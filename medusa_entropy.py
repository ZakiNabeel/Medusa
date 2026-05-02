"""
medusa_entropy.py — MedusaEntropyCalculator
============================================
PyOpenCL wrapper around entropy_kernel.cl.

Two public entry-points:

  calculator.calculate(probs)
      Input:  float32 ndarray (num_nodes, vocab_size) — already softmax'd
      Output: float32 ndarray (num_nodes,)             — entropy per node

  calculator.calculate_from_logits(logits)
      Input:  float32 ndarray (num_nodes, vocab_size) — raw logits
      Output: float32 ndarray (num_nodes,)             — entropy per node
      (fused softmax+entropy, no intermediate global write)

Designed to share the OpenCL context/queue already created by
build_opencl_context() in medusa_opencl_verify so that both kernels
can be enqueued on the same command queue without synchronisation overhead.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyopencl as cl

# ── Constants ────────────────────────────────────────────────────────────────
# Must be a power of 2 and ≤ the device's max workgroup size.
# 256 is safe on virtually every modern GPU and CPU OpenCL device.
_LOCAL_SIZE: int = 256

_KERNEL_FILE: Path = Path(__file__).with_name("entropy_kernel.cl")


# ── Helper ───────────────────────────────────────────────────────────────────

def _next_pow2(n: int) -> int:
    """Smallest power-of-2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _build_entropy_program(
    ctx: cl.Context,
    local_size: int,
    sub_group_size: int = 0,
) -> cl.Program:
    """
    Compile entropy_kernel.cl with the requested LOCAL_SIZE baked in.
    sub_group_size > 0 enables the shuffle fast path (Nvidia / Intel).
    """
    src = _KERNEL_FILE.read_text()

    defines = (
        f"-DLOCAL_SIZE={local_size} "
        f"-cl-fast-relaxed-math "       # enables native_* accuracy trade-offs
        f"-cl-mad-enable "
        f"-cl-no-signed-zeros "
    )
    if sub_group_size > 0:
        defines += f"-DSUB_GROUP_SIZE={sub_group_size} "

    program = cl.Program(ctx, src).build(options=defines)
    return program


# ── Main class ───────────────────────────────────────────────────────────────

class MedusaEntropyCalculator:
    """
    Computes per-node Shannon entropy over vocab distributions.

    Parameters
    ----------
    ctx : cl.Context, optional
        Reuse an existing OpenCL context (e.g. from build_opencl_context).
        If None a new context is created from the first GPU device found.
    queue : cl.CommandQueue, optional
        Must belong to `ctx`. Created automatically when ctx is supplied
        without one.
    local_size : int
        Work-group size for the reduction kernel. Must be a power of 2
        and ≤ the device's CL_DEVICE_MAX_WORK_GROUP_SIZE.
    entropy_threshold : float
        Branches whose entropy exceeds this value will be flagged for
        pruning by the `prune_mask` helper.  ln(V) ≈ 10.82 for V=50257.
        A good starting value is 2.0–4.0 (low-entropy = high confidence).
    """

    def __init__(
        self,
        ctx: Optional[cl.Context] = None,
        queue: Optional[cl.CommandQueue] = None,
        local_size: int = _LOCAL_SIZE,
        entropy_threshold: float = 3.0,
    ) -> None:
        # ── Context & queue ──────────────────────────────────────────
        if ctx is None:
            ctx = cl.create_some_context(interactive=False)
        self.ctx = ctx

        if queue is None:
            queue = cl.CommandQueue(
                ctx,
                properties=cl.command_queue_properties.PROFILING_ENABLE,
            )
        self.queue = queue

        # ── Device capability check ──────────────────────────────────
        device: cl.Device = ctx.devices[0]
        max_wg = device.max_work_group_size

        if local_size > max_wg:
            # Gracefully fall back to the largest supported power-of-2
            local_size = _next_pow2(max_wg) >> 1
            print(
                f"[MedusaEntropyCalculator] Warning: requested local_size "
                f"exceeds device max ({max_wg}). Falling back to {local_size}."
            )
        self.local_size: int = local_size

        # Detect sub-group size for optional shuffle fast-path
        sub_group_size = 0
        try:
            sg = device.preferred_vector_width_float   # rough proxy; real query below
            sub_group_size = getattr(
                device,
                "preferred_work_group_size_multiple",  # cl.device_info enum
                0,
            )
        except Exception:
            pass

        # ── Compile program ──────────────────────────────────────────
        self.program: cl.Program = _build_entropy_program(
            ctx, local_size, sub_group_size
        )
        self._k_entropy = self.program.compute_entropy
        self._k_fused   = self.program.logits_to_entropy

        # ── Kernel arg type declarations ─────────────────────────────
        # pyopencl needs explicit scalar types
        self._k_entropy.set_scalar_arg_dtypes([None, None, np.int32])
        self._k_fused.set_scalar_arg_dtypes([None, None, np.int32])

        # ── Persistent output buffer (resized lazily) ────────────────
        self._out_buf: Optional[cl.Buffer] = None
        self._out_num_nodes: int = 0

        self.entropy_threshold: float = entropy_threshold

        # ── Timing accumulator ───────────────────────────────────────
        self._timings_ms: list[float] = []

    # ── Buffer management ────────────────────────────────────────────────────

    def _ensure_output_buffer(self, num_nodes: int) -> cl.Buffer:
        """Lazily allocate / reuse the output buffer."""
        if self._out_buf is None or self._out_num_nodes != num_nodes:
            self._out_buf = cl.Buffer(
                self.ctx,
                cl.mem_flags.WRITE_ONLY,
                size=num_nodes * np.dtype(np.float32).itemsize,
            )
            self._out_num_nodes = num_nodes
        return self._out_buf

    # ── Public API ───────────────────────────────────────────────────────────

    def calculate(self, probs: np.ndarray) -> np.ndarray:
        """
        Compute entropy from a pre-computed softmax probability array.

        Parameters
        ----------
        probs : ndarray, shape (num_nodes, vocab_size), dtype float32
            Row i contains the probability distribution for node i.
            Each row must sum to 1.0 (already softmax-normalised).

        Returns
        -------
        entropies : ndarray, shape (num_nodes,), dtype float32
        """
        probs = np.ascontiguousarray(probs, dtype=np.float32)
        num_nodes, vocab_size = probs.shape

        # Upload to device
        in_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=probs,
        )
        out_buf = self._ensure_output_buffer(num_nodes)

        # Launch: global_size = num_nodes groups × local_size lanes
        global_size = (num_nodes * self.local_size,)
        local_size  = (self.local_size,)

        t0 = time.perf_counter()
        evt = self._k_entropy(
            self.queue,
            global_size,
            local_size,
            in_buf,
            out_buf,
            np.int32(vocab_size),
        )
        evt.wait()
        self._timings_ms.append((time.perf_counter() - t0) * 1e3)

        result = np.empty(num_nodes, dtype=np.float32)
        cl.enqueue_copy(self.queue, result, out_buf)
        self.queue.finish()
        return result

    def calculate_from_logits(self, logits: np.ndarray) -> np.ndarray:
        """
        Fused softmax + entropy from raw logits.  Never materialises the
        softmax in global memory — faster and more numerically stable.

        Parameters
        ----------
        logits : ndarray, shape (num_nodes, vocab_size), dtype float32

        Returns
        -------
        entropies : ndarray, shape (num_nodes,), dtype float32
        """
        logits = np.ascontiguousarray(logits, dtype=np.float32)
        num_nodes, vocab_size = logits.shape

        in_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=logits,
        )
        out_buf = self._ensure_output_buffer(num_nodes)

        global_size = (num_nodes * self.local_size,)
        local_size  = (self.local_size,)

        t0 = time.perf_counter()
        evt = self._k_fused(
            self.queue,
            global_size,
            local_size,
            in_buf,
            out_buf,
            np.int32(vocab_size),
        )
        evt.wait()
        self._timings_ms.append((time.perf_counter() - t0) * 1e3)

        result = np.empty(num_nodes, dtype=np.float32)
        cl.enqueue_copy(self.queue, result, out_buf)
        self.queue.finish()
        return result

    # ── Pruning helper ───────────────────────────────────────────────────────

    def prune_mask(self, entropies: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask where True = node should be pruned (high entropy).

        A node's entropy exceeds the threshold when the Medusa head is
        uncertain about that position.  Pruning these branches avoids
        wasting the tree-attention forward pass on unlikely candidates.

        Parameters
        ----------
        entropies : ndarray, shape (num_nodes,)

        Returns
        -------
        mask : bool ndarray, shape (num_nodes,)
            True  → prune this node
            False → keep this node
        """
        return entropies > self.entropy_threshold

    # ── Diagnostics ─────────────────────────────────────────────────────────

    @property
    def avg_kernel_time_ms(self) -> float:
        return float(np.mean(self._timings_ms)) if self._timings_ms else 0.0

    def reset_timings(self) -> None:
        self._timings_ms.clear()

    def __repr__(self) -> str:
        return (
            f"MedusaEntropyCalculator("
            f"local_size={self.local_size}, "
            f"threshold={self.entropy_threshold:.2f}, "
            f"calls={len(self._timings_ms)}, "
            f"avg_ms={self.avg_kernel_time_ms:.3f})"
        )