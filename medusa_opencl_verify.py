"""
medusa_opencl_verify.py  –  Local-Memory Tiling + Parallel Reduction
=====================================================================
Optimised OpenCL implementation of the Medusa tree-verification step.

Background
----------
In medusa_model.py → medusa_generate() the inner loop calls:
    best_candidate, accept_length = evaluate_posterior(logits, candidates, ...)

Under the greedy fast-path (temperature=0, fast=True) that reduces to:
    is_match[i] = 1  if  argmax(logits[i]) == candidates[i]  else  0

Why the naive version is slow
------------------------------
The v1 kernel assigned ONE work-item to each row. That work-item had to
read all 32 000 floats serially from global memory — no parallelism inside
the row, terrible memory access latency.

Optimisation strategy (this file)
-----------------------------------
Assign ONE work-group of 256 work-items to each row:

  Phase 1 – Chunked load + thread-local reduce (registers)
    Work-item `lid` reads vocab columns  lid, lid+256, lid+512, …
    Each work-item keeps a running (best_val, best_idx) in registers.
    Strided access across 256 threads is coalesced on Intel iGPU.

  Phase 2 – Parallel reduction in __local memory
    All 256 (val, idx) pairs land in local_vals[256] / local_idxs[256].
    A power-of-two reduction tree folds them to a single winner.
    LDS used: 256 × (4 bytes float + 4 bytes int) = 2 048 bytes  (3.1 % of 64 KB).

  Phase 3 – Verification
    Work-item 0 compares final argmax against drafts[node] and writes 0/1.

Grid mapping
-------------
  global_size = (num_nodes * 256,)
  local_size  = (256,)
  → one work-group per row, 256 work-items per work-group.

Hardware target
---------------
  Intel UHD Graphics 620
    Max work-group size : 256
    Local memory        : 64 KB

Usage
-----
    pip install pyopencl
    python medusa_opencl_verify.py        # benchmark + correctness check
"""

import time
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  OpenCL kernel
# ─────────────────────────────────────────────────────────────────────────────

WG_SIZE = 256   # must match #define WG_SIZE in the kernel string

KERNEL_SOURCE = r"""
/* ------------------------------------------------------------------ *
 *  medusa_verify_local                                                 *
 *                                                                      *
 *  Grid: one work-group (WG_SIZE = 256 work-items) per logit row.     *
 *                                                                      *
 *  Phase 1 – each WI strides over vocab in registers                  *
 *  Phase 2 – parallel reduction in __local memory                     *
 *  Phase 3 – WI-0 writes the match verdict                            *
 * ------------------------------------------------------------------ */

#define WG_SIZE 256

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void medusa_verify_local(
    __global const float* restrict logits,    /* [num_nodes * vocab_size]  row-major */
    __global const int*   restrict drafts,    /* [num_nodes]  candidate token ids    */
    __global       int*   restrict is_match,  /* [num_nodes]  output: 0 or 1         */
    const int vocab_size
) {
    /* ── Identifiers ──────────────────────────────────────────────── */
    const int lid     = get_local_id(0);   /* 0 … WG_SIZE-1  within work-group */
    const int node_id = get_group_id(0);   /* which row of logits this WG owns  */

    /* Pointer to the start of this row in global memory. */
    const __global float* row = logits + (long)node_id * vocab_size;

    /* ── Phase 1: register-level reduce ──────────────────────────── *
     * Access pattern (lid=0): cols 0, 256, 512, …                   *
     *               (lid=1): cols 1, 257, 513, …   → coalesced read */
    float my_val = -INFINITY;
    int   my_idx = 0;

    for (int v = lid; v < vocab_size; v += WG_SIZE) {
        float val = row[v];
        if (val > my_val) {
            my_val = val;
            my_idx = v;
        }
    }

    /* ── Phase 2: parallel reduction in __local memory ───────────── */
    __local float local_vals[WG_SIZE];
    __local int   local_idxs[WG_SIZE];

    local_vals[lid] = my_val;
    local_idxs[lid] = my_idx;

    /* All work-items must have written before anyone reads. */
    barrier(CLK_LOCAL_MEM_FENCE);

    /*
     * Standard power-of-two reduction tree.
     * Each step halves the active work-items.
     * Tie-break rule: keep the smaller index — matches numpy.argmax behaviour
     * (returns the *first* occurrence of the maximum).
     */
    for (int stride = WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            float other_val = local_vals[lid + stride];
            int   other_idx = local_idxs[lid + stride];

            if (other_val > local_vals[lid] ||
               (other_val == local_vals[lid] && other_idx < local_idxs[lid])) {
                local_vals[lid] = other_val;
                local_idxs[lid] = other_idx;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* ── Phase 3: work-item 0 writes result ──────────────────────── */
    if (lid == 0) {
        is_match[node_id] = (local_idxs[0] == drafts[node_id]) ? 1 : 0;
    }
}
"""

# ─────────────────────────────────────────────────────────────────────────────
#  One-time setup: compile kernel, create context + queue
# ─────────────────────────────────────────────────────────────────────────────

def build_opencl_context(platform_idx: int = 0, device_idx: int = 0):
    """
    Compile the kernel once and return (ctx, queue, program).
    Pass these objects to run_medusa_verify_opencl() on every call
    to avoid recompilation overhead.
    """
    import pyopencl as cl

    platform = cl.get_platforms()[platform_idx]
    device   = platform.get_devices()[device_idx]
    ctx      = cl.Context([device])
    queue    = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
    )

    # -cl-fast-relaxed-math : allow approximate fp, fused multiply-add
    # -cl-mad-enable         : enable fused multiply-add explicitly
    program = cl.Program(ctx, KERNEL_SOURCE).build(
        options="-cl-fast-relaxed-math -cl-mad-enable"
    )

    lds_bytes = WG_SIZE * (4 + 4)   # float + int per work-item
    print(f"[OpenCL] Platform  : {platform.name}")
    print(f"[OpenCL] Device    : {device.name}")
    print(f"[OpenCL] Max WG    : {device.max_work_group_size}")
    print(f"[OpenCL] LDS total : {device.local_mem_size // 1024} KB")
    print(f"[OpenCL] LDS used  : {lds_bytes} bytes  "
          f"({lds_bytes / device.local_mem_size * 100:.1f}% of limit per WG)")

    return ctx, queue, program


# ─────────────────────────────────────────────────────────────────────────────
#  Per-call launch
# ─────────────────────────────────────────────────────────────────────────────

def run_medusa_verify_opencl(
    drafts_np: np.ndarray,
    logits_np: np.ndarray,
    ctx=None,
    queue=None,
    program=None,
    platform_idx: int = 0,
    device_idx:   int = 0,
):
    """
    Parameters
    ----------
    drafts_np        : int32   array, shape (num_nodes,)
    logits_np        : float32 array, shape (num_nodes, vocab_size)
    ctx, queue, program : pre-built OpenCL objects (recommended for benchmarks).
                          Built on the fly when None.

    Returns
    -------
    is_match : int32 array, shape (num_nodes,)   — 1 where argmax == draft
    elapsed  : float — kernel-only wall-clock time in seconds
                       (excludes H2D/D2H transfers; use for fair comparison)
    """
    import pyopencl as cl

    if ctx is None:
        ctx, queue, program = build_opencl_context(platform_idx, device_idx)

    num_nodes, vocab_size = logits_np.shape

    # Ensure contiguous C-order arrays with correct dtypes (no-op if already OK)
    drafts_np = np.ascontiguousarray(drafts_np, dtype=np.int32)
    logits_np = np.ascontiguousarray(logits_np, dtype=np.float32)

    # ── Buffers ───────────────────────────────────────────────────────────────
    mf = cl.mem_flags
    logits_buf = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=logits_np)
    drafts_buf = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=drafts_np)
    match_buf  = cl.Buffer(ctx, mf.WRITE_ONLY, size=drafts_np.nbytes)

    # ── Launch ────────────────────────────────────────────────────────────────
    # One work-group of WG_SIZE per row  →  num_nodes work-groups total.
    global_size = (num_nodes * WG_SIZE,)
    local_size  = (WG_SIZE,)

    # Fetch kernel explicitly to prevent the RepeatedKernelRetrieval warning
    kernel = cl.Kernel(program, "medusa_verify_local")
    
    event = kernel(
        queue,
        global_size,
        local_size,
        logits_buf,
        drafts_buf,
        match_buf,
        np.int32(vocab_size),
    )
    event.wait()

    # Profiling timestamps are in nanoseconds
    elapsed = (event.profile.end - event.profile.start) * 1e-9

    # ── Readback ──────────────────────────────────────────────────────────────
    is_match = np.empty(num_nodes, dtype=np.int32)
    cl.enqueue_copy(queue, is_match, match_buf)
    queue.finish()

    return is_match, elapsed


# ─────────────────────────────────────────────────────────────────────────────
#  NumPy baseline  (the sequential loop being replaced)
# ─────────────────────────────────────────────────────────────────────────────

def run_medusa_verify_python(drafts_np: np.ndarray, logits_np: np.ndarray):
    """Greedy fast-path of evaluate_posterior() from medusa_model.py."""
    t0       = time.perf_counter()
    argmaxes = np.argmax(logits_np, axis=1)
    is_match = (argmaxes == drafts_np).astype(np.int32)
    return is_match, time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
#  Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # vicuna_7b_stage2 tree  →  63 candidate nodes; typical LLM vocab size.
    NUM_NODES  = 63
    VOCAB_SIZE = 32_000
    WARMUP     = 3     # discarded passes (JIT, cache warm-up)
    REPEATS    = 20    # timed passes; we report the median

    RNG       = np.random.default_rng(42)
    logits_np = RNG.standard_normal((NUM_NODES, VOCAB_SIZE)).astype(np.float32)
    drafts_np = RNG.integers(0, VOCAB_SIZE, size=NUM_NODES, dtype=np.int32)

    sep = "=" * 64
    print(sep)
    print(f"  num_nodes  = {NUM_NODES}")
    print(f"  vocab_size = {VOCAB_SIZE}")
    print(f"  WG_SIZE    = {WG_SIZE}   (256 work-items, one WG per row)")
    print(f"  LDS/WG     = {WG_SIZE * 8} bytes  (float[256] + int[256])")
    print(sep)

    # ── NumPy baseline ────────────────────────────────────────────────────────
    for _ in range(WARMUP):
        run_medusa_verify_python(drafts_np, logits_np)

    py_times  = [run_medusa_verify_python(drafts_np, logits_np)[1] for _ in range(REPEATS)]
    py_result = run_medusa_verify_python(drafts_np, logits_np)[0]
    py_med    = float(np.median(py_times)) * 1e3

    print(f"\n[Python/NumPy]            median = {py_med:.4f} ms"
          f"   matches = {py_result.sum()}")

    # ── OpenCL optimised kernel ───────────────────────────────────────────────
    try:
        import pyopencl as cl   # noqa: F401

        print()
        ctx, queue, program = build_opencl_context()

        for _ in range(WARMUP):
            run_medusa_verify_opencl(drafts_np, logits_np,
                                     ctx=ctx, queue=queue, program=program)

        cl_times = [
            run_medusa_verify_opencl(drafts_np, logits_np,
                                     ctx=ctx, queue=queue, program=program)[1]
            for _ in range(REPEATS)
        ]
        cl_result, _ = run_medusa_verify_opencl(
            drafts_np, logits_np, ctx=ctx, queue=queue, program=program
        )
        cl_med = float(np.median(cl_times)) * 1e3

        print(f"\n[OpenCL local-reduce]     median = {cl_med:.4f} ms"
              f"   matches = {cl_result.sum()}")

        # ── Correctness check ─────────────────────────────────────────────────
        print()
        if np.array_equal(py_result, cl_result):
            print("✓  Results match — kernel is correct.")
        else:
            bad = np.where(py_result != cl_result)[0]
            print(f"✗  Mismatch at {len(bad)} node(s): indices {bad}")

        # ── Speedup ───────────────────────────────────────────────────────────
        speedup = py_med / cl_med if cl_med > 0 else float("inf")
        print(f"\n  Speedup vs NumPy (median kernel time only) : {speedup:.2f}×")
        print(sep)

    except ImportError:
        print("\n[OpenCL] pyopencl not found — install with:  pip install pyopencl")
    except Exception as exc:
        print(f"\n[OpenCL] Error during execution: {exc}")
        raise


if __name__ == "__main__":
    main()