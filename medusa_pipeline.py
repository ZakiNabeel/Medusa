"""
medusa_pipeline.py  –  End-to-End Speculative Decoding Pipeline
================================================================
A single-script inference pipeline that fuses:
  • GPT-2 + custom Medusa heads  (neural inference, torch/transformers)
  • Entropy-adaptive tree construction  (numpy)
  • Hardware-accelerated candidate verification  (pyopencl, Intel UHD 620)
  • Longest-path acceptance + speedup reporting  (numpy)

Usage
-----
    python medusa_pipeline.py
    python medusa_pipeline.py --prompt "The Eiffel Tower is located in"
    python medusa_pipeline.py --prompt "Once upon a time" --heads 4

Dependencies
------------
    pip install torch transformers numpy pyopencl
    (No Medusa repo required.)
"""

import argparse
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 0 – GPT-2 + Medusa Heads
# ══════════════════════════════════════════════════════════════════════════════

class GPT2Medusa(nn.Module):
    """
    GPT-2 base model augmented with lightweight Medusa speculative-decoding
    heads.  Each head is a single linear projection from hidden-state space
    to the full vocabulary, predicting the token t+k for head k.

    Parameters
    ----------
    base_model_name : str
        Any HuggingFace causal-LM identifier (default: "gpt2").
    num_heads : int
        Number of speculative lookahead heads (default: 3).
        Head 0 predicts t+1, head 1 predicts t+2, …
    """

    def __init__(self, base_model_name: str = "gpt2", num_heads: int = 3):
        super().__init__()
        self.tokenizer  = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

        self.hidden_size = self.base_model.config.n_embd
        self.vocab_size  = self.base_model.config.vocab_size
        self.num_heads   = num_heads

        # Medusa heads: untrained linear layers (fine-tuning is out of scope).
        # In production these would be loaded from a trained checkpoint.
        self.medusa_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            for _ in range(num_heads)
        ])

    def forward(self, text_prompt: str) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Run one full forward pass for a text prompt.

        Returns
        -------
        base_logits   : float32 ndarray (vocab_size,)
            Raw logit distribution of the base model over the next token.
        medusa_logits : list of float32 ndarray, each (vocab_size,)
            One logit vector per Medusa head.
        """
        inputs  = self.tokenizer(text_prompt, return_tensors="pt")
        outputs = self.base_model(**inputs, output_hidden_states=True)

        # Last-position logits from the base model
        base_logits      = outputs.logits[0, -1, :].detach().numpy()          # (V,)
        # Last-position hidden state fed into each Medusa head
        last_hidden      = outputs.hidden_states[-1][0, -1, :]                # (H,)

        medusa_logits: List[np.ndarray] = []
        for head in self.medusa_heads:
            medusa_logits.append(head(last_hidden).detach().numpy())           # (V,)

        return base_logits, medusa_logits


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 – Shannon Entropy + Adaptive Threshold
# ══════════════════════════════════════════════════════════════════════════════

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp     = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def shannon_entropy(logits: np.ndarray) -> float:
    """
    Compute Shannon entropy H = -Σ p·log(p) (nats) from a raw logit vector.

    Parameters
    ----------
    logits : float32 (vocab_size,)

    Returns
    -------
    float  – entropy in nats.  Range: [0, ln(vocab_size)].
    """
    probs = _softmax(logits.astype(np.float32))
    # Numerically safe: ignore zero-probability tokens
    mask  = probs > 0.0
    return float(-np.sum(probs[mask] * np.log(probs[mask])))


def entropy_to_threshold(
    entropy: float,
    max_entropy: float,
    low_thr: float  = 0.02,
    high_thr: float = 0.15,
) -> float:
    """
    Map Shannon entropy → confidence threshold for tree pruning.

    Logic
    -----
    • High entropy (uncertain model)  →  *low* threshold  (wide tree, more guesses)
    • Low  entropy (confident model)  →  *high* threshold  (narrow tree, prune aggressively)

    The mapping is linear between [0, max_entropy] → [high_thr, low_thr].

    Parameters
    ----------
    entropy     : entropy of this head's distribution (nats).
    max_entropy : ln(vocab_size), the theoretical maximum.
    low_thr     : threshold used when entropy is at maximum.
    high_thr    : threshold used when entropy is at zero.

    Returns
    -------
    float – confidence_threshold in (low_thr, high_thr).
    """
    if max_entropy <= 0:
        return high_thr
    ratio = min(entropy / max_entropy, 1.0)         # 0 = certain, 1 = uniform
    return high_thr + ratio * (low_thr - high_thr)  # linearly interpolated


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 – Dynamic Candidate Tree Builder  (unchanged from medusa_opencl_verify)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DynamicTreeOutput:
    """Flat BFS-ordered candidate tree returned by build_dynamic_tree()."""
    candidates:      np.ndarray   # int32   (num_nodes,)
    parent_indices:  np.ndarray   # int32   (num_nodes,)
    node_logits:     np.ndarray   # float32 (num_nodes, vocab_size)
    num_nodes:       int
    num_levels:      int
    pruned_branches: int


def build_dynamic_tree(
    medusa_head_logits: List[np.ndarray],
    confidence_threshold: float = 0.15,
    max_nodes: int = 256,
) -> DynamicTreeOutput:
    """
    Build a BFS candidate tree from Medusa-head logits.

    Node 0 is always the root placeholder (token=-1, parent=-1).
    Real candidates start at index 1.

    Parameters
    ----------
    medusa_head_logits   : list of float32 (vocab_size,), one per head.
    confidence_threshold : minimum softmax probability to keep a token.
    max_nodes            : hard cap on tree size.

    Returns
    -------
    DynamicTreeOutput  – pass .candidates / .node_logits to the OpenCL kernel.
    """
    vocab_size   = medusa_head_logits[0].shape[0]
    num_heads    = len(medusa_head_logits)
    pruned_total = 0

    head_probs: List[np.ndarray] = [
        _softmax(h.astype(np.float32)) for h in medusa_head_logits
    ]

    candidates_list: List[int] = [-1]
    parent_list:     List[int] = [-1]
    node_level:      List[int] = [0]
    node_head_idx:   List[int] = [0]

    frontier: List[int]  = [0]
    num_levels_active    = 0

    for head_idx in range(num_heads):
        if not frontier or len(candidates_list) >= max_nodes:
            break

        probs        = head_probs[head_idx]
        passing_mask = probs > confidence_threshold
        passing_ids  = np.where(passing_mask)[0]
        pruned_total += int((~passing_mask).sum())

        if len(passing_ids) == 0:
            break

        next_frontier: List[int] = []

        for parent_node_idx in frontier:
            for token_id in passing_ids:
                new_node_idx = len(candidates_list)
                if new_node_idx >= max_nodes:
                    break
                candidates_list.append(int(token_id))
                parent_list.append(parent_node_idx)
                node_level.append(head_idx + 1)
                node_head_idx.append(head_idx)
                next_frontier.append(new_node_idx)
            if len(candidates_list) >= max_nodes:
                break

        frontier = next_frontier
        num_levels_active += 1

    num_nodes      = len(candidates_list)
    candidates     = np.array(candidates_list, dtype=np.int32)
    parent_indices = np.array(parent_list,     dtype=np.int32)

    # node_logits[i] = raw logit vector for the head that produced node i.
    # Row 0 (root placeholder) is zero-padded and ignored by the kernel.
    node_logits = np.zeros((num_nodes, vocab_size), dtype=np.float32)
    for i in range(1, num_nodes):
        node_logits[i] = medusa_head_logits[node_head_idx[i]]

    return DynamicTreeOutput(
        candidates      = candidates,
        parent_indices  = parent_indices,
        node_logits     = node_logits,
        num_nodes       = num_nodes,
        num_levels      = num_levels_active,
        pruned_branches = pruned_total,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 – OpenCL Hardware-Accelerated Verification
#  (kernel source identical to medusa_opencl_verify.py – Intel UHD 620 tuned)
# ══════════════════════════════════════════════════════════════════════════════

WG_SIZE = 256   # must match #define WG_SIZE inside the kernel

KERNEL_SOURCE = r"""
/* ------------------------------------------------------------------ *
 *  medusa_verify_local                                                 *
 *                                                                      *
 *  Grid: one work-group (WG_SIZE = 256 work-items) per logit row.     *
 *                                                                      *
 *  Phase 1 – each WI strides over vocab in registers                  *
 *  Phase 2 – parallel reduction in __local memory                     *
 *  Phase 3 – WI-0 writes the match verdict                            *
 *                                                                      *
 *  LDS used: 2 048 bytes / WG  (3.1 % of 64 KB on Intel UHD 620)     *
 * ------------------------------------------------------------------ */

#define WG_SIZE 256

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void medusa_verify_local(
    __global const float* restrict logits,    /* [num_nodes * vocab_size]  row-major */
    __global const int*   restrict drafts,    /* [num_nodes]  candidate token ids    */
    __global       int*   restrict is_match,  /* [num_nodes]  output: 0 or 1         */
    const int vocab_size
) {
    const int lid     = get_local_id(0);
    const int node_id = get_group_id(0);

    const __global float* row = logits + (long)node_id * vocab_size;

    /* Phase 1 – register-level strided reduce */
    float my_val = -INFINITY;
    int   my_idx = 0;
    for (int v = lid; v < vocab_size; v += WG_SIZE) {
        float val = row[v];
        if (val > my_val) { my_val = val; my_idx = v; }
    }

    /* Phase 2 – parallel reduction in __local memory */
    __local float local_vals[WG_SIZE];
    __local int   local_idxs[WG_SIZE];

    local_vals[lid] = my_val;
    local_idxs[lid] = my_idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            float ov = local_vals[lid + stride];
            int   oi = local_idxs[lid + stride];
            if (ov > local_vals[lid] ||
               (ov == local_vals[lid] && oi < local_idxs[lid])) {
                local_vals[lid] = ov;
                local_idxs[lid] = oi;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Phase 3 – WI-0 writes match result */
    if (lid == 0) {
        is_match[node_id] = (local_idxs[0] == drafts[node_id]) ? 1 : 0;
    }
}
"""


def build_opencl_context(platform_idx: int = 0, device_idx: int = 0):
    """Compile the OpenCL kernel once; return (ctx, queue, program)."""
    import pyopencl as cl

    platform = cl.get_platforms()[platform_idx]
    device   = platform.get_devices()[device_idx]
    ctx      = cl.Context([device])
    queue    = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
    )
    program  = cl.Program(ctx, KERNEL_SOURCE).build(
        options="-cl-fast-relaxed-math -cl-mad-enable"
    )

    lds_bytes = WG_SIZE * (4 + 4)
    print(f"  [OpenCL] Platform  : {platform.name}")
    print(f"  [OpenCL] Device    : {device.name}")
    print(f"  [OpenCL] Max WG    : {device.max_work_group_size}")
    print(f"  [OpenCL] LDS total : {device.local_mem_size // 1024} KB")
    print(f"  [OpenCL] LDS/WG    : {lds_bytes} bytes "
          f"({lds_bytes / device.local_mem_size * 100:.1f}% of limit)")

    return ctx, queue, program


def run_medusa_verify_opencl(
    drafts_np: np.ndarray,
    logits_np: np.ndarray,
    ctx=None, queue=None, program=None,
    platform_idx: int = 0,
    device_idx:   int = 0,
) -> Tuple[np.ndarray, float]:
    """
    Launch medusa_verify_local on the GPU.

    Parameters
    ----------
    drafts_np : int32   (num_nodes,)
    logits_np : float32 (num_nodes, vocab_size)

    Returns
    -------
    is_match : int32 (num_nodes,)  – 1 where argmax(logits_row) == draft token
    elapsed  : float               – kernel-only time in seconds
    """
    import pyopencl as cl

    if ctx is None:
        ctx, queue, program = build_opencl_context(platform_idx, device_idx)

    num_nodes, vocab_size = logits_np.shape
    drafts_np = np.ascontiguousarray(drafts_np, dtype=np.int32)
    logits_np = np.ascontiguousarray(logits_np, dtype=np.float32)

    mf         = cl.mem_flags
    logits_buf = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=logits_np)
    drafts_buf = cl.Buffer(ctx, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=drafts_np)
    match_buf  = cl.Buffer(ctx, mf.WRITE_ONLY, size=drafts_np.nbytes)

    kernel = cl.Kernel(program, "medusa_verify_local")
    event  = kernel(
        queue,
        (num_nodes * WG_SIZE,),
        (WG_SIZE,),
        logits_buf, drafts_buf, match_buf, np.int32(vocab_size),
    )
    event.wait()

    elapsed  = (event.profile.end - event.profile.start) * 1e-9
    is_match = np.empty(num_nodes, dtype=np.int32)
    cl.enqueue_copy(queue, is_match, match_buf)
    queue.finish()

    return is_match, elapsed


def run_medusa_verify_numpy(
    drafts_np: np.ndarray,
    logits_np: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """CPU/NumPy baseline – sequential argmax per row."""
    t0       = time.perf_counter()
    argmaxes = np.argmax(logits_np, axis=1)
    is_match = (argmaxes == drafts_np).astype(np.int32)
    return is_match, time.perf_counter() - t0


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 – Path Tracing  (unchanged from medusa_opencl_verify)
# ══════════════════════════════════════════════════════════════════════════════

def trace_longest_path(
    is_match:       np.ndarray,
    parent_indices: np.ndarray,
) -> Tuple[int, int]:
    """
    Walk the verified tree and return the longest contiguous accepted path.

    A node is accepted only when it matches *and* its entire ancestor chain
    back to the root is also accepted.

    Parameters
    ----------
    is_match       : int32 (num_nodes,)  – 1 = verified match
    parent_indices : int32 (num_nodes,)  – -1 = connected to root

    Returns
    -------
    best_node   : int  – index of the terminal node of the longest path
    path_length : int  – number of accepted tokens
    """
    valid_path_lengths = np.zeros_like(is_match)

    for i in range(len(is_match)):
        if is_match[i] == 1:
            parent = parent_indices[i]
            if parent == -1:
                valid_path_lengths[i] = 1
            elif valid_path_lengths[parent] > 0:
                valid_path_lengths[i] = valid_path_lengths[parent] + 1

    best_node   = int(np.argmax(valid_path_lengths))
    path_length = int(valid_path_lengths[best_node])
    return best_node, path_length


# ══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    prompt:       str  = "The capital of France is",
    num_heads:    int  = 3,
    low_thr:      float = 0.02,
    high_thr:     float = 0.15,
    max_nodes:    int   = 256,
) -> None:
    """
    Execute all four stages for a single prompt and print a structured report.

    Stages
    ------
    1. Neural inference   – GPT-2 + Medusa heads → base_logits, medusa_logits
    2. Tree construction  – entropy → adaptive threshold → DynamicTreeOutput
    3. GPU verification   – OpenCL kernel argmax-vs-draft comparison
    4. Logic resolution   – trace_longest_path → accepted tokens + speedup
    """
    SEP  = "═" * 68
    sep2 = "─" * 68

    print(f"\n{SEP}")
    print("  MEDUSA SPECULATIVE DECODING PIPELINE")
    print(SEP)
    print(f"  Prompt   : \"{prompt}\"")
    print(f"  Heads    : {num_heads}  |  max_nodes : {max_nodes}")
    print(f"  Threshold range : [{low_thr}, {high_thr}]  "
          "(low=uncertain, high=confident)")
    print(sep2)

    # ──────────────────────────────────────────────────────────────────────────
    #  STAGE 1 – Neural Inference
    # ──────────────────────────────────────────────────────────────────────────
    print("\n  STAGE 1 — Neural Inference")
    print(sep2)

    print("  Loading GPT-2 + Medusa heads …")
    t_inf_start = time.perf_counter()
    model       = GPT2Medusa(num_heads=num_heads)

    print(f"  Running forward pass …")
    base_logits, medusa_logits = model(prompt)
    t_inf_end   = time.perf_counter()
    t_inference = t_inf_end - t_inf_start

    vocab_size  = base_logits.shape[0]
    max_entropy = float(np.log(vocab_size))   # H_max = ln(V)

    print(f"  Vocab size        : {vocab_size:,}")
    print(f"  Inference time    : {t_inference * 1e3:.1f} ms")
    print(f"  Base argmax token : {int(np.argmax(base_logits))}  "
          f"(\"{model.tokenizer.decode([int(np.argmax(base_logits))])}\".strip())")
    print()

    # Per-head entropy
    entropies:   List[float] = []
    thresholds:  List[float] = []
    print(f"  {'Head':>4}  {'Entropy (nats)':>16}  {'Max H':>8}  "
          f"{'Threshold':>10}  {'Direction'}")
    print(f"  {'----':>4}  {'---------------':>16}  {'-------':>8}  "
          f"{'----------':>10}  {'----------'}")

    for i, h_logits in enumerate(medusa_logits):
        H   = shannon_entropy(h_logits)
        thr = entropy_to_threshold(H, max_entropy, low_thr, high_thr)
        entropies.append(H)
        thresholds.append(thr)
        direction = "← wide (uncertain)" if H > 0.5 * max_entropy else "→ narrow (confident)"
        print(f"  {i:>4}  {H:>16.4f}  {max_entropy:>8.4f}  {thr:>10.4f}  {direction}")

    # ──────────────────────────────────────────────────────────────────────────
    #  STAGE 2 – Entropy-Based Tree Construction
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{sep2}")
    print("  STAGE 2 — Entropy-Based Tree Construction")
    print(sep2)

    # Use the mean threshold across heads as a single tree-level threshold.
    # This keeps the tree builder interface clean (one scalar threshold).
    mean_threshold = float(np.mean(thresholds))
    print(f"  Mean adaptive threshold : {mean_threshold:.5f}")

    t_tree_start = time.perf_counter()
    tree = build_dynamic_tree(
        medusa_logits,
        confidence_threshold = mean_threshold,
        max_nodes            = max_nodes,
    )
    t_tree_end = time.perf_counter()

    print(f"  Tree build time         : {(t_tree_end - t_tree_start) * 1e3:.2f} ms")
    print(f"  Total nodes (incl. root): {tree.num_nodes}")
    print(f"  Active levels           : {tree.num_levels}  (of {num_heads} heads)")
    print(f"  Branches pruned         : {tree.pruned_branches}")
    print(f"  node_logits shape       : {tree.node_logits.shape}")

    # ── Prepare verification arrays (skip root placeholder at index 0) ────────
    verify_candidates  = tree.candidates[1:]       # (num_nodes-1,)
    verify_logits      = tree.node_logits[1:]      # (num_nodes-1, vocab_size)
    verify_parents_raw = tree.parent_indices[1:]   # parent indices in full tree

    # Re-index parents so the verified sub-array is self-consistent:
    # full-tree root (0) → -1 in the verified view.
    verify_parents = np.where(
        verify_parents_raw == 0, -1, verify_parents_raw - 1
    ).astype(np.int32)

    num_verify = len(verify_candidates)

    # ──────────────────────────────────────────────────────────────────────────
    #  STAGE 3 – Hardware-Accelerated Verification
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{sep2}")
    print(f"  STAGE 3 — GPU Verification  ({num_verify} candidate nodes)")
    print(sep2)

    # NumPy baseline (sequential CPU)
    is_match_np, t_numpy = run_medusa_verify_numpy(verify_candidates, verify_logits)
    print(f"  [NumPy baseline]  time = {t_numpy * 1e3:.4f} ms  "
          f"matches = {is_match_np.sum()}")

    # Attempt OpenCL GPU path
    use_opencl   = False
    is_match     = is_match_np
    t_gpu        = t_numpy    # fallback for speedup calculation

    if num_verify == 0:
        print("  [OpenCL] Skipped – tree is empty (all branches pruned).")
    else:
        try:
            import pyopencl as cl  # noqa: F401

            ctx, queue, program = build_opencl_context()
            is_match_cl, t_cl   = run_medusa_verify_opencl(
                verify_candidates, verify_logits,
                ctx=ctx, queue=queue, program=program,
            )

            print(f"\n  [OpenCL kernel]   time = {t_cl * 1e3:.4f} ms  "
                  f"matches = {is_match_cl.sum()}")

            if np.array_equal(is_match_np, is_match_cl):
                print("  ✓  NumPy and OpenCL results match.")
            else:
                bad = np.where(is_match_np != is_match_cl)[0]
                print(f"  ✗  Mismatch at {len(bad)} node(s): indices {bad[:10]}")

            is_match   = is_match_cl
            t_gpu      = t_cl
            use_opencl = True

        except ImportError:
            print("\n  [OpenCL] pyopencl not installed — falling back to NumPy.")
            print("           Install with:  pip install pyopencl")
        except Exception as exc:
            print(f"\n  [OpenCL] Error: {exc}")
            print("  Falling back to NumPy result.")

    # ──────────────────────────────────────────────────────────────────────────
    #  STAGE 4 – Logic Resolution
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{sep2}")
    print("  STAGE 4 — Logic Resolution")
    print(sep2)

    # FIX: Check if we actually have nodes to verify
    if num_verify > 0:
        best_node, accepted_length = trace_longest_path(is_match, verify_parents)

        if accepted_length > 0:
            path_tokens = []
            cur = best_node
            while cur != -1:
                path_tokens.append(int(verify_candidates[cur]))
                parent = int(verify_parents[cur])
                cur = parent if parent >= 0 else -1
            path_tokens.reverse()

            accepted_text = model.tokenizer.decode(path_tokens)

            print(f"  Accepted length   : {accepted_length} token(s)")
            print(f"  Terminal node     : {best_node}")
            print(f"  Token IDs         : {path_tokens}")
            print(f"  Decoded text      : \"{accepted_text}\"")
        else:
            print("  No speculative tokens accepted (no matching path found).")
    else:
        accepted_length = 0
        print("  Tree is empty – all candidate branches were pruned due to low confidence.")
        print("  (This is expected since Medusa heads are currently untrained.)")

    # ── Speedup report ────────────────────────────────────────────────────────
    # Sequential baseline = 1 forward pass per accepted token.
    # Speculative baseline = 1 forward pass for all num_heads candidates
    # at once, then verification.
    #
    # Empirical speedup metric used here:
    #   speedup = (accepted_length + 1) / 1
    # because Medusa accepts `accepted_length` draft tokens in one model call
    # that would have required `accepted_length + 1` sequential calls
    # (the +1 accounts for the base-model step that also runs).
    #
    # Verification speedup (kernel vs. NumPy) is reported separately.

    sequential_tokens   = max(accepted_length, 1)   # tokens that would need separate calls
    speculative_speedup = (accepted_length + 1)      # tokens produced in one forward pass

    print()
    print(f"  ─── Speedup Summary ───────────────────────────────────────")
    if accepted_length > 0:
        token_speedup = (accepted_length + 1) / 1.0
        print(f"  Token-level speedup  : {token_speedup:.1f}×  "
              f"({accepted_length + 1} tokens per base-model call "
              f"vs. 1 without speculation)")
    else:
        print("  Token-level speedup  : 1.0×  (no drafts accepted this step)")

    if use_opencl and t_numpy > 0 and t_gpu > 0:
        verify_speedup = t_numpy / t_gpu
        print(f"  Verification speedup : {verify_speedup:.2f}×  "
              f"(OpenCL vs. NumPy,  "
              f"{t_numpy * 1e3:.4f} ms → {t_gpu * 1e3:.4f} ms)")
    else:
        print("  Verification speedup : N/A  (OpenCL not active)")

    print(f"\n{SEP}")
    print("  Pipeline complete.")
    print(SEP)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Medusa Speculative Decoding Pipeline (GPT-2 + OpenCL)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Text prompt to run speculative decoding on.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=3,
        help="Number of Medusa speculative heads (default: 3).",
    )
    parser.add_argument(
        "--low-thr",
        type=float,
        default=0.02,
        help="Threshold when entropy is maximum (default: 0.02).",
    )
    parser.add_argument(
        "--high-thr",
        type=float,
        default=0.15,
        help="Threshold when entropy is minimum (default: 0.15).",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=256,
        help="Hard cap on candidate tree size (default: 256).",
    )
    args = parser.parse_args()

    run_pipeline(
        prompt    = args.prompt,
        num_heads = args.heads,
        low_thr   = args.low_thr,
        high_thr  = args.high_thr,
        max_nodes = args.max_nodes,
    )