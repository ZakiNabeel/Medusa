"""
medusa_opencl_verify.py  –  Dynamic Tree Candidate Generation + OpenCL Verification
====================================================================================
Extends the local-memory / parallel-reduction OpenCL kernel with a fully dynamic
Medusa candidate-tree builder that replaces the static medusa_choices.py topology.

Two-stage pipeline
------------------
Stage 1 – Dynamic tree builder  (CPU, runs once per generation step)
  build_dynamic_tree(medusa_head_logits, confidence_threshold=0.15)

  Replaces the fixed vicuna_7b_stage2 / mc_sim_7b_63 choice lists.
  Works level-by-level:
    a. Softmax each Medusa head's logit vector  →  probability distribution.
    b. Keep only tokens whose probability > confidence_threshold.
    c. Expand the Cartesian product of surviving tokens across all heads,
       tracking the parent index for each new node.
    d. Returns two flat arrays ready for the OpenCL kernel:
         candidates     int32 (num_nodes,)   — draft token at each node
         parent_indices int32 (num_nodes,)   — parent node index (-1 = root)
       plus a companion array:
         node_logits    float32 (num_nodes, vocab_size)  — row-major logit
                        matrix to feed straight into run_medusa_verify_opencl()

Stage 2 – OpenCL verification  (GPU, local-memory tiling + parallel reduction)
  run_medusa_verify_opencl(candidates, node_logits, ...)
  → is_match int32 (num_nodes,)

Stage 3 – Path tracing  (CPU)
  trace_longest_path(is_match, parent_indices)
  → (best_node, accepted_length)

Background on the static approach
----------------------------------
medusa_choices.py stores hand-tuned trees (e.g. vicuna_7b_stage2 has 63 nodes).
They are topology-optimal for average inputs but waste capacity on low-confidence
branches and miss high-confidence branches outside the fixed fan-out.
The dynamic builder adapts the tree to the model's actual output distribution
at every step, typically reducing num_nodes while maintaining or improving
acceptance length.

OpenCL kernel (unchanged from v2)
-----------------------------------
  global_size = (num_nodes * 256,)  — one 256-wide work-group per row
  local_size  = (256,)
  LDS used    : 2 048 bytes per work-group  (3.1 % of 64 KB)

Hardware target
---------------
  Intel UHD Graphics 620
    Max work-group size : 256
    Local memory        : 64 KB

Usage
-----
    pip install pyopencl
    python medusa_opencl_verify.py
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1 – Dynamic Candidate Tree Builder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DynamicTreeOutput:
    """
    Everything produced by build_dynamic_tree() in one object.

    Attributes
    ----------
    candidates : int32 (num_nodes,)
        The draft token id stored at each tree node.  Node 0 is always the
        root placeholder (token = -1, never verified); real candidates start
        at index 1.

    parent_indices : int32 (num_nodes,)
        parent_indices[i] = index of node i's parent in the same flat array.
        Root node has parent -1.

    node_logits : float32 (num_nodes, vocab_size)
        Row i holds the verification logits for node i — i.e. the logit
        distribution over the full vocabulary *at the position being
        predicted by node i*.  Row 0 (root) is all-zero padding and is
        never read by the OpenCL kernel in practice.

    num_nodes : int
        Total nodes including the root placeholder.

    num_levels : int
        Number of Medusa heads that contributed at least one node.

    pruned_branches : int
        How many candidate tokens were discarded by the threshold.
    """
    candidates:      np.ndarray          # int32   (num_nodes,)
    parent_indices:  np.ndarray          # int32   (num_nodes,)
    node_logits:     np.ndarray          # float32 (num_nodes, vocab_size)
    num_nodes:       int
    num_levels:      int
    pruned_branches: int


def _softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable row-wise softmax.
    logits : float32 (..., vocab_size)
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)   # prevent overflow
    exp     = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def build_dynamic_tree(
    medusa_head_logits: List[np.ndarray],
    confidence_threshold: float = 0.15,
    max_nodes: int = 256,
) -> DynamicTreeOutput:
    """
    Build a candidate tree dynamically from Medusa-head logits.

    This replaces the static vicuna_7b_stage2 / mc_sim_7b_63 choice lists
    from medusa_choices.py.  At each generation step the tree topology is
    derived from the model's own confidence, so only probable branches
    are expanded and dead branches are pruned early.

    Parameters
    ----------
    medusa_head_logits : list of float32 arrays, each shape (vocab_size,)
        One logit vector per Medusa head, in head order (head 0 predicts
        token t+1, head 1 predicts t+2, …).  These are the raw pre-softmax
        outputs from model.medusa_head[i].

    confidence_threshold : float, default 0.15
        Minimum softmax probability for a token to be added as a child node.
        Tokens below this value are pruned.  Typical useful range: 0.05–0.30.
        A lower value grows larger trees; a higher value produces sparser,
        more confident trees.

    max_nodes : int, default 256
        Hard ceiling on tree size.  Prevents pathological growth when many
        tokens exceed the threshold at a low-confidence level.  Expansion
        stops (breadth-first) once this limit is reached.

    Returns
    -------
    DynamicTreeOutput
        See dataclass docstring.  Pass .candidates and .node_logits directly
        to run_medusa_verify_opencl(); pass .parent_indices to
        trace_longest_path().

    Algorithm
    ---------
    The tree is built level-by-level (BFS).  Each level corresponds to one
    Medusa head:

      Level 0 (root, index 0):
        Placeholder node, token = -1, parent = -1.
        Not submitted to the OpenCL kernel.

      Level k  (head k-1, predicts position t+k):
        For every live node at level k-1, apply softmax to head-(k-1) logits
        and keep all tokens with probability > confidence_threshold.
        Each surviving token becomes a new child node whose parent_index
        points to the node that spawned it.

    The resulting flat arrays are ordered BFS (level by level), so
    parent_indices[i] < i always holds — a property trace_longest_path()
    relies on.
    """
    vocab_size    = medusa_head_logits[0].shape[0]
    num_heads     = len(medusa_head_logits)
    pruned_total  = 0

    # ── Pre-compute softmax for every head once ──────────────────────────────
    head_probs: List[np.ndarray] = []
    for raw_logits in medusa_head_logits:
        head_probs.append(_softmax(raw_logits.astype(np.float32)))

    # ── Flat tree storage ────────────────────────────────────────────────────
    # Node 0 is the root placeholder; real nodes start at 1.
    candidates_list:     List[int]        = [-1]          # token id at each node
    parent_list:         List[int]        = [-1]          # parent node index
    node_level:          List[int]        = [0]           # which head produced this node
    # node_logits will be filled after we know num_nodes
    # We collect (node_index, head_index) pairs to look up logits later.
    node_head_idx:       List[int]        = [0]           # placeholder uses head 0

    # Current frontier: list of node indices at the latest level
    frontier: List[int] = [0]   # starts at root

    num_levels_active = 0

    # ── BFS expansion ────────────────────────────────────────────────────────
    for head_idx in range(num_heads):
        if not frontier or len(candidates_list) >= max_nodes:
            break

        probs        = head_probs[head_idx]                 # (vocab_size,)
        # Indices of tokens that pass the threshold, sorted descending by prob
        passing_mask = probs > confidence_threshold
        passing_ids  = np.where(passing_mask)[0]
        pruned_total += int((~passing_mask).sum())

        if len(passing_ids) == 0:
            # No token at this level clears the bar — entire subtree pruned
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

    # ── Assemble flat numpy arrays ────────────────────────────────────────────
    num_nodes = len(candidates_list)

    candidates     = np.array(candidates_list, dtype=np.int32)
    parent_indices = np.array(parent_list,     dtype=np.int32)

    # node_logits[i] = the verification logit row for node i.
    # For node i produced by head h, we use head h's raw logit vector as the
    # "what does the model think comes next at this position" signal —
    # exactly what the OpenCL argmax-vs-draft check needs.
    # Row 0 (root) is zero-padded and never used by the kernel.
    node_logits = np.zeros((num_nodes, vocab_size), dtype=np.float32)
    for i in range(1, num_nodes):
        h = node_head_idx[i]
        node_logits[i] = medusa_head_logits[h]

    return DynamicTreeOutput(
        candidates      = candidates,
        parent_indices  = parent_indices,
        node_logits     = node_logits,
        num_nodes       = num_nodes,
        num_levels      = num_levels_active,
        pruned_branches = pruned_total,
    )


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
    # ── Synthetic inputs that mimic one step of medusa_generate() ─────────────
    # vicuna_7b_stage2 uses 5 Medusa heads and vocab_size = 32 000.
    NUM_HEADS  = 5
    VOCAB_SIZE = 32_000
    WARMUP     = 3
    REPEATS    = 20
    # confidence_threshold for the demo.
    # Real Medusa heads produce sharply-peaked distributions (top token often
    # >0.40).  Random normal logits over 32 000 tokens produce a much flatter
    # softmax (~3e-5 per token), so we use a lower threshold here so the demo
    # produces a non-trivial tree.  In production keep this at 0.05–0.20.
    CONF_THR   = 3e-5

    RNG = np.random.default_rng(42)

    # Simulate raw Medusa-head logit outputs: one (vocab_size,) vector per head.
    # In production these come from model.medusa_head[i](hidden_states).
    medusa_head_logits: List[np.ndarray] = [
        RNG.standard_normal(VOCAB_SIZE).astype(np.float32)
        for _ in range(NUM_HEADS)
    ]

    sep = "=" * 68
    print(sep)
    print("  Stage 1 — Dynamic Candidate Tree Builder")
    print(sep)

    # ── Stage 1: build dynamic tree ───────────────────────────────────────────
    tree = build_dynamic_tree(
        medusa_head_logits,
        confidence_threshold = CONF_THR,
        max_nodes            = 256,
    )

    print(f"  confidence_threshold : {CONF_THR}")
    print(f"  num_heads            : {NUM_HEADS}")
    print(f"  vocab_size           : {VOCAB_SIZE}")
    print(f"  ─────────────────────────────────────────")
    print(f"  nodes in tree        : {tree.num_nodes}  "
          f"(incl. root placeholder)")
    print(f"  active levels        : {tree.num_levels}  "
          f"(of {NUM_HEADS} heads)")
    print(f"  branches pruned      : {tree.pruned_branches}")
    print(f"  node_logits shape    : {tree.node_logits.shape}")
    print()
    print("  candidates[:8]      :", tree.candidates[:8].tolist())
    print("  parent_indices[:8]  :", tree.parent_indices[:8].tolist())

    # ── Verification nodes: skip root placeholder (index 0) ──────────────────
    # The root token=-1 is never a real draft; we verify nodes 1..num_nodes-1.
    verify_candidates = tree.candidates[1:]       # int32  (num_nodes-1,)
    verify_logits     = tree.node_logits[1:]      # float32 (num_nodes-1, vocab_size)
    verify_parents    = tree.parent_indices[1:]   # for path tracing, re-index to 0-based
    # Re-index parents so that the verified sub-array is self-consistent:
    # original parent=0 (root) becomes -1 in the verified view.
    verify_parents_reindexed = np.where(
        verify_parents == 0, -1, verify_parents - 1
    ).astype(np.int32)

    num_verify = len(verify_candidates)

    print()
    print(sep)
    print(f"  Stage 2 — OpenCL Verification  ({num_verify} nodes)")
    print(sep)
    print(f"  WG_SIZE  = {WG_SIZE}   (one work-group per node row)")
    print(f"  LDS/WG   = {WG_SIZE * 8} bytes")

    # ── NumPy baseline (the sequential loop being replaced) ──────────────────
    for _ in range(WARMUP):
        run_medusa_verify_python(verify_candidates, verify_logits)

    py_times  = [
        run_medusa_verify_python(verify_candidates, verify_logits)[1]
        for _ in range(REPEATS)
    ]
    py_result = run_medusa_verify_python(verify_candidates, verify_logits)[0]
    py_med    = float(np.median(py_times)) * 1e3
    print(f"\n  [Python/NumPy]           median = {py_med:.4f} ms"
          f"   matches = {py_result.sum()}")

    # ── OpenCL kernel ─────────────────────────────────────────────────────────
    try:
        import pyopencl as cl  # noqa: F401

        print()
        ctx, queue, program = build_opencl_context()

        for _ in range(WARMUP):
            run_medusa_verify_opencl(verify_candidates, verify_logits,
                                     ctx=ctx, queue=queue, program=program)

        cl_times = [
            run_medusa_verify_opencl(verify_candidates, verify_logits,
                                     ctx=ctx, queue=queue, program=program)[1]
            for _ in range(REPEATS)
        ]
        cl_result, _ = run_medusa_verify_opencl(
            verify_candidates, verify_logits,
            ctx=ctx, queue=queue, program=program,
        )
        cl_med = float(np.median(cl_times)) * 1e3

        print(f"\n  [OpenCL local-reduce]    median = {cl_med:.4f} ms"
              f"   matches = {cl_result.sum()}")

        # Correctness
        print()
        if np.array_equal(py_result, cl_result):
            print("  ✓  NumPy and OpenCL results match.")
        else:
            bad = np.where(py_result != cl_result)[0]
            print(f"  ✗  Mismatch at {len(bad)} node(s): {bad}")

        speedup = py_med / cl_med if cl_med > 0 else float("inf")
        print(f"\n  Speedup vs NumPy (kernel time only) : {speedup:.2f}×")

        # ── Stage 3: path tracing ─────────────────────────────────────────────
        print()
        print(sep)
        print("  Stage 3 — Tree Path Tracing")
        print(sep)
        best_node, path_len = trace_longest_path(cl_result, verify_parents_reindexed)

        if path_len > 0:
            # Reconstruct the accepted token sequence for logging
            path_tokens: List[int] = []
            cur = best_node
            while cur != -1:
                path_tokens.append(int(verify_candidates[cur]))
                parent = int(verify_parents_reindexed[cur])
                cur = parent if parent >= 0 else -1
            path_tokens.reverse()

            print(f"  Accepted length : {path_len} token(s)")
            print(f"  Terminal node   : {best_node}")
            print(f"  Token sequence  : {path_tokens}")
        else:
            print("  No valid path found — no speculative tokens accepted.")

        print(sep)

    except ImportError:
        print("\n[OpenCL] pyopencl not found — install with:  pip install pyopencl")
        # Still show path-tracing on the NumPy result so the demo is useful
        if len(py_result) > 0:
            best_node, path_len = trace_longest_path(py_result, verify_parents_reindexed)
            print(f"\n[Path trace on NumPy result]  length={path_len}  node={best_node}")
        else:
            print("\n[Path trace] No nodes to trace (tree was empty).")
    except Exception as exc:
        print(f"\n[OpenCL] Error: {exc}")
        raise



# ─────────────────────────────────────────────────────────────────────────────
#  Tree Path Tracing (Medusa Logic)
# ─────────────────────────────────────────────────────────────────────────────

def trace_longest_path(is_match: np.ndarray, parent_indices: np.ndarray):
    """
    Finds the longest continuous path of correct guesses in the candidate tree.
    A node is only valid if it matches AND its parent is valid.
    
    parent_indices[i] = the index of node i's parent. 
    A parent of -1 means it connects directly to the root (the original prompt).
    """
    valid_path_lengths = np.zeros_like(is_match)
    
    for i in range(len(is_match)):
        if is_match[i] == 1:
            parent = parent_indices[i]
            # If it's attached to the root, it's a valid path of length 1
            if parent == -1:
                valid_path_lengths[i] = 1
            # If its parent is part of a valid path, extend that path by 1
            elif valid_path_lengths[parent] > 0:
                valid_path_lengths[i] = valid_path_lengths[parent] + 1

    # Find the node that ended the longest valid path
    best_node = np.argmax(valid_path_lengths)
    max_length = valid_path_lengths[best_node]
    
    return best_node, max_length


if __name__ == "__main__":
    main()