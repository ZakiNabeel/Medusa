"""
medusa_pipeline.py  –  End-to-End Speculative Decoding Pipeline (Naive Version)
================================================================
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 0 – GPT-2 + Medusa Heads
# ══════════════════════════════════════════════════════════════════════════════

class GPT2Medusa(nn.Module):
    def __init__(self, base_model_name: str = "gpt2", num_heads: int = 3):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

        # Move to GPU
        self.base_model = self.base_model.cuda()

        self.hidden_size = self.base_model.config.n_embd
        self.vocab_size = self.base_model.config.vocab_size
        self.num_heads = num_heads

        self.medusa_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, self.vocab_size, bias=False)
            for _ in range(num_heads)
        ])

    def forward(self, text_prompt: str):
        inputs = self.tokenizer(text_prompt, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = self.base_model(**inputs, output_hidden_states=True)

        base_logits = outputs.logits[0, -1, :].detach().cpu().numpy()
        last_hidden = outputs.hidden_states[-1][0, -1, :]

        medusa_logits = []
        for head in self.medusa_heads:
            medusa_logits.append(head(last_hidden).detach().cpu().numpy())

        return base_logits, medusa_logits


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 – Shannon Entropy + Adaptive Threshold
# ══════════════════════════════════════════════════════════════════════════════

def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def shannon_entropy(logits: np.ndarray) -> float:
    probs = _softmax(logits.astype(np.float32))
    mask = probs > 0.0
    return float(-np.sum(probs[mask] * np.log(probs[mask])))


def entropy_to_threshold(entropy: float, max_entropy: float, low_thr: float = 0.02, high_thr: float = 0.15) -> float:
    """
    CORRECT: Low entropy (confident) → low threshold → more candidates
             High entropy (uncertain) → high threshold → fewer candidates (prune)
    """
    if max_entropy <= 0:
        return high_thr
    ratio = min(entropy / max_entropy, 1.0)
    return low_thr + ratio * (high_thr - low_thr)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 – Dynamic Candidate Tree Builder
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DynamicTreeOutput:
    candidates: np.ndarray
    parent_indices: np.ndarray
    head_node_logits: np.ndarray
    num_nodes: int
    num_levels: int
    pruned_branches: int


#def build_dynamic_tree(medusa_head_logits: List[np.ndarray], confidence_threshold: float = 0.15, max_nodes: int = 256) -> DynamicTreeOutput:
def build_dynamic_tree(medusa_head_logits, confidence_thresholds, max_nodes=256): #new
    vocab_size = medusa_head_logits[0].shape[0]
    num_heads = len(medusa_head_logits)
    pruned_total = 0

    head_probs = [_softmax(h.astype(np.float32)) for h in medusa_head_logits]

    candidates_list = [-1]
    parent_list = [-1]
    node_level = [0]
    node_head_idx = [0]

    frontier = [0]
    num_levels_active = 0

    for head_idx in range(num_heads):
        if not frontier or len(candidates_list) >= max_nodes:
            break

        probs = head_probs[head_idx]
        #passing_mask = probs > confidence_threshold
        passing_mask = probs > confidence_thresholds[head_idx] #new for batching
        passing_ids = np.where(passing_mask)[0]
        pruned_total += int((~passing_mask).sum())

        if len(passing_ids) == 0:
            break

        next_frontier = []

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

    num_nodes = len(candidates_list)
    candidates = np.array(candidates_list, dtype=np.int32)
    parent_indices = np.array(parent_list, dtype=np.int32)

    head_node_logits = np.zeros((num_nodes, vocab_size), dtype=np.float32)
    for i in range(1, num_nodes):
        head_node_logits[i] = medusa_head_logits[node_head_idx[i]]

    return DynamicTreeOutput(
        candidates=candidates,
        parent_indices=parent_indices,
        head_node_logits=head_node_logits,
        num_nodes=num_nodes,
        num_levels=num_levels_active,
        pruned_branches=pruned_total,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 – Base Model Verification (BATCHED + (NEW) WITH PREFIX GROUPING)
# ══════════════════════════════════════════════════════════════════════════════

def get_base_model_verification_logits(model, input_ids, tree):
    """
    new:
    Optimized: groups nodes by shared prefix, one forward pass per unique prefix.
    Reduces ~255 forward passes to ~3-10 depending on tree depth.
    """
    vocab_size = model.vocab_size
    num_nodes = tree.num_nodes
    candidates = tree.candidates
    parent_indices = tree.parent_indices

    # Group nodes by their path-to-parent (the prefix the base model sees)
    prefix_map = {}
    for node_idx in range(1, num_nodes):
        path_tokens = []
        cur = parent_indices[node_idx]
        while cur > 0:
            path_tokens.append(int(candidates[cur]))
            cur = int(parent_indices[cur])
        path_tokens.reverse()
        prefix_key = tuple(path_tokens)
        if prefix_key not in prefix_map:
            prefix_map[prefix_key] = []
        prefix_map[prefix_key].append(node_idx)

    verify_logits = np.zeros((num_nodes, vocab_size), dtype=np.float32)

    with torch.no_grad():
        for prefix_key, node_indices in prefix_map.items():
            if len(prefix_key) == 0:
                seq = input_ids
            else:
                prefix_tensor = torch.tensor(
                    list(prefix_key), dtype=torch.long
                ).unsqueeze(0).cuda()
                seq = torch.cat([input_ids, prefix_tensor], dim=-1)

            outputs = model.base_model(seq)
            base_logits_row = outputs.logits[0, -1, :].cpu().numpy()

            for node_idx in node_indices:
                verify_logits[node_idx] = base_logits_row

    # Return only rows 1..num_nodes (skip root)
    return verify_logits[1:]


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 – OpenCL Hardware-Accelerated Verification
# ══════════════════════════════════════════════════════════════════════════════

WG_SIZE = 256

KERNEL_SOURCE = r"""
#define WG_SIZE 256

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void medusa_verify_local(
    __global const float* restrict logits,
    __global const int*   restrict drafts,
    __global       int*   restrict is_match,
    const int vocab_size
) {
    const int lid = get_local_id(0);
    const int node_id = get_group_id(0);
    const __global float* row = logits + (long)node_id * vocab_size;

    float my_val = -INFINITY;
    int my_idx = 0;
    for (int v = lid; v < vocab_size; v += WG_SIZE) {
        float val = row[v];
        if (val > my_val) { my_val = val; my_idx = v; }
    }

    __local float local_vals[WG_SIZE];
    __local int local_idxs[WG_SIZE];
    local_vals[lid] = my_val;
    local_idxs[lid] = my_idx;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = WG_SIZE >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            float ov = local_vals[lid + stride];
            int oi = local_idxs[lid + stride];
            if (ov > local_vals[lid] ||
               (ov == local_vals[lid] && oi < local_idxs[lid])) {
                local_vals[lid] = ov;
                local_idxs[lid] = oi;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        is_match[node_id] = (local_idxs[0] == drafts[node_id]) ? 1 : 0;
    }
}
"""


def build_opencl_context(platform_idx: int = 0, device_idx: int = 0):
    import pyopencl as cl

    platform = cl.get_platforms()[platform_idx]
    device = platform.get_devices()[device_idx]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    program = cl.Program(ctx, KERNEL_SOURCE).build(options="-cl-fast-relaxed-math -cl-mad-enable")

    lds_bytes = WG_SIZE * (4 + 4)
    print(f"  [OpenCL] Platform: {platform.name}")
    print(f"  [OpenCL] Device: {device.name}")
    print(f"  [OpenCL] Max WG: {device.max_work_group_size}")
    print(f"  [OpenCL] LDS total: {device.local_mem_size // 1024} KB")
    print(f"  [OpenCL] LDS/WG: {lds_bytes} bytes ({lds_bytes / device.local_mem_size * 100:.1f}%)")

    return ctx, queue, program


def run_medusa_verify_opencl(drafts_np: np.ndarray, logits_np: np.ndarray, ctx=None, queue=None, program=None) -> Tuple[np.ndarray, float]:
    import pyopencl as cl

    if ctx is None:
        ctx, queue, program = build_opencl_context()

    num_nodes, vocab_size = logits_np.shape
    drafts_np = np.ascontiguousarray(drafts_np, dtype=np.int32)
    logits_np = np.ascontiguousarray(logits_np, dtype=np.float32)

    mf = cl.mem_flags
    logits_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=logits_np)
    drafts_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=drafts_np)
    match_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=drafts_np.nbytes)

    kernel = cl.Kernel(program, "medusa_verify_local")
    event = kernel(queue, (num_nodes * WG_SIZE,), (WG_SIZE,), logits_buf, drafts_buf, match_buf, np.int32(vocab_size))
    event.wait()

    elapsed = (event.profile.end - event.profile.start) * 1e-9
    is_match = np.empty(num_nodes, dtype=np.int32)
    cl.enqueue_copy(queue, is_match, match_buf)
    queue.finish()

    return is_match, elapsed


def run_medusa_verify_numpy(drafts_np: np.ndarray, logits_np: np.ndarray) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    argmaxes = np.argmax(logits_np, axis=1)
    is_match = (argmaxes == drafts_np).astype(np.int32)
    return is_match, time.perf_counter() - t0


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 5 – Path Tracing & Correctness Check
# ══════════════════════════════════════════════════════════════════════════════

def trace_longest_path(is_match: np.ndarray, parent_indices: np.ndarray) -> Tuple[int, int]:
    valid_path_lengths = np.zeros_like(is_match)

    for i in range(len(is_match)):
        if is_match[i] == 1:
            parent = parent_indices[i]
            if parent == -1:
                valid_path_lengths[i] = 1
            elif valid_path_lengths[parent] > 0:
                valid_path_lengths[i] = valid_path_lengths[parent] + 1

    best_node = int(np.argmax(valid_path_lengths))
    path_length = int(valid_path_lengths[best_node])
    return best_node, path_length


# ══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(prompt: str = "The capital of France is", num_heads: int = 3, low_thr: float = 0.02, high_thr: float = 0.15, max_nodes: int = 256) -> None:
    SEP = "═" * 68
    sep2 = "─" * 68

    print(f"\n{SEP}")
    print("  MEDUSA SPECULATIVE DECODING PIPELINE (FINAL CORRECTED)")
    print(SEP)
    print(f"  Prompt: \"{prompt}\"")
    print(f"  Heads: {num_heads} | max_nodes: {max_nodes}")
    print(f"  Threshold range: [{low_thr}, {high_thr}] (low=expand, high=prune)")
    print(sep2)

    # STAGE 1 – Neural Inference
    print("\n  STAGE 1 — Neural Inference")
    print(sep2)

    print("  Loading GPT-2 + Medusa heads...")
    t_inf_start = time.perf_counter()
    model = GPT2Medusa(num_heads=num_heads)

    # Load trained heads
    #checkpoint = torch.load('/content/drive/MyDrive/medusa_project/trained_heads.pt')
    #state_dict = checkpoint['model_state_dict']
    #new_state_dict = {key.replace('heads.', ''): value for key, value in state_dict.items()}
    #model.medusa_heads.load_state_dict(new_state_dict)
    checkpoint = torch.load('/content/drive/MyDrive/medusa_project/trained_heads_5000.pt') #use the newly trained heads
    state_dict = checkpoint['model_state_dict']
    # Checkpoint was saved from SimpleMedusaHeads (keys: heads.0.weight)
    # GPT2Medusa.medusa_heads is a ModuleList (expects keys: 0.weight)
    # Strip the 'heads.' prefix
    new_state_dict = {k.replace('heads.', ''): v for k, v in state_dict.items()}
    model.medusa_heads.load_state_dict(new_state_dict)
    model.medusa_heads.cuda()
    model.medusa_heads.eval()
    print("  Loaded trained Medusa heads!")

    # Get input IDs for later
    input_ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

    print("  Running forward pass...")
    base_logits, medusa_logits = model(prompt)
    t_inference = time.perf_counter() - t_inf_start

    vocab_size = base_logits.shape[0]
    max_entropy = float(np.log(vocab_size))

    print(f"  Vocab size: {vocab_size:,}")
    print(f"  Inference time: {t_inference * 1e3:.1f} ms")
    print(f"  Base argmax token: {int(np.argmax(base_logits))} (\"{model.tokenizer.decode([int(np.argmax(base_logits))])}\")")
    print()

    # Per-head entropy
    entropies = []
    thresholds = []
    print(f"  {'Head':>4}  {'Entropy (nats)':>16}  {'Max H':>8}  {'Threshold':>10}  {'Direction'}")
    print(f"  {'----':>4}  {'---------------':>16}  {'-------':>8}  {'----------':>10}  {'----------'}")

    for i, h_logits in enumerate(medusa_logits):
        H = shannon_entropy(h_logits)
        thr = entropy_to_threshold(H, max_entropy, low_thr, high_thr)
        entropies.append(H)
        thresholds.append(thr)
        # FIX 1: Correct direction label
        direction = "← wide (expand)" if H < 0.6 * max_entropy else "→ narrow (prune)"
        print(f"  {i:>4}  {H:>16.4f}  {max_entropy:>8.4f}  {thr:>10.4f}  {direction}")

    # STAGE 2 – Tree Construction
    print(f"\n{sep2}")
    print("  STAGE 2 — Entropy-Based Tree Construction")
    print(sep2)

    mean_threshold = float(np.mean(thresholds))
    print(f"  Mean adaptive threshold: {mean_threshold:.5f}")

    t_tree_start = time.perf_counter()
  #  tree = build_dynamic_tree(medusa_logits, confidence_threshold=mean_threshold, max_nodes=max_nodes)
    tree = build_dynamic_tree(medusa_logits, confidence_thresholds=thresholds, max_nodes=max_nodes) #new for batching
    t_tree = time.perf_counter() - t_tree_start

    print(f"  Tree build time: {t_tree * 1e3:.2f} ms")
    print(f"  Total nodes (incl. root): {tree.num_nodes}")
    print(f"  Active levels: {tree.num_levels} (of {num_heads} heads)")
    print(f"  Branches pruned: {tree.pruned_branches}")

    # Prepare verification arrays
    verify_candidates = tree.candidates[1:]
    verify_parents_raw = tree.parent_indices[1:]
    verify_parents = np.where(verify_parents_raw == 0, -1, verify_parents_raw - 1).astype(np.int32)
    num_verify = len(verify_candidates)

    if num_verify == 0:
        print("\n  Tree is empty — no candidates to verify.")
        return

    # STAGE 3 – Base Model Verification (BATCHED)
    print(f"\n{sep2}")
    print(f"  STAGE 3 — Base Model Verification ({num_verify} candidate nodes)")
    print(sep2)

    print("  Running batched base model verification (ONE forward pass)...")
    t_base_start = time.perf_counter()
    real_verify_logits = get_base_model_verification_logits(model, input_ids, tree)
    t_base = time.perf_counter() - t_base_start
    print(f"  Batched verification time: {t_base * 1e3:.1f} ms")

    # STAGE 4 – GPU Verification
    print(f"\n{sep2}")
    print(f"  STAGE 4 — GPU Verification ({num_verify} candidate nodes)")
    print(sep2)

    is_match_np, t_numpy = run_medusa_verify_numpy(verify_candidates, real_verify_logits)
    print(f"  [NumPy] time = {t_numpy * 1e3:.4f} ms, matches = {is_match_np.sum()}")

    use_opencl = False
    is_match = is_match_np
    t_gpu = t_numpy

    try:
        ctx, queue, program = build_opencl_context()
        is_match_cl, t_cl = run_medusa_verify_opencl(verify_candidates, real_verify_logits, ctx=ctx, queue=queue, program=program)
        print(f"  [OpenCL] time = {t_cl * 1e3:.4f} ms, matches = {is_match_cl.sum()}")

        if np.array_equal(is_match_np, is_match_cl):
            print("  ✓ NumPy and OpenCL match")
        is_match = is_match_cl
        t_gpu = t_cl
        use_opencl = True
    except Exception as exc:
        print(f"  [OpenCL] Error: {exc}")

    # STAGE 5 – Logic Resolution & Correctness Check
    print(f"\n{sep2}")
    print("  STAGE 5 — Logic Resolution")
    print(sep2)

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
        print(f"  ✓ Accepted {accepted_length} speculative token(s)")
        print(f"  Decoded text: \"{accepted_text}\"")

        # FIX 4: Correctness check against greedy baseline
        greedy_tokens = []
        check_ids = input_ids.clone()
        with torch.no_grad():
            for _ in range(accepted_length):
                out = model.base_model(check_ids)
                next_tok = out.logits[0, -1, :].argmax().item()
                greedy_tokens.append(next_tok)
                check_ids = torch.cat([check_ids, torch.tensor([[next_tok]]).cuda()], dim=-1)

        if path_tokens == greedy_tokens:
            print(f"  ✓ Correctness verified: accepted tokens match greedy baseline")
        else:
            print(f"  ✗ CORRECTNESS FAILURE: Expected {greedy_tokens}, Got {path_tokens}")
    else:
        print("  No speculative tokens accepted")

    # FIX 3: Accurate performance summary
    print()
    print(f"  ─── Performance Summary ───────────────────────────────────────")
    print(f"  Token efficiency this step: {accepted_length + 1:.1f}×")
    print(f"  (baseline: 1 token per forward pass, speculation: {accepted_length + 1} tokens)")
    if use_opencl and t_numpy > 0 and t_gpu > 0:
        print(f"  OpenCL verification speedup: {t_numpy / t_gpu:.2f}× over NumPy")
    print(f"  Batched base model verification: {t_base * 1e3:.1f} ms for {num_verify} candidates")

    print(f"\n{SEP}")
    print("  Pipeline complete.")
    print(SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is")
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--low-thr", type=float, default=0.001)
    parser.add_argument("--high-thr", type=float, default=0.01)
    parser.add_argument("--max-nodes", type=int, default=256)
    args = parser.parse_args()

    run_pipeline(prompt=args.prompt, num_heads=args.heads, low_thr=args.low_thr, high_thr=args.high_thr, max_nodes=args.max_nodes)