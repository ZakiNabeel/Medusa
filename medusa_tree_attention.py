"""
medusa_tree_attention.py — Full Generation Loop with Tree Attention
===================================================================
Milestone 3 implementation.
Uses repo's generate_medusa_buffers and evaluate_posterior.
Single GPT-2 forward pass over all tree candidates simultaneously.
Supports both hand-crafted and calibrated tree topologies.

PATCH (entropy pruning):
  - MedusaEntropyCalculator is constructed once alongside the existing
    OpenCL context so both kernels share a single device queue.
  - After the tree forward pass produces per-node logits, we compute
    entropy for each of the 15 candidate nodes and build a prune mask.
  - Logits belonging to pruned nodes are masked to -inf BEFORE
    evaluate_posterior, so the accept/reject step never considers them.
  - Pruning stats are collected and included in the returned metrics dict.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F

from medusa_opencl_verify import run_medusa_verify_opencl, build_opencl_context
from medusa_entropy import MedusaEntropyCalculator          # ← NEW
from medusa.model.utils import generate_medusa_buffers, evaluate_posterior, generate_candidates

# Hand-crafted 3-head tree (15 nodes)
MEDUSA_3HEAD_HANDCRAFTED = [
    [0], [1], [2], [3], [4],
    [0, 0], [0, 1], [0, 2],
    [1, 0], [1, 1],
    [2, 0],
    [0, 0, 0], [0, 0, 1],
    [0, 1, 0],
]

MEDUSA_3HEAD_CALIBRATED = None  # filled in after calibration runs


def medusa_generate_tree_attention(
    prompt,
    model,
    medusa_heads,
    tokenizer,
    max_new_tokens=50,
    verbose=False,
    tree_topology=None,
    # ── Entropy pruning controls ─────────────────────────────────────
    entropy_threshold: float = 3.0,
    # Set False to disable pruning without code changes (useful for ablation)
    enable_entropy_pruning: bool = True,
):
    """
    Full Medusa generation with tree attention + entropy-based pruning.

    Args:
        tree_topology: list of paths to use as tree. If None, uses
                       MEDUSA_3HEAD_HANDCRAFTED.
        entropy_threshold: nodes whose entropy > this are masked out before
                           evaluate_posterior.  ln(50257) ≈ 10.82.
                           Start at 3.0 and tune per your acceptance curves.
        enable_entropy_pruning: kill-switch for A/B comparisons.
    """
    tree = tree_topology if tree_topology is not None else MEDUSA_3HEAD_HANDCRAFTED

    input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
    generated = input_ids.clone()

    buffers = generate_medusa_buffers(tree, device='cuda')
    medusa_attn_mask = buffers['medusa_attn_mask']
    tree_indices     = buffers['tree_indices']
    position_ids     = buffers['medusa_position_ids']
    retrieve_indices = buffers['retrieve_indices']

    # ── OpenCL context (shared between verify and entropy kernels) ────────
    ctx, queue, program = build_opencl_context()

    # ── NEW: construct entropy calculator on the SAME ctx/queue ──────────
    entropy_calc = MedusaEntropyCalculator(
        ctx=ctx,
        queue=queue,
        entropy_threshold=entropy_threshold,
    ) if enable_entropy_pruning else None

    start_time = time.time()
    total_tokens = 0
    step_times = []
    accepted_lengths = []
    kernel_times_event = []

    # ── Pruning bookkeeping ───────────────────────────────────────────────
    pruned_counts: list[int] = []           # nodes pruned per step
    entropy_values: list[np.ndarray] = []   # per-step entropy vectors

    with torch.no_grad():
        while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
            step_start = time.time()
            prompt_len = generated.shape[1]

            # Step 1: base model forward pass
            outputs = model(generated, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            base_logits = outputs.logits

            # Step 2: head logits
            last_hidden = hidden_states[:, -1:, :]
            head_logits_list = []
            for head in medusa_heads.heads:
                head_logits_list.append(head(last_hidden[:, 0, :]))
            medusa_logits = torch.stack(head_logits_list, dim=0).unsqueeze(2)

            # Step 3: generate candidates
            cart_candidates, tree_candidates = generate_candidates(
                medusa_logits,
                base_logits,
                tree_indices,
                retrieve_indices,
                temperature=0,
                fast=True,
            )

            # Step 4: build attention mask and position IDs
            tree_len = tree_candidates.shape[1]
            full_len = prompt_len + tree_len

            tree_position_ids = position_ids + prompt_len
            prompt_pos = torch.arange(prompt_len, device='cuda').unsqueeze(0)
            full_position_ids = torch.cat(
                [prompt_pos, tree_position_ids.unsqueeze(0)], dim=1
            )

            full_input = torch.cat([generated, tree_candidates], dim=1)

            full_mask = torch.full(
                (1, 1, full_len, full_len),
                torch.finfo(torch.float32).min,
                device='cuda',
                dtype=torch.float32
            )

            causal_mask = torch.triu(
                torch.ones(prompt_len, prompt_len, device='cuda') * torch.finfo(torch.float32).min,
                diagonal=1
            )
            full_mask[0, 0, :prompt_len, :prompt_len] = causal_mask
            full_mask[0, 0, prompt_len:, :prompt_len] = 0.0

            tree_mask_bool = medusa_attn_mask[0, 0].bool()
            tree_attend_vals = torch.where(
                tree_mask_bool,
                torch.zeros(tree_len, tree_len, device='cuda'),
                torch.full((tree_len, tree_len), torch.finfo(torch.float32).min, device='cuda')
            )
            full_mask[0, 0, prompt_len:, prompt_len:] = tree_attend_vals

            # Step 5: single forward pass through transformer blocks
            gpt2_transformer = model.transformer
            token_embeds = gpt2_transformer.wte(full_input)
            pos_embeds   = gpt2_transformer.wpe(full_position_ids)
            hidden       = gpt2_transformer.drop(token_embeds + pos_embeds)

            for block in gpt2_transformer.h:
                hidden = block(hidden, attention_mask=full_mask, use_cache=False)[0]

            hidden = gpt2_transformer.ln_f(hidden)
            tree_logits_full = model.lm_head(hidden)
            tree_logits = tree_logits_full[0, prompt_len:, :]

            logits_reshaped = tree_logits[retrieve_indices]
            # logits_reshaped: (num_paths, path_len, vocab_size)

            # ── ─── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ────
            # PATCH: Entropy-based branch pruning
            #
            # logits_reshaped has shape (num_paths, path_len, vocab).
            # We need per-NODE entropy, not per-path.  The tree has
            # `tree_len` nodes; their logits sit at tree_logits[0..tree_len].
            # We use tree_logits directly (shape: tree_len × vocab) so we
            # avoid re-indexing through retrieve_indices.
            #
            # Use the fused logits_to_entropy kernel: no round-trip softmax.
            # ── ─── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ────
            if entropy_calc is not None:
                node_logits_np = tree_logits.cpu().numpy()   # (tree_len, vocab)

                # Fused softmax+entropy — single OpenCL dispatch ──────────
                node_entropies = entropy_calc.calculate_from_logits(node_logits_np)
                # node_entropies: float32 ndarray of shape (tree_len,)

                # Boolean mask: True = high entropy → prune this node
                prune_mask = entropy_calc.prune_mask(node_entropies)
                # shape: (tree_len,)

                entropy_values.append(node_entropies)
                pruned_counts.append(int(prune_mask.sum()))

                # Map node prune_mask → path prune_mask via retrieve_indices
                # retrieve_indices: (num_paths, path_len) — each element is a
                # flat node index into tree_logits.
                # A path is pruned if ANY position along it maps to a pruned node.
                # ─────────────────────────────────────────────────────────────
                # retrieve_indices lives on CUDA; bring to CPU for numpy indexing
                ri_cpu = retrieve_indices.cpu().numpy()           # (num_paths, path_len)
                path_prune_mask = prune_mask[ri_cpu].any(axis=1) # (num_paths,)

                if path_prune_mask.any() and not path_prune_mask.all():
                    # Mask out pruned paths in logits_reshaped so that
                    # evaluate_posterior assigns them probability -inf.
                    # We set the *entire logit row* for every position in
                    # the pruned path to -inf so argmax / softmax ignores them.
                    pruned_idx = torch.from_numpy(
                        np.where(path_prune_mask)[0]
                    ).to(logits_reshaped.device)

                    logits_reshaped = logits_reshaped.clone()       # avoid in-place on leaf
                    logits_reshaped[pruned_idx] = torch.finfo(torch.float32).min

                    # Also zero out the corresponding candidate rows so
                    # evaluate_posterior can't accidentally select them via
                    # the greedy-fallback path.
                    cart_candidates = cart_candidates.clone()
                    cart_candidates[pruned_idx] = -1   # sentinel value

                elif path_prune_mask.all():
                    # Every branch is high-entropy → no speculation benefit.
                    # Skip evaluate_posterior and fall through to greedy step.
                    next_token = base_logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
                    generated = torch.cat([generated, next_token], dim=-1)
                    total_tokens += 1
                    accepted_lengths.append(0)
                    step_times.append(time.time() - step_start)
                    continue
            else:
                pruned_counts.append(0)
            # ── END PATCH ─────────────────────────────────────────────────

            # Step 6: OpenCL verification kernel (unchanged)
            num_paths, path_len = cart_candidates.shape
            verify_candidates_flat = cart_candidates[:, 1:].reshape(-1).cpu().numpy().astype(np.int32)
            verify_logits_flat = logits_reshaped[:, :-1, :].reshape(-1, model.config.vocab_size).cpu().numpy()

            is_match_cl, t_kernel = run_medusa_verify_opencl(
                verify_candidates_flat,
                verify_logits_flat,
                ctx=ctx, queue=queue, program=program
            )
            kernel_times_event.append(t_kernel)

            # Step 7: evaluate posterior
            best_candidate, accept_length = evaluate_posterior(
                logits_reshaped,
                cart_candidates,
                temperature=0,
                fast=True,
            )
            accept_length = accept_length.item()

            # Step 8: update sequence
            if accept_length > 0:
                accepted_tokens = cart_candidates[best_candidate, :accept_length + 1]
                generated = torch.cat([generated, accepted_tokens.unsqueeze(0)], dim=-1)
                total_tokens += accept_length + 1
            else:
                next_token = base_logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=-1)
                total_tokens += 1

            accepted_lengths.append(accept_length)
            step_times.append(time.time() - step_start)

            if verbose and len(step_times) % 10 == 0:
                avg_pruned = np.mean(pruned_counts[-10:]) if pruned_counts else 0
                print(f"Step {len(step_times)}: {total_tokens} tokens, "
                      f"avg_accepted={np.mean(accepted_lengths[-10:]):.2f}, "
                      f"kernel={np.mean(kernel_times_event[-10:])*1000:.3f}ms, "
                      f"avg_pruned_nodes={avg_pruned:.1f}")

    elapsed = time.time() - start_time

    return {
        'text': tokenizer.decode(generated[0], skip_special_tokens=True),
        'tokens_per_second': total_tokens / elapsed,
        'total_tokens': total_tokens,
        'time_seconds': elapsed,
        'avg_step_time_ms': np.mean(step_times) * 1000,
        'avg_accepted_per_step': np.mean(accepted_lengths),
        'acceptance_rate': np.mean([a > 0 for a in accepted_lengths]),
        'avg_kernel_time_ms': np.mean(kernel_times_event) * 1000,
        'num_steps': len(step_times),
        'tree_size': len(tree),
        # ── Entropy pruning diagnostics ────────────────────────────
        'avg_pruned_nodes_per_step': float(np.mean(pruned_counts)) if pruned_counts else 0.0,
        'pruning_rate': float(np.mean([p > 0 for p in pruned_counts])) if pruned_counts else 0.0,
        'entropy_calc_avg_ms': entropy_calc.avg_kernel_time_ms if entropy_calc else 0.0,
        'entropy_threshold_used': entropy_threshold,
    }