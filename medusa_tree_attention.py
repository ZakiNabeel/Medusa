
import time
import numpy as np
import torch
from medusa_opencl_verify import run_medusa_verify_opencl, build_opencl_context

# Import directly from the repo
from medusa.model.medusa_choices import mc_sim_7b_63
from medusa.model.utils import generate_medusa_buffers, evaluate_posterior, generate_candidates

def medusa_generate_tree_attention(
    prompt,
    model,           # GPT2LMHeadModel
    medusa_heads,    # SimpleMedusaHeads
    tokenizer,
    max_new_tokens=50,
    verbose=False
):
    """
    Medusa generation using the repo's tree attention buffers.
    Uses mc_sim_7b_63 as the static tree topology (63 nodes).
    GPT-2 forward pass with custom 4D attention mask.
    """
    # ── Setup ────────────────────────────────────────────────────────────────
    input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
    generated = input_ids.clone()

    # Simple 3-head tree compatible with our 3 Medusa heads
    # Each entry is a path: [head0_rank, head1_rank, head2_rank]
    # This is a small hand-crafted tree that works with TOPK=10, num_heads=3
    medusa_3head = [
        [0], [1], [2], [3], [4],           # depth 1: top 5 from head 0
        [0, 0], [0, 1], [0, 2],            # depth 2: top 3 continuations of top-1
        [1, 0], [1, 1],                    # depth 2: top 2 continuations of top-2
        [2, 0],                            # depth 2: top 1 continuation of top-3
        [0, 0, 0], [0, 0, 1],              # depth 3: top 2 from best path
        [0, 1, 0],                         # depth 3: one more
    ]

    # Generate tree buffers once — reused every step
    buffers = generate_medusa_buffers(medusa_3head, device='cuda')
    medusa_attn_mask = buffers['medusa_attn_mask']   # (1, 1, 64, 64)
    tree_indices     = buffers['tree_indices']        # (64,)
    position_ids     = buffers['medusa_position_ids'] # (64,)
    retrieve_indices = buffers['retrieve_indices']    # (num_paths, max_path_len)

    TOPK = 10  # must match what generate_medusa_buffers expects

    ctx, queue, program = build_opencl_context()

    start_time = time.time()
    total_tokens = 0
    step_times = []
    accepted_lengths = []
    kernel_times_event = []

    with torch.no_grad():
        while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
            step_start = time.time()
            prompt_len = generated.shape[1]

            # ── Step 1: Run base model on current sequence ────────────────
            outputs = model(generated, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # (1, seq_len, 768)
            base_logits = outputs.logits               # (1, seq_len, vocab_size)

            # ── Step 2: Get head logits ───────────────────────────────────
            # Heads take last position hidden state
            last_hidden = hidden_states[:, -1:, :]  # (1, 1, 768)
            head_logits_list = []
            for head in medusa_heads.heads:
                head_logits_list.append(head(last_hidden[:, 0, :]))  # (1, vocab_size)
            # Stack into (num_heads, 1, 1, vocab_size) to match repo's format
            medusa_logits = torch.stack(head_logits_list, dim=0).unsqueeze(2)
            # shape: (num_heads, 1, 1, vocab_size)

            # ── Step 3: Generate candidates using repo function ───────────
            # generate_candidates returns:
            #   cart_candidates: (num_paths, path_len) — for evaluate_posterior
            #   tree_candidates: (1, 64) — to feed into tree attention forward pass
            cart_candidates, tree_candidates = generate_candidates(
                medusa_logits,
                base_logits,
                tree_indices,
                retrieve_indices,
                temperature=0,  # greedy
                fast=True,
            )

            # ── Step 4: Build tree attention input ────────────────────────
            tree_len = tree_candidates.shape[1]  # 64
            full_len = prompt_len + tree_len

            # Tree position IDs
            tree_position_ids = position_ids + prompt_len  # (64,)
            prompt_pos = torch.arange(prompt_len, device='cuda').unsqueeze(0)
            full_position_ids = torch.cat(
                [prompt_pos, tree_position_ids.unsqueeze(0)], dim=1
            )  # (1, full_len)

            # Full input: prompt tokens + tree candidate tokens
            full_input = torch.cat([generated, tree_candidates], dim=1)  # (1, full_len)

            # Build 4D attention mask
            # GPT-2 accepts past_key_values + a 2D mask for padding
            # But for custom tree attention we need to go through the raw attention
            # Build (1, 1, full_len, full_len) additive mask
            full_mask = torch.full(
                (1, 1, full_len, full_len),
                torch.finfo(torch.float32).min,  # use min float instead of -inf to avoid NaN
                device='cuda',
                dtype=torch.float32
            )

            # Prompt: causal mask
            # Prompt: causal mask — upper triangle is -inf, lower triangle is 0
            causal_mask = torch.triu(
                torch.ones(prompt_len, prompt_len, device='cuda') * torch.finfo(torch.float32).min,
                diagonal=1
            )
            full_mask[0, 0, :prompt_len, :prompt_len] = causal_mask

            # Tree nodes attend to full prompt
            full_mask[0, 0, prompt_len:, :prompt_len] = 0.0

            # Tree nodes attend to each other per ancestry mask
            tree_mask_bool = medusa_attn_mask[0, 0].bool()  # (64, 64)
            tree_attend_vals = torch.where(
                tree_mask_bool,
                torch.zeros(tree_len, tree_len, device='cuda'),
                torch.full((tree_len, tree_len), torch.finfo(torch.float32).min, device='cuda')
            )
            full_mask[0, 0, prompt_len:, prompt_len:] = tree_attend_vals

            # ── Step 5: Single forward pass with tree attention ───────────
            # GPT-2 TransfoXL-style: pass head_mask=None, use the raw model
            # We need to call the transformer directly to pass our custom mask
            gpt2_transformer = model.transformer

            token_embeds = gpt2_transformer.wte(full_input)          # (1, full_len, 768)
            pos_embeds = gpt2_transformer.wpe(full_position_ids)     # (1, full_len, 768)
            hidden = gpt2_transformer.drop(token_embeds + pos_embeds)

            # Run through each transformer block with our custom attention mask
            for block in gpt2_transformer.h:
                outputs_block = block(
                    hidden,
                    attention_mask=full_mask,
                    use_cache=False,
                )
                hidden = outputs_block[0]

            hidden = gpt2_transformer.ln_f(hidden)

            # Get logits via the LM head (weight tying with embeddings)
            tree_logits_full = model.lm_head(hidden)  # (1, full_len, vocab_size)

            # Extract only tree node positions
            tree_logits = tree_logits_full[0, prompt_len:, :]  # (64, vocab_size)

            # Map tree logits to candidate paths using retrieve_indices
            logits_reshaped = tree_logits[retrieve_indices]  # (num_paths, path_len, vocab_size)

            # ── Step 6: OpenCL verification (our contribution) ────────────
            # Flatten candidates and logits for the kernel
            num_paths, path_len = cart_candidates.shape
            verify_candidates_flat = cart_candidates[:, 1:].reshape(-1).cpu().numpy().astype(np.int32)
            verify_logits_flat = logits_reshaped[:, :-1, :].reshape(-1, model.config.vocab_size).cpu().numpy()
            
            is_match_cl, t_kernel = run_medusa_verify_opencl(
                verify_candidates_flat,
                verify_logits_flat,
                ctx=ctx, queue=queue, program=program
            )
            kernel_times_event.append(t_kernel)

            # ── Step 7: Evaluate posterior using repo function ────────────
            best_candidate, accept_length = evaluate_posterior(
                logits_reshaped,
                cart_candidates,
                temperature=0,
                fast=True,
            )
            accept_length = accept_length.item()

            # ── Step 8: Update generated sequence ────────────────────────
            if accept_length > 0:
                accepted_tokens = cart_candidates[best_candidate, :accept_length + 1]
                generated = torch.cat(
                    [generated, accepted_tokens.unsqueeze(0)], dim=-1
                )
                total_tokens += accept_length + 1
            else:
                # Accept at least the base model's next token
                next_token = base_logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=-1)
                total_tokens += 1

            accepted_lengths.append(accept_length)
            step_times.append(time.time() - step_start)

            if verbose and len(step_times) % 10 == 0:
                avg_acc = np.mean(accepted_lengths[-10:])
                avg_kernel = np.mean(kernel_times_event[-10:]) * 1000
                print(f"Step {len(step_times)}: {total_tokens} tokens, "
                      f"avg_accepted={avg_acc:.2f}, kernel={avg_kernel:.3f}ms")

    elapsed = time.time() - start_time
    tps = total_tokens / elapsed

    return {
        'text': tokenizer.decode(generated[0], skip_special_tokens=True),
        'tokens_per_second': tps,
        'total_tokens': total_tokens,
        'time_seconds': elapsed,
        'avg_step_time_ms': np.mean(step_times) * 1000,
        'avg_accepted_per_step': np.mean(accepted_lengths),
        'acceptance_rate': np.mean([a > 0 for a in accepted_lengths]),
        'avg_kernel_time_ms': np.mean(kernel_times_event) * 1000,
        'num_steps': len(step_times),
    }
