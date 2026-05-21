"""Lossless greedy speculative decoding.

Algorithm per step:
  1. Draft phase  — run draft model γ times autoregressively from current_token
  2. Verify phase — run target model over [current_token, draft[0..γ-1]] in ONE forward
  3. Accept/reject — greedy scan: find first position where target disagrees with draft
  4. Commit k+1 tokens; rollback target KV to committed_start + k + 1 if k < γ;
     bonus token (logits[γ]) if all γ accepted.
  5. Sync draft_cache.past_len to target_cache.past_len before next iteration.

Rollback is a direct integer assignment to cache.past_len — no zeroing needed; stale
slots above the new cursor are overwritten by the next verify forward.
"""
from __future__ import annotations

import torch

from minispecir.cache import KVCache
from minispecir.model.gpt2 import GPT2Model

EOS_TOKEN_ID = 50256  # GPT-2 <|endoftext|>


def generate_spec(
    target: GPT2Model,
    draft: GPT2Model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    gamma: int = 4,
    eos_token_id: int = EOS_TOKEN_ID,
) -> tuple[torch.Tensor, dict]:
    """
    Greedy speculative decoding.

    Args:
        target:         Target model (e.g. gpt2)
        draft:          Draft model (e.g. distilgpt2) — must share the same tokenizer vocab
        input_ids:      [T] or [1, T] long tensor
        max_new_tokens: maximum tokens to generate beyond the prompt
        gamma:          number of draft tokens to propose per step
        eos_token_id:   stop when this token is produced

    Returns:
        (output_ids, stats) where output_ids is [1, T + generated] and stats is
        {"steps": int, "total_draft": int, "accepted": int}.
        acceptance_rate = stats["accepted"] / stats["total_draft"] when total_draft > 0.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # [1, T]

    device = target.device
    tokens = input_ids.to(device)

    target_cache = KVCache.from_arch(target.arch, B=1, device=device)
    draft_cache = KVCache.from_arch(draft.arch, B=1, device=device)

    generated_ids: list[int] = []
    stats = {"steps": 0, "total_draft": 0, "accepted": 0}

    with torch.no_grad():
        # --- Prefill both models on the prompt ---
        target_logits = target.forward(tokens, cache=target_cache)  # past_len → T_prompt
        draft.forward(tokens, cache=draft_cache)                     # past_len → T_prompt

        # First committed token from target prefill (K/V not yet in either cache)
        current_token: torch.Tensor = target_logits[0, -1].argmax()  # scalar

        if current_token.item() == eos_token_id:
            output = torch.cat([tokens, current_token.view(1, 1)], dim=-1)
            return output, stats

        generated_ids.append(current_token.item())

        # --- Speculative loop ---
        while len(generated_ids) < max_new_tokens:
            # Snapshot committed cursor before verify (advance fires inside forward)
            committed_start = target_cache.past_len

            # -- Draft phase --
            draft_tokens: list[torch.Tensor] = []
            x = current_token.view(1, 1).to(device)
            for _ in range(gamma):
                d_logits = draft.forward(x, cache=draft_cache)   # advances draft_cache by 1
                d_tok = d_logits[0, 0].argmax()
                draft_tokens.append(d_tok)
                x = d_tok.view(1, 1)
                if d_tok.item() == eos_token_id:
                    break

            actual_gamma = len(draft_tokens)

            # -- Verify phase --
            # verify_ids: [1, actual_gamma + 1] = [current_token, draft[0..γ-1]]
            verify_ids = torch.cat(
                [current_token.view(1, 1),
                 torch.stack(draft_tokens).view(1, -1)],
                dim=-1,
            ).to(device)
            # One forward: advances target_cache by actual_gamma + 1
            verify_logits = target.forward(verify_ids, cache=target_cache)  # [1, γ+1, V]

            # -- Accept/reject scan --
            accept_len = 0
            for i in range(actual_gamma):
                target_choice = verify_logits[0, i].argmax()
                if target_choice.item() == draft_tokens[i].item():
                    accept_len += 1
                else:
                    break

            stats["steps"] += 1
            stats["total_draft"] += actual_gamma
            stats["accepted"] += accept_len

            if accept_len == actual_gamma:
                # All draft tokens accepted; bonus token from logits[γ]
                bonus = verify_logits[0, actual_gamma].argmax()
                # target_cache already at committed_start + actual_gamma + 1 — correct
                draft_cache.past_len = target_cache.past_len
                next_token = bonus
                # Emit all draft tokens + bonus
                for dt in draft_tokens:
                    if len(generated_ids) >= max_new_tokens:
                        break
                    generated_ids.append(dt.item())
                if len(generated_ids) < max_new_tokens:
                    generated_ids.append(next_token.item())
            else:
                # Mismatch at position accept_len; take target's correction
                next_token = verify_logits[0, accept_len].argmax()
                # Rollback: only committed_start + accept_len + 1 positions are valid
                target_cache.past_len = committed_start + accept_len + 1
                draft_cache.past_len = target_cache.past_len
                # Emit accepted draft tokens + correction
                for i in range(accept_len):
                    if len(generated_ids) >= max_new_tokens:
                        break
                    generated_ids.append(draft_tokens[i].item())
                if len(generated_ids) < max_new_tokens:
                    generated_ids.append(next_token.item())

            # Check EOS in the newly emitted tokens
            if generated_ids and generated_ids[-1] == eos_token_id:
                break

            # Trim to max_new_tokens
            if len(generated_ids) >= max_new_tokens:
                break

            current_token = torch.tensor(generated_ids[-1], dtype=torch.long, device=device)

    output = torch.cat(
        [tokens, torch.tensor([generated_ids], dtype=torch.long, device=device)],
        dim=-1,
    )
    return output, stats
