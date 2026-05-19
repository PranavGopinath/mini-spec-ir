"""Greedy inference engine.

generate_vanilla — full recompute each step, O(T²), Phase 2 reference.
generate_kv      — pre-allocated KV cache, O(T_new · T_full) per step, Phase 3.
"""
from __future__ import annotations

import torch

from minispecir.cache import KVCache
from minispecir.model.gpt2 import GPT2Model

EOS_TOKEN_ID = 50256  # GPT-2 <|endoftext|>


def generate_vanilla(
    model: GPT2Model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    eos_token_id: int = EOS_TOKEN_ID,
) -> torch.Tensor:
    """
    Greedy autoregressive generation with full recompute each step.

    Args:
        model:          GPT2Model with weights on its device
        input_ids:      [T] or [1, T] long tensor
        max_new_tokens: maximum tokens to generate beyond the prompt
        eos_token_id:   stop when this token is produced

    Returns:
        [1, T + generated] — prompt + all generated tokens (including EOS if hit)
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    tokens = input_ids.to(model.device)  # [1, T]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model.forward(tokens)      # [1, T_cur, V]
            next_id = logits[0, -1].argmax()    # scalar
            tokens = torch.cat([tokens, next_id.view(1, 1)], dim=-1)
            if next_id.item() == eos_token_id:
                break

    return tokens


def generate_kv(
    model: GPT2Model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    eos_token_id: int = EOS_TOKEN_ID,
) -> torch.Tensor:
    """
    Greedy autoregressive generation with a pre-allocated KV cache.

    Prefill runs a single forward over the full prompt to populate K/V buffers.
    Each decode step processes exactly one new token — no full recompute.
    No torch.cat on K/V tensors in the decode loop.

    Args:
        model:          GPT2Model with weights on its device
        input_ids:      [T] or [1, T] long tensor
        max_new_tokens: maximum tokens to generate beyond the prompt
        eos_token_id:   stop when this token is produced

    Returns:
        [1, T + generated] — prompt + all generated tokens (including EOS if hit)
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    tokens = input_ids.to(model.device)  # [1, T_prompt]
    arch = model.arch
    cache = KVCache.from_arch(arch, B=1, device=model.device)

    with torch.no_grad():
        # Prefill: populate K/V for all prompt positions
        logits = model.forward(tokens, cache=cache)   # [1, T_prompt, V]; cache.past_len → T_prompt

        # First generated token from prefill logits
        next_id = logits[0, -1].argmax()              # scalar
        tokens = torch.cat([tokens, next_id.view(1, 1)], dim=-1)
        if next_id.item() == eos_token_id:
            return tokens

        # Decode loop: one token at a time, no KV recompute
        for _ in range(max_new_tokens - 1):
            logits = model.forward(next_id.view(1, 1), cache=cache)  # [1, 1, V]
            next_id = logits[0, 0].argmax()                          # index 0 (T_new=1)
            tokens = torch.cat([tokens, next_id.view(1, 1)], dim=-1)
            if next_id.item() == eos_token_id:
                break

    return tokens
