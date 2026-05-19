"""Greedy inference engine — Phase 2: full recompute (no KV cache).

Each decode step runs the full forward pass over all tokens so far.
This is correct but O(T²) in compute; Phase 3 replaces it with KV cache.
"""
from __future__ import annotations

import torch

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
