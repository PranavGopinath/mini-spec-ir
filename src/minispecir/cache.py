"""Pre-allocated KV cache for autoregressive decoding.

Supports both MHA (GPT-2, n_kv_head == n_head) and GQA (Llama, n_kv_head < n_head).
Buffers are allocated once at construction; each decode step writes K/V
in-place via index assignment — no torch.cat in the hot loop.
"""
from __future__ import annotations

import torch


class KVCache:
    """
    Per-layer key/value cache for one batch entry.

    Shapes:
        k_cache, v_cache: [n_layer, B, n_kv_head, max_seq, head_dim]
        past_len: int — number of token positions committed so far

    For MHA (GPT-2): n_kv_head == n_head.
    For GQA (Llama): n_kv_head < n_head; the model expands KV heads before attention.

    Protocol:
        1. model.forward() calls cache.write(layer, k_new, v_new) then cache.read(layer, T_new)
           for each attention block — past_len is NOT advanced here.
        2. model.forward() calls cache.advance(T_new) once, after all blocks.
    """

    def __init__(
        self,
        n_layer: int,
        B: int,
        n_kv_head: int,
        max_seq: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (n_layer, B, n_kv_head, max_seq, head_dim)
        self.k_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.past_len: int = 0
        self.max_seq = max_seq

    def write(
        self,
        layer_idx: int,
        k: torch.Tensor,  # [B, n_head, T_new, head_dim]
        v: torch.Tensor,  # [B, n_head, T_new, head_dim]
    ) -> None:
        """Write new K/V at the current cursor. Does not advance past_len."""
        T_new = k.shape[2]
        end = self.past_len + T_new
        assert end <= self.max_seq, (
            f"KV cache overflow: {end} > {self.max_seq}. "
            "Prompt + generated tokens exceed max_seq_len."
        )
        self.k_cache[layer_idx, :, :, self.past_len : end, :] = k
        self.v_cache[layer_idx, :, :, self.past_len : end, :] = v

    def read(
        self,
        layer_idx: int,
        T_new: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return full K/V for attention: all past positions + the newly written ones.

        Returns:
            k, v — each [B, n_head, past_len + T_new, head_dim], contiguous.
        """
        end = self.past_len + T_new
        k = self.k_cache[layer_idx, :, :, :end, :].contiguous()
        v = self.v_cache[layer_idx, :, :, :end, :].contiguous()
        return k, v

    def advance(self, T: int) -> None:
        """Commit T new positions. Called once per forward() after all layers."""
        self.past_len += T

    def reset(self) -> None:
        """Zero buffers and reset cursor. Allows reuse across sequences."""
        self.past_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()

    @classmethod
    def from_arch(
        cls,
        arch: object,
        *,
        B: int = 1,
        max_seq: int | None = None,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "KVCache":
        """
        Convenience constructor from a GPT2Architecture or LlamaArchitecture.

        Both architectures expose .n_layer, .n_kv_head, .n_positions, .head_dim.
        Pass max_seq to override arch.n_positions (useful when arch.n_positions is
        very large, e.g. Llama 3.1's 131072).
        """
        return cls(
            n_layer=arch.n_layer,
            B=B,
            n_kv_head=arch.n_kv_head,
            max_seq=max_seq if max_seq is not None else arch.n_positions,
            head_dim=arch.head_dim,
            device=device,
            dtype=dtype,
        )
