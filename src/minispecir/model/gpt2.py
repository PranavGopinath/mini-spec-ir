"""Manual GPT-2 forward pass.

Weights come from a HF state_dict (Conv1D layout: shape [in, out]).
No model.forward() in the hot path — only this file runs at inference time.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from minispecir.weights import GPT2Architecture, StateDict


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """GELU with tanh approximation (matches HF gelu_new)."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class GPT2Model:
    """
    GPT-2 decoder-only transformer.

    All weights are pre-loaded onto *device* at construction.
    forward() expects input_ids on the same device.
    """

    def __init__(
        self,
        arch: GPT2Architecture,
        state: StateDict,
        device: torch.device,
    ) -> None:
        self.arch = arch
        self.device = device
        self.state: dict[str, torch.Tensor] = {
            k: v.to(device) for k, v in state.items()
        }

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def _ln(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], weight, bias, eps)

    def _attn(self, x: torch.Tensor, block_idx: int) -> torch.Tensor:
        """Multi-head causal self-attention for one block."""
        n_head = self.arch.n_head
        head_dim = self.arch.head_dim
        C = self.arch.n_embd

        # Conv1D weights are stored [in, out] → compute as x @ W
        W_qkv = self.state[f"transformer.h.{block_idx}.attn.c_attn.weight"]  # [C, 3C]
        b_qkv = self.state[f"transformer.h.{block_idx}.attn.c_attn.bias"]    # [3C]
        qkv = x @ W_qkv + b_qkv  # [B, T, 3C]

        q, k, v = qkv.split(C, dim=-1)  # each [B, T, C]

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # [B, T, C] → [B, n_head, T, head_dim]
            B, T, _ = t.shape
            return t.view(B, T, n_head, head_dim).transpose(1, 2)

        q = split_heads(q)  # [B, n_head, T, head_dim]
        k = split_heads(k)
        v = split_heads(v)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, n_head, T, T]

        T = x.shape[-2]
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        attn_w = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_w, v)  # [B, n_head, T, head_dim]

        # Merge heads → [B, T, C]
        B = out.shape[0]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        W_proj = self.state[f"transformer.h.{block_idx}.attn.c_proj.weight"]  # [C, C]
        b_proj = self.state[f"transformer.h.{block_idx}.attn.c_proj.bias"]
        return out @ W_proj + b_proj

    def _mlp(self, x: torch.Tensor, block_idx: int) -> torch.Tensor:
        W_fc   = self.state[f"transformer.h.{block_idx}.mlp.c_fc.weight"]    # [C, 4C]
        b_fc   = self.state[f"transformer.h.{block_idx}.mlp.c_fc.bias"]
        W_proj = self.state[f"transformer.h.{block_idx}.mlp.c_proj.weight"]  # [4C, C]
        b_proj = self.state[f"transformer.h.{block_idx}.mlp.c_proj.bias"]

        h = gelu_new(x @ W_fc + b_fc)
        return h @ W_proj + b_proj

    def block(self, x: torch.Tensor, block_idx: int) -> torch.Tensor:
        """One transformer block: pre-norm attn + pre-norm MLP, both with residuals."""
        ln1_w = self.state[f"transformer.h.{block_idx}.ln_1.weight"]
        ln1_b = self.state[f"transformer.h.{block_idx}.ln_1.bias"]
        ln2_w = self.state[f"transformer.h.{block_idx}.ln_2.weight"]
        ln2_b = self.state[f"transformer.h.{block_idx}.ln_2.bias"]

        x = x + self._attn(self._ln(x, ln1_w, ln1_b), block_idx)
        x = x + self._mlp(self._ln(x, ln2_w, ln2_b), block_idx)
        return x

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [T] or [B, T] long tensor on self.device

        Returns:
            logits: [T, vocab_size] or [B, T, vocab_size]
        """
        squeeze = input_ids.dim() == 1
        if squeeze:
            input_ids = input_ids.unsqueeze(0)

        B, T = input_ids.shape

        wte = self.state["transformer.wte.weight"]  # [V, C]
        wpe = self.state["transformer.wpe.weight"]  # [n_pos, C]

        pos_ids = torch.arange(T, device=self.device)
        x = wte[input_ids] + wpe[pos_ids]  # [B, T, C]

        for i in range(self.arch.n_layer):
            x = self.block(x, i)

        ln_f_w = self.state["transformer.ln_f.weight"]
        ln_f_b = self.state["transformer.ln_f.bias"]
        x = self._ln(x, ln_f_w, ln_f_b)

        # LM head tied to token embeddings
        logits = x @ wte.T  # [B, T, V]

        if squeeze:
            logits = logits.squeeze(0)
        return logits
