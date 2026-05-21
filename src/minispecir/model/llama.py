"""Manual Llama 3 forward pass.

Differences from GPT-2:
  - RoPE (rotary position embeddings) applied to Q/K inside attention; no wpe table
  - RMSNorm instead of LayerNorm (no bias, no mean centering)
  - GQA: n_kv_head < n_head; KV heads expanded via repeat_interleave before attention
  - SwiGLU MLP: silu(gate(x)) * up(x) fed through down projection; no bias
  - Standard nn.Linear weights (shape [out, in]); apply as F.linear(x, W)
  - LM head is NOT tied to token embeddings

No model.forward() in the hot path — only this file runs at inference time.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from minispecir.weights import LlamaArchitecture, StateDict

if TYPE_CHECKING:
    from minispecir.cache import KVCache


def _compute_llama3_inv_freq(
    head_dim: int,
    rope_theta: float,
    rope_scaling: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute RoPE inverse frequencies with Llama 3 scaling.

    Matches HF _compute_llama3_parameters: high-freq bands kept as-is,
    low-freq bands divided by factor, medium band smoothly interpolated.
    """
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )

    factor = float(rope_scaling["factor"])
    low_freq_factor = float(rope_scaling["low_freq_factor"])
    high_freq_factor = float(rope_scaling["high_freq_factor"])
    old_ctx = float(rope_scaling["original_max_position_embeddings"])

    low_freq_wavelen = old_ctx / low_freq_factor    # longer wavelength → lower freq
    high_freq_wavelen = old_ctx / high_freq_factor  # shorter wavelength → higher freq

    wavelen = 2 * math.pi / inv_freq  # [head_dim // 2]

    smooth = (old_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smooth_inv_freq = (1 - smooth) * inv_freq / factor + smooth * inv_freq

    new_inv_freq = torch.where(
        wavelen < high_freq_wavelen,
        inv_freq,                   # high freq: keep
        torch.where(
            wavelen > low_freq_wavelen,
            inv_freq / factor,      # low freq: scale down
            smooth_inv_freq,        # medium: smooth blend
        ),
    )
    return new_inv_freq


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of head_dim: [-x2, x1] where x = cat(x1, x2)."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class LlamaModel:
    """
    Llama 3 decoder-only transformer.

    All weights are pre-loaded onto *device* at construction.
    forward() expects input_ids on the same device.
    """

    def __init__(
        self,
        arch: LlamaArchitecture,
        state: StateDict,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.arch = arch
        self.device = device
        self.dtype = dtype
        # Move weights to device, filter precomputed RoPE buffers (we recompute them)
        self.state: dict[str, torch.Tensor] = {
            k: v.to(device=device, dtype=dtype)
            for k, v in state.items()
            if "rotary_emb.inv_freq" not in k
        }
        self.inv_freq = self._build_inv_freq()  # [head_dim // 2]

    # ------------------------------------------------------------------
    # RoPE
    # ------------------------------------------------------------------

    def _build_inv_freq(self) -> torch.Tensor:
        arch = self.arch
        head_dim = arch.head_dim

        if (
            arch.rope_scaling is not None
            and arch.rope_scaling.get("rope_type") == "llama3"
        ):
            return _compute_llama3_inv_freq(
                head_dim, arch.rope_theta, arch.rope_scaling, self.device
            )

        return 1.0 / (
            arch.rope_theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=self.device)
                / head_dim
            )
        )

    def _rope_cos_sin(self, past_len: int, T_new: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin tables for positions [past_len, past_len + T_new)."""
        positions = torch.arange(past_len, past_len + T_new, device=self.device)
        freqs = torch.outer(positions.float(), self.inv_freq)  # [T_new, head_dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)                # [T_new, head_dim]
        return emb.cos(), emb.sin()

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def _rmsnorm(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """RMSNorm in float32, output cast back to input dtype (matches HF)."""
        dtype = x.dtype
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(variance + eps)
        return (weight * x_norm).to(dtype)

    def _attn(
        self,
        x: torch.Tensor,       # [B, T_new, C]
        block_idx: int,
        cache: "KVCache | None" = None,
    ) -> torch.Tensor:
        """GQA self-attention with RoPE."""
        arch = self.arch
        n_head = arch.n_head
        n_kv_head = arch.n_kv_head
        head_dim = arch.head_dim
        T_new = x.shape[-2]

        pfx = f"model.layers.{block_idx}.self_attn"
        W_q = self.state[f"{pfx}.q_proj.weight"]  # [n_head*head_dim, C]
        W_k = self.state[f"{pfx}.k_proj.weight"]  # [n_kv_head*head_dim, C]
        W_v = self.state[f"{pfx}.v_proj.weight"]  # [n_kv_head*head_dim, C]
        W_o = self.state[f"{pfx}.o_proj.weight"]  # [C, n_head*head_dim]

        q = F.linear(x, W_q)      # [B, T_new, n_head*head_dim]
        k_new = F.linear(x, W_k)  # [B, T_new, n_kv_head*head_dim]
        v_new = F.linear(x, W_v)  # [B, T_new, n_kv_head*head_dim]

        def split_heads(t: torch.Tensor, n_h: int) -> torch.Tensor:
            B, T, _ = t.shape
            return t.view(B, T, n_h, head_dim).transpose(1, 2)  # [B, n_h, T, head_dim]

        q = split_heads(q, n_head)
        k_new_h = split_heads(k_new, n_kv_head)
        v_new_h = split_heads(v_new, n_kv_head)

        # RoPE: apply to Q and (new) K before writing to cache
        past_len = cache.past_len if cache is not None else 0
        cos, sin = self._rope_cos_sin(past_len, T_new)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T_new, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)

        q = q * cos + _rotate_half(q) * sin
        k_new_h = k_new_h * cos + _rotate_half(k_new_h) * sin

        # KV cache: write new K/V, read full sequence
        if cache is not None:
            cache.write(block_idx, k_new_h, v_new_h)
            k, v = cache.read(block_idx, T_new)  # [B, n_kv_head, T_full, head_dim]
            T_full = past_len + T_new
        else:
            k, v = k_new_h, v_new_h
            T_full = T_new

        # Expand KV heads to Q heads (GQA)
        groups = n_head // n_kv_head
        if groups > 1:
            k = k.repeat_interleave(groups, dim=1)  # [B, n_head, T_full, head_dim]
            v = v.repeat_interleave(groups, dim=1)

        # Causal mask [T_new, T_full]
        if past_len > 0:
            causal_mask = torch.ones(T_new, T_full, device=x.device).bool()
            causal_mask[:, past_len:] = torch.tril(
                torch.ones(T_new, T_new, device=x.device)
            ).bool()
        else:
            causal_mask = torch.tril(
                torch.ones(T_new, T_full, device=x.device)
            ).bool()

        # Scaled dot-product attention (upcast to float32 for numerical stability)
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        attn_w = F.softmax(scores, dim=-1).to(q.dtype)

        out = torch.matmul(attn_w, v)  # [B, n_head, T_new, head_dim]
        B = out.shape[0]
        out = out.transpose(1, 2).contiguous().view(B, T_new, n_head * head_dim)

        return F.linear(out, W_o)  # [B, T_new, C]

    def _mlp(self, x: torch.Tensor, block_idx: int) -> torch.Tensor:
        """SwiGLU: down(silu(gate(x)) * up(x)), no bias."""
        pfx = f"model.layers.{block_idx}.mlp"
        W_gate = self.state[f"{pfx}.gate_proj.weight"]  # [n_inner, C]
        W_up   = self.state[f"{pfx}.up_proj.weight"]    # [n_inner, C]
        W_down = self.state[f"{pfx}.down_proj.weight"]  # [C, n_inner]

        gate = F.silu(F.linear(x, W_gate))
        up   = F.linear(x, W_up)
        return F.linear(gate * up, W_down)

    def block(
        self,
        x: torch.Tensor,
        block_idx: int,
        cache: "KVCache | None" = None,
    ) -> torch.Tensor:
        """One transformer block: pre-norm attn + pre-norm MLP, both with residuals."""
        eps = self.arch.rms_norm_eps
        ln1_w = self.state[f"model.layers.{block_idx}.input_layernorm.weight"]
        ln2_w = self.state[f"model.layers.{block_idx}.post_attention_layernorm.weight"]

        x = x + self._attn(self._rmsnorm(x, ln1_w, eps), block_idx, cache=cache)
        x = x + self._mlp(self._rmsnorm(x, ln2_w, eps), block_idx)
        return x

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: "KVCache | None" = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [T] or [B, T] long tensor on self.device
            cache:     optional KVCache; if provided, reads past K/V and writes new ones

        Returns:
            logits: [T, vocab_size] or [B, T, vocab_size]
        """
        squeeze = input_ids.dim() == 1
        if squeeze:
            input_ids = input_ids.unsqueeze(0)

        B, T_new = input_ids.shape

        wte = self.state["model.embed_tokens.weight"]  # [V, C]
        x = wte[input_ids]  # [B, T_new, C] — no positional embedding; RoPE is in _attn

        for i in range(self.arch.n_layer):
            x = self.block(x, i, cache=cache)

        if cache is not None:
            cache.advance(T_new)

        norm_w = self.state["model.norm.weight"]
        x = self._rmsnorm(x, norm_w, self.arch.rms_norm_eps)

        lm_head_w = self.state["lm_head.weight"]  # [V, C], NOT tied to wte
        logits = F.linear(x, lm_head_w)  # [B, T_new, V]

        if squeeze:
            logits = logits.squeeze(0)
        return logits
