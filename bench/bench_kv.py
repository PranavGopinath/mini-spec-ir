"""Benchmark TTFT, ITL, and TPS for vanilla (full recompute) vs KV cache generation."""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from minispecir.cache import KVCache
from minispecir.engine import EOS_TOKEN_ID
from minispecir.model.gpt2 import GPT2Model


@dataclass
class BenchResult:
    mode: str           # "vanilla" or "kv"
    prompt: str
    prompt_len: int     # input tokens
    generated: int      # tokens produced (excluding prompt)
    ttft_ms: float      # time to first token (prefill + argmax)
    itl_ms: float       # mean inter-token latency across decode steps
    tps: float          # decode tokens / total decode time
    total_ms: float     # wall time for the whole call
    device: str
    dtype: str


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _bench_kv(
    model: GPT2Model,
    input_ids: torch.Tensor,   # [1, T]
    prompt_text: str,
    max_new_tokens: int,
    eos_token_id: int,
) -> BenchResult:
    tokens = input_ids.to(model.device)
    cache = KVCache.from_arch(model.arch, B=1, device=model.device)

    with torch.no_grad():
        # -- TTFT: prefill + first token --
        t0 = time.perf_counter()
        logits = model.forward(tokens, cache=cache)
        next_id = logits[0, -1].argmax()
        _sync(model.device)
        ttft_s = time.perf_counter() - t0

        tokens = torch.cat([tokens, next_id.view(1, 1)], dim=-1)
        decode_times: list[float] = []

        if next_id.item() != eos_token_id:
            for _ in range(max_new_tokens - 1):
                t_step = time.perf_counter()
                logits = model.forward(next_id.view(1, 1), cache=cache)
                next_id = logits[0, 0].argmax()
                _sync(model.device)
                decode_times.append(time.perf_counter() - t_step)
                tokens = torch.cat([tokens, next_id.view(1, 1)], dim=-1)
                if next_id.item() == eos_token_id:
                    break

    generated = tokens.shape[1] - input_ids.shape[1]
    itl_ms = (sum(decode_times) / len(decode_times) * 1000) if decode_times else 0.0
    tps = len(decode_times) / sum(decode_times) if decode_times else 0.0
    total_ms = (ttft_s + sum(decode_times)) * 1000

    return BenchResult(
        mode="kv",
        prompt=prompt_text,
        prompt_len=input_ids.shape[1],
        generated=generated,
        ttft_ms=ttft_s * 1000,
        itl_ms=itl_ms,
        tps=tps,
        total_ms=total_ms,
        device=str(model.device),
        dtype="float32",
    )


def _bench_vanilla(
    model: GPT2Model,
    input_ids: torch.Tensor,
    prompt_text: str,
    max_new_tokens: int,
    eos_token_id: int,
) -> BenchResult:
    tokens = input_ids.to(model.device)

    with torch.no_grad():
        # TTFT: first full forward + argmax
        t0 = time.perf_counter()
        logits = model.forward(tokens)
        next_id = logits[0, -1].argmax()
        _sync(model.device)
        ttft_s = time.perf_counter() - t0

        tokens = torch.cat([tokens, next_id.view(1, 1)], dim=-1)
        decode_times: list[float] = []

        if next_id.item() != eos_token_id:
            for _ in range(max_new_tokens - 1):
                t_step = time.perf_counter()
                logits = model.forward(tokens)
                next_id = logits[0, -1].argmax()
                _sync(model.device)
                decode_times.append(time.perf_counter() - t_step)
                tokens = torch.cat([tokens, next_id.view(1, 1)], dim=-1)
                if next_id.item() == eos_token_id:
                    break

    generated = tokens.shape[1] - input_ids.shape[1]
    itl_ms = (sum(decode_times) / len(decode_times) * 1000) if decode_times else 0.0
    tps = len(decode_times) / sum(decode_times) if decode_times else 0.0
    total_ms = (ttft_s + sum(decode_times)) * 1000

    return BenchResult(
        mode="vanilla",
        prompt=prompt_text,
        prompt_len=input_ids.shape[1],
        generated=generated,
        ttft_ms=ttft_s * 1000,
        itl_ms=itl_ms,
        tps=tps,
        total_ms=total_ms,
        device=str(model.device),
        dtype="float32",
    )


def run_bench(
    model: GPT2Model,
    prompts: list[tuple[str, list[int]]],   # [(text, token_ids), ...]
    max_new_tokens: int = 50,
    warmup: int = 1,
    eos_token_id: int = EOS_TOKEN_ID,
    run_vanilla: bool = True,
) -> list[BenchResult]:
    """
    Benchmark generate_kv (and optionally generate_vanilla) on a list of prompts.

    Args:
        warmup: number of warm-up runs per mode (results discarded)
    """
    results: list[BenchResult] = []

    first_ids = torch.tensor([prompts[0][1]], dtype=torch.long)

    if run_vanilla:
        for _ in range(warmup):
            _bench_vanilla(model, first_ids, "", max_new_tokens, eos_token_id)

    for _ in range(warmup):
        _bench_kv(model, first_ids, "", max_new_tokens, eos_token_id)

    for text, token_ids in prompts:
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        if run_vanilla:
            results.append(_bench_vanilla(model, input_ids, text, max_new_tokens, eos_token_id))
        results.append(_bench_kv(model, input_ids, text, max_new_tokens, eos_token_id))

    return results
