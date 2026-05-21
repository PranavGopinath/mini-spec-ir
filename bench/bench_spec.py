"""Benchmark speculative decoding: acceptance rate, TTFT, TPS."""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from minispecir.engine import EOS_TOKEN_ID
from minispecir.model.gpt2 import GPT2Model
from minispecir.speculate import generate_spec


@dataclass
class SpecBenchResult:
    mode: str = "spec"
    prompt: str = ""
    prompt_len: int = 0
    generated: int = 0
    gamma: int = 4
    total_steps: int = 0
    total_draft: int = 0
    accepted: int = 0
    acceptance_rate: float = 0.0
    ttft_ms: float = 0.0
    tps: float = 0.0
    total_ms: float = 0.0
    device: str = ""
    dtype: str = "float32"


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _bench_spec(
    target: GPT2Model,
    draft: GPT2Model,
    input_ids: torch.Tensor,   # [1, T]
    prompt_text: str,
    max_new_tokens: int,
    gamma: int,
    eos_token_id: int,
) -> SpecBenchResult:
    tokens = input_ids.to(target.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output, stats = generate_spec(
            target, draft, tokens,
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            eos_token_id=eos_token_id,
        )
    _sync(target.device)
    total_s = time.perf_counter() - t0

    generated = output.shape[1] - input_ids.shape[1]
    acceptance_rate = (
        stats["accepted"] / stats["total_draft"]
        if stats["total_draft"] > 0 else 0.0
    )
    tps = generated / total_s if total_s > 0 else 0.0

    return SpecBenchResult(
        mode="spec",
        prompt=prompt_text,
        prompt_len=input_ids.shape[1],
        generated=generated,
        gamma=gamma,
        total_steps=stats["steps"],
        total_draft=stats["total_draft"],
        accepted=stats["accepted"],
        acceptance_rate=acceptance_rate,
        ttft_ms=0.0,   # not separately measured (prefill is bundled inside generate_spec)
        tps=tps,
        total_ms=total_s * 1000,
        device=str(target.device),
        dtype="float32",
    )


def run_spec_bench(
    target: GPT2Model,
    draft: GPT2Model,
    prompts: list[tuple[str, list[int]]],  # [(text, token_ids), ...]
    max_new_tokens: int = 50,
    gamma: int = 4,
    warmup: int = 1,
    eos_token_id: int = EOS_TOKEN_ID,
) -> list[SpecBenchResult]:
    """
    Benchmark generate_spec on a list of prompts.

    Args:
        warmup: number of warm-up runs (results discarded)
    """
    first_ids = torch.tensor([prompts[0][1]], dtype=torch.long)
    for _ in range(warmup):
        _bench_spec(target, draft, first_ids, "", max_new_tokens, gamma, eos_token_id)

    results: list[SpecBenchResult] = []
    for text, token_ids in prompts:
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        results.append(_bench_spec(target, draft, input_ids, text,
                                   max_new_tokens, gamma, eos_token_id))
    return results
