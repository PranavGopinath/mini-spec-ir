"""Render BenchResult lists as a Markdown report."""
from __future__ import annotations

import datetime
from pathlib import Path

import torch

from bench.bench_kv import BenchResult
from bench.bench_spec import SpecBenchResult


_HEADER = """\
# mini-spec-ir benchmark report

**Generated:** {date}
**Device:** {device}
**dtype:** {dtype}
**PyTorch:** {torch_version}

"""

_TABLE_HEADER = """\
| mode | prompt | p_len | gen | TTFT (ms) | ITL (ms) | TPS |
|------|--------|------:|----:|----------:|---------:|----:|
"""

_ROW = "| {mode} | {prompt} | {p_len} | {gen} | {ttft:.1f} | {itl:.1f} | {tps:.1f} |\n"

_NOTES = """
## Notes

- **TTFT** — time from first input token to first output token (prefill + argmax).
- **ITL** — mean inter-token latency across decode steps.
- **TPS** — decode tokens per second (excludes prefill).
- vanilla mode recomputes the full attention matrix every step (O(T²)); kv mode writes K/V once per layer per step.
"""


def render_markdown(results: list[BenchResult]) -> str:
    if not results:
        return ""

    device = results[0].device
    dtype = results[0].dtype
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [_HEADER.format(date=date, device=device, dtype=dtype, torch_version=torch.__version__)]
    lines.append(_TABLE_HEADER)
    for r in results:
        prompt_short = (r.prompt[:28] + "…") if len(r.prompt) > 29 else r.prompt
        lines.append(_ROW.format(
            mode=r.mode,
            prompt=prompt_short,
            p_len=r.prompt_len,
            gen=r.generated,
            ttft=r.ttft_ms,
            itl=r.itl_ms,
            tps=r.tps,
        ))
    lines.append(_NOTES)
    return "".join(lines)


_SPEC_TABLE_HEADER = """\
## Speculative decoding

| γ | prompt | p_len | gen | steps | accept% | TPS |
|--:|--------|------:|----:|------:|--------:|----:|
"""

_SPEC_ROW = "| {gamma} | {prompt} | {p_len} | {gen} | {steps} | {accept:.1f}% | {tps:.1f} |\n"

_SPEC_NOTES = """
- **γ** — draft tokens proposed per speculative step.
- **accept%** — fraction of draft tokens accepted by the target (higher = more efficient).
- Spec TPS measures total generated tokens / wall time (prefill included).
"""


def render_spec_section(results: list[SpecBenchResult]) -> str:
    """Render a Markdown section for speculative decoding results."""
    if not results:
        return ""
    lines = [_SPEC_TABLE_HEADER]
    for r in results:
        prompt_short = (r.prompt[:28] + "…") if len(r.prompt) > 29 else r.prompt
        lines.append(_SPEC_ROW.format(
            gamma=r.gamma,
            prompt=prompt_short,
            p_len=r.prompt_len,
            gen=r.generated,
            steps=r.total_steps,
            accept=r.acceptance_rate * 100,
            tps=r.tps,
        ))
    lines.append(_SPEC_NOTES)
    return "".join(lines)


def write_report(results: list[BenchResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"bench_{date_str}.md"
    path.write_text(render_markdown(results))
    return path
