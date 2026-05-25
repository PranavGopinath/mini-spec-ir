from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from minispecir import __version__
from minispecir.config import DEFAULT_MODEL, DEFAULT_RUNTIME, resolved_model_dir
from minispecir.device import log_device_info, resolve_device
from minispecir.weights import download_model, is_local_snapshot_ready

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )


def cmd_info(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    log_device_info(device)
    model_dir = resolved_model_dir(DEFAULT_MODEL)
    local_ready = is_local_snapshot_ready(model_dir)
    print(f"mini-spec-ir {__version__}")
    print(f"device: {device}")
    print(f"model_id: {DEFAULT_MODEL.model_id}")
    print(f"model_dir: {model_dir} ({'ready' if local_ready else 'missing — run minispecir download'})")
    print(f"local_files_only: {DEFAULT_MODEL.local_files_only}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    path = download_model(
        args.model,
        model_dir=args.output,
        cache_dir=args.cache_dir,
    )
    print(f"Downloaded {args.model} → {path}")
    return 0


def _load_model(args: argparse.Namespace, device: "torch.device"):
    """Load arch + weights + model object based on --arch flag."""
    import torch
    from minispecir.engine import EOS_TOKEN_ID as GPT2_EOS

    model_dir = Path(args.model_dir) if args.model_dir else None

    if args.arch == "llama":
        from minispecir.weights import load_llama_architecture, load_llama_state_dict
        from minispecir.model.llama import LlamaModel
        arch = load_llama_architecture(args.model, model_dir=model_dir, local_files_only=False)
        state = load_llama_state_dict(args.model, model_dir=model_dir, local_files_only=False)
        model = LlamaModel(arch, state, device)
        eos_token_id = 128001  # Llama 3 <|end_of_text|>
    else:
        from minispecir.weights import load_gpt2_architecture, load_hf_state_dict
        from minispecir.model.gpt2 import GPT2Model
        arch = load_gpt2_architecture(args.model, model_dir=model_dir, local_files_only=False)
        state = load_hf_state_dict(args.model, model_dir=model_dir, local_files_only=False)
        model = GPT2Model(arch, state, device)
        eos_token_id = GPT2_EOS

    return model, eos_token_id


def cmd_generate(args: argparse.Namespace) -> int:
    import time
    import torch
    from minispecir.tokenizer import TokenizerWrapper
    from minispecir.engine import generate_kv
    from minispecir.cache import KVCache

    device = resolve_device(args.device)

    print(f"Loading {args.model} on {device}…", file=sys.stderr)
    model, eos_token_id = _load_model(args, device)
    tokenizer = TokenizerWrapper.from_pretrained(args.model)

    prompt_ids = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    t0 = time.perf_counter()
    output_ids = generate_kv(
        model, input_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_token_id,
    )
    elapsed = time.perf_counter() - t0

    text = tokenizer.decode(output_ids[0].tolist())
    print(text)

    n_new = output_ids.shape[1] - len(prompt_ids)
    print(f"\n[{n_new} tokens in {elapsed:.2f}s — {n_new / elapsed:.1f} tok/s]", file=sys.stderr)
    return 0


def cmd_spec(args: argparse.Namespace) -> int:
    import time
    import torch
    from minispecir.tokenizer import TokenizerWrapper
    from minispecir.weights import load_gpt2_architecture, load_hf_state_dict
    from minispecir.model.gpt2 import GPT2Model
    from minispecir.speculate import generate_spec

    device = resolve_device(args.device)
    model_dir = Path(args.model_dir) if args.model_dir else None
    draft_dir = Path(args.draft_model_dir) if args.draft_model_dir else None

    print(f"Loading target {args.model} on {device}…", file=sys.stderr)
    target_arch = load_gpt2_architecture(args.model, model_dir=model_dir, local_files_only=False)
    target_state = load_hf_state_dict(args.model, model_dir=model_dir, local_files_only=False)
    target = GPT2Model(target_arch, target_state, device)

    print(f"Loading draft {args.draft_model} on {device}…", file=sys.stderr)
    draft_arch = load_gpt2_architecture(args.draft_model, model_dir=draft_dir, local_files_only=False)
    draft_state = load_hf_state_dict(args.draft_model, model_dir=draft_dir, local_files_only=False)
    draft = GPT2Model(draft_arch, draft_state, device)

    tokenizer = TokenizerWrapper.from_pretrained(args.model)
    prompt_ids = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    t0 = time.perf_counter()
    output_ids, stats = generate_spec(
        target, draft, input_ids,
        max_new_tokens=args.max_new_tokens,
        gamma=args.gamma,
    )
    elapsed = time.perf_counter() - t0

    text = tokenizer.decode(output_ids[0].tolist())
    print(text)

    n_new = output_ids.shape[1] - len(prompt_ids)
    accept_pct = (
        stats["accepted"] / stats["total_draft"] * 100
        if stats["total_draft"] > 0 else 0.0
    )
    print(
        f"\n[{n_new} tokens in {elapsed:.2f}s — {n_new / elapsed:.1f} tok/s, "
        f"acceptance {accept_pct:.1f}% (γ={args.gamma})]",
        file=sys.stderr,
    )
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    import sys as _sys
    import torch
    from minispecir.tokenizer import TokenizerWrapper
    from minispecir.weights import load_gpt2_architecture, load_hf_state_dict
    from minispecir.model.gpt2 import GPT2Model
    from bench.bench_kv import run_bench
    from bench.report import write_report

    device = resolve_device(args.device)
    model_dir = Path(args.model_dir) if args.model_dir else None

    print(f"Loading {args.model} on {device}…", file=sys.stderr)
    arch = load_gpt2_architecture(args.model, model_dir=model_dir, local_files_only=False)
    state = load_hf_state_dict(args.model, model_dir=model_dir, local_files_only=False)
    model = GPT2Model(arch, state, device)
    tokenizer = TokenizerWrapper.from_pretrained(args.model)

    prompts_text = [
        "Hello, my name is",
        "The quick brown fox",
        "Once upon a time",
        "In the beginning",
        "To be or not to be",
    ]
    prompts = [(t, tokenizer.encode(t)) for t in prompts_text]

    print(f"Running bench (max_new_tokens={args.max_new_tokens}, warmup={args.warmup})…", file=sys.stderr)
    results = run_bench(
        model,
        prompts,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
        run_vanilla=not args.kv_only,
    )

    from bench.report import render_markdown, write_report
    print(render_markdown(results))

    report_path = write_report(results, Path(args.output))
    print(f"\nReport saved → {report_path}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minispecir",
        description="mini-spec-ir — minimal speculative inference runtime for local LLMs",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    info_p = sub.add_parser("info", help="Show device and config defaults")
    info_p.add_argument(
        "--model",
        default=DEFAULT_MODEL.model_id,
        help="Model id (default: gpt2)",
    )
    info_p.add_argument(
        "--device",
        default=DEFAULT_RUNTIME.device,
        help="Force device (e.g. cpu, mps). Default: auto",
    )
    info_p.set_defaults(func=cmd_info)

    dl_p = sub.add_parser(
        "download",
        help="Download model snapshot to models/gpt2 (one-time, needs network)",
    )
    dl_p.add_argument("--model", default=DEFAULT_MODEL.model_id)
    dl_p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Local directory (default: {resolved_model_dir(DEFAULT_MODEL)})",
    )
    dl_p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HF hub cache while downloading (optional)",
    )
    dl_p.set_defaults(func=cmd_download)

    gen_p = sub.add_parser("generate", help="Greedy generation with KV cache")
    gen_p.add_argument("prompt", help="Input prompt text")
    gen_p.add_argument("--max-new-tokens", type=int, default=32, help="Tokens to generate (default: 32)")
    gen_p.add_argument("--model", default=DEFAULT_MODEL.model_id, help="Model id (default: gpt2)")
    gen_p.add_argument("--model-dir", default=None, help="Local model snapshot directory")
    gen_p.add_argument("--arch", default="gpt2", choices=["gpt2", "llama"], help="Model architecture (default: gpt2)")
    gen_p.add_argument("--device", default=DEFAULT_RUNTIME.device, help="Device override (default: auto)")
    gen_p.set_defaults(func=cmd_generate)

    spec_p = sub.add_parser("spec", help="Speculative decoding (draft=distilgpt2, target=gpt2)")
    spec_p.add_argument("prompt", help="Input prompt text")
    spec_p.add_argument("--max-new-tokens", type=int, default=32, help="Tokens to generate (default: 32)")
    spec_p.add_argument("--gamma", type=int, default=DEFAULT_RUNTIME.gamma, help=f"Draft tokens per step (default: {DEFAULT_RUNTIME.gamma})")
    spec_p.add_argument("--model", default=DEFAULT_MODEL.model_id, help="Target model id (default: gpt2)")
    spec_p.add_argument("--model-dir", default=None, help="Local target model snapshot directory")
    spec_p.add_argument("--draft-model", default="distilgpt2", help="Draft model id (default: distilgpt2)")
    spec_p.add_argument("--draft-model-dir", default=None, help="Local draft model snapshot directory")
    spec_p.add_argument("--device", default=DEFAULT_RUNTIME.device, help="Device override (default: auto)")
    spec_p.set_defaults(func=cmd_spec)

    bench_p = sub.add_parser("bench", help="Benchmark TTFT / ITL / TPS (vanilla vs KV)")
    bench_p.add_argument("--output", type=str, default="reports/", help="Report output directory")
    bench_p.add_argument("--model", default=DEFAULT_MODEL.model_id, help="Model id (default: gpt2)")
    bench_p.add_argument("--model-dir", default=None, help="Local model snapshot directory")
    bench_p.add_argument("--device", default=DEFAULT_RUNTIME.device, help="Device override (default: auto)")
    bench_p.add_argument("--max-new-tokens", type=int, default=50, help="Tokens to generate per prompt (default: 50)")
    bench_p.add_argument("--warmup", type=int, default=1, help="Warm-up runs before timing (default: 1)")
    bench_p.add_argument("--kv-only", action="store_true", help="Skip vanilla (slow) mode")
    bench_p.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
