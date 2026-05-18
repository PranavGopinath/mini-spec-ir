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


def cmd_generate(args: argparse.Namespace) -> int:
    print("generate: not implemented yet (Phase 2+)", file=sys.stderr)
    return 1


def cmd_bench(args: argparse.Namespace) -> int:
    print("bench: not implemented yet (Phase 4+)", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minispecir",
        description="mini-spec-ir — minimal speculative inference runtime for local GPT-2",
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

    gen_p = sub.add_parser("generate", help="Greedy generation (Phase 2+)")
    gen_p.add_argument("prompt", help="Input prompt text")
    gen_p.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Tokens to generate",
    )
    gen_p.set_defaults(func=cmd_generate)

    bench_p = sub.add_parser("bench", help="Run benchmarks (Phase 4+)")
    bench_p.add_argument(
        "--output",
        type=str,
        default="reports/",
        help="Report output directory",
    )
    bench_p.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
