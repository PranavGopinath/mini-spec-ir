from __future__ import annotations

import argparse
import logging
import sys

from minispecir import __version__
from minispecir.config import DEFAULT_MODEL, DEFAULT_RUNTIME
from minispecir.device import log_device_info, resolve_device

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
    print(f"mini-spec-ir {__version__}")
    print(f"device: {device}")
    print(f"default model: {args.model}")
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
