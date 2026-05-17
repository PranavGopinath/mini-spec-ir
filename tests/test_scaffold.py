import torch

from minispecir import __version__
from minispecir.device import resolve_device


def test_version_is_string() -> None:
    assert isinstance(__version__, str)
    assert __version__


def test_resolve_device_auto() -> None:
    device = resolve_device(None)
    assert device.type in ("cpu", "mps", "cuda")


def test_resolve_device_explicit_cpu() -> None:
    device = resolve_device("cpu")
    assert device == torch.device("cpu")


def test_cli_help() -> None:
    import pytest

    from minispecir.cli.main import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
