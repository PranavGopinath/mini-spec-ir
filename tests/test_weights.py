import torch
from pathlib import Path

import pytest

from minispecir.config import ModelConfig, resolved_model_dir
from minispecir.weights import (
    GPT2Architecture,
    download_model,
    is_local_snapshot_ready,
    load_gpt2_architecture,
    load_gpt2_architecture_from_config,
    load_hf_state_dict,
    load_hf_state_dict_from_config,
    resolve_pretrained_source,
)

EXPECTED_GPT2 = GPT2Architecture(
    n_layer=12,
    n_head=12,
    n_embd=768,
    n_positions=1024,
    vocab_size=50257,
    n_inner=3072,
)

# Use Hub cache when no local snapshot (e.g. fresh CI without download).
HUB_FALLBACK = ModelConfig(local_files_only=False, model_dir=None)


def test_load_architecture() -> None:
    arch = load_gpt2_architecture("gpt2", local_files_only=False)
    assert arch == EXPECTED_GPT2
    assert arch.head_dim == 64


def test_load_architecture_from_config() -> None:
    arch = load_gpt2_architecture_from_config(HUB_FALLBACK)
    assert arch.n_layer == 12


def test_load_state_dict_keys_and_shapes() -> None:
    state = load_hf_state_dict("gpt2", local_files_only=False)

    assert "transformer.wte.weight" in state
    assert "transformer.wpe.weight" in state
    assert "transformer.h.0.attn.c_attn.weight" in state
    assert "transformer.ln_f.weight" in state

    assert state["transformer.wte.weight"].shape == (50257, 768)
    assert state["transformer.wpe.weight"].shape == (1024, 768)
    assert state["transformer.h.0.attn.c_attn.weight"].shape == (768, 2304)

    for tensor in state.values():
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"


def test_load_state_dict_from_config() -> None:
    state = load_hf_state_dict_from_config(HUB_FALLBACK)
    assert len(state) > 0


def test_resolve_local_snapshot(tmp_path: Path) -> None:
    download_model("gpt2", model_dir=tmp_path)
    assert is_local_snapshot_ready(tmp_path)

    source, local_only = resolve_pretrained_source(
        "gpt2",
        tmp_path,
        local_files_only=True,
    )
    assert local_only is True
    assert Path(source) == tmp_path.resolve()

    state = load_hf_state_dict("gpt2", model_dir=tmp_path, local_files_only=True)
    assert state["transformer.wte.weight"].shape == (50257, 768)


def test_local_files_only_raises_without_snapshot(tmp_path: Path) -> None:
    missing = tmp_path / "empty"
    missing.mkdir()
    with pytest.raises(FileNotFoundError, match="minispecir download"):
        resolve_pretrained_source("gpt2", missing, local_files_only=True)


def test_default_config_points_at_models_gpt2() -> None:
    assert resolved_model_dir(ModelConfig()) == Path("models/gpt2")
