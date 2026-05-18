from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from minispecir.config import (
    DEFAULT_MODEL_DIR,
    ModelConfig,
    resolved_model_dir,
)

StateDict = dict[str, torch.Tensor]


@dataclass(frozen=True)
class GPT2Architecture:
    """Architecture hyperparameters for GPT-2 (from HF config.json)."""

    n_layer: int
    n_head: int
    n_embd: int
    n_positions: int
    vocab_size: int
    n_inner: int

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


def _cache_dir_arg(cache_dir: Path | str | None) -> str | None:
    if cache_dir is None:
        return None
    return str(cache_dir)


def is_local_snapshot_ready(model_dir: Path) -> bool:
    return (Path(model_dir) / "config.json").is_file()


def resolve_pretrained_source(
    model_id: str,
    model_dir: Path | None,
    *,
    local_files_only: bool,
) -> tuple[str, bool]:
    """
    Pick load path for ``from_pretrained``.

    Returns ``(source, use_local_files_only)`` where *source* is a Hub id or
    a directory containing ``config.json``.
    """
    if model_dir is not None:
        model_dir = Path(model_dir)
        if is_local_snapshot_ready(model_dir):
            return str(model_dir.resolve()), True
        if local_files_only:
            raise FileNotFoundError(
                f"No local model at {model_dir} (missing config.json). "
                f"Run: minispecir download --output {model_dir}"
            )

    if local_files_only:
        raise FileNotFoundError(
            "local_files_only=True but model_dir is not set. "
            f"Run: minispecir download --output {DEFAULT_MODEL_DIR}"
        )

    return model_id, False


def download_model(
    model_id: str = "gpt2",
    *,
    model_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> Path:
    """
    Download a full model snapshot into *model_dir* (one-time, needs network).

    After this, loads with ``local_files_only=True`` read only from disk.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download models. "
            "Install with: pip install huggingface_hub"
        ) from exc

    dest = Path(model_dir if model_dir is not None else DEFAULT_MODEL_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        cache_dir=_cache_dir_arg(cache_dir),
    )
    if not is_local_snapshot_ready(dest):
        raise RuntimeError(f"Download finished but config.json missing under {dest}")
    return dest.resolve()


def load_gpt2_architecture(
    model_id: str = "gpt2",
    *,
    model_dir: Path | str | None = None,
    local_files_only: bool = True,
    cache_dir: Path | str | None = None,
) -> GPT2Architecture:
    """Load GPT-2 hyperparameters from a local snapshot or the Hub."""
    source, local_only = resolve_pretrained_source(
        model_id,
        Path(model_dir) if model_dir is not None else None,
        local_files_only=local_files_only,
    )
    config = GPT2Config.from_pretrained(
        source,
        local_files_only=local_only,
        cache_dir=_cache_dir_arg(cache_dir),
    )
    n_inner = config.n_inner if config.n_inner is not None else 4 * config.n_embd
    return GPT2Architecture(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        n_positions=config.n_positions,
        vocab_size=config.vocab_size,
        n_inner=n_inner,
    )


def load_hf_state_dict(
    model_id: str = "gpt2",
    *,
    model_dir: Path | str | None = None,
    local_files_only: bool = True,
    cache_dir: Path | str | None = None,
) -> StateDict:
    """
    Load GPT-2 weights into a CPU float32 state_dict.

    Prefer a local snapshot under ``models/gpt2`` (see :func:`download_model`).
    With ``local_files_only=True``, never contacts the Hub at load time.
    """
    source, local_only = resolve_pretrained_source(
        model_id,
        Path(model_dir) if model_dir is not None else None,
        local_files_only=local_files_only,
    )
    model = GPT2LMHeadModel.from_pretrained(
        source,
        local_files_only=local_only,
        cache_dir=_cache_dir_arg(cache_dir),
        torch_dtype=torch.float32,
    )
    model.eval()
    return {
        key: tensor.detach().cpu().clone()
        for key, tensor in model.state_dict().items()
    }


def load_hf_state_dict_from_config(config: ModelConfig) -> StateDict:
    return load_hf_state_dict(
        config.model_id,
        model_dir=resolved_model_dir(config),
        local_files_only=config.local_files_only,
        cache_dir=config.cache_dir,
    )


def load_gpt2_architecture_from_config(config: ModelConfig) -> GPT2Architecture:
    return load_gpt2_architecture(
        config.model_id,
        model_dir=resolved_model_dir(config),
        local_files_only=config.local_files_only,
        cache_dir=config.cache_dir,
    )
