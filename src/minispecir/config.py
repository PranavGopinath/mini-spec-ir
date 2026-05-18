from dataclasses import dataclass
from pathlib import Path

# Default on-disk snapshot (gitignored). Run: minispecir download
DEFAULT_MODEL_DIR = Path("models/gpt2")


@dataclass
class ModelConfig:
    """Model checkpoint location and loading policy."""

    model_id: str = "gpt2"
    model_dir: Path | None = None  # None → DEFAULT_MODEL_DIR
    local_files_only: bool = True  # no Hub network at load time when local snapshot exists
    cache_dir: Path | None = None  # HF hub cache used only while downloading


@dataclass
class RuntimeConfig:
    """Inference runtime knobs (expanded in later phases)."""

    max_seq_len: int = 1024
    dtype: str = "float32"
    device: str | None = None  # None → auto (MPS if available, else CPU)
    gamma: int = 4  # draft tokens per speculative step (v1.0)


DEFAULT_MODEL = ModelConfig()
DEFAULT_RUNTIME = RuntimeConfig()


def resolved_model_dir(config: ModelConfig) -> Path:
    return config.model_dir if config.model_dir is not None else DEFAULT_MODEL_DIR
