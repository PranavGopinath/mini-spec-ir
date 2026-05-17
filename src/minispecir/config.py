from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Hugging Face model id and on-disk cache location."""

    model_id: str = "gpt2"
    cache_dir: Path | None = None


@dataclass
class RuntimeConfig:
    """Inference runtime knobs (expanded in later phases)."""

    max_seq_len: int = 1024
    dtype: str = "float32"
    device: str | None = None  # None → auto (MPS if available, else CPU)
    gamma: int = 4  # draft tokens per speculative step (v1.0)


DEFAULT_MODEL = ModelConfig()
DEFAULT_RUNTIME = RuntimeConfig()
