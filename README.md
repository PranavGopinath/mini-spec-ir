# mini-spec-ir

Minimal **spec**ulative **i**nference **r**untime for local GPT-2 (Apple Silicon / MPS).

**Status:** Core decoder + KV cache → v0.1; speculative decoding → v1.0.

## Setup

Requires **Python 3.11+** (3.13 recommended on Apple Silicon).

```bash
python3.13 -m venv .venv   # or python3.11+
source .venv/bin/activate
pip install -e ".[dev]"
```

## CLI

```bash
# Help
python -m minispecir.cli.main --help
minispecir --help

# Device / version info (shows whether models/gpt2 is present)
minispecir info

# One-time download to ./models/gpt2 (production-style local snapshot)
minispecir download
```

Weights load from **`models/gpt2`** with `local_files_only=True` by default (no Hub at inference load time). First time only:

```bash
minispecir download   # ~500MB for gpt2, needs network once
```

Override: set `ModelConfig(model_dir=..., local_files_only=True)` or pass `local_files_only=False` to fall back to the HF hub cache (dev only).

## Tests

```bash
pytest
```

## License

MIT
