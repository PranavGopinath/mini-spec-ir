# mini-spec-ir

Minimal **spec**ulative **i**nference **r**untime for local GPT-2 (Apple Silicon / MPS).

**Status:** v0.1 in progress — Phase 0 scaffold. Core decoder + KV cache → v0.1; speculative decoding → v1.0.

See [PLAN.md](PLAN.md) for the full roadmap.

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

# Device / version info
minispecir info
```

Weights are loaded via Hugging Face cache in Phase 1 (`weights.py`); no separate download command until then.

## Tests

```bash
pytest
```

## License

MIT
