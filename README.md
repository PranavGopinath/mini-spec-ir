# mini-spec-ir

Minimal **spec**ulative **i**nference **r**untime for local LLMs (Apple Silicon / MPS).

Supports **Llama 3 8B** and **GPT-2** with a custom forward pass (no `model.generate()`), pre-allocated KV cache, and speculative decoding.

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
minispecir --help

# Device / version info
minispecir info

# Download a model snapshot (needs network once)
minispecir download --model meta-llama/Meta-Llama-3-8B-Instruct   # ~16GB
minispecir download --model gpt2                                    # ~500MB

# Greedy generation with KV cache
minispecir generate "The quick brown fox" --arch llama --model meta-llama/Meta-Llama-3-8B-Instruct
minispecir generate "The quick brown fox" --arch gpt2  --model gpt2

# Speculative decoding (target=gpt2, draft=distilgpt2)
minispecir spec "The quick brown fox" --model gpt2 --draft-model distilgpt2 --gamma 4

# Benchmark TTFT / ITL / TPS (vanilla vs KV)
minispecir bench --model gpt2
```

Weights load with `local_files_only=True` by default. Use `--model-dir` to point at a local snapshot directory.

## Tests

```bash
pytest
```

## License

MIT
