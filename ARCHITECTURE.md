# mini-spec-ir — End-to-end architecture

> **Preview note:** Cursor/VS Code’s built-in Markdown preview does **not** render Mermaid by default. This doc uses **ASCII diagrams** so they show everywhere. On GitHub, Mermaid also works if you paste blocks into [mermaid.live](https://mermaid.live).

Your mental model is close. The main fix: **KV cache isn’t a step before the model** — it’s **state the model’s attention uses** during each forward pass. Between model and text you also have **logits → (greedy) sample → token id**, not straight to text.

This document reflects the [PLAN.md](PLAN.md) and **what exists in the repo today** (no extra features assumed).

---

## End-to-end flow (v0.1 MVP)

```
  [prompt text]
       |
       v
  +----------+
  |   CLI    |
  +----------+
       |
       v
  +----------+     token ids      +----------+
  |tokenizer | -----------------> |  engine  |
  +----------+                    +----------+
                                       |
                    +------------------+------------------+
                    | prefill (all prompt tokens)          |
                    | decode  (one new token per step)     |
                    v                                      v
              +----------+    read/write K,V      +----------+
              |  model   | <--------------------> | KV cache |
              | gpt2.py  |                        | cache.py |
              +----------+                        +----------+
                    ^                                      ^
                    | weights.py                           |
                    | device.py                            |
                    +--------------------------------------+
                    |
                    v logits
              +----------+
              |  engine  | -- greedy argmax --> next token id
              +----------+         |
                    ^              | loop until max_tokens / EOS
                    +--------------+
                    |
                    v
              +----------+
              |tokenizer | --> [output text]
              +----------+
```

### Pipeline (one line per step)

```text
text
  → tokenizer.encode          → token ids
  → engine.prefill            → model + KV fill for prompt
  → loop: engine.decode_step  → model (uses KV) → logits → greedy sample → new id
  → tokenizer.decode          → text
```

KV is **updated inside** attention during each forward — not a separate stage before the model.

---

## Prefill vs decode (sequence)

```text
User          Tokenizer       Engine          Model           KVCache
  |               |              |               |                |
  |-- prompt ---->|              |               |                |
  |               |-- ids ------>|               |                |
  |               |              |               |                |
  |               |         [ PREFILL ]          |                |
  |               |              |-- forward --->|                |
  |               |              |               |-- write K,V -->|
  |               |              |<-- logits ----|                |
  |               |              |               |                |
  |               |         [ DECODE LOOP ]      |                |
  |               |              | argmax        |                |
  |               |              |-- 1 token --->|                |
  |               |              |               |-- append K,V ->|
  |               |              |<-- logits ----|                |
  |               |              |    (repeat)   |                |
  |               |              |               |                |
  |               |<-- ids ------|               |                |
  |<-- text ------|              |               |                |
```

---

## v1.0: speculative decoding (one decode step)

**Vanilla (v0.1):**

```text
  Target model (gpt2)  <-->  Target KV
       |
       +-- one token out per step
```

**Speculative (v1.0):**

```text
  Draft (distilgpt2)  -- proposes gamma tokens -->
                              |
                              v
  Target (gpt2)  <-->  Target KV  -- verify batch -->
                              |
                              v
                    accept prefix / fix first mismatch
                    (0 .. gamma tokens committed)
```

- Same tokenizer for draft and target.
- Draft has its own weights; target KV is for verification, not shared with draft’s internal steps.

---

## Full component map

```text
                         +--------+
                         |  CLI   |
                         +--------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
       +-------------+                  +-------------+
       |  tokenizer  |                  |   engine    |
       +-------------+                  +-------------+
                                              |
              +-------------------------------+-------------------------------+
              |                               |                               |
              v                               v                               v
       +-------------+                  +-------------+                  +-------------+
       |   weights   |                  |    model    |                  |  KV cache   |
       +-------------+                  +-------------+                  +-------------+
              |                               ^                               ^
              v                               |                               |
       +-------------+                         +-------- greedy sampler -----+
       |   device    |                                  (inside engine)
       +-------------+

  v1.0 only:  engine --> speculate.py --> draft model
                                    \--> target model

  observability:  bench/* --> engine        metrics.py --> bench

  dev only:  tests/parity --> HF transformers (reference, not hot path)
```

### Component responsibilities

| Component | Responsibility |
|-----------|----------------|
| **config.py** | `model_id`, `max_seq_len`, `device`, `gamma` (spec), cache dirs |
| **device.py** | Resolve CPU vs MPS; logging |
| **tokenizer.py** | `str` ↔ `list[int]` (model-specific vocab) |
| **weights.py** | Load HF checkpoint; map keys → your `nn.Module` params |
| **model/gpt2.py** | Embeddings, 12 blocks, `lm_head`; `forward` → logits |
| **model/layers.py** | LayerNorm, mask, attention math, GELU |
| **cache.py** | Pre-allocated per-layer K/V; append on decode |
| **engine.py** | Owns session: prefill → decode loop → stop condition |
| **Sampler** | v0.1: greedy argmax on last logits (lives in engine) |
| **speculate.py** (v1.0) | Draft γ tokens → target verify → accept/reject |
| **metrics.py** | TTFT, ITL, TPS, memory |
| **bench/** | Reproducible runs + reports |
| **cli/** | User-facing commands |
| **tests/** | Parity vs HF; cache correctness; spec == vanilla |

---

## Project requirements

### Functional

1. Accept **text prompt** (CLI).
2. **Encode** with the correct model tokenizer.
3. Run **GPT-2 forward** you implemented (not `model.generate()`).
4. **Prefill** prompt; **decode** autoregressively.
5. **KV cache** so decode doesn’t recompute past tokens.
6. **Greedy** generation matching Hugging Face on fixed prompts (v0.1).
7. **Speculative** greedy matching vanilla (v1.0).
8. **Decode** token ids back to text.

### Non-functional (from plan)

- fp32 until parity holds; CPU first, then MPS.
- Parity tests vs `transformers` in **tests only**.
- Bench: TTFT, ITL, TPS, memory vs sequence length.
- Single request, no HTTP server, no continuous batching, no quant in v1.

### Out of scope (unless expanded later)

- HTTP API / multi-user serving
- Continuous batching, paged KV
- Quantization (int8/fp16)
- Llama backend (optional v2 — see [PLAN.md](PLAN.md))
- Sampling (top-p/temperature) in v0.1 — greedy only

---

## Gaps vs a simple mental model

| Simple view | Nuance |
|-------------|--------|
| Text → tokenizer → KV → model → text | Add **engine**, **weights load**, **logits → sample**, **prefill vs decode** |
| KV before model | KV is **updated by** attention during forward, not a preprocessor |
| Model → text | Model → **logits** → **argmax** → **token ids** (loop) → tokenizer → text |
| Single model | v1.0 adds **draft model** + **speculate.py** |
| — | **Stop rules**: `max_new_tokens`, optional EOS id |
| — | **Position ids** + **causal mask** (inside model) |
| — | **Session state**: generated ids, cache length, positions |

---

## Implementation order

```text
Phase 0: scaffold, device, CLI, tokenizer          ✓ (tokenizer done)
Phase 1: weights + model + prefill logits (parity)
Phase 2: engine + greedy decode without KV
Phase 3: cache.py + decode with KV                   → v0.1 MVP
Phase 4: CLI generate + bench + README
Phase 5: speculate + distilgpt2                      → v1.0
v2 (optional): model/llama.py + new parity
```

---

## GPT-2 vs Llama (optional v2)

Same runtime shell (engine, bench, spec algorithm); swap **model blocks**, **weights map**, **tokenizer**, and **KV layout** (GQA + RoPE). Details in [PLAN.md](PLAN.md).

---

## Optional: Mermaid in GitHub / extension

If you install **Markdown Preview Mermaid Support** in Cursor/VS Code, you can preview graph versions on [mermaid.live](https://mermaid.live) or on GitHub when this file is pushed.
