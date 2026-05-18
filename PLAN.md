# mini-spec-ir — Implementation Plan (revised)

**mini-spec-ir** = **Mini** (local, educational scale) + **Spec** (speculative decoding) + **IR** (inference runtime).

**Plan philosophy:** ship **correctness first**, then **KV-cache speedup**, then **speculative decoding** as a gated bonus. Do not block the portfolio on 2× TPS.

**Hardware:** Apple Silicon Mac, ~48 GB unified memory. **PyTorch + MPS** (fallback CPU). No CUDA.

**Honest positioning:** This is a **research / learning runtime**, not a competitor to [MLX](https://github.com/ml-explore/mlx) or llama.cpp on raw speed. Interview value = you can explain and implement what those tools hide.

---

## Naming vs “done” (decided)

**Repo name stays `mini-spec-ir`** — it describes the **full project**, not the first tag.

| Term | Meaning |
|------|---------|
| **Portfolio checkpoint** | **v0.1** — engine + KV + parity + bench. OK to put on resume as **WIP** / “core IR shipped.” |
| **Project complete** | **v1.0** — lossless greedy speculative decoding passes tests. Only then say the project “matches the name.” |

Do **not** claim speculative decoding is done at v0.1. README/resume: *“speculative decode → v1.0”*.

---

## Two release targets

| Release | Scope | “Done” means | Rough time |
|---------|--------|--------------|------------|
| **v0.1 (MVP)** | GPT-2 engine + prefill/decode + KV cache + parity tests + minimal CLI + README | Greedy output matches HF; KV decode faster than no-KV; you can whiteboard the loop | **~2 weeks** |
| **v1.0 (full)** | MVP + lossless greedy speculative decoding + bench report | v0.1 + spec tokens == vanilla; acceptance rate logged (speedup optional) | **+1–2 weeks** |

If you run out of time, **ship v0.1**. A correct engine with KV profiling already signals inference engineering. Spec is the chapter that maps directly to the optimization JD, not a prerequisite for MVP.

---

## What you are building

### v0.1 (MVP)

1. Load GPT-2 weights (prefer **safetensors** + memory-mapped or page-cached reads).
2. **Manual decoder forward** (attention + MLP + residuals), verified against `transformers` logits.
3. Explicit **prefill** vs **decode** paths.
4. **Pre-allocated KV cache**; greedy generation with cache matches without cache and matches HF.
5. **CLI** `generate` + **bench** script: TTFT, ITL, TPS, memory vs sequence length.

### v1.0 (adds “Spec”)

6. Second LM as **draft** (`distilgpt2` → target `gpt2`, same tokenizer).
7. **Greedy speculative** loop with verify pass that extends existing target KV correctly.
8. Tests: spec output **byte-for-byte identical** to vanilla greedy on fixed prompts.
9. Report **acceptance rate**; report speedup if any, explain if none.

---

## What this is *not*

| Not this | Why |
|----------|-----|
| `model.generate()` in the hot path | Defeats the purpose; HF only for parity tests |
| Beating MLX / llama.cpp on Mac TPS | Wrong bar; they use optimized C++/Metal |
| Full vLLM | No paged KV, continuous batching, multi-GPU in v1 |
| Llama-from-scratch in week 2 | RoPE + GQA + RMSNorm is a **second engine**, not a tweak |
| Guaranteed 2–3× spec speedup | Often **memory-bound** on laptop; may be ≤1× — that’s a valid README section |

---

## Corrections vs. the original chat plan

Things we are **explicitly not** assuming anymore:

| Original claim | Revised plan |
|----------------|--------------|
| 2× spec speedup on Mac is expected | **Optional.** Success = lossless greedy + acceptance rate logged. |
| Bigram / 1-layer “draft” | **Removed.** Draft = `distilgpt2` (or `gpt2` + `gpt2-medium` pair). |
| `torch.load(..., map_location="mps")` | **Load on CPU** (mmap/safetensors) → copy weights to MPS per module or at init. |
| KV + spec in 14 days flat | **MVP in ~14 days;** spec gated behind green parity suite (+7 days buffer). |
| “90% same as Llama, 5-line swap” | Llama = **v2 stretch** with its own parity milestone. |
| Spec before KV is stable | **Forbidden.** No `speculate.py` until KV greedy == HF on 5 prompts. |

---

## Core concepts (reference)

### KV cache

Store per-layer K/V for past positions; each decode step only computes Q/K/V for the **new** token. **Pre-allocate** `[max_seq]` buffers; index writes — no `torch.cat` in the hot loop.

### Speculative decoding (v1.0 only)

1. Draft autoregressively proposes γ tokens (cheap model).
2. Target runs **one causal forward** over those γ positions **starting from current KV state** (hard part).
3. Greedy verify: accept longest matching prefix; on mismatch, take target’s token at that position; discard rest of draft.
4. **Lossless (greedy):** token ids match target-only greedy generation.

**Interview note:** Production also uses EAGLE/Medusa (extra heads on one model). Mention in README; out of scope for v1.0.

### Prefill vs decode

- **Prefill:** all prompt tokens in one (or chunked) forward; fill KV.
- **Decode:** one new token per step (vanilla) or γ draft + verify forward (spec).

---

## Model strategy

**Default path:** GPT-2 only through v0.1 (MVP) and v1.0 (spec). **Llama 3 is optional v2** — not Phase 2, not required for the portfolio story.

### MVP + v1.0: GPT-2 family only

| Role | Model | Notes |
|------|--------|------|
| Target | `gpt2` (124M) | Parity reference; all phases until v1.0 |
| Draft (v1.0) | `distilgpt2` | Same GPT-2 BPE tokenizer — do not use a custom bigram |

**Tokenizer:** one `tokenizers` instance / vocab for both target and draft.

---

## GPT-2 vs Llama 3 — what’s actually different

You **cannot** load Llama 3 weights into the GPT-2 `forward()` and expect sensible output. Checkpoints are tied to a specific computation graph. “New backend” in this repo means **`model/llama.py` + Llama weight mapping + Llama parity tests** — not a second bench suite, CLI, or speculative algorithm.

### Architecture comparison (why the forward pass differs)

| Component | GPT-2 (`gpt2`) | Llama 3 |
|-----------|----------------|---------|
| Position encoding | Learned absolute embeddings `wpe` added to token embeds | **RoPE** — rotary encodings applied to Q and K; no `wpe` table |
| Normalization | **LayerNorm** (ε=1e-5, bias) before each sublayer | **RMSNorm** (no mean centering; usually no bias) |
| Attention | Multi-head attention (MHA): 12 Q / 12 K / 12 V heads @ 124M | **GQA**: many query heads, fewer KV heads (e.g. 32 Q / 8 KV on larger configs) |
| MLP | `Linear → GELU → Linear` (3072 hidden on 124M) | **SwiGLU**: gate + up projections, SiLU activation |
| LM head | **Tied** to token embedding `wte` | Typically **untied** separate head |
| HF weight layout | `Conv1D` (weight shape `[in, out]`, applied as `x @ W`) | Standard `nn.Linear`; still needs careful key mapping |

Same high-level picture (decoder-only transformer: attn → residual → MLP → residual), **different per-layer math and tensor shapes**.

### What reuses vs what you rewrite (v2 Llama)

| Layer of the stack | GPT-2 path | Llama 3 path |
|--------------------|------------|--------------|
| `model/gpt2.py` / `model/llama.py` | **Implement** | **Rewrite** (RoPE, RMSNorm, GQA, SwiGLU) |
| `weights.py` key mapping | GPT-2 `state_dict` keys | New mapping table |
| `tokenizer.py` | GPT-2 BPE | Llama tokenizer (SentencePiece) |
| `cache.py` | **Same idea**, different shapes | KV heads = `num_kv_heads`, not `num_heads`; RoPE position = `past_len` offset on Q/K |
| `engine.py` decode loop | prefill → `decode_step` | **Same loop**; calls `LlamaModel`, passes correct position offset for RoPE |
| `speculate.py` | draft=`distilgpt2` | draft must be **Llama-family** + same tokenizer |
| `bench/*`, `cli`, `metrics.py` | **Reuse** | Same metrics (TTFT, ITL, TPS, memory); different `model_id` in config |
| Parity tests | vs `GPT2LMHeadModel` | vs `AutoModelForCausalLM` (Llama); full suite re-run |

**KV cache:** concept is identical (store past K/V, append each step, attend over cache). Implementation differs because (1) **GQA** stores fewer K/V heads, (2) **RoPE** must rotate Q/K with the correct absolute position when writing new cache slots — a common v2 bug is off-by-one position vs cache length.

### Mental model

```
bench / CLI / engine loop / speculative algorithm  →  reuse
model/*.py + weight map + KV layout + RoPE positions  →  swap per model family
```

### v2 (optional): Llama-style module

Only consider after **GPT-2 v0.1 is green** (KV + bench) or **v1.0** (spec) if you want one thing at a time.

| v2 deliverable | Effort driver |
|----------------|---------------|
| `model/llama.py` | RoPE trig + GQA einsums + SwiGLU |
| `weights.py` | New HF key → parameter map |
| `tests/test_parity_llama.py` | Same milestone order as Phase 1 (block → full prefill) |
| Re-run Phase 3 gates on Llama | KV + RoPE interaction |

Suggested v2 model: a **small** Llama (e.g. `Llama-3.2-1B`) before 8B — faster parity iteration. Treat as **weeks**, not days.

**Out of scope unless explicitly chosen:** v2 is not implied by the repo name; GPT-2 + spec is a complete project.

---

## Platform: Mac / MPS / weights

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

**Weight loading (recommended):**

1. Download HF weights (safetensors preferred).
2. `mmap` / lazy read on **CPU**.
3. Instantiate modules on CPU or directly on MPS; **verify parity on CPU first** if MPS debugging is painful.
4. Second run should be fast via OS page cache — measure and mention in README.

**MPS:** call `.contiguous()` after slicing KV views.

**Precision:** **fp32 only** until all parity tests pass.

**Benchmarks:** warm-up forwards; laptop plugged in; cooldown between runs; report variance if numbers jitter.

---

## Repository layout

```
mini-spec-ir/
├── PLAN.md
├── README.md
├── pyproject.toml
├── src/minispecir/
│   ├── config.py
│   ├── device.py
│   ├── weights.py
│   ├── tokenizer.py
│   ├── model/
│   │   ├── gpt2.py       # Phase 1–v1.0
│   │   ├── llama.py      # v2 optional (RoPE, RMSNorm, GQA)
│   │   └── layers.py     # shared primitives where possible (masks); model-specific norms
│   ├── cache.py
│   ├── engine.py          # MVP: prefill, decode_step, generate_vanilla
│   ├── speculate.py       # v1.0 only
│   └── metrics.py
├── bench/
│   ├── parity.py
│   ├── bench_kv.py
│   ├── bench_spec.py      # v1.0
│   └── report.py
├── cli/main.py
└── tests/
    ├── test_parity.py
    ├── test_cache.py
    └── test_speculative.py  # v1.0
```

---

## Phases (revised)

### Phase 0 — Scaffold

- [x] `pyproject.toml`, package layout, `pytest`
- [x] `device.py`, documented model download
- [ ] CI optional: parity on CPU only

**Done when:** `python -m minispecir.cli.main --help` works. ✓

---

### Phase 1 — Parity: one block, then full model (Days 1–5)

**Hardest risk:** silent shape/transpose bugs. Budget time here.

- [ ] `weights.py` + `tokenizer.py`
- [ ] One transformer block → logits match HF (prefill), **fp32, CPU first**
- [ ] Full stack → prefill logits match (atol `1e-4` or tight rtol)
- [ ] Same tests on MPS once CPU is green

**Acceptance**

- [ ] `test_parity.py` passes CPU (+ MPS when stable)
- [ ] No `model.forward()` in your engine code path

---

### Phase 2 — Vanilla generation without KV (Days 5–7)

- [ ] `engine.py`: full recompute each token (slow but simple)
- [ ] Greedy decode matches `transformers` on **5 fixed prompts** (short + medium length)

**Acceptance**

- [ ] End-to-end token-id equality vs HF greedy

**Gate:** do not start KV cache until this passes.

---

### Phase 3 — KV cache (Days 8–14) — **MVP critical path**

**Hardest implementation:** position ids, causal mask with `past_len`, contiguous KV views, verify decode step equals “recompute all” for one step.

Milestones (do in order):

1. [ ] Single `decode_step` with cache matches one step of no-cache forward (logit parity).
2. [ ] Full greedy with cache == full greedy without cache.
3. [ ] Full greedy with cache == HF on 5 prompts.
4. [ ] Pre-allocated cache; audit: no `cat` in hot loop.

**Acceptance (MVP)**

- [ ] Token-id match vs HF on 5 prompts
- [ ] `bench_kv.py`: measurable decode speedup at seq ≥ 128 **or** document why not (bandwidth-bound) with profiler notes
- [ ] Plot memory vs seq length

**Not required for MVP:** spec, fancy CLI colors, fp16.

---

### Phase 4 — MVP polish (Days 12–14, can overlap Phase 3)

- [ ] `cli generate --prompt "..."`
- [ ] `bench/report.py` → Markdown with TTFT, ITL, TPS, device, dtype
- [ ] README v0.1: architecture diagram, prefill/decode, KV explanation, honest comparison to MLX/llama.cpp

### **MVP SHIP CHECKPOINT** ✓

You can stop here and still have a strong project.

---

### Phase 5 — Speculative decoding (Days 15–21, v1.0) — **gated**

**Prerequisite:** Phase 3 green on CPU; MPS nice-to-have.

**Hardest implementation:** `verify_forward(prompt_state, draft_tokens)` that:

- Extends target KV by γ positions in **one** forward,
- Produces per-position logits aligned with what vanilla would have produced,
- Feeds back into the same cache object for the next loop iteration.

Sub-milestones:

1. [ ] Load draft weights; draft-only greedy runs.
2. [ ] Target verify forward: logits at each draft position match “vanilla one-step” checks (unit test per γ).
3. [ ] Full spec loop; greedy token stream == vanilla on 5 prompts.
4. [ ] Log acceptance rate per step; tune γ ∈ {2, 4, 6}.

**Acceptance (v1.0)**

- [ ] `test_speculative.py`: spec greedy == vanilla greedy
- [ ] Acceptance rate reported in bench output
- [ ] **Speedup:** nice-to-have. If TPS(spec) ≤ TPS(vanilla), README section: “Why speculation didn’t win on this setup” (Python overhead, small model, MPS, low acceptance).

**Out of scope for v1.0:** sampling + spec, fp16 spec, Medusa/EAGLE.

---

### Phase 6 — v1.0 README & interview prep

- [ ] Spec loop diagram (draft → verify → accept/reject)
- [ ] Table: vanilla / KV / spec (correctness ✓, TPS, acceptance %)
- [ ] Talking point: lossless greedy spec; know when production uses EAGLE instead of two-model

---

## Definition of done

### v0.1 — shippable / portfolio checkpoint (required)

1. Greedy tokens match Hugging Face `gpt2` on 5 fixed prompts.
2. KV path matches no-KV path and HF.
3. Reproducible `bench` output + README explaining prefill, decode, KV.
4. You can draw the data flow on a whiteboard without the repo open.

**Not required for v0.1:** speculative decoding, spec speedup.

### v1.0 — project complete (name fully earned)

5. Greedy speculative == greedy vanilla (same prompts).
6. Acceptance rate in bench report.
7. Honest analysis of whether spec helped on Mac (yes or no).

---

## Testing strategy

| Test | When |
|------|------|
| Block + full prefill logits vs HF | Phase 1 |
| One decode step: cache vs recompute | Phase 3 |
| 5-prompt greedy vs HF | Phase 2–3 |
| Spec greedy vs vanilla | Phase 5 |
| Bench thresholds in CI | Optional, loose (MPS noise) |

**Rule:** `transformers` only inside `tests/` and `bench/parity.py`, never inside `engine.py` / `speculate.py`.

---

## Known troubles (expect these)

| Trouble | What to do |
|---------|------------|
| Parity fails for days | Debug one block, one head, CPU, compare intermediate tensors |
| KV works for 20 tokens, drifts later | Off-by-one in `past_len` / positions / mask |
| Spec “runs” but text diverges | Do not demo until `test_speculative` passes; verify forward is wrong |
| No spec speedup | Ship anyway with acceptance % + explanation |
| MPS flaky | Develop parity on CPU; MPS for bench only |
| Scope creep (Llama, quant, UI) | v2 / never |

---

## Metrics to report

**MVP:** TTFT, ITL, TPS (vanilla vs KV), memory vs seq len, device/dtype.

**v1.0 add:** acceptance rate, TPS(spec), optional speedup ratio.

Do **not** headline unmeasured “2.4×” — only numbers from your `bench` command.

---

## Suggested schedule (flexible)

| Week | Focus |
|------|--------|
| 1 | Phases 0–2: scaffold, parity, vanilla greedy |
| 2 | Phase 3–4: KV cache, MVP bench + README → **tag v0.1** |
| 3 | Phase 5–6: speculative + v1.0 docs → **tag v1.0** |

Buffer days belong in week 2–3 (KV and verify are where projects slip).

---

## JD alignment (what to say)

| Theme | MVP | + v1.0 |
|-------|-----|--------|
| Transformer internals | ✓ | ✓ |
| KV cache | ✓ | ✓ |
| Speculative decoding | — | ✓ |
| Profiling | ✓ bench | + acceptance |
| Production humility | “Not vLLM; learned by implementing decode state” | + lossless spec story |

---

## Immediate next action

**Phase 0 → Phase 1 on CPU:** scaffold, load `gpt2` weights, one block, prefill logit parity vs `transformers`.

Do not open `speculate.py` until `test_cache` and 5-prompt HF match are green.
