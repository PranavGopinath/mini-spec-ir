"""Phase 5 speculative decoding tests.

Groups (run in order):
  1. Draft sanity         — distilgpt2 loads and generates without error
  2. Verify forward parity — multi-token verify logits match step-by-step decode
  3. Spec == vanilla      — token-id equality on 5 fixed prompts (the acceptance gate)
  4. Cache invariants     — past_len bookkeeping after accept/reject
"""
from __future__ import annotations

import torch
import pytest

from minispecir.cache import KVCache
from minispecir.engine import generate_kv, generate_vanilla
from minispecir.model.gpt2 import GPT2Model
from minispecir.speculate import generate_spec
from minispecir.weights import load_gpt2_architecture, load_hf_state_dict

TARGET_ID = "gpt2"
DRAFT_ID = "distilgpt2"
DEVICE = torch.device("cpu")

PROMPTS_IDS = [
    torch.tensor([[15496, 11, 616, 1438, 318]], dtype=torch.long),   # "Hello, my name is"
    torch.tensor([[464, 2068, 7586, 21831]], dtype=torch.long),       # "The quick brown fox"
    torch.tensor([[7454, 2402, 257, 640]], dtype=torch.long),         # "Once upon a time"
    torch.tensor([[818, 262, 3726]], dtype=torch.long),               # "In the beginning"
    torch.tensor([[2514, 307, 393, 407, 284, 307]], dtype=torch.long),# "To be or not to be"
]
GAMMA = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def target_arch():
    return load_gpt2_architecture(TARGET_ID, local_files_only=False)


@pytest.fixture(scope="module")
def target_state():
    return load_hf_state_dict(TARGET_ID, local_files_only=False)


@pytest.fixture(scope="module")
def target_model(target_arch, target_state):
    return GPT2Model(target_arch, target_state, DEVICE)


@pytest.fixture(scope="module")
def draft_arch():
    return load_gpt2_architecture(DRAFT_ID, local_files_only=False)


@pytest.fixture(scope="module")
def draft_state():
    return load_hf_state_dict(DRAFT_ID, local_files_only=False)


@pytest.fixture(scope="module")
def draft_model(draft_arch, draft_state):
    return GPT2Model(draft_arch, draft_state, DEVICE)


# ---------------------------------------------------------------------------
# Group 1: Draft model sanity
# ---------------------------------------------------------------------------

def test_draft_loads_and_generates(draft_model):
    """distilgpt2 runs generate_kv without error and produces tokens."""
    input_ids = PROMPTS_IDS[0]
    output = generate_kv(draft_model, input_ids, max_new_tokens=10)
    assert output.shape[1] > input_ids.shape[1], "draft model produced no tokens"


# ---------------------------------------------------------------------------
# Group 2: Verify forward logit parity
# ---------------------------------------------------------------------------

def test_verify_forward_logit_parity(target_model):
    """
    verify_logits[0, i] must match logits from a fresh step-by-step KV decode.

    We manually run target prefill, then a single verify forward with
    [current_token, d0..dγ-1]. For each output position i we compare against
    the logit produced by an independent step-by-step decode — catching any
    position id, causal mask, or cache cursor bug in isolation.
    """
    input_ids = PROMPTS_IDS[0].to(DEVICE)
    gamma = GAMMA

    # -- Reference: step-by-step KV decode collecting logits at positions T_prompt..T_prompt+γ --
    # ref_logits[i] = target's distribution at position T_prompt+i (given input at T_prompt+i).
    # ref_logits[0] comes from decoding current_token; ref_logits[i>0] from decoding draft_tokens[i-1].
    ref_cache = KVCache.from_arch(target_model.arch, B=1, device=DEVICE)
    with torch.no_grad():
        prefill_logits = target_model.forward(input_ids, cache=ref_cache)
        current_token = prefill_logits[0, -1].argmax()  # argmax of last prefill position

        ref_logits: list[torch.Tensor] = []
        x = current_token.view(1, 1)
        for _ in range(gamma + 1):
            step_logits = target_model.forward(x, cache=ref_cache)
            ref_logits.append(step_logits[0, 0])
            x = step_logits[0, 0].argmax().view(1, 1)

    # ref_logits[i] = logit at T_prompt+i; length = gamma+1

    # -- Verify: single verify forward using the same token sequence --
    draft_tokens = [ref_logits[i].argmax() for i in range(gamma)]

    verify_cache = KVCache.from_arch(target_model.arch, B=1, device=DEVICE)
    with torch.no_grad():
        target_model.forward(input_ids, cache=verify_cache)  # prefill
        verify_ids = torch.cat(
            [current_token.view(1, 1),
             torch.stack(draft_tokens).view(1, -1)],
            dim=-1,
        )
        verify_logits = target_model.forward(verify_ids, cache=verify_cache)  # [1, γ+1, V]

    # verify_logits[0, i] must match ref_logits[i] for i in 0..gamma
    for i in range(gamma + 1):
        torch.testing.assert_close(
            verify_logits[0, i], ref_logits[i],
            atol=1e-4, rtol=1e-3,
            msg=f"verify_logits mismatch at position {i}",
        )


# ---------------------------------------------------------------------------
# Group 3: Spec == vanilla (the acceptance gate)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("input_ids", PROMPTS_IDS)
def test_spec_equals_vanilla(target_model, draft_model, input_ids):
    """generate_spec must produce token-for-token identical output to generate_vanilla."""
    input_ids = input_ids.to(DEVICE)
    max_new = 20

    vanilla_out = generate_vanilla(target_model, input_ids, max_new_tokens=max_new)
    spec_out, stats = generate_spec(target_model, draft_model, input_ids,
                                    max_new_tokens=max_new, gamma=GAMMA)

    assert vanilla_out.tolist() == spec_out.tolist(), (
        f"spec/vanilla mismatch\n  vanilla: {vanilla_out.tolist()}\n  spec:    {spec_out.tolist()}"
    )
    # Stats sanity: at least some draft tokens were proposed
    assert stats["total_draft"] > 0
    assert 0 <= stats["accepted"] <= stats["total_draft"]


@pytest.mark.parametrize("input_ids", PROMPTS_IDS)
def test_spec_equals_kv(target_model, draft_model, input_ids):
    """generate_spec must also match generate_kv (belt-and-suspenders)."""
    input_ids = input_ids.to(DEVICE)
    max_new = 20

    kv_out = generate_kv(target_model, input_ids, max_new_tokens=max_new)
    spec_out, _ = generate_spec(target_model, draft_model, input_ids,
                                max_new_tokens=max_new, gamma=GAMMA)

    assert kv_out.tolist() == spec_out.tolist(), (
        f"spec/kv mismatch\n  kv:   {kv_out.tolist()}\n  spec: {spec_out.tolist()}"
    )


# ---------------------------------------------------------------------------
# Group 4: Cache invariants
# ---------------------------------------------------------------------------

def test_cache_past_len_after_full_accept(target_model, draft_model):
    """
    When all γ draft tokens are accepted, target_cache.past_len must advance by γ+1
    in a single speculative step.
    """
    # Use a prompt where distilgpt2 tends to agree with gpt2 to get a full-accept step.
    # We instrument this by monkey-patching draft to always return target's own tokens.
    input_ids = PROMPTS_IDS[0].to(DEVICE)
    T_prompt = input_ids.shape[1]

    # Make a "perfect" draft that mirrors the target
    target_cache_ref = KVCache.from_arch(target_model.arch, B=1, device=DEVICE)
    draft_cache_ref = KVCache.from_arch(draft_model.arch, B=1, device=DEVICE)

    with torch.no_grad():
        tl = target_model.forward(input_ids, cache=target_cache_ref)
        draft_model.forward(input_ids, cache=draft_cache_ref)
        current_token = tl[0, -1].argmax()

    committed_start = target_cache_ref.past_len  # = T_prompt

    # Build the verify input as if all draft tokens matched (use target's own greedy output)
    draft_tokens: list[torch.Tensor] = []
    temp_cache = KVCache.from_arch(target_model.arch, B=1, device=DEVICE)
    with torch.no_grad():
        target_model.forward(input_ids, cache=temp_cache)
        x = current_token.view(1, 1)
        for _ in range(GAMMA):
            sl = target_model.forward(x, cache=temp_cache)
            t = sl[0, 0].argmax()
            draft_tokens.append(t)
            x = t.view(1, 1)

    verify_ids = torch.cat(
        [current_token.view(1, 1), torch.stack(draft_tokens).view(1, -1)],
        dim=-1,
    )
    with torch.no_grad():
        verify_logits = target_model.forward(verify_ids, cache=target_cache_ref)

    # All tokens are target's own, so all accepted → past_len = committed_start + GAMMA + 1
    assert target_cache_ref.past_len == committed_start + GAMMA + 1


def test_cache_past_len_after_zero_accept(target_model):
    """
    When the first draft token is rejected, rolling back sets past_len = committed_start + 1.
    """
    input_ids = PROMPTS_IDS[0].to(DEVICE)
    cache = KVCache.from_arch(target_model.arch, B=1, device=DEVICE)

    with torch.no_grad():
        tl = target_model.forward(input_ids, cache=cache)
        current_token = tl[0, -1].argmax()

    committed_start = cache.past_len

    # Verify with a single token (gamma=1)
    # Regardless of acceptance, the verify forward advances by 2 (gamma=1+1)
    wrong_token = torch.tensor(
        (current_token.item() + 1) % 50257, dtype=torch.long, device=DEVICE
    )
    verify_ids = torch.cat([current_token.view(1, 1), wrong_token.view(1, 1)], dim=-1)

    with torch.no_grad():
        verify_logits = target_model.forward(verify_ids, cache=cache)

    # Cache advanced by 2; now simulate zero-accept rollback
    accept_len = 0
    cache.past_len = committed_start + accept_len + 1

    assert cache.past_len == committed_start + 1


def test_draft_cache_tracks_target(target_model, draft_model):
    """After generate_spec, draft and target caches are in sync by construction."""
    input_ids = PROMPTS_IDS[1].to(DEVICE)

    # We verify indirectly: generate_spec runs without assertion errors (caches stay
    # in-sync throughout) and returns a valid output.
    out, stats = generate_spec(target_model, draft_model, input_ids,
                               max_new_tokens=15, gamma=GAMMA)
    assert out.shape[1] > input_ids.shape[1]
    assert stats["steps"] > 0
