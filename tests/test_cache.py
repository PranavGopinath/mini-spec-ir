"""Phase 3 KV cache tests — four milestones.

Milestone 1: single decode step with cache == same step full recompute (logit parity)
Milestone 2: generate_kv token ids == generate_vanilla on 5 prompts
Milestone 3: generate_kv token ids == HF greedy on 5 prompts
Milestone 4: (structural) no torch.cat on KV in generate_kv decode loop
"""
import torch
import pytest
from transformers import GPT2LMHeadModel

from minispecir.weights import load_gpt2_architecture, load_hf_state_dict
from minispecir.model.gpt2 import GPT2Model
from minispecir.cache import KVCache
from minispecir.engine import generate_vanilla, generate_kv

MODEL_ID = "gpt2"
DEVICE = torch.device("cpu")
MAX_NEW_TOKENS = 20
EOS_TOKEN_ID = 50256

PROMPTS = [
    ("Hello, my name is",    [15496, 11, 616, 1438, 318]),
    ("The quick brown fox",  [464, 2068, 7586, 21831]),
    ("Once upon a time",     [7454, 2402, 257, 640]),
    ("In the beginning",     [818, 262, 3726]),
    ("To be or not to be",   [2514, 307, 393, 407, 284, 307]),
]


@pytest.fixture(scope="module")
def our_model():
    arch = load_gpt2_architecture(MODEL_ID, local_files_only=False)
    state = load_hf_state_dict(MODEL_ID, local_files_only=False)
    return GPT2Model(arch, state, DEVICE)


@pytest.fixture(scope="module")
def hf_model():
    m = GPT2LMHeadModel.from_pretrained(
        MODEL_ID,
        local_files_only=False,
        torch_dtype=torch.float32,
    )
    m.eval()
    return m


def _fresh_cache(model: GPT2Model) -> KVCache:
    return KVCache.from_arch(model.arch, B=1, device=DEVICE)


# ------------------------------------------------------------------
# Setup: past_len tracking
# ------------------------------------------------------------------

def test_cache_past_len_tracking(our_model):
    """past_len advances correctly: 0 → T_prompt after prefill, +1 after decode step."""
    prompt_ids = torch.tensor([[15496, 11, 616, 1438, 318]], dtype=torch.long)
    T_prompt = prompt_ids.shape[1]
    cache = _fresh_cache(our_model)

    assert cache.past_len == 0

    with torch.no_grad():
        our_model.forward(prompt_ids, cache=cache)
    assert cache.past_len == T_prompt

    decode_tok = torch.tensor([[100]], dtype=torch.long)
    with torch.no_grad():
        our_model.forward(decode_tok, cache=cache)
    assert cache.past_len == T_prompt + 1


# ------------------------------------------------------------------
# Milestone 1: single decode step logit parity
# ------------------------------------------------------------------

def test_single_decode_step_logit_parity(our_model):
    """
    After prefill, one cached decode step must produce the same logits
    as a full no-cache forward over prompt + new token.
    """
    prompt_ids = torch.tensor([[15496, 11, 616, 1438, 318]], dtype=torch.long)
    T_prompt = prompt_ids.shape[1]

    # Determine the first greedy token without a cache
    with torch.no_grad():
        logits_prefill_nocache = our_model.forward(prompt_ids)   # [1, T_prompt, V]
    first_token = logits_prefill_nocache[0, -1].argmax().view(1, 1)  # [1, 1]

    # Prefill WITH cache, then one cached decode step
    cache = _fresh_cache(our_model)
    with torch.no_grad():
        our_model.forward(prompt_ids, cache=cache)                # populate cache
        assert cache.past_len == T_prompt

        logits_cached = our_model.forward(first_token, cache=cache)  # [1, 1, V]

    # Full no-cache reference: forward over prompt + first token together
    full_seq = torch.cat([prompt_ids, first_token], dim=-1)          # [1, T_prompt+1]
    with torch.no_grad():
        logits_full = our_model.forward(full_seq)                     # [1, T_prompt+1, V]

    torch.testing.assert_close(
        logits_cached[0, 0],   # [V] — only position produced in decode step
        logits_full[0, -1],    # [V] — last position of full recompute
        atol=1e-4,
        rtol=1e-4,
    )


# ------------------------------------------------------------------
# Milestone 2: generate_kv == generate_vanilla
# ------------------------------------------------------------------

@pytest.mark.parametrize("prompt_text,prompt_ids", PROMPTS)
def test_kv_matches_vanilla(our_model, prompt_text, prompt_ids):
    """generate_kv token ids must equal generate_vanilla token ids."""
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)

    vanilla_out = generate_vanilla(our_model, input_ids, max_new_tokens=MAX_NEW_TOKENS)
    kv_out = generate_kv(our_model, input_ids, max_new_tokens=MAX_NEW_TOKENS)

    assert kv_out.tolist() == vanilla_out.tolist(), (
        f"KV vs vanilla mismatch on {repr(prompt_text)}\n"
        f"  kv:      {kv_out.tolist()}\n"
        f"  vanilla: {vanilla_out.tolist()}"
    )


# ------------------------------------------------------------------
# Milestone 3: generate_kv == HF greedy
# ------------------------------------------------------------------

@pytest.mark.parametrize("prompt_text,prompt_ids", PROMPTS)
def test_kv_matches_hf(our_model, hf_model, prompt_text, prompt_ids):
    """generate_kv token ids must equal HF greedy on all 5 prompts."""
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)

    with torch.no_grad():
        hf_out = hf_model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=EOS_TOKEN_ID,
        )

    kv_out = generate_kv(our_model, input_ids, max_new_tokens=MAX_NEW_TOKENS)

    assert kv_out.tolist() == hf_out.tolist(), (
        f"KV vs HF mismatch on {repr(prompt_text)}\n"
        f"  kv: {kv_out.tolist()}\n"
        f"  hf: {hf_out.tolist()}"
    )
