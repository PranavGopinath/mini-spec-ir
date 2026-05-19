"""Phase 2: end-to-end greedy generation matches HF on 5 fixed prompts.

Rules (from PLAN.md):
- transformers only in tests, never in engine.py
- token-id equality (not just text equality)
- CPU / fp32
"""
import pytest
import torch
from transformers import GPT2LMHeadModel

from minispecir.weights import load_gpt2_architecture, load_hf_state_dict
from minispecir.model.gpt2 import GPT2Model
from minispecir.engine import generate_vanilla

MODEL_ID = "gpt2"
DEVICE = torch.device("cpu")
MAX_NEW_TOKENS = 20
EOS_TOKEN_ID = 50256

# 5 fixed prompts, pre-tokenised with GPT-2 BPE
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


@pytest.mark.parametrize("prompt_text,prompt_ids", PROMPTS)
def test_greedy_matches_hf(our_model, hf_model, prompt_text, prompt_ids):
    """Our greedy output token-ids must be identical to HF greedy on every prompt."""
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)

    # HF greedy reference
    with torch.no_grad():
        hf_out = hf_model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=EOS_TOKEN_ID,
        )  # [1, T + new_tokens]

    # Our engine
    our_out = generate_vanilla(our_model, input_ids, max_new_tokens=MAX_NEW_TOKENS)

    assert our_out.tolist() == hf_out.tolist(), (
        f"Mismatch on prompt {repr(prompt_text)}\n"
        f"  ours: {our_out.tolist()}\n"
        f"  HF:   {hf_out.tolist()}"
    )
