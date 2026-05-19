"""Phase 1 parity tests: our GPT-2 forward vs HF GPT2LMHeadModel.

Rules (from PLAN.md):
- fp32, CPU only (MPS tested separately once these pass)
- transformers only inside tests/, never in engine code
- atol 1e-4 / rtol 1e-3 tolerance
"""
import pytest
import torch
from transformers import GPT2LMHeadModel

from minispecir.weights import load_gpt2_architecture, load_hf_state_dict
from minispecir.model.gpt2 import GPT2Model

MODEL_ID = "gpt2"
DEVICE = torch.device("cpu")

# Token ids for fixed test prompts (pre-tokenised to avoid tokenizer dependency here)
# "Hello, my name is"  →  [15496, 11, 616, 1438, 318]
# "The quick brown fox" → [464, 2068, 7586, 21831]
PROMPTS_IDS = [
    torch.tensor([[15496, 11, 616, 1438, 318]], dtype=torch.long),
    torch.tensor([[464, 2068, 7586, 21831]], dtype=torch.long),
]


@pytest.fixture(scope="module")
def arch():
    return load_gpt2_architecture(MODEL_ID, local_files_only=False)


@pytest.fixture(scope="module")
def state():
    return load_hf_state_dict(MODEL_ID, local_files_only=False)


@pytest.fixture(scope="module")
def our_model(arch, state):
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


# ------------------------------------------------------------------
# Single-block parity
# ------------------------------------------------------------------

def test_single_block_parity(our_model, hf_model, state):
    """Block 0 output must match HF hidden_states[1] (first block output)."""
    input_ids = PROMPTS_IDS[0]  # [1, T]
    T = input_ids.shape[1]

    with torch.no_grad():
        hf_out = hf_model(input_ids, output_hidden_states=True)
        # hidden_states[0] = embedding output, [1] = after block 0
        hf_block0_out = hf_out.hidden_states[1]  # [1, T, C]

    # Our embedding
    wte = state["transformer.wte.weight"]
    wpe = state["transformer.wpe.weight"]
    pos_ids = torch.arange(T, device=DEVICE)
    x = wte[input_ids] + wpe[pos_ids]  # [1, T, C]

    with torch.no_grad():
        our_block0_out = our_model.block(x, 0)

    torch.testing.assert_close(our_block0_out, hf_block0_out, atol=1e-4, rtol=1e-3)


# ------------------------------------------------------------------
# Full prefill parity
# ------------------------------------------------------------------

@pytest.mark.parametrize("input_ids", PROMPTS_IDS)
def test_full_prefill_parity(our_model, hf_model, input_ids):
    """Full logits must match HF on fixed prompts (atol 1e-4)."""
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits  # [1, T, V]
        our_logits = our_model.forward(input_ids)  # [1, T, V]

    torch.testing.assert_close(our_logits, hf_logits, atol=1e-4, rtol=1e-3)


def test_next_token_argmax_matches_hf(our_model, hf_model):
    """The greedy next token (argmax of last position logits) must match HF."""
    input_ids = PROMPTS_IDS[0]

    with torch.no_grad():
        hf_next = hf_model(input_ids).logits[0, -1].argmax().item()
        our_next = our_model.forward(input_ids)[0, -1].argmax().item()

    assert our_next == hf_next
