"""Llama 3 parity tests: our LlamaModel forward vs HF AutoModelForCausalLM.

Rules:
- bfloat16, CPU only
- atol 0.05 / rtol 0.02 (bfloat16 has ~7-bit mantissa; wider than GPT-2 fp32 tests)
- transformers only inside tests/, never in model code
- Skipped automatically if the model is not accessible (authentication required)

To run: ensure HF_TOKEN is set and you have access to meta-llama/Meta-Llama-3.1-8B.
"""
import pytest
import torch

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
DEVICE = torch.device("cpu")
DTYPE = torch.bfloat16

# Pre-tokenised prompts (Llama 3 BPE)
# "Hello, my name is" → [9906, 11, 856, 836, 374]
# "The quick brown fox" → [791, 4062, 14198, 39935]
PROMPTS_IDS = [
    torch.tensor([[9906, 11, 856, 836, 374]], dtype=torch.long),
    torch.tensor([[791, 4062, 14198, 39935]], dtype=torch.long),
]


def _model_accessible() -> bool:
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(MODEL_ID, local_files_only=False)
        return True
    except Exception:
        return False


requires_llama = pytest.mark.skipif(
    not _model_accessible(),
    reason=f"{MODEL_ID} not accessible — set HF_TOKEN and accept the license",
)


@pytest.fixture(scope="module")
def arch():
    from minispecir.weights import load_llama_architecture
    return load_llama_architecture(MODEL_ID, local_files_only=False)


@pytest.fixture(scope="module")
def state():
    from minispecir.weights import load_llama_state_dict
    return load_llama_state_dict(MODEL_ID, local_files_only=False, dtype=DTYPE)


@pytest.fixture(scope="module")
def our_model(arch, state):
    from minispecir.model.llama import LlamaModel
    return LlamaModel(arch, state, DEVICE, dtype=DTYPE)


@pytest.fixture(scope="module")
def hf_model():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        local_files_only=False,
        torch_dtype=DTYPE,
        device_map="cpu",
    )
    m.eval()
    return m


# ------------------------------------------------------------------
# Single-block parity
# ------------------------------------------------------------------

@requires_llama
def test_single_block_parity(our_model, hf_model, state):
    """Block 0 hidden-state output must match HF (atol 0.05, bfloat16)."""
    input_ids = PROMPTS_IDS[0]
    T = input_ids.shape[1]

    with torch.no_grad():
        hf_out = hf_model(input_ids, output_hidden_states=True)
        hf_block0_out = hf_out.hidden_states[1]  # after block 0

    # Our embedding (no wpe for Llama)
    wte = state["model.embed_tokens.weight"].to(DTYPE)
    x = wte[input_ids]

    with torch.no_grad():
        our_block0_out = our_model.block(x, 0)

    torch.testing.assert_close(our_block0_out, hf_block0_out, atol=0.05, rtol=0.02)


# ------------------------------------------------------------------
# Full prefill parity
# ------------------------------------------------------------------

@requires_llama
@pytest.mark.parametrize("input_ids", PROMPTS_IDS)
def test_full_prefill_parity(our_model, hf_model, input_ids):
    """Full logits must match HF on fixed prompts (atol 0.05, bfloat16)."""
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits   # [1, T, V]
        our_logits = our_model.forward(input_ids) # [1, T, V]

    torch.testing.assert_close(our_logits, hf_logits, atol=0.05, rtol=0.02)


@requires_llama
def test_next_token_argmax_matches_hf(our_model, hf_model):
    """Greedy next token (argmax of last logit position) must match HF."""
    input_ids = PROMPTS_IDS[0]

    with torch.no_grad():
        hf_next = hf_model(input_ids).logits[0, -1].argmax().item()
        our_next = our_model.forward(input_ids)[0, -1].argmax().item()

    assert our_next == hf_next
