import pytest

from minispecir.tokenizer import TokenizerWrapper

# Fixed strings used again in parity / generation tests.
SAMPLE_TEXTS = [
    "Hello, world!",
    "The capital of France is",
    "",
    "GPT-2 uses byte-level BPE.",
]


@pytest.fixture(scope="module")
def tokenizer() -> TokenizerWrapper:
    return TokenizerWrapper.from_pretrained("gpt2")


@pytest.fixture(scope="module")
def hf_tokenizer():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


def test_round_trip(tokenizer: TokenizerWrapper) -> None:
    for text in SAMPLE_TEXTS:
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text


def test_encode_batch_matches_encode(tokenizer: TokenizerWrapper) -> None:
    batch = tokenizer.encode_batch(SAMPLE_TEXTS)
    assert len(batch) == len(SAMPLE_TEXTS)
    for text, ids in zip(SAMPLE_TEXTS, batch, strict=True):
        assert ids == tokenizer.encode(text)


def test_matches_huggingface(tokenizer: TokenizerWrapper, hf_tokenizer) -> None:
    for text in SAMPLE_TEXTS:
        expected = hf_tokenizer.encode(text, add_special_tokens=False)
        assert tokenizer.encode(text) == expected


def test_vocab_size_gpt2(tokenizer: TokenizerWrapper) -> None:
    assert tokenizer.vocab_size == 50257
