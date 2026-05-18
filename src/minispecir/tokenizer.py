from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

from minispecir.config import ModelConfig


@dataclass
class TokenizerWrapper:
    """Thin wrapper around a Hugging Face fast tokenizer (GPT-2 byte-level BPE)."""

    _tokenizer: Tokenizer
    model_id: str

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "gpt2",
        cache_dir: Path | str | None = None,
    ) -> TokenizerWrapper:
        """Load tokenizer.json from the Hugging Face hub (or local cache)."""
        if cache_dir is not None:
            local = Path(cache_dir)
            tokenizer_json = local / "tokenizer.json"
            if tokenizer_json.is_file():
                return cls(Tokenizer.from_file(str(tokenizer_json)), model_id=model_id)

        tokenizer = Tokenizer.from_pretrained(model_id)
        return cls(tokenizer, model_id=model_id)

    @classmethod
    def from_config(cls, config: ModelConfig) -> TokenizerWrapper:
        return cls.from_pretrained(config.model_id, cache_dir=config.cache_dir)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_batch(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool = False,
    ) -> list[list[int]]:
        encodings = self._tokenizer.encode_batch(
            texts,
            add_special_tokens=add_special_tokens,
        )
        return [enc.ids for enc in encodings]

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self,
        batch_ids: list[list[int]],
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in batch_ids
        ]

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size(with_added_tokens=True)
