"""Token counting / windowing used by the chunker.

`HFTokenizer` uses the embedding model's own tokenizer so chunk budgets are exact (no silent
truncation at the model's 512-token window). `SimpleTokenizer` is a dependency-free stand-in for tests.
"""

from __future__ import annotations

import re
from typing import Protocol


class Tokenizer(Protocol):
    def count(self, texts: list[str]) -> list[int]:
        """Number of tokens in each text (without the model's special start/end tokens)."""
        ...

    def windows(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        """Split text into pieces of at most `max_tokens` tokens; consecutive pieces share `overlap`
        tokens so a sentence cut at a boundary still appears whole in one of them."""
        ...


def _window_bounds(n: int, max_tokens: int, overlap: int) -> list[tuple[int, int]]:
    """Token index ranges of the windows. n=700, max=320, overlap=64 -> (0,320), (256,576), (512,700)."""
    if n <= max_tokens:
        return [(0, n)]
    step = max(1, max_tokens - overlap)
    bounds = []
    start = 0
    while True:
        end = min(start + max_tokens, n)
        bounds.append((start, end))
        if end == n:
            return bounds
        start += step


class HFTokenizer:
    def __init__(self, hf_tokenizer) -> None:
        from transformers.utils import logging as hf_logging

        # counting tokens of long paragraphs (before splitting them) is intentional
        hf_logging.set_verbosity_error()
        self.tok = hf_tokenizer

    @classmethod
    def from_pretrained(cls, model_id: str, revision: str | None = None) -> HFTokenizer:
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(model_id, revision=revision))

    def count(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        enc = self.tok(
            texts, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False
        )
        return [len(ids) for ids in enc["input_ids"]]

    def windows(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        # offset_mapping gives each token's (start, end) character position in `text`, so token
        # windows can be cut back out of the original string (not rebuilt from tokens).
        enc = self.tok(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        offsets = enc["offset_mapping"]
        n = len(offsets)
        if n <= max_tokens:
            return [text]
        # word_ids()[i] = which word token i belongs to; "pneumatic" may be several sub-word
        # tokens ("pneu", "##matic") that share one word id.
        word_ids = enc.word_ids()
        out = []
        for start, end in _window_bounds(n, max_tokens, overlap):
            # never cut inside a word: move the start forward / the end backward to word boundaries
            while 0 < start < end - 1 and word_ids[start] is not None and word_ids[start] == word_ids[start - 1]:
                start += 1
            while start + 1 < end < n and word_ids[end] is not None and word_ids[end] == word_ids[end - 1]:
                end -= 1
            out.append(text[offsets[start][0] : offsets[end - 1][1]].strip())
        return [w for w in out if w]


class SimpleTokenizer:
    """Word/punctuation tokenizer (tests only)."""

    _re = re.compile(r"\w+|[^\w\s]")

    def count(self, texts: list[str]) -> list[int]:
        return [len(self._re.findall(t)) for t in texts]

    def windows(self, text: str, max_tokens: int, overlap: int) -> list[str]:
        spans = [m.span() for m in self._re.finditer(text)]
        if len(spans) <= max_tokens:
            return [text]
        return [
            text[spans[s][0] : spans[e - 1][1]].strip()
            for s, e in _window_bounds(len(spans), max_tokens, overlap)
        ]
