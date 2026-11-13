"""Embedding backends: text -> vectors. All return L2-normalised float32 matrices.

L2-normalised means every vector has length 1. For unit vectors, cosine similarity is just the dot
product (cos θ = a·b / (|a||b|) = a·b), so the search code can score a query against every chunk
with a single matrix-vector multiplication.
"""

from __future__ import annotations

import os
import zlib
from typing import Protocol

import numpy as np

from .config import MODEL_ID, MODEL_REVISION, QUERY_PREFIX
from .tokens import HFTokenizer, SimpleTokenizer, Tokenizer


class Embedder(Protocol):
    """What the rest of the code needs from an embedding model (so tests can swap in HashEmbedder)."""

    model_id: str
    revision: str | None
    dim: int
    tokenizer: Tokenizer

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...

    def encode_passages(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray: ...


def _pick_device(device: str | None) -> str:
    """Use the Apple-silicon GPU (MPS) or an NVIDIA GPU (CUDA) when present; otherwise the CPU."""
    if device:
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SentenceTransformerEmbedder:
    """The real embedder: a Hugging Face model run through the sentence-transformers library."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: str | None = MODEL_REVISION,
        device: str | None = None,
        query_prefix: str = QUERY_PREFIX,
        dtype: str = "float32",
    ) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        self.model_id = model_id
        self.revision = revision if model_id == MODEL_ID else None
        self.device = _pick_device(device)
        self.model = SentenceTransformer(model_id, revision=self.revision, device=self.device)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        self.max_seq_length = int(self.model.max_seq_length)
        self.query_prefix = query_prefix if "bge" in model_id.lower() else ""
        self.tokenizer = HFTokenizer(self.model.tokenizer)
        # float16 roughly halves GPU time; on CPU it is slower, so it is only used on MPS/CUDA
        self.dtype = dtype if self.device in ("mps", "cuda") else "float32"
        if self.dtype == "float16":
            self.model.half()

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """For search queries: adds the model's query instruction (see config.QUERY_PREFIX)."""
        texts = [self.query_prefix + t for t in texts]
        return self._encode(texts, batch_size=32, show_progress=False)

    def encode_passages(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """For the text being searched (chunks) and for passage-to-passage matching (pasted claims)."""
        return self._encode(texts, batch_size=batch_size, show_progress=show_progress)

    def _encode(self, texts: list[str], batch_size: int, show_progress: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # batch_size texts go through the model at once (GPU-friendly); sentence-transformers sorts
        # them by length internally to minimise padding and returns them in the original order.
        # normalize_embeddings=True makes every vector unit length (cosine == dot product).
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        out = np.asarray(out, dtype=np.float32)
        if self.dtype == "float16":
            bad = ~np.isfinite(out).all(axis=1)
            if bad.any():  # fp16 overflow is rare; recompute those rows in float32
                self.model.float()
                out[bad] = self.model.encode([t for t, b in zip(texts, bad) if b], batch_size=batch_size,
                                             convert_to_numpy=True, normalize_embeddings=True)
                self.model.half()
            out /= np.linalg.norm(out, axis=1, keepdims=True)  # renormalise in float32
        return np.ascontiguousarray(out)


class HashEmbedder:
    """Deterministic signed-hashing bag of words. No model download; used in tests.

    Identical texts get identical vectors and word overlap gives positive cosine, which is all the
    engine tests need.
    """

    def __init__(self, dim: int = 256) -> None:
        self.model_id = f"hash-{dim}"
        self.revision = None
        self.dim = dim
        self.tokenizer = SimpleTokenizer()

    def _vec(self, text: str) -> np.ndarray:
        # "hashing trick": each word adds +1 or -1 to one of `dim` slots chosen by its hash, so texts
        # sharing words point in similar directions. crc32 (not hash()) keeps it stable across runs.
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            h = zlib.crc32(tok.encode("utf-8"))
            v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode_passages(texts)

    def encode_passages(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)


def load_embedder(model_id: str = MODEL_ID, device: str | None = None, dtype: str = "float32") -> Embedder:
    """'hash-256' -> HashEmbedder (no download, for smoke tests); anything else -> a real model."""
    if model_id.startswith("hash-"):
        return HashEmbedder(int(model_id.split("-", 1)[1]))
    return SentenceTransformerEmbedder(model_id=model_id, device=device, dtype=dtype)
