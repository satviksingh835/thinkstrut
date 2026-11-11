"""Central configuration. Everything that affects the built index is part of the build hash."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repository root (the folder with pyproject.toml)
DEFAULT_DATA_DIR = ROOT / "data" / "patent_data_small"  # where the provided JSON files live
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts"  # each build writes artifacts/<build_id>/ (see store.py)
LATEST_POINTER = "LATEST"  # text file inside the artifacts dir naming the most recent build

# Bump when the on-disk index layout changes. It is part of the build id, so an index written by
# older code is never mistaken for a current one.
SCHEMA_VERSION = 1

# The embedding model turns text into vectors. bge-small-en-v1.5 outputs 384 numbers per text and
# reads at most 512 tokens per input. It is "asymmetric": a search query gets an instruction prefix
# (QUERY_PREFIX) while the passages being searched are embedded as-is — that is how the model was
# trained to match short questions to longer passages.
MODEL_ID = "BAAI/bge-small-en-v1.5"
# Pinned model version (a Hugging Face commit hash) so every rebuild produces identical vectors.
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class ChunkConfig:
    """Token budgets (model tokenizer tokens, excluding [CLS]/[SEP]).

    embed_text = title (<= ~45 tok) + optional parent-claim context (<= parent_tokens) + chunk
    (<= max_tokens), which stays under the model's 512-token window, so nothing is silently truncated.

    parent_tokens defaults to 0 (off): prepending the parent claim made every dependent claim a
    near-copy of its parent, so evidence pointed at dependent claims whose own text did not match.
    """

    target_tokens: int = 250  # description paragraphs are greedily merged up to this size
    max_tokens: int = 320  # any single chunk body longer than this is split into windows
    overlap_tokens: int = 64  # overlap between windows of an oversized paragraph/claim
    parent_tokens: int = 0  # leading tokens of the parent claim prepended to dependent claims (0 = off)
    model_max_tokens: int = 512

    def to_dict(self) -> dict:
        return asdict(self)
