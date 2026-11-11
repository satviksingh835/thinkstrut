"""Turn a patent into retrievable chunks.

Units: the abstract, each (re-assembled) claim, and description paragraphs greedily merged up to
~250 tokens. Each chunk keeps two texts:
  * display_text — what the user sees as evidence (cleaned source text)
  * embed_text   — title + display_text (optionally also the start of the parent claim for
    dependent claims; off by default, see ChunkConfig). The title restores context the upstream
    parser stripped (claim preambles) and anchors short fragments ("a width Wr ...") to the
    invention they belong to.
"""

from __future__ import annotations

from dataclasses import dataclass

from .claims import ClaimSet
from .config import ChunkConfig
from .load import Patent
from .normalize import title_for_embedding
from .tokens import Tokenizer

# Chunk kinds, stored as small integers per chunk (chunk_kind.i8.npy) so searches can be limited to,
# say, claims only ("scope") with a cheap array lookup.
KIND_ABSTRACT, KIND_CLAIM, KIND_DESCRIPTION, KIND_TITLE = 0, 1, 2, 3
KIND_NAMES = {KIND_ABSTRACT: "abstract", KIND_CLAIM: "claim", KIND_DESCRIPTION: "description", KIND_TITLE: "title"}
KIND_CODES = {v: k for k, v in KIND_NAMES.items()}


@dataclass
class Chunk:
    """One retrievable piece of a patent. Each chunk gets its own vector (one row of the matrix).

    label: what the UI shows as evidence ("abstract", "claim 3", "claims 13–20", "¶12–14").
    claim_lo/claim_hi, para_lo/para_hi: which claims / description paragraphs the chunk covers.
    """

    kind: int
    label: str
    display_text: str
    embed_text: str
    claim_lo: int | None = None
    claim_hi: int | None = None
    para_lo: int | None = None
    para_hi: int | None = None
    is_independent: bool = False


def _join(*parts: str) -> str:
    return "\n".join(p for p in parts if p)


def _parts_label(base: str, i: int, n: int) -> str:
    return base if n == 1 else f"{base} (part {i + 1}/{n})"


def _para_label(lo: int, hi: int) -> str:
    return f"¶{lo}" if lo == hi else f"¶{lo}–{hi}"


def chunk_patent(p: Patent, claims: ClaimSet, tok: Tokenizer, cfg: ChunkConfig) -> list[Chunk]:
    """A patent's chunks in a fixed order: abstract, claims (in claim order), description.

    The order matters downstream: the build appends them contiguously, so all of one patent's
    vectors sit in consecutive rows of the embedding matrix.
    """
    title = title_for_embedding(p.title)
    chunks: list[Chunk] = []

    # --- abstract -----------------------------------------------------------------------------
    if p.abstract:
        pieces = tok.windows(p.abstract, cfg.max_tokens, cfg.overlap_tokens)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(KIND_ABSTRACT, _parts_label("abstract", i, len(pieces)), piece, _join(title, piece))
            )

    # --- claims -------------------------------------------------------------------------------
    parent_ctx: dict[int, str] = {}
    for unit in claims.units:
        ctx = ""
        if unit.depends_on is not None and cfg.parent_tokens > 0:
            parent = claims.find(unit.depends_on)
            if parent is not None and parent is not unit:
                if parent.lo not in parent_ctx:
                    parent_ctx[parent.lo] = tok.windows(parent.text, cfg.parent_tokens, 0)[0]
                ctx = parent_ctx[parent.lo]
        pieces = tok.windows(unit.text, cfg.max_tokens, cfg.overlap_tokens)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    KIND_CLAIM,
                    _parts_label(unit.label, i, len(pieces)),
                    piece,
                    _join(title, ctx, piece),
                    claim_lo=unit.lo,
                    claim_hi=unit.hi,
                    is_independent=unit.is_independent,
                )
            )

    # --- description: greedy merge of consecutive paragraphs --------------------------------
    # Many paragraphs are a single short sentence, so they are merged until ~target_tokens:
    # chunks then carry enough context to embed well, and there are fewer vectors to search.
    paras = p.description
    counts = tok.count(paras)  # token count per paragraph, computed in one batch
    buf: list[str] = []  # paragraphs collected for the chunk being built
    buf_tokens = 0
    buf_lo = 0  # 1-based number of the first paragraph in `buf`

    def flush(hi: int) -> None:
        """Emit `buf` (paragraphs buf_lo..hi) as one description chunk and start a new one."""
        nonlocal buf, buf_tokens
        if buf:
            text = " ".join(buf)
            chunks.append(
                Chunk(KIND_DESCRIPTION, _para_label(buf_lo, hi), text, _join(title, text), para_lo=buf_lo, para_hi=hi)
            )
        buf, buf_tokens = [], 0

    for i, (para, n_tok) in enumerate(zip(paras, counts), start=1):
        if n_tok > cfg.max_tokens:
            # oversized paragraph: close the current chunk, then split this one into windows
            flush(i - 1)
            pieces = tok.windows(para, cfg.max_tokens, cfg.overlap_tokens)
            for j, piece in enumerate(pieces):
                chunks.append(
                    Chunk(
                        KIND_DESCRIPTION,
                        _parts_label(_para_label(i, i), j, len(pieces)),
                        piece,
                        _join(title, piece),
                        para_lo=i,
                        para_hi=i,
                    )
                )
            continue
        if buf and buf_tokens + n_tok > cfg.target_tokens:
            flush(i - 1)  # adding this paragraph would overshoot the target: close the chunk first
        if not buf:
            buf_lo = i
        buf.append(para)
        buf_tokens += n_tok
    flush(len(paras))

    # --- fallback: a patent with nothing but a title is still findable -----------------------
    if not chunks and title:
        chunks.append(Chunk(KIND_TITLE, "title", p.title, title))
    return chunks
