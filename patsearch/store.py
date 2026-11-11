"""On-disk index layout and SQLite access.

artifacts/<build_id>/
  patents.sqlite        docs, aliases, claims, chunks (text), docs_fts (FTS5 over title+abstract)
  chunks.f32.npy        L2-normalised chunk embeddings, row i == chunk_id i   (memory-mapped)
  docvec.f32.npy        per-patent vector (normalised mean of abstract + claim chunks)
  chunk_doc.i32.npy     chunk_id -> doc_id
  chunk_kind.i8.npy     chunk_id -> kind code (see chunk.KIND_*)
  doc_chunk_lo.i64.npy  doc_id -> first chunk row   (chunks of a doc are contiguous)
  doc_chunk_hi.i64.npy  doc_id -> one past last chunk row
  manifest.json         model, counts, data-quality stats, per-stage build timings

doc_ids are assigned in (cpc_subclass, pub_date, doc_number) order, so every CPC-subclass filter
selects one contiguous slice of the embedding matrix.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from .config import DEFAULT_ARTIFACTS_DIR, LATEST_POINTER

# Tables:
#   docs     one row per patent. doc_id is the patent's position in the matrices (0..n-1);
#            chunk_lo/chunk_hi are the rows of its chunks in chunks.f32.npy. The cpc_*, title_norm
#            and pub_date columns are indexed: those indexes make the hybrid filters fast lookups
#            instead of scans.
#   aliases  other publication numbers of the same patent (A9 republications) -> doc_id
#   claims   the re-assembled claims (claims.py), for display and claim-number lookups
#   chunks   the text of every chunk, by chunk_id (= row in the embedding matrix). Text is only
#            fetched for the final top-k results, so it never has to sit in memory.
#   docs_fts an FTS5 full-text index over title + abstract — an inverted index (word -> patents),
#            which answers "abstract contains 'bicycle'" without reading every abstract.
SCHEMA = """
CREATE TABLE docs (
    doc_id          INTEGER PRIMARY KEY,
    doc_number      TEXT NOT NULL UNIQUE,
    kind            TEXT,
    title           TEXT NOT NULL,
    title_norm      TEXT NOT NULL,
    abstract        TEXT NOT NULL,
    pub_date        INTEGER,
    cpc_raw         TEXT,
    cpc_subclass    TEXT,
    cpc_group       TEXT,
    cpc_compact     TEXT,
    cpc_display     TEXT,
    cpc_ambiguous   INTEGER NOT NULL DEFAULT 0,
    has_description INTEGER NOT NULL,
    n_claim_units   INTEGER NOT NULL,
    n_canceled      INTEGER NOT NULL,
    abstract_hash   TEXT,
    flags           TEXT NOT NULL,
    filename        TEXT,
    source_file     TEXT,
    bibtex          TEXT,
    chunk_lo        INTEGER NOT NULL,
    chunk_hi        INTEGER NOT NULL
);
CREATE INDEX docs_cpc_subclass ON docs(cpc_subclass);
CREATE INDEX docs_cpc_group    ON docs(cpc_group);
CREATE INDEX docs_cpc_compact  ON docs(cpc_compact);
CREATE INDEX docs_title_norm   ON docs(title_norm);
CREATE INDEX docs_pub_date     ON docs(pub_date);
CREATE INDEX docs_abstract_hash ON docs(abstract_hash);

CREATE TABLE aliases (
    alias  TEXT PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES docs(doc_id)
);

CREATE TABLE claims (
    doc_id     INTEGER NOT NULL REFERENCES docs(doc_id),
    lo         INTEGER NOT NULL,
    hi         INTEGER,             -- NULL = open-ended block ("claims 14+")
    label      TEXT NOT NULL,
    exact      INTEGER NOT NULL,
    depends_on INTEGER,
    text       TEXT NOT NULL,
    PRIMARY KEY (doc_id, lo)
);

CREATE TABLE chunks (
    chunk_id       INTEGER PRIMARY KEY,   -- == row in chunks.f32.npy
    doc_id         INTEGER NOT NULL REFERENCES docs(doc_id),
    kind           INTEGER NOT NULL,
    label          TEXT NOT NULL,
    claim_lo       INTEGER,
    claim_hi       INTEGER,
    para_lo        INTEGER,
    para_hi        INTEGER,
    is_independent INTEGER NOT NULL DEFAULT 0,
    text           TEXT NOT NULL
);
CREATE INDEX chunks_doc ON chunks(doc_id);

-- keyword constraints: exact words (no stemming), case- and diacritic-insensitive
CREATE VIRTUAL TABLE docs_fts USING fts5(
    title, abstract,
    content='docs', content_rowid='doc_id',
    tokenize='unicode61 remove_diacritics 2'
);
"""

DOC_COLUMNS = (
    "doc_id doc_number kind title title_norm abstract pub_date cpc_raw cpc_subclass cpc_group cpc_compact "
    "cpc_display cpc_ambiguous has_description n_claim_units n_canceled abstract_hash flags filename "
    "source_file bibtex chunk_lo chunk_hi"
).split()
CHUNK_COLUMNS = "chunk_id doc_id kind label claim_lo claim_hi para_lo para_hi is_independent text".split()
CLAIM_COLUMNS = "doc_id lo hi label exact depends_on text".split()

# numpy arrays next to the database. The per-chunk/per-doc arrays are small (a few bytes per row)
# and are what the search loop actually touches; SQLite is only used for filters and final results.
ARRAY_FILES = {
    "emb": "chunks.f32.npy",
    "docvec": "docvec.f32.npy",
    "chunk_doc": "chunk_doc.i32.npy",
    "chunk_kind": "chunk_kind.i8.npy",
    "doc_chunk_lo": "doc_chunk_lo.i64.npy",
    "doc_chunk_hi": "doc_chunk_hi.i64.npy",
}


def _insert_sql(table: str, cols: Sequence[str]) -> str:
    return f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"


class StoreWriter:
    """Creates a fresh patents.sqlite during the build and bulk-inserts rows into it."""

    def __init__(self, path: Path) -> None:
        path = Path(path)
        if path.exists():
            path.unlink()
        self.con = sqlite3.connect(path)
        # The build writes into a temporary directory and is all-or-nothing, so SQLite's crash
        # protection (journal + fsync on every commit) is skipped for speed.
        self.con.execute("PRAGMA journal_mode=OFF")
        self.con.execute("PRAGMA synchronous=OFF")
        self.con.executescript(SCHEMA)

    def write_docs(self, rows: Iterable[dict]) -> None:
        self.con.executemany(_insert_sql("docs", DOC_COLUMNS), ([r[c] for c in DOC_COLUMNS] for r in rows))

    def write_chunks(self, rows: Iterable[dict]) -> None:
        self.con.executemany(_insert_sql("chunks", CHUNK_COLUMNS), ([r[c] for c in CHUNK_COLUMNS] for r in rows))

    def write_claims(self, rows: Iterable[dict]) -> None:
        self.con.executemany(_insert_sql("claims", CLAIM_COLUMNS), ([r[c] for c in CLAIM_COLUMNS] for r in rows))

    def write_aliases(self, pairs: Iterable[tuple[str, int]]) -> None:
        self.con.executemany("INSERT OR IGNORE INTO aliases (alias, doc_id) VALUES (?, ?)", pairs)

    def finalize(self) -> None:
        # fill the keyword index from the docs table in one pass
        self.con.execute("INSERT INTO docs_fts(docs_fts) VALUES ('rebuild')")
        # gather table statistics so SQLite's query planner picks the right index for each filter
        self.con.execute("ANALYZE")
        self.con.commit()
        self.con.close()


class Store:
    """Read-only access to a built index: the SQLite database plus the numpy arrays.

    mmap=True memory-maps the big embedding matrix: the OS pages it in from disk on first use
    instead of reading it all at start-up (quick start for the CLI; the server uses mmap=False).
    """

    def __init__(self, artifact_dir: Path, mmap: bool = True) -> None:
        self.dir = Path(artifact_dir)
        db = self.dir / "patents.sqlite"
        if not db.exists():
            raise FileNotFoundError(f"no index at {self.dir} (run `patsearch build` first)")
        self.con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        mode = "r" if mmap else None
        self.emb = np.load(self.dir / ARRAY_FILES["emb"], mmap_mode=mode)
        self.docvec = np.load(self.dir / ARRAY_FILES["docvec"])
        self.chunk_doc = np.load(self.dir / ARRAY_FILES["chunk_doc"])
        self.chunk_kind = np.load(self.dir / ARRAY_FILES["chunk_kind"])
        self.doc_chunk_lo = np.load(self.dir / ARRAY_FILES["doc_chunk_lo"])
        self.doc_chunk_hi = np.load(self.dir / ARRAY_FILES["doc_chunk_hi"])

    @property
    def n_docs(self) -> int:
        return int(self.doc_chunk_lo.shape[0])

    @property
    def n_chunks(self) -> int:
        return int(self.emb.shape[0])

    # --- lookups -------------------------------------------------------------------------------
    def resolve(self, doc_number: str) -> int | None:
        """Publication number (or an alias such as an A9 republication) -> internal doc_id, or None."""
        doc_number = normalize_doc_number(doc_number)
        row = self.con.execute("SELECT doc_id FROM docs WHERE doc_number = ?", (doc_number,)).fetchone()
        if row:
            return int(row[0])
        row = self.con.execute("SELECT doc_id FROM aliases WHERE alias = ?", (doc_number,)).fetchone()
        return int(row[0]) if row else None

    def docs(self, doc_ids: Sequence[int], columns: str = "*") -> dict[int, sqlite3.Row]:
        """Fetch rows of the docs table for several doc_ids in one query: {doc_id: row}."""
        if not len(doc_ids):
            return {}
        ids = [int(i) for i in doc_ids]
        q = f"SELECT {columns} FROM docs WHERE doc_id IN ({','.join('?' * len(ids))})"
        return {int(r["doc_id"]): r for r in self.con.execute(q, ids)}

    def chunks(self, chunk_ids: Sequence[int]) -> dict[int, sqlite3.Row]:
        """Fetch chunk rows (label, text, ...) by chunk_id: {chunk_id: row}."""
        if not len(chunk_ids):
            return {}
        ids = [int(i) for i in chunk_ids]
        q = f"SELECT * FROM chunks WHERE chunk_id IN ({','.join('?' * len(ids))})"
        return {int(r["chunk_id"]): r for r in self.con.execute(q, ids)}

    def aliases_of(self, doc_ids: Sequence[int]) -> dict[int, list[str]]:
        if not len(doc_ids):
            return {}
        ids = [int(i) for i in doc_ids]
        q = f"SELECT doc_id, alias FROM aliases WHERE doc_id IN ({','.join('?' * len(ids))}) ORDER BY alias"
        out: dict[int, list[str]] = {}
        for r in self.con.execute(q, ids):
            out.setdefault(int(r[0]), []).append(r[1])
        return out

    def claims_of(self, doc_id: int) -> list[sqlite3.Row]:
        return list(self.con.execute("SELECT * FROM claims WHERE doc_id = ? ORDER BY lo", (doc_id,)))

    def claim_chunk_ids(self, doc_id: int, lo: int) -> list[int]:
        """Rows of the chunk(s) holding the claim unit that starts at `lo` (long claims span several)."""
        q = "SELECT chunk_id FROM chunks WHERE doc_id = ? AND kind = 1 AND claim_lo = ? ORDER BY chunk_id"
        return [int(r[0]) for r in self.con.execute(q, (doc_id, lo))]


def normalize_doc_number(s: str) -> str:
    """Accept '20240051333', 'US20240051333A1', 'US 2024/0051333 A1' -> '20240051333'."""
    s = re.sub(r"\s+", "", s.strip().upper())
    s = re.sub(r"^US", "", s)
    s = re.sub(r"[A-Z]\d?$", "", s)  # trailing kind code (A1, A9, B2, ...)
    return re.sub(r"\D", "", s)


def resolve_artifact_dir(path: Path | None = None) -> Path:
    """Accept a build dir, or an artifacts root containing a LATEST pointer."""
    root = Path(path) if path else DEFAULT_ARTIFACTS_DIR
    if (root / "patents.sqlite").exists():
        return root
    pointer = root / LATEST_POINTER
    if pointer.exists():
        return root / pointer.read_text().strip()
    raise FileNotFoundError(f"no index found under {root} (run `patsearch build` first)")
