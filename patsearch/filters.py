"""Hybrid (hard) constraints: CPC codes, keywords in title/abstract, exact title, title phrase, dates.

Two implementations with identical semantics:
  * `candidate_doc_ids` — the pre-filter: one SQL query (B-tree indexes + FTS5) -> sorted doc ids.
  * `DocPredicate`      — the naive post-filter used as a benchmark baseline: a Python check run
                        per ranked candidate.

Keyword semantics: a term matches when its words appear consecutively in the field (case- and
diacritic-insensitive, no stemming); a trailing '*' makes the last word a prefix ("groove*").
User text never reaches FTS5 verbatim: terms are re-tokenised into alphanumeric tokens first, so
input like `B60-B`, `"` or `*` cannot produce an FTS5 syntax error.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from dataclasses import asdict, dataclass, field

import numpy as np

from .normalize import CPC, CpcFilter, cpc_matches, fts_tokens, norm_title, parse_cpc, parse_cpc_filter


@dataclass
class FilterSpec:
    """The user's hard constraints exactly as typed. compile_filter() validates and parses them.

    Constraints of different types are AND'ed: cpc=["B60B"], keywords=["bicycle"] means
    "classified B60B *and* 'bicycle' in the title or abstract".
    """

    cpc: list[str] = field(default_factory=list)  # OR'ed
    keywords: list[str] = field(default_factory=list)
    kw_mode: str = "all"  # all | any
    kw_field: str = "both"  # both | title | abstract
    title: str | None = None  # exact (normalised) title
    title_contains: str | None = None  # phrase inside the title
    date_from: str | int | None = None  # publication date, inclusive (YYYY-MM-DD or YYYYMMDD)
    date_to: str | int | None = None

    def is_empty(self) -> bool:
        return not (self.cpc or self.keywords or self.title or self.title_contains or self.date_from or self.date_to)

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if v not in (None, [], "")}
        if not self.keywords:  # keyword options are meaningless without keywords
            d.pop("kw_mode", None)
            d.pop("kw_field", None)
        return d


@dataclass(frozen=True)
class Term:
    """One keyword or phrase, split into words the same way the FTS5 index splits text.

    "Sound absorber" -> tokens ("sound", "absorber"); "groove*" -> tokens ("groove",), prefix=True.
    """

    raw: str
    tokens: tuple[str, ...]
    prefix: bool

    def fts(self) -> str:
        """FTS5 query syntax: '"sound absorber"' (a phrase) or '"groove"*' (prefix match)."""
        return '"' + " ".join(self.tokens) + '"' + ("*" if self.prefix else "")

    def matches(self, toks: list[str]) -> bool:
        """Python version of the same test: do the term's words appear consecutively in `toks`?"""
        n = len(self.tokens)
        for i in range(len(toks) - n + 1):
            if all(toks[i + j] == self.tokens[j] for j in range(n - 1)):
                last = toks[i + n - 1]
                if (last.startswith(self.tokens[-1]) if self.prefix else last == self.tokens[-1]):
                    return True
        return False


def _term(raw: str) -> Term | None:
    """User text -> Term, or None if it has no letters/digits at all (e.g. '"' or '*')."""
    s = raw.strip()
    prefix = s.endswith("*")
    toks = tuple(fts_tokens(s.rstrip("*")))
    return Term(raw, toks, prefix) if toks else None


def _date(v: str | int | None) -> int | None:
    """'2025-02-01' / '20250201' -> 20250201 (dates are stored as YYYYMMDD integers)."""
    if v in (None, ""):
        return None
    s = str(v).strip().replace("-", "").replace("/", "")
    d = dt.datetime.strptime(s, "%Y%m%d")  # raises ValueError on bad input
    return int(d.strftime("%Y%m%d"))


@dataclass
class CompiledFilter:
    """A validated FilterSpec: CPC filters parsed, keywords tokenised, dates as integers."""

    spec: FilterSpec
    cpc: list[CpcFilter]
    keywords: list[Term]
    title_phrase: Term | None
    title_norm: str | None
    date_from: int | None
    date_to: int | None
    warnings: list[str]

    @property
    def is_empty(self) -> bool:
        return not (self.cpc or self.keywords or self.title_phrase or self.title_norm or self.date_from or self.date_to)


def compile_filter(spec: FilterSpec | None) -> CompiledFilter:
    """Validate and parse a FilterSpec. Raises ValueError on bad input (shown to the user as-is)."""
    spec = spec or FilterSpec()
    warnings: list[str] = []
    if spec.kw_mode not in ("all", "any"):
        raise ValueError("kw_mode must be 'all' or 'any'")
    if spec.kw_field not in ("both", "title", "abstract"):
        raise ValueError("kw_field must be 'both', 'title' or 'abstract'")
    cpc = [parse_cpc_filter(c) for c in spec.cpc if c and c.strip()]
    kws = []
    for raw in spec.keywords:
        t = _term(raw)
        if t is None:
            warnings.append(f"ignored keyword {raw!r} (no searchable characters)")
        else:
            kws.append(t)
    title_phrase = _term(spec.title_contains) if spec.title_contains else None
    if spec.title_contains and title_phrase is None:
        warnings.append(f"ignored title phrase {spec.title_contains!r} (no searchable characters)")
    tn = norm_title(spec.title) if spec.title else None
    return CompiledFilter(spec, cpc, kws, title_phrase, tn or None, _date(spec.date_from), _date(spec.date_to), warnings)


# ----------------------------------------------------------------------------------------------
# Pre-filter: SQL + FTS5
# ----------------------------------------------------------------------------------------------
_FTS_COLS = {"both": "{title abstract}", "title": "title", "abstract": "abstract"}


def fts_expression(cf: CompiledFilter) -> str | None:
    """The FTS5 MATCH string for the keyword/title-phrase constraints, e.g.

        keywords ["tread", "sound absorber"], title_contains "tire"
        -> '{title abstract} : ("tread" AND "sound absorber") AND title : ("tire")'

    `{title abstract} :` restricts the terms to those two columns.
    """
    parts = []
    if cf.keywords:
        joiner = " AND " if cf.spec.kw_mode == "all" else " OR "
        parts.append(f"{_FTS_COLS[cf.spec.kw_field]} : ({joiner.join(t.fts() for t in cf.keywords)})")
    if cf.title_phrase:
        parts.append(f"title : ({cf.title_phrase.fts()})")
    return " AND ".join(parts) if parts else None


def _next_prefix(p: str) -> str:
    """'B60B' -> 'B60C': every string starting with 'B60B' sorts in ['B60B', 'B60C')."""
    return p[:-1] + chr(ord(p[-1]) + 1)


def filter_sql(cf: CompiledFilter) -> tuple[str, list]:
    """One SQL query returning the doc_ids that pass every constraint, e.g. for cpc=B60B + kw=bicycle:

        SELECT doc_id FROM docs
        WHERE ((cpc_subclass >= 'B60B' AND cpc_subclass < 'B60C'))
        AND doc_id IN (SELECT rowid FROM docs_fts WHERE docs_fts MATCH '{title abstract} : ("bicycle")')
        ORDER BY doc_id

    The range condition is how a prefix match ("starts with B60B") uses the B-tree index.
    Values are passed as `?` parameters, never pasted into the SQL text.
    """
    where, params = [], []
    if cf.cpc:
        ors = []
        for f in cf.cpc:
            if f.kind == "prefix":  # index range scan == prefix match on the 4-char subclass
                ors.append("(cpc_subclass >= ? AND cpc_subclass < ?)")
                params += [f.value, _next_prefix(f.value)]
            elif f.kind == "group":
                ors.append("cpc_group = ?")
                params.append(f.value)
            else:
                ors.append("cpc_compact = ?")
                params.append(f.value)
        where.append("(" + " OR ".join(ors) + ")")
    if cf.title_norm:
        where.append("title_norm = ?")
        params.append(cf.title_norm)
    if cf.date_from:
        where.append("pub_date >= ?")
        params.append(cf.date_from)
    if cf.date_to:
        where.append("pub_date <= ?")
        params.append(cf.date_to)
    fts = fts_expression(cf)
    if fts:
        where.append("doc_id IN (SELECT rowid FROM docs_fts WHERE docs_fts MATCH ?)")
        params.append(fts)
    sql = "SELECT doc_id FROM docs" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY doc_id"
    return sql, params


def candidate_doc_ids(con: sqlite3.Connection, cf: CompiledFilter) -> np.ndarray:
    """Run the pre-filter: a sorted array of the doc_ids that satisfy every constraint."""
    sql, params = filter_sql(cf)
    rows = con.execute(sql, params).fetchall()
    return np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))


# ----------------------------------------------------------------------------------------------
# Naive post-filter predicate (benchmark baseline) — same semantics, evaluated per candidate
# ----------------------------------------------------------------------------------------------
@dataclass
class DocMeta:
    """The fields a post-filter check needs, kept in memory for every patent."""

    title: str
    abstract: str
    title_norm: str
    cpc: CPC | None
    pub_date: int | None


def load_doc_meta(con: sqlite3.Connection) -> list[DocMeta]:
    rows = con.execute("SELECT doc_id, title, abstract, title_norm, cpc_raw, pub_date FROM docs ORDER BY doc_id")
    return [DocMeta(r[1], r[2], r[3], parse_cpc(r[4]), r[5]) for r in rows]


class DocPredicate:
    """`pred(doc_id) -> bool`: does this one patent pass the filters? Used by the naive strategies,
    which rank first and then check candidates one at a time (slow for keywords: it re-reads and
    re-tokenises each abstract, which is exactly the work the FTS5 index avoids)."""

    def __init__(self, cf: CompiledFilter, meta: list[DocMeta]) -> None:
        self.cf = cf
        self.meta = meta

    def __call__(self, doc_id: int) -> bool:
        cf, m = self.cf, self.meta[doc_id]
        if cf.cpc and not any(cpc_matches(f, m.cpc) for f in cf.cpc):
            return False
        if cf.title_norm and m.title_norm != cf.title_norm:
            return False
        if cf.date_from and (m.pub_date is None or m.pub_date < cf.date_from):
            return False
        if cf.date_to and (m.pub_date is None or m.pub_date > cf.date_to):
            return False
        if cf.keywords or cf.title_phrase:
            title_toks = fts_tokens(m.title)
            if cf.title_phrase and not cf.title_phrase.matches(title_toks):
                return False
            if cf.keywords:
                fields = {
                    "both": (title_toks, fts_tokens(m.abstract)),
                    "title": (title_toks,),
                    "abstract": (fts_tokens(m.abstract),),
                }[cf.spec.kw_field]
                hits = (any(t.matches(f) for f in fields) for t in cf.keywords)
                if not (all(hits) if cf.spec.kw_mode == "all" else any(hits)):
                    return False
        return True
