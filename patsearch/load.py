"""Stream the weekly JSON files and coerce each record into a `Patent`.

The loader is tolerant by design (Part 3 will feed it much larger, messier data): absent keys,
nulls and wrong types become empty values plus a quality flag. A record is dropped only when it has
no usable document number or no text at all; drop counts are reported in the build manifest.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .normalize import clean_text, sha1

# "US20240051333A1-20240215.XML" -> num 20240051333, kind A1 (publication type), date 20240215 (published).
# The large dataset omits the date ("US20240000148A1.XML"), so the date part is optional.
_FILENAME_RE = re.compile(r"^US(?P<num>\d+)(?P<kind>[A-Z]\d?)?(?:-(?P<date>\d{8}))?", re.I)
# fallback date from the weekly file name: "patents_ipa240215.json" -> "240215"
_SOURCE_DATE_RE = re.compile(r"ipa(\d{6})")


@dataclass
class Patent:
    """One patent application after cleaning. Text fields are never None (missing -> "" or []);
    anything that was missing, empty or of the wrong type is recorded in `flags`."""

    doc_number: str
    title: str
    abstract: str
    description: list[str]  # cleaned, blank paragraphs removed
    claims_raw: list[str]  # raw fragments (see claims.py)
    classification: str
    filename: str
    bibtex: str
    pub_date: int | None  # YYYYMMDD
    kind: str | None  # publication kind code, e.g. A1 / A9
    source_file: str
    flags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)  # other doc_numbers deduplicated into this one

    @property
    def has_description(self) -> bool:
        return bool(self.description)


@dataclass
class LoadStats:
    """Data-quality counters for the build manifest: what was read, what was dropped and why."""

    files: int = 0
    file_errors: list[tuple[str, str]] = field(default_factory=list)
    records: int = 0
    loaded: int = 0
    dropped_not_a_dict: int = 0
    dropped_no_doc_number: int = 0
    dropped_no_text: int = 0
    duplicate_doc_numbers: int = 0  # repeated records of the same patent, removed by dedupe()
    flags: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "files": self.files,
            "file_errors": self.file_errors,
            "records": self.records,
            "loaded": self.loaded,
            "dropped_not_a_dict": self.dropped_not_a_dict,
            "dropped_no_doc_number": self.dropped_no_doc_number,
            "dropped_no_text": self.dropped_no_text,
            "duplicate_doc_numbers": self.duplicate_doc_numbers,
            "flags": dict(self.flags.most_common()),
        }


def list_data_files(data_dir: Path) -> list[Path]:
    """Data files in a stable order: the small dataset's weekly patents_ipa*.json, or the large
    dataset's job_*/patents_batch_*.json; otherwise any *.json (except the large set's sample.json)."""
    d = Path(data_dir)
    files = sorted(d.glob("patents_ipa*.json")) or sorted(d.glob("job_*/patents_batch_*.json"))
    return files or sorted(p for p in d.glob("*.json") if p.name != "sample.json")


def _as_str(rec: dict, key: str, flags: list[str]) -> str:
    """Read `rec[key]` as a string without ever failing; note in `flags` if it was absent/null/odd."""
    v = rec.get(key)
    if key not in rec:
        flags.append(f"absent:{key}")
        return ""
    if v is None:
        flags.append(f"null:{key}")
        return ""
    if isinstance(v, str):
        if not v.strip():
            flags.append(f"empty:{key}")
        return v
    if isinstance(v, (int, float)):
        flags.append(f"coerced:{key}")
        return str(v)
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        flags.append(f"coerced:{key}")
        return " ".join(v)
    flags.append(f"bad_type:{key}")
    return ""


def _as_str_list(rec: dict, key: str, flags: list[str]) -> list[str]:
    """Read `rec[key]` as a list of strings (claims, description paragraphs) without ever failing."""
    v = rec.get(key)
    if key not in rec:
        flags.append(f"absent:{key}")
        return []
    if v is None:
        flags.append(f"null:{key}")
        return []
    if isinstance(v, str):
        flags.append(f"coerced:{key}")
        v = [v]
    if not isinstance(v, list):
        flags.append(f"bad_type:{key}")
        return []
    items = [x for x in v if isinstance(x, str)]
    if len(items) != len(v):
        flags.append(f"non_str_items:{key}")
    if not any(x.strip() for x in items):
        flags.append(f"empty:{key}")
    return items


def coerce_record(rec: Any, source_file: str, stats: LoadStats) -> Patent | None:
    """One raw JSON record -> Patent, or None (counted in `stats`) if it is unusable."""
    if not isinstance(rec, dict):
        stats.dropped_not_a_dict += 1
        return None
    flags: list[str] = []
    title = clean_text(_as_str(rec, "title", flags))
    doc_number = _as_str(rec, "doc_number", flags).strip()
    filename = _as_str(rec, "filename", flags).strip()
    abstract = clean_text(_as_str(rec, "abstract", flags))
    description_raw = _as_str_list(rec, "detailed_description", flags)
    claims_raw = _as_str_list(rec, "claims", flags)
    bibtex = _as_str(rec, "bibtex", flags)
    classification = _as_str(rec, "classification", flags).strip()

    # The undocumented `filename` field carries the publication date and kind code (A1/A9) and
    # can stand in for a missing doc_number.
    fm = _FILENAME_RE.match(filename) if filename else None
    if not doc_number and fm:
        doc_number = fm.group("num")
        flags.append("doc_number_from_filename")
    if not doc_number:
        stats.dropped_no_doc_number += 1
        return None

    pub_date: int | None = None
    kind: str | None = None
    if fm:
        kind = (fm.group("kind") or "").upper() or None
        if fm.group("date"):
            pub_date = int(fm.group("date"))
    if pub_date is None:
        sm = _SOURCE_DATE_RE.search(source_file)
        if sm:
            pub_date = int("20" + sm.group(1))
            flags.append("pub_date_from_source_file")

    # drop the "" spacer paragraphs (about half of all paragraphs); [""] becomes [] = no description
    description = [t for t in (clean_text(p) for p in description_raw) if t]
    if not description:
        flags.append("no_description")
    # keep the patent as long as *something* is searchable (it is then indexed on what it has)
    if not (title or abstract or description or any(c.strip() for c in claims_raw)):
        stats.dropped_no_text += 1
        return None

    for f in flags:
        stats.flags[f] += 1
    return Patent(
        doc_number=doc_number,
        title=title,
        abstract=abstract,
        description=description,
        claims_raw=claims_raw,
        classification=classification,
        filename=filename,
        bibtex=bibtex,
        pub_date=pub_date,
        kind=kind,
        source_file=source_file,
        flags=flags,
    )


def content_hash(p: Patent) -> str:
    """Fingerprint of a patent's text; equal hashes = byte-identical publications."""
    payload = json.dumps([p.title, p.abstract, p.claims_raw, p.description], ensure_ascii=False)
    return sha1(payload)


def dedupe(patents: list[Patent], stats: LoadStats | None = None) -> tuple[list[Patent], dict[str, str]]:
    """Collapse byte-identical publications (e.g. an A1 and its A9 corrected republication).

    The earliest publication is kept (its date is the one that matters for prior art); the other
    doc_numbers become aliases of it. Returns (kept patents, {alias: canonical doc_number}).
    Patents that merely share an abstract (family members with different claims) are kept.
    """
    # Pass 1: the same doc_number appearing more than once (re-scraped records in the large dataset)
    # keeps only its first copy.
    unique: dict[str, Patent] = {}
    for p in patents:
        if p.doc_number in unique:
            if stats is not None:
                stats.duplicate_doc_numbers += 1
            continue
        unique[p.doc_number] = p
    # Pass 2: different doc_numbers with byte-identical text become aliases.
    groups: dict[str, list[Patent]] = {}
    for p in unique.values():
        groups.setdefault(content_hash(p), []).append(p)
    kept: list[Patent] = []
    alias_map: dict[str, str] = {}
    for group in groups.values():
        group.sort(key=lambda p: (p.pub_date or 99999999, p.doc_number))  # earliest publication first
        canon = group[0]
        for other in group[1:]:
            if other.doc_number == canon.doc_number:
                continue  # literal duplicate record
            canon.aliases.append(other.doc_number)
            alias_map[other.doc_number] = canon.doc_number
        kept.append(canon)
    return kept, alias_map


def iter_patents(files: list[Path], stats: LoadStats) -> Iterator[Patent]:
    """Stream patents file by file (one JSON file in memory at a time)."""
    for path in files:
        stats.files += 1
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            stats.file_errors.append((path.name, repr(exc)))
            continue
        if not isinstance(data, list):
            stats.file_errors.append((path.name, f"top-level JSON is {type(data).__name__}, expected list"))
            continue
        for rec in data:
            stats.records += 1
            p = coerce_record(rec, path.name, stats)
            if p is not None:
                stats.loaded += 1
                yield p
