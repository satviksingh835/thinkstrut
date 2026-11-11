"""Text normalisation, title normalisation, CPC parsing, FTS-compatible tokenisation and hashing."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

# ----------------------------------------------------------------------------------------------
# Text cleaning
# ----------------------------------------------------------------------------------------------
# U+2061-2064 (invisible function application/times/separator/plus) appear ~400 times inside
# formulas; zero-width chars and soft hyphens likewise carry no meaning for search.
_INVISIBLE_RE = re.compile("[­​‌‍⁠⁡⁢⁣⁤﻿]")
_SPACE_RE = re.compile(r"[\s  -   　]+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,;:])")  # "claim 1 , wherein" -> "claim 1, wherein"
_PAREN_OPEN_RE = re.compile(r"\(\s+")  # "( 18 )" -> "(18)"
_PAREN_CLOSE_RE = re.compile(r"\s+\)")
_TRAILING_DOT_RE = re.compile(r"\s+\.$")


def clean_text(s: str) -> str:
    """Light cleanup used for both display and embedding. Deliberately keeps '<' and '>' (math)."""
    if not s:
        return ""
    s = _INVISIBLE_RE.sub("", s)
    s = _SPACE_RE.sub(" ", s)
    s = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", s)
    s = _PAREN_OPEN_RE.sub("(", s)
    s = _PAREN_CLOSE_RE.sub(")", s)
    s = _TRAILING_DOT_RE.sub(".", s)
    return s.strip()


def norm_title(s: str) -> str:
    """Key for exact title matching: NFKC, casefold, punctuation -> space, collapsed whitespace."""
    s = unicodedata.normalize("NFKC", s or "").casefold()
    s = re.sub(r"[\W_]+", " ", s)
    return " ".join(s.split())


def title_for_embedding(title: str) -> str:
    """'PNEUMATIC TIRE' -> 'Pneumatic tire' (the model is uncased; this mainly helps display/debug)."""
    title = clean_text(title)
    return title.capitalize() if title.isupper() else title


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def abstract_hash(abstract: str) -> str | None:
    """Same-family marker: patents whose abstracts are identical after normalisation."""
    key = norm_title(abstract)
    return sha1(key)[:16] if key else None


# ----------------------------------------------------------------------------------------------
# FTS5-compatible tokenisation (used by the naive post-filter so it has the same semantics as the
# SQLite `unicode61 remove_diacritics 2` tokenizer used by the pre-filter)
# ----------------------------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[^\W_]+")


def fts_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric words with accents removed: 'Café-Racer HUB' -> ['cafe', 'racer', 'hub']."""
    # NFKD splits "é" into "e" + a combining accent, which the next line then drops
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _TOKEN_RE.findall(text.casefold())


# ----------------------------------------------------------------------------------------------
# CPC / IPC classification codes
# ----------------------------------------------------------------------------------------------
# Stored codes look like "B60C1103FI": subclass "B60C", then main group and subgroup digits with the
# separator stripped, then a constant suffix ("FI"). The digit split is not recoverable with
# certainty, so we use: 3 digits -> X/YY, 4 -> XX/YY, 5 -> XX/YYY, 6 -> XXX/YYY, and flag >= 5
# digits as ambiguous (e.g. 25138 could be 25/138 or 2/5138). Codes that already contain "/" are
# parsed exactly.
#
# Anatomy of a CPC code, from broad to narrow:  B (section) · B60 (class) · B60C (subclass: tyres)
#   · B60C 11 (main group: tread patterns) · B60C 11/03 (subgroup).
# Groups: section letter, class digits, subclass letter, group digits, explicit "/subgroup", suffix.
_CPC_RE = re.compile(r"^([A-HY])(\d{2})([A-Z])\s*(\d+)?\s*(?:/\s*(\d+))?\s*([A-Z]*)$")
_SUBCLASS_PREFIX_RE = re.compile(r"^[A-HY](\d{1,2}([A-Z])?)?$")  # "B", "B6", "B60", "B60B"


@dataclass(frozen=True)
class CPC:
    """A parsed classification code, e.g. raw "B60C1103FI" -> subclass B60C, group B60C11, subgroup 03."""

    raw: str
    subclass: str  # "B60C"
    group: str | None  # "B60C11"  (subclass + main group digits)
    subgroup: str | None  # "03"
    ambiguous: bool

    @property
    def compact(self) -> str:
        """Separator-free code without suffix: 'B60C1103'. Unambiguous for equality matching."""
        if self.group is None:
            return self.subclass
        return self.group + (self.subgroup or "")

    @property
    def display(self) -> str:
        if self.group is None:
            return self.subclass
        main = self.group[len(self.subclass):]
        return f"{self.subclass} {main}/{self.subgroup}" if self.subgroup else f"{self.subclass} {main}"


def _split_digits(d: str) -> tuple[str, str | None, bool]:
    """Separator-free group digits -> (main group, subgroup, ambiguous?). '1103' -> ('11', '03', False)."""
    n = len(d)
    if n <= 2:
        return d, None, False
    if n == 3:
        return d[:1], d[1:], False
    if n == 4:
        return d[:2], d[2:], False
    if n == 5:
        return d[:2], d[2:], True
    return d[: n - 3], d[n - 3:], True


def parse_cpc(raw: str | None) -> CPC | None:
    """Parse a stored classification string; None if it is empty or not a recognisable code."""
    if not raw or not raw.strip():
        return None
    s = raw.strip().upper()
    m = _CPC_RE.match(s)
    if not m:
        return None
    # "B60C1103FI" -> sec "B", cls "60", sub "C", digits "1103", explicit_sub None, suffix "FI"
    sec, cls, sub, digits, explicit_sub, _suffix = m.groups()
    subclass = f"{sec}{cls}{sub}"
    if digits is None:
        return CPC(raw=raw, subclass=subclass, group=None, subgroup=None, ambiguous=False)
    if explicit_sub is not None:
        return CPC(raw=raw, subclass=subclass, group=subclass + digits, subgroup=explicit_sub, ambiguous=False)
    main, subgroup, ambiguous = _split_digits(digits)
    return CPC(raw=raw, subclass=subclass, group=subclass + main, subgroup=subgroup, ambiguous=ambiguous)


@dataclass(frozen=True)
class CpcFilter:
    """kind: 'prefix' (on subclass, <= 4 chars), 'group' (main-group equality), 'full' (compact equality)."""

    kind: str
    value: str


def parse_cpc_filter(text: str) -> CpcFilter:
    """User input -> filter.  'B60B' -> prefix; 'B60C 11' -> group; 'B60C 11/03' or 'B60C1103' -> full."""
    t = re.sub(r"\s+", "", (text or "").upper())
    if not t:
        raise ValueError("empty CPC filter")
    if _SUBCLASS_PREFIX_RE.match(t):
        return CpcFilter("prefix", t)
    m = re.match(r"^([A-HY]\d{2}[A-Z])(\d+)/(\d+)$", t)
    if m:
        return CpcFilter("full", m.group(1) + m.group(2) + m.group(3))
    m = re.match(r"^([A-HY]\d{2}[A-Z])(\d+)$", t)
    if m:
        sub, digits = m.groups()
        # 1-2 digits = a main group ("B60C11"); 3+ digits = a compact full code ("B60C1103")
        return CpcFilter("group", sub + digits) if len(digits) <= 2 else CpcFilter("full", sub + digits)
    raise ValueError(f"unrecognised CPC filter {text!r} (examples: B60, B60B, 'B60C 11', 'B60C 11/03')")


def cpc_matches(f: CpcFilter, cpc: CPC | None) -> bool:
    """Python predicate with the same semantics as the SQL filter (used by the post-filter baseline)."""
    if cpc is None:
        return False
    if f.kind == "prefix":
        return cpc.subclass.startswith(f.value)
    if f.kind == "group":
        return cpc.group == f.value
    return cpc.compact == f.value
