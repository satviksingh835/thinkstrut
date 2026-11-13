"""Re-assemble claims from the fragmented `claims` lists in the source data.

What the upstream parser did (observed in the data):
  * simple claims appear intact with their number:   "2 . The spoke according to claim 1 , wherein ..."
  * claims with nested sub-elements lost their number and preamble; only their sub-element texts
    appear, as unnumbered fragments:                  "a tread portion extending in ...;"
  * canceled claims appear as a range marker:         "1 - 12 . (canceled)"

Gap-aware segmentation: a numbered fragment N (> last seen number) starts claim N; unnumbered
fragments are collected and assigned to the numbering gap between the previous and the next
numbered claim. A gap of exactly one claim number gives an exact claim; a larger gap gives an
ambiguous block ("claims 13–20"); fragments after the last number form an open block ("claims 14+").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .normalize import clean_text

# Claim-number prefix: "7 . ", "1 - 5 . " (a range), "1 .- ", "4) ". Group 1 = number, group 2 = range end.
_NUM_RE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?(?:\.\s*-|\.|-|\))\s*")
# What follows the number when the claim(s) were withdrawn: "(canceled)" or "(cancelled)".
_CANCELED_RE = re.compile(r"^\(\s*cancell?ed\s*\)\s*\.?\s*", re.I)
# A reference to another claim: "according to claim 1", "of claims 2 to 4", ...
_DEP_RE = re.compile(r"\bclaims?\s+(\d+)", re.I)


@dataclass
class ClaimUnit:
    """One re-assembled claim (lo == hi), or a block of claims lo..hi whose boundaries were lost."""

    lo: int
    hi: int | None  # None = open-ended block (numbering lost after the last numbered claim)
    text: str
    depends_on: int | None = None

    @property
    def exact(self) -> bool:
        return self.hi == self.lo

    @property
    def headless(self) -> bool:
        """Starts mid-sentence: the preamble (and any "according to claim K") was lost upstream."""
        return bool(self.text) and self.text[0].islower()

    @property
    def is_independent(self) -> bool:
        # Claim 1 is always independent. A headless later claim may have lost its "according to
        # claim K" together with the preamble, so its independence is unknown -> not flagged.
        return self.exact and self.depends_on is None and (self.lo == 1 or not self.headless)

    @property
    def label(self) -> str:
        if self.exact:
            return f"claim {self.lo}"
        if self.hi is None:
            return f"claims {self.lo}+"
        return f"claims {self.lo}–{self.hi}"

    def contains(self, n: int) -> bool:
        return self.lo <= n and (self.hi is None or n <= self.hi)


@dataclass
class ClaimSet:
    """All claim units of one patent, in claim order, plus the canceled ranges (not indexed)."""

    units: list[ClaimUnit] = field(default_factory=list)
    canceled: list[tuple[int, int]] = field(default_factory=list)

    @property
    def n_canceled(self) -> int:
        return sum(b - a + 1 for a, b in self.canceled)

    @property
    def fully_exact(self) -> bool:
        return all(u.exact for u in self.units)

    def find(self, n: int) -> ClaimUnit | None:
        for u in self.units:
            if u.contains(n):
                return u
        return None


def _depends_on(text: str, own_lo: int) -> int | None:
    """'The tire according to claim 6, wherein ...' -> 6. Dependencies are stated in the claim's
    opening words, so only the first 250 characters are searched."""
    for m in _DEP_RE.finditer(text[:250]):
        k = int(m.group(1))
        if k < own_lo:  # only references to an earlier claim count as a dependency
            return k
    return None


def segment_claims(fragments: list[str]) -> ClaimSet:
    """Re-assemble one patent's claim fragments (rules in the module docstring).

    ["1 - 5 . (canceled)", "a tread portion ...;", "7 . The tire according to claim 6 , ..."]
    -> canceled (1, 5); claim 6 = "a tread portion ...;"; claim 7 = "The tire according to claim 6, ..."
    """
    out = ClaimSet()
    pending: list[str] = []  # unnumbered fragments not yet assigned to a claim number
    carry: list[str] = []  # text that precedes the first numbered claim: joins that claim
    last = 0  # highest claim number accounted for so far

    def flush(next_num: int | None) -> None:
        """Turn the pending fragments into one unit covering the numbering gap last+1 .. next_num-1
        (next_num=None: no numbered claim follows, so the unit is open-ended)."""
        nonlocal pending
        if not pending:
            return
        text = " ".join(t for t in (clean_text(p) for p in pending) if t)
        pending = []
        if not text:
            return
        lo = last + 1
        hi = next_num - 1 if next_num is not None else None
        if hi is not None and hi < lo:
            # no numbering gap -> these fragments continue the previous claim
            if out.units:
                out.units[-1].text = f"{out.units[-1].text} {text}"
            else:
                # Nothing to continue yet (text before claim 1, or after a canceled range): it is
                # the start of the next numbered claim. (Making it its own unit created two units
                # with the same number, which the large dataset exposed.)
                carry.append(text)
            return
        out.units.append(ClaimUnit(lo=lo, hi=hi, text=text))

    for frag in fragments:
        if not isinstance(frag, str) or not frag.strip():
            continue
        m = _NUM_RE.match(frag)
        # A number prefix starts a new claim only if it is higher than anything seen so far;
        # otherwise it is ordinary text that happens to begin with a number.
        if m and int(m.group(1)) > last:
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) else a  # "1 - 5 ." covers claims 1..5
            if b < a:
                b = a
            flush(a)  # fragments collected so far fill the gap before claim a
            rest = frag[m.end():]
            cm = _CANCELED_RE.match(rest)
            if cm:
                out.canceled.append((a, b))
                last = b
                leftover = rest[cm.end():].strip()
                if leftover:  # text after a canceled marker belongs to the next claim(s)
                    pending.append(leftover)
                continue
            text = clean_text(rest)
            if carry:
                text = " ".join(carry + [text]).strip()
                carry.clear()
            out.units.append(ClaimUnit(lo=a, hi=b, text=text))
            last = b
        else:  # unnumbered: a sub-element of a claim whose number and preamble were lost upstream
            pending.append(frag)
    flush(None)  # fragments after the last numbered claim -> open-ended block ("claims 14+")
    if carry:  # carried text that no numbered claim followed (e.g. only canceled claims after it)
        out.units.append(ClaimUnit(lo=last + 1, hi=None, text=" ".join(carry)))

    # second pass: record dependencies now that every unit's number is known
    for u in out.units:
        u.depends_on = _depends_on(u.text, u.lo)
    return out
