"""Quality cost of an efficiency change, measured against a reference (baseline) index.

  # simulate embedding only the first N description chunks (no re-embedding: rows are masked)
  uv run python scripts/eval_agreement.py --ref artifacts/large_5k_baseline --desc-chunks 4,8,16,32
  # compare another build (e.g. fp16 vectors) with the reference
  uv run python scripts/eval_agreement.py --ref artifacts/large_5k_baseline --cand artifacts/large_5k_fp16

Query sets (seeded): claim 1 and the abstract of 200 random patents (own patent excluded), the text
queries of bench/queries_large.jsonl, and a known-item test: a description chunk from late in a patent
(beyond the truncation point) must still retrieve its own patent in the top 10.
Metrics: agreement@10 = share of the reference top-10 patents also returned; known-item recall@10, MRR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from patsearch.store import Store, resolve_artifact_dir  # noqa: E402

K, N_QUERIES, BLOCK = 10, 200, 64
TEXT_QUERIES = Path(__file__).resolve().parent.parent / "bench" / "queries_large.jsonl"


class Index:
    def __init__(self, path: Path) -> None:
        self.store = Store(resolve_artifact_dir(path), mmap=False)
        s = self.store
        self.emb = np.asarray(s.emb, dtype=np.float32)
        self.lo, self.hi, self.kind = s.doc_chunk_lo, s.doc_chunk_hi, s.chunk_kind
        rows = s.con.execute("SELECT doc_id, doc_number FROM docs").fetchall()
        self.doc_number = {r[0]: r[1] for r in rows}
        self.doc_id = {r[1]: r[0] for r in rows}
        self.labels = [r[0] for r in s.con.execute("SELECT label FROM chunks ORDER BY chunk_id")]
        self.row_of = {}  # (doc_number, label) -> row
        for row, d in enumerate(s.chunk_doc.tolist()):
            self.row_of.setdefault((self.doc_number[d], self.labels[row]), row)
        is_desc = (self.kind == 2).astype(np.int64)
        csum = np.cumsum(is_desc)
        before_doc = np.repeat(csum[self.lo] - is_desc[self.lo], self.hi - self.lo)
        self.desc_ordinal = np.where(is_desc == 1, csum - before_doc - 1, -1)  # 0-based within each patent
        self.text_len = np.asarray([r[0] for r in s.con.execute(
            "SELECT length(text) FROM chunks ORDER BY chunk_id")], dtype=np.int64)

    def top_docs(self, Q: np.ndarray, exclude: list[int | None], mask: np.ndarray | None = None) -> list[list[str]]:
        """Top-K doc_numbers per query row (best chunk per patent), optionally ignoring masked rows."""
        out = []
        for a in range(0, len(Q), BLOCK):
            S = self.emb @ Q[a:a + BLOCK].T  # rows x block
            if mask is not None:
                S[~mask] = -np.inf
            best = np.maximum.reduceat(S, self.lo, axis=0)  # docs x block
            for j in range(best.shape[1]):
                col = best[:, j]
                ex = exclude[a + j]
                if ex is not None:
                    col[ex] = -np.inf
                top = np.argpartition(-col, K)[:K]
                top = top[np.argsort(-col[top], kind="stable")]
                out.append([self.doc_number[int(d)] for d in top])
        return out


def build_queries(ref: Index, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    docs = rng.choice(ref.store.n_docs, size=min(N_QUERIES, ref.store.n_docs), replace=False)
    sets = {"claim 1": [], "abstract": [], "known-item (late description)": []}
    for d in docs.tolist():
        rows = np.arange(ref.lo[d], ref.hi[d])
        claims = rows[ref.kind[rows] == 1]
        abstract = rows[ref.kind[rows] == 0]
        desc = rows[ref.kind[rows] == 2]
        if len(claims):
            sets["claim 1"].append((int(d), int(claims[0])))
        if len(abstract):
            sets["abstract"].append((int(d), int(abstract[0])))
        if len(desc) > 32:  # late passage: beyond every truncation point tested
            sets["known-item (late description)"].append((int(d), int(rng.choice(desc[32:]))))
    return sets


def embed_text_queries(ref: Index) -> np.ndarray | None:
    if not TEXT_QUERIES.exists():
        return None
    texts = [json.loads(line)["text"] for line in TEXT_QUERIES.read_text().splitlines()
             if line and json.loads(line)["type"] == "text"]
    from patsearch.embed import load_embedder
    return load_embedder(ref.store.manifest["model"]["id"]).encode_queries(texts) if texts else None


def agreement(a: list[list[str]], b: list[list[str]]) -> float:
    return float(np.mean([len(set(x) & set(y)) / K for x, y in zip(a, b)])) if a else float("nan")


def known_item(results: list[list[str]], sources: list[str]) -> tuple[float, float]:
    hits = [r.index(s) + 1 if s in r else 0 for r, s in zip(results, sources)]
    return float(np.mean([h > 0 for h in hits])), float(np.mean([1 / h if h else 0 for h in hits]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--cand", type=Path)
    ap.add_argument("--desc-chunks", default="", help="comma-separated N values to simulate")
    ap.add_argument("--out", type=Path, default=Path("results/part3_quality.json"))
    args = ap.parse_args()

    ref = Index(args.ref)
    sets = build_queries(ref)
    Qtext = embed_text_queries(ref)
    report = {"ref": str(args.ref), "n_docs": ref.store.n_docs, "n_rows": int(len(ref.emb)), "rows": []}

    def evaluate(name: str, run) -> dict:
        row = {"config": name}
        for set_name, items in sets.items():
            Q = np.stack([ref.emb[r] for _, r in items])
            src = [ref.doc_number[d] for d, _ in items]
            keep_source = set_name.startswith("known-item")  # the known-item test must find its own patent
            base = run(Q, items, ref_side=True, keep_source=keep_source)
            got = run(Q, items, ref_side=False, keep_source=keep_source)
            if set_name.startswith("known-item"):
                row["known-item recall@10"], row["known-item MRR"] = known_item(got, src)
                row["known-item recall@10 (reference)"] = known_item(base, src)[0]
            else:
                row[f"agreement@10 {set_name}"] = round(agreement(base, got), 3)
        if Qtext is not None:
            none = [None] * len(Qtext)
            row["agreement@10 text"] = round(agreement(run(Qtext, none, True, text=True),
                                                       run(Qtext, none, False, text=True)), 3)
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}

    for n in [int(x) for x in args.desc_chunks.split(",") if x.strip()]:
        mask = (ref.kind != 2) | (ref.desc_ordinal < n)
        kept = float(ref.text_len[mask].sum() / ref.text_len.sum())

        def run(Q, items, ref_side, text=False, mask=mask, keep_source=False):
            exclude = [None] * len(Q) if text or keep_source else [d for d, _ in items]
            return ref.top_docs(Q, exclude, None if ref_side else mask)

        row = evaluate(f"first {n} description chunks", run)
        row["share of text embedded"] = round(kept, 3)
        report["rows"].append(row)
        print(json.dumps(row), flush=True)

    if args.cand:
        cand = Index(args.cand)

        def run(Q, items, ref_side, text=False, keep_source=False):
            if ref_side:
                exclude = [None] * len(Q) if text or keep_source else [d for d, _ in items]
                return ref.top_docs(Q, exclude)
            if text:
                return cand.top_docs(Q, [None] * len(Q))
            # same query chunk, but the candidate index's own vector for it
            rows = [cand.row_of.get((ref.doc_number[d], ref.labels[r])) for d, r in items]
            Qc = np.stack([cand.emb[r] if r is not None else Q[i] for i, r in enumerate(rows)])
            exclude = [None] * len(Q) if keep_source else [cand.doc_id[ref.doc_number[d]] for d, _ in items]
            return cand.top_docs(Qc, exclude)

        row = evaluate(f"candidate {args.cand}", run)
        report["rows"].append(row)
        print(json.dumps(row), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
