import sqlite3

import numpy as np

from patsearch.embed import HashEmbedder
from patsearch.fastbuild import fast_build
from patsearch.filters import FilterSpec
from patsearch.search import SearchEngine


def _rows(path, sql):
    con = sqlite3.connect(path / "patents.sqlite")
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def test_fast_build_matches_baseline_build(built, corpus_dir, tmp_path):
    base_dir, base_manifest = built
    fast_dir, m = fast_build(corpus_dir, tmp_path, model_id="hash-256", embedder=HashEmbedder(), workers=2,
                             log=lambda s: None)
    assert m["counts"]["docs"] == base_manifest["counts"]["docs"]
    assert m["counts"]["load"]["records"] == base_manifest["counts"]["load"]["records"]
    q = "SELECT doc_id, doc_number, cpc_subclass, chunk_lo, chunk_hi FROM docs ORDER BY doc_id"
    assert _rows(fast_dir, q) == _rows(base_dir, q)  # same doc_id order (CPC subclass first)
    q = "SELECT chunk_id, doc_id, kind, label, text FROM chunks ORDER BY chunk_id"
    assert _rows(fast_dir, q) == _rows(base_dir, q)
    assert _rows(fast_dir, "SELECT alias, doc_id FROM aliases") == _rows(base_dir, "SELECT alias, doc_id FROM aliases")
    np.testing.assert_allclose(np.load(fast_dir / "chunks.f32.npy"), np.load(base_dir / "chunks.f32.npy"), atol=1e-3)
    a = SearchEngine(base_dir, embedder=HashEmbedder()).search("bicycle hub pawls", FilterSpec(cpc=["B60B"]), k=3)
    b = SearchEngine(fast_dir, embedder=HashEmbedder()).search("bicycle hub pawls", FilterSpec(cpc=["B60B"]), k=3)
    assert [h.doc_number for h in a.hits] == [h.doc_number for h in b.hits]


def test_fast_build_max_desc_chunks(corpus_dir, tmp_path):
    _, m = fast_build(corpus_dir, tmp_path, model_id="hash-256", embedder=HashEmbedder(), workers=2,
                      max_desc_chunks=0, log=lambda s: None)
    assert "description" not in m["counts"]["chunks_by_kind"]
    assert m["counts"]["description_chunks_dropped"] > 0
