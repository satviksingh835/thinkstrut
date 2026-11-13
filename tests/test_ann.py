import numpy as np

from patsearch.ann import AnnIndex, build_ann
from patsearch.embed import HashEmbedder
from patsearch.search import SearchEngine


def test_ann_matches_exact_on_small_corpus(built):
    out_dir, _ = built
    build_ann(out_dir, m=8, ef_construction=40)
    eng = SearchEngine(out_dir, embedder=HashEmbedder())
    assert isinstance(eng.index, AnnIndex)
    ann = AnnIndex(eng.exact, eng.index.index, exact_max_rows=0)  # force the graph path even when filtered
    q = eng.embed_query("bicycle hub with ratchet and pawls")
    for doc_ids, exclude in [(None, None), (np.array([0, 1, 2, 3, 4]), None), (None, np.array([1]))]:
        a_ids, a_s, path = ann.top_docs(q, 3, doc_ids, None, exclude)
        e_ids, e_s, _ = eng.exact.top_docs(q, 3, doc_ids, None, exclude)
        assert path == "hnsw" and a_ids.tolist() == e_ids.tolist()
        np.testing.assert_allclose(a_s, e_s, atol=1e-6)
    (out_dir / "chunks.hnsw.faiss").unlink()  # keep the shared fixture exact for other tests
