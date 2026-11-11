import json
from pathlib import Path

import numpy as np
import pytest

from patsearch.config import DEFAULT_DATA_DIR
from patsearch.filters import FilterSpec
from patsearch.vindex import ExactIndex


def _nums(resp):
    return [h.doc_number for h in resp.hits]


def test_manifest_counts(built):
    _, m = built
    c = m["counts"]
    assert c["docs"] == 8  # 9 records, A9 duplicate collapsed
    assert c["alias_map"] == {"20240000009": "20240000003"}
    assert c["docs_without_description"] == 1
    assert c["chunks_truncated"] == 0
    assert c["claims_canceled"] == 3


def test_doc_ids_are_ordered_by_cpc_subclass(engine):
    rows = engine.store.con.execute("SELECT cpc_subclass FROM docs ORDER BY doc_id").fetchall()
    subclasses = [r[0] for r in rows]
    assert subclasses == sorted(subclasses)


def test_text_search_finds_relevant_patent(engine):
    resp = engine.search("sound absorbing foam in the tire cavity", k=3)
    assert resp.hits[0].doc_number == "20240000002"
    assert resp.hits[0].evidence  # explains why it matched


QUERIES = ["tire tread grooves", "bicycle hub pawls", "carbon fiber spoke", "wheel cover", "hitch receiver"]
FILTERS = [
    FilterSpec(),
    FilterSpec(cpc=["B60B"]),
    FilterSpec(cpc=["B60C"], keywords=["tire"]),
    FilterSpec(keywords=["hub"], kw_mode="any"),
    FilterSpec(title="tire"),
    FilterSpec(cpc=["B60D"]),
]


@pytest.mark.parametrize("spec", FILTERS, ids=lambda s: str(s.to_dict()))
@pytest.mark.parametrize("query", QUERIES)
def test_prefilter_equals_naive_postfilter(engine, query, spec):
    pre = engine.search(query, spec, k=4, strategy="pre")
    post = engine.search(query, spec, k=4, strategy="post")
    assert _nums(pre) == _nums(post)
    np.testing.assert_allclose([h.score for h in pre.hits], [h.score for h in post.hits], atol=1e-5)


def test_post_topn_equals_pre_when_n_covers_corpus(engine):
    # POST_TOPN (100) >= corpus size here, so filtering the unfiltered top-N loses nothing
    spec = FilterSpec(cpc=["B60B"])
    pre = engine.search("tire", spec, k=3, strategy="pre")
    topn = engine.search("tire", spec, k=3, strategy="post-topN")
    assert _nums(topn) == _nums(pre)


@pytest.mark.parametrize("path", ["full", "slices", "gather"])
def test_scoring_paths_agree(engine, path):
    idx: ExactIndex = engine.index
    q = engine.embed_query("tire with grooves and hub")
    for doc_ids in (np.array([0, 1, 2]), np.array([1, 3, 5, 7]), np.arange(idx.n_docs), np.array([], dtype=np.int64)):
        ref_ids, ref_best, _ = idx.doc_scores(q, doc_ids, None, "full")
        ids, best, used = idx.doc_scores(q, doc_ids, None, path)
        assert used == path
        np.testing.assert_array_equal(ids, ref_ids)
        np.testing.assert_allclose(best, ref_best, atol=1e-6)


def test_claim_mapping_own_patent_ranks_first_when_not_excluded(engine):
    resp = engine.claim("20240000002:1", exclude_source=False, k=3)
    assert resp.hits[0].doc_number == "20240000002"
    resp = engine.claim("20240000002:1", k=3)
    assert "20240000002" not in _nums(resp)


def test_claim_inside_ambiguous_block_warns(engine):
    # 20240000010: claims 1-3 canceled, claim 4 exact
    with pytest.raises(KeyError):
        engine.claim("20240000010:2")
    resp = engine.claim("20240000010:4")
    assert resp.query["label"] == "claim 4" and not resp.warnings


def test_similar_excludes_self_and_resolves_alias(engine):
    resp = engine.similar("20240000009", k=10)  # A9 alias of 20240000003
    assert resp.query["doc_number"] == "20240000003"
    assert "20240000003" not in _nums(resp) and "20240000009" not in _nums(resp)
    assert len(resp.hits) == engine.n_docs - 1


def test_scope_description_skips_patents_without_description(engine):
    resp = engine.search("height adjustable hitch receiver", k=10, scope="description")
    assert "20240000005" not in _nums(resp)
    resp = engine.search("height adjustable hitch receiver", k=1, scope="abstract,claims")
    assert _nums(resp) == ["20240000005"]


def test_browse_by_filter_only(engine):
    resp = engine.search(None, FilterSpec(title="TIRE"), k=10)
    assert resp.mode == "browse" and sorted(_nums(resp)) == ["20240000001", "20240000002"]
    with pytest.raises(ValueError):
        engine.search(None, FilterSpec())


def test_passage_granularity_and_json(engine):
    resp = engine.search("pawls ratchet", k=4, granularity="passage")
    assert [h.rank for h in resp.hits] == [1, 2, 3, 4]
    assert all(len(h.evidence) == 1 for h in resp.hits)
    json.dumps(resp.to_dict())  # serialisable


def test_also_published_as(engine):
    resp = engine.search("bicycle hub ratchet", FilterSpec(cpc=["B60B 27/02"]), k=1)
    assert resp.hits[0].also_published_as == ["20240000009"]


# ------------------------------------------------------------------------------------------------
# Integration check on the real dataset (skipped when data/ is not present)
# ------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not Path(DEFAULT_DATA_DIR).exists(), reason="dataset not present")
def test_real_dataset_profile():
    from patsearch.claims import segment_claims
    from patsearch.load import LoadStats, dedupe, iter_patents, list_data_files

    st = LoadStats()
    patents = list(iter_patents(list_data_files(DEFAULT_DATA_DIR), st))
    assert st.records == 640 and st.loaded == 640
    kept, alias = dedupe(patents)
    assert len(kept) == 636 and len(alias) == 4
    assert all(a9.startswith("202402274") for a9 in alias)  # the A9 republications
    assert sum(not p.has_description for p in kept) == 116
    units = [u for p in kept for u in segment_claims(p.claims_raw).units]
    assert sum(u.exact for u in units) == 8192
