import pytest

from patsearch.filters import DocPredicate, FilterSpec, candidate_doc_ids, compile_filter, load_doc_meta

SPECS = [
    FilterSpec(cpc=["B60B"]),
    FilterSpec(cpc=["B60C"]),
    FilterSpec(cpc=["B60C 11"]),
    FilterSpec(cpc=["B60C 1"]),  # must NOT match B60C 11/03
    FilterSpec(cpc=["B60C 7/14"]),
    FilterSpec(cpc=["B60B", "B60D"]),
    FilterSpec(cpc=["B6"]),
    FilterSpec(keywords=["tire"]),
    FilterSpec(keywords=["tire", "foam"]),
    FilterSpec(keywords=["tire", "hub"], kw_mode="any"),
    FilterSpec(keywords=["tread band"]),
    FilterSpec(keywords=["band tread"]),
    FilterSpec(keywords=["groove*"]),
    FilterSpec(keywords=["zigzag"], kw_field="title"),
    FilterSpec(keywords=["zigzag"], kw_field="abstract"),
    FilterSpec(keywords=["cafe"]),  # diacritic-insensitive: matches "café"
    FilterSpec(title="tire"),
    FilterSpec(title="Wheel Bearing Hub, Café Design"),
    FilterSpec(title_contains="hub"),
    FilterSpec(title_contains="pneumatic tire"),
    FilterSpec(date_from="2024-02-22"),
    FilterSpec(date_to="20240215"),
    FilterSpec(cpc=["B60B"], keywords=["bicycle"], date_from="2024-02-01"),
]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: str(s.to_dict()))
def test_sql_prefilter_matches_python_predicate(engine, spec):
    cf = compile_filter(spec)
    sql_ids = candidate_doc_ids(engine.store.con, cf).tolist()
    pred = DocPredicate(cf, load_doc_meta(engine.store.con))
    py_ids = [d for d in range(engine.n_docs) if pred(d)]
    assert sql_ids == py_ids


def _numbers(engine, ids):
    meta = engine.store.docs(ids, "doc_id, doc_number")
    return sorted(meta[i]["doc_number"] for i in ids)


def test_expected_matches(engine):
    def ids(**kw):
        return _numbers(engine, candidate_doc_ids(engine.store.con, compile_filter(FilterSpec(**kw))).tolist())

    assert ids(cpc=["B60C 11"]) == ["20240000001"]
    assert ids(cpc=["B60C 1"]) == ["20240000002"]
    assert ids(title="TIRE") == ["20240000001", "20240000002"]
    assert ids(keywords=["cafe"]) == ["20240000011"]
    assert ids(keywords=["tread band"]) == ["20240000010"]
    assert ids(keywords=["band tread"]) == []  # phrase order matters
    assert ids(keywords=["groove*"]) == ["20240000001"]


@pytest.mark.parametrize("bad", ['B60-B', '"', "*", 'a"b', "NEAR(", "AND", "OR NOT", "title:", "{title}", "^"])
def test_hostile_keyword_input_never_breaks_fts(engine, bad):
    cf = compile_filter(FilterSpec(keywords=[bad], title_contains=bad))
    candidate_doc_ids(engine.store.con, cf)  # must not raise


def test_bad_inputs_raise_clean_errors():
    with pytest.raises(ValueError):
        compile_filter(FilterSpec(cpc=["???"]))
    with pytest.raises(ValueError):
        compile_filter(FilterSpec(date_from="2024-13-45"))
    with pytest.raises(ValueError):
        compile_filter(FilterSpec(kw_mode="sometimes"))
