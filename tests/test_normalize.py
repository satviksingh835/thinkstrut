import pytest

from patsearch.load import LoadStats, coerce_record, dedupe
from patsearch.normalize import clean_text, cpc_matches, norm_title, parse_cpc, parse_cpc_filter


@pytest.mark.parametrize(
    "raw, display, group, ambiguous",
    [
        ("B60C1103FI", "B60C 11/03", "B60C11", False),
        ("B60C100FI", "B60C 1/00", "B60C1", False),
        ("B60C714FI", "B60C 7/14", "B60C7", False),
        ("B60B2700FI", "B60B 27/00", "B60B27", False),
        ("B60C25138FI", "B60C 25/138", "B60C25", True),
        ("B60C 11/0306", "B60C 11/0306", "B60C11", False),  # explicit separator is parsed exactly
    ],
)
def test_parse_cpc(raw, display, group, ambiguous):
    c = parse_cpc(raw)
    assert (c.display, c.group, c.ambiguous) == (display, group, ambiguous)
    assert c.subclass == raw[:4]


def test_parse_cpc_garbage():
    assert parse_cpc("") is None and parse_cpc(None) is None and parse_cpc("hello") is None


def test_cpc_filter_semantics():
    c_11_03, c_1_00 = parse_cpc("B60C1103FI"), parse_cpc("B60C100FI")
    group11 = parse_cpc_filter("B60C 11")
    assert group11.kind == "group"
    assert cpc_matches(group11, c_11_03) and not cpc_matches(group11, c_1_00)  # naive prefix "B60C1" would match both
    full = parse_cpc_filter("B60C 11/03")
    assert cpc_matches(full, c_11_03) and not cpc_matches(full, c_1_00)
    assert cpc_matches(parse_cpc_filter("B60C1103"), c_11_03)  # compact input works too
    for p in ("B", "B6", "B60", "B60C", "b60c"):
        assert cpc_matches(parse_cpc_filter(p), c_11_03)
    assert not cpc_matches(parse_cpc_filter("B60B"), c_11_03)
    with pytest.raises(ValueError):
        parse_cpc_filter("not a code")


def test_clean_text():
    s = "The spoke according to claim 1 , wherein ( 18 ) p1_⁢0  x < y ."
    assert clean_text(s) == "The spoke according to claim 1, wherein (18) p1_0 x < y."


def test_norm_title():
    assert norm_title("HUB, IN PARTICULAR FOR BICYCLES") == norm_title("Hub in particular  for bicycles")


def _rec(**kw):
    base = {
        "title": "TIRE", "doc_number": "20240000001", "filename": "US20240000001A1-20240215.XML",
        "abstract": "A tire.", "detailed_description": ["p1", "", "p2"], "claims": ["a tread;"],
        "bibtex": "@patent{...}", "classification": "B60C1103FI",
    }
    base.update(kw)
    return base


def test_coerce_tolerates_missing_and_bad_fields():
    st = LoadStats()
    p = coerce_record(_rec(abstract=None, detailed_description=[""], classification=5), "patents_ipa240215.json", st)
    assert p.abstract == "" and p.description == [] and not p.has_description
    assert "null:abstract" in p.flags and "no_description" in p.flags and "coerced:classification" in p.flags
    assert p.pub_date == 20240215 and p.kind == "A1"
    q = coerce_record({k: v for k, v in _rec().items() if k != "doc_number"}, "x.json", st)
    assert q.doc_number == "20240000001" and "doc_number_from_filename" in q.flags
    assert coerce_record(_rec(doc_number="", filename=""), "x.json", st) is None
    assert coerce_record({"doc_number": "1"}, "x.json", st) is None  # no text at all
    assert coerce_record("not a dict", "x.json", st) is None
    assert st.dropped_no_doc_number == 1 and st.dropped_no_text == 1 and st.dropped_not_a_dict == 1


def test_coerce_drops_blank_paragraphs():
    p = coerce_record(_rec(), "x.json", LoadStats())
    assert p.description == ["p1", "p2"]


def test_dedupe_keeps_earliest_publication_and_records_alias():
    st = LoadStats()
    a9 = coerce_record(_rec(doc_number="20240000009", filename="US20240000009A9-20240711.XML"), "x.json", st)
    a1 = coerce_record(_rec(), "x.json", st)
    other = coerce_record(_rec(doc_number="20240000002", claims=["different"]), "x.json", st)
    kept, alias = dedupe([a9, a1, other])
    assert sorted(p.doc_number for p in kept) == ["20240000001", "20240000002"]
    assert alias == {"20240000009": "20240000001"}
    assert next(p for p in kept if p.doc_number == "20240000001").aliases == ["20240000009"]


def test_large_dataset_filename_without_date():
    p = coerce_record(_rec(filename="US20240000148A1.XML"), "patents_batch_0001.json", LoadStats())
    assert p.kind == "A1" and p.pub_date is None


def test_list_data_files_nested_large_layout(tmp_path):
    from patsearch.load import list_data_files

    for job in ("job_000", "job_001"):
        (tmp_path / job).mkdir()
        for i in (2, 1):
            (tmp_path / job / f"patents_batch_000{i}.json").write_text("[]")
    (tmp_path / "sample.json").write_text("{}")
    names = [f"{p.parent.name}/{p.name}" for p in list_data_files(tmp_path)]
    assert names == ["job_000/patents_batch_0001.json", "job_000/patents_batch_0002.json",
                     "job_001/patents_batch_0001.json", "job_001/patents_batch_0002.json"]


def test_dedupe_repeated_doc_number_keeps_first():
    st = LoadStats()
    first = coerce_record(_rec(), "a.json", st)
    again = coerce_record(_rec(abstract="A re-scraped copy with different text."), "b.json", st)
    kept, alias = dedupe([first, again], st)
    assert [p.abstract for p in kept] == ["A tire."] and alias == {} and st.duplicate_doc_numbers == 1
