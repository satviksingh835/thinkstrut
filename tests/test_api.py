import pytest
from fastapi.testclient import TestClient

from patsearch.api import create_app
from patsearch.embed import HashEmbedder


@pytest.fixture(scope="module")
def client(built):
    return TestClient(create_app(built[0], embedder=HashEmbedder()))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["docs"] == 8


def test_search_with_filters(client):
    r = client.get("/search", params={"q": "tire foam", "cpc": ["B60C"], "kw": ["tire"], "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["filters"]["cpc"] == ["B60C"] and len(body["hits"]) <= 2
    assert all(h["cpc"].startswith("B60C") for h in body["hits"])
    assert "timings_ms" in body


def test_strategy_param_validated(client):
    assert client.get("/search", params={"q": "tire", "strategy": "post"}).status_code == 200
    assert client.get("/search", params={"q": "tire", "strategy": "bogus"}).status_code == 422


def test_similar_resolves_alias(client):
    r = client.get("/similar/20240000009", params={"k": 3})
    assert r.status_code == 200 and r.json()["query"]["doc_number"] == "20240000003"


def test_errors_map_to_http_codes(client):
    assert client.get("/similar/999").status_code == 404
    assert client.get("/claim/20240000010/2").status_code == 404  # canceled claim
    assert client.get("/search", params={"q": "x", "cpc": ["??"]}).status_code == 400


def test_ui_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "patsearch" in r.text


def test_meta(client):
    m = client.get("/meta").json()
    assert m["docs"] == 8 and m["date_min"] == "2024-02-15" and m["date_max"] == "2024-02-22"
    assert {"B60B", "B60C", "B60D"} <= set(m["subclasses"])
    assert any(g["code"] == "B60C 11" for g in m["groups"])


def test_patent_detail(client):
    p = client.get("/patent/20240000009").json()  # alias resolves to the A1
    assert p["doc_number"] == "20240000003" and p["also_published_as"] == ["20240000009"]
    assert [c["label"] for c in p["claims"]] == ["claim 1", "claim 2"] and p["claims"][1]["depends_on"] == 1
    assert p["description"] and p["has_description"]
    assert client.get("/patent/123").status_code == 404


def test_claim_by_text_and_blank_filters_ignored(client):
    r = client.get("/claim", params={"text": "spokes connecting a hub and a tread band", "k": 2, "cpc": [""], "title": ""})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "claim" and body["filters"] == {} and body["hits"]
