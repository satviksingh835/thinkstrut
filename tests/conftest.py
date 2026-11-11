import json

import pytest

from patsearch.build import build_index
from patsearch.embed import HashEmbedder
from patsearch.search import SearchEngine


def _p(num, title, cpc, abstract, claims, desc, date="20240215", kind="A1"):
    return {
        "title": title,
        "doc_number": num,
        "filename": f"US{num}{kind}-{date}.XML",
        "abstract": abstract,
        "detailed_description": desc,
        "claims": claims,
        "bibtex": f"@patent{{{num}}}",
        "classification": cpc,
    }


HUB_ABSTRACT = "A bicycle hub with a ratchet ring and pawls that engage when pedalling."
HUB_CLAIMS = ["a hub shell; a ratchet ring; and pawls;", "2 . The bicycle hub according to claim 1 , wherein the pawls are spring loaded."]
HUB_DESC = ["The hub includes a freewheel.", "", "Pawls engage the ratchet ring."]

WEEK1 = [
    _p("20240000001", "TIRE", "B60C1103FI", "A pneumatic tire whose tread has zigzag grooves for wet grip.",
       ["a tread portion having circumferential grooves;", "2 . The tire according to claim 1 , wherein the grooves are zigzag."],
       ["The tread has four main grooves.", "", "The grooves improve wet grip and drainage."]),
    _p("20240000002", "TIRE", "B60C100FI", "A tire with sound absorbing foam attached inside the tire cavity to reduce noise.",
       ["1 . A tire comprising a sound absorbing foam in the cavity.", "2 . The tire of claim 1 , wherein the foam is polyurethane."],
       ["Cavity noise is reduced by the foam.", "The foam is glued to the inner liner."]),
    _p("20240000003", "BICYCLE HUB", "B60B2702FI", HUB_ABSTRACT, HUB_CLAIMS, HUB_DESC),
    _p("20240000004", "SPOKE", "B60B104FI", "A carbon fiber spoke with threaded connecting ends.",
       ["an axle body made of carbon fiber;", "2 . The spoke according to claim 1 , wherein the ends are threaded."],
       ["The spoke body is made of carbon fiber reinforced plastic."]),
]
WEEK2 = [
    _p("20240000005", "HEIGHT ADJUSTABLE HITCH", "B60D100FI", "A height adjustable hitch receiver for trailers.",
       ["1 . A hitch receiver comprising a vertical bar with holes."], [""], date="20240222"),
    _p("20240000006", "WHEEL COVER", "B60B706FI", "A floating wheel cover that stays upright while the wheel rotates.",
       ["1 . A wheel cover comprising a weighted disc."], ["The cover is weighted at the bottom."], date="20240222"),
    _p("20240000009", "BICYCLE HUB", "B60B2702FI", HUB_ABSTRACT, HUB_CLAIMS, HUB_DESC, date="20240711", kind="A9"),
    _p("20240000010", "NON-PNEUMATIC TIRE", "B60C714FI", "A non-pneumatic tire where spokes connect the hub to a tread band.",
       ["1 - 3 . (canceled)", "4 . A non-pneumatic tire comprising spokes connecting a hub and a tread band."],
       ["Spokes are made of rubber.", "The tread band is reinforced."], date="20240222"),
    _p("20240000011", "Wheel bearing hub; café design", "B60B2700FI", "A wheel bearing hub assembly for café racer motorcycles.",
       ["1 . A hub assembly comprising a bearing."], ["The bearing is sealed."], date="20240222"),
]


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    (d / "patents_ipa240215.json").write_text(json.dumps(WEEK1))
    (d / "patents_ipa240222.json").write_text(json.dumps(WEEK2))
    return d


@pytest.fixture(scope="session")
def built(corpus_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    out_dir, manifest = build_index(corpus_dir, out, embedder=HashEmbedder(), progress=False, log=lambda s: None)
    return out_dir, manifest


@pytest.fixture(scope="session")
def engine(built):
    return SearchEngine(built[0], embedder=HashEmbedder())
