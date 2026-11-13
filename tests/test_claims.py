from patsearch.claims import segment_claims

# Fragment patterns copied (shortened) from real records.
SPOKE = [  # 20240051333: claim 1 headless, the rest intact
    "an axle body, having a middle segment and two connecting segments;",
    "2 . The spoke according to claim 1 , wherein the two friction enhancing structures each comprise ...",
    "3 . The spoke according to claim 2 , wherein the convex portions ...",
]
CANCELED_THEN_NESTED = [  # 20240051338: canceled range, nested claims lost their numbers
    "1 - 5 . (canceled)",
    "a tread portion extending in a tire circumferential direction;",
    "7 . The pneumatic tire according to claim 6 , wherein a cis-1,4 bond content ...",
    "8 . The pneumatic tire according to claim 6 , wherein a storage modulus ...",
    "the undertread layer contains a rubber composition ...",
    "10 . The pneumatic tire according to claim 6 , wherein a ratio h/G ...",
]
OPEN_TAIL = [  # 20240300267: canceled range then only unnumbered fragments
    "1 - 12 . (canceled)",
    "a tread portion extending in a tire circumferential direction;",
    "a width Wr",
    "an inclination angle θ",
]


def test_headless_first_claim_and_numbered_claims():
    cs = segment_claims(SPOKE)
    assert [u.label for u in cs.units] == ["claim 1", "claim 2", "claim 3"]
    assert cs.units[0].text.startswith("an axle body")
    assert cs.units[0].is_independent  # claim 1 is always independent
    assert cs.units[1].depends_on == 1 and not cs.units[1].is_independent
    assert cs.units[2].depends_on == 2
    assert "claim 1, wherein" in cs.units[1].text  # " ," spacing repaired
    assert cs.fully_exact


def test_canceled_range_and_single_gaps_become_exact():
    cs = segment_claims(CANCELED_THEN_NESTED)
    assert cs.canceled == [(1, 5)] and cs.n_canceled == 5
    labels = [u.label for u in cs.units]
    assert labels == ["claim 6", "claim 7", "claim 8", "claim 9", "claim 10"]
    claim9 = cs.find(9)
    assert claim9.text.startswith("the undertread layer")
    assert claim9.headless and not claim9.is_independent  # independence unknown -> not claimed
    assert cs.find(6).is_independent is False  # headless claim 6 (> 1): unknown
    assert cs.find(3) is None  # canceled


def test_open_tail_block():
    cs = segment_claims(OPEN_TAIL)
    assert len(cs.units) == 1
    u = cs.units[0]
    assert (u.lo, u.hi, u.label) == (13, None, "claims 13+")
    assert "a width Wr" in u.text and not u.exact
    assert cs.find(20) is u


def test_multi_claim_gap_is_ambiguous_block():
    cs = segment_claims(["1 . A wheel comprising a hub.", "frag a;", "frag b;", "5 . The wheel of claim 1 , wherein x."])
    assert [u.label for u in cs.units] == ["claim 1", "claims 2–4", "claim 5"]
    assert cs.units[1].text == "frag a; frag b;"


def test_fragments_without_gap_continue_previous_claim():
    cs = segment_claims(["1 . A tire comprising:", "a tread;", "2 . The tire of claim 1 ."])
    # "a tread;" has no numbering gap to fill (next number is 2) -> continuation of claim 1
    assert [u.label for u in cs.units] == ["claim 1", "claim 2"]
    assert cs.units[0].text == "A tire comprising: a tread;"


def test_numbered_variants_and_noise():
    cs = segment_claims(["1 .- A hub.", "", None, "2 - 3 . (cancelled)", "4) The hub of claim 1."])
    assert [u.label for u in cs.units] == ["claim 1", "claim 4"]
    assert cs.canceled == [(2, 3)]


def test_text_before_first_numbered_claim_joins_it():
    # regression (large dataset): leading unnumbered text used to become a second "claim 1"
    cs = segment_claims(["a frame;", "1 . A bicycle comprising the frame.", "2 . The bicycle of claim 1 ."])
    assert [u.label for u in cs.units] == ["claim 1", "claim 2"]
    assert cs.units[0].text == "a frame; A bicycle comprising the frame."
    cs = segment_claims(["a hub;", "1 - 3 . (canceled)", "4 . A wheel."])
    assert [u.label for u in cs.units] == ["claim 4"] and cs.units[0].text.startswith("a hub;")
    assert len({u.lo for u in cs.units}) == len(cs.units)
