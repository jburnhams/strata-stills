import random

from stratastills import arena


def votes(*triples):
    return [dict(pic="p", a=a, b=b, winner=w, t=i) for i, (a, b, w) in enumerate(triples)]


def test_elo_and_bradley_terry_rank_the_winner_first():
    vs = votes(*[("x", "y", "a")] * 8 + [("y", "x", "b")] * 4 + [("x", "y", "b")] * 2)
    e, bt = arena.elo(vs), arena.bradley_terry(vs)
    assert e["x"] > e["y"] and bt["x"] > bt["y"] and abs(bt["x"] + bt["y"] - 3000) < 1


def test_ties_move_the_ratings_together_and_retired_variants_are_skipped():
    e = arena.elo(votes(("x", "y", "a"), ("x", "y", "tie")))
    assert e["x"] > e["y"]
    assert (
        arena.elo(votes(("x", "y", "a")), retired={"y"}) == {}
        and arena.bradley_terry(votes(("x", "y", "a")), retired={"x"}) == {}
    )


def test_head_to_head_counts_by_sorted_pair():
    assert arena.head_to_head(votes(("y", "x", "a"), ("x", "y", "a"), ("x", "y", "tie"))) == {("x", "y"): [1, 1, 1]}


def test_bootstrap_gives_an_interval_around_each_rating():
    vs = votes(*[("x", "y", "a")] * 20 + [("x", "y", "b")] * 10)
    ci = arena.bootstrap(vs, n=30)
    assert ci["x"][0] <= arena.bradley_terry(vs)["x"] <= ci["x"][1] and arena.bootstrap([]) == {}


def test_pick_pair_spreads_over_pictures_skips_identical_pictures_and_hides_the_order():
    h = lambda s: dict(file=s, hash=s)
    pool = {"p1": {"lr": h("a"), "auto1": h("b"), "auto2": h("b")}, "p2": {"lr": h("c"), "auto1": h("d")}}
    rng = random.Random(1)
    seen = set()
    for _ in range(40):
        pic, a, b = arena.pick_pair(pool, [], rng)
        assert {a, b} != {"auto1", "auto2"}
        seen.add((pic, a))
    assert {p for p, _ in seen} == {"p1", "p2"} and {a for _, a in seen} >= {"lr", "auto1"}
    done = [dict(pic="p1", a="lr", b="auto1", winner="a", t=1)]
    assert arena.pick_pair(pool, done, random.Random(2))[0] == "p2"
    assert arena.pick_pair(pool, [], rng, want={"auto2"})[0] == "p1" and arena.pick_pair({}, [], rng) is None


def test_votes_round_trip_through_the_file(tmp_path):
    f = str(tmp_path / "a" / "votes.jsonl")
    arena.append_vote(f, dict(pic="p", a="x", b="y", winner="a", t=1))
    open(f, "a").write('{"half')
    assert arena.load_votes(f) == [dict(pic="p", a="x", b="y", winner="a", t=1)] and arena.load_votes(f + ".none") == []


def test_pick_pair_only_offers_pictures_of_the_named_set_and_load_set_reads_it(tmp_path):
    import json

    import pytest

    e = lambda h: dict(file="x", hash=h)
    pool = {p: {"auto1": e(p + "1"), "lightroom": e(p + "2")} for p in ("p1", "p2", "p3")}
    rng = random.Random(1)
    assert {arena.pick_pair(pool, [], rng, only={"p2"})[0] for _ in range(20)} == {"p2"} and arena.pick_pair(
        pool, [], rng, only={"nope"}
    ) is None
    (tmp_path / "sets").mkdir()
    (tmp_path / "sets" / "hard.json").write_text(json.dumps(["p1", "p3"]))
    (tmp_path / "sets" / "obj.json").write_text(json.dumps({"pics": ["p2"]}))
    assert (
        arena.load_set(str(tmp_path), "hard") == {"p1", "p3"}
        and arena.load_set(str(tmp_path), "obj") == {"p2"}
        and arena.load_set(str(tmp_path), "") is None
    )
    for bad in ("missing", "../x"):
        with pytest.raises(ValueError):
            arena.load_set(str(tmp_path), bad)


def _votes(*rows):
    return [dict(pic="p", a=a, b=b, winner=w, t=i) for i, (a, b, w) in enumerate(rows)]


def test_side_term_takes_a_left_preference_out_of_the_strengths():
    import random as R

    rng, vs = R.Random(3), []
    for _ in range(600):  # x is truly stronger (70 %), the left picture gets a boost on top
        a, b = ("x", "y") if rng.random() < 0.5 else ("y", "x")
        px = 0.7 if a == "x" else 0.3
        left = 1.6 * px / (1.6 * px + (1 - px))
        vs.append(dict(pic="p", a=a, b=b, winner="a" if rng.random() < left else "b", t=0))
    assert 1.3 < arena.side_advantage(vs) < 2.0
    assert arena.bradley_terry(vs, side=False)["x"] > arena.bradley_terry(vs, side=False)["y"]
    # always-left wins with no preference between the versions: side term explains it all, strengths stay level
    flat = _votes(*[("x", "y", "a"), ("y", "x", "a")] * 20)
    bt = arena.bradley_terry(flat)
    assert abs(bt["x"] - bt["y"]) < 5 and arena.side_advantage(flat) > 5
    assert arena.side_advantage([]) == 1.0


def test_priority_weights_clear_pairs_of_interesting_variants_and_faces_but_never_drop_a_pair():
    pool = {p: {v: {"file": f"{p}{v}", "hash": f"{p}{v}"} for v in ("auto4", "auto6", "camera")} for p in ("p1", "p2")}
    pr = dict(
        faces=["p2"],
        dist={"p1": {"auto4|auto6": 1.0, "auto4|camera": 12.0, "auto6|camera": 12.0}},
    )
    seen, pn = {}, {}
    assert arena.pair_weight(pr, "p1", "auto4", "auto6", seen, pn) < arena.pair_weight(
        pr, "p1", "auto4", "camera", seen, pn
    )
    assert arena.pair_weight(pr, "p2", "auto4", "auto6", seen, pn) > arena.pair_weight(
        pr, "p1", "auto4", "auto6", seen, pn
    )
    assert arena.pair_weight(pr, "p1", "auto4", "auto6", seen, pn) > 0
    rng, hits = random.Random(0), {}
    for _ in range(400):
        pic, a, b = arena.pick_pair(pool, [], rng, priority=pr)
        hits[pic] = hits.get(pic, 0) + 1
    assert hits["p2"] > hits["p1"]


def test_load_priorities_reads_the_file_or_says_none(tmp_path):
    assert arena.load_priorities(str(tmp_path)) is None
    (tmp_path / "priorities.json").write_text('{"interest": {"auto6": 2}}')
    assert arena.load_priorities(str(tmp_path)) == {"interest": {"auto6": 2}}
    (tmp_path / "priorities.json").write_text("[1]")
    assert arena.load_priorities(str(tmp_path)) is None


def test_current_votes_keep_votes_on_the_pictures_shown_now_or_judged_the_same():
    pool = {
        "p": {
            "x": dict(file="x", hash="x2", same=["x1"], changed=50),
            "y": dict(file="y", hash="y1"),
        }
    }
    vs = [
        dict(pic="p", a="x", b="y", winner="a", t=10, ha="x1", hb="y1"),  # an earlier render judged the same
        dict(pic="p", a="x", b="y", winner="a", t=60, ha="x0", hb="y1"),  # a render from before a real change
        dict(pic="p", a="y", b="x", winner="a", t=60, ha="y1", hb="x2"),  # the current pictures
        dict(pic="p", a="x", b="y", winner="b", t=10),  # no hashes, cast before the last real change
        dict(pic="p", a="x", b="y", winner="b", t=70),  # no hashes, cast after it
        dict(pic="gone", a="x", b="y", winner="b", t=1, ha="?", hb="?"),  # a picture no longer in the pool
    ]
    assert [vs.index(v) for v in arena.current_votes(vs, pool)] == [0, 2, 4, 5]
