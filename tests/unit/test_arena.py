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
