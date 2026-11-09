"""Arena picture names to photos: matching needs no database, linking is checked on SQLite (`pytest --pg` runs it on Postgres)."""

import sys

import pytest
from sqlalchemy import select

from stratastills.db import hashes, mapping
from stratastills.db.models import ArenaPic, Photo

# The fixtures are POSIX paths ("/data/raw/..."); the code under test normalises with os.path, which on Windows turns them into "\\data\\raw" and nothing matches.
# Real Windows paths are exercised by the application, not here: rewrite the fixtures with tmp_path to run these on Windows too.
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-path fixtures; see the comment above")

ROOT = "/data/raw"
H = {n: f"{i:064x}" for i, n in enumerate("abcdefg")}


def rec(rel, key):
    return dict(path=f"{ROOT}/{rel}", sha256=H[key], size=1, mtime_ns=1)


RECORDS = [
    rec("2018/2018-05-28/DSC08525.ARW", "a"),
    rec("2016/2016-04-28/DSC08525.ARW", "b"),  # one stem in two years
    rec("2018/2018-06-16 Title/DSC08694.ARW", "c"),  # exact (a titled folder)
    rec("2016/2016-07-24/DSC09723.ARW", "d"),
    rec("2018/2018-09-29/DSC09723.ARW", "e"),  # stem within 3 days of one of them only
    rec("2019/2019-01-01/DSC00001.ARW", "f"),
    rec("2019/2019-01-02/DSC00001.ARW", "g"),
]  # two raws near one name: ambiguous
CATALOGUE = [
    dict(raw=f"{ROOT}/2018/2018-05-28/DSC08525.ARW", final="/p/2018/2018-05-26 - Trip/DSC08525-Edit.jpg")
]  # a suffixed export: the arena names it after the RAW's stem


def test_the_rules_in_order_and_nothing_is_guessed():
    idx = mapping.build_index(RECORDS, CATALOGUE, [ROOT])
    assert mapping.match("2018-06-16_DSC08694", idx) == ("exact", f"{ROOT}/2018/2018-06-16 Title/DSC08694.ARW")
    assert mapping.match("2018-05-26_DSC08525", idx) == (
        "catalogue",
        f"{ROOT}/2018/2018-05-28/DSC08525.ARW",
    )  # the export's date differs from the folder's
    assert mapping.match("2016-07-23_DSC09723", idx) == ("window", f"{ROOT}/2016/2016-07-24/DSC09723.ARW")
    assert mapping.match("2019-01-01_DSC00001", idx)[0] == "exact" and mapping.match("2019-01-03_DSC00001", idx) == (
        "ambiguous",
        None,
    )
    assert mapping.match("2020-01-01_DSC77777", idx) == ("none", None) and mapping.match("not a name", idx) == (
        "none",
        None,
    )


def test_plan_counts_without_a_database():
    p = mapping.plan(
        ["2018-06-16_DSC08694", "2018-05-26_DSC08525", "2019-01-03_DSC00001", "x"], RECORDS, CATALOGUE, [ROOT]
    )
    assert p["counts"] == dict(exact=1, catalogue=1, ambiguous=1, none=1, mapped=2, unmapped=2, names=4) and {
        u["name"]: u["why"] for u in p["unmapped"]
    } == {"2019-01-03_DSC00001": "ambiguous", "x": "none"}


def test_write_fills_empty_links_once_and_never_overwrites(db_session):
    hashes.write(
        db_session, hashes.plan([dict(path=r["path"], sha256=r["sha256"], size=1, mtime_ns=1) for r in RECORDS], [ROOT])
    )
    db_session.add_all([ArenaPic(name="2018-06-16_DSC08694"), ArenaPic(name="2018-05-26_DSC08525"), ArenaPic(name="x")])
    db_session.commit()
    p = mapping.plan(["2018-06-16_DSC08694", "2018-05-26_DSC08525", "x"], RECORDS, CATALOGUE, [ROOT])
    assert mapping.write(db_session, p) == dict(linked=2, already=0, missing_pic=0, conflicts=0)
    assert mapping.write(db_session, p) == dict(linked=0, already=2, missing_pic=0, conflicts=0)
    pid = {s: i for s, i in db_session.execute(select(Photo.sha256, Photo.id))}
    assert (
        db_session.get(ArenaPic, "2018-06-16_DSC08694").photo_id == pid[H["c"]]
        and db_session.get(ArenaPic, "x").photo_id is None
    )
    db_session.get(ArenaPic, "x").photo_id = pid[H["a"]]
    db_session.commit()
    p2 = dict(rows=[dict(name="x", sha256=H["b"], tier="exact", raw="")], unmapped=[], counts={})
    assert mapping.write(db_session, p2)["conflicts"] == 1 and db_session.get(ArenaPic, "x").photo_id == pid[H["a"]]


def test_write_refuses_when_the_hashes_are_not_loaded(db_session):
    db_session.add(ArenaPic(name="2018-06-16_DSC08694"))
    db_session.commit()
    with pytest.raises(mapping.MappingMismatch):
        mapping.write(db_session, mapping.plan(["2018-06-16_DSC08694"], RECORDS, CATALOGUE, [ROOT]))


def test_arena_name_follows_build_arena_and_reads_windows_paths():
    assert (
        mapping.arena_name("/p/2018/2018-05-26 - Trip/DSC08525-Edit.jpg", "/r/2018/2018-05-28/DSC08525.ARW")
        == "2018-05-26_DSC08525"
    )
    assert (
        mapping.arena_name(
            "D:\\Photos\\2018\\2018-05-26 - Trip\\DSC08525.jpg", "D:\\raw\\2018\\2018-05-28\\DSC08525.ARW"
        )
        == "2018-05-26_DSC08525"
    )
    assert (
        mapping.arena_name("/p/Misc/Selected/DSC1.jpg", "/r/x/DSC1.ARW") is None
        and mapping.arena_name(None, "/r/x.ARW") is None
    )
