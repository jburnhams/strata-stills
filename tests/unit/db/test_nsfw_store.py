"""The nudity filter's database store: the SAME contract as the file store (SQLite here; `--pg` on the real thing), what is specific to the database, and the filter itself running on it."""

import json
import os
import sys

import pytest
from photos import make_jpeg
from sqlalchemy import func, select

from stratastills import nsfw, nsfw_store
from stratastills.db import hashes
from stratastills.db.models import File, NsfwDecision, NsfwResult, Photo
from stratastills.db.nsfw import DbStore


class Env:
    """A store plus a way to make a file version known to the database (a no-op for the file store)."""

    def __init__(self, st, root, db):
        self.st, self.root, self.db = st, root, db

    def link(self, key, photo=None):
        if not self.db:
            return None
        _, path, size, mt = nsfw_store.parse_key(key)
        r, rel = self.st._where(path)
        with self.st._session() as s:
            if photo is None:
                ph = Photo(sha256=os.urandom(32).hex())
                s.add(ph)
                s.flush()
                photo = ph.id
            f = self.st._file(s, r, rel, size, mt, True)
            f.photo_id = photo
            s.commit()
            return photo


@pytest.fixture(params=["files", "db"])
def env(request, tmp_path, db_url):
    root = tmp_path / "shoot"
    root.mkdir()
    if request.param == "files":
        return Env(nsfw_store.FileStore(), str(root), False)
    return Env(DbStore(db_url, roots=[str(root)]), str(root), True)


def key(root, name, size=10, mt=5):
    return f"nudenet-320n|{os.path.join(root, name)}|{size}|{mt}"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the test mixes POSIX and drive-letter keys by hand; make the keys with os.path.join to run it on Windows",
)
def test_contract_scores_round_trip_and_update(env):
    st, root = env.st, env.root
    assert st.load() == {}
    k1, k2, far = (
        key(root, "a.jpg"),
        key(root, "sub/b.jpg", 20, 6),
        key("/elsewhere/c", "c.jpg", 30, 7),
    )  # the last one is under no root
    for k in (k1, k2, far):
        env.link(k)
    cache = {}
    for k, v in ((k1, [0.1, ""]), (k2, [0.9, "BUTTOCKS_EXPOSED"]), (far, [1.0, "unreadable"])):
        cache[k] = v
        st.put(k, v, cache)
    assert {k: list(v) for k, v in st.load().items()} == {
        k1: [0.1, ""],
        k2: [0.9, "BUTTOCKS_EXPOSED"],
        far: [1.0, "unreadable"],
    }
    cache[k1] = [0.2, "X"]
    st.put(k1, [0.2, "X"], cache)
    assert list(st.load()[k1]) == [0.2, "X"] and len(st.load()) == 3


def test_contract_decisions_are_per_file_version_and_can_be_taken_back(env):
    st, root = env.st, env.root
    k, other = key(root, "a.jpg"), key(root, "a.jpg", 11, 6)  # the same path after it changed
    p = env.link(k)
    env.link(other, photo=p)  # both versions belong to ONE photo
    assert st.allowed_among([k, other]) == set()
    st.set_allowed(k, True)
    assert st.allowed_among([k, other]) == {k}  # allowing one version does not allow the other
    st.set_allowed(other, False)
    assert st.allowed_among([k, other]) == {k}  # taking back a version that was not cleared changes nothing
    st.set_allowed(k, False)
    assert st.allowed_among([k, other]) == set()


def test_contract_skip_log_is_a_file(env):
    env.st.record_skip({"path": "x", "score": 0.9, "reason": "nudity"})
    assert (
        json.loads(open(os.path.join(os.environ["STRATASTILLS_HOME"], "nsfw_skipped.jsonl")).readline())["path"] == "x"
    )


def test_one_decision_per_photo_the_latest_version_wins(tmp_path, db_url):
    root = tmp_path / "s"
    root.mkdir()
    st = DbStore(db_url, roots=[str(root)])
    e = Env(st, str(root), True)
    k, other = key(str(root), "a.jpg"), key(str(root), "a.jpg", 11, 6)
    p = e.link(k)
    e.link(other, photo=p)
    st.set_allowed(k, True)
    st.set_allowed(other, True)
    assert st.allowed_among([k, other]) == {other}  # clearing the new version replaces the decision on the old one
    with st._session() as s:
        assert s.scalar(select(func.count()).select_from(NsfwDecision)) == 1


def test_scores_go_to_the_database_only_for_files_it_knows_and_nothing_is_lost(tmp_path, db_url):
    root = tmp_path / "s"
    root.mkdir()
    st = DbStore(db_url, roots=[str(root)])
    e = Env(st, str(root), True)
    known, unknown = key(str(root), "k.jpg"), key(str(root), "u.ARW", 7, 7)
    e.link(known)
    nsfw_store.FileStore().put(
        "old|/gone/x.jpg|1|1", [0.5, "F"], {"old|/gone/x.jpg|1|1": [0.5, "F"]}
    )  # an entry only the file has
    cache = st.load()
    assert "old|/gone/x.jpg|1|1" in cache  # the filter's cache starts from the files plus the database
    for k in (known, unknown):
        cache[k] = [0.3, ""]
        st.put(k, [0.3, ""], cache)
    with st._session() as s:
        assert (
            s.scalar(select(func.count()).select_from(NsfwResult)) == 1
            and s.scalar(select(func.count()).select_from(File)) == 1
        )  # no placeholder file row for the unknown raw
    assert set(nsfw_store.FileStore().load()) == {
        "old|/gone/x.jpg|1|1",
        known,
        unknown,
    }  # dual-write MERGED into the file cache: the file-only entry survived
    assert set(st.load_db()) == {known} and set(st.load()) == {"old|/gone/x.jpg|1|1", known, unknown}


def test_the_hash_importer_still_works_on_a_file_that_was_scored_first(tmp_path, db_url, db_session_url=None):
    from omarchy_db import connect, session

    root = tmp_path / "s"
    root.mkdir()
    st = DbStore(db_url, roots=[str(root)])
    k = key(str(root), "u.ARW", 7, 7)
    st.put(k, [0.3, ""], {k: [0.3, ""]})
    with session(connect(db_url)) as s:
        p = hashes.plan([dict(path=str(root / "u.ARW"), sha256="a" * 64, size=7, mtime_ns=7)], [str(root)])
        assert hashes.write(s, p)["file"] == 1  # no HashMismatch from a placeholder


def test_the_database_store_counts_unconfigured_roots(tmp_path, db_url):
    root = tmp_path / "shoot"
    root.mkdir()
    st = DbStore(db_url, roots=[str(root)])
    e = Env(st, str(root), True)
    k = key(str(root), "a.jpg")
    e.link(k)
    st.put(k, [0.3, ""], {k: [0.3, ""]})
    other = DbStore(db_url, roots=[str(tmp_path / "else")])
    assert (
        other.load_db() == {} and other.unresolved == 1
    )  # a root that is not configured here cannot be turned back into a path


def test_the_filter_runs_on_the_database(tmp_path, db_url, monkeypatch):
    d = tmp_path / "shoot"
    d.mkdir()
    make_jpeg(str(d / "A.jpg"), w=200, h=150, seed=1)
    monkeypatch.setenv("STRATASTILLS_NSFW", "1")
    monkeypatch.setenv("STRATASTILLS_DATABASE_URL", db_url)
    monkeypatch.setenv("STRATASTILLS_DB_READS", "nsfw")
    monkeypatch.setenv("STRATASTILLS_SOURCES", str(d))
    nsfw_store._stores.clear()
    nsfw.reset()
    calls = []
    nsfw.set_detector(lambda bgr: calls.append(1) or [dict(score=0.9, **{"class": "BUTTOCKS_EXPOSED"})])
    try:
        p = str(d / "A.jpg")
        hit = nsfw.flagged(p)
        assert hit and hit["cls"] == "BUTTOCKS_EXPOSED" and len(calls) == 1
        nsfw.reset()
        nsfw.set_detector(
            lambda bgr: pytest.fail("looked again: the score should come from the cache")
        )  # a new process: the cache is loaded again (files plus database)
        assert nsfw.flagged(p)["score"] == 0.9 and [r["path"] for r in nsfw.review()] == [p]
    finally:
        nsfw.set_detector(None)
        nsfw.reset()
        nsfw_store._stores.clear()
