"""The nudity filter's file store and the choice of store (no database packages needed): the files as before, and the database only on its own opt-in."""

import json
import os

from stratastills import nsfw_store


def test_file_store_round_trips_scores_decisions_and_the_skip_log():
    st = nsfw_store.FileStore()
    assert st.load() == {} and st.allowed_among(["k"]) == set()
    cache = {"m|/a.jpg|1|2": [0.1, ""], "m|/b.jpg|3|4": [0.9, "X"]}
    for k, v in cache.items():
        st.put(k, v, cache)
    assert st.load() == cache
    st.set_allowed("m|/b.jpg|3|4", True)
    assert st.allowed_among(list(cache)) == {"m|/b.jpg|3|4"}
    st.set_allowed("m|/b.jpg|3|4", False)
    assert st.allowed_among(list(cache)) == set()
    st.record_skip({"path": "x", "reason": "nudity"})
    assert (
        json.loads(open(os.path.join(os.environ["STRATASTILLS_HOME"], "nsfw_skipped.jsonl")).readline())["path"] == "x"
    )


def test_parse_key_splits_a_path_that_contains_bars():
    assert nsfw_store.parse_key("nudenet-320n|/p/a|b.jpg|10|5") == ("nudenet-320n", "/p/a|b.jpg", 10, 5)


def test_a_database_url_alone_does_not_select_the_database_store(monkeypatch):
    nsfw_store._stores.clear()
    monkeypatch.setenv("STRATASTILLS_DATABASE_URL", "postgresql+psycopg:///x")
    monkeypatch.delenv("STRATASTILLS_DB_READS", raising=False)
    assert nsfw_store.get().kind == "files"  # the arena has its own switch; a read path needs its own entry in db_reads
    monkeypatch.delenv("STRATASTILLS_DATABASE_URL")
    monkeypatch.setenv("STRATASTILLS_DB_READS", "nsfw")
    assert nsfw_store.get().kind == "files"  # an entry without a URL does not either
    nsfw_store._stores.clear()


def test_db_reads_comes_from_the_environment_or_the_settings_file(monkeypatch):
    from stratastills import settings

    monkeypatch.delenv("STRATASTILLS_DB_READS", raising=False)
    assert settings.db_reads() == set()
    os.makedirs(os.environ["STRATASTILLS_HOME"], exist_ok=True)
    with open(os.path.join(os.environ["STRATASTILLS_HOME"], "settings.json"), "w") as f:
        json.dump({"db_reads": ["nsfw", " projects "]}, f)
    assert settings.db_reads() == {"nsfw", "projects"}
    monkeypatch.setenv("STRATASTILLS_DB_READS", "metadata, nsfw")
    assert settings.db_reads() == {"metadata", "nsfw"}  # the environment wins
    monkeypatch.setenv("STRATASTILLS_DB_READS", "")
    assert settings.db_reads() == set()
