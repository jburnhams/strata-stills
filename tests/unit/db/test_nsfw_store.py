"""The nudity filter's store: the SAME contract on the file store and on the database store (SQLite here; `--pg` on the real thing), and the filter itself running on the database."""
import json, os

import pytest

pytest.importorskip('omarchy_db'); pytest.importorskip('pgvector')
from photos import make_jpeg
from sqlalchemy import func, select

from stratastills import nsfw, nsfw_store
from stratastills.db import hashes
from stratastills.db.models import File, NsfwDecision, NsfwResult, Photo


@pytest.fixture(params=['files', 'db'])
def store(request, tmp_path, db_url):
    root = tmp_path / 'shoot'; root.mkdir()
    if request.param == 'files': return nsfw_store.FileStore(), str(root), lambda key: None
    st = nsfw_store.DbStore(db_url, roots=[str(root)])
    def link(key):                                                         # a decision needs a file that belongs to a photo in the database
        _, path, size, mt = nsfw_store.parse_key(key); r, rel = st._where(path)
        with st._session() as s:
            ph = Photo(sha256=os.urandom(32).hex()); s.add(ph); s.flush(); f = st._file(s, r, rel, size, mt, True); f.photo_id = ph.id; s.commit()
    return st, str(root), link


def key(root, name, size=10, mt=5): return f'nudenet-320n|{os.path.join(root, name)}|{size}|{mt}'


def test_contract_scores_round_trip_and_update(store):
    st, root, _ = store; assert st.load() == {}
    k1, k2, far = key(root, 'a.jpg'), key(root, 'sub/b.jpg', 20, 6), key('/elsewhere/c', 'c.jpg', 30, 7)                          # the last one is under no root
    cache = {}
    for k, v in ((k1, [0.1, '']), (k2, [0.9, 'BUTTOCKS_EXPOSED']), (far, [1.0, 'unreadable'])): cache[k] = v; st.put(k, v, cache)
    assert {k: list(v) for k, v in st.load().items()} == {k1: [0.1, ''], k2: [0.9, 'BUTTOCKS_EXPOSED'], far: [1.0, 'unreadable']}
    cache[k1] = [0.2, 'X']; st.put(k1, [0.2, 'X'], cache); assert list(st.load()[k1]) == [0.2, 'X'] and len(st.load()) == 3


def test_contract_decisions_are_per_file_version_and_can_be_taken_back(store):
    st, root, link = store; k, other = key(root, 'a.jpg'), key(root, 'a.jpg', 11, 6)                                              # the same path after it changed
    cache = {k: [0.9, 'X'], other: [0.9, 'X']}
    for kk in (k, other): st.put(kk, cache[kk], cache)
    link(k)
    assert st.allowed_among([k, other]) == set()
    st.set_allowed(k, True); assert st.allowed_among([k, other]) == {k}                                                           # the changed file has to be cleared again
    st.set_allowed(k, False); assert st.allowed_among([k, other]) == set()


def test_contract_skip_log_is_a_file(store):
    st, root, _ = store; st.record_skip({'path': 'x', 'score': 0.9, 'reason': 'nudity'})
    assert json.loads(open(os.path.join(os.environ['STRATASTILLS_HOME'], 'nsfw_skipped.jsonl')).readline())['path'] == 'x'


def test_the_database_store_also_keeps_the_files_fresh_and_counts_unconfigured_roots(tmp_path, db_url):
    root = tmp_path / 'shoot'; root.mkdir(); st = nsfw_store.DbStore(db_url, roots=[str(root)]); k = key(str(root), 'a.jpg'); st.put(k, [0.3, ''], {k: [0.3, '']})
    assert nsfw_store.FileStore().load() == {k: [0.3, '']}                                                                         # dual-write
    other = nsfw_store.DbStore(db_url, roots=[str(tmp_path / 'else')]); assert other.load() == {} and other.unresolved == 1       # a root that is not configured here cannot be turned back into a path


def test_the_filter_runs_on_the_database(tmp_path, db_url, monkeypatch, db_session_for=None):
    d = tmp_path / 'shoot'; d.mkdir(); make_jpeg(str(d / 'A.jpg'), w=200, h=150, seed=1)
    monkeypatch.setenv('STRATASTILLS_NSFW', '1'); monkeypatch.setenv('STRATASTILLS_DATABASE_URL', db_url); monkeypatch.setenv('STRATASTILLS_DB_READS', 'nsfw'); monkeypatch.setenv('STRATASTILLS_SOURCES', str(d)); nsfw_store._stores.clear(); nsfw.reset()
    calls = []; nsfw.set_detector(lambda bgr: calls.append(1) or [dict(score=0.9, **{'class': 'BUTTOCKS_EXPOSED'})])
    try:
        p = str(d / 'A.jpg'); hit = nsfw.flagged(p); assert hit and hit['cls'] == 'BUTTOCKS_EXPOSED' and len(calls) == 1
        nsfw.reset(); nsfw.set_detector(lambda bgr: pytest.fail('looked again: the score should come from the database'))               # a new process: the cache comes from schema photos
        assert nsfw.flagged(p)['score'] == 0.9 and [r['path'] for r in nsfw.review()] == [p]
        from omarchy_db import connect, session
        with session(connect(db_url)) as s: assert s.scalar(select(func.count()).select_from(NsfwResult)) == 1 and s.scalar(select(func.count()).select_from(File)) == 1
    finally: nsfw.set_detector(None); nsfw.reset(); nsfw_store._stores.clear()


def test_a_database_url_alone_does_not_switch_the_filter(tmp_path, db_url, monkeypatch):
    monkeypatch.setenv('STRATASTILLS_DATABASE_URL', db_url); monkeypatch.delenv('STRATASTILLS_DB_READS', raising=False); nsfw_store._stores.clear()
    assert nsfw_store.get().kind == 'files'                                                                       # the arena has its own switch; a read path needs its own entry in db_reads
    monkeypatch.setenv('STRATASTILLS_DB_READS', 'nsfw,projects'); assert nsfw_store.get().kind == 'database'
    monkeypatch.delenv('STRATASTILLS_DATABASE_URL'); assert nsfw_store.get().kind == 'files'
    nsfw_store._stores.clear()
