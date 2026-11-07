"""The arena on the database: the store contract for both stores, and the server on schema photos (SQLite here; `--pg` on the real thing)."""
import json

import pytest

pytest.importorskip('omarchy_db'); pytest.importorskip('pgvector')
from fastapi.testclient import TestClient
from omarchy_db import connect, session as _session
from sqlalchemy import func, select

from stratastills import arena_store
from stratastills.db import importer
from stratastills.db.models import ArenaFlagEvent, ArenaVote
from stratastills.server.app import create_app


@pytest.fixture
def db_session(db_url):
    """A session on the SAME fresh database the stores connect to (the plugin's db_session is another database)."""
    with _session(connect(db_url)) as s: yield s


@pytest.fixture
def arena(tmp_path, monkeypatch):
    a = tmp_path / 'arena'; (a / 'img').mkdir(parents=True); monkeypatch.setenv('STRATASTILLS_ARENA', str(a))
    for n in ('lr', 'auto1'): (a / 'img' / f'{n}.jpg').write_bytes(b'jpg-' + n.encode())
    (a / 'pool.json').write_text('{"p1": {"lr": {"file": "img/lr.jpg", "hash": "a"}, "auto1": {"file": "img/auto1.jpg", "hash": "b", "key": "k1"}}}')
    return a


@pytest.fixture(params=['files', 'database'])
def store(request, arena, db_url, db_session, monkeypatch):
    if request.param == 'database':
        importer.write(db_session, importer.plan(str(arena))); monkeypatch.setenv('STRATASTILLS_DATABASE_URL', db_url); arena_store._stores.clear()
    s = arena_store.get(); assert s.kind == request.param; return s


def test_the_store_contract(store):
    assert store.pool()['p1']['auto1'] == {'file': 'img/auto1.jpg', 'hash': 'b', 'key': 'k1'} and store.retired() == set() and store.votes() == [] and store.undo() is None
    store.add_vote(dict(pic='p1', a='lr', b='auto1', winner='a', t=1.5, ms=7)); store.add_vote(dict(pic='p1', a='auto1', b='lr', winner='tie', t=2.5, ms=8))
    assert [v['winner'] for v in store.votes()] == ['a', 'tie'] and store.votes()[0]['ms'] == 7
    gone, left = store.undo(); assert gone['winner'] == 'tie' and left == 1 and len(store.votes()) == 1
    store.add_alert_event(dict(id='f1', token='t', pic='p1', variant='auto1', other='lr', side='left', note='x', t=3.0, ms=1)); store.add_alert_event(dict(id='f1', status='investigating', note='smear', t=4.0)); store.add_alert_event(dict(id='f1', resolved=True, t=5.0))
    from stratastills import arena as AR
    a = AR.alerts(store.alert_events())[0]; assert (a['id'], a['status'], a['note'], a['resolved']) == ('f1', 'fixed', 'smear', True)


def test_undo_in_the_database_marks_the_vote_and_keeps_the_row(arena, db_url, db_session, monkeypatch):
    monkeypatch.setenv('STRATASTILLS_DATABASE_URL', db_url); arena_store._stores.clear(); importer.write(db_session, importer.plan(str(arena)))
    s = arena_store.get(); s.add_vote(dict(pic='p1', a='lr', b='auto1', winner='a', t=1.0)); s.undo(); db_session.expire_all()
    assert db_session.scalar(select(func.count()).select_from(ArenaVote)) == 1 and db_session.scalar(select(ArenaVote.undone_at)) is not None and s.votes() == []


def test_a_rebuilt_picture_supersedes_the_old_entry(arena, db_url, db_session, monkeypatch):
    importer.write(db_session, importer.plan(str(arena)))
    (arena / 'pool.json').write_text('{"p1": {"lr": {"file": "img/lr.jpg", "hash": "a"}, "auto1": {"file": "img/auto1.jpg", "hash": "NEW", "key": "k2"}}}')
    added = importer.write(db_session, importer.plan(str(arena))); assert added['arena_entry'] == 1
    monkeypatch.setenv('STRATASTILLS_DATABASE_URL', db_url); arena_store._stores.clear(); assert arena_store.get().pool()['p1']['auto1']['hash'] == 'NEW'


def test_the_server_runs_the_arena_on_the_database(arena, db_url, db_session, monkeypatch, tmp_path):
    monkeypatch.setenv('STRATASTILLS_PROJECTS', str(tmp_path / 'projects')); monkeypatch.setenv('STRATASTILLS_HOME', str(tmp_path / 'home'))
    importer.write(db_session, importer.plan(str(arena))); monkeypatch.setenv('STRATASTILLS_DATABASE_URL', db_url); arena_store._stores.clear()
    c = TestClient(create_app()); p = c.get('/api/arena/next').json()
    assert {c.get('/arena/' + p['left']).content, c.get('/arena/' + p['right']).content} == {b'jpg-lr', b'jpg-auto1'}
    assert c.post('/api/arena/vote', json=dict(token=p['token'], choice='left', ms=900)).json()['votes'] == 1
    q = c.get('/api/arena/next').json(); al = c.post('/api/arena/alert', json=dict(token=q['token'], side='left', note=' banding ', ms=5)).json()
    assert c.post('/api/arena/alert', json=dict(token=q['token'], side='left')).json()['id'] == al['id'] and c.get('/api/arena/next').status_code == 404
    c.post('/api/alerts/status', json=dict(id=al['id'], status='fixed', note='ok')); got = c.get('/api/alerts').json()['alerts'][0]; assert got['status'] == 'fixed' and got['note'] == 'ok' and got['resolved']
    assert c.get('/api/arena/ratings').json()['votes'] == 1 and c.post('/api/arena/undo').json()['votes'] == 0 and c.post('/api/arena/undo').status_code == 404
    with __import__('omarchy_db').session(arena_store.get().engine) as s: assert s.scalar(select(func.count()).select_from(ArenaFlagEvent)) == 2       # the flag and its status change are history rows
    assert not (arena / 'votes.jsonl').exists() and not (arena / 'alerts.jsonl').exists()                                                         # nothing is written to the files any more
