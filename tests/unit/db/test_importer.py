"""The arena import: parsing is checked without a database, writing on SQLite (the `db_session` fixture)."""
import json, os

import pytest

pytest.importorskip('omarchy_db'); pytest.importorskip('pgvector')
from sqlalchemy import func, select

from stratastills.db import importer
from stratastills.db.models import ArenaEntry, ArenaFlag, ArenaFlagEvent, ArenaPic, ArenaVariant, ArenaVote, ReportFlag


def lines(path, rows): path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf8')


@pytest.fixture
def arena(tmp_path):
    a, f = tmp_path / 'arena', tmp_path / 'flags'; a.mkdir(); f.mkdir()
    lines(a / 'votes.jsonl', [dict(pic='p1', a='lightroom', b='auto1', winner='a', t=1.0, ms=100), dict(pic='p1', a='lightroom', b='auto1', winner='a', t=1.0, ms=100), dict(pic='p2', a='neutral', b='auto2', winner='tie', t=2.0, ms=5)])
    with open(a / 'votes.jsonl', 'a') as h: h.write('{"half": ')                                      # a half-written last line
    lines(a / 'alerts.jsonl', [dict(id='f1', pic='p1', variant='auto1', other='lightroom', side='right', token='t', note='', t=3.0, ms=9, votes_on_pic=1, votes_on_variant=1), dict(id='f1', status='investigating', note='smear', t=4.0), dict(id='f1', status='fixed', t=5.0), dict(id='nope', status='fixed', t=6.0)])
    (a / 'pool.json').write_text(json.dumps({'p1': {'lightroom': dict(file='img/p1/lightroom.jpg', hash='h1', key='k1'), 'auto1': dict(file='img/p1/auto1.jpg', hash='h2')}}), encoding='utf8')
    (f / 'auto.json').write_text(json.dumps({'p3': dict(flag=True, rating='medium', reasons=['too flat / dull'], note='', status='open', at='2026-11-01T14:29:21', first='2026-11-01T14:00:00')}), encoding='utf8')
    return a, f


def test_plan_counts_and_problems_without_a_database(arena):
    p = importer.plan(*arena)
    assert p['counts'] == dict(arena_pic=2, arena_variant=4, arena_entry=2, arena_vote=3, arena_flag=1, arena_flag_event=3, report_flag=1)
    assert any('unparsable' in x for x in p['problems']) and any('unknown flag' in x for x in p['problems'])
    assert 'p3' not in p['pics']                                                                      # a report's picture ids are not the arena's pool names
    assert len({v['legacy_key'] for v in p['votes']}) == 3                                           # two identical lines stay two votes
    assert p['flags'][0]['status'] == 'fixed' and p['flags'][0]['note'] == 'smear'


def test_write_then_write_again_adds_nothing(arena, db_session):
    p = importer.plan(*arena); added = importer.write(db_session, p)
    assert added['arena_vote'] == 3 and added['arena_flag_event'] == 3 and added['report_flag'] == 1
    assert importer.write(db_session, p) == {k: 0 for k in added}
    assert db_session.scalar(select(func.count()).select_from(ArenaVote)) == 3 and db_session.get(ArenaFlag, 'f1').status == 'fixed'
    assert db_session.get(ReportFlag, ('auto', 'p3')).reasons == ['too flat / dull']
    assert db_session.scalar(select(func.count()).select_from(ArenaEntry)) == 2 and db_session.get(ArenaVariant, 'auto1').retired is False


def test_retired_variants_are_marked(arena, db_session):
    a, f = arena; (a / 'variants.json').write_text(json.dumps({'retired': ['auto2']}), encoding='utf8')
    importer.write(db_session, importer.plan(a, f)); assert db_session.get(ArenaVariant, 'auto2').retired is True


def test_the_source_files_are_not_touched(arena, db_session):
    a, f = arena; before = {n: (a / n).read_bytes() for n in os.listdir(a)}
    importer.write(db_session, importer.plan(a, f)); assert {n: (a / n).read_bytes() for n in os.listdir(a)} == before
