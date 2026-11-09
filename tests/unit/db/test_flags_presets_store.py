"""Report flags and pipeline presets: the SAME contract on the file stores and the database stores (SQLite here; `--pg` on the real thing), and the server and the pipeline running on the database."""
import json, os

import pytest

pytest.importorskip('omarchy_db'); pytest.importorskip('pgvector')
from sqlalchemy import func, select

from stratastills import flags_store, pipeline as PL, presets_store
from stratastills.db.flags import DbStore as FlagDb
from stratastills.db.models import PipelinePreset, ReportFlag
from stratastills.db.presets import DbStore as PresetDb


@pytest.fixture(autouse=True)
def own_flags_folder(tmp_path, monkeypatch): monkeypatch.setenv('STRATASTILLS_REPORT_FLAGS', str(tmp_path / 'report-flags'))                 # not out/flags of the repository


@pytest.fixture(params=['files', 'db'])
def stores(request, db_url):
    if request.param == 'files': return flags_store.FileStore(), presets_store.FileStore(), request.param
    return FlagDb(db_url), PresetDb(db_url), request.param


E1 = dict(flag=True, rating='bad', reasons=['too dark', 'too flat / dull'], note='corner smear', status='open', at='2026-11-01T14:30:21', first='2026-11-01T14:30:18')


def test_contract_flags(stores):
    fl, _, kind = stores; assert fl.read('auto') == {}
    assert fl.write('auto', 'p1', E1) == E1 and fl.read('auto') == {'p1': E1} and fl.read('neutral') == {}
    e2 = dict(E1, rating='good', status='fixed', at='2026-11-02T10:00:00'); fl.write('auto', 'p1', e2); assert fl.read('auto')['p1'] == e2                          # a change keeps `first`
    fl.write('auto', 'p2', dict(flag=True, rating='', reasons=[], note='x', status='open', at='2026-11-02T10:00:00', first='2026-11-02T10:00:00')); assert sorted(fl.read('auto')) == ['p1', 'p2']
    assert fl.write('auto', 'p1', None) == {} and sorted(fl.read('auto')) == ['p2']                                                                                        # taken off
    fl.write('auto', 'p1', E1); assert fl.read('auto')['p1'] == E1                                                                                                       # and flagged again
    if kind == 'db':
        with fl._session() as s: assert s.scalar(select(func.count()).select_from(ReportFlag)) == 2 and s.get(ReportFlag, ('auto', 'p1')).removed_at is None
        fl.write('auto', 'p1', None)
        with fl._session() as s: assert s.get(ReportFlag, ('auto', 'p1')).removed_at is not None and s.scalar(select(func.count()).select_from(ReportFlag)) == 2                       # marked, not deleted


def test_contract_presets(stores):
    _, pr, kind = stores; assert pr.list() == [] and pr.delete('nope') is False
    pr.save('mine', 'My preset', [{'stage': 'develop'}]); pr.save('other', None, [{'stage': 'resize'}])
    assert [(p['name'], p['label']) for p in pr.list()] == [('mine', 'My preset'), ('other', 'other')] and pr.list()[0]['stages'] == [{'stage': 'develop'}]
    pr.save('mine', 'Renamed', [{'stage': 'sharpen'}]); assert pr.list()[0]['label'] == 'Renamed' and pr.list()[0]['stages'] == [{'stage': 'sharpen'}]
    assert pr.delete('mine') is True and [p['name'] for p in pr.list()] == ['other'] and pr.delete('mine') is False
    if kind == 'db':
        with pr._session() as s: assert [r.name for r in s.scalars(select(PipelinePreset))] == ['other']


def test_the_database_stores_also_keep_the_files_and_show_a_file_only_entry(tmp_path, db_url):
    fl, pr = FlagDb(db_url), PresetDb(db_url); fs, ps = flags_store.FileStore(), presets_store.FileStore()
    fs.write('auto', 'only_file', dict(E1)); ps.save('only_file', 'F', [{'stage': 'develop'}])
    assert 'only_file' in fl.read('auto') and 'only_file' in [p['name'] for p in pr.list()]                                   # the files' entries show under the database's
    fl.write('auto', 'p1', E1); pr.save('mine', 'M', [{'stage': 'develop'}])
    assert 'p1' in fs.read('auto') and 'mine' in [p['name'] for p in ps.list()]                                               # dual-write
    fl.write('auto', 'only_file', None); assert 'only_file' not in fl.read('auto') and 'only_file' not in fs.read('auto')    # taken off in both


def test_the_pipeline_presets_and_the_flag_endpoints_run_on_the_database(tmp_path, db_url, monkeypatch):
    from fastapi.testclient import TestClient
    from stratastills.server.app import create_app
    monkeypatch.setenv('STRATASTILLS_DATABASE_URL', db_url); monkeypatch.setenv('STRATASTILLS_DB_READS', 'flags,presets'); monkeypatch.setenv('STRATASTILLS_REPORT_FLAGS', str(tmp_path / 'flags'))
    for m in (flags_store, presets_store): m._stores.clear()
    try:
        assert presets_store.get().kind == 'database' and flags_store.get().kind == 'database'
        PL.save_preset('mine', 'Mine', [{'stage': 'develop'}, {'stage': 'encode', 'format': 'jpeg', 'quality': 90}]); assert 'mine' in [p['name'] for p in PL.list_presets() if not p['builtin']]
        c = TestClient(create_app()); r = c.put('/api/report-flags/auto/p1', json=dict(rating='bad', reasons=['too dark'], note='n')); assert r.status_code == 200 and r.json()['rating'] == 'bad'
        assert c.get('/api/report-flags/auto').json()['p1']['reasons'] == ['too dark']
        c.put('/api/report-flags/auto/p1', json=dict(rating='', reasons=[], note='')); assert c.get('/api/report-flags/auto').json() == {}
        PL.delete_preset('mine'); assert 'mine' not in [p['name'] for p in PL.list_presets()]
    finally:
        for m in (flags_store, presets_store): m._stores.clear()
