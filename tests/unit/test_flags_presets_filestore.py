"""The report-flag and preset file stores and the choice of stores (no database packages needed), and the server's flag endpoints on the files."""
import json, os

import pytest

from stratastills import flags_store, presets_store


@pytest.fixture(autouse=True)
def own_flags_folder(tmp_path, monkeypatch): monkeypatch.setenv('STRATASTILLS_REPORT_FLAGS', str(tmp_path / 'report-flags'))                 # not out/flags of the repository


def test_flag_file_store_sets_and_removes_one_picture(tmp_path):
    st = flags_store.FileStore(); assert st.read('auto') == {}
    assert st.write('auto', 'p1', dict(flag=True, rating='bad', reasons=['too dark'], note='n', status='open', at='2026-01-01T00:00:00', first='2026-01-01T00:00:00'))['rating'] == 'bad'
    st.write('auto', 'p2', dict(flag=True, rating='good')); assert sorted(st.read('auto')) == ['p1', 'p2'] and st.read('neutral') == {}
    assert st.write('auto', 'p1', None) == {} and sorted(st.read('auto')) == ['p2']


def test_preset_file_store_round_trips_and_skips_unreadable_files():
    st = presets_store.FileStore(); assert st.list() == [] and st.delete('nope') is False
    st.save('mine', 'My preset', [{'stage': 'develop'}]); st.save('other', None, [{'stage': 'resize'}])
    open(os.path.join(st._dir(), 'broken.json'), 'w').write('{not json')
    assert [(p['name'], p['label']) for p in st.list()] == [('mine', 'My preset'), ('other', 'other')] and st.list()[0]['stages'] == [{'stage': 'develop'}]
    assert st.delete('mine') is True and [p['name'] for p in st.list()] == ['other']


def test_a_database_url_alone_does_not_select_either_database_store(monkeypatch):
    for m in (flags_store, presets_store): m._stores.clear()
    monkeypatch.setenv('STRATASTILLS_DATABASE_URL', 'postgresql+psycopg:///x'); monkeypatch.delenv('STRATASTILLS_DB_READS', raising=False)
    assert flags_store.get().kind == 'files' and presets_store.get().kind == 'files'
    monkeypatch.setenv('STRATASTILLS_DB_READS', 'nsfw,projects,metadata'); assert flags_store.get().kind == 'files' and presets_store.get().kind == 'files'      # other read paths' entries do not either
    monkeypatch.delenv('STRATASTILLS_DATABASE_URL'); monkeypatch.setenv('STRATASTILLS_DB_READS', 'flags,presets'); assert flags_store.get().kind == 'files' and presets_store.get().kind == 'files'   # an entry without a URL does not either
