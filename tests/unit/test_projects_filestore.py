"""The projects file store and the choice of store (no database packages needed): the files as before, and the database only on its own opt-in."""
import json, os

from stratastills import projects_store


def test_file_store_round_trips_document_and_choices(tmp_path):
    st = projects_store.FileStore(); f = str(tmp_path)
    assert st.read_doc('p', f) is None and st.read_state('p', f) == {}
    st.write_doc('p', f, {'name': 'P', 'sources': []}); st.write_state('p', f, {'a~b': {'primary': 'raw'}})
    assert st.read_doc('p', f) == {'name': 'P', 'sources': []} and st.read_state('p', f) == {'a~b': {'primary': 'raw'}} and json.load(open(os.path.join(f, 'project.json')))['name'] == 'P'
    open(os.path.join(f, 'items.json'), 'w').write('[1]'); assert st.read_state('p', f) == {}                    # not a dict: no choices


def test_a_database_url_alone_does_not_select_the_database_store(monkeypatch):
    projects_store._stores.clear(); monkeypatch.setenv('STRATASTILLS_DATABASE_URL', 'postgresql+psycopg:///x'); monkeypatch.delenv('STRATASTILLS_DB_READS', raising=False)
    assert projects_store.get().kind == 'files'
    monkeypatch.setenv('STRATASTILLS_DB_READS', 'nsfw'); assert projects_store.get().kind == 'files'                # another read path's entry does not either
    monkeypatch.delenv('STRATASTILLS_DATABASE_URL'); monkeypatch.setenv('STRATASTILLS_DB_READS', 'projects'); assert projects_store.get().kind == 'files'
    projects_store._stores.clear()
