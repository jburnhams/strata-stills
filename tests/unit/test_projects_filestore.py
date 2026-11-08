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


def _gen(folder, item, key, source, file_field=None, ext='.jpg', picture=True, output=(100, 50)):
    d = os.path.join(folder, 'generated', item); os.makedirs(d, exist_ok=True)
    if picture: open(os.path.join(d, key + ext), 'wb').write(b'x')
    rec = dict(key=key, item=item, source=source, output=list(output), file=file_field if file_field is not None else os.path.join(d, key + ext), created='2026-01-01T00:00:00')
    json.dump(rec, open(os.path.join(d, key + '.json'), 'w')); return rec


UNC = '\\\\Server\\Drone\\shoot'


def test_generated_where_ids_and_paths_match_nothing_changes(tmp_path):
    from stratastills import projects as P
    f = str(tmp_path); iid = P.item_id(P.source_id(UNC), 'S1'); _gen(f, iid, 'k1', UNC + '\\S1.DNG')
    st = projects_store.FileStore(); got = st.generated('p', f, iid, [UNC])
    assert [r['key'] for r in got] == ['k1'] and got[0]['file'] == os.path.join(f, 'generated', iid, 'k1.jpg') and st.read_sidecar('p', f, iid, 'k1', [UNC])['key'] == 'k1' and st.read_sidecar('p', f, iid, 'zz', [UNC]) is None
    assert st.generated('p', f, P.item_id(P.source_id(UNC), 'OTHER'), [UNC]) == []


def test_generated_made_on_another_machine_is_found_by_stem_in_its_source_and_by_file_name(tmp_path):
    from stratastills import projects as P
    f = str(tmp_path); win = '4069159d~S1'; local = P.item_id(P.source_id(UNC), 'S1'); assert local != win
    _gen(f, win, 'k1', UNC + '\\S1.DNG', file_field='C:\\Code\\strata-stills\\projects\\p\\generated\\4069159d~S1\\k1.jpg')                  # the id differs and the recorded path is a Windows path that does not exist here
    st = projects_store.FileStore(); got = st.generated('p', f, local, [UNC])
    assert [r['key'] for r in got] == ['k1'] and got[0]['file'] == os.path.join(f, 'generated', win, 'k1.jpg')
    assert st.generated('p', f, P.item_id(P.source_id('\\\\Server\\Drone\\other'), 'S1'), [UNC, '\\\\Server\\Drone\\other']) == []                 # the same stem in ANOTHER source of the project is not this item's
    assert [r['key'] for r in st.generated('p', f, P.item_id('whatever', 'S1'), [UNC])] == ['k1']                                                 # a one-source project: an id made elsewhere names that source


def test_a_record_whose_picture_is_nowhere_is_skipped_and_sidecars_round_trip(tmp_path):
    f = str(tmp_path); st = projects_store.FileStore(); iid = 'x~S1'
    _gen(f, iid, 'gone', UNC + '\\S1.DNG', picture=False); assert st.generated('p', f, iid, [UNC]) == []
    d = os.path.join(f, 'generated', iid); open(os.path.join(d, 'k2.tif'), 'wb').write(b'x')
    st.write_sidecar('p', f, iid, 'k2', dict(key='k2', file=os.path.join(d, 'k2.tif'), output=[1, 2], source=UNC + '\\S1.DNG')); assert [r['key'] for r in st.generated('p', f, iid, [UNC])] == ['k2']
