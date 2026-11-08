"""The project document and the item choices: the SAME contract on the file store and the database store (SQLite here; `--pg` on the real thing), and the projects API running on the database."""
import json, os
from datetime import datetime

import pytest

pytest.importorskip('omarchy_db'); pytest.importorskip('pgvector')
from sqlalchemy import func, select

from stratastills import projects as P, projects_store
from stratastills.db import hashes
from stratastills.db.models import ItemChoice, Project, ProjectRule, ProjectSource
from stratastills.db.projects import DbStore

UNC = '\\\\Server\\Drone\\shoot'


@pytest.fixture(params=['files', 'db'])
def env(request, tmp_path, db_url):
    root = tmp_path / 'drone'; (root / 'shoot').mkdir(parents=True); folder = tmp_path / 'projects' / 'p1'; folder.mkdir(parents=True)
    st = projects_store.FileStore() if request.param == 'files' else DbStore(db_url, roots=[str(root)])
    if request.param == 'db':
        from omarchy_db import connect, session
        with session(connect(db_url)) as s: hashes.write(s, hashes.plan([dict(path=str(root / 'shoot' / 'S1.DNG'), sha256='a' * 64, size=1, mtime_ns=1)], [str(root)]))
    return st, str(folder), str(root), request.param


DOC = dict(schema=1, name='P One', created='2026-10-22T03:10:41+01:00', sources=[dict(path=UNC, added='2026-10-22T03:10:41+01:00')], auto=[dict(source='*', pipeline='best')])


def instant(s): return datetime.fromisoformat(s).timestamp() if datetime.fromisoformat(s).tzinfo else datetime.fromisoformat(s).replace(tzinfo=__import__('datetime').timezone.utc).timestamp()


def test_contract_document_round_trip_and_update(env):
    st, folder, root, kind = env; assert st.read_doc('p1', folder) is None
    st.write_doc('p1', folder, DOC); d = st.read_doc('p1', folder)
    assert d['name'] == 'P One' and [x['path'] for x in d['sources']] == [UNC] and d['auto'] == [dict(source='*', pipeline='best')] and d['schema'] == 1
    assert instant(d['created']) == instant(DOC['created']) and instant(d['sources'][0]['added']) == instant(DOC['sources'][0]['added'])        # the same instant, whatever the offset it is written with
    two = dict(DOC, name='Renamed', sources=DOC['sources'] + [dict(path=os.path.join(root, 'other'), added='2026-11-01T00:00:00')], auto=[dict(source='*', pipeline='best', variant='raw'), dict(source='x', pipeline='fast')])
    st.write_doc('p1', folder, two); d = st.read_doc('p1', folder)
    assert d['name'] == 'Renamed' and len(d['sources']) == 2 and d['auto'] == two['auto']
    one = dict(two, sources=two['sources'][:1]); st.write_doc('p1', folder, one); assert len(st.read_doc('p1', folder)['sources']) == 1                  # a source taken out stays out
    assert json.load(open(os.path.join(folder, 'project.json')))['name'] == 'Renamed'                                                                    # the files are kept either way


def test_contract_choices_round_trip(env):
    st, folder, root, kind = env; st.write_doc('p1', folder, dict(DOC, sources=[dict(path=os.path.join(root, 'shoot'), added='2026-10-22T03:10:41')]))
    sid = P.source_id(os.path.join(root, 'shoot')); known, unknown = P.item_id(sid, 'S1'), P.item_id(sid, 'NEW')                                           # S1 is a photo in the database, NEW is not
    assert st.read_state('p1', folder) == {}
    st.write_state('p1', folder, {known: {'primary': 'jpeg'}, unknown: {'primary': 'raw'}})
    assert st.read_state('p1', folder) == {known: {'primary': 'jpeg'}, unknown: {'primary': 'raw'}}
    st.write_state('p1', folder, {known: {'primary': 'raw'}, unknown: {'primary': 'raw'}}); assert st.read_state('p1', folder)[known] == {'primary': 'raw'}
    if kind == 'db':
        with st._session() as s: assert s.scalar(select(func.count()).select_from(ItemChoice)) == 1                                                       # only the item with a photo is in the database


def test_the_projects_api_runs_on_the_database(tmp_path, db_url, monkeypatch):
    pr = tmp_path / 'projects'; pr.mkdir(); drone = tmp_path / 'drone'; (drone / 'shoot').mkdir(parents=True)
    monkeypatch.setenv('STRATASTILLS_PROJECTS', str(pr)); monkeypatch.setenv('STRATASTILLS_SOURCES', str(drone)); monkeypatch.setenv('STRATASTILLS_DATABASE_URL', db_url); monkeypatch.setenv('STRATASTILLS_DB_READS', 'projects'); projects_store._stores.clear()
    try:
        doc = P.create('Shoot Day', sources=[str(drone / 'shoot')]); pid = doc['id']; assert projects_store.get().kind == 'database'
        from omarchy_db import connect, session
        with session(connect(db_url)) as s: assert s.scalar(select(func.count()).select_from(Project)) == 1 and s.scalar(select(func.count()).select_from(ProjectSource)) == 1
        P.add_source(pid, str(drone)); assert len(P.load(pid)['sources']) == 2
        assert [p['id'] for p in P.list_projects()] == [pid]
    finally: projects_store._stores.clear()


def _two_sources(tmp_path, db_url):
    from omarchy_db import connect, session
    drone = tmp_path / 'drone'; (drone / 'a').mkdir(parents=True); (drone / 'b').mkdir(); folder = tmp_path / 'projects' / 'p1'; folder.mkdir(parents=True)
    st = DbStore(db_url, roots=[str(drone)])
    with session(connect(db_url)) as s:
        hashes.write(s, hashes.plan([dict(path=str(drone / 'b' / 'S1.DNG'), sha256='b' * 64, size=1, mtime_ns=1)], [str(drone)]))                  # S1 exists only in source b
    doc = dict(DOC, sources=[dict(path=str(drone / 'a'), added='2026-10-22T03:10:41'), dict(path=str(drone / 'b'), added='2026-10-22T03:10:41')])
    st.write_doc('p1', str(folder), doc); return st, str(folder), str(drone)


def test_an_item_is_matched_in_its_own_source_only(tmp_path, db_url):
    st, folder, drone = _two_sources(tmp_path, db_url); sa, sb = P.source_id(os.path.join(drone, 'a')), P.source_id(os.path.join(drone, 'b'))
    st.write_state('p1', folder, {P.item_id(sa, 'S1'): {'primary': 'jpeg'}, P.item_id(sb, 'S1'): {'primary': 'raw'}})                                  # S1 in source a is NOT imported; in b it is
    with st._session() as s: assert [c.primary_variant for c in s.scalars(select(ItemChoice))] == ['raw']                                           # the choice for a/S1 did not land on b's photo
    assert st.read_state('p1', folder) == {P.item_id(sa, 'S1'): {'primary': 'jpeg'}, P.item_id(sb, 'S1'): {'primary': 'raw'}}                         # a/S1 comes from the file only; a/S1 is not overwritten by b's choice


def test_a_raw_present_at_two_paths_and_a_superseded_version(tmp_path, db_url):
    from omarchy_db import connect, session
    st, folder, drone = _two_sources(tmp_path, db_url); sb = P.source_id(os.path.join(drone, 'b')); iid = P.item_id(sb, 'S1')
    with session(connect(db_url)) as s:
        hashes.write(s, hashes.plan([dict(path=os.path.join(drone, 'a', 'S1.DNG'), sha256='b' * 64, size=1, mtime_ns=1), dict(path=os.path.join(drone, 'zz', 'S1.DNG'), sha256='b' * 64, size=1, mtime_ns=1)], [drone]))      # the same raw also at other paths (one outside the project)
    st.write_state('p1', folder, {iid: {'primary': 'jpeg'}})
    with st._session() as s: assert s.scalar(select(func.count()).select_from(ItemChoice)) == 1 and st.read_state('p1', folder)[iid] == {'primary': 'jpeg'}
    from omarchy_db import connect as c2, session as s2
    with s2(c2(db_url)) as s: hashes.write(s, hashes.plan([dict(path=os.path.join(drone, 'b', 'S1.DNG'), sha256='c' * 64, size=2, mtime_ns=2)], [drone]))   # b/S1 changed: a new photo, the old file version is no longer current
    st.write_state('p1', folder, {iid: {'primary': 'raw'}})
    state = st.read_state('p1', folder)
    assert state[iid] == {'primary': 'raw'}                                                                                                          # the new selection is not reverted by the old photo's choice
    with st._session() as s: assert sorted(c.primary_variant for c in s.scalars(select(ItemChoice))) == ['jpeg', 'raw']


def _gen_files(folder, item, key, source, file_field=None):
    d = os.path.join(folder, 'generated', item); os.makedirs(d, exist_ok=True); open(os.path.join(d, key + '.jpg'), 'wb').write(b'x')
    rec = dict(key=key, item=item, source=source, output=[100, 50], file=file_field or os.path.join(d, key + '.jpg'), created='2026-01-01T00:00:00', preset='best', stages=[{'stage': 'develop'}], summary='s', bytes=1, quality=90, source_variant='raw')
    json.dump(rec, open(os.path.join(d, key + '.json'), 'w')); return rec


def test_contract_generated_variants_and_sidecars(env):
    st, folder, root, kind = env; src = os.path.join(root, 'shoot'); st.write_doc('p1', folder, dict(DOC, sources=[dict(path=src, added='2026-10-22T03:10:41')]))
    iid = P.item_id(P.source_id(src), 'S1'); new = P.item_id(P.source_id(src), 'NEW')                                                                  # S1 is a photo in the database, NEW is not
    assert st.generated('p1', folder, iid, [src]) == []
    st.write_sidecar('p1', folder, iid, 'k1', _gen_files(folder, iid, 'k1', src + '/S1.DNG')); st.write_sidecar('p1', folder, new, 'k2', _gen_files(folder, new, 'k2', src + '/NEW.DNG'))
    assert [r['key'] for r in st.generated('p1', folder, iid, [src])] == ['k1'] and [r['key'] for r in st.generated('p1', folder, new, [src])] == ['k2']
    assert st.read_sidecar('p1', folder, iid, 'k1', [src])['output'] == [100, 50] and st.read_sidecar('p1', folder, iid, 'nope', [src]) is None
    if kind == 'db':
        from stratastills.db.models import Render
        with st._session() as s: assert [(r.cache_key, r.out_w, r.path) for r in s.scalars(select(Render))] == [('k1', 100, f'generated/{iid}/k1.jpg')]            # only the item with a photo is in the database


def test_the_database_finds_outputs_made_on_another_machine_and_the_projects_api_shows_them(tmp_path, db_url, monkeypatch):
    from omarchy_db import connect, session
    from stratastills.db.models import Render
    drone = tmp_path / 'drone'; (drone / 'shoot').mkdir(parents=True); pr = tmp_path / 'projects'; folder = pr / 'p1'; folder.mkdir(parents=True)
    with session(connect(db_url)) as s: hashes.write(s, hashes.plan([dict(path=str(drone / 'shoot' / 'S1.DNG'), sha256='a' * 64, size=1, mtime_ns=1)], [str(drone)]))
    src = str(drone / 'shoot'); st = DbStore(db_url, roots=[str(drone)]); st.write_doc('p1', str(folder), dict(DOC, sources=[dict(path=src, added='2026-10-22T03:10:41')]))
    win = '4069159d~S1'; rec = _gen_files(str(folder), win, 'k1', src + '\\S1.DNG', file_field='C:\\Code\\projects\\p1\\generated\\4069159d~S1\\k1.jpg')                 # made elsewhere: another id, a Windows path
    with session(connect(db_url)) as s:
        from stratastills.db.models import Photo
        ph = s.scalars(select(Photo)).one(); s.add(Render(photo_id=ph.id, cache_key='k1', kind='generated', out_w=100, out_h=50, path=f'generated/{win}/k1.jpg', info=rec)); s.commit()
    got = st.generated('p1', str(folder), P.item_id(P.source_id(src), 'S1'), [src]); assert [r['key'] for r in got] == ['k1'] and got[0]['file'] == str(folder / 'generated' / win / 'k1.jpg')
    monkeypatch.setenv('STRATASTILLS_PROJECTS', str(pr)); monkeypatch.setenv('STRATASTILLS_SOURCES', str(drone)); monkeypatch.setenv('STRATASTILLS_DATABASE_URL', db_url); monkeypatch.setenv('STRATASTILLS_DB_READS', 'projects'); projects_store._stores.clear()
    try: assert [v['key'] for v in P.generated_variants('p1', P.item_id(P.source_id(src), 'S1'))] == ['k1']
    finally: projects_store._stores.clear()
