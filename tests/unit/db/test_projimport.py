"""Project import: planning needs no database; writing is checked on SQLite (`pytest --pg` runs it on Postgres)."""
import json

import pytest

pytest.importorskip('omarchy_db'); pytest.importorskip('pgvector')
from sqlalchemy import func, select

from stratastills.db import hashes, projimport as P
from stratastills.db.models import File, Photo, Project, ProjectRule, ProjectSource, Render

UNC = '\\\\Server\\Drone\\20250402 - Shoot'
A, B, C = 'a' * 64, 'b' * 64, 'c' * 64


@pytest.fixture
def world(tmp_path):
    root = tmp_path / 'drone'; src = root / '20250402 - Shoot'; src.mkdir(parents=True); proj = tmp_path / 'projects' / 'shoot'; (proj / 'generated' / 'dca8e83e~DJI_1').mkdir(parents=True); (proj / 'generated' / 'dca8e83e~DJI_2').mkdir(); (proj / 'cache').mkdir()
    (proj / 'project.json').write_text(json.dumps(dict(schema=1, name='Shoot', created='2026-10-22T03:10:41+01:00', sources=[dict(path=UNC, added='2026-10-22T03:10:41+01:00')], auto=[dict(source='*', pipeline='best')])), encoding='utf8')
    side = lambda key, **kw: dict(dict(item='dca8e83e~DJI_1', stem='DJI_1', key=key, source_variant='raw', preset='best', stages=[{'stage': 'develop'}], summary='JPEG q90', output=[100, 50], bytes=7, quality=90, created='2026-10-22T14:39:59+01:00'), **kw)
    (proj / 'generated' / 'dca8e83e~DJI_1' / 'k1.json').write_text(json.dumps(side('k1')), encoding='utf8'); (proj / 'generated' / 'dca8e83e~DJI_1' / 'k2.json').write_text(json.dumps(side('k2', preset=None)), encoding='utf8')
    (proj / 'generated' / 'dca8e83e~DJI_2' / 'k3.json').write_text(json.dumps(side('k3')), encoding='utf8'); (proj / 'cache' / 'abc.json').write_text('{}', encoding='utf8')
    recs = [dict(path=str(src / f), sha256=s, size=n, mtime_ns=n) for f, s, n in (('DJI_1.DNG', A, 1), ('DJI_1.JPG', B, 2), ('DJI_2.dng', C, 3), ('DJI_3.DNG', 'd' * 64, 4))]
    return root, proj, recs


def test_plan_finds_items_by_stem_and_resolves_the_windows_source(world):
    root, proj, recs = world; p = P.plan(proj, recs, [root])
    assert p['counts'] == dict(project=1, project_source=1, project_rule=1, photos_linked=2, file_jpeg=1, render=3, dng_hashed=3, dng_without_item=1, problems=0)
    assert p['sources'][0]['root'] == 'drone' and p['sources'][0]['relpath'] == '20250402 - Shoot' and p['sources'][0]['original_path'] == UNC
    assert p['photos'][('drone', '20250402 - Shoot', 'DJI_1')]['item'] == 'dca8e83e~DJI_1' and p['renders'][0]['path'] == 'generated/dca8e83e~DJI_1/k1.jpg' and p['renders'][0]['cache_key'] == 'k1' and p['renders'][0]['out_w'] == 100
    assert any(x.startswith('FATAL') for x in P.plan(proj, recs, [root.parent])['problems'])                                                    # the source is on none of the roots
    assert P.plan(proj, [r for r in recs if 'DJI_2' not in r['path']], [root])['problems'] == ['dca8e83e~DJI_2: no DNG named DJI_2 among the hashed files of its source']


def test_write_is_idempotent_and_keeps_the_legacy_id_and_the_cache_keys(world, db_session):
    root, proj, recs = world; s = db_session; p = P.plan(proj, recs, [root])
    with pytest.raises(P.ProjectMismatch): P.write(s, p)                                                                                         # the DNGs are not loaded
    hashes.write(s, hashes.plan([r for r in recs if r['path'].lower().endswith('.dng')], [root]))
    assert P.write(s, p) == dict(project=1, project_source=1, project_rule=1, file_jpeg=1, render=3) and P.write(s, p) == dict(project=0, project_source=0, project_rule=0, file_jpeg=0, render=0)
    assert s.scalar(select(func.count()).select_from(Render)) == 3 and {r.cache_key for r in s.scalars(select(Render))} == {'k1', 'k2', 'k3'}
    ph = s.scalars(select(Photo).where(Photo.sha256 == A)).one(); assert ph.legacy_item_id == 'dca8e83e~DJI_1'
    j = s.scalars(select(File).where(File.role == 'jpeg')).one(); assert (j.photo_id, j.sha256, j.relpath) == (ph.id, B, '20250402 - Shoot/DJI_1.JPG')
    r = s.scalars(select(Render).where(Render.cache_key == 'k2')).one(); assert r.preset is None and r.info['key'] == 'k2' and r.created_at.year == 2026 and r.photo_id == ph.id
    assert s.scalars(select(ProjectSource)).one().original_path == UNC and s.scalars(select(Project)).one().slug == 'shoot' and s.scalars(select(ProjectRule)).one().pipeline == 'best'


def test_a_different_legacy_id_is_a_conflict_not_an_overwrite(world, db_session):
    root, proj, recs = world; s = db_session; hashes.write(s, hashes.plan([r for r in recs if r['path'].lower().endswith('.dng')], [root]))
    s.scalars(select(Photo).where(Photo.sha256 == A)).one().legacy_item_id = 'other~DJI_1'; s.commit()
    with pytest.raises(P.ProjectMismatch): P.write(s, P.plan(proj, recs, [root]))
    assert s.scalar(select(func.count()).select_from(Render)) == 0


def _project(tmp_path, sources, items):
    """A project with several sources (stored as UNC paths) and items: [(item, stem, source path or None, file name or None, key)]."""
    proj = tmp_path / 'projects' / 'multi'; proj.mkdir(parents=True)
    (proj / 'project.json').write_text(json.dumps(dict(schema=1, name='Multi', sources=[dict(path=sp) for sp in sources], auto=[])), encoding='utf8')
    for item, stem, src, fname, key in items:
        (proj / 'generated' / item).mkdir(parents=True, exist_ok=True)
        (proj / 'generated' / item / f'{key}.json').write_text(json.dumps(dict(item=item, stem=stem, key=key, source=src, file=fname, format='tiff' if fname and fname.endswith('.tif') else 'jpeg')), encoding='utf8')
    return proj


def test_sources_sharing_a_stem_never_mix_and_an_ambiguous_stem_is_rejected(tmp_path):
    root = tmp_path / 'drone'; (root / 'one').mkdir(parents=True); (root / 'two').mkdir()
    recs = [dict(path=str(root / 'one' / 'DJI_1.DNG'), sha256=A, size=1, mtime_ns=1), dict(path=str(root / 'two' / 'DJI_1.DNG'), sha256=B, size=1, mtime_ns=1),
            dict(path=str(root / 'two' / 'DJI_9.DNG'), sha256=C, size=1, mtime_ns=1), dict(path=str(root / 'two' / 'DJI_9.dng'), sha256='d' * 64, size=1, mtime_ns=1)]
    proj = _project(tmp_path, ['\\\\S\\Drone\\one', '\\\\S\\Drone\\two'], [('x~DJI_1', 'DJI_1', '\\\\S\\Drone\\one\\DJI_1.DNG', 'k1.jpg', 'k1'), ('y~DJI_1', 'DJI_1', '\\\\S\\Drone\\two\\DJI_1.DNG', 'k2.jpg', 'k2'),
                                  ('y~DJI_9', 'DJI_9', '\\\\S\\Drone\\two\\DJI_9.DNG', 'k3.jpg', 'k3'), ('z~DJI_1', 'DJI_1', None, 'k4.jpg', 'k4')])
    p = P.plan(proj, recs, [root]); got = {k: v['sha256'] for k, v in p['photos'].items()}
    assert got == {('drone', 'one', 'DJI_1'): A, ('drone', 'two', 'DJI_1'): B}                                                                    # each item found its own source's file
    assert sorted(p['problems']) == ['y~DJI_9: 2 DNGs named DJI_9 in its source (ambiguous)', 'z~DJI_1: cannot tell which source of the project it comes from'] and p['counts']['render'] == 2


def test_the_render_path_follows_the_recorded_file_name(tmp_path):
    root = tmp_path / 'drone'; (root / 'one').mkdir(parents=True); recs = [dict(path=str(root / 'one' / 'DJI_1.DNG'), sha256=A, size=1, mtime_ns=1)]
    proj = _project(tmp_path, ['\\\\S\\Drone\\one'], [('x~DJI_1', 'DJI_1', None, 'C:\\Code\\projects\\multi\\generated\\x~DJI_1\\k9.tif', 'k9')])
    assert P.plan(proj, recs, [root])['renders'][0]['path'] == 'generated/x~DJI_1/k9.tif'


def test_an_older_jpeg_snapshot_imported_later_leaves_one_current_row(world, db_session):
    root, proj, recs = world; s = db_session; hashes.write(s, hashes.plan([r for r in recs if r['path'].lower().endswith('.dng')], [root]))
    P.write(s, P.plan(proj, recs, [root]))                                                                                                        # JPG at mtime 2
    older = [dict(r, mtime_ns=1, size=9, sha256='e' * 64) if r['path'].endswith('.JPG') else r for r in recs]; P.write(s, P.plan(proj, older, [root]))
    rows = list(s.scalars(select(File).where(File.role == 'jpeg').order_by(File.mtime_ns))); assert [(f.mtime_ns, f.current) for f in rows] == [(1, False), (2, True)]
    assert P.write(s, P.plan(proj, older, [root]))['file_jpeg'] == 0 and [f.current for f in s.scalars(select(File).where(File.role == 'jpeg').order_by(File.mtime_ns))] == [False, True]
