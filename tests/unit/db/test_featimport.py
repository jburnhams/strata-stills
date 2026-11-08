"""Derived feature matrices into analysis rows: the row-order proof and planning need no database; writing is checked on SQLite (`pytest --pg` runs it on Postgres). Uses numpy only for the arrays."""
import math

import pytest

pytest.importorskip('omarchy_db'); pytest.importorskip('pgvector'); np = pytest.importorskip('numpy')
from sqlalchemy import func, select

from stratastills.db import featimport as F, hashes
from stratastills.db.models import Analysis

RAW = '/data/raw'
CAT = [dict(raw=f'{RAW}/2018/2018-05-28/A.ARW', final='/p/a.jpg', Exposure2012='0'), dict(raw=f'{RAW}/x/E.ARW', final='/p/e.jpg', error='broken'), dict(raw=f'{RAW}/2019/2019-01-01/B.ARW', final='/p/b.jpg', Exposure2012='1')]   # the filtered catalogue is rows 0 and 2
H = {CAT[0]['raw']: 'a' * 64, CAT[2]['raw']: 'b' * 64}
TAGS = {CAT[0]['raw']: {'ExifIFD:ISO': 100, 'ExifIFD:FNumber': 8, 'ExifIFD:DateTimeOriginal': '2018:05:28 09:00:26'}, CAT[2]['raw']: {'ExifIFD:ISO': 400, 'ExifIFD:FNumber': 2.8, 'ExifIFD:DateTimeOriginal': '2019:01:01 10:00:00'}}


def arrays(iso=(100, 400)):
    idx = np.array([0, 1])
    return dict(clip=dict(idx=idx, labels=np.array([[.1, .9], [.5, .5]], np.float32), label_names=np.array(['where:in', 'where:out'])),
                meta=dict(idx=idx, M=np.array([[iso[0], 8.0, np.nan], [iso[1], 2.8, 3.0]], np.float32), names=np.array(['ExifIFD:ISO', 'ExifIFD:FNumber', 'x'])),
                features=dict(idx=idx, X=np.array([[math.log2(100), 8.0, 2018], [math.log2(400), 2.8, 2019]], np.float32), names=np.array(['log_iso', 'fnum', 'year']), F=np.arange(20, dtype=np.float32).reshape(2, 10), C=np.arange(12, dtype=np.float32).reshape(2, 6)),
                fit=dict(idx=idx, params=np.ones((2, 3), np.float32), decoded=np.zeros((2, 2), np.float32), before=np.ones((2, 2), np.float32), after=np.zeros((2, 2), np.float32), names=np.array(['p0', 'p1', 'p2'])))


KEYS = {'clip.npz': 'k1', 'meta.npz': 'k2', 'features_all.npz': 'k3', 'fit.npz': 'k4'}


def test_plan_proves_the_order_and_names_every_column():
    p = F.plan(CAT, H, TAGS, arrays(), KEYS)
    assert p['counts'] == dict(clip_labels=2, exif_features=2, look_features=2, final_stats=2, lr_fit=2, duplicate_photo_skipped=0, problems=0)
    r = p['rows']; assert r['exif_features'][0]['data'] == dict(names=['ExifIFD:ISO', 'ExifIFD:FNumber', 'x'], values=[100.0, 8.0, None]) and r['clip_labels'][1]['sha256'] == 'b' * 64
    assert r['final_stats'][0]['data']['names'][:3] == ['L_p1', 'L_p5', 'L_p10'] and len(r['final_stats'][0]['data']['values']) == 16 and r['lr_fit'][0]['data']['names'] == ['p0', 'p1', 'p2']


def test_a_matrix_in_the_wrong_order_is_fatal_and_so_is_an_unproven_idx():
    p = F.plan(CAT, H, TAGS, arrays(iso=(400, 100)), KEYS); assert any(x.startswith('FATAL') and 'row order' in x for x in p['problems']) and p['counts']['clip_labels'] == 0
    a = arrays(); a['fit']['idx'] = np.array([0, 7]); q = F.plan(CAT, H, TAGS, a, KEYS); assert any(x.startswith('FATAL') and 'fit.npz' in x and 'outside' in x for x in q['problems'])
    a = arrays(); a['clip']['idx'] = np.array([1, 1]); assert any('appears twice' in x for x in F.plan(CAT, H, TAGS, a, KEYS)['problems'])


def test_write_is_idempotent_and_a_regenerated_file_is_a_new_row(db_session):
    s = db_session; hashes.write(s, hashes.plan([dict(path=k, sha256=v, size=1, mtime_ns=1) for k, v in H.items()], [RAW])); p = F.plan(CAT, H, TAGS, arrays(), KEYS)
    assert F.write(s, p) == dict(clip_labels=2, exif_features=2, look_features=2, final_stats=2, lr_fit=2) and F.write(s, p) == dict(clip_labels=0, exif_features=0, look_features=0, final_stats=0, lr_fit=0)
    assert s.scalar(select(func.count()).select_from(Analysis)) == 10
    assert F.write(s, F.plan(CAT, H, TAGS, arrays(), dict(KEYS, **{'meta.npz': 'new'})))['exif_features'] == 2 and s.scalar(select(func.count()).select_from(Analysis).where(Analysis.kind == 'exif_features')) == 4
    with pytest.raises(F.FeatureMismatch): F.write(s, F.plan(CAT, H, TAGS, arrays(iso=(400, 100)), KEYS))


def test_each_matrix_is_read_through_its_own_row_map():
    a = arrays(); a['features'].update(idx=np.array([1, 0]), X=a['features']['X'][::-1].copy(), F=a['features']['F'][::-1].copy(), C=a['features']['C'][::-1].copy())          # features rows in the order of their idx; meta stays in catalogue order
    a['clip'].update(idx=np.array([1]), labels=np.array([[.5, .5]], np.float32))                                                                                              # clip has only catalogue row 1
    a['fit'].update(idx=np.array([1, 0]), params=a['fit']['params'][::-1].copy() * np.array([[1], [2]], np.float32), decoded=a['fit']['decoded'], before=a['fit']['before'], after=a['fit']['after'])
    p = F.plan(CAT, H, TAGS, a, KEYS); assert p['counts']['problems'] == 0, p['problems']
    by = lambda kind: {r['sha256']: r['data'] for r in p['rows'][kind]}
    assert by('exif_features')['a' * 64]['values'][0] == 100.0 and by('exif_features')['b' * 64]['values'][0] == 400.0                                                     # meta row c is catalogue row c
    assert by('look_features')['a' * 64]['values'][1] == 8.0 and by('look_features')['b' * 64]['values'][1] == pytest.approx(2.8)                                          # features row i is catalogue row idx[i]
    assert list(by('clip_labels')) == ['b' * 64] and p['counts']['clip_labels'] == 1                                                                                          # a photo the matrix has no row for gets no row
    assert by('lr_fit')['b' * 64]['params'] == [1.0, 1.0, 1.0] and by('lr_fit')['a' * 64]['params'] == [2.0, 2.0, 2.0]


def test_a_row_that_cannot_be_checked_is_fatal_not_skipped():
    tags = dict(TAGS); del tags[CAT[2]['raw']]                                                            # no exiftool record for the second picture
    p = F.plan(CAT, H, tags, arrays(), KEYS); assert any(x.startswith('FATAL') and 'no exiftool record' in x for x in p['problems']) and p['counts']['clip_labels'] == 0
    a = arrays(); a['meta']['M'][1][0] = np.nan                                                          # one side absent, the other present: a mismatch, not a skip
    assert any(x.startswith('FATAL') and 'differ' in x for x in F.plan(CAT, H, TAGS, a, KEYS)['problems'])
    t2 = dict(TAGS); t2[CAT[2]['raw']] = dict(TAGS[CAT[2]['raw']], **{'ExifIFD:ISO': None}); b = arrays(); b['meta']['M'][1][0] = np.nan; b['features']['X'][1][0] = np.nan     # the second picture has no ISO on either side: agreed; the other rows and columns still prove the order
    assert F.plan(CAT, H, t2, b, KEYS)['counts']['problems'] == 0
