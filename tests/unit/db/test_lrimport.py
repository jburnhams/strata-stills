"""Lightroom-side importers: planning needs no database; writing is checked on SQLite (`pytest --pg` runs it on Postgres)."""
import pytest

pytest.importorskip('omarchy_db'); pytest.importorskip('pgvector')
from sqlalchemy import func, select

from stratastills.db import hashes, lrimport as L
from stratastills.db.models import Analysis, Detection, Embedding, File, LightroomEdit, NsfwResult, Photo

RAW, PERS = '/data/raw', '/data/personal'
A, B, C = 'a' * 64, 'b' * 64, 'c' * 64
HASHED = {f'{RAW}/2018/2018-05-28/DSC1.ARW': A, f'{RAW}/2018/2018-06-16/DSC2.ARW': B, f'{RAW}/2019/2019-01-01/DSC3.ARW': C}
CAT = [dict(raw=f'{RAW}/2018/2018-05-28/DSC1.ARW', final=f'{PERS}/2018/2018-05-26 - Trip/DSC1-Edit.jpg', Exposure2012='+0.5', CropLeft='0.1', HasCrop='True', Version='7.0'),
       dict(raw=f'{RAW}/2018/2018-06-16/DSC2.ARW', final=f'{PERS}/2018/2018-06-16/DSC2.jpg', error='no xmp'),                                    # not in the faces/clip order
       dict(raw=f'{RAW}/2019/2019-01-01/DSC3.ARW', final=f'{PERS}/2019/2019-01-01/DSC3.jpg', Exposure2012='-1')]
STAT = {f'{PERS}/2018/2018-05-26 - Trip/DSC1-Edit.jpg': (100, 1), f'{PERS}/2018/2018-06-16/DSC2.jpg': (200, 2), f'{PERS}/2019/2019-01-01/DSC3.jpg': (300, 3)}
NSFW = {f'nudenet-320n|{PERS}/2018/2018-05-26 - Trip/DSC1-Edit.jpg|100|1': [0.2, ''], f'nudenet-320n|{PERS}/2019/2019-01-01/DSC3.jpg|299|3': [0.9, 'x'], 'broken': [0, '']}
FACES = {'0': [dict(box=[.1, .2, .3, .4], score=.9, cam=[1, 2, 3], fin=[4, 5, 6])], '1': []}                                                     # indexes into the FILTERED catalogue (rows 0 and 2)


@pytest.fixture
def loaded(db_session):
    hashes.write(db_session, hashes.plan([dict(path=p, sha256=s, size=1, mtime_ns=1) for p, s in HASHED.items()], [RAW])); return db_session


def test_filtered_rows_follow_lr_dataset():
    assert [r['raw'][-8:] for r in L.filtered_rows(CAT)] == ['DSC1.ARW', 'DSC3.ARW']


def test_plans_without_a_database():
    p = L.plan_lightroom(CAT, HASHED, [PERS], stat=STAT.get); assert p['counts'] == dict(lightroom_edit=3, file=3, problems=0)
    e = p['edits'][0]; assert e['crop'] == dict(CropLeft='0.1', HasCrop='True') and e['settings'] == dict(Exposure2012='+0.5', Version='7.0') and p['edits'][1]['error'] == 'no xmp'
    n = L.plan_nsfw(NSFW, CAT, HASHED, [PERS]); assert n['counts'] == dict(nsfw_result=2, file=2, problems=1) and n['results'][1]['file'][2] == 299
    f = L.plan_faces(CAT, FACES, HASHED); assert f['counts'] == dict(analysis=2, detection=1, problems=0) and f['pictures'][1]['sha256'] == C
    assert any(x.startswith('FATAL') for x in L.plan_faces(CAT, {'0': []}, HASHED)['problems'])                                                  # a count that does not fit: the order cannot be trusted
    v = L.plan_embeddings(CAT, dict(idx=[0, 1], pca=[[1, 2], [3, 4]]), HASHED); assert v['counts'] == dict(embedding=2, problems=0) and v['vectors'][1]['sha256'] == C


def test_writes_are_idempotent_and_keep_the_scanned_version(loaded):
    s = loaded; lr = L.plan_lightroom(CAT, HASHED, [PERS], stat=STAT.get); ns = L.plan_nsfw(NSFW, CAT, HASHED, [PERS])
    assert L.write_lightroom(s, lr) == dict(lightroom_edit=3, file=3) and L.write_lightroom(s, lr) == dict(lightroom_edit=0, file=0)
    assert L.write_nsfw(s, ns) == dict(nsfw_result=2, file=1) and L.write_nsfw(s, ns) == dict(nsfw_result=0, file=0)           # DSC1's scanned version is the one now on disk; DSC3 was scanned at another size
    assert s.scalar(select(func.count()).select_from(File).where(File.role == 'final')) == 4 and s.scalar(select(func.count()).select_from(NsfwResult)) == 2
    assert s.scalars(select(LightroomEdit).order_by(LightroomEdit.id)).first().crop == dict(CropLeft='0.1', HasCrop='True')
    fc = L.plan_faces(CAT, FACES, HASHED); assert L.write_faces(s, fc) == dict(analysis=2, detection=1) and L.write_faces(s, fc) == dict(analysis=0, detection=0)
    d = s.scalars(select(Detection)).one(); assert (d.x, d.w, d.label, d.data['fin']) == (.1, .3, 'face', [4, 5, 6])
    em = L.plan_embeddings(CAT, dict(idx=[0, 1], pca=[[1, 2], [3, 4]]), HASHED); assert L.write_embeddings(s, em) == dict(embedding=2) and L.write_embeddings(s, em) == dict(embedding=0)
    assert s.scalar(select(func.count()).select_from(Embedding)) == 2 and s.scalar(select(func.count()).select_from(Analysis)) == 2


def test_refuses_without_photos_or_with_a_fatal_plan(db_session):
    with pytest.raises(L.ImportMismatch): L.write_lightroom(db_session, L.plan_lightroom(CAT, HASHED, [PERS], stat=STAT.get))             # the hashes are not loaded
    with pytest.raises(L.ImportMismatch): L.write_faces(db_session, L.plan_faces(CAT, {'0': []}, HASHED))
    assert db_session.scalar(select(func.count()).select_from(Photo)) == 0
