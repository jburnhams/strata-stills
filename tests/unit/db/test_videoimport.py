"""Video import: SRT parsing and planning need no database; writing is checked on SQLite (`pytest --pg` runs it on Postgres)."""
import pytest

pytest.importorskip('omarchy_db'); pytest.importorskip('pgvector')
from datetime import datetime

from sqlalchemy import func, select

from stratastills.db import videoimport as V
from stratastills.db.models import Analysis, File, Photo, PhotoMeta

SRT = '''1
00:00:00,000 --> 00:00:00,016
<font size="28">FrameCnt: 1, DiffTime: 16ms
2025-04-02 18:55:45.754
[iso: 340] [shutter: 1/100.0] [fnum: 1.7] [ev: -0.3] [color_md: default] [focal_len: 24.00] [latitude: 51.538685] [longitude: -0.199394] [rel_alt: 0.000 abs_alt: -19.413] [ct: 6928] </font>

2
00:00:00,016 --> 00:00:00,032
<font size="28">FrameCnt: 2, DiffTime: 16ms
2025-04-02 18:55:45.772
[iso: 400] [shutter: 1/200.0] [fnum: 1.7] [ev: -0.3] [color_md: default] [focal_len: 24.00] [latitude: 51.538686] [longitude: -0.199395] [rel_alt: 0.100 abs_alt: -19.3] [ct: 6900] </font>
'''
A, B, C = 'a' * 64, 'b' * 64, 'c' * 64
ROOT = '/data/drone'
TAGS = {'QuickTime:Duration': 0.45, 'Track1:ImageWidth': 3840, 'Track1:ImageHeight': 2160, 'Track1:VideoFrameRate': 59.94, 'Track1:CompressorID': 'hvc1', 'QuickTime:CreateDate': '2025:04:02 17:55:45'}


def recs():
    d = f'{ROOT}/20250402 - Hazelmere'
    return [dict(path=f'{d}/DJI_1.MP4', sha256=A, size=10, mtime_ns=1), dict(path=f'{d}/DJI_1.SRT', sha256=B, size=2, mtime_ns=1), dict(path=f'{d}/DJI_1.LRF', sha256=C, size=3, mtime_ns=1), dict(path=f'{d}/DJI_2.SRT', sha256='d' * 64, size=2, mtime_ns=1)]


def test_parse_srt_columns_and_units():
    s = V.parse_srt(SRT); c = s['columns']
    assert (s['frames'], s['first'], s['last']) == (2, '2025-04-02 18:55:45.754', '2025-04-02 18:55:45.772')
    assert c['iso'] == [340.0, 400.0] and c['shutter'] == [0.01, 0.005] and c['abs_alt'] == [-19.413, -19.3] and c['rel_alt'] == [0.0, 0.1] and c['color_md'] == ['default', 'default'] and V.parse_srt('nothing here') is None


def test_plan_groups_by_stem_and_takes_camera_time_from_the_srt():
    p = V.plan(recs(), [dict(sha256=A, tags=TAGS)], [ROOT], read_text=lambda path: SRT)
    assert p['counts'] == dict(photo=1, file=3, analysis=1, problems=1) and p['problems'] == ['DJI_2: no MP4, skipped']                    # DJI_2 has only an SRT
    v = p['videos'][0]; m = v['meta']
    assert [(f['role'], f['tag']) for f in v['files']] == [('video', None), ('sidecar', 'lrf'), ('sidecar', 'srt')] and m['taken_at'] == datetime(2025, 4, 2, 18, 55, 45, 754000)
    assert (m['width'], m['height'], m['iso'], m['exposure_s'], m['f_number'], m['focal_mm'], m['gps_lat'], m['gps_alt']) == (3840, 2160, 340, 0.01, 1.7, 24.0, 51.538685, -19.413) and m['dji']['frame_rate'] == 59.94 and m['exif_raw']['QuickTime:CreateDate'] == '2025:04:02 17:55:45'


def test_write_is_idempotent_and_keeps_the_telemetry(db_session):
    s = db_session; p = V.plan(recs(), [dict(sha256=A, tags=TAGS)], [ROOT], read_text=lambda path: SRT)
    assert V.write(s, p) == dict(photo=1, file=3, photo_meta=1, analysis=1) and V.write(s, p) == dict(photo=0, file=0, photo_meta=0, analysis=0)
    ph = s.scalars(select(Photo)).one(); assert (ph.kind, ph.sha256, ph.stem) == ('video', A, 'DJI_1')
    assert {(f.role, f.tag, f.sha256) for f in s.scalars(select(File))} == {('video', None, A), ('sidecar', 'srt', B), ('sidecar', 'lrf', C)} and all(f.current for f in s.scalars(select(File)))
    a = s.scalars(select(Analysis)).one(); assert (a.kind, a.input_key, a.data['frames'], a.data['columns']['iso']) == ('dji_srt', B, 2, [340.0, 400.0]) and a.file_id is not None
    assert s.scalar(select(func.count()).select_from(PhotoMeta)) == 1 and s.scalars(select(PhotoMeta)).one().dji['srt_frames'] == 2
