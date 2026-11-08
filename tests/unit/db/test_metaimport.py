"""photo_meta import: columns from exiftool tags need no database; writing is checked on SQLite (`pytest --pg` runs it on Postgres)."""
import pytest

pytest.importorskip('omarchy_db'); pytest.importorskip('pgvector')
from datetime import datetime

from sqlalchemy import func, select

from stratastills.db import hashes, metaimport as M
from stratastills.db.models import PhotoMeta

A, B = 'a' * 64, 'b' * 64
SONY = {'IFD0:Make': 'SONY', 'IFD0:Model': 'NEX-5N', 'IFD0:Orientation': 6, 'ExifIFD:DateTimeOriginal': '2018:05:26 14:53:39', 'ExifIFD:ExposureTime': 0.0025, 'ExifIFD:FNumber': 8, 'ExifIFD:ISO': 100, 'ExifIFD:FocalLength': 16,
        'ExifIFD:LensModel': 'E PZ 16-50mm', 'Composite:ImageSize': '4928 3276', 'Composite:LensID': 65535, 'XMP-xmp:Rating': 3, 'System:FileSize': 5}
DJI = {'IFD0:Make': 'DJI', 'ExifIFD:DateTimeOriginal': '2025:04:02 19:01:11', 'GPS:GPSLatitude': 51.5, 'GPS:GPSLongitude': -0.12, 'GPS:GPSAltitude': 80.2, 'Composite:ImageSize': '5280x3956', 'Composite:LensID': 'x', 'XMP-drone-dji:GimbalPitchDegree': -90.0}


def test_columns_from_tags():
    c = M.columns(SONY)
    assert (c['make'], c['model'], c['lens'], c['iso'], c['exposure_s'], c['f_number'], c['focal_mm'], c['orientation'], c['width'], c['height']) == ('SONY', 'NEX-5N', 'E PZ 16-50mm', 100, 0.0025, 8.0, 16.0, 6, 4928, 3276)
    assert c['taken_at'] == datetime(2018, 5, 26, 14, 53, 39) and c['dji'] is None and c['gps_lat'] is None
    d = M.columns(DJI); assert (d['gps_lat'], d['gps_lon'], d['gps_alt'], d['width'], d['height'], d['lens'], d['dji']) == (51.5, -0.12, 80.2, 5280, 3956, 'x', {'GimbalPitchDegree': -90.0})
    assert M.columns({}) == dict.fromkeys(M.columns({}))                                                                                         # nothing known: all empty, nothing raised


def test_plan_splits_xmp_reports_duplicates_and_bad_lines(tmp_path):
    f = tmp_path / 'm.jsonl'; f.write_text('{"path": "/a", "sha256": "%s", "tags": {"IFD0:Make": "X", "XMP-xmp:Rating": 3}}\n{"path": "/b", "sha256": "%s", "tags": {}}\n{"half": ' % (A, A), encoding='utf8'); problems = []
    p = M.plan(M.read_jsonl(f, problems), problems)
    assert p['counts']['photo_meta'] == 1 and p['counts']['duplicate_photo_skipped'] == 1 and p['counts']['problems'] == 2 and p['rows'][0]['xmp_raw'] == {'XMP-xmp:Rating': 3} and p['rows'][0]['exif_raw'] == {'IFD0:Make': 'X'}
    gone = []; M.read_jsonl(tmp_path / 'nope', gone); assert gone[0].startswith('FATAL')


def test_write_is_idempotent_leaves_existing_rows_and_needs_the_photos(db_session):
    s = db_session; p = M.plan([dict(sha256=A, tags=SONY, path='/a'), dict(sha256=B, tags=DJI, path='/b')])
    with pytest.raises(M.MetaMismatch): M.write(s, p)                                                                                          # the hashes are not loaded
    hashes.write(s, hashes.plan([dict(path='/r/a.ARW', sha256=A, size=1, mtime_ns=1), dict(path='/r/b.ARW', sha256=B, size=1, mtime_ns=1)], ['/r']))
    assert M.write(s, p) == dict(photo_meta=2, already=0) and M.write(s, p) == dict(photo_meta=0, already=2)
    m = s.scalars(select(PhotoMeta).where(PhotoMeta.make == 'SONY')).one(); assert m.exif_raw['IFD0:Model'] == 'NEX-5N' and m.xmp_raw == {'XMP-xmp:Rating': 3} and s.scalar(select(func.count()).select_from(PhotoMeta)) == 2
    m.make = 'EDITED'; s.commit(); assert M.write(s, p)['photo_meta'] == 0 and s.get(PhotoMeta, m.photo_id).make == 'EDITED'                         # a rerun does not overwrite


def test_gps_is_signed_from_the_refs_or_the_composite():
    c = M.columns({'GPS:GPSLatitude': 33.9, 'GPS:GPSLatitudeRef': 'S', 'GPS:GPSLongitude': 18.4, 'GPS:GPSLongitudeRef': 'W', 'GPS:GPSAltitude': 12.5, 'GPS:GPSAltitudeRef': 1})
    assert (c['gps_lat'], c['gps_lon'], c['gps_alt']) == (-33.9, -18.4, -12.5)
    n = M.columns({'GPS:GPSLatitude': 51.5, 'GPS:GPSLatitudeRef': 'N', 'GPS:GPSLongitude': 0.12, 'GPS:GPSLongitudeRef': 'E', 'GPS:GPSAltitude': 80.2, 'GPS:GPSAltitudeRef': 0}); assert (n['gps_lat'], n['gps_lon'], n['gps_alt']) == (51.5, 0.12, 80.2)
    comp = M.columns({'Composite:GPSLatitude': -33.9, 'Composite:GPSLongitude': -18.4, 'Composite:GPSAltitude': -12.5, 'GPS:GPSLatitude': 33.9}); assert (comp['gps_lat'], comp['gps_lon'], comp['gps_alt']) == (-33.9, -18.4, -12.5)


def test_an_exiftool_error_record_makes_no_row():
    p = M.plan([dict(sha256=A, tags={'ExifTool:Error': 'File format error'}, path='/bad'), dict(sha256=B, tags=SONY, path='/ok')])
    assert p['counts']['photo_meta'] == 1 and p['counts']['problems'] == 1 and 'error' in p['problems'][0] and p['rows'][0]['sha256'] == B


def test_missing_json_is_sql_null_not_json_null(db_session):
    s = db_session; hashes.write(s, hashes.plan([dict(path='/r/a.ARW', sha256=A, size=1, mtime_ns=1)], ['/r']))
    M.write(s, M.plan([dict(sha256=A, tags={'IFD0:Make': 'X'}, path='/a')]))                                                                      # no XMP, no DJI group
    assert s.scalar(select(func.count()).select_from(PhotoMeta).where(PhotoMeta.dji.is_(None), PhotoMeta.xmp_raw.is_(None))) == 1
