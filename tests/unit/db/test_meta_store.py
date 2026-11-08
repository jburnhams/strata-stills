"""The DJI fields from the database against the XMP in the file: the same answer (SQLite here; `--pg` on the real thing)."""
import pytest

pytest.importorskip('omarchy_db'); pytest.importorskip('pgvector')
from omarchy_db import connect, session

from stratastills import meta, meta_store
from stratastills.db import hashes, metaimport
from stratastills.db.meta import DbStore, fields
from stratastills.db.models import File

XMP = (b'\xff\xd8junk<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:Description drone-dji:GimbalPitchDegree="-90.00" drone-dji:GpsLatitude="+51.538792090" drone-dji:GpsLongitude="-0.199639326" '
       b'drone-dji:AbsoluteAltitude="+22.143" drone-dji:ProductName="Mini4 Pro" drone-dji:GpsStatus="Normal" drone-dji:FlightYawDegree="-111.3"/></x:xmpmeta>')
TAGS = {'IFD0:Make': 'DJI', 'XMP-drone-dji:GimbalPitchDegree': -90.0, 'XMP-drone-dji:GPSLatitude': '+51.538792090', 'XMP-drone-dji:GPSLongitude': -0.199639326, 'XMP-drone-dji:AbsoluteAltitude': '+22.143',
        'XMP-drone-dji:ProductName': 'Mini4 Pro', 'XMP-drone-dji:GpsStatus': 'Normal', 'XMP-drone-dji:FlightYawDegree': -111.3, 'XMP-drone-dji:SelfData': ''}


def test_fields_turns_exiftool_names_and_types_into_what_dji_fields_gives():
    assert fields({'GPSLatitude': '+51.5', 'GimbalPitchDegree': -90.0, 'AbsoluteAltitude': '+22.143', 'ProductName': 'M', 'SelfData': '', 'FlightYawDegree': 'x'}) == {'GpsLatitude': 51.5, 'GimbalPitchDegree': -90.0, 'AbsoluteAltitude': 22.143, 'ProductName': 'M'}
    assert fields(None) == {}


def test_database_and_file_stores_agree_and_an_unknown_file_falls_back(tmp_path, db_url):
    drone = tmp_path / 'drone'; (drone / 'shoot').mkdir(parents=True); jpg = drone / 'shoot' / 'S1.JPG'; jpg.write_bytes(XMP + b'x' * 50); other = drone / 'shoot' / 'NEW.JPG'; other.write_bytes(XMP.replace(b'-90.00', b'-45.00'))
    with session(connect(db_url)) as s:
        hashes.write(s, hashes.plan([dict(path=str(drone / 'shoot' / 'S1.DNG'), sha256='a' * 64, size=1, mtime_ns=1)], [str(drone)]))
        metaimport.write(s, metaimport.plan([dict(sha256='a' * 64, tags=TAGS, path='x')]))
        ph = s.query(File).one().photo_id; s.add(File(root='drone', relpath='shoot/S1.JPG', size=1, mtime_ns=1, photo_id=ph, role='jpeg', current=True)); s.commit()           # the JPEG beside the DNG, as import_projects makes it
    db, files = DbStore(db_url, roots=[str(drone)]), meta_store.FileStore()
    assert db.dji(str(jpg)) == files.dji(str(jpg)) and meta.is_nadir(db.dji(str(jpg)))                                                                                       # same answer from either
    assert db.dji(str(other)) == files.dji(str(other)) and db.dji(str(other))['GimbalPitchDegree'] == -45.0                                                                  # not in the database: the file is read
    assert DbStore(db_url, roots=[str(tmp_path / 'else')]).dji(str(jpg)) == files.dji(str(jpg))                                                                              # a root that is not configured: the file is read
