import datetime as dt
from stratastills import meta

XMP = b'junk<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:Description xmp:CreateDate="2025-04-02T19:01:11+01:00" drone-dji:GimbalPitchDegree="-90.00" drone-dji:RelativeAltitude="+41.900" drone-dji:ProductName="Mini4 Pro" drone-dji:GimbalYawDegree="bad"/></x:xmpmeta>junk'


def test_read_xmp_and_dji_fields(tmp_path):
    p = tmp_path / 'a.jpg'; p.write_bytes(XMP)
    d = meta.dji_fields(meta.read_xmp(str(p)))
    assert d == dict(GimbalPitchDegree=-90.0, RelativeAltitude=41.9, ProductName='Mini4 Pro')
    assert meta.is_nadir(d) and not meta.is_nadir(dict(GimbalPitchDegree=-20.0)) and not meta.is_nadir({})


def test_read_xmp_missing_file_or_packet(tmp_path):
    assert meta.read_xmp(str(tmp_path / 'nope.jpg')) == {}
    (tmp_path / 'b.jpg').write_bytes(b'no packet'); assert meta.read_xmp(str(tmp_path / 'b.jpg')) == {}


def test_parse_time_formats():
    assert meta.parse_time('2025-04-02T19:01:30.1+01:00').hour == 19
    assert meta.parse_time('2025:04:02 19:01:11') == dt.datetime(2025, 4, 2, 19, 1, 11)
    assert meta.parse_time('nonsense') is None and meta.parse_time(None) is None


def test_photo_info_from_a_real_jpeg(tmp_path):
    from PIL import Image
    p = tmp_path / 'a.jpg'; Image.new('RGB', (64, 48)).save(p)
    i = meta.photo_info(str(p)); assert (i['width'], i['height']) == (64, 48) and i['dji'] == {}
