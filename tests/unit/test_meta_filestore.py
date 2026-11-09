"""The metadata file store and the choice of store (no database packages needed)."""
import os

from stratastills import meta, meta_store

XMP = (b'\xff\xd8junk<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:Description drone-dji:GimbalPitchDegree="-90.00" drone-dji:GpsLatitude="+51.538792090" '
       b'drone-dji:ProductName="Mini4 Pro" drone-dji:RelativeAltitude="+41.900"/></x:xmpmeta>')


def test_file_store_reads_the_xmp_in_the_head_of_the_file(tmp_path):
    p = tmp_path / 'a.jpg'; p.write_bytes(XMP + b'x' * 100)
    d = meta_store.FileStore().dji(str(p)); assert d == {'GpsLatitude': 51.53879209, 'RelativeAltitude': 41.9, 'GimbalPitchDegree': -90.0, 'ProductName': 'Mini4 Pro'} and meta.is_nadir(d)
    assert meta.dji_for(str(p)) == d and meta_store.FileStore().dji(str(tmp_path / 'none.jpg')) == {}


def test_a_database_url_alone_does_not_select_the_database_store(monkeypatch):
    meta_store._stores.clear(); monkeypatch.setenv('STRATASTILLS_DATABASE_URL', 'postgresql+psycopg:///x'); monkeypatch.delenv('STRATASTILLS_DB_READS', raising=False)
    assert meta_store.get().kind == 'files'
    monkeypatch.setenv('STRATASTILLS_DB_READS', 'nsfw,projects'); assert meta_store.get().kind == 'files'
    monkeypatch.delenv('STRATASTILLS_DATABASE_URL'); monkeypatch.setenv('STRATASTILLS_DB_READS', 'metadata'); assert meta_store.get().kind == 'files'
    meta_store._stores.clear()
