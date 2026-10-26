"""Tests for ingest.py over a real folder structure."""
import os
from stratastills import ingest

def test_ingest_scan_shoot(shoot):
    shots = ingest.scan(shoot)
    assert len(shots) == 4

    by = {s.stem: s for s in shots}

    assert by['DJI_20250402190111_0001_D'].kind == "still"
    assert by['DJI_20250402190111_0001_D'].jpeg.endswith('DJI_20250402190111_0001_D.JPG')

    assert by['DJI_20250402190111_0001_D-Pano'].kind == "pano"
    assert by['DJI_20250402190111_0001_D-Pano'].parent == 'DJI_20250402190111_0001_D'
    assert by['DJI_20250402190111_0001_D-Pano'].raw.endswith('DJI_20250402190111_0001_D-Pano.dng')

    assert by['DJI_20250402190211_0002_D'].kind == "still"

    assert by['DJI_20250402190511_0009_D'].kind == "video"
    assert by['DJI_20250402190511_0009_D'].video.endswith('DJI_20250402190511_0009_D.MP4')


def test_ingest_scan_empty_folder(tmp_path):
    assert ingest.scan(str(tmp_path)) == []


def test_ingest_scan_ignored_and_unsupported_files(tmp_path):
    # .hidden, thumbs.db, foo.txt
    (tmp_path / ".hidden").touch()
    (tmp_path / "thumbs.db").touch()
    (tmp_path / "Thumbs.DB").touch() # test case insensitivity
    (tmp_path / "desktop.ini").touch()
    (tmp_path / "foo.txt").touch()
    assert ingest.scan(str(tmp_path)) == []


def test_ingest_scan_sidecars_and_videos(tmp_path):
    (tmp_path / "DJI_0001.DNG").touch()
    (tmp_path / "DJI_0001.XMP").touch()
    (tmp_path / "DJI_0002.MP4").touch()

    shots = ingest.scan(str(tmp_path))
    assert len(shots) == 2
    assert shots[0].stem == "DJI_0001"
    assert shots[0].raw == str(tmp_path / "DJI_0001.DNG")
    assert shots[0].sidecars == [str(tmp_path / "DJI_0001.XMP")]
    assert shots[0].kind == "still"

    assert shots[1].stem == "DJI_0002"
    assert shots[1].video == str(tmp_path / "DJI_0002.MP4")
    assert shots[1].kind == "video"
    assert shots[1].source is None


def test_ingest_scan_derived_files(tmp_path):
    (tmp_path / "DJI_0001.JPG").touch()
    (tmp_path / "DJI_0001-Edit.jpg").touch()
    (tmp_path / "DJI_0001-Edit.mp4").touch() # unsupported derived type
    (tmp_path / "Orphan-Edit.jpg").touch() # derived file without parent

    shots = ingest.scan(str(tmp_path))
    assert len(shots) == 2

    s0 = next(s for s in shots if s.stem == "DJI_0001")
    assert s0.derived == [("Edit", str(tmp_path / "DJI_0001-Edit.jpg"))]
    assert len(s0.variants()) == 2 # jpeg, derived

    s1 = next(s for s in shots if s.stem == "Orphan")
    assert s1.derived == [("Edit", str(tmp_path / "Orphan-Edit.jpg"))]
    assert s1.source == str(tmp_path / "Orphan-Edit.jpg")
    assert s1.kind == "still"


def test_ingest_scan_pano(tmp_path):
    (tmp_path / "DJI_0001.JPG").touch()
    (tmp_path / "DJI_0001-Pano.jpg").touch() # Pano jpeg
    (tmp_path / "DJI_0002.JPG").touch()
    (tmp_path / "DJI_0002-Pano.dng").touch() # Pano raw

    shots = ingest.scan(str(tmp_path))
    assert len(shots) == 4

    pano1 = next(s for s in shots if s.stem == "DJI_0001-Pano")
    assert pano1.pano is True
    assert pano1.parent == "DJI_0001"
    assert pano1.jpeg == str(tmp_path / "DJI_0001-Pano.jpg")
    assert pano1.kind == "pano"

    pano2 = next(s for s in shots if s.stem == "DJI_0002-Pano")
    assert pano2.pano is True
    assert pano2.parent == "DJI_0002"
    assert pano2.raw == str(tmp_path / "DJI_0002-Pano.dng")
    assert pano2.kind == "pano"


def test_ingest_scan_recursive(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    (d / "DJI_0001.JPG").touch()

    shots = ingest.scan(str(tmp_path), recursive=True)
    assert len(shots) == 1
    assert shots[0].stem == "DJI_0001"
    assert shots[0].jpeg == str(d / "DJI_0001.JPG")


def test_to_json(tmp_path):
    (tmp_path / "DJI_0001.JPG").touch()
    shots = ingest.scan(str(tmp_path))
    assert len(shots) == 1
    assert shots[0].to_json()["stem"] == "DJI_0001"
