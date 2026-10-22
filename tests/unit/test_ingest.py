from stratastills import ingest


def touch(d, *names):
    for n in names: (d / n).write_bytes(b'x')


def test_split_name_recognises_editor_suffixes():
    assert ingest.split_name('DJI_20250402190130_0041_D-Pano.jpg') == ('DJI_20250402190130_0041_D', 'Pano', '.jpg')
    assert ingest.split_name('DJI_20250406151728_0101_D-Enhanced-NR.dng') == ('DJI_20250406151728_0101_D', 'Enhanced-NR', '.dng')
    assert ingest.split_name('DJI_20250402190111_0037_D.DNG') == ('DJI_20250402190111_0037_D', '', '.dng')


def test_scan_pairs_raw_and_jpeg_and_attaches_derived(tmp_path):
    touch(tmp_path, 'DJI_1_0001_D.DNG', 'DJI_1_0001_D.JPG', 'DJI_1_0001_D-Pano.jpg', 'DJI_1_0002_D.JPG', 'Thumbs.db', 'notes.txt')
    shots = ingest.scan(str(tmp_path))
    assert [s.stem for s in shots] == ['DJI_1_0001_D', 'DJI_1_0002_D']
    a, b = shots
    assert a.raw.endswith('.DNG') and a.jpeg.endswith('.JPG') and a.source == a.raw
    assert [t for t, _ in a.derived] == ['Pano']
    assert b.raw is None and b.source == b.jpeg


def test_video_group_is_not_a_still(tmp_path):
    touch(tmp_path, 'DJI_1_0003_D.MP4', 'DJI_1_0003_D.LRF', 'DJI_1_0003_D.SRT')
    (s,) = ingest.scan(str(tmp_path))
    assert s.kind == 'video' and s.source is None and len(s.sidecars) == 2


def test_edited_file_without_original_is_listed_but_has_no_source(tmp_path):
    touch(tmp_path, 'IMG_9-Pano.jpg')
    (s,) = ingest.scan(str(tmp_path))
    assert s.source is None and s.derived
