from stratastills import ingest


def touch(d, *names):
    for n in names: (d / n).write_bytes(b'x')


def test_split_name_recognises_editor_suffixes():
    assert ingest.split_name('DJI_20250402190130_0041_D-Pano.jpg') == ('DJI_20250402190130_0041_D', 'Pano', '.jpg')
    assert ingest.split_name('DJI_20250406151728_0101_D-Enhanced-NR.dng') == ('DJI_20250406151728_0101_D', 'Enhanced-NR', '.dng')
    assert ingest.split_name('DJI_20250402190111_0037_D.DNG') == ('DJI_20250402190111_0037_D', '', '.dng')


def test_scan_pairs_raw_and_jpeg_and_attaches_enhanced_variants(tmp_path):
    touch(tmp_path, 'DJI_1_0001_D.DNG', 'DJI_1_0001_D.JPG', 'DJI_1_0001_D-Enhanced-NR.dng', 'DJI_1_0002_D.JPG', 'Thumbs.db', 'notes.txt')
    a, b = ingest.scan(str(tmp_path))
    assert [a.stem, b.stem] == ['DJI_1_0001_D', 'DJI_1_0002_D']
    assert a.raw.endswith('.DNG') and a.jpeg.endswith('.JPG') and a.source == a.raw
    assert [v['id'] for v in a.variants()] == ['raw', 'jpeg', 'enhanced-nr.dng'] and a.variants()[2]['kind'] == 'derived'
    assert b.raw is None and b.source == b.jpeg and [v['id'] for v in b.variants()] == ['jpeg']


def test_panorama_is_its_own_item_with_a_parent(tmp_path):
    touch(tmp_path, 'DJI_1_0001_D.DNG', 'DJI_1_0001_D-Pano.jpg', 'DJI_1_0001_D-Pano.dng')
    shot, pano = ingest.scan(str(tmp_path))
    assert shot.kind == 'still' and shot.derived == [] and pano.kind == 'pano' and pano.parent == 'DJI_1_0001_D' and pano.stem == 'DJI_1_0001_D-Pano'
    assert pano.raw.endswith('.dng') and pano.jpeg.endswith('.jpg')


def test_video_group_is_not_a_still(tmp_path):
    touch(tmp_path, 'DJI_1_0003_D.MP4', 'DJI_1_0003_D.LRF', 'DJI_1_0003_D.SRT')
    (s,) = ingest.scan(str(tmp_path))
    assert s.kind == 'video' and s.source is None and len(s.sidecars) == 2 and s.variants() == []


def test_edited_file_without_original_is_an_item_with_a_derived_variant(tmp_path):
    touch(tmp_path, 'IMG_9-Enhanced.jpg')
    (s,) = ingest.scan(str(tmp_path))
    assert s.kind == 'still' and s.source.endswith('.jpg') and [v['kind'] for v in s.variants()] == ['derived']
