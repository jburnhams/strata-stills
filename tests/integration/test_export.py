import json, os, time, io
import numpy as np
import pytest

from stratastills import export as X, fullres, projects as P

def test_export_measure_detail_and_cache(project, shoot, monkeypatch):
    pid, sid = project
    nadir_dng = os.path.join(shoot, 'DJI_20250402190111_0001_D-Pano.dng')

    res1 = X.measure_detail(pid, nadir_dng)
    assert 'native' in res1 and 'recommended' in res1
    assert res1['native'] == [320, 240]

    # Verify cache was written
    from stratastills import images, detail, develop
    cache_path = images.cache_path(P.folder(pid), images.cache_key(nadir_dng, 'detail', develop.VERSION, detail.TILE), '.json')
    assert os.path.exists(cache_path)

    # Modify the cache to test that it is being read from cache
    with open(cache_path, 'w') as f:
        json.dump({'native': [100, 100], 'recommended': 100, 'from_cache': True}, f)

    res2 = X.measure_detail(pid, nadir_dng)
    assert res2['from_cache'] is True
    assert res2['native'] == [100, 100]

    # Test reading with `luma` directly instead of reading image again
    os.remove(cache_path)
    mock_luma = np.zeros((100, 200), dtype=np.float32)
    res3 = X.measure_detail(pid, nadir_dng, luma=mock_luma)
    assert res3['native'] == [200, 100]

def test_export_choose_size():
    from stratastills.projects import ProjectError

    # Test auto (without recommended) -> native size
    size, reason = X.choose_size('auto', 4000, {})
    assert size == 4000
    assert 'native' in reason

    # Test auto (with recommended) -> recommended size
    size, reason = X.choose_size('auto', 4000, {'recommended': 2000})
    assert size == 2000
    assert 'measured' in reason

    # Test native -> native size
    size, reason = X.choose_size('native', 4000, {'recommended': 2000})
    assert size == 4000
    assert 'as asked' in reason

    # Test explicit size -> explicit size
    size, reason = X.choose_size(1500, 4000, {})
    assert size == 1500
    assert 'as asked' in reason

    # Test explicit size limited by native -> native size
    size, reason = X.choose_size(5000, 4000, {})
    assert size == 4000
    assert 'limited to native' in reason

    # Test too small size -> raises ProjectError
    with pytest.raises(ProjectError, match='too small'):
        X.choose_size(63, 4000, {})

def test_export_write_picture(tmp_path):
    rgb = np.zeros((100, 100, 3), dtype=np.float32)
    rgb[0, 0] = [1.0, 0.5, 0.0]

    # Write TIFF
    tiff_path = tmp_path / "out.tiff"
    X.write_picture(str(tiff_path), rgb, fmt='tiff')
    assert tiff_path.exists()
    import tifffile
    tiff_img = tifffile.imread(str(tiff_path))
    assert tiff_img.shape == (100, 100, 3)
    assert tiff_img.dtype == np.uint16
    assert tiff_img[0, 0, 0] == 65535
    assert tiff_img[0, 0, 1] == pytest.approx(32768, abs=1)
    assert tiff_img[0, 0, 2] == 0

    # Write JPEG without EXIF
    jpeg_path = tmp_path / "out.jpg"
    X.write_picture(str(jpeg_path), rgb, fmt='jpeg', quality=85)
    assert jpeg_path.exists()
    from PIL import Image
    with Image.open(str(jpeg_path)) as img:
        assert img.format == 'JPEG'
        assert img.size == (100, 100)
        assert img.info.get('icc_profile') is not None
        assert img.getexif().get(0x0112) is None # no exif written

    # Write JPEG with dummy EXIF
    dummy_exif = b"Exif\x00\x00II*\x00\x08\x00\x00\x00\x01\x00\x12\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00"
    jpeg_exif_path = tmp_path / "out_exif.jpg"
    X.write_picture(str(jpeg_exif_path), rgb, fmt='jpeg', quality=85, exif=dummy_exif)
    assert jpeg_exif_path.exists()
    with Image.open(str(jpeg_exif_path)) as img:
        exif = img.info.get('exif')
        assert exif is not None

def test_exif_of_extraction(tmp_path):
    # Construct a dummy item with variants
    item = {
        'variants': [
            {'kind': 'generated', 'ext': '.jpg', 'path': 'does_not_matter'},
            {'kind': 'source', 'ext': '.dng', 'path': 'does_not_matter'},
        ]
    }
    assert X._exif_of(item) is None

    # Using an actual image
    import PIL.Image

    f_path = tmp_path / "test.jpg"
    img = PIL.Image.new('RGB', (10, 10))
    exif = img.getexif()
    exif[0x0112] = 3 # orientation
    img.save(str(f_path), exif=exif)

    item = {
        'variants': [
            {'kind': 'source', 'ext': '.jpg', 'path': str(f_path)}
        ]
    }
    res_exif = X._exif_of(item)
    assert res_exif is not None

    img2 = PIL.Image.new('RGB', (10, 10))
    f_path2 = tmp_path / "test_2.jpg"
    img2.save(str(f_path2), exif=res_exif)
    img3 = PIL.Image.open(str(f_path2))

    assert img3.getexif()[0x0112] == 1

def test_fullres_dims(shoot):
    nadir_dng = os.path.join(shoot, 'DJI_20250402190111_0001_D-Pano.dng')
    dims = fullres.dims(nadir_dng)
    assert dims == (320, 240)

def test_fullres_lifecycle(project, shoot):
    pid, sid = project
    nadir_dng = os.path.join(shoot, 'DJI_20250402190111_0001_D-Pano.dng')

    # Status should be idle initially
    stat = fullres.status(pid, nadir_dng)
    assert stat['state'] == 'idle'

    # Get should return None initially
    arr = fullres.get(pid, nadir_dng)
    assert arr is None

    # Prepare should kick off the process
    prep_stat = fullres.prepare(pid, nadir_dng)
    assert prep_stat['state'] in ('running', 'ready')

    # Wait until it's ready (poll with a deadline)
    deadline = time.time() + 10.0
    while time.time() < deadline:
        stat = fullres.status(pid, nadir_dng)
        if stat['state'] == 'ready':
            break
        time.sleep(0.05)
    else:
        raise AssertionError("fullres did not become ready")

    # Get should now return a numpy array (memory map)
    arr = fullres.get(pid, nadir_dng)
    assert arr is not None
    assert arr.shape == (240, 320, 3)
    assert arr.dtype == np.uint8

    # Now that it's ready, status should just say ready
    stat = fullres.status(pid, nadir_dng)
    assert stat['state'] == 'ready'
    prep_stat = fullres.prepare(pid, nadir_dng)
    assert prep_stat['state'] == 'ready'

def test_fullres_error_handling(project, tmp_path):
    pid, sid = project
    bad_file = tmp_path / "bad.jpg"
    with open(bad_file, "w") as f:
        f.write("not an image")

    # Prepare should fail when trying to decode
    fullres.prepare(pid, str(bad_file))

    deadline = time.time() + 3.0
    while time.time() < deadline:
        stat = fullres.status(pid, str(bad_file))
        if stat['state'] == 'failed':
            break
        time.sleep(0.05)
    else:
        raise AssertionError("fullres did not fail as expected")
