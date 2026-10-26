"""images.py: reading pictures Pillow cannot (16-bit RGB TIFFs, demosaiced Lightroom DNGs), the error paths, and the header-only size of a file."""
import cv2, numpy as np, pytest, tifffile

from stratastills import images


def _rgb16(h, w, v): return np.full((h, w, 3), v, np.uint16)


def test_a_16_bit_rgb_tiff_is_read_with_tifffile_to_8_bit_bgr_and_fitted(tmp_path):
    p = str(tmp_path / 'a.tif'); a = _rgb16(60, 80, 0); a[..., 0] = 65535                               # pure red, in RGB order
    tifffile.imwrite(p, a)
    im = images.load(p, 40)
    assert im.dtype == np.uint8 and im.shape == (30, 40, 3)                                              # the long side fitted to 40, the aspect kept
    assert im[0, 0].tolist() == [0, 0, 255]                                                              # red, as BGR


def test_a_grey_tiff_becomes_three_equal_planes_and_a_picture_already_small_enough_is_not_resized(tmp_path):
    p = str(tmp_path / 'g.tif'); tifffile.imwrite(p, np.full((10, 20), 51, np.uint8))
    im = images._read_tiff(p, 100)
    assert im.shape == (10, 20, 3) and (im == 51).all()


def test_a_linear_dng_uses_the_smallest_level_that_is_big_enough_else_the_biggest(tmp_path):
    p = str(tmp_path / 'lin.dng'); tifffile.imwrite(p, _rgb16(64, 96, 32768)); tifffile.imwrite(p, _rgb16(32, 48, 32768), append=True)
    small = images._read_linear_dng(p, 40)                                                               # the 48-wide level is enough: no need to read the big one
    assert small.shape == (27, 40, 3) and small[0, 0].tolist() == [128, 128, 128]                        # 32768/65535 of 255, rounded
    assert images._read_linear_dng(p, 500).shape == (64, 96, 3)                                          # nothing is big enough: the biggest, not enlarged


def test_a_dng_without_a_16_bit_rgb_level_says_so(tmp_path):
    p = str(tmp_path / 'x.dng'); tifffile.imwrite(p, np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(images.ImageError, match='no 16-bit RGB level'): images._read_linear_dng(p, 10)


def test_a_raw_is_developed_half_size_and_returned_as_bgr(monkeypatch, tmp_path):
    seen = {}
    def fake(path, **kw): seen.update(kw); return np.dstack([np.ones((10, 20)), np.zeros((10, 20)), np.zeros((10, 20))])        # a red picture, RGB floats
    monkeypatch.setattr('stratastills.develop.develop', fake)
    im = images._read_raw(str(tmp_path / 'r.arw'), 100)
    assert seen == dict(half=True, denoise='chroma') and im[0, 0].tolist() == [0, 0, 255]


@pytest.mark.parametrize('name', ['bad.dng', 'bad.arw'])
def test_a_raw_that_neither_route_can_read_names_the_file_and_both_reasons(tmp_path, name):
    p = tmp_path / name; p.write_bytes(b'not a picture at all')
    with pytest.raises(images.ImageError, match=f'cannot read {name}: .*; '): images.load(str(p), 100)


@pytest.mark.parametrize('name', ['bad.tif', 'bad.jpg'])
def test_a_broken_ordinary_picture_is_an_image_error_even_after_the_tiff_retry(tmp_path, name):
    p = tmp_path / name; p.write_bytes(b'not a picture at all')
    with pytest.raises(images.ImageError, match=f'cannot read {name}'): images.load(str(p), 100)


def test_the_size_of_a_dng_is_its_biggest_level_and_is_remembered_per_file_state(tmp_path):
    p = str(tmp_path / 'm.dng')
    with tifffile.TiffWriter(p) as tw: tw.write(_rgb16(10, 20, 1), subifds=1); tw.write(_rgb16(40, 30, 1), subfiletype=1)                  # a small main IFD (as in a camera DNG) with the real picture in a sub-IFD
    assert images.dims(p) == (30, 40) and images.dims(p) == (30, 40)                                     # (width, height) of the biggest level; the second answer is from the memo
    with pytest.raises(OSError): images.dims(str(tmp_path / 'missing.dng'))                              # no stat: straight to reading, which fails


def test_the_size_of_an_ordinary_picture_comes_from_pillow(tmp_path):
    p = str(tmp_path / 'p.png'); cv2.imwrite(p, np.zeros((7, 11, 3), np.uint8))
    assert images.dims(p) == (11, 7)
