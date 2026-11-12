import struct
import numpy as np
import pytest

from stratastills import lineardng


class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class Page:
    def __init__(self, photo_name, array, tags=None, pages=None):
        self.photometric = NS(name=photo_name)
        self.shape = array.shape
        self._array = array
        self.tags = tags or {}
        self.pages = pages or []

    def asarray(self):
        return self._array


def tiff(*pages):
    return NS(pages=list(pages))


def make_opcode_blob(plane, planes, top, bottom, left, right, coef, oid=8, ver=1, flags=0, pitch=False):
    count = 1
    degree = len(coef) - 1
    nb = 36 + 8 * (degree + 1)
    if pitch:
        rp, cp = 2, 2
    else:
        rp, cp = 1, 1
    out = bytearray(4 + 16 + nb)
    struct.pack_into(">I", out, 0, count)
    struct.pack_into(">IIII", out, 4, oid, ver, flags, nb)
    struct.pack_into(">9I", out, 20, top, left, bottom, right, plane, planes, rp, cp, degree)
    struct.pack_into(f">{degree + 1}d", out, 56, *coef)
    return bytes(out)


def test_is_linear_finds_the_biggest_linear_raw_page(monkeypatch):
    t1 = Page("RGB", np.zeros((10, 10, 3)))
    t2 = Page("LINEAR_RAW", np.zeros((20, 20, 3)))
    t3 = Page("LINEAR_RAW", np.zeros((30, 30, 3)))

    tf = tiff(t1)
    tf.pages[0].pages = [t2, t3]

    import tifffile
    class TiffFile:
        def __init__(self, path):
            self.path = path
        def __enter__(self):
            return tf
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(tifffile, "TiffFile", TiffFile)

    assert lineardng.is_linear("dummy.dng")
    assert lineardng.raw_page(tf) is t3


def test_is_linear_returns_false_for_non_linear_or_missing_file():
    assert not lineardng.is_linear("missing.dng")


def test_map_polynomials_extracts_coefficients_and_bounds():
    coef = (0.1, 0.2, 0.3)
    blob = make_opcode_blob(0, 3, 10, 100, 20, 200, coef)
    ops = lineardng.map_polynomials(blob)
    assert len(ops) == 1
    plane, planes, top, bottom, left, right, parsed_coef = ops[0]
    assert plane == 0 and planes == 3 and top == 10 and bottom == 100 and left == 20 and right == 200
    assert parsed_coef == pytest.approx(coef)

    with pytest.raises(ValueError, match="not supported"):
        lineardng.map_polynomials(make_opcode_blob(0, 3, 10, 100, 20, 200, coef, oid=9))

    with pytest.raises(ValueError, match="pitch is not supported"):
        lineardng.map_polynomials(make_opcode_blob(0, 3, 10, 100, 20, 200, coef, pitch=True))


def test_valid_box_finds_the_largest_rectangle_in_the_mask():
    mask = np.zeros((200, 400), np.uint8)
    mask[20:180, 50:350] = 255
    mask[20:60, 50:90] = 0  # notch
    t = tiff(Page("RGB", np.zeros((10,10,3))))
    t.pages[0].pages = [Page("LINEAR_RAW", np.zeros((200, 400, 3), np.uint16)), Page("MASK", mask)]
    box = lineardng.valid_box(t, margin=0)
    assert box == (20, 180, 90, 350)


def test_valid_box_returns_none_when_no_mask_or_too_small():
    t = tiff(Page("RGB", np.zeros((10,10,3))))
    t.pages[0].pages = [Page("LINEAR_RAW", np.zeros((200, 400, 3), np.uint16))]
    assert lineardng.valid_box(t) is None
    mask = np.zeros((200, 400), np.uint8)
    mask[10:15, 10:15] = 255 # too small
    t.pages[0].pages.append(Page("MASK", mask))
    assert lineardng.valid_box(t) is None


def test_read_applies_processing(monkeypatch):
    import tifffile
    raw_array = np.full((100, 100, 3), 1000, dtype=np.uint16)

    class FakeTag:
        def __init__(self, value, dtype=5):
            self.value = value
            self.dtype = dtype

    black_level = FakeTag([100, 1, 200, 1, 300, 1], dtype=5) # rational
    white_level = FakeTag([2000], dtype=4)
    neutral = FakeTag([2, 1, 1, 1, 2, 1], dtype=5) # rational 2.0, 1.0, 2.0

    # 0.5 + x * 1.0
    ops = make_opcode_blob(0, 3, 10, 90, 10, 90, (0.5, 1.0))

    main = Page("RGB", np.zeros((10,10,3)), tags={"AsShotNeutral": neutral})
    raw_page = Page("LINEAR_RAW", raw_array, tags={
        "BlackLevel": black_level,
        "WhiteLevel": white_level,
        "OpcodeList2": FakeTag(ops, 7)
    })

    mask = np.zeros((100, 100), np.uint8)
    mask[10:90, 10:90] = 255
    mask_page = Page("MASK", mask)

    tf = tiff(main)
    tf.pages[0].pages = [raw_page, mask_page]

    class FakeTiffFile:
        def __init__(self, path): pass
        def __enter__(self): return tf
        def __exit__(self, *a): pass

    monkeypatch.setattr(tifffile, "TiffFile", FakeTiffFile)

    out = lineardng.read("dummy.dng")
    assert out.dtype == np.uint16
    assert out.shape == (64, 64, 3) # valid_box returns slightly different due to downsizing margin 8

    # Check that half=True works
    out_half = lineardng.read("dummy.dng", half=True)
    assert out_half.shape[0] == out.shape[0] // 2

    # Check max_pixels works
    out_max = lineardng.read("dummy.dng", max_pixels=2000)
    assert out_max.shape[0] < out.shape[0]


def test_read_without_optional_tags(monkeypatch):
    import tifffile
    raw_array = np.full((100, 100, 3), 32768, dtype=np.uint16)
    main = Page("RGB", np.zeros((10,10,3)))
    raw_page = Page("LINEAR_RAW", raw_array)
    tf = tiff(main)
    tf.pages[0].pages = [raw_page]

    class FakeTiffFile:
        def __init__(self, path): pass
        def __enter__(self): return tf
        def __exit__(self, *a): pass

    monkeypatch.setattr(tifffile, "TiffFile", FakeTiffFile)

    out = lineardng.read("dummy.dng")
    assert out.shape == (100, 100, 3)
    assert np.allclose(out, 32768)


def test_read_raises_if_not_linear_dng(monkeypatch):
    import tifffile
    tf = tiff(Page("RGB", np.zeros((10,10,3))))
    class FakeTiffFile:
        def __init__(self, path): pass
        def __enter__(self): return tf
        def __exit__(self, *a): pass

    monkeypatch.setattr(tifffile, "TiffFile", FakeTiffFile)

    with pytest.raises(ValueError, match="not a linear DNG"):
        lineardng.read("dummy.dng")
