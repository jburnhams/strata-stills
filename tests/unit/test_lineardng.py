import pytest
import numpy as np
from stratastills import lineardng

def test_placeholder():
    assert True

def test_black_level():
    import struct
    class TagRational:
        dtype = 'RATIONAL'
        value = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    assert np.allclose(lineardng._black(TagRational, 3), [0, 0, 0])

    class TagOther:
        dtype = 3
        value = [5.0]
    assert np.allclose(lineardng._black(TagOther, 3), [5.0, 5.0, 5.0])

def test_raw_page():
    class Page:
        def __init__(self, name, shape, pages=None):
            class Photo:
                pass
            self.photometric = Photo()
            self.photometric.name = name
            self.shape = shape
            self.pages = pages or []

    class Tiff:
        def __init__(self, pages):
            self.pages = pages

    # Simple linear raw
    t1 = Tiff([Page('LINEAR_RAW', (100, 100))])
    assert lineardng.raw_page(t1) is not None

    # Larger linear raw
    p1 = Page('LINEAR_RAW', (100, 100))
    p2 = Page('LINEAR_RAW', (200, 200))
    t2 = Tiff([Page('OTHER', (10, 10), [p1, p2])])
    assert lineardng.raw_page(t2) == p2

    # No linear raw
    t3 = Tiff([Page('OTHER', (10, 10))])
    assert lineardng.raw_page(t3) is None

def test_is_linear(monkeypatch):
    import tifffile
    class Tiff:
        def __init__(self, pages):
            self.pages = pages
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class Page:
        def __init__(self, name):
            class Photo: pass
            self.photometric = Photo()
            self.photometric.name = name
            self.shape = (10, 10)

    monkeypatch.setattr(tifffile, 'TiffFile', lambda p: Tiff([Page('LINEAR_RAW')]))
    assert lineardng.is_linear('p') is True

    monkeypatch.setattr(tifffile, 'TiffFile', lambda p: Tiff([Page('OTHER')]))
    assert lineardng.is_linear('p') is False

    monkeypatch.setattr(tifffile, 'TiffFile', lambda p: (_ for _ in ()).throw(Exception()))
    assert lineardng.is_linear('p') is False

def test_map_polynomials():
    import struct
    # count = 1, opcode = 8, flags etc, length = 36 + 8 * 2 = 52
    # header = 4
    # body = 16 + 52 = 68
    blob = struct.pack('>I', 1) # count
    blob += struct.pack('>IIII', 8, 1, 0, 52) # opcode header
    blob += struct.pack('>9I', 0, 0, 100, 100, 0, 3, 1, 1, 1) # params
    blob += struct.pack('>2d', 0.5, 1.5) # coefs

    ops = lineardng.map_polynomials(blob)
    assert len(ops) == 1
    assert ops[0] == (0, 3, 0, 100, 0, 100, (0.5, 1.5))

    # test wrong opcode
    blob_wrong = struct.pack('>I', 1) + struct.pack('>IIII', 9, 1, 0, 0)
    with pytest.raises(ValueError):
        lineardng.map_polynomials(blob_wrong)

    # test wrong pitch
    blob_pitch = struct.pack('>I', 1) + struct.pack('>IIII', 8, 1, 0, 52) + struct.pack('>9I', 0, 0, 10, 10, 0, 3, 2, 1, 1) + struct.pack('>2d', 0.5, 1.5)
    with pytest.raises(ValueError):
        lineardng.map_polynomials(blob_pitch)

def test_valid_box():
    import numpy as np
    class Page:
        def __init__(self, name, shape, arr=None, pages=None):
            class Photo:
                pass
            self.photometric = Photo()
            self.photometric.name = name
            self.shape = shape
            self.arr = arr
            self.pages = pages or []
        def asarray(self): return self.arr

    class Tiff:
        def __init__(self, pages):
            self.pages = pages

    # Valid box
    mask_arr = np.zeros((100, 100), np.uint8)
    mask_arr[20:80, 20:80] = 255
    t = Tiff([Page('OTHER', (100, 100), None, pages=[
        Page('LINEAR_RAW', (100, 100)),
        Page('MASK', (100, 100), mask_arr)
    ])])
    # Tiff page 0 must be accessible
    t.pages[0].pages = [Page('LINEAR_RAW', (100, 100)), Page('MASK', (100, 100), mask_arr)]

    box = lineardng.valid_box(t, margin=2)
    assert box is not None
    assert box[0] >= 20 and box[1] <= 80 and box[2] >= 20 and box[3] <= 80

    # Empty box
    mask_empty = np.zeros((100, 100), np.uint8)
    t.pages[0].pages = [Page('LINEAR_RAW', (100, 100)), Page('MASK', (100, 100), mask_empty)]
    assert lineardng.valid_box(t) is None


def test_read_no_linear(monkeypatch):
    import tifffile
    class Tiff:
        def __init__(self, pages):
            self.pages = pages
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class Page:
        def __init__(self, name):
            class Photo: pass
            self.photometric = Photo()
            self.photometric.name = name
            self.shape = (10, 10)
            self.pages = []

    monkeypatch.setattr(tifffile, 'TiffFile', lambda p: Tiff([Page('OTHER')]))
    with pytest.raises(ValueError):
        lineardng.read('p')

def test_read_success(monkeypatch):
    import tifffile
    class Tag:
        def __init__(self, val, dtype=3):
            self.value = val
            self.dtype = dtype
    class Page:
        def __init__(self, name, shape, arr, tags, pages=None):
            class Photo: pass
            self.photometric = Photo()
            self.photometric.name = name
            self.shape = shape
            self.arr = arr
            self.tags = tags
            self.pages = pages or []
        def asarray(self): return self.arr

    class Tiff:
        def __init__(self, pages):
            self.pages = pages
        def __enter__(self): return self
        def __exit__(self, *args): pass

    arr = np.ones((100, 100, 3), np.uint16) * 1000

    # We will test max_pixels to branch `k < 1.0`
    pg = Page('LINEAR_RAW', (100, 100), arr, {
        'BlackLevel': Tag([0.0, 0.0, 0.0]),
        'WhiteLevel': Tag([65535.0, 65535.0, 65535.0])
    })

    # Empty main tags
    main = Page('MAIN', (100, 100), None, {
        'AsShotNeutral': Tag([1, 1, 1, 1, 1, 1], dtype='RATIONAL')
    }, [pg])

    monkeypatch.setattr(tifffile, 'TiffFile', lambda p: Tiff([main]))

    out = lineardng.read('p', max_pixels=50*50)
    assert out.shape == (50, 50, 3)

    # Test half=True and opcodes and box
    import struct
    blob = struct.pack('>I', 1) + struct.pack('>IIII', 8, 1, 0, 52) + struct.pack('>9I', 0, 0, 100, 100, 0, 3, 1, 1, 1) + struct.pack('>2d', 0.5, 1.5)
    pg.tags['OpcodeList2'] = Tag(blob)

    mask_arr = np.zeros((100, 100), np.uint8)
    mask_arr[20:80, 20:80] = 255
    pg.pages = [Page('MASK', (100, 100), mask_arr, {})]

    out_half = lineardng.read('p', half=True)
    # Box should be extracted around 20..80 with some margin, then halved
    assert out_half.shape[0] <= 50
