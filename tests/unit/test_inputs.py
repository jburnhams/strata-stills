import struct
from types import SimpleNamespace as NS

import numpy as np
import pytest

from stratastills import inputs, lineardng


class Page:
    def __init__(self, kind, arr, tags=None): self.photometric = NS(name=kind); self._a = arr; self.shape = arr.shape; self.tags = tags or {}
    def asarray(self): return self._a


def tiff(*subs): return NS(pages=[NS(pages=list(subs), photometric=NS(name='RGB'))])


def poly_op(plane, coef, box=(0, 0, 10, 10)):
    body = struct.pack('>9I', *box, plane, 1, 1, 1, len(coef) - 1) + struct.pack(f'>{len(coef)}d', *coef); return struct.pack('>IIII', 8, 0, 0, len(body)) + body


def test_black_level_rationals_and_plain_numbers():
    rat = NS(value=(1533952, 256, 0, 256, 0, 256), dtype=5); assert np.allclose(lineardng._black(rat, 3), [5992, 0, 0])
    plain = NS(value=(4100, 4100, 4100), dtype=3); assert np.allclose(lineardng._black(plain, 3), 4100) and np.allclose(lineardng._black(NS(value=(64,), dtype=3), 3), 64)


def test_map_polynomial_opcodes_are_parsed_and_others_refused():
    blob = struct.pack('>I', 2) + poly_op(0, [0.0, 0.0625, 0.0, 0.9375]) + poly_op(2, [0.1, 0.9])
    ops = lineardng.map_polynomials(blob); assert [o[0] for o in ops] == [0, 2] and ops[0][-1] == (0.0, 0.0625, 0.0, 0.9375) and ops[1][2:6] == (0, 10, 0, 10)
    with pytest.raises(ValueError): lineardng.map_polynomials(struct.pack('>I', 1) + struct.pack('>IIII', 9, 0, 0, 0))


def test_the_biggest_valid_rectangle_of_a_stitch_mask():
    mask = np.zeros((200, 400), np.uint8); mask[20:180, 50:350] = 255; mask[20:60, 50:90] = 0                                  # a notch cut out of one corner
    t = tiff(Page('LINEAR_RAW', np.zeros((200, 400, 3), np.uint16)), Page('MASK', mask)); top, bottom, left, right = lineardng.valid_box(t, margin=0)
    assert (top, bottom, left, right) == (60, 180, 50, 350) or (bottom - top) * (right - left) >= 0.9 * 300 * 120 and top >= 20 and bottom <= 180 and left >= 50 and right <= 350
    assert lineardng.valid_box(tiff(Page('LINEAR_RAW', np.zeros((200, 400, 3), np.uint16))), margin=0) is None                       # no mask: nothing to crop


def test_a_device_profile_supplies_the_tone_curve_only_when_the_file_has_none():
    own = [0.0, 0.0, 1.0, 1.0]; assert inputs.tone_curve(dict(ProfileToneCurve=own, Make=b'DJI', Model=b'FC8482')) == own
    c = inputs.tone_curve(dict(Make=b'DJI\x00', Model='FC8482')); assert c is not None and len(c) == 256 and c[-1] == 1.0
    assert inputs.tone_curve(dict(Make='Nobody', Model='X1')) is None and inputs.tone_curve({}) is None


def test_lightroom_adobe_standard_files_get_the_cameras_own_colour_profile_back():
    adobe = dict(ProfileName='Adobe Standard', Make='DJI', Model='FC8482', ColorMatrix1=(1, 2), ProfileHueSatMapDims=(90, 30, 1), AsShotNeutral=(4, 10, 1, 1, 6, 10)); out = inputs.adjust_tags(dict(adobe))
    assert out['ProfileHueSatMapDims'] == (36, 10, 1) and len(out['ProfileToneCurve']) == 256 and len(out['ColorMatrix1']) == 18 and out['AsShotNeutral'] == adobe['AsShotNeutral']       # the white balance stays the file's own
    own = dict(ProfileName='Embedded', Make='DJI', Model='FC8482', ColorMatrix1=(1, 2)); assert inputs.adjust_tags(dict(own)) == own
    other = dict(adobe, Make='Nobody'); assert inputs.adjust_tags(dict(other)) == other                                                          # no device profile: left alone


def test_describe_tells_the_kind_of_file(tmp_path):
    p = tmp_path / 'a.jpg'; p.write_bytes(b'x'); assert inputs.describe(str(p)) == dict(kind='rendered', make=None, model=None, profile=None)
    assert inputs.profile('DJI', 'FC8482')['label'] == 'DJI Mini 4 Pro' and inputs.profile('x', 'y') == {}
