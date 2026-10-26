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
    own = dict(ProfileName='Embedded', Make='DJI', Model='FC8482', ColorMatrix1=(1, 2)); got = inputs.adjust_tags(dict(own)); assert {k: v for k, v in got.items() if k != 'LookTable'} == own and 'LookTable' in got           # the device's look table is added, nothing else changes
    other = dict(adobe, Make='Nobody'); assert inputs.adjust_tags(dict(other)) == other                                                          # no device profile: left alone


def test_describe_tells_the_kind_of_file(tmp_path):
    p = tmp_path / 'a.jpg'; p.write_bytes(b'x'); assert inputs.describe(str(p)) == dict(kind='rendered', make=None, model=None, profile=None)
    assert inputs.profile('DJI', 'FC8482')['label'] == 'DJI Mini 4 Pro' and inputs.profile('x', 'y') == {}


def fake_raw(monkeypatch, wb=(2000.0, 1024.0, 1900.0, 1024.0)):
    import contextlib, sys, types
    r = NS(rgb_xyz_matrix=np.array([[0.7, -0.2, -0.05], [-0.5, 1.3, 0.25], [-0.1, 0.18, 0.65], [0, 0, 0]]), camera_whitebalance=list(wb), sizes=NS(crop_width=60, crop_height=40, iwidth=62, iheight=42, flip=0))
    mod = types.ModuleType('rawpy'); mod.imread = lambda path: contextlib.nullcontext(r); monkeypatch.setitem(sys.modules, 'rawpy', mod)


def test_noise_profile_is_interpolated_in_log_iso_and_held_beyond_the_ends():
    t = [[100, 1e-6, 1e-10], [400, 4e-6, 4e-10]]
    assert inputs.noise_profile(t, 200) == pytest.approx((2e-6, 2e-10)) and inputs.noise_profile(t, 50) == pytest.approx((1e-6, 1e-10)) and inputs.noise_profile(t, 6400) == pytest.approx((4e-6, 4e-10))


def test_a_non_dng_raw_gets_its_colour_from_the_device_profile_and_its_noise_for_the_iso(monkeypatch):
    fake_raw(monkeypatch); monkeypatch.setattr(inputs, '_iso', lambda path: 320.0)
    t = inputs.synthesize_tags('x.arw', dict(Make=b'SONY\x00', Model='ILCE-7C'))
    assert len(t['ColorMatrix1']) == 18 and len(t['ColorMatrix2']) == 18 and t['CalibrationIlluminant1'] == 17 and len(t['ProfileToneCurve']) % 2 == 0
    assert 'ProfileHueSatMapDims' not in t and t['NoiseProfile'][0] > 0 and t['NoiseProfile'][1] > 0                                           # Adobe's hue/saturation map is a look, left out
    assert [a / b for a, b in zip(t['AsShotNeutral'][0::2], t['AsShotNeutral'][1::2])] == pytest.approx([1024 / 2000, 1.0, 1024 / 1900])


def test_a_non_dng_raw_of_an_unknown_camera_gets_libraws_matrix(monkeypatch):
    fake_raw(monkeypatch); t = inputs.synthesize_tags('x.arw', dict(Make='Nobody', Model='X1'))
    assert t['ColorMatrix1'] == t['ColorMatrix2'] and t['ProfileName'] == 'LibRaw' and 'NoiseProfile' not in t and 'ProfileToneCurve' not in t


def test_files_that_have_colour_tags_or_are_dng_are_left_alone(monkeypatch):
    fake_raw(monkeypatch); own = dict(Make='SONY', Model='ILCE-7C', ColorMatrix1=(1, 2)); assert inputs.synthesize_tags('x.arw', dict(own)) == own
    plain = dict(Make='SONY', Model='ILCE-7C'); assert inputs.synthesize_tags('x.dng', dict(plain)) == plain


def test_a_device_looktable_reaches_every_kind_of_file_of_that_device(monkeypatch):
    spec = dict(hue=24, data=[0.0, 1.0] * (24 * 5)); monkeypatch.setitem(inputs.PROFILES, ('acme', 'x1'), dict(label='Acme', bundle='acme_profile.json')); monkeypatch.setitem(inputs._cache, 'acme_profile.json', dict(LookTable=spec, ColorMatrix1=[1, 1]))
    t = inputs.adjust_tags(dict(Make='ACME', Model='X1', ColorMatrix1=(5, 1))); assert t['LookTable'] == spec and t['ColorMatrix1'] == (5, 1)                                  # a file of the device that has its own colour tags keeps them
    t = inputs.adjust_tags(dict(Make='ACME', Model='X1', ProfileName='Adobe Standard', ColorMatrix1=(5, 1))); assert t['LookTable'] == spec and t['ColorMatrix1'] == (1, 1) and t['ProfileName'] == 'Embedded'
    assert 'LookTable' not in inputs.adjust_tags(dict(Make='Nobody', Model='Y'))


def test_a_devices_noise_scale_multiplies_the_noise_profile_of_any_file_of_it(monkeypatch):
    monkeypatch.setitem(inputs.PROFILES, ('acme', 'x1'), dict(label='Acme', bundle='acme_profile.json')); monkeypatch.setitem(inputs._cache, 'acme_profile.json', dict(NoiseScale=8.0))
    assert inputs.adjust_tags(dict(Make='ACME', Model='X1', NoiseProfile=(0.001, 1e-6)))['NoiseProfile'] == pytest.approx((0.008, 8e-6)) and 'NoiseProfile' not in inputs.adjust_tags(dict(Make='ACME', Model='X1'))
    monkeypatch.setitem(inputs._cache, 'acme_profile.json', dict()); assert inputs.adjust_tags(dict(Make='ACME', Model='X1', NoiseProfile=(0.001, 1e-6)))['NoiseProfile'] == (0.001, 1e-6)


def test_noise_scale_can_be_a_table_over_the_iso():
    assert inputs.noise_scale(8.0, 400) == 8.0 and inputs.noise_scale(None, 400) is None
    t = [[100, 4.0], [400, 16.0], [1600, 64.0]]
    assert inputs.noise_scale(t, 100) == pytest.approx(4.0) and inputs.noise_scale(t, 200) == pytest.approx(8.0) and inputs.noise_scale(t, 6400) == pytest.approx(64.0) and inputs.noise_scale(t, 50) == pytest.approx(4.0) and inputs.noise_scale(t, None) == pytest.approx(16.0)


def test_the_nex5n_profile_carries_what_add_device_fits():
    p = inputs.profile('SONY', 'NEX-5N'); b = inputs._bundle(p['bundle'])
    assert p['label'] == 'Sony NEX-5N' and {'ColorMatrix1', 'NoiseTable', 'ProfileToneCurve', 'LookTable', 'NoiseScale'} <= set(b)
    assert all(1 <= s <= 32 for _, s in b['NoiseScale'])                                                                                      # inside the scales the fit tried


def test_the_other_raw_programs_are_found_or_named_as_missing(monkeypatch, tmp_path):
    import pytest
    from stratastills import externaltools as ET
    monkeypatch.setattr(ET.shutil, 'which', lambda n: None); monkeypatch.setattr(ET, 'EXTRA', {'darktable': [], 'rawtherapee': []})
    assert ET.missing() == ['darktable', 'rawtherapee']
    with pytest.raises(SystemExit) as e: ET.require()
    assert 'darktable' in str(e.value) and 'install' in str(e.value).lower() or 'pacman' in str(e.value) or 'brew' in str(e.value) or 'winget' in str(e.value)
    monkeypatch.setattr(ET.shutil, 'which', lambda n: str(tmp_path / n)); monkeypatch.setattr(ET, 'rt_profiles', lambda: str(tmp_path)); assert ET.missing() == [] and ET.require() is None
