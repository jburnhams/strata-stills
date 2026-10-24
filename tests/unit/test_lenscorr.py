import struct

import numpy as np
import pytest

from stratastills import develop as D, lenscorr as L

N = 16
TABLES = dict(dist=[-1, 1, 5, 12, 21, 31, 45, 59, 75, 90, 105, 118, 128, 132, 130, 117], ca_r=[-640, -384, -128, 128, 256, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384], ca_b=[-256, -512, -768, -896, -1024, -1024, -1024, -1024, -896, -896, -896, -768, -768, -768, -640, 128],
              vig=[0, 368, 800, 1296, 1856, 2480, 3168, 3952, 4768, 5600, 6464, 7344, 8240, 9152, 10096, 10992])


def test_the_knots_and_the_conventions_of_the_sony_tables():
    k, d, r, b, v = L.curves(TABLES)
    assert np.allclose(k, (np.arange(N) + 0.5) / (N - 1)) and abs(d[-1] - (1 + 117 * 2.0 ** -14)) < 1e-12 and abs(r[0] - (1 - 640 * 2.0 ** -21)) < 1e-12 and abs(b[-1] - (1 + 128 * 2.0 ** -21)) < 1e-12
    assert v[0] == 1.0 and abs(v[-1] - 1 / 2 ** (0.5 - 2 ** (10992 / 8192 - 1))) < 1e-12 and 1.69 < v[-1] < 1.71 and (np.diff(v) > 0).all()                      # a gain that grows to 0.77 stop at the corner


def test_the_gain_grid_is_radial_in_the_half_diagonal_of_the_picture():
    k, _d, _r, _b, v = L.curves(TABLES); g = L.gain_grid(TABLES, 600, 400, 65)
    assert g.shape == (65, 65) and abs(g[32, 32] - 1.0) < 1e-3 and abs(g[0, 0] - np.interp(1.0, k, v)) < 1e-5 and np.allclose(g, g[::-1, ::-1], atol=1e-6)
    assert abs(g[32, 64] - np.interp(300 / np.hypot(300, 200), k, v)) < 1e-5                                                                                 # the middle of a long edge is at 0.83 of the half diagonal, not at the corner


def test_the_warp_polynomial_follows_distortion_and_colour_of_each_plane():
    c = L.warp_coefficients(TABLES); k, d, red, blue, _ = L.curves(TABLES); rr = np.linspace(0, 1, 60); dist = np.interp(rr, k, d)
    for p, want in enumerate((dist * np.interp(rr, k, red), dist, dist * np.interp(rr, k, blue))): assert np.abs(np.vander(rr * rr, 4, increasing=True) @ c[p, :4] - want).max() < 3e-4
    assert (c[:, 4:] == 0).all()


def test_the_warp_keeps_the_scale_where_asked():
    k, d, *_ = L.curves(TABLES); n = 0.55; c = L.warp_coefficients(TABLES, n)
    assert sum(c[1, i] * n ** (2 * i) for i in range(4)) == pytest.approx(1.0, abs=2e-3) and L.warp_coefficients(TABLES, None)[1, 0] != pytest.approx(c[1, 0])


def test_the_opcodes_are_read_back_by_the_development_and_correct_a_vignetted_picture():
    w, h = 300, 200; ops = D.parse_opcodes(L.opcodes(TABLES, w, h)); assert [i for i, _ in ops] == [9, 1]
    m, plane, planes, box, sp, org = D.gain_map(ops[0][1]); assert m.shape == (65, 65, 1) and (plane, planes, box) == (0, 3, (0, 0, h, w)) and abs(m[32, 32, 0] - 1) < 1e-3
    coef, centre = D.warp_params(ops[1][1]); assert coef.shape == (3, 6) and centre == (0.5, 0.5)
    yy, xx = np.mgrid[0:h, 0:w]; r = np.hypot(xx - w / 2, yy - h / 2) / np.hypot(w / 2, h / 2); k, _d, _r, _b, v = L.curves(TABLES); vign = (1 / np.interp(r, k, v)).astype(np.float32)           # a flat scene seen through the lens
    img = np.dstack([vign * 0.5] * 3); out, field = D.apply_gain_map(img.copy(), m, plane, planes); assert np.abs(out[:, :, 1] - 0.5).max() < 0.012 and field is not None                      # flat again to within the grid's accuracy


def test_a_file_without_a_valid_set_of_tables_gets_no_opcodes(tmp_path):
    p = tmp_path / 'x.arw'; p.write_bytes(b'not a tiff'); assert L.read_tables(str(p)) is None and L.from_file(str(p), 10, 10) is None


def test_the_lens_setting_picks_the_corrections_asked_for():
    on = dict(distortion=17, shading=257, ca=1); off = dict(distortion=0, shading=257, ca=1)
    assert L.parts('auto') == (True, True) and L.parts('off') == (False, False) and L.parts('shading') == (True, False) and L.parts('geometry') == (False, True)
    assert L.parts('camera', on) == (False, True) and L.parts('camera', off) == (False, False) and L.parts('camera', dict(distortion=255)) == (False, False) and L.parts('camera', None) == (False, False)           # the camera's JPEG: distortion only when the camera's setting is on, never shading
    full = L.opcodes(dict(TABLES, settings=on), 60, 40, 'auto'); assert [i for i, _ in D.parse_opcodes(full)] == [9, 1]
    assert [i for i, _ in D.parse_opcodes(L.opcodes(dict(TABLES, settings=on), 60, 40, 'camera'))] == [1] and L.opcodes(dict(TABLES, settings=off), 60, 40, 'camera') is None and [i for i, _ in D.parse_opcodes(L.opcodes(TABLES, 60, 40, 'shading'))] == [9]


def test_a_dngs_own_opcodes_can_be_cut_to_one_part():
    full = L.opcodes(TABLES, 60, 40); g = D.parse_opcodes(L.keep_opcodes(full, {9})); w = D.parse_opcodes(L.keep_opcodes(full, {1}))
    assert [i for i, _ in g] == [9] and [i for i, _ in w] == [1] and g[0][1] == D.parse_opcodes(full)[0][1] and w[0][1] == D.parse_opcodes(full)[1][1] and D.parse_opcodes(L.keep_opcodes(full, set())) == []


def test_each_part_comes_from_the_best_source(monkeypatch):
    from stratastills import lensfundb
    lf_gain = lambda r: np.full_like(r, 2.0); lf_warp = np.ones((3, 6)) * 0.5
    monkeypatch.setattr(lensfundb, 'corrections_from_exif', lambda p, w, h: (lf_gain, lf_warp)); cam = dict(TABLES, settings=dict(distortion=17, shading=257, ca=1))
    ids = lambda ops: [i for i, _ in D.parse_opcodes(ops)] if ops else []
    monkeypatch.setattr(L, 'read_tables', lambda p: cam)
    ops = D.parse_opcodes(L.from_file('x.arw', 60, 40, 'auto')); assert [i for i, _ in ops] == [9, 1]
    m = D.gain_map(ops[0][1])[0]; assert m[32, 32, 0] == pytest.approx(1.0, abs=1e-3) and m[0, 0, 0] < 1.8                                                   # the shading is the camera's table (about 1.7 in the corner), not lensfun's 2.0 everywhere
    assert D.warp_params(ops[1][1])[0][0, 0] == pytest.approx(0.5)                                                                                         # the geometry is lensfun's when it has it
    cw = D.warp_params(D.parse_opcodes(L.from_file('x.arw', 60, 40, 'camera'))[0][1])[0]; assert cw[1, 0] == pytest.approx(1 / np.interp(40 / np.hypot(60, 40), *L.curves(cam)[:2]), abs=3e-3) and cw[1, 0] != pytest.approx(0.5)      # 'camera': the table, scaled to keep the middle of the short side
    monkeypatch.setattr(lensfundb, 'corrections_from_exif', lambda p, w, h: (lf_gain, None)); assert ids(L.from_file('x.arw', 60, 40, 'auto')) == [9, 1]            # no lensfun geometry, the camera corrects: its table
    monkeypatch.setattr(L, 'read_tables', lambda p: dict(cam, settings=dict(distortion=0))); assert ids(L.from_file('x.arw', 60, 40, 'auto')) == [9, 1] and ids(L.from_file('x.arw', 60, 40, 'camera')) == []           # the camera does not correct: still the only data for a lens without a profile, but 'camera' (what the JPEG shows) has none
    monkeypatch.setattr(L, 'read_tables', lambda p: cam); monkeypatch.setattr(lensfundb, 'corrections_from_exif', lambda p, w, h: (lf_gain, lf_warp))
    assert ids(L.from_file('x.arw', 60, 40, 'camera')) == [1] and ids(L.from_file('x.arw', 60, 40, 'shading')) == [9] and ids(L.from_file('x.arw', 60, 40, 'off')) == []
    monkeypatch.setattr(L, 'read_tables', lambda p: dict(cam, settings=dict(distortion=0))); assert ids(L.from_file('x.arw', 60, 40, 'camera')) == []                # the camera's distortion setting is off: its JPEG is not corrected
    monkeypatch.setattr(L, 'read_tables', lambda p: None); ops = D.parse_opcodes(L.from_file('x.arw', 60, 40, 'auto')); assert [i for i, _ in ops] == [9, 1] and D.gain_map(ops[0][1])[0][0, 0, 0] == pytest.approx(2.0)   # no table in the file: the shading from lensfun too
    monkeypatch.setattr(lensfundb, 'corrections_from_exif', lambda p, w, h: (None, None)); assert L.from_file('x.arw', 60, 40, 'auto') is None                          # nothing describes this lens
