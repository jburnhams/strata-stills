import struct

import numpy as np

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


def test_the_opcodes_are_read_back_by_the_development_and_correct_a_vignetted_picture():
    w, h = 300, 200; ops = D.parse_opcodes(L.opcodes(TABLES, w, h)); assert [i for i, _ in ops] == [9, 1]
    m, plane, planes, box, sp, org = D.gain_map(ops[0][1]); assert m.shape == (65, 65, 1) and (plane, planes, box) == (0, 3, (0, 0, h, w)) and abs(m[32, 32, 0] - 1) < 1e-3
    coef, centre = D.warp_params(ops[1][1]); assert coef.shape == (3, 6) and centre == (0.5, 0.5)
    yy, xx = np.mgrid[0:h, 0:w]; r = np.hypot(xx - w / 2, yy - h / 2) / np.hypot(w / 2, h / 2); k, _d, _r, _b, v = L.curves(TABLES); vign = (1 / np.interp(r, k, v)).astype(np.float32)           # a flat scene seen through the lens
    img = np.dstack([vign * 0.5] * 3); out, field = D.apply_gain_map(img.copy(), m, plane, planes); assert np.abs(out[:, :, 1] - 0.5).max() < 0.012 and field is not None                      # flat again to within the grid's accuracy


def test_a_file_without_a_valid_set_of_tables_gets_no_opcodes(tmp_path):
    p = tmp_path / 'x.arw'; p.write_bytes(b'not a tiff'); assert L.read_tables(str(p)) is None and L.from_file(str(p), 10, 10) is None
