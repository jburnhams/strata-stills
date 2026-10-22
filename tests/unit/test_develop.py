import struct

import cv2, numpy as np, pytest
from stratastills import develop as D


def opcode_list(*ops):
    b = struct.pack('>I', len(ops))
    for oid, p in ops: b += struct.pack('>IIII', oid, 0x01030000, 0, len(p)) + p
    return b


def gain_param(m, plane=0, planes=3):
    nv, nh, mp = m.shape
    return struct.pack('>10I', 0, 0, 100, 100, plane, planes, 1, 1, nv, nh) + struct.pack('>4d', 1 / max(nv - 1, 1), 1 / max(nh - 1, 1), 0.0, 0.0) + struct.pack('>I', mp) + m.astype('>f4').tobytes()


def warp_param(coefs, centre=(0.5, 0.5)):
    return struct.pack('>I', len(coefs)) + b''.join(struct.pack('>6d', *c) for c in coefs) + struct.pack('>2d', *centre)


def test_opcode_lists_parse_back():
    m = np.ones((4, 4, 3), np.float32) * 2; ops = D.parse_opcodes(opcode_list((9, gain_param(m)), (1, warp_param([(1, 0, 0, 0, 0, 0)] * 3))))
    assert [o for o, _ in ops] == [9, 1]
    gm, plane, planes, box, sp, org = D.gain_map(ops[0][1]); assert gm.shape == (4, 4, 3) and float(gm.min()) == 2.0 and planes == 3 and box == (0, 0, 100, 100)
    coef, centre = D.warp_params(ops[1][1]); assert coef.shape == (3, 6) and centre == (0.5, 0.5)


def test_gain_map_lifts_the_corners_and_returns_the_field():
    m = np.ones((5, 5, 3), np.float32); m[0, 0] = 4.0
    img, field = D.apply_gain_map(np.ones((50, 60, 3), np.float32), m, 0, 3)
    assert img[0, 0, 0] == pytest.approx(4.0) and img[25, 30, 0] == pytest.approx(1.0, abs=0.05) and field.shape == (50, 60)


def test_warp_identity_leaves_the_picture_and_a_small_k_moves_the_edges():
    rng = np.random.default_rng(0); img = cv2.GaussianBlur(rng.random((120, 160, 3)).astype(np.float32), (0, 0), 2)
    same = D.warp_rectilinear(img, np.array([[1, 0, 0, 0, 0, 0]] * 3, float), (0.5, 0.5)); assert np.abs(same - img).max() < 1e-3
    moved = D.warp_rectilinear(img, np.array([[1.0, 0.1, 0, 0, 0, 0]] * 3, float), (0.5, 0.5)); assert np.abs(moved - img).mean() > 0.005 and np.allclose(moved[60, 80], img[60, 80], atol=1e-2)
    ca = D.warp_rectilinear(img, np.array([[1.0, 0.05, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1.0, -0.05, 0, 0, 0, 0]], float), (0.5, 0.5)); assert np.abs(ca[:, :, 0] - ca[:, :, 2]).mean() > np.abs(img[:, :, 0] - img[:, :, 2]).mean()


def test_colour_matrix_keeps_neutral_neutral_whatever_the_white_balance():
    cm1 = np.array([[1.2, -0.4, -0.1], [-0.3, 1.1, 0.2], [0.0, 0.1, 0.7]]); cm2 = np.array([[0.9, -0.2, -0.1], [-0.4, 1.2, 0.2], [0.0, 0.1, 0.8]])
    for neutral in (np.array([0.42, 1.0, 0.64]), np.array([0.6, 1.0, 0.45])):
        m = D.camera_to_srgb(D.color_matrix(cm1, cm2, neutral), neutral); assert m @ np.ones(3) == pytest.approx([1, 1, 1], abs=1e-9) and np.isfinite(m).all()


def test_srgb_curves_round_trip():
    x = np.linspace(0, 1, 11).astype(np.float32); assert np.allclose(D.srgb_eotf(D.srgb_oetf(x)), x, atol=1e-4)


def test_shoulder_compresses_highlights_but_keeps_hue_and_shadows():
    x = np.array([[[0.1, 0.2, 0.3], [2.0, 1.0, 0.5]]], np.float32); y = D.shoulder(x)
    assert np.allclose(y[0, 0], x[0, 0]) and y.max() < 1.0 and y[0, 1, 0] / y[0, 1, 1] == pytest.approx(2.0, rel=1e-4)
    assert D.shoulder(np.array([[[D.WHITE] * 3]], np.float32)).max() == pytest.approx(1.0, abs=1e-5)


def test_auto_ev_brings_a_dark_picture_to_the_target_median():
    rng = np.random.default_rng(1); img = (rng.random((80, 80, 3)) * 0.04 + 0.01).astype(np.float32); ev = D.auto_ev(img, None)
    out = D.finish(img * 2.0 ** ev, None); assert 0.37 < float(np.median(out @ np.array([0.2126, 0.7152, 0.0722]))) < 0.47 and 1 < ev < D.MAX_EV
    assert D.auto_ev(np.full((40, 40, 3), 0.9, np.float32), None) < -0.99           # an already bright picture is darkened, but only so far


def test_local_tone_lifts_shadows_and_keeps_detail():
    img = np.full((200, 300, 3), 0.5, np.float32); img[:, :150] = 0.02; rng = np.random.default_rng(2); img *= (1 + 0.1 * rng.standard_normal((200, 300, 1))).astype(np.float32)
    out = D.local_tone(img, 0.5); assert out[:, :140].mean() > img[:, :140].mean() * 1.4 and out[:, 160:].mean() < img[:, 160:].mean()
    assert D.local_tone(img, 0) is img and np.std(out[:, :140, 0]) / out[:, :140, 0].mean() == pytest.approx(np.std(img[:, :140, 0]) / img[:, :140, 0].mean(), rel=0.3)


def test_denoise_never_smooths_luminance_in_the_middle_but_cleans_colour():
    rng = np.random.default_rng(3); tex = (0.2 + 0.05 * cv2.GaussianBlur(rng.standard_normal((120, 160)).astype(np.float32), (0, 0), 1.2))[..., None] * np.ones(3, np.float32)
    assert D.denoise_linear(tex, None) is tex
    out = D.denoise_linear(tex.copy(), (0.02, 1e-4)); assert np.abs(out @ np.array([0.2126, 0.7152, 0.0722], np.float32) - tex @ np.array([0.2126, 0.7152, 0.0722], np.float32)).max() < 1e-4          # luminance (the texture) is untouched
    chroma = tex + rng.normal(0, 0.02, tex.shape).astype(np.float32) * np.array([1, -1, 0], np.float32); cl = D.denoise_linear(np.clip(chroma, 0, None), (0.02, 1e-4))
    chroma_of = lambda a: a - (a @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    assert chroma_of(cl).std() < chroma_of(chroma).std() * 0.7                                                                                              # colour noise is smoothed
    assert D.visible_noise(tex, (0.02, 1e-4)) > D.visible_noise(tex, (1e-5, 1e-9))
    forced = D.denoise_linear(np.clip(tex + rng.normal(0, 0.03, tex.shape).astype(np.float32), 0, None), (0.02, 1e-4), 4.0); assert forced.std() < tex.std() * 2


def test_denoise_treats_only_the_lifted_corners_when_given_the_gain_field():
    rng = np.random.default_rng(4); img = np.full((120, 160, 3), 0.2, np.float32) + rng.normal(0, 0.02, (120, 160, 3)).astype(np.float32); gain = np.ones((120, 160), np.float32); gain[:, 120:] = 5.0
    out = D.denoise_linear(np.clip(img, 0, None), (0.01, 1e-4), 'auto', gain); ly = lambda a: a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    assert ly(out)[:, 130:].std() < ly(img)[:, 130:].std() * 0.8 and np.abs(ly(out)[:, :100] - ly(img)[:, :100]).max() < 1e-4


def test_develop_runs_the_whole_recipe_on_a_synthetic_raw(monkeypatch):
    monkeypatch.setattr(D.denoise_ai, 'available', lambda: False)
    rng = np.random.default_rng(5); raw = (rng.random((60, 80, 3)) * 0.03 + 0.01).astype(np.float32)
    neutral = (4096, 9776, 4096, 4096, 4096, 6393); cm = tuple(v for x in ((12758, 10000), (-6160, 10000), (916, 10000), (-5992, 10000), (14682, 10000), (5340, 10000), (-251, 10000), (792, 10000), (9511, 10000)) for v in x)
    tags = dict(ColorMatrix1=cm, ColorMatrix2=cm, AsShotNeutral=neutral, BaselineExposure=(19, 100), NoiseProfile=(0.0004, 1e-7), ProfileToneCurve=tuple(np.repeat(np.linspace(0, 1, 16) ** 0.5, 2)[:0]) or tuple(v for x in np.linspace(0, 1, 16) for v in (x, x ** 0.5)),
                OpcodeList3=opcode_list((9, gain_param(np.full((4, 4, 3), 1.5, np.float32))), (1, warp_param([(1, 0, 0, 0, 0, 0)] * 3))))
    monkeypatch.setattr(D, 'read_dng_tags', lambda p: tags); monkeypatch.setattr(D, '_raw_linear', lambda p, half: raw.copy())
    out = D.develop('x.dng', half=True); assert out.shape == raw.shape and out.dtype == np.float32 and 0.3 < np.median(out) < 0.6 and np.isfinite(out).all()
    assert D.develop('x.dng', ev=-3.0, denoise='off', recover=0).mean() < out.mean()
    monkeypatch.setattr(D, 'read_dng_tags', lambda p: {}); assert D.develop('x.dng', ev=0).shape == raw.shape                  # a file with none of the tags is still developed


def test_analysis_luma_is_gamma_encoded_and_full_size(monkeypatch):
    monkeypatch.setattr(D, '_raw_linear', lambda p, half: np.full((20, 30, 3), 0.02, np.float32)); y = D.analysis_luma('x.dng')
    assert y.shape == (20, 30) and float(y.mean()) == pytest.approx(float(D.srgb_oetf(np.float32(0.18))), abs=0.01)
