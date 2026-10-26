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


def test_the_noise_map_counts_the_lens_shading_gain_and_denoise_leaves_luminance_to_the_coring():
    rng = np.random.default_rng(4); img = np.full((120, 160, 3), 0.2, np.float32) + rng.normal(0, 0.02, (120, 160, 3)).astype(np.float32); gain = np.ones((120, 160), np.float32); gain[:, 120:] = 5.0; npf = (0.01, 1e-4)
    plain = D.noise_map8(img, npf); aware = D.noise_map8(img, npf, gain=gain)
    assert np.allclose(aware[:, :80], plain[:, :80], rtol=0.02) and aware[:, 140:].mean() > plain[:, 140:].mean() * 2.0                          # the lifted corner is noisier by the gain, the middle is as before
    ly = lambda a: a @ np.array([0.2126, 0.7152, 0.0722], np.float32); out = D.denoise_linear(np.clip(img, 0, None), npf, 'auto', gain); assert np.abs(ly(out) - ly(np.clip(img, 0, None))).max() < 1e-4        # no luminance pass any more: only colour
    assert D.denoise_linear(np.clip(img, 0, None), npf, 3.0).shape == img.shape


def test_develop_runs_the_whole_recipe_on_a_synthetic_raw(monkeypatch):
    monkeypatch.setattr(D.denoise_ai, 'available', lambda m=None: False)
    rng = np.random.default_rng(5); raw = (rng.random((60, 80, 3)) * 0.03 + 0.01).astype(np.float32)
    neutral = (4096, 9776, 4096, 4096, 4096, 6393); cm = tuple(v for x in ((12758, 10000), (-6160, 10000), (916, 10000), (-5992, 10000), (14682, 10000), (5340, 10000), (-251, 10000), (792, 10000), (9511, 10000)) for v in x)
    tags = dict(ColorMatrix1=cm, ColorMatrix2=cm, AsShotNeutral=neutral, BaselineExposure=(19, 100), NoiseProfile=(0.0004, 1e-7), ProfileToneCurve=tuple(np.repeat(np.linspace(0, 1, 16) ** 0.5, 2)[:0]) or tuple(v for x in np.linspace(0, 1, 16) for v in (x, x ** 0.5)),
                OpcodeList3=opcode_list((9, gain_param(np.full((4, 4, 3), 1.5, np.float32))), (1, warp_param([(1, 0, 0, 0, 0, 0)] * 3))))
    monkeypatch.setattr(D, 'read_dng_tags', lambda p, lens='auto': tags); monkeypatch.setattr(D, '_raw_u16', lambda p, half: (raw * 65535).astype(np.uint16))
    out = D.develop('x.dng', half=True); assert out.shape == raw.shape and out.dtype == np.float32 and 0.3 < np.median(out) < 0.6 and np.isfinite(out).all()
    assert D.develop('x.dng', ev=-3.0, denoise='off', recover=0).mean() < out.mean()
    monkeypatch.setattr(D, 'read_dng_tags', lambda p, lens='auto': {}); assert D.develop('x.dng', ev=0).shape == raw.shape                  # a file with none of the tags is still developed


def test_analysis_luma_is_gamma_encoded_and_full_size(monkeypatch):
    monkeypatch.setattr(D, '_raw_linear', lambda p, half: np.full((20, 30, 3), 0.02, np.float32)); y = D.analysis_luma('x.dng')
    assert y.shape == (20, 30) and float(y.mean()) == pytest.approx(float(D.srgb_oetf(np.float32(0.18))), abs=0.01)


LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def test_clean_colour_greys_colour_noise_keeps_real_colour_and_leaves_luminance_alone():
    rng = np.random.default_rng(6); h, w = 160, 240; y = np.full((h, w), 0.15, np.float32) + 0.02 * cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 1.5)
    noise = rng.normal(0, 0.03, (h, w, 2)).astype(np.float32); img = np.dstack([y + noise[..., 0], y, y + noise[..., 1]]); img[:, 120:] = np.clip(np.dstack([y[:, 120:] * 0.5, y[:, 120:] * 1.0, y[:, 120:] * 3.0]), 0, 1)       # left: grey with colour noise; right: a real blue
    out = D.clean_colour(np.clip(img, 0, 1), np.full((h, w), 8.0, np.float32)); chroma = lambda a: (a[..., 2] - a @ LUMA)
    assert chroma(out)[:, :100].std() < chroma(img)[:, :100].std() * 0.35
    assert chroma(out)[:, 130:].mean() > chroma(img)[:, 130:].mean() * 0.7 and np.abs(out @ LUMA - np.clip(img, 0, 1) @ LUMA).max() < 0.02


def test_clean_colour_rolls_colour_off_near_white_and_does_not_bleed_across_luminance_edges():
    img = np.full((64, 96, 3), 0.97, np.float32); img[:, :, 0] += 0.03; img[:, :, 1] -= 0.03; fringe = D.clean_colour(np.clip(img, 0, 1), np.full((64, 96), 2.0, np.float32))
    assert abs(float((fringe[..., 0] - fringe[..., 1]).mean())) < abs(float((np.clip(img, 0, 1)[..., 0] - np.clip(img, 0, 1)[..., 1]).mean())) * 0.7
    edge = np.full((64, 96, 3), 0.2, np.float32); edge[:, 48:] = [0.8, 0.4, 0.4]; o = D.clean_colour(edge, np.full((64, 96), 8.0, np.float32)); assert o[:, 52:].mean(0).mean(0)[0] - o[:, 52:].mean(0).mean(0)[1] > 0.2 and abs(o[:, :44, 0] - o[:, :44, 1]).max() < 0.03


def test_core_luma_removes_fine_grain_but_keeps_strong_edges():
    rng = np.random.default_rng(7); y = np.full((120, 160), 0.2, np.float32); y[:, 80:] = 0.5; img = np.dstack([y] * 3) + rng.normal(0, 0.02, (120, 160, 1)).astype(np.float32); out = D.core_luma(np.clip(img, 0, 1), np.full((120, 160), 6.0, np.float32))
    assert out[:, :60].std() < img[:, :60].std() * 0.6 and abs(float(out[:, 90:].mean() - out[:, :70].mean()) - 0.3) < 0.02


def test_exposure_protects_the_highlights_but_never_more_than_a_stop():
    img = np.full((80, 80, 3), 0.03, np.float32); img[:20] = 0.45                                                            # a quarter of the picture is a bright wall, the rest dark
    ev = D.auto_ev(img, None); ev_median_only = D.auto_ev(np.where(img > 0.1, 0.03, img).astype(np.float32), None)
    assert 0.45 * 2.0 ** ev <= 1.05 and ev < ev_median_only - 0.3 and ev >= ev_median_only - D.MAX_DARKER - 0.05


def test_local_tone_compresses_big_bright_regions_more_than_a_flat_squeeze():
    img = np.full((200, 300, 3), 0.1, np.float32); img[:, 150:] = 0.8; rng = np.random.default_rng(8); img *= (1 + 0.05 * rng.standard_normal((200, 300, 1))).astype(np.float32)
    a = D.local_tone(img, 0.35, highlights=0.0); b = D.local_tone(img, 0.35, highlights=0.55); assert b[:, 180:].mean() < a[:, 180:].mean() * 0.95


def test_highlight_detail_boosts_contrast_in_bright_areas_but_not_their_brightness_or_the_dark():
    rng = np.random.default_rng(9); h, w = 160, 240; tex = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 1.2) * 0.02
    img = np.dstack([np.full((h, w), 0.2, np.float32) + tex] * 3); img[:, 120:] = (0.85 + tex[:, 120:, None] * 1.0) * np.ones(3, np.float32)                 # left dark, right bright: both with the same texture
    out = D.highlight_detail(np.clip(img, 0, 1), 1.0)
    assert out[:, 130:].std() > img[:, 130:].std() * 1.5 and abs(float(out[:, 130:].mean() - np.clip(img, 0, 1)[:, 130:].mean())) < 0.01        # more contrast, same brightness
    assert np.abs(out[:, :100] - img[:, :100]).max() < 1e-3 and out.max() <= 1.0 and D.highlight_detail(np.clip(img, 0, 1), 0.0).shape == img.shape


def test_default_shoulder_is_the_soft_one():
    assert D.WHITE == 16.0 and D.KNEE == 0.55 and 0 < D.HIGHLIGHT_CONTRAST <= 1.5


def test_band_coring_removes_mottling_at_two_to_four_pixels_but_keeps_edges_and_is_stronger_in_bright_areas():
    rng = np.random.default_rng(10); h, w = 200, 300
    mottle = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 1.6); mottle *= 0.012 / mottle.std()                                   # grain correlated over 2 to 4 px
    dark = np.full((h, w), 0.25, np.float32) + mottle; bright = np.full((h, w), 0.85, np.float32) + mottle
    sig = np.full((h, w), 8.0, np.float32)
    for lab, a in (('dark', dark), ('bright', bright)):
        out = D.core_luma(np.dstack([a] * 3), sig); assert out[..., 0].std() < a.std() * 0.8, lab
    od = D.core_luma(np.dstack([dark] * 3), sig)[..., 0].std() / dark.std(); ob = D.core_luma(np.dstack([bright] * 3), sig)[..., 0].std() / bright.std(); assert ob < od        # stronger in the bright area
    edge = np.full((h, w), 0.5, np.float32); edge[:, 150:] = 0.9; e = D.core_luma(np.dstack([edge + mottle] * 3), sig)[..., 0]
    assert abs(float(e[:, 160:190].mean() - e[:, 110:140].mean()) - 0.4) < 0.01 and D.core_luma(np.dstack([edge] * 3), sig * 0)[..., 0].std() == pytest.approx(edge.std(), rel=1e-3)      # a strong edge is untouched, and no noise means no coring


def test_the_contrast_boost_runs_after_the_grain_is_cored(monkeypatch):
    order = []; rng = np.random.default_rng(5); raw = (rng.random((60, 80, 3)) * 0.03 + 0.01).astype(np.float32)
    monkeypatch.setattr(D, 'read_dng_tags', lambda p, lens='auto': dict(NoiseProfile=(0.0004, 1e-7))); monkeypatch.setattr(D, '_raw_u16', lambda p, half: (raw * 65535).astype(np.uint16)); monkeypatch.setattr(D.denoise_ai, 'available', lambda m=None: False)
    monkeypatch.setattr(D, 'core_luma', lambda r, s, k=None: order.append('core') or r); monkeypatch.setattr(D, 'highlight_detail', lambda r, a=1.0, lo=0.62, hi=0.92: order.append('contrast') or r)
    D.develop('x.dng', half=True, backend='cpu'); assert order == ['core', 'contrast']


def test_rgbtone_keeps_the_hue_per_channel_does_not():
    lut = (np.linspace(0, 1, 4096) ** 0.55).astype(np.float32); orange = np.full((4, 4, 3), 0.0, np.float32); orange[...] = [0.55, 0.25, 0.08]
    per = D.finish(orange, lut, 1.0); tone = D.finish(orange, lut, 0.0); mix = D.finish(orange, lut, 0.5)
    f = lambda v: np.interp(v, np.linspace(0, 1, 4096), lut); assert np.allclose(tone[0, 0, 0], f(D.shoulder(orange)[0, 0, 0]), atol=1e-4) and np.allclose(tone[0, 0, 2], f(D.shoulder(orange)[0, 0, 2]), atol=1e-4)      # largest and smallest channel go through the curve
    mid = lambda a: (a[0, 0, 1] - a[0, 0, 2]) / (a[0, 0, 0] - a[0, 0, 2]); assert abs(mid(tone) - mid(D.shoulder(orange))) < 1e-4 and abs(mid(per) - mid(D.shoulder(orange))) > 0.02 and mid(tone) < mid(mix) < mid(per)      # the middle channel keeps its place between the others (the hue)
    assert np.allclose(D.finish(orange, None, 0.0), D.finish(orange, None, 1.0)) and D.finish(orange, lut)[0, 0, 0] == per[0, 0, 0]


def test_colour_through_prophoto_composes_to_the_direct_matrix_and_an_identity_map_changes_nothing():
    cm = np.array([[0.9, -0.2, -0.1], [-0.3, 1.1, 0.1], [0.0, 0.1, 0.8]]); n = np.array([0.5, 1.0, 0.6])
    assert np.allclose(D.prophoto_to_srgb() @ D.camera_to_prophoto(cm, n), D.camera_to_srgb(cm, n), atol=1e-6); assert np.allclose(D.camera_to_prophoto(cm, n) @ np.ones(3), 1)
    x = (np.random.default_rng(0).random((30, 40, 3)) * 1.5).astype(np.float32); ident = np.zeros((36, 10, 3), np.float32); ident[..., 1:] = 1
    assert np.allclose(D.hsv_to_rgb(D.rgb_to_hsv(x)), x, atol=1e-5) and np.abs(D.apply_hue_sat_map(x, ident) - x).max() < 1e-5


def test_hue_sat_map_shifts_hue_scales_saturation_and_blends_the_two_illuminants():
    red = np.full((2, 2, 3), 0.0, np.float32); red[..., 0] = 0.8; red[..., 1] = 0.2; red[..., 2] = 0.2; tb = np.zeros((36, 10, 3), np.float32); tb[..., 1:] = 1
    tb[..., 1] = 0.5; out = D.apply_hue_sat_map(red, tb); assert np.allclose(out[0, 0], [0.8, 0.5, 0.5], atol=1e-5)                                  # saturation halved: 0.75 -> 0.375 of the maximum
    tb[..., 1] = 1; tb[..., 0] = 60; out = D.rgb_to_hsv(D.apply_hue_sat_map(red, tb)); assert abs(out[0, 0, 0] - 1.0) < 1e-5                                    # 60 degrees = one sextant
    tags = dict(ProfileHueSatMapDims=(36, 10, 1), ProfileHueSatMapData1=np.ones(1080), ProfileHueSatMapData2=np.full(1080, 3.0)); mixed = D.hue_sat_table(tags, 0.25); assert np.allclose(mixed, 0.25 * 1 + 0.75 * 3)
    assert D.hue_sat_table({}, 0.5) is None


def test_limit_chroma_leaves_soft_colour_alone_and_rolls_off_strong_colour_keeping_luma():
    soft = np.full((2, 2, 3), 0.5, np.float32); soft[..., 0] = 0.55; strong = np.zeros((2, 2, 3), np.float32); strong[...] = [0.95, 0.45, 0.05]
    assert np.allclose(D.limit_chroma(soft), soft, atol=1e-6); out = D.limit_chroma(strong); w = np.array([0.2126, 0.7152, 0.0722])
    assert abs(out[0, 0] @ w - strong[0, 0] @ w) < 1e-4 and (out[0, 0].max() - out[0, 0].min()) < (strong[0, 0].max() - strong[0, 0].min())

def test_a_non_dng_raw_is_cut_to_the_makers_active_area(monkeypatch):
    import contextlib, sys, types
    from types import SimpleNamespace as NS
    z = NS(iwidth=24, iheight=16, crop_width=20, crop_height=12, crop_left_margin=2, crop_top_margin=4, flip=0)
    class R:
        sizes = z
        def postprocess(self, half_size, **kw): z.flip = 0; n = 2 if half_size else 1; return np.zeros((16 // n, 24 // n, 3), np.uint16)
    mod = types.ModuleType('rawpy'); mod.imread = lambda p: contextlib.nullcontext(R()); mod.ColorSpace = NS(raw=0); mod.DemosaicAlgorithm = NS(DHT=0); mod.HighlightMode = NS(Blend=0); monkeypatch.setitem(sys.modules, 'rawpy', mod)
    assert D._raw_u16('x.arw', False).shape == (12, 20, 3) and D._raw_u16('x.arw', True).shape == (6, 10, 3) and D._raw_u16('x.dng', False).shape == (12, 20, 3)           # a DNG without opcodes is cut like any raw
    monkeypatch.setattr(D, '_has_opcodes', lambda path: True); assert D._raw_u16('x.dng', False).shape == (16, 24, 3) and D._raw_u16('x.arw', False).shape == (12, 20, 3)           # one with opcodes (sensor geometry) keeps its full area
    for flip, shape in ((3, (12, 20, 3)), (6, (20, 12, 3)), (5, (20, 12, 3))): z.flip = flip; assert D._raw_u16('x.arw', False).shape == shape                                                                  # turned upright by the camera's orientation


def test_camera_exposure_is_no_change_and_a_number_is_stops_from_it():
    assert D.ev_stops('camera') == 0.0 and D.ev_stops(1) == 1.0 and D.ev_stops('-0.5') == -0.5


def test_a_dng_with_an_orientation_is_turned_upright_and_its_opcodes_are_turned_with_it(monkeypatch):
    import contextlib, sys, types
    from types import SimpleNamespace as NS
    z = NS(iwidth=24, iheight=16, crop_width=0, crop_height=0, crop_left_margin=0, crop_top_margin=0, flip=6)
    class R:
        sizes = z
        def postprocess(self, half_size, **kw): z.flip = 0; return np.zeros((16, 24, 3), np.uint16)
    mod = types.ModuleType('rawpy'); mod.imread = lambda p: contextlib.nullcontext(R()); mod.ColorSpace = NS(raw=0); mod.DemosaicAlgorithm = NS(DHT=0); mod.HighlightMode = NS(Blend=0); monkeypatch.setitem(sys.modules, 'rawpy', mod)
    assert D._raw_u16('x.dng', False).shape == (24, 16, 3)
    m = np.arange(12, dtype='>f4').reshape(3, 4); body = struct.pack('>10I', 0, 0, 16, 24, 0, 3, 1, 1, 3, 4) + struct.pack('>4d', .5, .5, 0, 0) + struct.pack('>I', 1) + m.tobytes()
    assert D.gain_map(body)[0][:, :, 0].shape == (3, 4) and D.gain_map(body, 6)[0][:, :, 0].shape == (4, 3) and np.array_equal(D.gain_map(body, 6)[0][:, :, 0], np.rot90(m.astype(np.float32), -1)) and np.array_equal(D.gain_map(body, 3)[0][:, :, 0], np.rot90(m.astype(np.float32), 2))
    wp = struct.pack('>I', 1) + struct.pack('>6d', 1, 0, 0, 0, 0, 0) + struct.pack('>2d', 0.25, 0.4)
    assert D.warp_params(wp)[1] == (0.25, 0.4) and np.allclose(D.warp_params(wp, 6)[1], (0.6, 0.25)) and np.allclose(D.warp_params(wp, 5)[1], (0.4, 0.75)) and np.allclose(D.warp_params(wp, 3)[1], (0.75, 0.6))


def test_match_from_decode_uses_the_card_and_gives_no_tone_without_a_preview(monkeypatch):
    pytest.importorskip('torch'); from stratastills import tonematch
    monkeypatch.setattr(D, 'read_dng_tags', lambda path, lens: {}); monkeypatch.setattr(tonematch, 'embedded_jpeg', lambda path: None)
    u16 = (np.random.default_rng(0).random((64, 48, 3)) * 65535).astype(np.uint16); an = {}
    assert D._match_from_decode('x.dng', u16, 'auto', an) is None and an['tone'] is None
