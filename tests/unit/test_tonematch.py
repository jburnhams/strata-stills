import numpy as np

from stratastills import develop as D, tonematch as T

RNG = np.random.default_rng(3)


def scene(h=120, w=180):
    """A linear picture with a spread of luminance and some colour, and a 'camera JPEG' of it: a fixed gamma-ish curve on the luminance, colour at 0.6 of the picture's."""
    lin = (RNG.random((h, w)) ** 3 * 0.9 + 0.002).astype(np.float32)[..., None] * np.array([1.0, 0.8, 0.5], np.float32); y = lin @ T.LUMA
    yenc = np.clip((y * 1.5) ** (1 / 2.0), 0, 1); base = D.srgb_oetf(np.clip(lin, 0, 1)); jpg = np.clip(yenc[..., None] + (base - D.srgb_oetf(np.clip(y, 0, 1))[..., None]) * 0.6, 0, 1)
    return lin, (jpg * 255 + 0.5).astype(np.uint8)


def test_the_matched_curve_takes_the_pictures_luminance_to_the_previews():
    lin, jpg = scene(); m = T.match_curve(lin, jpg); assert m is not None
    c = np.asarray(m['curve']); x, y = c[0::2], c[1::2]; assert x[0] == 0 and y[0] == 0 and x[-1] == 1 and (np.diff(x) > 0).all() and (np.diff(y) >= 0).all()
    for v in (0.02, 0.1, 0.3): assert abs(np.interp(v, x, y) - min(1, (v * 1.5) ** 0.5)) < 0.04


def test_the_chroma_scale_measures_the_previews_colour_against_ours_after_the_curve():
    lin, jpg = scene(); m = T.match_curve(lin, jpg); ch = np.asarray(m['chroma']); s = ch[1::2]
    assert (ch[0::2] >= 0).all() and (ch[0::2] <= 1).all() and 0.2 <= s.min() and s.max() <= 1.5 and np.median(s) < 1.0                                    # a curve applied per channel over-colours; the camera's colour is a part of that


def test_a_preview_that_is_another_picture_or_too_small_gives_no_curve():
    lin, jpg = scene(); assert T.match_curve(lin, np.ascontiguousarray(jpg[::-1, ::-1][:, ::-1] * 0 + 128)) is None and T.match_curve(lin, jpg[:20, :30]) is None and T.match_curve(lin, None) is None
    noise = (RNG.random(jpg.shape) * 255).astype(np.uint8); assert T.match_curve(lin, noise) is None


def test_a_preview_turned_by_the_cameras_orientation_is_found_again():
    lin, jpg = scene(180, 120); m0 = T.match_curve(lin, jpg); m1 = T.match_curve(lin, np.rot90(jpg)); m2 = T.match_curve(lin, np.rot90(jpg, 2)); assert m0 and m1 and m2


def test_chroma_scale_changes_colour_but_keeps_luminance():
    rgb = RNG.random((30, 40, 3)).astype(np.float32); lut = np.full(64, 0.5, np.float32); out = D.chroma_scale(rgb, (lut, None)); w = np.array([0.2126, 0.7152, 0.0722])
    assert np.allclose(out @ w, rgb @ w, atol=1e-5) and np.allclose(D.chroma_scale(rgb, None), rgb)
    assert np.abs((out - (out @ w)[..., None])).mean() < np.abs((rgb - (rgb @ w)[..., None])).mean()


def test_tone_parts_takes_a_list_a_dict_or_nothing():
    own = [0.0, 0.0, 1.0, 1.0]; assert D.tone_parts(own, {}) == (own, None) and D.tone_parts(None, dict(Make='Nobody')) == (None, None)
    c, (ch, tab) = D.tone_parts(dict(curve=own, chroma=[0.0, 1.0, 1.0, 0.5]), {}); assert c == own and tab is None and len(ch) == 64 and abs(ch[0] - 1.0) < 1e-6 and abs(ch[-1] - 0.5) < 1e-6


def hue_table(rot=0.0, scale=1.0, h=24): return np.dstack([np.full((h, len(D.LOOK_CHROMA)), rot), np.full((h, len(D.LOOK_CHROMA)), scale)]).astype(np.float32)


def test_hue_look_with_an_identity_table_changes_nothing_and_keeps_luminance_when_it_turns_and_scales():
    rgb = (RNG.random((40, 50, 3)) * 0.8 + 0.1).astype(np.float32); w = np.array([0.2126, 0.7152, 0.0722], np.float32)
    assert np.allclose(D.hue_look(rgb, hue_table()), rgb, atol=1e-5)
    out = D.hue_look(rgb, hue_table(30.0, 0.5)); inside = (out > 0.001).all(-1) & (out < 0.999).all(-1); assert np.allclose((out @ w)[inside], (rgb @ w)[inside], atol=1e-4)
    d0, d1 = rgb - (rgb @ w)[..., None], out - (out @ w)[..., None]; assert np.allclose(np.linalg.norm(d1, axis=-1)[inside], 0.5 * np.linalg.norm(d0, axis=-1)[inside], atol=1e-4)


def test_hue_look_turns_a_colour_by_the_tables_angle():
    w = np.array([0.2126, 0.7152, 0.0722], np.float32); c = np.array([[[0.6, 0.4, 0.3]]], np.float32); out = D.hue_look(c, hue_table(40.0)); d0 = c[0, 0] - c[0, 0] @ w; d1 = out[0, 0] - out[0, 0] @ w
    e1, e2 = D._E1, D._E2; a0 = np.degrees(np.arctan2(d0 @ e2, d0 @ e1)); a1 = np.degrees(np.arctan2(d1 @ e2, d1 @ e1)); assert abs(((a1 - a0 + 180) % 360 - 180) - 40.0) < 0.5


def test_a_device_look_table_is_used_only_with_a_matched_tone():
    spec = dict(hue=24, data=hue_table(10.0, 0.9).ravel().tolist()); tags = dict(LookTable=spec); own = [0.0, 0.0, 1.0, 1.0]
    assert D.tone_parts(own, tags)[1] is None and D.tone_parts(None, tags)[1] is None
    c, (lut, tab) = D.tone_parts(dict(curve=own), tags); assert lut is None and tab.shape == (24, len(D.LOOK_CHROMA), 2) and D.tone_parts(dict(curve=own), {})[1] is None


def fake_rawpy(monkeypatch, fmt, data=None, fail=False):
    import contextlib, enum, sys, types
    class TF(enum.Enum): JPEG = 1; BITMAP = 2; OTHER = 3
    mod = types.ModuleType('rawpy'); mod.ThumbFormat = TF
    class R:
        def extract_thumb(self):
            if fail: raise RuntimeError('no thumbnail')
            return types.SimpleNamespace(format=getattr(TF, fmt), data=data)
    mod.imread = lambda path: contextlib.nullcontext(R()); monkeypatch.setitem(sys.modules, 'rawpy', mod)


def test_the_embedded_jpeg_is_decoded_to_rgb_and_a_bitmap_thumbnail_is_passed_through(monkeypatch):
    import cv2
    bgr = np.zeros((6, 8, 3), np.uint8); bgr[..., 2] = 255                                               # red, as OpenCV stores it
    ok, buf = cv2.imencode('.png', bgr); fake_rawpy(monkeypatch, 'JPEG', buf.tobytes())                  # imdecode reads any format; PNG keeps the colour exact
    rgb = T.embedded_jpeg('x.arw'); assert rgb.shape == (6, 8, 3) and rgb[0, 0].tolist() == [255, 0, 0]
    bm = np.full((4, 5, 3), 7, np.uint8); fake_rawpy(monkeypatch, 'BITMAP', bm); assert (T.embedded_jpeg('x.arw') == 7).all()


def test_an_undecodable_unknown_or_missing_thumbnail_is_none(monkeypatch):
    fake_rawpy(monkeypatch, 'JPEG', b'not an image'); assert T.embedded_jpeg('x.arw') is None
    fake_rawpy(monkeypatch, 'OTHER', b''); assert T.embedded_jpeg('x.arw') is None
    fake_rawpy(monkeypatch, 'JPEG', fail=True); assert T.embedded_jpeg('x.arw') is None


def test_a_picture_is_cropped_to_the_centre_to_match_an_aspect_ratio():
    img = np.arange(10 * 20).reshape(10, 20)
    assert T._crop_to_aspect(img, 2.0) is img and T._crop_to_aspect(img, 2.005) is img                    # already that shape (within 1 %)
    wide = T._crop_to_aspect(img, 1.0); assert wide.shape == (10, 10) and wide[0, 0] == img[0, 5]         # too wide: sides trimmed equally
    tall = T._crop_to_aspect(img.T, 1.0); assert tall.shape == (10, 10) and tall[0, 0] == img.T[5, 0]     # too tall: top and bottom trimmed equally


def test_a_preview_with_a_step_in_its_tone_gives_a_smooth_curve_with_no_flat_run():
    x = np.sort(RNG.random(20000) ** 3).astype(np.float32); y = D.srgb_eotf(np.where(x < 0.08, 0.2 + x * 4, 0.62 + (x - 0.08) * 0.5).clip(0, 1) + RNG.normal(0, 0.01, x.size))   # a jump of the preview's tone at 0.08 (the posterising case), with noise
    kx, ky = T.fit(x, np.clip(y, 0, 1)); ex = T._enc(kx); s = (np.diff(ky) / np.diff(ex))[1:][ky[2:] < 1]                                                     # the first step (the preview's black) and the clipped top are the picture's, not the fit's
    assert kx[0] == 0 and ky[0] == 0 and (np.diff(kx) > 0).all() and (np.diff(ky) >= 0).all() and s.min() >= T.SLOPE_MIN - 1e-6 and s.max() <= T.SLOPE_MAX + 1e-6


def test_isotonic_pools_a_dip_instead_of_holding_the_peak():
    assert np.allclose(T._isotonic([0.1, 0.5, 0.3, 0.6]), [0.1, 0.4, 0.4, 0.6])
