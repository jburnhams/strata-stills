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
    rgb = RNG.random((30, 40, 3)).astype(np.float32); lut = np.full(64, 0.5, np.float32); out = D.chroma_scale(rgb, lut); w = np.array([0.2126, 0.7152, 0.0722])
    assert np.allclose(out @ w, rgb @ w, atol=1e-5) and np.allclose(D.chroma_scale(rgb, None), rgb)
    assert np.abs((out - (out @ w)[..., None])).mean() < np.abs((rgb - (rgb @ w)[..., None])).mean()


def test_tone_parts_takes_a_list_a_dict_or_nothing():
    own = [0.0, 0.0, 1.0, 1.0]; assert D.tone_parts(own, {}) == (own, None) and D.tone_parts(None, dict(Make='Nobody')) == (None, None)
    c, ch = D.tone_parts(dict(curve=own, chroma=[0.0, 1.0, 1.0, 0.5]), {}); assert c == own and len(ch) == 64 and abs(ch[0] - 1.0) < 1e-6 and abs(ch[-1] - 0.5) < 1e-6
