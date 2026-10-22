import cv2, numpy as np

from stratastills import match as M


def scene(h=900, w=1200, seed=1):
    """A textured scene with structure at several scales (features SIFT can find) and colour."""
    r = np.random.default_rng(seed); img = np.zeros((h, w, 3), np.float32)
    for s, a in ((90, 0.5), (30, 0.3), (9, 0.2)):
        n = cv2.resize(r.random((h // s + 2, w // s + 2, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC); img += a * n
    return np.clip(img / img.max(), 0, 1).astype(np.float32)


def test_the_curves_recover_a_tone_change_between_two_views_of_a_scene(monkeypatch):
    monkeypatch.setattr(M, 'FLAT', 0.3)                                                                                                       # the synthetic scene is all texture
    ref = scene(); crop = ref[100:800, 150:1050]; pano = np.clip(crop, 0, 1) ** 1.4 * np.array([1.0, 0.95, 1.05], np.float32)                  # the panorama sees part of it, darker and tinted
    pano = cv2.resize(pano, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA); pano = np.clip(pano, 0, 1); out, rec = M.match_levels(pano, [ref])
    assert rec['applied'] and rec['points'] >= M.MIN_POINTS and rec['rms_after'] < 0.5 * rec['rms_before']
    small = cv2.resize(ref[100:800, 150:1050], (pano.shape[1], pano.shape[0]), interpolation=cv2.INTER_AREA); assert np.abs(out - small).mean() < 0.5 * np.abs(pano - small).mean()
    assert out.min() >= 0 and out.max() <= 1 and out.dtype == np.float32


def test_unrelated_pictures_are_left_alone():
    a, b = scene(seed=2), scene(seed=3); out, rec = M.match_levels(a, [b]); assert not rec['applied'] and 'too few' in rec['reason'] and out is a


def test_fit_is_monotone_pinned_and_pulled_to_the_identity_with_few_points():
    r = np.random.default_rng(0); x = r.random((M.MIN_POINTS, 3)); data = np.stack([x, np.clip(x * 0.7, 0, 1)], 1); f = M.fit(data)
    assert f is not None and (np.diff(f['curves'], axis=1) >= -1e-6).all() and np.allclose(f['curves'][:, 0], 0, atol=1e-6) and np.allclose(f['curves'][:, -1], 1, atol=1e-6)
    mid = f['curves'][0][128]; assert 0.7 * 0.5 < mid < 0.5                                                                                         # moved towards the target, not all the way: 40 points is little evidence
    assert M.fit(data[:M.MIN_POINTS - 1]) is None


def test_references_are_the_single_shots_just_before_the_panorama():
    mk = lambda stem, kind='still': dict(id='s~' + stem, stem=stem, kind=kind, video=None)
    pano = mk('DJI_20250402190211_0046_D-Pano', 'pano'); items = [pano, mk('DJI_20250402190130_0041_D'), mk('DJI_20250402190206_0045_D'), mk('DJI_20250402190211_0046_D'), mk('DJI_20250402190252_0048_D'), mk('DJI_20250402190011_0030_D'), mk('IMG_0001')]
    assert [i['stem'][-6:] for i in M.references(pano, items)] == ['0046_D', '0045_D', '0041_D'] and M.references(mk('DJI_20250402190206_0045_D-Pano', 'pano'), items)[0]['stem'].endswith('0045_D') and M.references(mk('IMG_0001'), items) == []
