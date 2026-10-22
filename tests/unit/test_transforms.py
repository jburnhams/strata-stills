import cv2, numpy as np, pytest
from stratastills import transforms as T


def grid(deg, w=800, h=600):
    """A square grid turned `deg` degrees (positive = anticlockwise on screen)."""
    img = np.full((h, w, 3), 200, np.uint8)
    for x in range(0, w, 60): cv2.line(img, (x, 0), (x, h), (30, 30, 30), 3)
    for y in range(0, h, 60): cv2.line(img, (0, y), (w, y), (30, 30, 30), 3)
    return cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0), (w, h), borderValue=(200, 200, 200))


def gray(i): return cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)


@pytest.mark.parametrize('deg', [-7.0, 3.0, 20.0])
def test_nadir_level_finds_the_turn_and_undoes_it(deg):
    img = grid(deg); r = T.estimate_level(gray(img), True)
    assert r['angle'] == pytest.approx(-deg, abs=0.5) and r['confidence'] > 0.9          # the scene is turned `deg` anticlockwise: the lines are `-deg` from square
    out, info = T.level(img, dict(nadir=True)); assert info['applied'] and out.shape == img.shape
    assert abs(T.estimate_level(gray(out), True)['angle']) < 0.3


def test_already_level_and_featureless_are_left_alone():
    out, info = T.level(grid(0.0), dict(nadir=True)); assert not info['applied'] and info['reason'] == 'already level'
    flat = np.full((300, 400, 3), 128, np.uint8); out, info = T.level(flat, dict(nadir=True)); assert not info['applied'] and 'few' in info['reason'] and out is flat


def test_oblique_ignores_big_angles():
    out, info = T.level(grid(20.0), dict(nadir=False)); assert not info['applied']


def test_inscribed_scale_fills_the_frame():
    assert T.inscribed_scale(800, 600, 0) == pytest.approx(1.0) and T.inscribed_scale(800, 600, 10) > 1.0
    out = T.rotate_fill(np.full((600, 800, 3), 255, np.uint8), 12); assert out.min() == 255          # no empty corner is brought in


def test_unknown_transform():
    with pytest.raises(KeyError): T.apply('nope', np.zeros((4, 4, 3), np.uint8), {})
    img = np.zeros((4, 4, 3), np.uint8); assert T.apply('original', img, {})[0] is img
