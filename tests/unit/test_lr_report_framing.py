"""scripts/build_lr_report.py: our picture laid on the person's final (Lightroom's crop, the way up, then registered), on made pictures."""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import build_lr_report as L


@pytest.fixture(scope="module")
def scene():
    """A 900 x 600 picture full of corners, with more of them in the middle than at the edges."""
    rng = np.random.default_rng(1)
    im = np.full((600, 900, 3), 90, np.uint8)
    for n, sx, sy in ((500, 300, 200), (500, 110, 80)):
        for _ in range(n):
            x, y = int(rng.normal(450, sx)), int(rng.normal(300, sy))
            w, h = rng.integers(4, 22, 2)
            cv2.rectangle(im, (x, y), (x + int(w), y + int(h)), rng.integers(0, 255, 3).tolist(), -1)
    return im


def off(a, b, margin=12):
    """The mean difference of two pictures of one size, away from their borders."""
    d = np.abs(a.astype(int) - b)[margin:-margin, margin:-margin]
    return float(d.mean())


CROP = dict(HasCrop="True", CropTop="0.1", CropLeft="0.15", CropBottom="0.9", CropRight="0.8")


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
@pytest.mark.parametrize("row", [{}, CROP, dict(CROP, CropAngle="3.5")])
def test_the_move_does_what_crop_and_orient_do(scene, row, orientation):
    want = L.D.orient(L.D.lr_crop(scene, row), orientation)
    h, w = want.shape[:2]
    T = L.sensor_to_crop(scene.shape, row, orientation, None, (w, h))
    got = cv2.warpPerspective(scene, T, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    real = np.full(scene.shape[:2], 255, np.uint8)
    ok = cv2.warpPerspective(real, T, (w, h), flags=cv2.INTER_NEAREST) > 0
    assert np.abs(want.astype(int) - got)[ok].mean() < 8  # to about a pixel: lr_crop rounds to whole ones


def test_a_final_a_little_off_is_met_at_the_edges(scene):
    M = np.array([[1.012, 0.004, -6], [-0.004, 1.012, 3], [0, 0, 1]], np.float32)
    final = cv2.warpPerspective(scene, M, (900, 600), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    out, how, (x0, y0, x1, y1) = L.to_final(scene, {}, 1, final, with_rect=True)
    assert how == "Lightroom crop, then registered"
    assert off(out[y0:y1, x0:x1], final[y0:y1, x0:x1]) < 0.4 * off(scene, final)


def test_a_changed_perspective_is_matched_only_when_the_person_changed_it(scene):
    q = np.float32([[0, 0], [900, 0], [900, 600], [0, 600]])
    H = cv2.getPerspectiveTransform(q, np.float32([[40, 25], [870, 5], [900, 600], [0, 590]]))
    final = cv2.warpPerspective(scene, H, (900, 600), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    row = dict(PerspectiveVertical="+20")
    out, how, (x0, y0, x1, y1) = L.to_final(scene, row, 1, final, with_rect=True)
    assert how.endswith("with perspective")
    assert off(out[y0:y1, x0:x1], final[y0:y1, x0:x1]) < 0.4 * off(scene, final)
    assert (
        x0 > 0 and y0 > 0
    )  # the final's corner there is outside our picture: known from the move, not guessed from the pixels
    flat, how = L.to_final(scene, {}, 1, final)
    assert not how.endswith("with perspective")


def test_pixels_outside_the_nominal_crop_are_used(scene):
    """A final that shows 10 px more on the left than its crop says: real pixels, so nothing is cut."""
    row = dict(HasCrop="True", CropTop="0.1", CropLeft="0.1", CropBottom="0.9", CropRight="0.9")
    final = scene[60:540, 80:800]
    out, _, rect = L.to_final(scene, row, 1, final, with_rect=True)
    assert rect == (0, 0, 720, 480) and off(out, final) < 0.4 * off(scene[60:540, 90:810], final)


def test_what_counts_as_reshaped():
    assert not L.reshaped({}) and not L.reshaped(dict(PerspectiveVertical="0", PerspectiveScale="100"))
    assert L.reshaped(dict(PerspectiveUpright="1")) and L.reshaped(dict(PerspectiveHorizontal="+12"))
    assert L.reshaped(dict(PerspectiveScale="95"))


def test_a_wild_transform_is_refused():
    H = np.array([[1, 0, 500], [0, 1, 0], [0, 0, 1]], float)
    assert not L.plausible(H, 900, 600, True) and not L.plausible(np.diag([1.2, 1.2, 1]), 900, 600, False)
    assert L.plausible(np.eye(3), 900, 600, True) and L.plausible(np.eye(3), 900, 600, False)


def test_a_picture_with_nothing_to_match_keeps_the_crop():
    flat = np.full((300, 400, 3), 120, np.uint8)
    out, how, rect = L.to_final(flat, {}, 1, flat, with_rect=True)
    assert how == "Lightroom crop" and rect == (0, 0, 400, 300) and (out == 120).all()


def test_a_registration_that_does_not_help_is_dropped(scene, monkeypatch):
    M = np.array([[1, 0, 25], [0, 1, 10], [0, 0, 1]], float)
    monkeypatch.setattr(L, "register_even", lambda *a: M)  # as a near-black frame's noise can give
    out, how = L.to_final(scene, {}, 1, scene)
    assert how == "Lightroom crop" and off(out, scene) < 1
    assert L.edge_agreement(scene[..., 0], scene[..., 0]) > 0.99
