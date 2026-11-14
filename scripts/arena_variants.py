"""The versions of a picture the rating arena (scripts/build_arena.py, /arena/) compares. To try an idea: add a function here and an entry in VARIANTS (auto4, auto5 ...); the arena makes its pictures for every picture already in the pool, and the votes decide. A version that is clearly worse is retired (`build_arena.py --retire NAME`): its votes are kept but it is no longer shown or rated.

Each function gets a context `c` (types.SimpleNamespace): fin (the person's final, BGR, the picture all others are laid on), neu (our neutral picture laid on it), cam (the camera's own picture laid on it), x (its features, made by stratastills.look from the camera's preview cut as the final was), fl (its faces, boxes in this frame), bundle and preds (the look's frozen models and what they predict) and returns a BGR uint8 picture of the same size. The auto versions ARE the pipeline's look stage (stratastills.look), so a version that wins here is what a project's 'Best quality' makes."""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stratastills import look


def match_brightness(img, ref, gpu=None):
    """`img` with its mean lightness brought to `ref`'s by a gamma on CIELAB L (a crude automatic exposure): the camera's and our neutral picture come out darker than the finals, and a blind choice should not be decided by that alone."""
    from stratastills.look import gpu as lr_gpu

    if lr_gpu.use() if gpu is None else gpu:
        return lr_gpu.match_brightness(img, ref)  # the same bisection on the device
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    L = lab[..., 0] / 100
    target = float(np.mean(cv2.cvtColor(ref.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[..., 0]))
    lo, hi = 0.2, 3.0
    for _ in range(30):
        g = (lo + hi) / 2
        if float(np.mean(100 * np.power(np.clip(L, 1e-4, 1), g))) > target:
            lo = g
        else:
            hi = g
    lab[..., 0] = 100 * np.power(np.clip(L, 1e-4, 1), (lo + hi) / 2)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) * 255 + 0.5, 0, 255).astype(np.uint8)


def lightroom(c):
    return c.fin


def neutral(c):
    return match_brightness(c.neu, auto1(c))  # exposure matched to auto1's


def camera(c):
    return match_brightness(c.cam, auto1(c))


def _look(name):
    """An auto version is the pipeline's look stage (stratastills.look.render) on our neutral picture: the arena judges exactly the code a project runs."""

    def f(c):
        return look.render(name, c.neu, c.x, c.fl, c.bundle, c.preds)[0]

    f.__name__ = name
    f.__doc__ = f"stratastills.look {name}"
    return f


auto1, auto2, auto3, auto4, auto5, auto6, auto7 = (_look(f"auto{i}") for i in range(1, 8))


VARIANTS = {
    "lightroom": lightroom,
    "camera": camera,
    "neutral": neutral,
    "auto1": auto1,
    "auto2": auto2,
    "auto3": auto3,
    "auto4": auto4,
    "auto5": auto5,
    "auto6": auto6,
    "auto7": auto7,
}
