"""The versions of a picture the rating arena (scripts/build_arena.py, /arena/) compares. To try an idea: add a function here and an entry in VARIANTS (auto4, auto5 ...); the arena makes its pictures for every picture already in the pool, and the votes decide. A version that is clearly worse is retired (`build_arena.py --retire NAME`): its votes are kept but it is no longer shown or rated.

Each function gets a context `c` (types.SimpleNamespace): fin (the person's final, BGR, the picture all others are laid on), neu (our neutral picture laid on it), cam (the camera's own picture laid on it), pred (the look model's row for it), x/names (its features), colour (the learned transform's parameters), face (dict(box, target) or None) and returns a BGR uint8 picture of the same size."""
import os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import lr_look as LK


def match_brightness(img, ref, gpu=None):
    """`img` with its mean lightness brought to `ref`'s by a gamma on CIELAB L (a crude automatic exposure): the camera's and our neutral picture come out darker than the finals, and a blind choice should not be decided by that alone."""
    import lr_gpu
    if lr_gpu.use() if gpu is None else gpu: return lr_gpu.match_brightness(img, ref)                                                            # the same bisection on the device
    lab = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2LAB); L = lab[..., 0] / 100; target = float(np.mean(cv2.cvtColor(ref.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[..., 0])); lo, hi = 0.2, 3.0
    for _ in range(30):
        g = (lo + hi) / 2
        if float(np.mean(100 * np.power(np.clip(L, 1e-4, 1), g))) > target: lo = g
        else: hi = g
    lab[..., 0] = 100 * np.power(np.clip(L, 1e-4, 1), (lo + hi) / 2)
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) * 255 + 0.5, 0, 255).astype(np.uint8)


def lightroom(c): return c.fin
def neutral(c): return match_brightness(c.neu, auto1(c))                                                          # exposure matched to auto1's
def camera(c): return match_brightness(c.cam, auto1(c))
def auto1(c): return LK.apply_look(c.neu, c.pred, dict(zip(c.names, c.x)))[0]
def auto2(c): return LK.apply_look(c.neu, c.pred, dict(zip(c.names, c.x)), colour=c.colour)[0]
def auto3(c): return LK.apply_look(c.neu, c.pred, dict(zip(c.names, c.x)), colour=c.colour, face=c.face)[0]               # the same as auto2 on a picture without a face


def auto4(c):                                                                                                                                     # the network trained on OUR neutral -> the final (scripts/lr_neutral_model.py): its whole transform (tone curve, chroma gains, cast, hue) applied to the neutral picture
    import lr_neutral_model as NM, lr_tmodel as TM
    return TM.apply_params(c.neu, NM.predict_params(c.neu, c.x))
def auto5(c):                                                                                                                                     # auto3's pipeline (percentile curve, local tone, vignette, clarity, skin guard) with its colour step taken from the neutral-trained network instead of the camera-trained transform model
    import lr_neutral_model as NM
    return LK.apply_look(c.neu, c.pred, dict(zip(c.names, c.x)), colour=NM.predict_params(c.neu, c.x), face=c.face)[0]


VARIANTS = {'lightroom': lightroom, 'camera': camera, 'neutral': neutral, 'auto1': auto1, 'auto2': auto2, 'auto3': auto3, 'auto4': auto4, 'auto5': auto5}
