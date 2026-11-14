"""Face first: after the automatic edit, move the picture so the main face's skin lands near the colour the person gives skin, trading a
little whole-picture accuracy for a face that looks right (people see a wrong face at once).

Faces: candidates/personal/faces2.pkl {catalogue row: [{box (x, y, w, h fractions of the final's frame), lm (68 landmarks, fractions),
pose, score, real (P(real person) from the strata-models face-kind role), kind, by (which look found it)}]}, from the re-detection of
2026-11-09 (plain + padded look, harder looks where Sonnet/CLIP expected more faces). Only real faces (real >= 0.5) are used, so a statue
or a poster never sets the skin colour.

Skin: the face outline from the landmarks (jaw and a strip of forehead) minus eyes, brows and mouth, eroded; its median CIELAB.

Target: XGBoost (out/cache/lr-skin-target.pkl, one model per L/a/b) predicts the final's skin minus the edited picture's skin, from the
picture features (lr_look.load), the edited and the neutral skin colour, the face size and pose; trained on 3,829 training faces with
the network's OUT-OF-FOLD edits (two networks, each trained on half the training shoots), so it learns what the edit actually gets wrong.

Step: the whole picture's a/b moved GLOBAL of the way to the target (kept off the blacks), then the rest of the way LOCAL of it on the
face only, through a mask feathered over a quarter of the face's size. L is left alone.
Held out (861 faces, 512 px): face colour distance 6.76 -> 5.49, whole picture 7.48 -> 7.57; tungsten faces 15.9 -> 11.0, whole 13.7 -> 12.8."""

import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stratastills.look import face_first as _FF
from stratastills.look.face_first import GLOBAL, LOCAL, MIN_AREA, MIN_REAL, MIN_SCORE, features, lab, skin, skin_mask  # noqa: F401  (moved)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FACES_FILE = os.path.join(ROOT, "candidates", "personal", "faces2.pkl")
MODEL_FILE = os.path.join(ROOT, "out", "cache", "lr-skin-target.pkl")
_cache = {}


def faces():
    if "faces" not in _cache:
        _cache["faces"] = pickle.load(open(FACES_FILE, "rb")) if os.path.exists(FACES_FILE) else {}
    return _cache["faces"]


def main_face(row):
    """The largest real face of catalogue row `row`, or None."""
    return _FF.main_face(faces().get(str(int(row)), []))


def models():
    """The skin-target models, or None when the file is missing (apply then leaves the picture as it is)."""
    if "models" not in _cache:
        _cache["models"] = pickle.load(open(MODEL_FILE, "rb")) if os.path.exists(MODEL_FILE) else None
    return _cache["models"]


def apply(out_bgr, neu_bgr, x, face, glob=GLOBAL, local=LOCAL):
    """look.face_first.apply with the skin-target models of out/cache (the picture as it is when the file is missing)."""
    return _FF.apply(out_bgr, neu_bgr, x, face, models(), glob, local)
