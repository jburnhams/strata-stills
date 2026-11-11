"""How much a remade arena picture changed, and whether the votes on the old one still count (Jonathan, 2026-11-11: decide from the size of the change rather than always keeping or always dropping them).

`change(old, new)` is the share of pixels a person could tell apart: both pictures at SIZE px on the long side (JPEG noise and resampling stay below it), CIELAB, a pixel counts when its distance is over JND. Pictures of different shapes changed completely (the crop moved). `revise` turns the old pool entry and the new picture's hash and change into the new entry: under KEEP the old hash joins `same` and the votes on it keep counting (arena.counts_now); over it `same` is emptied and `changed` set to now, so only votes on the new picture count."""

import cv2
import numpy as np

SIZE = 512
JND = 2.3  # CIELAB distance of a just noticeable difference
KEEP = 0.02  # up to 2 % of the pixels visibly different: the same picture for a vote


def _lab(bgr):
    h, w = bgr.shape[:2]
    s = SIZE / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)


def change(old, new):
    """Share (0..1) of the pixels whose colour moved by more than JND between two 8-bit BGR pictures; 1.0 when either is missing or their shapes differ."""
    if old is None or new is None or old.shape != new.shape:
        return 1.0
    d = np.linalg.norm(_lab(old) - _lab(new), axis=2)
    return float((d > JND).mean())


def revise(old, new_hash, share, now):
    """The fields the pool entry of a remade picture carries over: {'same': [...], 'changed': t, 'change': share} (`change` rounded, for the record). `old` is the previous entry (None for a new variant: nothing to carry). The same hash again (identical pixels) keeps the old fields."""
    if not old:
        return {}
    keep = {k: old[k] for k in ("same", "changed", "change") if k in old}
    if old.get("hash") == new_hash:
        return keep
    if share <= KEEP:
        same = [h for h in [*(old.get("same") or []), old["hash"]] if h != new_hash]
        return dict(keep, same=same, change=round(share, 4))
    return dict(same=[], changed=round(now, 1), change=round(share, 4))
