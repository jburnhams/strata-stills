"""The person's learned look, for any picture: the pipeline's `look` stage and the arena's auto versions run this one code.

A picture's look is predicted from its FEATURES (look/features.py: stats of the camera's preview, EXIF, the raw's metadata, what CLIP sees)
and its FACES, by models trained on the person's Lightroom catalogue (scripts/lr_look.py, lr_tmodel.py, lr_ftarget.py,
lr_neutral_model.py, lr_face_first.py train them). The trained models are frozen into a BUNDLE (`settings.look_dir()`, made by
scripts/export_look_bundle.py), so the app needs neither the catalogue nor the training data; the bundle is private (trained on the
person's photos) and never in the repository.

The versions (the arena's names), each applied to our neutral develop:
  auto1  the look model: tone curve through the predicted percentiles, Shadows/Highlights, Clarity, vignette, colour by predicted means
  auto2  auto1 with the colour from the camera-trained transform model
  auto3  auto2 with the face anchor (the largest face moved toward the predicted face of the final)
  auto4  the network trained on our neutral -> the final, its whole transform (the person's typical edit when it would flatten the picture)
  auto5  auto3 with its colour from that network
  auto6  auto4, then face first (the main real face's skin toward the predicted skin)
  auto7  auto5, then face first
Faces and CLIP come from the strata-models service (roles faces, face-kind, image-embed); without it a picture gets no faces and NaN
for the CLIP columns (the models were trained to cope with missing values), and the stage says so in what it did."""

import json
import os
import pickle
import threading

import numpy as np

from stratastills import settings
from stratastills.look import apply as A
from stratastills.look import face_first as FF
from stratastills.look import features as F

VERSION = 1  # of the stage's code here; the bundle's manifest is part of version() too
VARIANTS = ("auto1", "auto2", "auto3", "auto4", "auto5", "auto6", "auto7")
DEFAULT = "best"  # the version the arena's votes rate highest now (`winner`)
FALLBACK = "auto5"  # the winner when there are no ratings to go by (the arena's leader on 2026-11-14)
RATINGS_TTL = 600.0
PRIORITY = os.environ.get(
    "STRATASTILLS_LOOK_PRIORITY", "batch"
)  # of the strata-models jobs (faces, CLIP): behind interactive work; a one-off check can ask for interactive
PREVIEW_LONG = 960  # the preview the faces are found on (scripts/lr_faces.py)
FACE_MIN_SCORE, FACE_MIN_SIZE = 0.6, 24  # as scripts/lr_faces.py: lower is mostly toys, hair and the backs of heads


class LookUnavailable(RuntimeError):
    """No look bundle here (settings.look_dir()): the stage leaves the picture as it is and says why."""


class Bundle:
    """The frozen models of one bundle folder, loaded as first needed."""

    def __init__(self, path):
        self.path = path
        f = os.path.join(path, "manifest.json")
        if not os.path.isfile(f):
            raise LookUnavailable(f"no look bundle in {path} (make one with scripts/export_look_bundle.py)")
        self.manifest = json.load(open(f))
        self.names = self.manifest["names"]
        self._m = {}
        self._lock = threading.Lock()

    def _get(self, key, load):
        with self._lock:
            if key not in self._m:
                self._m[key] = load()
            return self._m[key]

    def _pickle(self, name):
        return self._get(name, lambda: pickle.load(open(os.path.join(self.path, name), "rb")))

    def look(self):
        return self._pickle("look.pkl")

    def tmodel(self):
        return self._pickle("tmodel.pkl")

    def ftarget(self):
        return self._pickle("ftarget.pkl")

    def skin(self):
        return self._pickle("skin.pkl") if os.path.exists(os.path.join(self.path, "skin.pkl")) else None

    def meta_encoder(self):
        return self._get("meta", lambda: json.load(open(os.path.join(self.path, "meta-encoder.json"))))

    def clip_basis(self):
        def load():
            z = np.load(os.path.join(self.path, "clip-basis.npz"))
            groups = json.load(open(os.path.join(self.path, "clip-prompts.json")))
            return dict(mu=z["mu"], vt=z["vt"], texts=[(g, ps, z[f"text_{g}"]) for g, ps in groups])

        return self._get("clip", load)

    def net(self):
        def load():
            from stratastills import gpu
            from stratastills.look import net

            dev = "cuda" if gpu.available() and gpu.torch_().cuda.is_available() else "cpu"
            r = np.load(os.path.join(self.path, "neutral-range.npz"))
            return dict(
                net.load(os.path.join(self.path, "neutral-net.pt"), dev), lo=r["lo"], hi=r["hi"], typical=r["typical"]
            )

        return self._get("net", load)


_bundles = {}
_ratings = {}


def ratings():
    """{version: dict(rating, votes)} of the auto versions from the arena's votes (Bradley-Terry, as /api/arena/ratings), read at most every RATINGS_TTL seconds; {} when the arena has no votes or cannot be read."""
    import time

    hit = _ratings.get("r")
    if hit and time.monotonic() - hit[0] < RATINGS_TTL:
        return hit[1]
    out = {}
    try:
        from stratastills import arena as AR
        from stratastills import arena_store

        st = arena_store.get()
        retired = st.retired()
        votes = AR.current_votes(st.votes(), st.pool())
        bt = AR.bradley_terry(votes, retired)
        n = {}
        for v in votes:
            for k in (v["a"], v["b"]):
                n[k] = n.get(k, 0) + 1
        out = {k: dict(rating=round(bt[k]), votes=n.get(k, 0)) for k in VARIANTS if k in bt and k not in retired}
    except Exception:
        out = {}
    _ratings["r"] = (time.monotonic(), out)
    return out


def winner():
    """The auto version the arena rates highest now (FALLBACK without ratings): what 'best' means."""
    r = ratings()
    return max(r, key=lambda k: r[k]["rating"]) if r else FALLBACK


def version():
    """The look stage's version: this code's VERSION and the bundle's identity (its files' hashes), or 'none' when there is no bundle."""
    try:
        b = bundle()
    except LookUnavailable:
        return f"{VERSION}-none"
    import hashlib

    return f"{VERSION}-" + hashlib.sha1(json.dumps(b.manifest["files"], sort_keys=True).encode()).hexdigest()[:10]


def preview(path):
    """The camera's own picture of a file, turned the way up: a raw's embedded preview, else the file itself (long side at most 2048)."""
    from stratastills import images

    p = images.preview_bgr(path, 2048) if images.is_camera_raw(path) else None
    return p if p is not None else images.load(path, 2048)


def bundle():
    """The bundle of settings.look_dir() (LookUnavailable when there is none)."""
    p = settings.look_dir()
    if p not in _bundles:
        _bundles[p] = Bundle(p)
    return _bundles[p]


# ---- what the models see of a picture ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def exif(raw):
    """(the EXIF the shot features read, the raw's whole metadata as scripts/lr_dataset.py meta reads it): two exiftool calls."""
    import subprocess

    one = lambda *a: (
        json.loads(
            subprocess.run(["exiftool", "-q", "-json", "-n", *a, raw], capture_output=True, text=True).stdout or "[{}]"
        )
        or [{}]
    )[0]
    return one(*F.EXIF_TAGS), one("-G1", "-fast2")


def _service():
    """The strata-models client when the service can answer now, else None: not installed, not running, or this process holds the whole card (the service could not run the job until the lease ends)."""
    try:
        import strata_models_client as sm
    except ImportError:
        return None
    from stratastills import gpu

    if gpu.lease_is_exclusive() or not sm.available():
        return None
    return sm


def clip_embedding(preview_bgr):
    """The CLIP B/32 image embedding of the preview squeezed to THUMB x THUMB (as the catalogue's), from strata-models; None when it is not running."""
    import tempfile

    import cv2

    sm = _service()
    if sm is None:
        return None
    sq = cv2.resize(preview_bgr, (F.THUMB, F.THUMB), interpolation=cv2.INTER_AREA)
    with tempfile.TemporaryDirectory(prefix="look-clip-") as d:
        np.savez(os.path.join(d, "t.npz"), cam=sq[None])
        sm.run(
            "image-embed",
            dict(
                array=dict(path=os.path.join(d, "t.npz"), key="cam", bgr=True),
                resize="square",
                out=os.path.join(d, "e.npz"),
            ),
            app="strata-stills",
            model="clip-vit-b-32",
            priority=PRIORITY,
        )
        return np.load(os.path.join(d, "e.npz"))["images"][0]


def feature_row(preview_bgr, ex, meta, embedding, b=None):
    """The feature row of a picture in the bundle's column order: `preview_bgr` the camera's preview turned the way up, `ex`/`meta` from `exif`, `embedding` from `clip_embedding` (None: NaN columns)."""
    b = b or bundle()
    row = dict(F.img_feats(preview_bgr), **F.raw_feats(ex))
    base = [row.get(n, np.nan) for n in b.manifest["base_names"]]
    m = F.meta_row(meta, b.meta_encoder())
    c = (
        F.clip_row(embedding, b.clip_basis())
        if embedding is not None
        else np.full(b.manifest["n_clip"], np.nan, np.float32)
    )
    x = np.concatenate([np.asarray(base, np.float32), m, c]).astype(np.float32)
    if len(x) != len(b.names):
        raise LookUnavailable(
            f"the bundle expects {len(b.names)} features, this code makes {len(x)}: export the bundle again"
        )
    return x


def faces(preview_bgr):
    """The faces of a picture on its preview (largest first), from strata-models: [{box (x, y, w, h fractions), score, cam (median CIELAB of the inner half of the box), lm (68 landmarks, fractions), pose, real, kind}]; [] when the service is not running."""
    import tempfile

    import cv2

    sm = _service()
    if sm is None:
        return []
    h, w = preview_bgr.shape[:2]
    k = PREVIEW_LONG / max(h, w)
    im = cv2.resize(preview_bgr, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else preview_bgr
    H, W = im.shape[:2]
    with tempfile.TemporaryDirectory(prefix="look-faces-") as d:
        p = os.path.join(d, "p.jpg")
        cv2.imwrite(p, im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        got = sm.faces(
            p, "strata-stills", priority=PRIORITY, fields=["bbox", "det_score", "landmark_3d_68", "pose"]
        ).get("faces", [])
        got = [
            f
            for f in got
            if f["det_score"] >= FACE_MIN_SCORE
            and min(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1]) >= FACE_MIN_SIZE
        ]
        got.sort(key=lambda f: -(f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        got = got[:6]
        kinds = (
            sm.run("face-kind", dict(path=p, boxes=[f["bbox"] for f in got]), app="strata-stills", priority=PRIORITY)[
                "faces"
            ]
            if got
            else []
        )
    out = []
    for f, kd in zip(got, kinds):
        x1, y1, x2, y2 = f["bbox"]
        cam = F.face_stats(im, (x1, y1, x2 - x1, y2 - y1))
        if cam is None:
            continue
        out.append(
            dict(
                box=[x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H],
                score=f["det_score"],
                cam=cam,
                lm=[[p[0] / W, p[1] / H] for p in f.get("landmark_3d_68") or []],
                pose=f.get("pose") or [0.0, 0.0, 0.0],
                real=kd["real"],
                kind=kd["kind"],
            )
        )
    return out


def map_faces(fl, ops):
    """Faces found on the camera's preview (fractions of its frame) moved into the frame of the finished picture through what the pipeline did to it: ('centre', f) a centred crop keeping fraction f of each side (pipeline.crop_to_image_circle), ('rotate', deg, w, h) a turn about the centre with the scale that fills the frame (transforms.rotate_fill). Resizing changes no fraction."""
    import cv2

    from stratastills import transforms

    def move(pts, op):
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        if op[0] == "centre":
            f = op[1]
            return (pts - (1 - f) / 2) / f
        _, deg, w, h = op
        M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, transforms.inscribed_scale(w, h, deg))
        px = pts * [w, h]
        return (px @ M[:, :2].T + M[:, 2]) / [w, h]

    out = []
    for f in fl:
        x, y, w, h = f["box"]
        corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        lm = np.asarray(f.get("lm") or [], np.float64).reshape(-1, 2)
        for op in ops:
            corners = move(corners, op)
            lm = move(lm, op) if len(lm) else lm
        x0, y0 = corners.min(0)
        x1, y1 = corners.max(0)
        if x1 <= 0 or y1 <= 0 or x0 >= 1 or y0 >= 1:
            continue  # cropped away
        out.append(dict(f, box=[float(x0), float(y0), float(x1 - x0), float(y1 - y0)], lm=lm.tolist()))
    return out


# ---- the versions ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def _face_target(models, x, face):
    """The predicted face (L, a, b) of the final (scripts/lr_ftarget.py face_target)."""
    L, a, b = face["cam"]
    f = np.array(
        [
            [
                1,
                face["box"][2] * face["box"][3],
                L,
                a,
                b,
                np.hypot(a, b),
                np.degrees(np.arctan2(b, a)) % 360,
                face["score"],
            ]
        ],
        np.float32,
    )
    return np.array([L, a, b]) + np.array([m.predict(np.hstack([x[None], f]))[0] for m in models])


def predictions(x, fl, b=None):
    """What the models predict for a picture: the look row, the camera-trained colour transform, and the face target of its largest face."""
    b = b or bundle()
    X = np.asarray(x, np.float32)[None]
    return dict(
        pred=A.predict(b.look(), X)[0],
        colour=np.stack([m.predict(X) for m in b.tmodel()], 1).astype(np.float32)[0],
        face=dict(box=fl[0]["box"], target=_face_target(b.ftarget(), np.asarray(x, np.float32), fl[0])) if fl else None,
    )


def apply_params(bgr, p):
    """The transform (look/transform.py) with parameters `p` applied to a picture (scripts/lr_tmodel.py apply_params): on the GPU when the picture fits."""
    import cv2
    import torch

    from stratastills.look import transform as T

    dev = "cuda" if torch.cuda.is_available() and bgr.shape[0] * bgr.shape[1] < 4_000_000 else "cpu"
    with torch.no_grad():
        out = (
            T.transform(T.to_lab(bgr[None]).to(dev), torch.from_numpy(np.asarray(p, np.float32))[None].to(dev))[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
    return np.clip(cv2.cvtColor(out.astype(np.float32), cv2.COLOR_LAB2BGR) * 255 + 0.5, 0, 255).astype(np.uint8)


def neutral_params(neu, x, b=None):
    from stratastills.look import net

    m = (b or bundle()).net()
    return net.predict_params(m, neu, x, m["lo"], m["hi"])


def render(variant, neu, x, fl, b=None, preds=None):
    """One version of a picture: `neu` our neutral picture (uint8 BGR), `x` its features, `fl` its faces (look.faces: boxes as fractions of `neu`'s frame). -> (picture, what was done)."""
    b = b or bundle()
    if variant not in VARIANTS:
        raise ValueError(f"unknown look {variant!r}")
    p = preds or predictions(x, fl, b)
    cam = dict(zip(b.names, x))
    if variant == "auto1":
        return A.apply_look(neu, p["pred"], cam)
    if variant == "auto2":
        return A.apply_look(neu, p["pred"], cam, colour=p["colour"])
    if variant == "auto3":
        return A.apply_look(neu, p["pred"], cam, colour=p["colour"], face=p["face"])
    if variant in ("auto4", "auto6"):
        out = apply_params(neu, neutral_params(neu, x, b))
        did = dict(transform="network")
        if out.std() < 0.3 * neu.std():
            out, did = (
                apply_params(neu, b.net()["typical"]),
                dict(transform="typical (the network's would flatten this picture)"),
            )
    else:  # auto5, auto7
        out, did = A.apply_look(neu, p["pred"], cam, colour=neutral_params(neu, x, b), face=p["face"])
    if variant in ("auto6", "auto7"):
        main = FF.main_face(fl)
        if main is not None and main.get("lm"):
            out, more = FF.apply(out, neu, x, main, b.skin())
            did = dict(did, **more)
    return out, did
