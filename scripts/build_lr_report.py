"""A visual report on the person's own edits: for pictures from shoots the look model never saw, the camera's picture, our neutral develop, the person's Lightroom final and our automatic version of their look (scripts/lr_look.py), all on the final's crop, as full overviews and crops where they differ most, with the raw's metadata, every Lightroom setting and what ours did.
  python scripts/build_lr_report.py [--n 30] [--out out/report/auto] [--stems DSC01234,...] [--seed 1]
Served by the app at /report/auto/, a peer of the neutral report (/report/neutral/, build_report.py). The automatic processing that matches the person's edits; the Lightroom export is the reference it learns from, and the report will grow with the auto-process as it supersedes this first model. PRIVATE: the pictures are the person's own, people included; never make this public.
Needs candidates/personal/ (scripts/lr_dataset.py pairs, catalogue, features)."""

import argparse
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import build_report as BR
import lr_dataset as D
import lr_ftarget as FT
import lr_look as LK
import lr_tmodel as TM

from stratastills import framing as FR
from stratastills import images

ROOT = BR.ROOT
LONG = 3000  # every picture brought to the final's framing at this long side
WIN = (200, 140)
ZOOM = 2
OVERVIEW = 1100
JPEG_Q = 82
NAMES = [
    "Camera",
    "Neutral (ours)",
    "Lightroom (yours)",
    "Auto (ours)",
    "Auto 2 (learned colour)",
    "Auto 3 (face anchor)",
]
KEY_SETTINGS = [
    "ProcessVersion",
    "CameraProfile",
    "WhiteBalance",
    "Temperature",
    "Tint",
    "Exposure2012",
    "Contrast2012",
    "Highlights2012",
    "Shadows2012",
    "Whites2012",
    "Blacks2012",
    "Texture",
    "Clarity2012",
    "Dehaze",
    "Vibrance",
    "Saturation",
    "ToneCurveName2012",
    "PostCropVignetteAmount",
    "LensProfileEnable",
    "HasCrop",
    "CropAngle",
]
CATEGORIES = [
    ("usual", "a usual picture", 0.4),
    ("dark_cam", "dark in camera", 0.12),
    ("bright_cam", "bright in camera", 0.12),
    ("kept_dark", "you kept it darker than the camera suggests", 0.12),
    ("brightest", "one of your brightest finals", 0.1),
    ("masks", "you used masks or brushes", 0.07),
    ("extreme", "a slider far from your usual", 0.07),
]


def categorise(rows, X, names, Y, idx):
    """{category: [positions in X]} for the picking: by the camera's and the final's median lightness (outer tenths), masks, and sliders outside their usual 2.5-97.5 % range."""
    inL = X[:, names.index("L_p50")]
    outL = Y[:, D.FINAL_L.index("L_p50")]
    lo, hi = np.percentile(inL, [10, 90])
    flo, fhi = np.percentile(outL, [10, 90])
    masks = np.array(
        [
            any(
                k in rows[i]
                for k in (
                    "GradientBasedCorrections",
                    "CircularGradientBasedCorrections",
                    "PaintBasedCorrections",
                    "MaskGroupBasedCorrections",
                )
            )
            for i in idx
        ]
    )
    sl = ["Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012", "Whites2012", "Blacks2012"]
    V = np.array([[D.num(rows[i].get(k)) for k in sl] for i in idx])
    lim = np.nanpercentile(V, [2.5, 97.5], axis=0)
    extreme = ((lim[0] > V) | (lim[1] < V)).any(axis=1)
    c = dict(
        dark_cam=inL < lo,
        bright_cam=inL > hi,
        kept_dark=(outL < flo) & (inL >= lo),
        brightest=outL > fhi,
        masks=masks,
        extreme=extreme,
    )
    c["usual"] = ~np.any(np.stack(list(c.values())), axis=0)
    return {k: np.flatnonzero(v) for k, v in c.items()}


def pick(cats, test, n, rng, X, names):
    """`n` held-out pictures over the categories (CATEGORIES shares), at most two per shoot, both cameras."""
    chosen, why, per_shoot = [], {}, {}
    a7c = X[:, names.index("a7c")] > 0.5
    for key, label, share in CATEGORIES:
        pool = [i for i in cats[key] if test[i] and i not in why]
        rng.shuffle(pool)
        want = max(1, round(n * share))
        got = 0
        pool.sort(key=lambda i: a7c[i] != (got % 3 == 0))  # about a third from the A7C
        for i in pool:
            if got >= want:
                break
            if per_shoot.get(i_shoot(i), 0) >= 2:
                continue
            chosen.append(i)
            why[i] = label
            per_shoot[i_shoot(i)] = per_shoot.get(i_shoot(i), 0) + 1
            got += 1
    return chosen, why


def i_shoot(i):
    return SHOOTS[i]


deferred = []  # raws whose develop came out blank while other GPU work ran beside it: the caller develops them again once alone (build_arena does, after its parallel pass)
_alone = threading.Lock()  # a retried develop runs with no other retry beside it


def readable(path, min_std=0.5):
    return images.readable_jpeg(path, min_std)  # stratastills.images: None for a missing, truncated or blank picture


def _develop_alone(raw, work):
    """The 'best' neutral of one raw in a project of its own (the retry of a bad develop) -> its generate() record."""
    from stratastills import pipeline as PL
    from stratastills import projects as P

    src = os.path.join(work, "retry-%d" % time.time_ns())
    os.makedirs(src)
    os.symlink(raw, os.path.join(src, os.path.basename(raw)))
    pid = P.create("lr-retry-%d" % time.time_ns(), [src])["id"]
    it = P.source_items(pid, P.source_id(P.load(pid)["sources"][0]["path"]))[0]
    return PL.generate(
        pid, it["id"], preset="neutral", options=dict(quality=95, level=False, size="native"), progress=lambda m: None
    )


def neutral(raws, work, say, rebuild=False, retry_now=True):
    """Our pipeline's 'neutral' preset ('best' without the look), without the levelling (the final's crop decides the angle), on each raw: {raw: (BGR picture, record)}. The raws are linked into one folder for a temporary project; the originals are only read. A raw whose neutral picture was made before by the same code (BR.code_hash) is read from out/cache/ instead: only the look model changes between runs, the neutral picture rarely does."""
    cdir = os.path.join(BR.ROOT, "out", "cache", "report-auto-neutral")
    os.makedirs(cdir, exist_ok=True)
    out = {}
    todo = []
    key = lambda r: hashlib.sha1(
        json.dumps([BR.CACHE_VERSION, BR.code_hash(), r, os.path.getsize(r), int(os.path.getmtime(r))]).encode()
    ).hexdigest()
    for r in raws:
        base = os.path.join(cdir, key(r))
        im = None if rebuild or not os.path.exists(base + ".json") else readable(base + ".jpg")
        if im is not None:
            out[r] = (im, json.load(open(base + ".json", encoding="utf-8")))
        else:
            todo.append(r)
    say(f"neutral pictures: {len(out)} cached, {len(todo)} to make")
    if not todo:
        return out
    os.environ["STRATASTILLS_PROJECTS"] = os.path.join(work, "projects")
    os.environ["STRATASTILLS_HOME"] = os.path.join(work, "home")
    from stratastills import pipeline as PL
    from stratastills import projects as P

    src = os.path.join(work, "raws-%d" % time.time_ns())
    os.makedirs(src, exist_ok=True)
    names = {}
    for k, r in enumerate(todo):
        n = f"{k:03d}_{os.path.basename(r)}"
        os.symlink(r, os.path.join(src, n))
        names[os.path.splitext(n)[0]] = r
    pid = P.create("lr-report-%d" % time.time_ns(), [src])["id"]
    sid = P.source_id(P.load(pid)["sources"][0]["path"])
    for it in P.source_items(pid, sid):
        r = names.get(it["stem"])
        if not r:
            continue
        t0 = time.time()
        try:
            rec = PL.generate(
                pid,
                it["id"],
                preset="neutral",
                options=dict(quality=95, level=False, size="native"),
                progress=lambda m: None,
            )
        except Exception as e:
            say(f"  {os.path.basename(r)}: FAILED ({type(e).__name__}: {e})")
            continue
        im = readable(rec["file"])
        if (
            im is None
        ):  # a blank or truncated develop (seen with two arena workers on the card at once): once more on its own, and never cached when still bad
            say(
                f"  {os.path.basename(r)}: blank or unreadable develop"
                + (", making it again on its own" if retry_now else "")
            )
            try:  # keep the evidence (picture and record: which stages ran on the card) for the open bug
                keep = os.path.join(
                    os.path.expanduser("~/work/strata-stills/blank-develops"),
                    "%d_%s" % (time.time_ns(), os.path.basename(r)),
                )
                os.makedirs(keep, exist_ok=True)
                if os.path.exists(rec["file"]):
                    shutil.copyfile(rec["file"], os.path.join(keep, "develop.jpg"))
                json.dump(
                    dict(rec, env={k: v for k, v in os.environ.items() if k.startswith("STRATASTILLS")}),
                    open(os.path.join(keep, "record.json"), "w"),
                    default=str,
                )
            except Exception:
                pass
            if not retry_now:
                deferred.append(r)
                say(f"  {os.path.basename(r)}: deferred until the parallel work is done")
                continue
            with _alone:  # a fresh one-picture project, so nothing of the bad run is reused
                try:
                    rec = _develop_alone(r, work)
                except Exception as e:
                    say(f"  {os.path.basename(r)}: FAILED ({type(e).__name__}: {e})")
                    continue
            im = readable(rec["file"])
            if im is None:
                say(f"  {os.path.basename(r)}: FAILED (blank develop twice; not cached)")
                continue
        out[r] = (im, rec)
        base = os.path.join(cdir, key(r))
        tmp = base + ".tmp.jpg"
        shutil.copyfile(rec["file"], tmp)
        os.replace(tmp, base + ".jpg")
        json.dump(rec, open(base + ".json", "w", encoding="utf-8"), default=str)
        say(f"  {os.path.basename(r)}: neutral in {time.time() - t0:.0f} s")
    return out


def turn_to_ref(c, ref):
    """`c` turned the way it lies in `ref` (Lightroom may have turned the picture, and some raws, the NEX-5N's among them, say Orientation 1 regardless). When the landscape/portrait of the two differ, a crop cannot explain it, so `c` must be turned a quarter; when either is (nearly) square, the shapes say nothing and `c` as it is competes with both quarter-turns. Which: the candidate whose features register on `ref` without a further turn (the catalogue's own orientation is not in the exported files), else the one whose coarse structure matches best (soft or heavily cropped pictures do not register)."""
    sq = lambda x: abs(np.log(x.shape[1] / x.shape[0])) < 0.05
    if not (sq(c) or sq(ref)) and (c.shape[1] > c.shape[0]) == (ref.shape[1] > ref.shape[0]):
        return c
    H, W = ref.shape[:2]
    k = 1500 / max(W, H)
    g = lambda x: cv2.cvtColor(
        cv2.resize(x, (round(W * k), round(H * k)), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY
    )
    cands = [cv2.rotate(c, r) for r in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    if sq(c) or sq(ref):
        cands = [c] + cands
    reg = [BR.register(g(ref), g(x), inliers=True) for x in cands]
    score = [
        n if M is not None and abs(M[0, 0] - 1) <= 0.08 and abs(M[1, 0]) <= 0.05 else 0 for M, n in reg
    ]  # register allows any rotation: a candidate it lays on `ref` only by another turn is the wrong way up
    if max(score) >= 20:
        return cands[int(np.argmax(score))]
    return cands[int(np.argmax([structure_corr(g(ref), g(x)) for x in cands]))]


def structure_corr(a, b, n=64):
    """Correlation of the coarse edges (Laplacian of a blurred n x n copy) of two gray pictures of one size: about 1 for the same picture, about 0 for one turned a quarter."""
    e = lambda x: (lambda y: (y - y.mean()) / (y.std() + 1e-6))(
        cv2.Laplacian(
            cv2.GaussianBlur(cv2.resize(x, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), 1.5),
            cv2.CV_32F,
        )
    )
    return float((e(a) * e(b)).mean())


def to_final(im, row, orientation, ref, with_rect=False):
    """`im` (BGR, in the sensor's frame) cut to Lightroom's crop, turned the way up, then laid exactly on `ref` (the final, BGR) by a similarity transform found from features (lens corrections and the programs' framing differ by a few pixels and a little scale); the crop alone if no transform is found. -> (picture at ref's size, how it was aligned)."""
    c = D.orient(D.lr_crop(im, row), orientation)
    H, W = ref.shape[:2]
    c = turn_to_ref(c, ref)
    c = cv2.resize(c, (W, H), interpolation=cv2.INTER_AREA if c.shape[1] > W else cv2.INTER_CUBIC)
    k = 1500 / max(W, H)
    g = lambda x: cv2.cvtColor(
        cv2.resize(x, (round(W * k), round(H * k)), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY
    )
    M = BR.register(g(ref), g(c))
    if M is None or abs(M[0, 0] - 1) > 0.08 or abs(M[1, 0]) > 0.05:
        return (c, "Lightroom crop", (0, 0, W, H)) if with_rect else (c, "Lightroom crop")
    M = M.copy()
    M[:, 2] /= k
    out = cv2.warpAffine(c, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if not with_rect:
        return out, "Lightroom crop, then registered"
    ok = (
        cv2.warpAffine(
            np.full((H, W), 255, np.uint8),
            M,
            (W, H),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )  # where the moved picture really has pixels: outside it the border is a smear of the last row or column
    return out, "Lightroom crop, then registered", FR.inner_rect(ok, pad=3)


def face_errors(ims, ref, faces):
    """For the largest face (scripts/lr_faces.py: box as fractions of the picture): how far the skin of each picture in `ims` is from `ref`'s, from the median CIELAB of the inner half of the box: (lightness, hue in degrees, chroma, CIE76 distance), all absolute. None without a face."""
    if not faces:
        return None
    H, W = ref.shape[:2]
    x, y, w, h = faces[0]["box"]
    x0, x1, y0, y1 = int((x + w / 4) * W), int((x + 3 * w / 4) * W), int((y + h / 4) * H), int((y + 3 * h / 4) * H)
    med = lambda im: (
        np.median(cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3), 0)
        - np.array([0, 128, 128], np.float32)
    )
    r = med(ref)
    out = []
    for im in ims:
        m = med(im)
        dh = (np.degrees(np.arctan2(m[2], m[1])) - np.degrees(np.arctan2(r[2], r[1])) + 180) % 360 - 180
        out.append(
            (
                abs(float(m[0] - r[0])) * 100 / 255,
                abs(float(dh)),
                abs(float(np.hypot(m[1], m[2]) - np.hypot(r[1], r[2]))),
                float(np.sqrt(((m - r) ** 2 * np.array([(100 / 255) ** 2, 1, 1])).sum())),
            )
        )
    return out


def errors(ims, ref):
    """Mean |lightness difference| and mean colour difference (CIE76, blurred 1.5 px so that noise and a pixel of misregistration do not count) of each picture from `ref`."""
    lab = lambda x: cv2.GaussianBlur(cv2.cvtColor(x.astype(np.float32) / 255, cv2.COLOR_BGR2LAB), (0, 0), 1.5)
    r = lab(ref)
    out = []
    for im in ims:
        d = lab(im) - r
        out.append((float(np.abs(d[..., 0]).mean()), float(np.sqrt((d**2).sum(axis=2)).mean())))
    return out


def tone_map(ref, im):
    lab = lambda x: cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), 3.0)
    return np.abs(lab(im) - lab(ref)).sum(axis=2)


def windows(maps, win, n):
    """`n` windows (x0, y0, kind) where the given maps (kind, map) are strongest, taken in turn, none overlapping."""
    h, w = maps[0][1].shape
    ww, wh = win
    taken, out = [], []
    for i in range(n):
        kind, m = maps[i % len(maps)]
        m = cv2.blur(m, (ww, wh))
        m[: wh // 2 + 4] = 0
        m[-(wh // 2 + 4) :] = 0
        m[:, : ww // 2 + 4] = 0
        m[:, -(ww // 2 + 4) :] = 0
        for cx, cy in taken:
            m[max(0, cy - wh) : cy + wh, max(0, cx - ww) : cx + ww] = 0
        cy, cx = np.unravel_index(int(np.argmax(m)), m.shape)
        taken.append((cx, cy))
        out.append((int(cx - ww // 2), int(cy - wh // 2), kind))
    return out


def table(d, cls="s"):
    return (
        f"<table class={cls}>"
        + "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in d.items())
        + "</table>"
    )


def build(a):
    global SHOOTS
    rows, X, names, Y, idx, G = LK.load()
    SHOOTS = G
    test = np.array([LK.test_shoot(g) for g in G])
    rng = np.random.default_rng(a.seed)
    say = lambda m: print(m, flush=True)
    say(
        f"{len(X)} pairs, {test.sum()} in {len(set(G[test]))} held-out shoots; training the look model on the other {(~test).sum()}"
    )
    tmodels = TM.train_cached(a.hard_weight)
    FACES = (
        json.load(open(os.path.join(D.OUT, "faces.json"))) if os.path.exists(os.path.join(D.OUT, "faces.json")) else {}
    )
    fmodels = FT.train_cached()
    models = LK.train_cached(X[~test], Y[~test])
    P = LK.predict(models, X[test])
    cam = X[test][:, [names.index(f"L_p{p}") for p in D.PCTS]]
    held = dict(
        model=np.abs(P[:, :9] - Y[test][:, :9]).mean(0),
        camera=np.abs(cam - Y[test][:, :9]).mean(0),
        avg=np.abs(np.median(Y[~test][:, :9], 0) - Y[test][:, :9]).mean(0),
    )
    if a.stems:
        want = set(a.stems.split(","))
        chosen = [
            i for i in range(len(idx)) if test[i] and os.path.splitext(os.path.basename(rows[idx[i]]["raw"]))[0] in want
        ]
        why = dict.fromkeys(chosen, "asked for")
    else:
        chosen, why = pick(categorise(rows, X, names, Y, idx), test, a.n, rng, X, names)
    say(
        f"{len(chosen)} pictures: "
        + ", ".join(f"{lab} {sum(1 for i in chosen if why[i] == lab)}" for _, lab, _ in CATEGORIES)
    )
    final_dir = os.path.abspath(a.out)
    out = final_dir + ".building"
    assets = os.path.join(out, "assets")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(assets)
    work = tempfile.mkdtemp(prefix="lr-report-", dir=os.path.dirname(out))
    sections, nav, summary, face_summary = [], [], [], []
    try:
        raws = [rows[idx[i]]["raw"] for i in chosen]
        ex = D.exif_of(raws)
        full_exif = {
            x["SourceFile"]: x
            for x in json.loads(
                subprocess.run(
                    ["exiftool", "-json", "-G1", "-a", "--ThumbnailImage", "--PreviewImage", *raws],
                    capture_output=True,
                    text=True,
                ).stdout
                or "[]"
            )
        }
        dev = neutral(raws, work, say, a.rebuild)
        for i in chosen:
            row = rows[idx[i]]
            raw = row["raw"]
            stem = os.path.splitext(os.path.basename(raw))[0]
            sid = f"{os.path.basename(os.path.dirname(row['final']))[:10]}_{stem}".replace(" ", "")
            fin = cv2.imread(row["final"])
            camera = D.camera_picture(raw)
            if fin is None or camera is None or raw not in dev:
                say(f"  {stem}: skipped (missing picture)")
                continue
            k = LONG / max(fin.shape[:2])
            fin = cv2.resize(
                fin,
                (round(fin.shape[1] * k), round(fin.shape[0] * k)),
                interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC,
            )
            H, W = fin.shape[:2]
            o = ex.get(raw, {}).get("Orientation", 1)
            nim, nrec = dev[raw]
            cam_a, how_c = to_final(camera, row, o, fin)
            neu_a, how_n = to_final(
                D.unorient(nim, o), row, o, fin
            )  # our output is turned the way up already: back to the sensor's frame for Lightroom's crop
            pred = LK.predict(models, X[i : i + 1])[0]
            auto, did = LK.apply_look(neu_a, pred, dict(zip(names, X[i])))
            auto2, _ = LK.apply_look(neu_a, pred, dict(zip(names, X[i])), colour=TM.predict(tmodels, X[i : i + 1])[0])
            fl = FACES.get(str(int(idx[i])), [])
            fa = dict(box=fl[0]["box"], target=FT.face_target(fmodels, X[i], fl[0])) if fl else None
            auto3, did3 = LK.apply_look(
                neu_a, pred, dict(zip(names, X[i])), colour=TM.predict(tmodels, X[i : i + 1])[0], face=fa
            )
            did = dict(did, **{k: v for k, v in did3.items() if k.startswith("face_")})
            ims = [cam_a, neu_a, fin, auto, auto2, auto3]
            err = errors([cam_a, neu_a, auto, auto2, auto3], fin)
            summary.append((why[i], err))
            fe = face_errors([cam_a, neu_a, auto, auto2, auto3], fin, fl)
            if fe:
                face_summary.append(fe)
            maps = [
                ("auto against yours", tone_map(fin, auto)),
                ("neutral against yours", tone_map(fin, neu_a)),
                ("camera against yours", tone_map(fin, cam_a)),
            ]
            kk = 1000 / max(W, H)
            small = lambda m: cv2.resize(m, (round(W * kk), round(H * kk)), interpolation=cv2.INTER_AREA)
            sel = windows([(kind, small(m)) for kind, m in maps], (round(WIN[0] * kk), round(WIN[1] * kk)), a.crops)
            tabs = []
            for j, (nm, im) in enumerate(zip(NAMES, ims)):
                fn = f"{sid}-ov{j}.jpg"
                cv2.imwrite(
                    os.path.join(assets, fn),
                    cv2.resize(im, (OVERVIEW, round(OVERVIEW * H / W)), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q],
                )
                tabs.append((nm, fn))
            blocks = []
            for n, (cx, cy, kind) in enumerate(sel, 1):
                x0 = min(max(0, int(round(cx / kk))), W - WIN[0])
                y0 = min(max(0, int(round(cy / kk))), H - WIN[1])
                pw, ph = WIN[0] * ZOOM, WIN[1] * ZOOM
                sheet = np.hstack(
                    [
                        BR.label(
                            cv2.resize(
                                im[y0 : y0 + WIN[1], x0 : x0 + WIN[0]], (pw, ph), interpolation=cv2.INTER_CUBIC
                            ).copy(),
                            nm,
                        )
                        for nm, im in zip(NAMES, ims)
                    ]
                )
                fn = f"{sid}-crop{n}.jpg"
                cv2.imwrite(os.path.join(assets, fn), sheet, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                blocks.append(
                    dict(n=n, kind=kind, x=x0 / W, y=y0 / H, w=WIN[0] / W, h=WIN[1] / H, sheet=fn, pw=pw, ph=ph)
                )
            sections.append(
                render(
                    sid,
                    stem,
                    row,
                    why[i],
                    W,
                    H,
                    tabs,
                    blocks,
                    err,
                    pred,
                    did,
                    nrec,
                    full_exif.get(raw, {}),
                    (how_c, how_n),
                    Y[i],
                    fl,
                    fe,
                )
            )
            nav.append((sid, f"{stem} ({why[i]})"))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if not sections:
        say("nothing to report on")
        return 1
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write(
        page(sections, nav, summary, held, len(set(G[test])), int(test.sum()), face_summary)
    )
    shutil.rmtree(final_dir, ignore_errors=True)
    os.rename(out, final_dir)
    say(f"report: {os.path.join(final_dir, 'index.html')} ({len(sections)} pictures)")
    return 0


def render(sid, stem, row, why, W, H, tabs, blocks, err, pred, did, nrec, exif, how, actual, faces=(), fe=None):
    tabs_html = (
        "<div class=tabs>"
        + "".join(
            f'<button data-src="assets/{fn}"{" class=on" if j == 2 else ""}>{html.escape(nm)}</button>'
            for j, (nm, fn) in enumerate(tabs)
        )
        + "</div>"
    )
    fboxes = "".join(
        f'<div class="box fbox" style="left:{f["box"][0] * 100:.2f}%;top:{f["box"][1] * 100:.2f}%;width:{f["box"][2] * 100:.2f}%;height:{f["box"][3] * 100:.2f}%"></div>'
        for f in faces
    )
    ftab = (
        "<table class=t><tr><th>largest face, off from your final by</th><th>lightness</th><th>hue (deg)</th><th>chroma</th><th>colour</th></tr>"
        + "".join(
            f"<tr><th>{nm}</th>" + "".join(f"<td>{v:.1f}</td>" for v in e) + "</tr>"
            for nm, e in zip([NAMES[0], NAMES[1], NAMES[3], NAMES[4], NAMES[5]], fe)
        )
        + "</table>"
        if fe
        else ""
    )
    boxes = fboxes + "".join(
        f'<div class=box style="left:{b["x"] * 100:.2f}%;top:{b["y"] * 100:.2f}%;width:{b["w"] * 100:.2f}%;height:{b["h"] * 100:.2f}%"><b>{b["n"]}</b></div>'
        for b in blocks
    )
    crops = "".join(
        f'<h3>Crop {b["n"]}: where {html.escape(b["kind"])} differ most</h3><div class=crop><div class=ov style="width:260px"><img src="assets/{tabs[2][1]}" width=260><div class=box style="left:{b["x"] * 100:.2f}%;top:{b["y"] * 100:.2f}%;width:{b["w"] * 100:.2f}%;height:{b["h"] * 100:.2f}%"></div></div><img src="assets/{b["sheet"]}" width={b["pw"] * 4 // 2} style="max-width:100%"></div>'
        for b in blocks
    )
    errs = (
        "<table class=t><tr><th></th><th>lightness off by</th><th>colour off by</th></tr>"
        + "".join(
            f"<tr><th>{nm}</th><td>{e[0]:.1f}</td><td>{e[1]:.1f}</td></tr>"
            for nm, e in zip([NAMES[0], NAMES[1], NAMES[3], NAMES[4], NAMES[5]], err)
        )
        + "</table><p class=m>Mean difference from your final, CIELAB units (lightness 0-100; colour: CIE76 distance), on the whole picture.</p>"
    )
    key = {k: row.get(k, "") for k in KEY_SETTINGS if k in row}
    rest = {k: v for k, v in sorted(row.items()) if k not in key and k not in ("final", "raw")}
    look = (
        "<table class=t><tr><th></th>"
        + "".join(f"<th>{t}</th>" for t in LK.TARGETS)
        + "</tr>"
        + "".join(
            f"<tr><th>{nm}</th>" + "".join(f"<td>{v:.1f}</td>" for v in vals) + "</tr>"
            for nm, vals in (("predicted", pred), ("your final", actual))
        )
        + "</table>"
    )
    did_html = table(
        {
            "tone curve on L (from, to)": " ".join(f"{x:g}->{y:g}" for x, y in did["curve"]),
            "neutral lightness percentiles": did["source_L"],
            "chroma scale": did["chroma_scale"],
            "local sliders used": did["sliders"],
            "colour cast moved (a, b)": did["cast"],
            "camera aligned": how[0],
            "neutral aligned": how[1],
        }
    )
    ex = {k: v for k, v in exif.items() if k != "SourceFile" and not str(v).startswith("(Binary")}
    sh = f"{exif.get('IFD0:Model', '')} · {exif.get('ExifIFD:LensModel', exif.get('Sony:LensType', ''))} · {exif.get('ExifIFD:FocalLength', '')} · f/{exif.get('ExifIFD:FNumber', '')} · {exif.get('ExifIFD:ExposureTime', '')} s · ISO {exif.get('ExifIFD:ISO', '')}"
    return f"""<section class=pic id="{sid}"><h2>{html.escape(stem)} <small class=why>{html.escape(os.path.basename(os.path.dirname(row["final"])))} · {html.escape(why)}</small></h2>{BR.flag_widget(sid)}<p class=m>{html.escape(sh)} · final {W}x{H} (brought to {LONG} px long side)<br>raw: {html.escape(row["raw"])}<br>final: {html.escape(row["final"])}</p>
<div class=ov style="width:{BR.fit_width(W, H, OVERVIEW)}"><img class=ovimg src="assets/{tabs[2][1]}">{boxes}</div>{tabs_html}{errs}{ftab}
<details open><summary>Lightroom: your main settings</summary>{table(key)}</details>
<details><summary>Auto (ours): the predicted look and what was done</summary>{look}{did_html}</details>
<details><summary>Neutral (ours): the pipeline's settings</summary>{BR.settings_html(nrec)}</details>
<details><summary>Lightroom: every setting in the export ({len(rest) + len(key)})</summary>{table(rest)}</details>
<details><summary>The raw's metadata ({len(ex)} tags)</summary>{table(ex)}</details>{crops}</section>"""


def face_table(fs):
    """The faces, which matter most: over the shown pictures that have a face, the mean distance of the largest face's skin from your final's."""
    if not fs:
        return ""
    mean = lambda j, k: float(np.mean([e[j][k] for e in fs]))
    rows = "".join(
        f"<tr><th>{nm}</th>" + "".join(f"<td>{mean(j, k):.1f}</td>" for k in range(4)) + "</tr>"
        for j, nm in enumerate(["Camera", "Neutral", "Auto", "Auto 2 (learned colour)", "Auto 3 (face anchor)"])
    )
    return f"<h3>Faces ({len(fs)} of the shown pictures have one)</h3><table class=t><tr><th>mean distance of the largest face from your final</th><th>lightness</th><th>hue (deg)</th><th>chroma</th><th>colour</th></tr>{rows}</table>"


def page(sections, nav, summary, held, n_shoots, n_pics, face_summary=()):
    med = lambda v: float(np.mean(v)) if v else float("nan")
    rows = "".join(
        f"<tr><th>{html.escape(lab)} ({sum(1 for w, _ in summary if w == lab)})</th>"
        + "".join(
            f"<td>{med([e[j][0] for w, e in summary if w == lab]):.1f} / {med([e[j][1] for w, e in summary if w == lab]):.1f}</td>"
            for j in range(5)
        )
        + "</tr>"
        for _, lab, _ in CATEGORIES
        if any(w == lab for w, _ in summary)
    )
    rows += (
        "<tr><th>all shown</th>"
        + "".join(
            f"<td>{med([e[j][0] for _, e in summary]):.1f} / {med([e[j][1] for _, e in summary]):.1f}</td>"
            for j in range(5)
        )
        + "</tr>"
    )
    hrows = "".join(
        f"<tr><th>{nm}</th>" + "".join(f"<td>{v:.1f}</td>" for v in held[k]) + "</tr>"
        for k, nm in (
            ("camera", "camera picture as it is"),
            ("avg", "your average final"),
            ("model", "auto (predicted)"),
        )
    )
    return f"""<!doctype html><meta charset=utf-8><title>stratastills: auto-process</title><style>{BR.CSS}</style><header><h1>Auto-process: camera, neutral, your Lightroom final and auto</h1>
<div class=note>PRIVATE: your own pictures. Generated {time.strftime("%Y-%m-%d %H:%M")} by scripts/build_lr_report.py, on pictures from shoots the look model was not trained on. Every picture is shown on the crop of your final, at {LONG} px on the long side: <b>Camera</b> (the camera's JPEG beside the raw, else the preview inside the raw: smaller, so softer), <b>Neutral</b> (our 'best' pipeline, not levelled), <b>Lightroom</b> (your export), <b>Auto</b> (our neutral bent to the look the model predicts you would have given it: version 1, one global tone curve, colour strength and cast; no local Highlights/Shadows yet). Crops are taken where auto, neutral and the camera differ most from yours. <a href="../neutral/">Neutral report</a> · <a href="../">All reports</a></div>
{BR.flag_header("auto")}<nav>{"".join(f'<a href="#{s}">{html.escape(t)}</a>' for s, t in nav)}</nav></header><main>
<h2>How close each comes to your final</h2><table class=t><tr><th>pictures shown</th><th>Camera</th><th>Neutral</th><th>Auto</th><th>Auto 2 (learned colour)</th><th>Auto 3 (face anchor)</th></tr>{rows}</table><p class=m>Mean lightness / colour difference from your final (CIELAB units), whole pictures.</p>{face_table(face_summary)}
<h3>All {n_pics} pictures of the {n_shoots} held-out shoots (the model's prediction only, no pictures made)</h3><table class=t><tr><th>error of the final's lightness at</th>{"".join(f"<th>p{p}</th>" for p in D.PCTS)}</tr>{hrows}</table>
{"".join(sections)}</main>{BR.flag_script("auto")}<script>document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{{const t=b.parentNode;t.querySelectorAll('button').forEach(x=>x.classList.remove('on'));b.classList.add('on');b.closest('section').querySelector('.ovimg').src=b.dataset.src}})</script>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hard-weight", type=float, default=3.0)
    ap.add_argument(
        "--rebuild", action="store_true", help="make the neutral pictures again, ignoring the cache in out/cache/"
    )
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "report", "auto"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--stems")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--crops", type=int, default=3)
    sys.exit(build(ap.parse_args()))
