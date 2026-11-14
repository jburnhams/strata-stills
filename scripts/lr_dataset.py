"""The person's own edits as training data: raws paired with the finished Lightroom exports of them, with the Lightroom settings each export carries.
  python scripts/lr_dataset.py pairs      [--finals ~/personal/Backup/Photos] [--raws ~/raw]   finals named after their raw (DSC01234*.jpg in <finals>/<year>/...) paired with the raw of that stem and year
  python scripts/lr_dataset.py catalogue                                                         every paired final's Lightroom settings (the crs: XMP at the head of the JPEG; no exiftool: 21k files in 8 minutes over SMB)
  python scripts/lr_dataset.py features                                                          per pair: stats of the camera's preview cropped as the final was, EXIF, and the final's lightness and colour (resumable, batches of 250)
Everything goes to candidates/personal/ (gitignored, on the data share): these are private photos, never committed and never in a public report. Findings: docs/progress.md, "Learning the person's own edits".
Lightroom's crop (crs:CropTop/Left/Bottom/Right, CropAngle) is in the frame of the sensor: rotate by +CropAngle about the crop's centre, cut the box, then turn by the EXIF orientation (99 % of 160 crops matched the export, against 74 % with the other sign)."""

import argparse
import collections
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stratastills import nsfw
from stratastills.look.features import (  # noqa: F401  (moved: the pipeline's look stage computes them for any picture)
    EXIF_TAGS,
    FINAL_C,
    FINAL_L,
    PCTS,
    SKIP_TAG,
    THUMB,
    img_feats,
    num,
    orient,
    raw_feats,
    unorient,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "candidates", "personal")
RAW_EXTS = (".arw", ".cr2", ".dng", ".nef")
STEM = re.compile(r"(DSC\d{5}|_MG_\d{4}|IMG_\d{4}|DJI_\d{4})", re.I)
BATCH = 250


def pairs(a):
    raws = collections.defaultdict(list)
    for root, _, fs in os.walk(a.raws):
        for f in fs:
            if os.path.splitext(f)[1].lower() in RAW_EXTS:
                raws[os.path.splitext(f)[0].upper()].append(os.path.join(root, f))
    out, first = [], {}
    for root, _, fs in os.walk(a.finals):
        year = os.path.relpath(root, a.finals).split(os.sep)[0]
        for f in sorted(fs):
            s, e = os.path.splitext(f)
            m = STEM.match(s)
            if (
                e.lower() not in (".jpg", ".jpeg", ".tif", ".tiff")
                or not m
                or "-Pano" in s
                or os.sep + "small" in root.lower()
            ):
                continue  # panoramas are stitched from several raws; small/ holds resized copies of the same edit
            c = [
                p for p in raws.get(m.group(1).upper(), []) if f"{os.sep}{year}{os.sep}" in p
            ]  # file numbers repeat over the years: same stem and same year
            if c and nsfw.flagged(os.path.join(root, f)):
                continue  # the nudity filter: such a picture feeds no report, model or arena
            if c:
                first.setdefault(c[0], os.path.join(root, f))
    out = [dict(final=f, raw=r) for r, f in first.items()]
    json.dump(out, open(os.path.join(OUT, "pairs.json"), "w"))
    print(f"{len(out)} raws with a final")


def _xmp(item):
    f = item["final"]
    row = dict(item)
    try:
        with open(f, "rb") as fh:
            b = fh.read(1 << 20)
    except OSError as e:
        row["error"] = str(e)
        return row
    i = b.find(b"<x:xmpmeta")
    j = b.find(b"</x:xmpmeta>", i)
    if i < 0 or j < 0:
        row["error"] = "no xmp"
        return row
    x = b[i:j].decode("utf8", "replace")
    row.update(re.findall(r'crs:(\w+)="([^"]*)"', x))
    for el in set(re.findall(r"<crs:(\w+)>", x)):
        row.setdefault(el, "present")  # masks, curves: element form; only their presence is kept
    return row


def catalogue(a):
    items = json.load(open(os.path.join(OUT, "pairs.json")))
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(_xmp, items))
    json.dump(rows, open(os.path.join(OUT, "catalogue.json"), "w"))
    print(f"{len(rows)} finals, {sum('error' in r for r in rows)} unreadable")


def load_catalogue():
    """The pairs with process-2012 settings (all but a handful), in a fixed order: the `idx` of the feature batches indexes this list."""
    return [r for r in json.load(open(os.path.join(OUT, "catalogue.json"))) if "error" not in r and "Exposure2012" in r]


def lr_crop(im, x):
    """The part of `im` (in the sensor's frame, not yet oriented) that Lightroom's crop `x` (a catalogue row) kept."""
    if str(x.get("HasCrop")) not in ("1", "True"):
        return im
    t, l, b, r, ang = (
        num(x.get(k), d)
        for k, d in (("CropTop", 0), ("CropLeft", 0), ("CropBottom", 1), ("CropRight", 1), ("CropAngle", 0))
    )
    h, w = im.shape[:2]
    cx, cy = w * (l + r) / 2, h * (t + b) / 2
    dx, dy = w * (r - l), h * (b - t)
    if ang:  # the two corners are opposite corners of the turned box: its own width and height along the turned axes
        im = cv2.warpAffine(im, cv2.getRotationMatrix2D((cx, cy), ang, 1.0), (w, h), flags=cv2.INTER_AREA)
        th = math.radians(ang)
        c, s = math.cos(th), math.sin(th)
        dx, dy = abs(dx * c + dy * s), abs(-dx * s + dy * c)
    out = im[
        max(0, int(round(cy - dy / 2))) : int(round(cy + dy / 2)),
        max(0, int(round(cx - dx / 2))) : int(round(cx + dx / 2)),
    ]
    return out if out.size else im


def final_stats(im):
    """The FINAL_L and FINAL_C numbers of a picture (BGR uint8)."""
    im = cv2.resize(im, (384, max(8, round(384 * im.shape[0] / im.shape[1]))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * 100 / 255
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    c = np.hypot(a, b)
    return [float(np.percentile(L, p)) for p in PCTS] + [float(L.mean())], [
        float(a.mean()),
        float(b.mean()),
        float(np.percentile(c, 50)),
        float(np.percentile(c, 90)),
        float(c.mean()),
        float(np.percentile(c, 99)),
    ]


def camera_picture(raw, tmp=None):
    """The camera's own rendering of a raw (BGR uint8, in the sensor's frame): the JPEG beside it when the camera wrote one, else the preview inside the raw."""
    for ext in (".JPG", ".jpg"):
        j = os.path.splitext(raw)[0] + ext
        if os.path.exists(j):
            return unorient(
                cv2.imread(j), (exif_of([raw])[raw].get("Orientation"))
            )  # the camera's JPEG is stored the way up: back to the sensor's frame, as the preview is
    b = subprocess.run(["exiftool", "-b", "-PreviewImage", raw], capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) if b else None


def exif_of(raws):
    return {
        x["SourceFile"]: x
        for x in json.loads(
            subprocess.run(["exiftool", "-q", "-json", "-n", *EXIF_TAGS, *raws], capture_output=True, text=True).stdout
            or "[]"
        )
    }


def features(a):
    rows = load_catalogue()
    d = os.path.join(OUT, "features")
    os.makedirs(d, exist_ok=True)
    for k in range(0, len(rows), BATCH):
        dst = os.path.join(d, f"b{k // BATCH:03d}.npz")
        if os.path.exists(dst):
            continue
        batch = rows[k : k + BATCH]
        tmp = tempfile.mkdtemp(dir=d)
        lst = os.path.join(tmp, "list.txt")
        open(lst, "w").write("\n".join(r["raw"] for r in batch))
        subprocess.run(
            ["exiftool", "-q", "-b", "-PreviewImage", "-w", f"{tmp}/%:1d_%f.jpg", "-@", lst], capture_output=True
        )
        ex = {
            x["SourceFile"]: x
            for x in json.loads(
                subprocess.run(
                    ["exiftool", "-q", "-json", "-n", *EXIF_TAGS, "-@", lst], capture_output=True, text=True
                ).stdout
                or "[]"
            )
        }

        def fin(r):
            im = cv2.imread(r["final"], cv2.IMREAD_REDUCED_COLOR_4)
            return None if im is None else final_stats(im)

        with ThreadPoolExecutor(8) as pool:
            fins = list(pool.map(fin, batch))
        X, F, C, keep, names = [], [], [], [], None
        for i, (r, fs) in enumerate(zip(batch, fins)):
            p = f"{tmp}{os.path.dirname(r['raw'])}/_{os.path.splitext(os.path.basename(r['raw']))[0]}.jpg"
            e = ex.get(r["raw"])
            im = cv2.imread(p) if os.path.exists(p) else None
            if im is None or e is None or fs is None:
                continue
            ft = img_feats(orient(lr_crop(im, r), e.get("Orientation")))
            ft.update(raw_feats(e))
            names = names or list(ft)
            X.append([ft[n] for n in names])
            F.append(fs[0])
            C.append(fs[1])
            keep.append(k + i)
        np.savez(
            dst,
            X=np.array(X, np.float32),
            F=np.array(F, np.float32),
            C=np.array(C, np.float32),
            idx=np.array(keep),
            names=np.array(names or []),
        )
        shutil.rmtree(tmp)
        print(f"batch {k // BATCH}: {len(keep)}/{len(batch)}", flush=True)


_features = {}  # OUT -> the stacked feature batches, read once per process


def load_features():
    """(X, names, F, C, idx) over every batch: features, the finals' lightness and colour, and the catalogue row of each. Read once per process, and kept stacked in one uncompressed file (`features_all.npz`, beside the batches, made again when a batch changes): 86 batches of small members took about 60-130 s to open and crc-check, the stacked file about a second."""
    import glob

    files = sorted(glob.glob(os.path.join(OUT, "features", "b*.npz")))
    sig = [(os.path.basename(p), os.stat(p).st_size, os.stat(p).st_mtime_ns) for p in files]
    if _features.get(OUT, (None,))[0] == sig:
        return _features[OUT][1]
    allf = os.path.join(OUT, "features_all.npz")
    out = None
    if os.path.exists(allf):
        try:
            z = np.load(allf, allow_pickle=False)
            if json.loads(str(z["sig"])) == [list(s) for s in sig]:
                out = (z["X"], [str(n) for n in z["names"]], z["F"], z["C"], z["idx"])
        except Exception:
            out = None
    if out is None:
        parts = [np.load(p, allow_pickle=True) for p in files]
        out = (
            np.concatenate([p["X"] for p in parts]),
            [str(n) for n in parts[0]["names"]],
            np.concatenate([p["F"] for p in parts]),
            np.concatenate([p["C"] for p in parts]),
            np.concatenate([p["idx"] for p in parts]),
        )
        try:
            tmp = f"{allf}.{os.getpid()}.tmp.npz"
            np.savez(tmp, sig=json.dumps(sig), X=out[0], names=np.array(out[1]), F=out[2], C=out[3], idx=out[4])
            os.replace(tmp, allf)
        except OSError:
            pass
    _features[OUT] = (sig, out)
    return out


def _meta_batch(raws):
    return {
        x["SourceFile"]: x
        for x in json.loads(
            subprocess.run(
                ["exiftool", "-q", "-json", "-n", "-G1", "-fast2", *raws], capture_output=True, text=True
            ).stdout
            or "[]"
        )
    }


def meta(a):
    """OUT/meta.npz: as much of every raw's metadata as is usable as model input, whatever the maker wrote (camera and lens, exposure, focus, drive and scene modes, picture style, white balance levels, flash, GPS, ...), one number per column: numbers as they are, short number lists split into columns, text as the frequency rank of its value, and the shot's time as hour of day, day of year, weekday and year. A tag counts when at least 20 % of the pictures have it and it takes more than one value; ids, offsets, sizes, paths and free text are skipped. Aligned with the catalogue rows."""
    import datetime

    rows = load_catalogue()
    raws = [r["raw"] for r in rows]
    recs = [None] * len(raws)
    chunks = [(k, raws[k : k + 400]) for k in range(0, len(raws), 400)]

    def work(c):
        k, rs = c
        ex = _meta_batch(rs)
        print(f"meta {k + len(rs)}/{len(raws)}", flush=True)
        return k, [ex.get(r) for r in rs]

    with ThreadPoolExecutor(6) as pool:
        for k, out in pool.map(work, chunks):
            recs[k : k + len(out)] = out
    cols = collections.defaultdict(lambda: [None] * len(raws))
    for i, e in enumerate(recs):
        if not e:
            continue
        for tag, v in e.items():
            if tag == "SourceFile" or SKIP_TAG.search(tag.split(":", 1)[-1]) or isinstance(v, (dict, list)):
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                cols[tag][i] = float(v)
                continue
            t = str(v).strip()
            if re.fullmatch(r"-?[\d.]+( -?[\d.]+){1,5}", t):
                for q, x in enumerate(t.split()):
                    cols[f"{tag}[{q}]"][i] = float(x)
            elif t and len(t) < 80:
                cols[tag][i] = t
        m = re.match(
            r"(\d{4}):(\d\d):(\d\d) (\d\d):(\d\d)",
            str(e.get("ExifIFD:DateTimeOriginal") or e.get("IFD0:ModifyDate") or ""),
        )
        if m:
            y, mo, d, h, mi = map(int, m.groups())
            try:
                dd = datetime.date(y, mo, d)
                cols["when:hour"][i] = h + mi / 60
                cols["when:doy"][i] = dd.timetuple().tm_yday
                cols["when:weekday"][i] = dd.weekday()
                cols["when:year"][i] = y
            except ValueError:
                pass
    names, mats = [], []
    for name, v in sorted(cols.items()):
        have = [x for x in v if x is not None]
        if len(have) < 0.2 * len(raws):
            continue
        if any(isinstance(x, str) for x in have):
            rank = {
                t: n
                for n, (t, _) in enumerate(collections.Counter(x for x in have if isinstance(x, str)).most_common())
            }
            col = [rank.get(x, np.nan) if isinstance(x, str) else np.nan for x in v]
        else:
            col = [np.nan if x is None else x for x in v]
        arr = np.array(col, np.float32)
        if not np.isfinite(arr).any() or np.nanmin(arr) == np.nanmax(arr):
            continue
        names.append(name)
        mats.append(arr)
    np.savez(os.path.join(OUT, "meta.npz"), M=np.stack(mats, 1), names=np.array(names))
    print(f"{len(names)} metadata columns:", ", ".join(names), flush=True)


def thumbs(a):
    """OUT/thumbs/t*.npz: for every catalogue row the camera's preview (cut to Lightroom's crop, turned the way up) and the person's final, both squeezed to THUMB x THUMB (uint8 BGR), in batches of BATCH rows: the pair data for fitting what the person does to a picture (scripts/lr_fit.py) and for image features. `idx` says which catalogue row each is, `ok` which rows had both pictures."""
    rows = load_catalogue()
    d = os.path.join(OUT, "thumbs")
    os.makedirs(d, exist_ok=True)
    for k in range(0, len(rows), BATCH):
        dst = os.path.join(d, f"t{k // BATCH:03d}.npz")
        if os.path.exists(dst):
            continue
        batch = rows[k : k + BATCH]
        tmp = tempfile.mkdtemp(dir=d)
        lst = os.path.join(tmp, "list.txt")
        open(lst, "w").write("\n".join(r["raw"] for r in batch))
        subprocess.run(
            ["exiftool", "-q", "-b", "-PreviewImage", "-w", f"{tmp}/%:1d_%f.jpg", "-@", lst], capture_output=True
        )
        ex = {
            x["SourceFile"]: x
            for x in json.loads(
                subprocess.run(
                    ["exiftool", "-q", "-json", "-n", "-Orientation", "-@", lst], capture_output=True, text=True
                ).stdout
                or "[]"
            )
        }
        with ThreadPoolExecutor(8) as pool:
            fins = list(pool.map(lambda r: cv2.imread(r["final"], cv2.IMREAD_REDUCED_COLOR_4), batch))
        C, F, keep = [], [], []
        for i, (r, fin) in enumerate(zip(batch, fins)):
            p = f"{tmp}{os.path.dirname(r['raw'])}/_{os.path.splitext(os.path.basename(r['raw']))[0]}.jpg"
            e = ex.get(r["raw"])
            im = cv2.imread(p) if os.path.exists(p) else None
            if im is None or e is None or fin is None:
                continue
            cam = orient(lr_crop(im, r), e.get("Orientation"))
            sq = lambda x: cv2.resize(x, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
            C.append(sq(cam))
            F.append(sq(fin))
            keep.append(k + i)
        np.savez(dst, cam=np.array(C, np.uint8), fin=np.array(F, np.uint8), idx=np.array(keep))
        shutil.rmtree(tmp)
        print(f"thumbs batch {k // BATCH}: {len(keep)}/{len(batch)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["pairs", "catalogue", "features", "meta", "thumbs"])
    ap.add_argument("--finals", default=os.path.expanduser("~/personal/Backup/Photos"))
    ap.add_argument("--raws", default=os.path.expanduser("~/raw"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"pairs": pairs, "catalogue": catalogue, "features": features, "meta": meta, "thumbs": thumbs}[a.step](a)
