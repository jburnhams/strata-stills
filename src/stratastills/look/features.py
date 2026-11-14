"""What the look models see of a picture (moved from scripts/lr_dataset.py, which builds the training data with them): stats of the camera's preview (`img_feats`), the shot's EXIF (`raw_feats`), the rest of the raw's metadata as numbers (`meta_row`, with the encoder frozen in the look bundle), and what CLIP sees (`clip_row`, with the PCA basis and prompt embeddings frozen in the bundle).
The models were trained on the preview cut to the person's Lightroom crop; a new picture has no crop, so its stats are of the whole preview."""

import collections
import datetime
import math
import re

import cv2
import numpy as np

PCTS = (1, 5, 10, 25, 50, 75, 90, 95, 99)

FINAL_L = [f"L_p{p}" for p in PCTS] + [
    "L_mean"
]  # the final's lightness (CIELAB L, 0-100) at these percentiles, and its mean

FINAL_C = [
    "a_mean",
    "b_mean",
    "C_p50",
    "C_p90",
    "C_mean",
    "C_p99",
]  # its colour: mean a/b (the cast) and chroma percentiles

EXIF_TAGS = [
    "-ISO",
    "-ExposureTime",
    "-FNumber",
    "-LightValue",
    "-BrightnessValue",
    "-Flash",
    "-FocalLengthIn35mmFormat",
    "-ExposureCompensation",
    "-Model",
    "-DateTimeOriginal",
    "-Orientation",
]

SKIP_TAG = re.compile(
    r"(Offset|Length|Size$|Checksum|Thumbnail|Preview|File|Directory|ExifTool|ImageUniqueID|Version|Sequence|Count$|Bytes|Table|Padding|Binary|Data$|MakerNote|Comment|Software|Profile|Copyright|Artist|Description|Title|URL|DocumentID|InstanceID|OriginalDocument|Derived|History|Rating|Label|Keywords|SubSec|Date|Time(?!r)|Trailer|Index|Warning|Error)",
    re.I,
)

THUMB = 128


def num(v, d=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def orient(im, o):
    return {
        6: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE),
        8: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE),
        3: cv2.rotate(im, cv2.ROTATE_180),
    }.get(int(o or 1), im)


def unorient(im, o):
    return {
        6: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE),
        8: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE),
        3: cv2.rotate(im, cv2.ROTATE_180),
    }.get(int(o or 1), im)


def img_feats(im):
    """Stats of a picture (BGR uint8, any size) that tell how it is exposed and what kind of scene it is: lightness percentiles and histogram, clipping, colour cast and strength, local contrast, where the light is."""
    im = cv2.resize(im, (384, max(8, round(384 * im.shape[0] / im.shape[1]))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * 100 / 255
    a = lab[..., 1] - 128
    b = lab[..., 2] - 128
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mx = im.max(axis=2)
    c = np.hypot(a, b)
    f = {f"L_p{p}": float(np.percentile(L, p)) for p in PCTS}
    f.update(
        L_mean=L.mean(),
        L_std=L.std(),
        clip_hi=(mx >= 250).mean(),
        near_hi=(g >= 235).mean(),
        clip_lo=(g <= 5).mean(),
        near_lo=(g <= 20).mean(),
        a_mean=a.mean(),
        b_mean=b.mean(),
        chroma_mean=c.mean(),
        chroma_p90=np.percentile(c, 90),
    )
    f.update({f"Lh{i}": v for i, v in enumerate(np.histogram(L, 16, (0, 100))[0] / L.size)})
    base = cv2.GaussianBlur(L, (0, 0), 12)
    f["local_contrast"] = np.abs(L - base).mean()
    f["base_spread"] = np.percentile(base, 95) - np.percentile(base, 5)
    f["lap"] = np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean()
    H, W = L.shape
    f["top_minus_bottom"] = L[: H // 3].mean() - L[-H // 3 :].mean()
    f["centre_minus_edge"] = L[H // 4 : -H // 4, W // 4 : -W // 4].mean() - L.mean()
    hue = np.degrees(np.arctan2(b, a)) % 360
    f.update({f"hue{i}": v for i, v in enumerate(np.histogram(hue, 12, (0, 360), weights=c)[0] / max(c.sum(), 1e-6))})
    return {k: float(v) for k, v in f.items()}


def raw_feats(e):
    t = num(e.get("ExposureTime"))
    return dict(
        log_iso=math.log2(max(num(e.get("ISO"), 100), 1)),
        log_t=math.log2(t) if t == t and t > 0 else np.nan,
        fnum=num(e.get("FNumber")),
        lv=num(e.get("LightValue")),
        bv=num(e.get("BrightnessValue")),
        flash=float(int(num(e.get("Flash"), 0)) & 1),
        f35=num(e.get("FocalLengthIn35mmFormat")),
        ev_comp=num(e.get("ExposureCompensation")),
        a7c=float(e.get("Model") == "ILCE-7C"),
        year=num(str(e.get("DateTimeOriginal", ""))[:4]),
    )


def meta_cols(e):
    """The metadata columns of one raw: exiftool's JSON record (-json -n -G1 -fast2) -> {column: number or text}, as scripts/lr_dataset.py meta reads them: numbers as they are, short number lists split into columns, short text kept (encoded by rank later), and the shot's time as hour, day of year, weekday and year."""
    cols = {}
    for tag, v in (e or {}).items():
        if tag == "SourceFile" or SKIP_TAG.search(tag.split(":", 1)[-1]) or isinstance(v, (dict, list)):
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            cols[tag] = float(v)
            continue
        t = str(v).strip()
        if re.fullmatch(r"-?[\d.]+( -?[\d.]+){1,5}", t):
            for q, x in enumerate(t.split()):
                cols[f"{tag}[{q}]"] = float(x)
        elif t and len(t) < 80:
            cols[tag] = t
    m = re.match(
        r"(\d{4}):(\d\d):(\d\d) (\d\d):(\d\d)",
        str((e or {}).get("ExifIFD:DateTimeOriginal") or (e or {}).get("IFD0:ModifyDate") or ""),
    )
    if m:
        y, mo, d, h, mi = map(int, m.groups())
        try:
            dd = datetime.date(y, mo, d)
            cols["when:hour"] = h + mi / 60
            cols["when:doy"] = dd.timetuple().tm_yday
            cols["when:weekday"] = dd.weekday()
            cols["when:year"] = y
        except ValueError:
            pass
    return cols


def meta_encoder(records):
    """{names, ranks}: the columns kept (at least 20 % of the records have them and they take more than one value) and, for text columns, each value's rank by frequency: what scripts/lr_dataset.py meta does to the catalogue, kept so one new picture can be encoded the same way."""
    cols = collections.defaultdict(lambda: [None] * len(records))
    for i, e in enumerate(records):
        for k, v in meta_cols(e).items():
            cols[k][i] = v
    names, ranks = [], {}
    for name, v in sorted(cols.items()):
        have = [x for x in v if x is not None]
        if len(have) < 0.2 * len(records):
            continue
        if any(isinstance(x, str) for x in have):
            rank = {
                t: n
                for n, (t, _) in enumerate(collections.Counter(x for x in have if isinstance(x, str)).most_common())
            }
            col = [rank.get(x, np.nan) if isinstance(x, str) else np.nan for x in v]
            ranks[name] = rank
        else:
            col = [np.nan if x is None else x for x in v]
        arr = np.array(col, np.float32)
        if not np.isfinite(arr).any() or np.nanmin(arr) == np.nanmax(arr):
            ranks.pop(name, None)
            continue
        names.append(name)
    return dict(names=names, ranks=ranks)


def meta_row(e, enc):
    """One raw's metadata (an exiftool record) as the row of numbers the models were trained on (NaN where it has no value, or a text value the catalogue never had)."""
    c = meta_cols(e)
    out = []
    for n in enc["names"]:
        v = c.get(n)
        if n in enc["ranks"]:
            out.append(float(enc["ranks"][n][v]) if isinstance(v, str) and v in enc["ranks"][n] else np.nan)
        else:
            out.append(float(v) if isinstance(v, (int, float)) else np.nan)
    return np.array(out, np.float32)


def clip_row(e, basis):
    """What CLIP sees, as the models saw it (scripts/lr_clip.py write): the image embedding `e` projected on the 32 principal components of the catalogue's, then the softmax over each group of scene prompts. `basis` = {mu, vt, texts: [(group, prompts, text embeddings)]} from the look bundle."""
    e = np.asarray(e, np.float64)
    pca = (e - basis["mu"]) @ basis["vt"].T
    labels = []
    for _g, _ps, T in basis["texts"]:
        sim = 100 * e @ np.asarray(T, np.float64).T
        p = np.exp(sim - sim.max())
        labels.append(p / p.sum())
    return np.concatenate([pca, *labels]).astype(np.float32)


def face_stats(img, box):
    """Median CIELAB (L 0-100, a, b) of the inner half of `box` (x, y, w, h in pixels) of a BGR image: the cheeks and forehead, not hair or background (moved from scripts/lr_faces.py)."""
    x, y, w, h = box
    x0, x1, y0, y1 = int(x + w / 4), int(x + 3 * w / 4), int(y + h / 4), int(y + 3 * h / 4)
    p = img[max(0, y0) : y1, max(0, x0) : x1]
    if p.size == 0:
        return None
    lab = cv2.cvtColor(p, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
    return [
        float(np.median(lab[:, 0]) * 100 / 255),
        float(np.median(lab[:, 1]) - 128),
        float(np.median(lab[:, 2]) - 128),
    ]
