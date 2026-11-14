"""Faces in the person's pairs: where they are and what colour they are in the camera's preview and in the final (scripts/lr_dataset.py's pairs), from bright-horizons' detector (SCRFD, insightface buffalo_l). Faces are what must not look wrong, the rest of a picture is more style, so the look model gets face statistics as inputs and targets (what the skin of the final looks like), and faces get their own score.
  ~/code/github/bright-horizons/.venv/bin/python scripts/lr_faces.py [--limit N]     (that venv has insightface and onnxruntime-gpu for the GTX 1050; this script needs only cv2, numpy and insightface)
-> candidates/personal/faces.json: {row: [{box: [x, y, w, h] as fractions of the cropped, turned-up preview, score, cam: [L, a, b], fin: [L, a, b]}, ...]} for every catalogue row with a face of score >= MIN_SCORE and short side >= MIN_SIZE px (on a PREVIEW_LONG px preview), the 6 largest. cam/fin are the median CIELAB of the inner half of the box (the cheeks and forehead, not hair or background), the final resized onto the preview's frame.
Resumable: rows already in the file are skipped."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2

sys.path.insert(0, os.path.dirname(__file__))
import lr_dataset as D

from stratastills.look.features import face_stats as stats

MIN_SCORE = 0.6  # as bright-horizons: lower is mostly toys, hair and the backs of heads
MIN_SIZE = 24
PREVIEW_LONG = 960
BATCH = 100


def main():
    from insightface.app import FaceAnalysis

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    app = FaceAnalysis(
        name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"], allowed_modules=["detection"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    rows = D.load_catalogue()[: a.limit]
    f = os.path.join(D.OUT, "faces.json")
    out = json.load(open(f)) if os.path.exists(f) else {}
    for k in range(0, len(rows), BATCH):
        if all(str(k + i) in out for i in range(min(BATCH, len(rows) - k))):
            continue
        batch = rows[k : k + BATCH]
        tmp = tempfile.mkdtemp()
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
        for i, r in enumerate(batch):
            p = f"{tmp}{os.path.dirname(r['raw'])}/_{os.path.splitext(os.path.basename(r['raw']))[0]}.jpg"
            e = ex.get(r["raw"])
            im = cv2.imread(p) if os.path.exists(p) else None
            fin = cv2.imread(r["final"], cv2.IMREAD_REDUCED_COLOR_4)
            if im is None or e is None or fin is None:
                out[str(k + i)] = []
                continue
            cam = D.orient(D.lr_crop(im, r), e.get("Orientation"))
            s = PREVIEW_LONG / max(cam.shape[:2])
            cam = cv2.resize(
                cam,
                (round(cam.shape[1] * s), round(cam.shape[0] * s)),
                interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC,
            )
            fin = cv2.resize(fin, (cam.shape[1], cam.shape[0]), interpolation=cv2.INTER_AREA)
            faces = []
            for fc in app.get(cam):
                x0, y0, x1, y1 = [float(v) for v in fc.bbox]
                w, h = x1 - x0, y1 - y0
                if fc.det_score < MIN_SCORE or min(w, h) < MIN_SIZE:
                    continue
                sc, sf = stats(cam, (x0, y0, w, h)), stats(fin, (x0, y0, w, h))
                if sc and sf:
                    faces.append(
                        dict(
                            box=[x0 / cam.shape[1], y0 / cam.shape[0], w / cam.shape[1], h / cam.shape[0]],
                            score=float(fc.det_score),
                            cam=sc,
                            fin=sf,
                        )
                    )
            out[str(k + i)] = sorted(faces, key=lambda q: -q["box"][2] * q["box"][3])[:6]
        shutil.rmtree(tmp)
        json.dump(out, open(f + ".tmp", "w"))
        os.replace(f + ".tmp", f)
        print(f"{k + len(batch)}/{len(rows)}: {sum(1 for v in out.values() if v)} rows with a face", flush=True)


if __name__ == "__main__":
    main()
