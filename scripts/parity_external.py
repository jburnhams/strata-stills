"""Score another program's renderings of the same raws against the camera's JPEGs, exactly as parity_jpeg.py scores ours (same size, same numbers), so the programs can be compared on the candidate set.
  python scripts/parity_external.py <folder of renderings: <stem>.jpg> <folder with the camera JPEGs> --candidates docs/a7c_candidates.json [--sheets dir]
The renderings are made by the other program's command line (scripts/render_external.ps1 for darktable-cli and rawtherapee-cli); this only reads them."""
import argparse, glob, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import parity_jpeg as PJ

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('renders'); ap.add_argument('cameras'); ap.add_argument('--candidates'); ap.add_argument('--sheets'); a = ap.parse_args()
    stems = [c['stem'] for c in json.load(open(a.candidates))] if a.candidates else sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(a.renders, '*.jpg')))
    rows = []
    if a.sheets: os.makedirs(a.sheets, exist_ok=True)
    for s in stems:
        r = next(iter(glob.glob(os.path.join(a.renders, s + '.[jJ][pP][gG]'))), None); c = next(iter(glob.glob(os.path.join(a.cameras, s + '.[jJ][pP][gG]'))), None)
        if not r or not c: print(s, 'missing'); continue
        j = cv2.imread(c, cv2.IMREAD_COLOR)[..., ::-1]; o = cv2.imread(r, cv2.IMREAD_COLOR)[..., ::-1]
        if (o.shape[1] > o.shape[0]) != (j.shape[1] > j.shape[0]): print(s, 'turned differently from the camera JPEG'); continue
        k = 1 / 8; size = (int(j.shape[1] * k), int(j.shape[0] * k)); a8 = cv2.resize(np.ascontiguousarray(o), size, interpolation=cv2.INTER_AREA); j8 = cv2.resize(np.ascontiguousarray(j), size, interpolation=cv2.INTER_AREA)
        if a.sheets: PJ.write_sheet(os.path.join(a.sheets, s + '.jpg'), cv2.resize(np.ascontiguousarray(o), (j.shape[1] // 2, j.shape[0] // 2), interpolation=cv2.INTER_AREA), cv2.resize(np.ascontiguousarray(j), (j.shape[1] // 2, j.shape[0] // 2), interpolation=cv2.INTER_AREA))
        d = PJ.metrics(a8, j8); rows.append(d); print(s, ' '.join(f'{k} {v:6.2f}' for k, v in d.items()), flush=True)
    print('MEDIAN', ' '.join(f'{k} {np.median([r[k] for r in rows]):6.2f}' for k in rows[0]))
