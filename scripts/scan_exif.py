"""Find pictures of a camera at the ISOs you need in the folders of a raw library: reads the EXIF (header only) of every raw that has a camera JPEG next to it and writes a table (folder, file, model, ISO, lens, shutter, aperture) as JSON lines.
  python scripts/scan_exif.py <folder> [<folder> ...] --out iso.jsonl [--model ILCE-7C] [--ext .ARW]"""
import argparse, glob, json, os
from concurrent.futures import ThreadPoolExecutor
import tifffile


def info(path):
    try:
        with tifffile.TiffFile(path) as t:
            pg = t.pages[0]; ex = pg.tags['ExifTag'].value; model = str(pg.tags['Model'].value)
            return dict(folder=os.path.basename(os.path.dirname(path)), file=os.path.basename(path), model=model, iso=int(ex.get('ISOSpeedRatings', 0)), lens=str(ex.get('LensModel', '')), exposure=float(ex['ExposureTime'][0] / ex['ExposureTime'][1]) if 'ExposureTime' in ex else None, fnumber=float(ex['FNumber'][0] / ex['FNumber'][1]) if 'FNumber' in ex else None)
    except Exception: return None


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('folders', nargs='+'); ap.add_argument('--out', required=True); ap.add_argument('--model'); ap.add_argument('--ext', default='.ARW'); a = ap.parse_args()
    files = [f for d in a.folders for f in sorted(glob.glob(os.path.join(d, '*' + a.ext))) if os.path.exists(os.path.splitext(f)[0] + '.JPG')]
    with ThreadPoolExecutor(8) as ex: rows = [r for r in ex.map(info, files) if r and (not a.model or r['model'].strip().upper() == a.model.upper())]
    with open(a.out, 'w') as f:
        for r in rows: f.write(json.dumps(r) + '\n')
    print(len(files), 'raws with a JPG,', len(rows), 'kept')
