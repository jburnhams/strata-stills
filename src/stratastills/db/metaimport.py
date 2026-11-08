"""EXIF/XMP metadata of the hashed raws into `photo_meta` (scripts/read_meta.py makes the JSONL with exiftool; scripts/import_meta.py loads it). `plan()` reads records (path, sha256, tags: {"Group:Tag": value}) and needs no database; `write()` adds a row for each photo that has none (an existing row is left as it is: re-reading a file is a deliberate act, not a side effect of a rerun), verifies and commits in ONE transaction.
Columns (the first tag present wins): taken_at = ExifIFD:DateTimeOriginal (CreateDate if missing) as the camera's WALL-CLOCK time, naive and not converted to UTC (it is UK local time by the photographer's decision of 2026-11-08: left as is; cameras rarely record the zone; where one does, OffsetTimeOriginal is in exif_raw); width/height = Composite:ImageSize; make, model, lens (ExifIFD:LensModel, else a text Composite:LensID), iso, exposure_s, f_number, focal_mm, orientation (IFD0), GPS (signed degrees and metres: the Composite values, or the GPS-group magnitudes with S/W/AltitudeRef applied). `exif_raw` = every non-XMP tag as read (undecoded tags were not kept), `xmp_raw` = the XMP groups, `dji` = the XMP-drone-dji group when a file has one. is_nadir is left empty (it needs the app's rule on the gimbal pitch)."""
from datetime import datetime

from sqlalchemy import select

from .hashes import _chunks
from .models import Photo, PhotoMeta


class MetaMismatch(Exception): pass


def _first(tags, *names):
    for n in names:
        v = tags.get(n)
        if v not in (None, ''): return v
    return None


def _num(v, cast):
    try: return cast(v)
    except (TypeError, ValueError): return None


def _time(v):
    if not isinstance(v, str): return None
    for fmt in ('%Y:%m:%d %H:%M:%S.%f', '%Y:%m:%d %H:%M:%S', '%Y:%m:%d %H:%M'):
        try: return datetime.strptime(v.strip()[:26], fmt)
        except ValueError: pass
    return None


def has_error(tags): return any(k == 'ExifTool:Error' or k.endswith(':Error') for k in tags)


def _gps(tags):
    """(lat, lon, alt) as signed numbers: the Composite:GPS* values are already signed; the GPS-group ones are magnitudes, so S / W / below-sea-level (AltitudeRef 1) are applied to them."""
    lat, lon, alt = (_num(_first(tags, f'Composite:GPS{n}'), float) for n in ('Latitude', 'Longitude', 'Altitude'))
    if lat is None:
        lat = _num(_first(tags, 'GPS:GPSLatitude'), float)
        if lat is not None and str(_first(tags, 'GPS:GPSLatitudeRef') or '').upper().startswith('S'): lat = -abs(lat)
    if lon is None:
        lon = _num(_first(tags, 'GPS:GPSLongitude'), float)
        if lon is not None and str(_first(tags, 'GPS:GPSLongitudeRef') or '').upper().startswith('W'): lon = -abs(lon)
    if alt is None:
        alt = _num(_first(tags, 'GPS:GPSAltitude'), float)
        if alt is not None and str(_first(tags, 'GPS:GPSAltitudeRef')).strip() in ('1', 'Below Sea Level'): alt = -abs(alt)
    return lat, lon, alt


def columns(tags):
    """The PhotoMeta columns of one file's tags."""
    size = _first(tags, 'Composite:ImageSize'); w = h = None
    if isinstance(size, str) and size.replace('x', ' ').split()[:2]:
        parts = size.replace('x', ' ').split()
        if len(parts) >= 2: w, h = _num(parts[0], int), _num(parts[1], int)
    if w is None: w, h = _num(_first(tags, 'SubIFD:ImageWidth', 'ExifIFD:ExifImageWidth', 'File:ImageWidth'), int), _num(_first(tags, 'SubIFD:ImageHeight', 'ExifIFD:ExifImageHeight', 'File:ImageHeight'), int)
    lens = _first(tags, 'ExifIFD:LensModel', 'Composite:LensID', 'XMP-aux:Lens'); lens = lens if isinstance(lens, str) else None
    lat, lon, alt = _gps(tags); dji = {k.split(':', 1)[1]: v for k, v in tags.items() if k.startswith('XMP-drone-dji:')} or None
    return dict(taken_at=_time(_first(tags, 'ExifIFD:DateTimeOriginal', 'ExifIFD:CreateDate')), width=w, height=h, make=_first(tags, 'IFD0:Make'), model=_first(tags, 'IFD0:Model'), lens=lens,
                iso=_num(_first(tags, 'ExifIFD:ISO'), int), exposure_s=_num(_first(tags, 'ExifIFD:ExposureTime'), float), f_number=_num(_first(tags, 'ExifIFD:FNumber'), float), focal_mm=_num(_first(tags, 'ExifIFD:FocalLength'), float),
                orientation=_num(_first(tags, 'IFD0:Orientation'), int), gps_lat=lat, gps_lon=lon, gps_alt=alt, dji=dji)


def read_jsonl(path, problems):
    import json
    recs = []
    try: f = open(path, encoding='utf8')
    except OSError as e: problems.append(f'FATAL unreadable {path}: {e}'); return recs
    with f:
        for i, ln in enumerate(f, 1):
            if not ln.strip(): continue
            try: r = json.loads(ln); recs.append(dict(sha256=r['sha256'], tags=r['tags'], path=r.get('path')))
            except (ValueError, KeyError, TypeError): problems.append(f'line {i}: unparsable or incomplete, skipped')
    return recs


def plan(records, problems=None):
    """{rows: [{sha256, columns..., exif_raw, xmp_raw}], problems, counts}. The first record of a hash wins; the others are reported ('duplicate photo, skipped')."""
    problems = list(problems or []); rows, seen = [], set(); dup = 0; filled = dict(taken_at=0, make=0, model=0, lens=0, iso=0, gps_lat=0)
    for r in records:
        if has_error(r['tags']): problems.append(f'{r.get("path")}: exiftool reported an error, skipped'); continue
        if r['sha256'] in seen: dup += 1; problems.append(f'{r.get("path")}: duplicate photo, skipped'); continue
        seen.add(r['sha256']); tags = r['tags']; cols = columns(tags)
        exif = {k: v for k, v in tags.items() if not k.startswith('XMP')}; xmp = {k: v for k, v in tags.items() if k.startswith('XMP')} or None
        rows.append(dict(sha256=r['sha256'], exif_raw=exif, xmp_raw=xmp, **cols))
        for k in filled: filled[k] += cols[k] is not None
    return dict(rows=rows, problems=problems, counts=dict(photo_meta=len(rows), duplicate_photo_skipped=dup, problems=len(problems), filled=filled))


def write(session, p):
    """Add the missing rows; returns {photo_meta: added, already: n}. Rolls back and raises MetaMismatch if a photo has no row, or a planned one is missing afterwards."""
    fatal = [x for x in p['problems'] if x.startswith('FATAL')]
    if fatal: raise MetaMismatch('refusing to write a plan with an unreadable source: ' + '; '.join(fatal))
    ids = {}
    for c in _chunks({r['sha256'] for r in p['rows']}): ids.update({s: i for s, i in session.execute(select(Photo.sha256, Photo.id).where(Photo.sha256.in_(c)))})
    absent = [r['sha256'] for r in p['rows'] if r['sha256'] not in ids]
    if absent: session.rollback(); raise MetaMismatch(f'{len(absent)} planned hash(es) have no photo row, e.g. {absent[:2]}; load the hashes first')
    have = {pid for (pid,) in session.execute(select(PhotoMeta.photo_id))}; new = []
    for r in p['rows']:
        pid = ids[r['sha256']]
        if pid in have: continue
        have.add(pid); new.append(PhotoMeta(photo_id=pid, **{k: v for k, v in r.items() if k != 'sha256'}))
    session.add_all(new); session.flush(); got = {pid for (pid,) in session.execute(select(PhotoMeta.photo_id))}
    if not {ids[r['sha256']] for r in p['rows']} <= got: session.rollback(); raise MetaMismatch('photo_meta: a planned row is missing after the write')
    session.commit(); return dict(photo_meta=len(new), already=len(p['rows']) - len(new))
