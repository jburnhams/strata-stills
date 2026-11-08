"""Videos (a drone clip = an MP4 with an SRT telemetry file and an LRF proxy) into schema `photos`: one `photo` of kind 'video' per MP4 (identity = the MP4's SHA-256; stem = the file name without the extension), `file` rows (the MP4 with role 'video'; the SRT and the LRF with role 'sidecar', `tag` 'srt' / 'lrf', each with its own sha256), a `photo_meta` row and the telemetry as one `analysis` (kind 'dji_srt').
`plan()` reads the hashes JSONL (scripts/hash_to_jsonl.py), the exiftool JSONL of the MP4s (scripts/read_meta.py) and the SRT files, and needs no database; `write()` adds what is missing in ONE transaction, skips what is there, verifies, commits.
Times: `taken_at` is the camera's wall-clock time as the SRT records it for the first frame (the same kind of time as a drone still's DateTimeOriginal); the MP4's own QuickTime dates are UTC and stay unconverted in exif_raw. The telemetry (the SRT per frame: ISO, shutter, f-number, EV, focal length, latitude, longitude, relative and absolute altitude, colour temperature) is kept column-wise in `analysis.data`."""
import os, re
from datetime import datetime

from sqlalchemy import select

from .hashes import _chunks, keep_newest_current, root_labels
from .metaimport import _first, _num
from .models import Analysis, File, Photo, PhotoMeta

SRT_PRODUCER = 'srt-parse'
BLOCK = re.compile(r'(\d+)\s*\n(\d\d:\d\d:\d\d,\d{3}) --> (\d\d:\d\d:\d\d,\d{3})\s*\n(.*?)(?:\n\s*\n|\Z)', re.S)
TIME = re.compile(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?)')
PAIR = re.compile(r'(\w+): ([^\s\]]+)')


class VideoMismatch(Exception): pass


def parse_srt(text):
    """{frames, first, last, columns: {name: [value per frame]}} of a DJI SRT; a field a frame lacks is None. Numbers are floats; `shutter` ('1/100.0') is in seconds. None for text with no frame."""
    times, cols, n = [], {}, 0
    for m in BLOCK.finditer(text.replace('\r\n', '\n')):
        body = m.group(4); t = TIME.search(body)
        if not t: continue
        row = {}
        for br in re.findall(r'\[([^\]]*)\]', body):
            for k, v in PAIR.findall(br):
                if k == 'shutter':
                    try: a, _, b = v.partition('/'); row[k] = float(a) / float(b) if b else float(a)
                    except (ValueError, ZeroDivisionError): row[k] = None
                else:
                    try: row[k] = float(v)
                    except ValueError: row[k] = v
        for k in row: cols.setdefault(k, [None] * n)
        for k in cols: cols[k].append(row.get(k))
        times.append(t.group(1)); n += 1
    if not n: return None
    return dict(frames=n, first=times[0], last=times[-1], columns=dict(time=times, **cols))


def _time(s):
    for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S'):
        try: return datetime.strptime(s, fmt)
        except (TypeError, ValueError): pass
    return None


def plan(hashed, meta, roots, read_text=None):
    """{videos: [{key, stem, sha256, files: [...], meta: {...}, srt: {...} or None}], problems, counts}. `hashed`: records (path, sha256, size, mtime_ns) of .mp4/.lrf/.srt files; `meta`: records of read_meta.py for the MP4s ({sha256, tags}); `read_text(path)` gives an SRT's text (default: read the file)."""
    labels = root_labels(roots); absroot = dict(labels); problems = []; groups = {}; read_text = read_text or (lambda p: open(p, encoding='utf8', errors='replace').read())
    for r in hashed:
        stem, ext = os.path.splitext(os.path.basename(r['path'])); ext = ext.lower(); loc = next((n for n, root in labels if os.path.normpath(r['path']).startswith(root + os.sep)), None)
        if loc is None: problems.append(f'{r["path"]}: outside the roots'); continue
        rel = os.path.relpath(os.path.normpath(r['path']), absroot[loc]).replace(os.sep, '/')
        if ext in ('.mp4', '.lrf', '.srt'): groups.setdefault((loc, os.path.dirname(rel), stem), {}).setdefault(ext, []).append(dict(root=loc, relpath=rel, size=r['size'], mtime_ns=r['mtime_ns'], sha256=r['sha256'], path=r['path']))
    tags_of = {m['sha256']: m['tags'] for m in meta}; videos = []
    for key, g in sorted(groups.items()):
        mp4 = g.get('.mp4', [])
        if len(mp4) != 1: problems.append(f'{key[2]}: ' + ('no MP4' if not mp4 else f'{len(mp4)} MP4s with that name (ambiguous)') + ', skipped'); continue
        files = [dict(mp4[0], role='video', tag=None)]
        for ext in ('.lrf', '.srt'):
            if len(g.get(ext, [])) > 1: problems.append(f'{key[2]}: {len(g[ext])} {ext[1:].upper()} files with that name (ambiguous), none imported')
            elif g.get(ext): files.append(dict(g[ext][0], role='sidecar', tag=ext[1:]))
        srt = None
        if g.get('.srt') and len(g['.srt']) == 1:
            try: srt = parse_srt(read_text(g['.srt'][0]['path']))
            except OSError as e: problems.append(f'{key[2]}: SRT unreadable ({e})')
            if srt is None: problems.append(f'{key[2]}: no frames in the SRT')
        tags = tags_of.get(mp4[0]['sha256'])
        if tags is None: problems.append(f'{key[2]}: no exiftool record for the MP4')
        videos.append(dict(key=key, stem=key[2], sha256=mp4[0]['sha256'], files=files, tags=tags or {}, srt=srt, meta=columns(tags or {}, srt)))
    counts = dict(photo=len(videos), file=sum(len(v['files']) for v in videos), analysis=sum(1 for v in videos if v['srt']), problems=len(problems))
    return dict(videos=videos, problems=problems, counts=counts)


def columns(tags, srt):
    """The PhotoMeta columns of a video: the SRT's first frame (camera time, exposure, GPS) over the MP4's tags (size)."""
    c = srt['columns'] if srt else {}; first = lambda k: (c.get(k) or [None])[0]
    dur = _num(_first(tags, 'QuickTime:Duration'), float)
    return dict(taken_at=_time(srt['first']) if srt else None, width=_num(_first(tags, 'Track1:ImageWidth', 'Composite:ImageWidth'), int), height=_num(_first(tags, 'Track1:ImageHeight', 'Composite:ImageHeight'), int),
                make=_first(tags, 'QuickTime:Make', 'UserData:Make'), model=_first(tags, 'QuickTime:Model', 'UserData:Model'), lens=None, iso=_num(first('iso'), int), exposure_s=_num(first('shutter'), float), f_number=_num(first('fnum'), float), focal_mm=_num(first('focal_len'), float),
                orientation=None, gps_lat=_num(first('latitude'), float), gps_lon=_num(first('longitude'), float), gps_alt=_num(first('abs_alt'), float),
                dji=dict(duration_s=dur, frame_rate=_num(_first(tags, 'Track1:VideoFrameRate'), float), codec=_first(tags, 'Track1:CompressorID'), srt_frames=srt['frames'] if srt else None), exif_raw=dict(tags) or None, xmp_raw=None)


def write(session, p):
    """Returns the rows added. Rolls back and raises VideoMismatch on a FATAL problem or if a planned row is missing afterwards."""
    fatal = [x for x in p['problems'] if x.startswith('FATAL')]
    if fatal: raise VideoMismatch('refusing to write a plan with a FATAL problem: ' + '; '.join(fatal))
    shas = [v['sha256'] for v in p['videos']]; ids = {}
    for c in _chunks(shas): ids.update({s: i for s, i in session.execute(select(Photo.sha256, Photo.id).where(Photo.sha256.in_(c)))})
    new = [Photo(sha256=v['sha256'], stem=v['stem'], kind='video') for v in p['videos'] if v['sha256'] not in ids]; session.add_all(new); session.flush(); ids.update({x.sha256: x.id for x in new})
    by_path = {}
    for c in _chunks({f['relpath'] for v in p['videos'] for f in v['files']}):
        for f in session.scalars(select(File).where(File.relpath.in_(c))): by_path.setdefault((f.root, f.relpath), []).append(f)
    have = {(f.root, f.relpath, f.size, f.mtime_ns) for fs in by_path.values() for f in fs}; nfile = 0; file_ids = {}
    for v in p['videos']:
        for f in v['files']:
            k = (f['root'], f['relpath'], f['size'], f['mtime_ns'])
            if k not in have:
                row = File(root=f['root'], relpath=f['relpath'], size=f['size'], mtime_ns=f['mtime_ns'], photo_id=ids[v['sha256']], role=f['role'], tag=f['tag'], sha256=f['sha256'], current=True); session.add(row); have.add(k); by_path.setdefault((f['root'], f['relpath']), []).append(row); nfile += 1
    session.flush()
    for k, fs in by_path.items():
        if any(f.role in ('video', 'sidecar') for f in fs): keep_newest_current(fs)
    for fs in by_path.values():
        for f in fs: file_ids[(f.root, f.relpath, f.size, f.mtime_ns)] = f.id
    have_meta = {pid for (pid,) in session.execute(select(PhotoMeta.photo_id))}; nmeta = 0
    have_an = {(a.photo_id, a.input_key) for a in session.scalars(select(Analysis).where(Analysis.kind == 'dji_srt', Analysis.producer == SRT_PRODUCER))}; nan = 0
    for v in p['videos']:
        pid = ids[v['sha256']]
        if pid not in have_meta: session.add(PhotoMeta(photo_id=pid, **v['meta'])); have_meta.add(pid); nmeta += 1
        srtf = next((f for f in v['files'] if f['tag'] == 'srt'), None)
        if v['srt'] and srtf and (pid, srtf['sha256']) not in have_an:
            session.add(Analysis(photo_id=pid, kind='dji_srt', producer=SRT_PRODUCER, producer_version='1', input_key=srtf['sha256'], file_id=file_ids[(srtf['root'], srtf['relpath'], srtf['size'], srtf['mtime_ns'])], score=None, label=str(v['srt']['frames']), data=v['srt'])); have_an.add((pid, srtf['sha256'])); nan += 1
    session.flush(); bad = []
    got = {s for c in _chunks(shas) for (s,) in session.execute(select(Photo.sha256).where(Photo.sha256.in_(c)))}
    if set(shas) - got: bad.append('photo: a planned video is missing after the write')
    if bad: session.rollback(); raise VideoMismatch('; '.join(bad))
    session.commit(); return dict(photo=len(new), file=nfile, photo_meta=nmeta, analysis=nan)
