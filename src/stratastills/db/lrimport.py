"""Lightroom-side data into schema `photos`, one importer per source, each with `plan_*()` (reads files, no database) and `write_*()` (one transaction, skips what is there, verifies, commits):
- `lightroom`: the catalogue rows (a raw and the export made from it, plus the crs: settings of the export) -> `file` rows of role 'final' + `lightroom_edit`;
- `nsfw`: ~/.stratastills/nsfw_cache.json (keys `model|path|size|mtime_ns` of the EXPORTS) -> `file` rows of role 'final' (the version that was scanned) + `nsfw_result`;
- `faces`: candidates/personal/faces.json (keys are indexes into lr_dataset.load_catalogue(): the rows with no 'error' and with 'Exposure2012') -> one `analysis` row (kind 'faces') + `detection` rows per picture;
- `embeddings`: candidates/personal/clip.npz ('pca' 32 numbers per row, `idx` indexes the same filtered list) -> `embedding` rows of model 'clip-pca32'.
A picture is found through the catalogue's raw -> its hash (the hashing job's JSONL) -> `photo`. A row whose raw was not hashed, or whose photo row does not exist yet, is a problem and is skipped (write() refuses a plan with a FATAL one). Roots: the export's source root is labelled by its folder name ('personal')."""
import json, os
from datetime import datetime, timezone

from sqlalchemy import select

from .hashes import _chunks, root_labels
from .models import Analysis, Detection, Embedding, File, LightroomEdit, NsfwResult, Photo

FACES_PRODUCER = 'scrfd-buffalo_l'
CLIP_MODEL = 'clip-pca32'


class ImportMismatch(Exception): pass


def filtered_rows(catalogue):
    """The rows lr_dataset.load_catalogue() returns, in its order: no 'error' and with 'Exposure2012' (faces.json and clip.npz index this list)."""
    return [r for r in catalogue if 'error' not in r and 'Exposure2012' in r]


def _where(path, labels):
    ap = os.path.normpath(path)
    for n, root in labels:
        if ap.startswith(root + os.sep): return n, os.path.relpath(ap, root).replace(os.sep, '/')
    return None


def _stat(path):
    try: st = os.stat(path); return st.st_size, st.st_mtime_ns
    except OSError: return None


def plan_lightroom(catalogue, hashed, roots, stat=_stat):
    """{edits: [{sha256, file: (root, relpath, size, mtime_ns), settings, crop, error}], problems, counts}. `hashed` = {raw path: sha256}; `stat` gives (size, mtime_ns) of an export (None = unreadable)."""
    labels = root_labels(roots); edits, problems = [], []
    for r in catalogue:
        raw, fin = r.get('raw'), r.get('final'); sha = hashed.get(raw); loc = _where(fin or '', labels); st = stat(fin) if loc else None
        if sha is None: problems.append(f'{raw}: raw not hashed'); continue
        if loc is None or st is None: problems.append(f'{fin}: export outside the roots or unreadable'); continue
        crop = {k: v for k, v in r.items() if k.startswith('Crop') or k == 'HasCrop'}; settings = {k: v for k, v in r.items() if k not in crop and k not in ('raw', 'final', 'error')}
        edits.append(dict(sha256=sha, file=(loc[0], loc[1], st[0], st[1]), settings=settings, crop=crop, error=r.get('error')))
    return dict(edits=edits, problems=problems, counts=dict(lightroom_edit=len(edits), file=len({e['file'] for e in edits}), problems=len(problems)))


def plan_nsfw(cache, catalogue, hashed, roots):
    """{results: [{sha256, file, model, score, label}], problems, counts}; the photo is found through the catalogue's export -> raw."""
    labels = root_labels(roots); photo_of = {os.path.normpath(r['final']): hashed.get(r['raw']) for r in catalogue if r.get('final') and r.get('raw')}; results, problems = [], []
    for key, val in cache.items():
        try: model, path, size, mtime = key.split('|'); size, mtime = int(size), int(mtime)
        except ValueError: problems.append(f'{key[:80]}: unparsable key'); continue
        sha = photo_of.get(os.path.normpath(path)); loc = _where(path, labels)
        if sha is None: problems.append(f'{path}: no hashed raw behind this export'); continue
        if loc is None: problems.append(f'{path}: outside the roots'); continue
        results.append(dict(sha256=sha, file=(loc[0], loc[1], size, mtime), model=model, score=float(val[0]), label=str(val[1] or '')))
    return dict(results=results, problems=problems, counts=dict(nsfw_result=len(results), file=len({r['file'] for r in results}), problems=len(problems)))


def plan_faces(catalogue, faces, hashed):
    """{pictures: [{sha256, faces: [...]}], problems, counts}; faces.json keys are indexes into filtered_rows()."""
    rows = filtered_rows(catalogue); pics, problems, n, seen, dup = [], [], 0, set(), 0                       # two raws with one hash are one photo: the first row in catalogue order wins, later ones are skipped and counted
    if len(faces) != len(rows): problems.append(f'FATAL faces.json has {len(faces)} entries but the filtered catalogue {len(rows)}: the order cannot be trusted')
    else:
        for k, fl in faces.items():
            try: row = rows[int(k)]
            except (ValueError, IndexError): problems.append(f'FATAL face key {k!r} is not an index of the catalogue'); continue
            sha = hashed.get(row.get('raw'))
            if sha is None: problems.append(f'{row.get("raw")}: raw not hashed'); continue
            if sha in seen: dup += 1; problems.append(f'{row.get("raw")}: duplicate photo, skipped'); continue
            seen.add(sha); pics.append(dict(sha256=sha, faces=fl)); n += len(fl)
    return dict(pictures=pics, problems=problems, counts=dict(analysis=len(pics), detection=n, duplicate_photo_skipped=dup, problems=len(problems)))


def plan_embeddings(catalogue, clip, hashed, model=CLIP_MODEL):
    """{vectors: [{sha256, vector}], problems, counts}; clip = {'idx': [...], 'pca': [[...], ...]} (numpy arrays or lists)."""
    rows = filtered_rows(catalogue); vecs, problems, seen, dup = [], [], set(), 0
    for i, ix in enumerate(clip['idx']):
        row = rows[int(ix)] if 0 <= int(ix) < len(rows) else None; sha = hashed.get(row.get('raw')) if row else None
        if sha is None: problems.append(f'embedding {i}: no hashed raw'); continue
        if sha in seen: dup += 1; problems.append(f'embedding {i}: duplicate photo, skipped'); continue
        seen.add(sha); vecs.append(dict(sha256=sha, vector=[float(x) for x in clip['pca'][i]]))
    return dict(vectors=vecs, model=model, problems=problems, counts=dict(embedding=len(vecs), duplicate_photo_skipped=dup, problems=len(problems)))


def _check(p):
    fatal = [x for x in p['problems'] if x.startswith('FATAL')]
    if fatal: raise ImportMismatch('refusing to write a plan with a FATAL problem: ' + '; '.join(fatal))


def _photo_ids(session, shas):
    ids = {}
    for c in _chunks(set(shas)): ids.update({s: i for s, i in session.execute(select(Photo.sha256, Photo.id).where(Photo.sha256.in_(c)))})
    missing = sorted(set(shas) - set(ids))
    if missing: session.rollback(); raise ImportMismatch(f'{len(missing)} planned hash(es) have no photo row, e.g. {missing[:2]}; load the hashes first')
    return ids


def _files(session, keys, photo_of):
    """{(root, relpath, size, mtime_ns): id}: existing rows, the missing ones created with role 'final' (photo from `photo_of[key]`). Per (root, relpath) exactly one version stays current: the newest (mtime_ns, then id) across the rows already there AND the new ones; the others are history and keep their own results. Returns (ids, number created)."""
    by_path = {}
    for c in _chunks({k[1] for k in keys}):
        for f in session.scalars(select(File).where(File.relpath.in_(c))): by_path.setdefault((f.root, f.relpath), []).append(f)
    have = {(f.root, f.relpath, f.size, f.mtime_ns): f for fs in by_path.values() for f in fs}; now = datetime.now(timezone.utc).replace(tzinfo=None)
    new = [File(root=k[0], relpath=k[1], size=k[2], mtime_ns=k[3], photo_id=photo_of[k], role='final', current=True, last_seen=now) for k in keys if k not in have]
    session.add_all(new); session.flush()
    for f in new: by_path.setdefault((f.root, f.relpath), []).append(f); have[(f.root, f.relpath, f.size, f.mtime_ns)] = f
    for k in {(k[0], k[1]) for k in keys}:
        rows = by_path[k]; keep = max(rows, key=lambda x: (x.mtime_ns, x.id))
        for f in rows: f.current = f is keep
    session.flush(); return {k: f.id for k, f in have.items()}, len(new)


def write_lightroom(session, p):
    _check(p); ids = _photo_ids(session, [e['sha256'] for e in p['edits']]); keys = {e['file']: ids[e['sha256']] for e in p['edits']}; fids, nfile = _files(session, keys, keys)
    have = {(e.photo_id, e.final_file_id) for e in session.scalars(select(LightroomEdit))}; new = []
    for e in p['edits']:
        k = (ids[e['sha256']], fids[e['file']])
        if k not in have: have.add(k); new.append(LightroomEdit(photo_id=k[0], final_file_id=k[1], settings=e['settings'], crop=e['crop'], error=e['error']))
    session.add_all(new); session.flush(); got = {(e.photo_id, e.final_file_id) for e in session.scalars(select(LightroomEdit))}
    if not {(ids[e['sha256']], fids[e['file']]) for e in p['edits']} <= got: session.rollback(); raise ImportMismatch('lightroom_edit: a planned row is missing after the write')
    session.commit(); return dict(lightroom_edit=len(new), file=nfile)


def write_nsfw(session, p):
    _check(p); ids = _photo_ids(session, [r['sha256'] for r in p['results']]); keys = {r['file']: ids[r['sha256']] for r in p['results']}; fids, nfile = _files(session, keys, keys)
    have = {(r.file_id, r.model) for r in session.scalars(select(NsfwResult))}; new = []
    for r in p['results']:
        k = (fids[r['file']], r['model'])
        if k not in have: have.add(k); new.append(NsfwResult(file_id=k[0], model=k[1], score=r['score'], label=r['label']))
    session.add_all(new); session.flush(); got = {(r.file_id, r.model) for r in session.scalars(select(NsfwResult))}
    if not {(fids[r['file']], r['model']) for r in p['results']} <= got: session.rollback(); raise ImportMismatch('nsfw_result: a planned row is missing after the write')
    session.commit(); return dict(nsfw_result=len(new), file=nfile)


def write_faces(session, p):
    _check(p); ids = _photo_ids(session, [x['sha256'] for x in p['pictures']]); have = {a.photo_id for a in session.scalars(select(Analysis).where(Analysis.kind == 'faces', Analysis.producer == FACES_PRODUCER))}; nd = 0; made = []
    for x in p['pictures']:
        if ids[x['sha256']] in have: continue
        have.add(ids[x['sha256']])
        a = Analysis(photo_id=ids[x['sha256']], kind='faces', producer=FACES_PRODUCER, producer_version='', input_key='', score=max(f['score'] for f in x['faces']) if x['faces'] else None, label=str(len(x['faces'])), data=dict(count=len(x['faces']))); session.add(a); made.append((a, x))
    session.flush()
    for a, x in made:
        for f in x['faces']: b = f['box']; session.add(Detection(analysis_id=a.id, label='face', x=b[0], y=b[1], w=b[2], h=b[3], score=f.get('score'), data=dict(cam=f.get('cam'), fin=f.get('fin')))); nd += 1
    session.flush(); got = {a.photo_id for a in session.scalars(select(Analysis).where(Analysis.kind == 'faces', Analysis.producer == FACES_PRODUCER))}
    if not {ids[x['sha256']] for x in p['pictures']} <= got: session.rollback(); raise ImportMismatch('analysis: a planned row is missing after the write')
    session.commit(); return dict(analysis=len(made), detection=nd)


def write_embeddings(session, p):
    _check(p); ids = _photo_ids(session, [v['sha256'] for v in p['vectors']]); model = p['model']
    have = {e.photo_id for e in session.scalars(select(Embedding).where(Embedding.model == model, Embedding.detection_id.is_(None)))}
    new = []
    for v in p['vectors']:
        if ids[v['sha256']] not in have: have.add(ids[v['sha256']]); new.append(Embedding(photo_id=ids[v['sha256']], model=model, vector=v['vector']))
    session.add_all(new); session.flush()
    got = {e.photo_id for e in session.scalars(select(Embedding).where(Embedding.model == model, Embedding.detection_id.is_(None)))}
    if not {ids[v['sha256']] for v in p['vectors']} <= got: session.rollback(); raise ImportMismatch('embedding: a planned row is missing after the write')
    session.commit(); return dict(embedding=len(new))
