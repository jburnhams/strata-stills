"""The derived feature matrices of the Lightroom look work (candidates/personal/*.npz) into `analysis` rows, one kind per matrix and ONE row per photo, the values with their column names so a row reads without the script that made it:
- `clip_labels`  (clip.npz `labels`, 28 scene-prompt scores)            producer 'open_clip ViT-B-32 laion2b_s34b_b79k'
- `exif_features` (meta.npz `M`, 91 numbers made from the exiftool tags)   producer 'lr_look.meta'     (90,000+ NaN where a camera lacks a tag, so JSON nulls, not an embedding: pgvector cannot hold NaN)
- `look_features` (features_all.npz `X`, 62 picture and exposure statistics) producer 'lr_dataset.features'
- `final_stats`  (features_all.npz `F` + `C`, the lightness percentiles and colour of the person's FINAL, 16 numbers) producer 'lr_dataset.final_stats'
- `lr_fit`       (fit.npz: 19 fitted look parameters, the 18 decoded ones, the distance to the final before and after) producer 'lr_fit'
`input_key` = the first 16 hex digits of the .npz file's SHA-256, so a regenerated file is a new row. NaN -> null.
ROW ORDER: every matrix is indexed by `idx` into lr_dataset.load_catalogue() (the catalogue rows with no 'error' and with 'Exposure2012'). It is PROVEN, not assumed: `prove_order()` checks the matrix columns that mirror exiftool tags (ISO, FNumber and the year of meta.npz; fnum, log2 ISO and year of features_all.npz) against the exiftool tags read from the very raw of row `idx[i]`, for every row; clip.npz and fit.npz carry no such columns and are accepted only when their `idx` equals the proven one. A failed proof is a FATAL problem and nothing is written. Not imported (and why): features/b*.npz (the batches features_all.npz was concatenated from), final_stats.json / sample.json / pairs.json (subsets or copies of the catalogue and F/C), tmodel_eval.npz (predictions for a test split, regenerable), candidates/look_pairs_*.npz (per-camera training data, not per photo)."""
import hashlib, json, math, os

from sqlalchemy import select

from .hashes import _chunks
from .lrimport import filtered_rows
from .models import Analysis, Photo

PCTS = (1, 5, 10, 25, 50, 75, 90, 95, 99)
FINAL_NAMES = [f'L_p{p}' for p in PCTS] + ['L_mean'] + ['a_mean', 'b_mean', 'C_p50', 'C_p90', 'C_mean', 'C_p99']
KINDS = {'clip_labels': ('open_clip ViT-B-32 laion2b_s34b_b79k', 'clip.npz'), 'exif_features': ('lr_look.meta', 'meta.npz'), 'look_features': ('lr_dataset.features', 'features_all.npz'),
         'final_stats': ('lr_dataset.final_stats', 'features_all.npz'), 'lr_fit': ('lr_fit', 'fit.npz')}


class FeatureMismatch(Exception): pass


def _clean(v):
    f = float(v); return None if math.isnan(f) or math.isinf(f) else f


def file_key(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(1 << 22): h.update(b)
    return h.hexdigest()[:16]


def _num(v):
    try: return float(v)
    except (TypeError, ValueError): return float('nan')


def prove_order(rows, idx, matrices, tags_of):
    """[] when every mirrored column equals the exiftool tag of its row's raw, else the reasons. `matrices`: {'meta': (names, M), 'features': (names, X)}; tags_of: {raw path: {tag: value}}."""
    bad = []; need = {'meta': {'ExifIFD:ISO': ('iso', 1.0), 'ExifIFD:FNumber': ('fnum', 1.0)}, 'features': {'fnum': ('fnum', 1.0), 'log_iso': ('log2iso', 1.0), 'year': ('year', 1.0)}}
    def truth(raw, what):
        t = tags_of.get(raw)
        if t is None: return float('nan')
        if what == 'iso': return _num(t.get('ExifIFD:ISO'))
        if what == 'fnum': return _num(t.get('ExifIFD:FNumber'))
        if what == 'log2iso': v = _num(t.get('ExifIFD:ISO')); return math.log2(v) if v > 0 else float('nan')
        return _num(str(t.get('ExifIFD:DateTimeOriginal'))[:4])
    for name, (names, M) in matrices.items():
        for col, (what, _) in need[name].items():
            if col not in names: bad.append(f'{name}: column {col} missing, cannot prove the row order'); continue
            j = names.index(col); n = k = 0
            for i, ix in enumerate(idx):
                a = float(M[i][j]); b = truth(rows[int(ix)]['raw'], what)
                if math.isnan(a) or math.isnan(b): continue
                n += 1; k += abs(a - b) > 1e-3
            if n == 0: bad.append(f'{name}.{col}: nothing to compare')
            elif k: bad.append(f'{name}.{col}: {k} of {n} rows differ from the exiftool tags: the row order is not the catalogue\'s')
    return bad


def plan(catalogue, hashed, tags_of, arrays, keys):
    """{rows: {kind: [{sha256, data}]}, problems, counts}. `arrays` = {'clip': {...}, 'meta': {...}, 'features': {...}, 'fit': {...}} of numpy-like npz contents, `keys` = {file name: input_key}. Reads only."""
    rows = filtered_rows(catalogue); problems = []; out = {k: [] for k in KINDS}; idx = arrays['features']['idx']
    bad = prove_order(rows, idx, {'meta': ([str(n) for n in arrays['meta']['names']], arrays['meta']['M']), 'features': ([str(n) for n in arrays['features']['names']], arrays['features']['X'])}, tags_of)
    problems += ['FATAL ' + b for b in bad]
    for name in ('clip', 'fit'):
        if [int(x) for x in arrays[name]['idx']] != [int(x) for x in idx]: problems.append(f'FATAL {name}.npz: its idx differs from the proven one, so its row order is not proven')
    if any(p.startswith('FATAL') for p in problems): return dict(rows=out, problems=problems, counts={k: 0 for k in KINDS})
    seen = set(); dup = 0
    for i, ix in enumerate(idx):
        raw = rows[int(ix)].get('raw'); sha = hashed.get(raw)
        if sha is None: problems.append(f'{raw}: raw not hashed'); continue
        if sha in seen: dup += 1; problems.append(f'{raw}: duplicate photo, skipped'); continue
        seen.add(sha); a = arrays
        out['clip_labels'].append(dict(sha256=sha, data=dict(names=[str(n) for n in a['clip']['label_names']], values=[_clean(v) for v in a['clip']['labels'][i]])))
        out['exif_features'].append(dict(sha256=sha, data=dict(names=[str(n) for n in a['meta']['names']], values=[_clean(v) for v in a['meta']['M'][i]])))
        out['look_features'].append(dict(sha256=sha, data=dict(names=[str(n) for n in a['features']['names']], values=[_clean(v) for v in a['features']['X'][i]])))
        out['final_stats'].append(dict(sha256=sha, data=dict(names=FINAL_NAMES, values=[_clean(v) for v in list(a['features']['F'][i]) + list(a['features']['C'][i])])))
        f = a['fit']; out['lr_fit'].append(dict(sha256=sha, data=dict(names=[str(n) for n in f['names']], params=[_clean(v) for v in f['params'][i]], decoded=[_clean(v) for v in f['decoded'][i]], before=[_clean(v) for v in f['before'][i]], after=[_clean(v) for v in f['after'][i]])))
    return dict(rows=out, keys=keys, problems=problems, counts=dict({k: len(v) for k, v in out.items()}, duplicate_photo_skipped=dup, problems=len(problems)))


def write(session, p):
    """Adds the missing rows of each kind (unique key photo, kind, producer, version, input_key) in ONE transaction; returns {kind: added}."""
    fatal = [x for x in p['problems'] if x.startswith('FATAL')]
    if fatal: raise FeatureMismatch('refusing to write a plan with a FATAL problem: ' + '; '.join(fatal))
    shas = {r['sha256'] for v in p['rows'].values() for r in v}; ids = {}
    for c in _chunks(shas): ids.update({s: i for s, i in session.execute(select(Photo.sha256, Photo.id).where(Photo.sha256.in_(c)))})
    if shas - set(ids): session.rollback(); raise FeatureMismatch(f'{len(shas - set(ids))} planned photo(s) are not in the database; load the hashes first')
    added = {}
    for kind, rows in p['rows'].items():
        producer, fname = KINDS[kind]; key = p['keys'][fname]; have = {pid for (pid,) in session.execute(select(Analysis.photo_id).where(Analysis.kind == kind, Analysis.producer == producer, Analysis.producer_version == '1', Analysis.input_key == key))}; n = 0
        for r in rows:
            pid = ids[r['sha256']]
            if pid in have: continue
            have.add(pid); session.add(Analysis(photo_id=pid, kind=kind, producer=producer, producer_version='1', input_key=key, data=r['data'])); n += 1
        added[kind] = n
    session.flush(); bad = []
    for kind, rows in p['rows'].items():
        producer, fname = KINDS[kind]; got = {pid for (pid,) in session.execute(select(Analysis.photo_id).where(Analysis.kind == kind, Analysis.producer == producer, Analysis.input_key == p['keys'][fname]))}
        if not {ids[r['sha256']] for r in rows} <= got: bad.append(f'{kind}: a planned row is missing after the write')
    if bad: session.rollback(); raise FeatureMismatch('; '.join(bad))
    session.commit(); return added


def load_arrays(folder):
    """The npz contents and their input keys from candidates/personal (numpy needed)."""
    import numpy as np
    out, keys = {}, {}
    for name, fname in (('clip', 'clip.npz'), ('meta', 'meta.npz'), ('features', 'features_all.npz'), ('fit', 'fit.npz')):
        z = np.load(os.path.join(folder, fname), allow_pickle=True); out[name] = {k: z[k] for k in z.files}; keys[fname] = file_key(os.path.join(folder, fname))
    if 'idx' not in out['meta']: out['meta']['idx'] = out['features']['idx']
    return out, keys
