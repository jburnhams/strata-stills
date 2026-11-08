"""Projects into schema `photos`: `project.json` (-> project, project_source, project_rule) and the generated sidecars (`generated/<item>/<key>.json` -> render rows of kind 'generated'), tied to photos by STEM within the project's source folder. `plan()` reads the folders and the hashes and needs no database; `write()` puts the plan into the session's transaction (skips what is there), verifies and commits; MISSING photos make it refuse. The app's file reads are not changed.
Item ids ('<sha1 of the source path>~<stem>') were made on Windows from the lower-cased UNC path; the sha1 differs on other machines, so the id is not recomputed: the stem finds the picture (the DNG in the source folder is its identity) and the id is kept as `photo.legacy_item_id`. The JPEG beside a DNG is a file row of role 'jpeg' with its own sha256 for the same photo. Cache keys are stored exactly as in the sidecar. `Render.path` is relative to the project folder; the sidecar as written is kept whole in `info`. The content-hashed cache JSONs of the project's `cache/` folder have no link to a picture and are not imported (they are re-creatable)."""
import json, os, posixpath
from datetime import datetime, timezone

from sqlalchemy import select

from .hashes import _chunks, keep_newest_current, root_labels
from .models import File, Photo, Project, ProjectRule, ProjectSource, Render


class ProjectMismatch(Exception): pass


def _rel_of_unc(path):
    parts = [x for x in path.split('\\') if x]
    return '/'.join(parts[2:]) if path.startswith('\\\\') and len(parts) > 2 else None


def locate(source_path, roots):
    """(root label, relpath) of a stored source path on THIS machine: under one of the roots directly, or a Windows UNC path whose part after the share is a folder of one of the roots (tried in turn, the first that exists); None if neither."""
    labels = root_labels(roots); rel = _rel_of_unc(source_path)
    for name, root in labels:
        ap = os.path.normpath(source_path)
        if ap.startswith(root + os.sep): return name, os.path.relpath(ap, root).replace(os.sep, '/')
    if rel:
        for name, root in labels:
            if os.path.isdir(os.path.join(root, *rel.split('/'))): return name, rel
    return None


def _utc(s):
    try: d = datetime.fromisoformat(s)
    except (TypeError, ValueError): return None
    return d.astimezone(timezone.utc).replace(tzinfo=None) if d.tzinfo else d


def _norm(p): return p.replace('\\', '/').rstrip('/').lower()


def _basename(p): return p.replace('\\', '/').rsplit('/', 1)[-1]


def _render_file(d, item, key):
    """The portable path of a render, relative to the project folder: the file name the sidecar recorded (a TIFF is written as .tif, say), else the key + the extension of the format."""
    name = _basename(d.get('file') or '') or key + {'jpeg': '.jpg', 'jpg': '.jpg', 'tiff': '.tif', 'tif': '.tif', 'png': '.png'}.get(str(d.get('format')).lower(), '.jpg')
    return posixpath.join('generated', item, name)


def plan(project_dir, hashed, roots):
    """{project, sources, rules, files, renders, photos: {(root, folder, stem): {sha256, item}}, problems, counts}. `hashed` = records (path, sha256, size, mtime_ns) of the source folders' DNG and JPEG files. Everything is keyed by (source folder, stem): two sources sharing a stem never mix, and a stem that matches more than one file of its own source (X.DNG and X.dng) is a problem, not a guess."""
    problems = []; labels = root_labels(roots); absroot = dict(labels)
    try: doc = json.load(open(os.path.join(project_dir, 'project.json'), encoding='utf8'))
    except (OSError, ValueError) as e: return dict(project=None, sources=[], rules=[], files=[], renders=[], photos={}, problems=[f'FATAL unreadable project.json: {e}'], counts={})
    slug = os.path.basename(os.path.normpath(project_dir)); project = dict(slug=slug, name=doc.get('name') or slug, created=_utc(doc.get('created')))
    sources, stored = [], []
    for s in doc.get('sources', []):
        loc = locate(s['path'], roots)
        if loc is None: problems.append(f'FATAL source {s["path"]!r} is on none of the roots'); continue
        sources.append(dict(root=loc[0], relpath=loc[1], added=_utc(s.get('added')), original_path=s['path'])); stored.append(_norm(s['path']))
    rules = [dict(source=r.get('source', '*'), pipeline=r['pipeline'], variant=r.get('variant')) for r in doc.get('auto', [])]
    dng, jpg = {}, {}
    for r in hashed:
        stem, ext = os.path.splitext(os.path.basename(r['path'])); loc = next((n for n, root in labels if os.path.normpath(r['path']).startswith(root + os.sep)), None)
        if loc is None: problems.append(f'{r["path"]}: outside the roots'); continue
        rel = os.path.relpath(os.path.normpath(r['path']), absroot[loc]).replace(os.sep, '/'); key = (loc, posixpath.dirname(rel), stem)
        if ext.lower() == '.dng': dng.setdefault(key, []).append(r)
        elif ext.lower() in ('.jpg', '.jpeg'): jpg.setdefault(key, []).append(dict(root=loc, relpath=rel, size=r['size'], mtime_ns=r['mtime_ns'], sha256=r['sha256']))
    photos, renders, files = {}, [], []; gen = os.path.join(project_dir, 'generated')
    for item in sorted(os.listdir(gen)) if os.path.isdir(gen) else []:
        stem = item.split('~', 1)[-1]; sidecars = []
        for f in sorted(os.listdir(os.path.join(gen, item))):
            if not f.endswith('.json'): continue
            try: sidecars.append((f, json.load(open(os.path.join(gen, item, f), encoding='utf8'))))
            except (OSError, ValueError) as e: problems.append(f'FATAL unreadable sidecar {item}/{f}: {e}')
        src = next((n for n in (_norm(d.get('source') or '') for _, d in sidecars) if n), None); idx = None
        if src is not None: idx = next((i for i, st in enumerate(stored) if src.startswith(st + '/')), None)
        if idx is None and len(sources) == 1 and src is None: idx = 0                              # a sidecar with no source: only a one-source project can tell
        if idx is None: problems.append(f'{item}: cannot tell which source of the project it comes from'); continue
        key = (sources[idx]['root'], sources[idx]['relpath'], stem); cands = dng.get(key, [])
        if len(cands) != 1: problems.append(f'{item}: ' + (f'no DNG named {stem} among the hashed files of its source' if not cands else f'{len(cands)} DNGs named {stem} in its source (ambiguous)')); continue
        photos[key] = dict(sha256=cands[0]['sha256'], item=item)
        if len(jpg.get(key, [])) > 1: problems.append(f'{item}: {len(jpg[key])} JPEGs named {stem} in its source (ambiguous), none imported')
        elif key in jpg: files.append(dict(jpg[key][0], photo=key))
        for f, d in sidecars:
            out = d.get('output') or [None, None]
            renders.append(dict(photo=key, cache_key=d['key'], kind='generated', variant=d.get('source_variant'), preset=d.get('preset'), stages=d.get('stages'), summary=d.get('summary'), out_w=out[0], out_h=out[1],
                                bytes=d.get('bytes'), quality=d.get('quality'), path=_render_file(d, item, f[:-5]), info=d, created=_utc(d.get('created'))))
    counts = dict(project=1, project_source=len(sources), project_rule=len(rules), photos_linked=len(photos), file_jpeg=len(files), render=len(renders), dng_hashed=sum(len(v) for v in dng.values()), dng_without_item=sum(1 for k in dng if k not in photos), problems=len(problems))
    return dict(project=project, sources=sources, rules=rules, files=files, renders=renders, photos=photos, problems=problems, counts=counts)


def write(session, p):
    """Returns the rows added per table. Rolls back and raises ProjectMismatch if a photo is missing or a planned row is missing afterwards."""
    fatal = [x for x in p['problems'] if x.startswith('FATAL')]
    if fatal or p['project'] is None: raise ProjectMismatch('refusing to write a plan with a FATAL problem: ' + '; '.join(fatal))
    shas = {v['sha256'] for v in p['photos'].values()}; ids = {}
    for c in _chunks(shas): ids.update({s: i for s, i in session.execute(select(Photo.sha256, Photo.id).where(Photo.sha256.in_(c)))})
    if shas - set(ids): session.rollback(); raise ProjectMismatch(f'{len(shas - set(ids))} planned photo(s) are not in the database; load the hashes first')
    pid_of = {key: ids[v['sha256']] for key, v in p['photos'].items()}; added = {}
    proj = session.scalars(select(Project).where(Project.slug == p['project']['slug'])).first()
    if proj is None: proj = Project(slug=p['project']['slug'], name=p['project']['name'], created=p['project']['created']); session.add(proj); session.flush(); added['project'] = 1
    else: added['project'] = 0
    have = {(s.root, s.relpath) for s in session.scalars(select(ProjectSource).where(ProjectSource.project_id == proj.id))}; n = 0
    for s in p['sources']:
        if (s['root'], s['relpath']) not in have: session.add(ProjectSource(project_id=proj.id, **s)); n += 1
    added['project_source'] = n; have_r = {(r.source, r.pipeline, r.variant) for r in session.scalars(select(ProjectRule).where(ProjectRule.project_id == proj.id))}; n = 0
    for r in p['rules']:
        if (r['source'], r['pipeline'], r['variant']) not in have_r: session.add(ProjectRule(project_id=proj.id, **r)); n += 1
    added['project_rule'] = n; conflicts = []
    for key, v in p['photos'].items():
        ph = session.get(Photo, pid_of[key])
        if ph.legacy_item_id is None: ph.legacy_item_id = v['item']
        elif ph.legacy_item_id != v['item']: conflicts.append(f'{key[2]}: legacy id {ph.legacy_item_id} differs from {v["item"]}')
    if conflicts: session.rollback(); raise ProjectMismatch('; '.join(conflicts[:5]))
    by_path = {}
    for c in _chunks({f['relpath'] for f in p['files']}):
        for f in session.scalars(select(File).where(File.relpath.in_(c))): by_path.setdefault((f.root, f.relpath), []).append(f)
    have_f = {(f.root, f.relpath, f.size, f.mtime_ns) for fs in by_path.values() for f in fs}; n = 0
    for f in p['files']:
        k = (f['root'], f['relpath'], f['size'], f['mtime_ns'])
        if k in have_f: continue
        row = File(root=f['root'], relpath=f['relpath'], size=f['size'], mtime_ns=f['mtime_ns'], photo_id=pid_of[f['photo']], role='jpeg', sha256=f['sha256'], current=True); session.add(row); have_f.add(k); by_path.setdefault((f['root'], f['relpath']), []).append(row); n += 1
    session.flush()
    for k in {(f['root'], f['relpath']) for f in p['files']}: keep_newest_current(by_path[k])                                  # one current version per path across old and new rows
    session.flush()
    added['file_jpeg'] = n; have_k = {(r.kind, r.cache_key) for c in _chunks({r['cache_key'] for r in p['renders']}) for r in session.scalars(select(Render).where(Render.cache_key.in_(c)))}; n = 0
    for r in p['renders']:
        if (r['kind'], r['cache_key']) in have_k: continue
        row = Render(photo_id=pid_of[r['photo']], cache_key=r['cache_key'], kind=r['kind'], variant=r['variant'], preset=r['preset'], stages=r['stages'], summary=r['summary'], out_w=r['out_w'], out_h=r['out_h'], bytes=r['bytes'], quality=r['quality'], path=r['path'], info=r['info'])
        if r['created']: row.created_at = r['created']
        session.add(row); have_k.add((r['kind'], r['cache_key'])); n += 1
    added['render'] = n; session.flush(); bad = []
    got = {(r.kind, r.cache_key) for c in _chunks({r['cache_key'] for r in p['renders']}) for r in session.scalars(select(Render).where(Render.cache_key.in_(c)))}
    if not {(r['kind'], r['cache_key']) for r in p['renders']} <= got: bad.append('render: a planned row is missing after the write')
    if len(session.scalars(select(ProjectSource).where(ProjectSource.project_id == proj.id)).all()) < len(p['sources']): bad.append('project_source: a planned row is missing')
    if bad: session.rollback(); raise ProjectMismatch('; '.join(bad))
    session.commit(); return added
