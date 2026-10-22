"""Projects. A project is a folder inside the projects directory (settings.projects_dir) holding `project.json`, the analysis and the output. It is separate from the photos:
a project has any number of SOURCE FOLDERS (a shoot on a drone card, a phone dump, a folder on the NAS ...), added and removed at any time. Source folders are only ever read.

  <projects_dir>/<id>/project.json   {schema, id, name, created, sources: [{path, added}]}
  <projects_dir>/<id>/...            everything the pipeline makes for the project

Projects may be grouped in plain sub-folders of the projects directory (`<projects_dir>/2025/hazelmere/`): `list_projects` looks `MAX_DEPTH` levels down. A project's id is its path relative to the projects directory, with forward slashes."""
import datetime as dt, json, os, re

from stratastills import ingest, settings

SCHEMA = 1
FILE = 'project.json'
MAX_DEPTH = 3


class ProjectError(ValueError): pass


def slugify(name):
    s = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
    if not s: raise ProjectError('a project needs a name')
    return s


def folder(pid, root=None):
    """The project's folder; refuses an id that would leave the projects directory."""
    root = root or settings.projects_dir(); p = os.path.abspath(os.path.join(root, *pid.split('/')))
    if os.path.commonpath([root, p]) != root or p == root: raise ProjectError(f'bad project id {pid!r}')
    return p


def load(pid, root=None):
    f = os.path.join(folder(pid, root), FILE)
    try: d = json.load(open(f))
    except (OSError, ValueError): raise ProjectError(f'no project {pid!r}')
    d.setdefault('sources', []); d['id'] = pid; return d


def _save(doc, root=None):
    d = folder(doc['id'], root); tmp = os.path.join(d, FILE + '.tmp')
    with open(tmp, 'w') as f: json.dump({k: v for k, v in doc.items() if k != 'id'}, f, indent=1)
    os.replace(tmp, os.path.join(d, FILE)); return doc


def create(name, sources=(), group=None, root=None):
    """A new project called `name` (its folder is the name slugified, inside `group` if given). Raises ProjectError if that folder already holds a project."""
    pid = '/'.join([slugify(g) for g in (group or '').split('/') if g] + [slugify(name)]); d = folder(pid, root)
    if os.path.exists(os.path.join(d, FILE)): raise ProjectError(f'project {pid!r} already exists')
    os.makedirs(d, exist_ok=True)
    doc = dict(schema=SCHEMA, id=pid, name=name.strip(), created=dt.datetime.now().astimezone().isoformat(timespec='seconds'), sources=[])
    _save(doc, root)
    for s in sources: doc = add_source(pid, s, root)
    return doc


def list_projects(root=None):
    """Every project under the projects directory, newest first: [{id, name, created, sources, n_sources}]. A folder that has a project.json is a project and is not searched further."""
    root = root or settings.projects_dir(); out = []

    def walk(d, depth):
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            if not os.path.isdir(p): continue
            if os.path.exists(os.path.join(p, FILE)):
                pid = os.path.relpath(p, root).replace(os.sep, '/')
                try: doc = load(pid, root)
                except ProjectError: continue
                out.append(dict(id=pid, name=doc.get('name', n), created=doc.get('created'), sources=doc['sources'], n_sources=len(doc['sources'])))
            elif depth < MAX_DEPTH: walk(p, depth + 1)
    walk(root, 1)
    return sorted(out, key=lambda p: p['created'] or '', reverse=True)


def add_source(pid, path, root=None):
    """Add a source folder (it must exist and not already be one of the project's). Paths are stored absolute; UNC paths (\\\\Server\\share) are kept as they are."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path): raise ProjectError(f'not a folder: {path}')
    doc = load(pid, root)
    if any(os.path.normcase(s['path']) == os.path.normcase(path) for s in doc['sources']): raise ProjectError('that folder is already a source of this project')
    doc['sources'].append(dict(path=path, added=dt.datetime.now().astimezone().isoformat(timespec='seconds'))); return _save(doc, root)


def remove_source(pid, path, root=None):
    doc = load(pid, root); n = len(doc['sources'])
    doc['sources'] = [s for s in doc['sources'] if os.path.normcase(s['path']) != os.path.normcase(os.path.abspath(path))]
    if len(doc['sources']) == n: raise ProjectError('not a source of this project')
    return _save(doc, root)


def shots(pid, root=None, recursive=False):
    """The shots of all the project's sources: [(source_path, Shot)] in source order, then by name. A source that is not reachable (a share that is off) is reported as a (path, None) entry rather than failing the rest."""
    out = []
    for s in load(pid, root)['sources']:
        try: out += [(s['path'], sh) for sh in ingest.scan(s['path'], recursive=recursive)]
        except OSError: out.append((s['path'], None))
    return out


# ---- sources, items and the choices made on them ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

STATE = 'items.json'                    # {item_id: {primary: variant_id}}: what the person chose, per item


def source_id(path):
    """A short stable id of a source folder (the same folder is the same id whatever the case or slashes)."""
    import hashlib
    return hashlib.sha1(os.path.normcase(os.path.abspath(path)).encode()).hexdigest()[:8]


def item_id(sid, stem): return f'{sid}~{stem}'


def _source(pid, sid, root=None):
    for s in load(pid, root)['sources']:
        if source_id(s['path']) == sid: return s
    raise ProjectError(f'no source {sid!r} in project {pid!r}')


def load_state(pid, root=None):
    try: d = json.load(open(os.path.join(folder(pid, root), STATE)))
    except (OSError, ValueError): d = {}
    return d if isinstance(d, dict) else {}


def _save_state(pid, d, root=None):
    p = os.path.join(folder(pid, root), STATE); tmp = p + '.tmp'
    with open(tmp, 'w') as f: json.dump(d, f, indent=1)
    os.replace(tmp, p)


def default_primary(shot):
    """The variant shown and processed unless the person picks another: the raw (a panorama's raw is a big linear file, so its JPEG), else the camera JPEG, else the first derived file."""
    v = {x['id']: x for x in shot.variants()}
    for k in (('jpeg', 'raw') if shot.pano else ('raw', 'jpeg')):
        if k in v: return k
    return next(iter(v), None)


def source_summaries(pid, root=None):
    """Each source folder with what is in it: [{id, path, reachable, stills, panos, videos}]. An unreachable folder (a share that is off) is reported, not an error."""
    out = []
    for s in load(pid, root)['sources']:
        d = dict(id=source_id(s['path']), path=s['path'], added=s.get('added'), reachable=True, stills=0, panos=0, videos=0)
        try:
            for sh in ingest.scan(s['path']):
                k = sh.kind; d['stills' if k == 'still' else 'panos' if k == 'pano' else 'videos'] += 1
        except OSError: d['reachable'] = False
        out.append(d)
    return out


def source_items(pid, sid, root=None):
    """The items of one source: [{id, stem, kind, parent, primary, variants: [...]}] (videos included, marked by kind)."""
    s = _source(pid, sid, root); st = load_state(pid, root); out = []
    for sh in ingest.scan(s['path']):
        iid = item_id(sid, sh.stem); vs = sh.variants(); prim = st.get(iid, {}).get('primary')
        if prim not in {v['id'] for v in vs}: prim = default_primary(sh)
        out.append(dict(id=iid, stem=sh.stem, kind=sh.kind, parent=sh.parent, primary=prim, chosen=iid in st, variants=vs, n_sidecars=len(sh.sidecars), video=sh.video))
    return out


def get_item(pid, iid, root=None):
    sid, _, stem = iid.partition('~')
    for it in source_items(pid, sid, root):
        if it['id'] == iid: return it
    raise ProjectError(f'no item {iid!r}')


def set_primary(pid, iid, variant, root=None):
    it = get_item(pid, iid, root)
    if variant not in {v['id'] for v in it['variants']}: raise ProjectError(f'item {iid!r} has no variant {variant!r}')
    st = load_state(pid, root); st.setdefault(iid, {})['primary'] = variant; _save_state(pid, st, root); return get_item(pid, iid, root)
