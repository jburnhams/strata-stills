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
