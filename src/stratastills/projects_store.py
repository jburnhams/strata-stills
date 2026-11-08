"""Where the app keeps a project's document (project.json) and the person's item choices (items.json). Two stores with the same methods, so projects.py does not care which one it has:
- `FileStore`: the two files in the project's folder, exactly as before. The default, and what the unit tests use.
- `DbStore` (src/stratastills/db/projects.py, imported only when it is switched on): schema `photos` (project, project_source, project_rule, item_choice), used when `settings.database_url()` is set AND 'projects' is in `settings.db_reads()` (the switch is deliberate, see settings.db_reads). It ALSO writes everything to the files (dual-write: the files stay a fresh backup) and where the database has nothing for an item it falls back to the files' entry.
The generated variants (`generated`, `read_sidecar`, `write_sidecar`) are the sidecar JSONs the app writes next to each generated picture. The picture itself always stays a file. Where the project folder holds outputs made elsewhere, `FileStore.generated` finds them by the item's stem within its source and by the sidecar's file NAME when the recorded path is missing (see its docstring): on machines where the ids and paths already match nothing changes."""
import json, os

from stratastills import settings

FILE = 'project.json'
STATE = 'items.json'
GENERATED = 'generated'                 # <project>/generated/<item id>/<key>.jpg|.tif + <key>.json


def _read(path):
    try: return json.load(open(path))
    except (OSError, ValueError): return None


def _norm(p): return p.replace('\\', '/').rstrip('/').lower()


def _source_id(path):
    import hashlib
    return hashlib.sha1(os.path.normcase(os.path.abspath(path)).encode()).hexdigest()[:8]


def _write(path, data, indent=1):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f: json.dump(data, f, indent=indent)
    os.replace(tmp, path)


class FileStore:
    kind = 'files'

    def read_doc(self, pid, folder): return _read(os.path.join(folder, FILE))

    def write_doc(self, pid, folder, doc): _write(os.path.join(folder, FILE), doc)

    def read_state(self, pid, folder):
        d = _read(os.path.join(folder, STATE)); return d if isinstance(d, dict) else {}

    def write_state(self, pid, folder, d): _write(os.path.join(folder, STATE), d)

    # ---- the generated variants: one sidecar JSON per picture, <folder>/generated/<item id>/<key>.json ----------------------------------------------------
    def generated_dir(self, folder, iid): return os.path.join(folder, GENERATED, iid.replace('/', '_'))

    def generated(self, pid, folder, iid, sources=()):
        """The sidecar records of the item's generated pictures whose picture file exists, [(record, path of the sidecar)] by file name. The item's own folder (<folder>/generated/<iid>) first. A project made on another machine has its outputs under that machine's item ids (the sha1 of the source path differs per machine) and its records name that machine's paths, so: (1) when the item's own folder does not exist, a folder `generated/*~<stem>` is the item's if one of its records says it came from the item's source (`source` under the stored path of the source whose id is the item's prefix; in a one-source project that source); (2) when a record's `file` does not exist, the file of that NAME beside the record is used."""
        d = self.generated_dir(folder, iid); dirs = [d] if os.path.isdir(d) else []                                  # the item's own folder (it may exist and be empty: a failed encode)
        sid, _, stem = iid.partition('~'); src = next((x for x in sources if _source_id(x) == sid), None) or (sources[0] if len(sources) == 1 else None); g = os.path.join(folder, GENERATED)
        if stem and src and os.path.isdir(g):
            want = _norm(src)                                                                                      # a folder made elsewhere counts when a sidecar's source file is directly IN the item's source folder (the scan is not recursive, so a sub-folder's items are other items)
            for n in sorted(os.listdir(g)):
                nd = os.path.join(g, n)
                if n.partition('~')[2] != stem or not os.path.isdir(nd) or nd in dirs: continue
                recs = [_read(os.path.join(nd, f)) for f in sorted(os.listdir(nd)) if f.endswith('.json')]
                if any(isinstance(r, dict) and r.get('source') and _norm(os.path.dirname(r['source'].replace('\\', '/'))) == want for r in recs): dirs.append(nd)
        out, seen = [], set()
        for d in dirs:                                                                                             # the item's own folder first: its record wins over a migrated one with the same key
            for n in sorted(os.listdir(d)):
                if not n.endswith('.json'): continue
                r = _read(os.path.join(d, n))
                if not isinstance(r, dict) or 'key' not in r or r['key'] in seen: continue
                f = r.get('file') or ''
                if not os.path.exists(f):
                    alt = os.path.join(d, f.replace('\\', '/').rsplit('/', 1)[-1]) if f else ''
                    if not (alt and os.path.exists(alt)): continue
                    r = dict(r, file=alt)
                seen.add(r['key']); out.append(r)
        return out

    def read_sidecar(self, pid, folder, iid, key, sources=()):
        return next((r for r in self.generated(pid, folder, iid, sources) if r.get('key') == key), None)

    def write_sidecar(self, pid, folder, iid, key, rec):
        d = self.generated_dir(folder, iid); os.makedirs(d, exist_ok=True); _write(os.path.join(d, key + '.json'), rec)


_stores = {}


def get():
    """The store to use now: the database when settings.database_url() is set AND 'projects' is in settings.db_reads(), else the files. A DbStore is made once per URL."""
    url = settings.database_url()
    if not url or 'projects' not in settings.db_reads(): return FileStore()
    if url not in _stores:
        from .db.projects import DbStore                                                            # needs requirements-db.txt; only imported when the database is switched on
        _stores[url] = DbStore(url)
    return _stores[url]
