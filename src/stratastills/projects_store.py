"""Where the app keeps a project's document (project.json) and the person's item choices (items.json). Two stores with the same methods, so projects.py does not care which one it has:
- `FileStore`: the two files in the project's folder, exactly as before. The default, and what the unit tests use.
- `DbStore` (src/stratastills/db/projects.py, imported only when it is switched on): schema `photos` (project, project_source, project_rule, item_choice), used when `settings.database_url()` is set AND 'projects' is in `settings.db_reads()` (the switch is deliberate, see settings.db_reads). It ALSO writes everything to the files (dual-write: the files stay a fresh backup) and where the database has nothing for an item it falls back to the files' entry.
Only the document and the choices are covered here; the generated variants and the sidecars are a separate step."""
import json, os

from stratastills import settings

FILE = 'project.json'
STATE = 'items.json'


def _read(path):
    try: return json.load(open(path))
    except (OSError, ValueError): return None


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


_stores = {}


def get():
    """The store to use now: the database when settings.database_url() is set AND 'projects' is in settings.db_reads(), else the files. A DbStore is made once per URL."""
    url = settings.database_url()
    if not url or 'projects' not in settings.db_reads(): return FileStore()
    if url not in _stores:
        from .db.projects import DbStore                                                            # needs requirements-db.txt; only imported when the database is switched on
        _stores[url] = DbStore(url)
    return _stores[url]
