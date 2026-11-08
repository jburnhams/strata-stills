"""Where the nudity filter keeps what it has learned. Two stores with the same methods, so nsfw.py does not care which one it has:
- `FileStore`: `<STRATASTILLS_HOME>/nsfw_cache.json` (key -> [score, class]), `nsfw_allowed.json` (key -> time of the review decision) and the append-only `nsfw_skipped.jsonl`, exactly as before. The default, and what the unit tests use.
- `DbStore`: schema `photos` of the shared database, used when `settings.database_url()` is set AND `db_reads` (settings.json, or STRATASTILLS_DB_READS) contains 'nsfw': scores in `nsfw_result` (one row per file version and model; the file in `file`), the person's decisions in `nsfw_decision`. It ALSO writes everything it records to the files (dual-write), so the files stay a fresh backup; it only reads the database. The skip log has no table and stays a file.
A key is `model|absolute path|size|mtime_ns`. The database knows a file as (root label, path below the root): the roots are the app's source roots plus the optional `extra_roots` list of settings.json, labelled by their folder name; a path under none of them is kept under root 'abs' with the absolute path. A database row whose root is not configured cannot be turned back into a path and is left out of `load()` (counted in `unresolved`).
A decision in the database belongs to a PICTURE (`nsfw_decision.photo_id`); the file store's belonged to one file version (size and time stamp). The DbStore keeps the narrower meaning: a decision counts for a key only when that exact file version is the one recorded for the photo, so a changed file has to be cleared again. A file that is not part of a photo in the database cannot be given a decision there (it is recorded in the file only)."""
import json, os, threading, time

from stratastills import settings


class FileStore:
    kind = 'files'

    def _home(self): return settings.home()

    def _cache_file(self): return os.path.join(self._home(), 'nsfw_cache.json')

    def _allowed_file(self): return os.path.join(self._home(), 'nsfw_allowed.json')

    def load(self):
        try: return json.load(open(self._cache_file()))
        except (OSError, ValueError): return {}

    def put(self, key, value, cache):
        """Remember one score; the file store rewrites the whole cache (`cache` is the in-memory dict that already holds `key`)."""
        os.makedirs(self._home(), exist_ok=True); tmp = f'{self._cache_file()}.{os.getpid()}.{threading.get_ident()}.tmp'
        with open(tmp, 'w') as f: json.dump(cache, f)
        os.replace(tmp, self._cache_file())

    def _allowed(self):
        try: return json.load(open(self._allowed_file()))
        except (OSError, ValueError): return {}

    def allowed_among(self, keys):
        a = self._allowed(); return {k for k in keys if k in a}

    def set_allowed(self, key, yes):
        a = self._allowed()
        if yes: a[key] = int(time.time())
        else: a.pop(key, None)
        os.makedirs(self._home(), exist_ok=True); tmp = f'{self._allowed_file()}.{os.getpid()}.{threading.get_ident()}.tmp'
        with open(tmp, 'w') as f: json.dump(a, f)
        os.replace(tmp, self._allowed_file())

    def record_skip(self, line):
        os.makedirs(self._home(), exist_ok=True)
        with open(os.path.join(self._home(), 'nsfw_skipped.jsonl'), 'a', encoding='utf8') as f: f.write(json.dumps(line) + '\n')


def parse_key(key):
    """(model, absolute path, size, mtime_ns) of a cache key."""
    model, _, rest = key.partition('|'); path, size, mt = rest.rsplit('|', 2); return model, path, int(size), int(mt)


class DbStore:
    kind = 'database'

    def __init__(self, url, roots=None):
        from omarchy_db import connect
        self.engine = connect(url); self.files = FileStore(); self.unresolved = 0; self._roots = roots

    def roots(self):
        """[(label, absolute dir)] longest dir first."""
        from .db.hashes import root_labels
        dirs = self._roots if self._roots is not None else list(settings.source_roots()) + list(settings.load().get('extra_roots') or [])
        return root_labels(dirs) if dirs else []

    def _session(self):
        from omarchy_db import session
        return session(self.engine)

    def _where(self, path):
        ap = os.path.normpath(path)
        for name, root in self.roots():
            if ap.startswith(root + os.sep): return name, os.path.relpath(ap, root).replace(os.sep, '/')
        return 'abs', ap.replace(os.sep, '/')

    def _path(self, root, relpath):
        if root == 'abs': return relpath
        for name, d in self.roots():
            if name == root: return os.path.join(d, *relpath.split('/'))
        return None

    def load(self):
        from sqlalchemy import select
        from .db.models import File, NsfwResult
        out, self.unresolved = {}, 0
        with self._session() as s:
            for m, score, label, root, rel, size, mt in s.execute(select(NsfwResult.model, NsfwResult.score, NsfwResult.label, File.root, File.relpath, File.size, File.mtime_ns).join(File, File.id == NsfwResult.file_id)):
                p = self._path(root, rel)
                if p is None: self.unresolved += 1; continue
                out[f'{m}|{os.path.abspath(p)}|{size}|{mt}'] = [score, label]
        return out

    def _file(self, s, root, rel, size, mt, create):
        from sqlalchemy import select
        from .db.models import File
        f = s.scalars(select(File).where(File.root == root, File.relpath == rel, File.size == size, File.mtime_ns == mt)).first()
        if f is None and create:
            f = File(root=root, relpath=rel, size=size, mtime_ns=mt, current=True); s.add(f); s.flush()
        return f

    def put(self, key, value, cache):
        from sqlalchemy import select
        from .db.models import NsfwResult
        model, path, size, mt = parse_key(key); root, rel = self._where(path)
        with self._session() as s:
            f = self._file(s, root, rel, size, mt, True)
            r = s.scalars(select(NsfwResult).where(NsfwResult.file_id == f.id, NsfwResult.model == model)).first()
            if r is None: s.add(NsfwResult(file_id=f.id, model=model, score=float(value[0]), label=str(value[1] or '')))
            else: r.score, r.label = float(value[0]), str(value[1] or '')
            s.commit()
        self.files.put(key, value, cache)                                                                         # dual-write: the files stay a fresh backup

    def allowed_among(self, keys):
        from sqlalchemy import select
        from .db.models import NsfwDecision
        out = set()
        with self._session() as s:
            if not s.scalars(select(NsfwDecision.photo_id).where(NsfwDecision.allowed.is_(True)).limit(1)).first(): return out              # no decision at all (the usual case): nothing to look up per key
            for k in keys:
                model, path, size, mt = parse_key(k); root, rel = self._where(path); f = self._file(s, root, rel, size, mt, False)
                if f is not None and f.photo_id is not None:
                    d = s.get(NsfwDecision, f.photo_id)
                    if d is not None and d.allowed: out.add(k)
        return out

    def set_allowed(self, key, yes):
        from .db.models import NsfwDecision
        model, path, size, mt = parse_key(key); root, rel = self._where(path)
        with self._session() as s:
            f = self._file(s, root, rel, size, mt, False)
            if f is not None and f.photo_id is not None:
                d = s.get(NsfwDecision, f.photo_id)
                if d is None: s.add(NsfwDecision(photo_id=f.photo_id, allowed=bool(yes)))
                else: d.allowed = bool(yes)
                s.commit()
        self.files.set_allowed(key, yes)

    def record_skip(self, line): self.files.record_skip(line)


_stores = {}


def get():
    """The store to use now: the database when settings.database_url() is set AND 'nsfw' is in settings.db_reads() (the switch is deliberate, see settings.db_reads), else the files. A DbStore is made once per URL."""
    url = settings.database_url()
    if not url or 'nsfw' not in settings.db_reads(): return FileStore()
    if url not in _stores: _stores[url] = DbStore(url)
    return _stores[url]
