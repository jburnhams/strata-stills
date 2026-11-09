"""Where the nudity filter keeps what it has learned. Two stores with the same methods, so nsfw.py does not care which one it has:
- `FileStore`: `<STRATASTILLS_HOME>/nsfw_cache.json` (key -> [score, class]), `nsfw_allowed.json` (key -> time of the review decision) and the append-only `nsfw_skipped.jsonl`, exactly as before. The default, and what the unit tests use.
- `DbStore` (src/stratastills/db/nsfw.py, imported only when it is switched on): schema `photos` of the shared database, used when `settings.database_url()` is set AND `db_reads` (settings.json, or STRATASTILLS_DB_READS) contains 'nsfw': scores in `nsfw_result` (one row per file version and model; the file in `file`), the person's decisions in `nsfw_decision`. It ALSO writes everything it records to the files (dual-write), so the files stay a fresh backup; it only reads the database. The skip log has no table and stays a file.
A key is `model|absolute path|size|mtime_ns`. The database knows a file as (root label, path below the root): the roots are the app's source roots plus the optional `extra_roots` list of settings.json, labelled by their folder name; a path under none of them is kept under root 'abs' with the absolute path. A database row whose root is not configured cannot be turned back into a path and is left out of `load()` (counted in `unresolved`).
A decision in the database belongs to a PICTURE (`nsfw_decision.photo_id`); the file store's belonged to one file version (size and time stamp). The DbStore keeps the narrower meaning: a decision counts for a key only when that exact file version is the one recorded for the photo, so a changed file has to be cleared again. A file that is not part of a photo in the database cannot be given a decision there (it is recorded in the file only)."""

import json
import os
import threading
import time

from stratastills import settings


class FileStore:
    kind = "files"

    def _home(self):
        return settings.home()

    def _cache_file(self):
        return os.path.join(self._home(), "nsfw_cache.json")

    def _allowed_file(self):
        return os.path.join(self._home(), "nsfw_allowed.json")

    def load(self):
        try:
            return json.load(open(self._cache_file()))
        except (OSError, ValueError):
            return {}

    def put(self, key, value, cache):
        """Remember one score; the file store rewrites the whole cache (`cache` is the in-memory dict that already holds `key`)."""
        os.makedirs(self._home(), exist_ok=True)
        tmp = f"{self._cache_file()}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, self._cache_file())

    def _allowed(self):
        try:
            return json.load(open(self._allowed_file()))
        except (OSError, ValueError):
            return {}

    def allowed_among(self, keys):
        a = self._allowed()
        return {k for k in keys if k in a}

    def set_allowed(self, key, yes):
        a = self._allowed()
        if yes:
            a[key] = int(time.time())
        else:
            a.pop(key, None)
        os.makedirs(self._home(), exist_ok=True)
        tmp = f"{self._allowed_file()}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w") as f:
            json.dump(a, f)
        os.replace(tmp, self._allowed_file())

    def record_skip(self, line):
        os.makedirs(self._home(), exist_ok=True)
        with open(os.path.join(self._home(), "nsfw_skipped.jsonl"), "a", encoding="utf8") as f:
            f.write(json.dumps(line) + "\n")


def parse_key(key):
    """(model, absolute path, size, mtime_ns) of a cache key."""
    model, _, rest = key.partition("|")
    path, size, mt = rest.rsplit("|", 2)
    return model, path, int(size), int(mt)


_stores = {}


def get():
    """The store to use now: the database when settings.database_url() is set AND 'nsfw' is in settings.db_reads() (the switch is deliberate, see settings.db_reads), else the files. A DbStore is made once per URL."""
    url = settings.database_url()
    if not url or "nsfw" not in settings.db_reads():
        return FileStore()
    if url not in _stores:
        from .db.nsfw import DbStore  # needs omarchy-db (dev group); only imported when the database is switched on

        _stores[url] = DbStore(url)
    return _stores[url]
