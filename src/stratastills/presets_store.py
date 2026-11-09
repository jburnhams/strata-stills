"""Where the person's own pipeline presets are kept (the built-in ones are code and are not stored). Two stores with the same methods, so pipeline.py does not care which one it has:
- `FileStore`: `<STRATASTILLS_HOME>/pipelines/<name>.json` ({label, stages}), exactly as before. The default, and what the unit tests use.
- `DbStore` (src/stratastills/db/presets.py, imported only when it is switched on): `pipeline_preset` of schema `photos`, used when `settings.database_url()` is set AND 'presets' is in `settings.db_reads()`. It also writes to the files (dual-write: the files stay a fresh backup) and reads the files' presets under the database's. Deleting removes the row and the file."""

import json
import os

from stratastills import settings


class FileStore:
    kind = "files"

    def _dir(self):
        return os.path.join(settings.home(), "pipelines")

    def list(self):
        """[{name, label, stages}] of the saved presets, by name; a file that cannot be read is left out."""
        d = self._dir()
        out = []
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if not f.endswith(".json"):
                    continue
                try:
                    r = json.load(open(os.path.join(d, f)))
                    out.append(dict(name=f[:-5], label=r.get("label") or f[:-5], stages=r["stages"]))
                except (OSError, ValueError, KeyError):
                    continue
        return out

    def save(self, name, label, stages):
        os.makedirs(self._dir(), exist_ok=True)
        with open(os.path.join(self._dir(), name + ".json"), "w") as f:
            json.dump(dict(label=label or name, stages=stages), f, indent=1)

    def delete(self, name):
        """True if there was one."""
        p = os.path.join(self._dir(), name + ".json")
        if not os.path.exists(p):
            return False
        os.remove(p)
        return True


_stores = {}


def get():
    """The store to use now: the database when settings.database_url() is set AND 'presets' is in settings.db_reads(), else the files. A DbStore is made once per URL."""
    url = settings.database_url()
    if not url or "presets" not in settings.db_reads():
        return FileStore()
    if url not in _stores:
        from .db.presets import DbStore  # needs omarchy-db (dev group); only imported when the database is switched on

        _stores[url] = DbStore(url)
    return _stores[url]
