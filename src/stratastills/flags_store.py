"""Where the flags set on a visual report's pictures are kept (`neutral`, `auto`): {pic: {flag, rating, reasons, note, status, at, first}}. Two stores with the same methods, so server/app.py does not care which one it has:
- `FileStore`: `<settings.report_flags_dir()>/<report>.json`, one small file per report, read, one picture changed, written whole through a temporary file, exactly as before. The default, and what the unit tests use.
- `DbStore` (src/stratastills/db/flags.py, imported only when it is switched on): `report_flag` of schema `photos`, used when `settings.database_url()` is set AND 'flags' is in `settings.db_reads()`. A flag taken off is MARKED removed in the table, not deleted. It also writes everything to the files (dual-write: the files stay a fresh backup) and reads the files' flags under the database's, so a flag only the file has still shows."""

import json
import os
import threading

from stratastills import settings

REPORTS = ("neutral", "auto")
_lock = threading.Lock()


class FileStore:
    kind = "files"

    def _f(self, report):
        return os.path.join(settings.report_flags_dir(), f"{report}.json")

    def read(self, report):
        try:
            d = json.load(open(self._f(report), encoding="utf8"))
        except (OSError, ValueError):
            return {}
        return d if isinstance(d, dict) else {}

    def write(self, report, pic, entry):
        """Set one picture's flag (`entry` a dict) or take it off (None); returns the picture's flag, {} when none."""
        with _lock:
            cur = self.read(report)
            if entry:
                cur[pic] = entry
            else:
                cur.pop(pic, None)
            f = self._f(report)
            os.makedirs(os.path.dirname(f), exist_ok=True)
            tmp = f + ".tmp"
            json.dump(cur, open(tmp, "w", encoding="utf8"), indent=1, sort_keys=True)
            os.replace(tmp, f)
        return cur.get(pic, {})


_stores = {}


def get():
    """The store to use now: the database when settings.database_url() is set AND 'flags' is in settings.db_reads(), else the files. A DbStore is made once per URL."""
    url = settings.database_url()
    if not url or "flags" not in settings.db_reads():
        return FileStore()
    if url not in _stores:
        from .db.flags import DbStore  # needs strata-python[db]; only imported when the database is switched on

        _stores[url] = DbStore(url)
    return _stores[url]
