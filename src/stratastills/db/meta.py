"""The DJI drone fields from `photo_meta.dji` (see meta_store.py for the contract it shares with FileStore, and for the switch: `database_url` AND 'metadata' in `db_reads`).
exiftool's XMP-drone-dji group is what the table holds: the same values as the XMP attributes the file store reads, with different spelling and types (`GPSLatitude` against `GpsLatitude`, numbers as signed strings like '+22.143'), so `fields()` turns them into exactly what `meta.dji_fields` gives. The file is found through `file` (root label + path below it; roots are the source roots plus the optional `extra_roots`), its photo through `file.photo_id`, a current file preferred. A file or photo the database does not know falls back to reading the file."""
import os

from sqlalchemy import select

from stratastills import meta, settings
from stratastills.meta_store import FileStore


def fields(dji):
    """`meta.dji_fields` of an exiftool XMP-drone-dji dict: the same keys, numbers as floats, text as is; a missing or unreadable one is left out."""
    low = {str(k).lower(): v for k, v in (dji or {}).items()}; out = {}
    for k in meta.DJI_NUMERIC:
        try: out[k] = float(low[k.lower()])
        except (KeyError, ValueError, TypeError): pass
    for k in meta.DJI_TEXT:
        if low.get(k.lower()): out[k] = low[k.lower()]
    return out


class DbStore:
    kind = 'database'

    def __init__(self, url, roots=None):
        from omarchy_db import connect
        self.engine = connect(url); self.files = FileStore(); self._roots = roots

    def roots(self):
        from .hashes import root_labels
        dirs = self._roots if self._roots is not None else list(settings.source_roots()) + list(settings.load().get('extra_roots') or [])
        return root_labels(dirs) if dirs else []

    def _where(self, path):
        ap = os.path.normpath(path)
        for name, root in self.roots():
            if ap.startswith(root + os.sep): return name, os.path.relpath(ap, root).replace(os.sep, '/')
        return None

    def dji(self, path):
        from .models import File, PhotoMeta
        loc = self._where(path)
        if loc is not None:
            from omarchy_db import session
            with session(self.engine) as s:
                f = s.scalars(select(File).where(File.root == loc[0], File.relpath == loc[1], File.photo_id.is_not(None)).order_by(File.current.desc(), File.id.desc())).first()
                m = s.get(PhotoMeta, f.photo_id) if f is not None else None
                if m is not None and m.dji: return fields(m.dji)
        return self.files.dji(path)
