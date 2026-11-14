"""Where the app gets a photo's DJI drone fields (gimbal and flight attitude, position) from. Two stores with the same method, so meta.py does not care which one it has:
- `FileStore`: the XMP packet in the head of the file (`meta.read_xmp` + `meta.dji_fields`), exactly as before. The default, and what the unit tests use.
- `DbStore` (src/stratastills/db/meta.py, imported only when it is switched on): `photo_meta.dji` of the photo the file belongs to (the metadata exiftool read from the drone raw, loaded by scripts/import_meta.py), used when `settings.database_url()` is set AND 'metadata' is in `settings.db_reads()`. A file the database does not know falls back to the file read, so a new shoot works before it is imported. Read-only: nothing is written, so there is no dual-write here.
Only the drone fields are covered: they are per PHOTO (a DNG and the JPEG beside it share the gimbal pitch). What `meta.photo_info` reports about the JPEG itself (its width and height, creator tool, panorama flag) is per FILE and stays a read of the file (it is the `scan --info` diagnostic, not a hot path)."""

from stratastills import settings


class FileStore:
    kind = "files"

    def dji(self, path):
        from stratastills import meta

        return meta.dji_fields(meta.read_xmp(path))


_stores = {}


def get():
    """The store to use now: the database when settings.database_url() is set AND 'metadata' is in settings.db_reads(), else the files. A DbStore is made once per URL."""
    url = settings.database_url()
    if not url or "metadata" not in settings.db_reads():
        return FileStore()
    if url not in _stores:
        from .db.meta import DbStore  # needs strata-python[db]; only imported when the database is switched on

        _stores[url] = DbStore(url)
    return _stores[url]
