"""The Browser's reads from schema `photos` (browser.py): which raws of a folder have a Lightroom export, and where that export is on this machine. Roots are the source roots plus the optional `extra_roots` (the exports are under `personal`), labelled by folder name as everywhere in the database."""

import os

from sqlalchemy import select
from sqlalchemy.orm import aliased

from stratastills import settings

_engines = {}


def _engine(url):
    if url not in _engines:
        from strata_python.db import connect

        _engines[url] = connect(url)
    return _engines[url]


def lightroom_finals(folder, url=None, roots=None):
    """{stem: absolute path of the Lightroom export} for the raws directly in `folder` (current file rows only; an export whose root is not configured here is left out; whether the file is on disk is checked when it is read, not here: on the archive disk that is a seek per picture)."""
    from strata_python.db import session

    from .hashes import root_labels
    from .models import File, LightroomEdit

    dirs = (
        roots if roots is not None else list(settings.source_roots()) + list(settings.load().get("extra_roots") or [])
    )
    labels = root_labels(dirs)
    where = dict(labels)
    ap = os.path.normpath(folder)
    loc = next(
        ((n, os.path.relpath(ap, r).replace(os.sep, "/")) for n, r in labels if ap == r or ap.startswith(r + os.sep)),
        None,
    )
    if loc is None:
        return {}
    prefix = "" if loc[1] == "." else loc[1] + "/"
    raw, final = aliased(File), aliased(File)
    out = {}
    with session(_engine(url or settings.database_url())) as s:
        rows = s.execute(
            select(raw.relpath, final.root, final.relpath)
            .join(LightroomEdit, LightroomEdit.photo_id == raw.photo_id)
            .join(final, final.id == LightroomEdit.final_file_id)
            .where(
                raw.root == loc[0],
                raw.role == "raw",
                raw.current.is_(True),
                raw.relpath.like(prefix.replace("%", r"\%").replace("_", r"\_") + "%", escape="\\"),
            )
        ).all()
    for rel, froot, frel in rows:
        rest = rel[len(prefix) :]
        if "/" in rest or froot not in where:
            continue
        out[os.path.splitext(rest)[0]] = os.path.join(where[froot], *frel.split("/"))
    return out
