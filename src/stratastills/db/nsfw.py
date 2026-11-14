"""The nudity filter's store on schema `photos` (see nsfw_store.py for the contract it shares with FileStore, and for the switch: `database_url` AND 'nsfw' in `db_reads`).
Files the database does not know: a score is kept in the database only for a file version that already has a `file` row (the hash importers make those, with their hash and photo); a score for any other file is kept in the files only, and `load()` adds the files' entries to the database's (the database wins), so nothing is lost and no placeholder `file` row without a hash is ever made (it would trip scripts/hash_new.py and db.hashes.write()). The dual-write MERGES the new score into the file cache that is on disk rather than writing the in-memory dict, so file-only entries survive whatever the roots or the database hold.
A decision is one row per photo (`nsfw_decision.photo_id`) and records the file version it was made on (`file_id`): it counts for that version only, and clearing another version of the same photo replaces it (the latest decision wins)."""

import os

from stratastills import settings
from stratastills.nsfw_store import FileStore, parse_key


class DbStore:
    kind = "database"

    def __init__(self, url, roots=None):
        from strata_python.db import connect

        self.engine = connect(url)
        self.files = FileStore()
        self.unresolved = 0
        self._roots = roots

    def roots(self):
        """[(label, absolute dir)] longest dir first."""
        from .hashes import root_labels

        dirs = (
            self._roots
            if self._roots is not None
            else list(settings.source_roots()) + list(settings.load().get("extra_roots") or [])
        )
        return root_labels(dirs) if dirs else []

    def _session(self):
        from strata_python.db import session

        return session(self.engine)

    def _where(self, path):
        ap = os.path.normpath(path)
        for name, root in self.roots():
            if ap.startswith(root + os.sep):
                return name, os.path.relpath(ap, root).replace(os.sep, "/")
        return "abs", ap.replace(os.sep, "/")

    def _path(self, root, relpath):
        if root == "abs":
            return relpath
        for name, d in self.roots():
            if name == root:
                return os.path.join(d, *relpath.split("/"))
        return None

    def load_db(self):
        """The scores the database holds (only), as cache entries."""
        from sqlalchemy import select

        from .models import File, NsfwResult

        out, self.unresolved = {}, 0
        with self._session() as s:
            for m, score, label, root, rel, size, mt in s.execute(
                select(
                    NsfwResult.model,
                    NsfwResult.score,
                    NsfwResult.label,
                    File.root,
                    File.relpath,
                    File.size,
                    File.mtime_ns,
                ).join(File, File.id == NsfwResult.file_id)
            ):
                p = self._path(root, rel)
                if p is None:
                    self.unresolved += 1
                    continue
                out[f"{m}|{os.path.abspath(p)}|{size}|{mt}"] = [score, label]
        return out

    def load(self):
        """The cache the filter starts from: the files' entries, with the database's on top."""
        return {**self.files.load(), **self.load_db()}

    def _file(self, s, root, rel, size, mt, create):
        from sqlalchemy import select

        from .models import File

        f = s.scalars(
            select(File).where(File.root == root, File.relpath == rel, File.size == size, File.mtime_ns == mt)
        ).first()
        if f is None and create:
            f = File(root=root, relpath=rel, size=size, mtime_ns=mt, current=True)
            s.add(f)
            s.flush()
        return f

    def put(self, key, value, cache):
        from sqlalchemy import select

        from .models import NsfwResult

        model, path, size, mt = parse_key(key)
        root, rel = self._where(path)
        with self._session() as s:
            f = self._file(s, root, rel, size, mt, False)
            if f is not None:  # only a file version the database already knows
                r = s.scalars(select(NsfwResult).where(NsfwResult.file_id == f.id, NsfwResult.model == model)).first()
                if r is None:
                    s.add(NsfwResult(file_id=f.id, model=model, score=float(value[0]), label=str(value[1] or "")))
                else:
                    r.score, r.label = float(value[0]), str(value[1] or "")
                s.commit()
        merged = self.files.load()
        merged[key] = list(value)
        self.files.put(key, value, merged)  # dual-write: merged into what is on disk

    def allowed_among(self, keys):
        from sqlalchemy import select

        from .models import NsfwDecision

        out = set()
        with self._session() as s:
            if not s.scalars(select(NsfwDecision.photo_id).where(NsfwDecision.allowed.is_(True)).limit(1)).first():
                return out  # no decision at all (the usual case): nothing to look up per key
            for k in keys:
                model, path, size, mt = parse_key(k)
                root, rel = self._where(path)
                f = self._file(s, root, rel, size, mt, False)
                if f is not None and f.photo_id is not None:
                    d = s.get(NsfwDecision, f.photo_id)
                    if d is not None and d.allowed and d.file_id == f.id:
                        out.add(k)
        return out

    def set_allowed(self, key, yes):
        from .models import NsfwDecision

        model, path, size, mt = parse_key(key)
        root, rel = self._where(path)
        with self._session() as s:
            f = self._file(s, root, rel, size, mt, False)
            if f is not None and f.photo_id is not None:
                d = s.get(NsfwDecision, f.photo_id)
                if yes:
                    if d is None:
                        s.add(NsfwDecision(photo_id=f.photo_id, allowed=True, file_id=f.id))
                    else:
                        d.allowed, d.file_id = True, f.id
                elif d is not None and d.file_id == f.id:
                    d.allowed = False
                s.commit()
        self.files.set_allowed(key, yes)

    def record_skip(self, line):
        self.files.record_skip(line)
