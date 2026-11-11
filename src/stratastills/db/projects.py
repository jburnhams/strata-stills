"""The project document and the item choices on schema `photos` (see projects_store.py for the contract it shares with FileStore, and for the switch: `database_url` AND 'projects' in `db_reads`).
The document: `project` (slug = the project id, name, created as UTC), `project_source` (root label + path below it as this machine knows it, and `original_path`, the path as the document stored it), `project_rule` (the `auto` rules). Reading: the file document is the base (other keys it may carry), the database's name, created, sources and rules go on top. `created` comes back as an ISO time in UTC (the file's offset is not kept; the instant is the same).
The choices: `item_choice` (project, photo, primary variant). An item id ('<sha1 of the source path>~<stem>') names a picture through its stem within one of the project's sources, so the choice is tied to the photo whose raw has that stem in that source folder; an item with no such photo (a shoot not imported yet) has its choice in the file only. Reading adds the files' choices under the database's. Every write goes to the files too (dual-write)."""

import os
from datetime import UTC, datetime

from sqlalchemy import select

from stratastills import settings
from stratastills.projects_store import FileStore


def _iso(d):
    return d.replace(tzinfo=UTC).isoformat(timespec="seconds") if d else None


def _utc(s):
    try:
        d = datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None
    return d.astimezone(UTC).replace(tzinfo=None) if d.tzinfo else d


class DbStore:
    kind = "database"

    def __init__(self, url, roots=None):
        from omarchy_db import connect

        self.engine = connect(url)
        self.files = FileStore()
        self._roots = roots

    def _session(self):
        from omarchy_db import session

        return session(self.engine)

    def roots(self):
        from .hashes import root_labels

        dirs = self._roots if self._roots is not None else list(settings.source_roots())
        return root_labels(dirs) if dirs else []

    def _locate(self, path):
        from .projimport import locate

        loc = locate(path, [d for _, d in self.roots()]) if self.roots() else None
        return loc or ("abs", os.path.normpath(path).replace(os.sep, "/"))

    # ---- the document -------------------------------------------------------------------------------------------------------------------------------------
    def read_doc(self, pid, folder):
        from .models import Project, ProjectRule, ProjectSource

        base = self.files.read_doc(pid, folder)
        with self._session() as s:
            p = s.scalars(select(Project).where(Project.slug == pid)).first()
            if p is None:
                return base
            doc = dict(base or {})
            doc.update(name=p.name, created=_iso(p.created))
            doc["sources"] = [
                dict(path=x.original_path or x.relpath, added=_iso(x.added), rel=x.relpath)
                for x in s.scalars(
                    select(ProjectSource).where(ProjectSource.project_id == p.id).order_by(ProjectSource.id)
                )
            ]
            doc["auto"] = [
                dict({"source": r.source, "pipeline": r.pipeline}, **({"variant": r.variant} if r.variant else {}))
                for r in s.scalars(select(ProjectRule).where(ProjectRule.project_id == p.id).order_by(ProjectRule.id))
            ]
            return doc

    def write_doc(self, pid, folder, doc):
        from .models import Project, ProjectRule, ProjectSource

        with self._session() as s:
            p = s.scalars(select(Project).where(Project.slug == pid)).first()
            if p is None:
                p = Project(slug=pid, name=doc.get("name") or pid, created=_utc(doc.get("created")))
                s.add(p)
                s.flush()
            else:
                p.name, p.created = doc.get("name") or p.name, _utc(doc.get("created")) or p.created
            have = {
                (x.root, x.relpath): x for x in s.scalars(select(ProjectSource).where(ProjectSource.project_id == p.id))
            }
            want = {}
            for src in doc.get("sources", []):
                root, rel = self._locate(src["path"])
                want[(root, rel)] = src
                if (root, rel) in have:
                    have[(root, rel)].original_path = src["path"]
                else:
                    s.add(
                        ProjectSource(
                            project_id=p.id,
                            root=root,
                            relpath=rel,
                            added=_utc(src.get("added")),
                            original_path=src["path"],
                        )
                    )
            for k, x in have.items():
                if k not in want:
                    s.delete(x)  # a source taken out of the document
            for r in s.scalars(select(ProjectRule).where(ProjectRule.project_id == p.id)):
                s.delete(r)
            s.flush()
            for r in doc.get("auto", []):
                s.add(
                    ProjectRule(
                        project_id=p.id, source=r.get("source", "*"), pipeline=r["pipeline"], variant=r.get("variant")
                    )
                )
            s.commit()
        self.files.write_doc(pid, folder, doc)

    # ---- the choices ---------------------------------------------------------------------------------------------------------------------------------------
    def _sources(self, s, p):
        from .models import ProjectSource

        return [
            (x.root, x.relpath, x.original_path)
            for x in s.scalars(select(ProjectSource).where(ProjectSource.project_id == p.id).order_by(ProjectSource.id))
        ]

    def _photo(self, s, sources, iid):
        """The photo an item id names: the CURRENT raw file named <stem>.* in the item's own source folder, exactly one photo. The item's source is the one whose id (the sha1 prefix) the item carries; an id made elsewhere (the prefix matches none of this machine's) names a source only in a one-source project. An item with no such photo stays file-only (None)."""
        from stratastills import projects as P

        from .models import File, Photo

        sid, _, stem = iid.partition("~")
        if not stem:
            return None
        mine = [x for x in sources if P.source_id(x[2] or x[1]) == sid] or (list(sources) if len(sources) == 1 else [])
        out = set()
        for root, rel, _ in mine:
            for f in s.scalars(
                select(File).where(
                    File.root == root, File.role == "raw", File.current.is_(True), File.relpath.like(f"{rel}/{stem}.%")
                )
            ):
                if (
                    os.path.splitext(os.path.basename(f.relpath))[0] == stem
                    and os.path.dirname(f.relpath) == rel
                    and f.photo_id
                ):
                    out.add(f.photo_id)
        return s.get(Photo, next(iter(out))) if len(out) == 1 else None

    def read_state(self, pid, folder):
        from stratastills import projects as P

        from .models import File, ItemChoice, Photo, Project

        out = self.files.read_state(pid, folder)
        with self._session() as s:
            p = s.scalars(select(Project).where(Project.slug == pid)).first()
            if p is None:
                return out
            sources = self._sources(s, p)
            for c in s.scalars(select(ItemChoice).where(ItemChoice.project_id == p.id)):
                ph = s.get(Photo, c.photo_id)
                if ph is None:
                    continue
                if ph is None:
                    continue
                for f in s.scalars(
                    select(File).where(File.photo_id == ph.id, File.role == "raw", File.current.is_(True))
                ):  # only the photo's current files: a superseded version names nothing
                    for r, rel, orig in sources:
                        if (
                            f.root == r
                            and os.path.dirname(f.relpath) == rel
                            and os.path.splitext(os.path.basename(f.relpath))[0] == ph.stem
                        ):
                            out.setdefault(P.item_id(P.source_id(orig or rel), ph.stem), {})["primary"] = (
                                c.primary_variant
                            )  # the id the app asks for on this machine
        return out

    def write_state(self, pid, folder, d):
        from .models import ItemChoice, Project

        with self._session() as s:
            p = s.scalars(select(Project).where(Project.slug == pid)).first()
            if p is not None:
                sources = self._sources(s, p)
                have = {c.photo_id: c for c in s.scalars(select(ItemChoice).where(ItemChoice.project_id == p.id))}
                for iid, v in d.items():
                    if not isinstance(v, dict) or not v.get("primary"):
                        continue
                    ph = self._photo(s, sources, iid)
                    if ph is None:
                        continue  # no photo for this item (yet): the file has it
                    if ph.id in have:
                        have[ph.id].primary_variant = v["primary"]
                    else:
                        s.add(ItemChoice(project_id=p.id, photo_id=ph.id, primary_variant=v["primary"]))
                s.commit()
        self.files.write_state(pid, folder, d)

    # ---- the generated variants ----------------------------------------------------------------------------------------------------------------------------
    def generated(self, pid, folder, iid, sources=()):
        """The sidecar records of the item's generated pictures: the `render` rows of the item's photo (the whole sidecar is in `info`, the picture's path is relative to the project folder), with the files' records for any key the database does not have (an item not imported yet). Only records whose picture file exists."""
        from .models import Project, Render

        recs = {r["key"]: r for r in self.files.generated(pid, folder, iid, sources)}
        with self._session() as s:
            p = s.scalars(select(Project).where(Project.slug == pid)).first()
            ph = self._photo(s, self._sources(s, p), iid) if p is not None else None
            if ph is not None:
                for r in s.scalars(
                    select(Render).where(Render.photo_id == ph.id, Render.kind == "generated").order_by(Render.id)
                ):
                    rec = dict(r.info or {}, key=r.cache_key)
                    f = os.path.join(folder, *(r.path or "").split("/")) if r.path else ""
                    own = os.path.abspath(folder) + os.sep
                    recorded = rec.get("file") or ""
                    if f and os.path.exists(f):
                        rec["file"] = f
                    elif recorded and os.path.abspath(recorded).startswith(own) and os.path.exists(recorded):
                        pass  # only a picture inside THIS project's folder: the same photo can be in other projects, whose renders share the row
                    else:
                        continue
                    recs[r.cache_key] = rec  # the database wins
        return list(recs.values())

    def read_sidecar(self, pid, folder, iid, key, sources=()):
        return next((r for r in self.generated(pid, folder, iid, sources) if r.get("key") == key), None)

    def write_sidecar(self, pid, folder, iid, key, rec):
        from .models import Project, Render

        with self._session() as s:
            p = s.scalars(select(Project).where(Project.slug == pid)).first()
            ph = self._photo(s, self._sources(s, p), iid) if p is not None else None
            if ph is not None:  # an item with no photo in the database keeps its sidecar in the file only
                out = rec.get("output") or [None, None]
                f = rec.get("file") or ""
                path = (
                    os.path.relpath(f, folder).replace(os.sep, "/")
                    if f and os.path.abspath(f).startswith(os.path.abspath(folder) + os.sep)
                    else f"generated/{iid.replace('/', '_')}/{os.path.basename(f.replace(chr(92), '/'))}"
                )
                r = s.scalars(select(Render).where(Render.kind == "generated", Render.cache_key == key)).first()
                vals = dict(
                    photo_id=ph.id,
                    variant=rec.get("source_variant"),
                    preset=rec.get("preset"),
                    stages=rec.get("stages"),
                    summary=rec.get("summary"),
                    out_w=out[0],
                    out_h=out[1],
                    bytes=rec.get("bytes"),
                    quality=rec.get("quality"),
                    path=path,
                    info=rec,
                )
                if r is None:
                    s.add(Render(cache_key=key, kind="generated", **vals))  # pyright: ignore[reportArgumentType]
                else:
                    for k, v in vals.items():
                        setattr(r, k, v)
                s.commit()
        self.files.write_sidecar(pid, folder, iid, key, rec)  # dual-write: the file stays a fresh backup
