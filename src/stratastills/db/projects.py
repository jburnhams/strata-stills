"""The project document and the item choices on schema `photos` (see projects_store.py for the contract it shares with FileStore, and for the switch: `database_url` AND 'projects' in `db_reads`).
The document: `project` (slug = the project id, name, created as UTC), `project_source` (root label + path below it as this machine knows it, and `original_path`, the path as the document stored it), `project_rule` (the `auto` rules). Reading: the file document is the base (other keys it may carry), the database's name, created, sources and rules go on top. `created` comes back as an ISO time in UTC (the file's offset is not kept; the instant is the same).
The choices: `item_choice` (project, photo, primary variant). An item id ('<sha1 of the source path>~<stem>') names a picture through its stem within one of the project's sources, so the choice is tied to the photo whose raw has that stem in that source folder; an item with no such photo (a shoot not imported yet) has its choice in the file only. Reading adds the files' choices under the database's. Every write goes to the files too (dual-write)."""
import os
from datetime import datetime, timezone

from sqlalchemy import select

from stratastills import settings
from stratastills.projects_store import FileStore


def _iso(d): return d.replace(tzinfo=timezone.utc).isoformat(timespec='seconds') if d else None


def _utc(s):
    try: d = datetime.fromisoformat(s)
    except (TypeError, ValueError): return None
    return d.astimezone(timezone.utc).replace(tzinfo=None) if d.tzinfo else d


class DbStore:
    kind = 'database'

    def __init__(self, url, roots=None):
        from omarchy_db import connect
        self.engine = connect(url); self.files = FileStore(); self._roots = roots

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
        return loc or ('abs', os.path.normpath(path).replace(os.sep, '/'))

    # ---- the document -------------------------------------------------------------------------------------------------------------------------------------
    def read_doc(self, pid, folder):
        from .models import Project, ProjectRule, ProjectSource
        base = self.files.read_doc(pid, folder)
        with self._session() as s:
            p = s.scalars(select(Project).where(Project.slug == pid)).first()
            if p is None: return base
            doc = dict(base or {}); doc.update(name=p.name, created=_iso(p.created))
            doc['sources'] = [dict(path=x.original_path or x.relpath, added=_iso(x.added), rel=x.relpath) for x in s.scalars(select(ProjectSource).where(ProjectSource.project_id == p.id).order_by(ProjectSource.id))]
            doc['auto'] = [dict({'source': r.source, 'pipeline': r.pipeline}, **({'variant': r.variant} if r.variant else {})) for r in s.scalars(select(ProjectRule).where(ProjectRule.project_id == p.id).order_by(ProjectRule.id))]
            return doc

    def write_doc(self, pid, folder, doc):
        from .models import Project, ProjectRule, ProjectSource
        with self._session() as s:
            p = s.scalars(select(Project).where(Project.slug == pid)).first()
            if p is None: p = Project(slug=pid, name=doc.get('name') or pid, created=_utc(doc.get('created'))); s.add(p); s.flush()
            else: p.name, p.created = doc.get('name') or p.name, _utc(doc.get('created')) or p.created
            have = {(x.root, x.relpath): x for x in s.scalars(select(ProjectSource).where(ProjectSource.project_id == p.id))}; want = {}
            for src in doc.get('sources', []):
                root, rel = self._locate(src['path']); want[(root, rel)] = src
                if (root, rel) in have: have[(root, rel)].original_path = src['path']
                else: s.add(ProjectSource(project_id=p.id, root=root, relpath=rel, added=_utc(src.get('added')), original_path=src['path']))
            for k, x in have.items():
                if k not in want: s.delete(x)                                                                                        # a source taken out of the document
            for r in s.scalars(select(ProjectRule).where(ProjectRule.project_id == p.id)): s.delete(r)
            s.flush()
            for r in doc.get('auto', []): s.add(ProjectRule(project_id=p.id, source=r.get('source', '*'), pipeline=r['pipeline'], variant=r.get('variant')))
            s.commit()
        self.files.write_doc(pid, folder, doc)

    # ---- the choices ---------------------------------------------------------------------------------------------------------------------------------------
    def _sources(self, s, p):
        from .models import ProjectSource
        return [(x.root, x.relpath, x.original_path) for x in s.scalars(select(ProjectSource).where(ProjectSource.project_id == p.id))]

    def _photo(self, s, sources, iid):
        """The photo an item id names: the raw file named <stem>.* in one of the project's source folders, exactly one."""
        from .models import File, Photo
        stem = iid.partition('~')[2]
        if not stem: return None
        out = set()
        for root, rel, _ in sources:
            for f in s.scalars(select(File).where(File.root == root, File.role == 'raw', File.relpath.like(f'{rel}/{stem}.%'))):
                if os.path.splitext(os.path.basename(f.relpath))[0] == stem and os.path.dirname(f.relpath) == rel and f.photo_id: out.add(f.photo_id)
        return s.get(Photo, next(iter(out))) if len(out) == 1 else None

    def read_state(self, pid, folder):
        from .models import File, ItemChoice, Photo, Project
        from stratastills import projects as P
        out = self.files.read_state(pid, folder)
        with self._session() as s:
            p = s.scalars(select(Project).where(Project.slug == pid)).first()
            if p is None: return out
            sources = self._sources(s, p)
            for c in s.scalars(select(ItemChoice).where(ItemChoice.project_id == p.id)):
                ph = s.get(Photo, c.photo_id); f = s.scalars(select(File).where(File.photo_id == ph.id, File.role == 'raw')).first()
                src = next(((r, rel, orig) for r, rel, orig in sources if f and f.root == r and os.path.dirname(f.relpath) == rel), None)
                if src is None: continue
                out.setdefault(P.item_id(P.source_id(src[2] or src[1]), ph.stem), {})['primary'] = c.primary_variant               # the id the app asks for on this machine
        return out

    def write_state(self, pid, folder, d):
        from .models import ItemChoice, Project
        with self._session() as s:
            p = s.scalars(select(Project).where(Project.slug == pid)).first()
            if p is not None:
                sources = self._sources(s, p); have = {c.photo_id: c for c in s.scalars(select(ItemChoice).where(ItemChoice.project_id == p.id))}
                for iid, v in d.items():
                    if not isinstance(v, dict) or not v.get('primary'): continue
                    ph = self._photo(s, sources, iid)
                    if ph is None: continue                                                                                          # no photo for this item (yet): the file has it
                    if ph.id in have: have[ph.id].primary_variant = v['primary']
                    else: s.add(ItemChoice(project_id=p.id, photo_id=ph.id, primary_variant=v['primary']))
                s.commit()
        self.files.write_state(pid, folder, d)
