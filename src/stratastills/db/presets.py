"""The saved pipeline presets on `pipeline_preset` (see presets_store.py for the contract it shares with FileStore, and for the switch: `database_url` AND 'presets' in `db_reads`). Reading: the files' presets with the database's laid over them, by name. Every write also goes to the files (dual-write); a delete removes the row and the file."""
from sqlalchemy import select

from stratastills.presets_store import FileStore


class DbStore:
    kind = 'database'

    def __init__(self, url):
        from omarchy_db import connect
        self.engine = connect(url); self.files = FileStore()

    def _session(self):
        from omarchy_db import session
        return session(self.engine)

    def list(self):
        """The files' presets with the database's laid over them, by name."""
        return self._merge({p['name']: p for p in self.files.list()})

    def list_db(self):
        """What the database alone holds (parity checks)."""
        return self._merge({})

    def _merge(self, out):
        from .models import PipelinePreset
        with self._session() as s:
            for r in s.scalars(select(PipelinePreset)): out[r.name] = dict(name=r.name, label=r.label, stages=r.stages)
        return [out[k] for k in sorted(out)]

    def save(self, name, label, stages):
        from .models import PipelinePreset
        with self._session() as s:
            r = s.get(PipelinePreset, name)
            if r is None: s.add(PipelinePreset(name=name, label=label or name, stages=stages))
            else: r.label, r.stages = label or name, stages
            s.commit()
        self.files.save(name, label, stages)

    def delete(self, name):
        from .models import PipelinePreset
        with self._session() as s:
            r = s.get(PipelinePreset, name); had = r is not None
            if had: s.delete(r); s.commit()
        return self.files.delete(name) or had
