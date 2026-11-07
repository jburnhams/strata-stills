"""The arena's store on schema `photos` (see arena_store.py for the contract it shares with FileStore)."""
import threading, time

from stratastills import arena as AR


class DbStore:
    kind = 'database'

    def __init__(self, url):
        from omarchy_db import connect
        self.engine = connect(url); self._lock = threading.Lock()

    def _session(self):
        from omarchy_db import session
        return session(self.engine)

    def pool(self):
        from sqlalchemy import select
        from .models import ArenaEntry
        out = {}
        with self._session() as s:
            for e in s.scalars(select(ArenaEntry).where(ArenaEntry.superseded_at.is_(None)).order_by(ArenaEntry.id)): out.setdefault(e.pic, {})[e.variant] = {k: v for k, v in dict(file=e.file, hash=e.hash, key=e.key).items() if v is not None}      # the newest entry of a (pic, variant) wins
        return out

    def retired(self):
        from sqlalchemy import select
        from .models import ArenaVariant
        with self._session() as s: return set(s.scalars(select(ArenaVariant.name).where(ArenaVariant.retired.is_(True))))

    def votes(self):
        from sqlalchemy import select
        from .models import ArenaVote
        with self._session() as s:
            return [dict(pic=v.pic, a=v.a, b=v.b, winner=v.winner, t=v.voted_at, ms=v.ms) for v in s.scalars(select(ArenaVote).where(ArenaVote.undone_at.is_(None)).order_by(ArenaVote.id))]

    def alert_events(self):
        from sqlalchemy import select
        from .models import ArenaFlagEvent
        with self._session() as s: return [e.raw for e in s.scalars(select(ArenaFlagEvent).order_by(ArenaFlagEvent.id)) if e.raw]

    def _ensure(self, s, model, key, **kw):
        if s.get(model, key) is None: s.add(model(**{('name' if hasattr(model, 'name') else 'id'): key}, **kw)); s.flush()

    def add_vote(self, vote):
        from .models import ArenaPic, ArenaVariant, ArenaVote
        with self._lock, self._session() as s:
            self._ensure(s, ArenaPic, vote['pic'])
            for n in (vote['a'], vote['b']): self._ensure(s, ArenaVariant, n)
            s.add(ArenaVote(pic=vote['pic'], a=vote['a'], b=vote['b'], winner=vote['winner'], voted_at=float(vote['t']), ms=vote.get('ms'), raw=vote)); s.commit()

    def add_alert_event(self, ev):
        from .models import ArenaFlag, ArenaFlagEvent, ArenaPic, ArenaVariant
        with self._lock, self._session() as s:
            if 'pic' in ev:                                                                                  # a new flag
                self._ensure(s, ArenaPic, ev['pic']); self._ensure(s, ArenaVariant, ev['variant'])
                s.add(ArenaFlag(id=ev['id'], pic=ev['pic'], variant=ev['variant'], status='open', note=ev.get('note') or '', other=ev.get('other'), side=ev.get('side'), token=ev.get('token'), votes_on_pic=ev.get('votes_on_pic'), votes_on_variant=ev.get('votes_on_variant'), ms=ev.get('ms'), flagged_at=ev.get('t')))
            else:
                f = s.get(ArenaFlag, ev.get('id'))
                if f is not None:                                                                            # the flag's current state follows the change (the same rules as arena.alerts)
                    if 'note' in ev: f.note = str(ev['note'])[:500]
                    if ev.get('status') in AR.STATUSES: f.status = ev['status']
                    elif 'resolved' in ev: f.status = 'fixed' if ev['resolved'] else 'open'
            s.flush(); s.add(ArenaFlagEvent(flag_id=ev['id'], at=float(ev.get('t', time.time())), status=ev.get('status') or (('fixed' if ev['resolved'] else 'open') if 'resolved' in ev else 'open' if 'pic' in ev else None), note=ev.get('note'), raw=ev)); s.commit()

    def undo(self):
        from datetime import datetime, timezone
        from sqlalchemy import func, select
        from .models import ArenaVote
        with self._lock, self._session() as s:
            v = s.scalars(select(ArenaVote).where(ArenaVote.undone_at.is_(None)).order_by(ArenaVote.id.desc()).limit(1)).first()
            if v is None: return None
            v.undone_at = datetime.now(timezone.utc).replace(tzinfo=None); gone = dict(pic=v.pic, a=v.a, b=v.b, winner=v.winner, t=v.voted_at, ms=v.ms); s.commit()
            return gone, s.scalar(select(func.count()).select_from(ArenaVote).where(ArenaVote.undone_at.is_(None)))
