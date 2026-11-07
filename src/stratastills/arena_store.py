"""Where the arena keeps its pool, votes and flags. Two stores with the same methods, so server/app.py does not care which one it has:
- `FileStore`: the files of `settings.arena_dir()` (pool.json, variants.json, votes.jsonl, alerts.jsonl), as before. The default, and what the unit tests use.
- `DbStore`: schema `photos` of the shared database (src/stratastills/db), used as soon as `settings.database_url()` is set (STRATASTILLS_DATABASE_URL or `database_url` in settings.json). Votes are never deleted (undo marks `undone_at`), every flag change is a row of the flag's history, and the pool is the non-superseded entries. The picture files stay on disk under `arena_dir()`.
Both hand back what arena.py's logic already takes: pool `{pic: {variant: {file, hash, key?}}}`, votes `[{pic, a, b, winner, t, ms}]` oldest first, flag events `[{id, ...}]` oldest first. Switching a running arena from files to the database is setting the URL and restarting; the files stay where they are (frozen), and `scripts/import_to_db.py --write` brings a rebuilt pool.json across."""
import json, os

from stratastills import arena as AR, settings


class FileStore:
    kind = 'files'

    def _f(self, name): return os.path.join(settings.arena_dir(), name)

    def pool(self):
        try: return json.load(open(self._f('pool.json'), encoding='utf8'))
        except (OSError, ValueError): return {}

    def retired(self):
        try: return set(json.load(open(self._f('variants.json'), encoding='utf8')).get('retired', []))
        except (OSError, ValueError): return set()

    def votes(self): return AR.load_votes(self._f('votes.jsonl'))

    def alert_events(self): return AR.load_votes(self._f('alerts.jsonl'))

    def add_vote(self, vote): AR.append_vote(self._f('votes.jsonl'), vote)

    def add_alert_event(self, ev): AR.append_vote(self._f('alerts.jsonl'), ev)

    def undo(self):
        """Remove the newest vote -> (that vote, votes left) or None when there is none."""
        f = self._f('votes.jsonl'); lines = open(f, encoding='utf8').read().splitlines() if os.path.exists(f) else []
        if not lines: return None
        gone = lines.pop(); open(f, 'w', encoding='utf8').write(''.join(x + '\n' for x in lines)); return json.loads(gone), len(lines)


_stores = {}


def get():
    """The store to use now: the database when settings.database_url() is set, else the files. A DbStore is made once per URL."""
    url = settings.database_url()
    if not url: return FileStore()
    if url not in _stores:
        from .db.store import DbStore                                                                       # needs requirements-db.txt; only imported when a database is configured
        _stores[url] = DbStore(url)
    return _stores[url]
