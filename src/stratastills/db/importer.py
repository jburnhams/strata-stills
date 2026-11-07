"""One-way, repeatable import of the arena's files into schema `photos`: votes.jsonl, alerts.jsonl (flags and their changes), pool.json, variants.json and the report flags (`out/flags/<report>.json`). The files are only read; nothing in them is changed, moved or deleted, and the arena keeps using them until reads move over (docs/progress.md).

Two steps so that a run can be looked at first: `plan()` reads and parses the files and needs no database (counts, problems, the rows to write); `write()` puts the plan into a session in ONE transaction, skips what is already there (`legacy_key`, primary keys), then compares row counts and a checksum of the keys with the plan and rolls everything back on a mismatch. Run it twice: the second run writes nothing.
Keys: a line's `legacy_key` is the SHA-1 of its canonical JSON (the Nth identical line gets `N:` in front), so the same line is the same row on every run. Arena pictures are named as the pool names them ('2023-01-11_DSC00010'); matching a name to a photo (SHA-256 of its raw) is a later step (`ArenaPic.photo_id` stays null here)."""
import hashlib, json, os
from datetime import datetime

from sqlalchemy import func, select

from stratastills import arena as AR
from .models import ArenaEntry, ArenaFlag, ArenaFlagEvent, ArenaPic, ArenaVariant, ArenaVote, ReportFlag

REPORTS = ('neutral', 'auto')


def _read_json(path):
    try:
        with open(path, encoding='utf8') as f: return json.load(f)
    except (OSError, ValueError): return None


def _keys(lines):
    """legacy keys for lines: SHA-1 of the canonical JSON, the Nth identical line prefixed with 'N:' so repeats stay distinct."""
    seen, out = {}, []
    for ln in lines:
        c = json.dumps(ln, sort_keys=True); n = seen.get(c, 0); seen[c] = n + 1
        out.append(hashlib.sha1((f'{n}:' if n else '') .encode() + c.encode()).hexdigest())
    return out


def _time(s):
    try: return datetime.fromisoformat(s) if s else None
    except ValueError: return None


def plan(arena_dir, flags_dir=None):
    """Parse the files: {votes, flags, flag_events, pics, variants, entries, report_flags: row dicts, problems: [...], counts}. Reads only."""
    problems = []; votes_f = os.path.join(arena_dir, 'votes.jsonl'); alerts_f = os.path.join(arena_dir, 'alerts.jsonl')
    raw_votes = AR.load_votes(votes_f); events = AR.load_votes(alerts_f)
    for name, path in (('votes.jsonl', votes_f), ('alerts.jsonl', alerts_f)):
        if os.path.exists(path):
            with open(path, encoding='utf8') as f: n = sum(1 for l in f if l.strip())
            if n != len(AR.load_votes(path)): problems.append(f'{name}: {n - len(AR.load_votes(path))} unparsable line(s) skipped')
    vkeys = _keys(raw_votes); votes = []
    for v, k in zip(raw_votes, vkeys):
        if not all(x in v for x in ('pic', 'a', 'b', 'winner', 't')): problems.append(f'vote without pic/a/b/winner/t: {v}'); continue
        votes.append(dict(pic=v['pic'], a=v['a'], b=v['b'], winner=v['winner'], voted_at=float(v['t']), ms=v.get('ms'), legacy_key=k, raw=v))
    ekeys = _keys(events); known = {e['id'] for e in events if 'pic' in e}; final = {a['id']: a for a in AR.alerts(events)}
    flags = [dict(id=a['id'], pic=a['pic'], variant=a['variant'], status=a['status'], note=a.get('note') or '', other=a.get('other'), side=a.get('side'), token=a.get('token'), votes_on_pic=a.get('votes_on_pic'), votes_on_variant=a.get('votes_on_variant'), ms=a.get('ms'), flagged_at=a.get('t')) for a in final.values()]
    flag_events = []
    for e, k in zip(events, ekeys):
        fid = e.get('id')
        if fid not in known: problems.append(f'alert change for an unknown flag: {e}'); continue
        status = e.get('status') or (('fixed' if e['resolved'] else 'open') if 'resolved' in e else 'open' if 'pic' in e else None)
        flag_events.append(dict(flag_id=fid, at=float(e.get('t', 0)), status=status, note=e.get('note'), legacy_key=k, raw=e))
    pool = _read_json(os.path.join(arena_dir, 'pool.json')) or {}; var = (_read_json(os.path.join(arena_dir, 'variants.json')) or {}).get('retired', [])
    entries = [dict(pic=p, variant=n, file=e['file'], hash=e['hash'], key=e.get('key')) for p, vs in sorted(pool.items()) for n, e in sorted(vs.items())]
    pics = sorted({v['pic'] for v in votes} | {f['pic'] for f in flags} | set(pool)); variants = sorted({e['variant'] for e in entries} | {x for v in votes for x in (v['a'], v['b'])} | {f['variant'] for f in flags} | set(var))
    report_flags = []
    for rep in REPORTS:
        d = _read_json(os.path.join(flags_dir, f'{rep}.json')) if flags_dir else None
        for pic, f in sorted((d or {}).items()): report_flags.append(dict(report=rep, pic=pic, rating=f.get('rating') or '', reasons=f.get('reasons') or [], note=f.get('note') or '', status=f.get('status') or 'open', first_at=_time(f.get('first')), at=_time(f.get('at'))))
    pics = sorted(set(pics) | {r['pic'] for r in report_flags})
    counts = dict(arena_pic=len(pics), arena_variant=len(variants), arena_entry=len(entries), arena_vote=len(votes), arena_flag=len(flags), arena_flag_event=len(flag_events), report_flag=len(report_flags))
    return dict(votes=votes, flags=flags, flag_events=flag_events, pics=pics, variants=variants, retired=set(var), entries=entries, report_flags=report_flags, problems=problems, counts=counts)


def _checksum(keys): return hashlib.sha1('\n'.join(sorted(keys)).encode()).hexdigest()


def write(session, p):
    """Write a plan in the session's transaction, skipping rows that exist, then verify: every planned vote/flag-event key and flag/pic/variant/report-flag key is in the database, and (for the tables this import owns) the counts match. Raises ImportMismatch (nothing committed, the session rolled back) otherwise; commits and returns {table: rows added} when it matches."""
    added = {}
    def have(model, col): return {r for (r,) in session.execute(select(col))}
    def add(name, rows): added[name] = len(rows); session.add_all(rows)
    have_pics = have(ArenaPic, ArenaPic.name); add('arena_pic', [ArenaPic(name=n) for n in p['pics'] if n not in have_pics]); session.flush()
    have_var = have(ArenaVariant, ArenaVariant.name); add('arena_variant', [ArenaVariant(name=n, retired=n in p['retired']) for n in p['variants'] if n not in have_var])
    for v in session.scalars(select(ArenaVariant).where(ArenaVariant.name.in_(list(p['retired'])))): v.retired = True
    have_e = {(a, b, c, d) for a, b, c, d in session.execute(select(ArenaEntry.pic, ArenaEntry.variant, ArenaEntry.hash, ArenaEntry.file))}
    session.flush(); add('arena_entry', [ArenaEntry(pic=e['pic'], variant=e['variant'], file=e['file'], hash=e['hash'], key=e['key']) for e in p['entries'] if (e['pic'], e['variant'], e['hash'], e['file']) not in have_e])
    have_v = have(ArenaVote, ArenaVote.legacy_key); add('arena_vote', [ArenaVote(**v) for v in p['votes'] if v['legacy_key'] not in have_v])
    have_f = have(ArenaFlag, ArenaFlag.id); add('arena_flag', [ArenaFlag(**f) for f in p['flags'] if f['id'] not in have_f]); session.flush()
    have_fe = have(ArenaFlagEvent, ArenaFlagEvent.legacy_key); add('arena_flag_event', [ArenaFlagEvent(**e) for e in p['flag_events'] if e['legacy_key'] not in have_fe])
    have_r = {(a, b) for a, b in session.execute(select(ReportFlag.report, ReportFlag.pic))}; add('report_flag', [ReportFlag(**r) for r in p['report_flags'] if (r['report'], r['pic']) not in have_r]); session.flush()
    bad = []
    for name, model, col, want in (('arena_vote', ArenaVote, ArenaVote.legacy_key, [v['legacy_key'] for v in p['votes']]), ('arena_flag_event', ArenaFlagEvent, ArenaFlagEvent.legacy_key, [e['legacy_key'] for e in p['flag_events']]), ('arena_flag', ArenaFlag, ArenaFlag.id, [f['id'] for f in p['flags']]), ('arena_pic', ArenaPic, ArenaPic.name, p['pics'])):
        got = have(model, col)
        if not set(want) <= got: bad.append(f'{name}: {len(set(want) - got)} planned row(s) missing after the write')
        elif name in ('arena_vote', 'arena_flag_event') and _checksum(want) != _checksum(k for k in got if k in set(want)): bad.append(f'{name}: checksum differs')
    n_votes = session.scalar(select(func.count()).select_from(ArenaVote))
    if n_votes < len(p['votes']): bad.append(f'arena_vote: {n_votes} rows for {len(p["votes"])} planned')
    if bad: session.rollback(); raise ImportMismatch('; '.join(bad))
    session.commit(); return added


class ImportMismatch(Exception): pass
