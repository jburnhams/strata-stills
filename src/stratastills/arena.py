"""The rating arena: blind pairwise choices between versions of the same picture (the person's Lightroom final and the automatic versions, auto1, auto2 ...), to find out by data which automatic version people prefer, and to compare any future one the same way (add a variant, make its pictures, vote).

Pure logic (no server, no images): pair choice, Elo and Bradley-Terry ratings. The server (server/app.py) holds the pool and the votes (`out/arena/`), scripts/build_arena.py makes the pictures.

Pool: {picture: {variant: {'file': relative path, 'hash': str}}}; a picture is offered only with the variants it has, and two variants whose pictures are identical (same hash) are never paired (a picture with no face gets the same pixels from two variants that differ only for faces).
Vote: {'pic', 'a', 'b' (variants as shown left and right), 'winner': 'a' | 'b' | 'tie', 't' (unix time), 'ms' (time to decide)}."""
import json, math, os, random

START = 1500.0
K = 24.0


def load_votes(path):
    if not os.path.exists(path): return []
    out = []
    for line in open(path, encoding='utf8'):
        line = line.strip()
        if line:
            try: out.append(json.loads(line))
            except ValueError: pass                                                   # a half-written last line
    return out


def append_vote(path, vote):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf8') as f: f.write(json.dumps(vote, sort_keys=True) + '\n')


STATUSES = ('open', 'investigating', 'fixed', 'wontfix')                                           # the queue of problems with our versions: `fixed` and `wontfix` are done (the version is offered again)


def alerts(events, pool=None):
    """The flags (a person saying 'something is wrong with this picture') from the event log, oldest first: [{id, pic, variant, other, side, note, status, t, ms, resolved, pool: the pool entry of that variant, votes_on_pic}]. A flag is just the press of a button; the reason (`note`) and `status` (STATUSES) are set afterwards on the alerts page. An event is a flag ({'id', 'pic', 'variant', ...}) or a change ({'id'} plus any of 'note', 'status', or the older 'resolved': bool); later ones win. `resolved` is derived from the status. `open_alerts` names the (picture, variant) pairs that must not be offered."""
    out = {}
    for e in events:
        if 'pic' in e: out[e['id']] = dict(e, status=e.get('status') or ('fixed' if e.get('resolved') else 'open'))
        elif e.get('id') in out:
            a = out[e['id']]
            if 'note' in e: a['note'] = str(e['note'])[:500]
            if e.get('status') in STATUSES: a['status'] = e['status']
            elif 'resolved' in e: a['status'] = 'fixed' if e['resolved'] else 'open'
    for a in out.values(): a['resolved'] = a['status'] in ('fixed', 'wontfix'); a['pool'] = ((pool or {}).get(a['pic']) or {}).get(a['variant'])
    return sorted(out.values(), key=lambda a: a.get('t', 0))


def open_alerts(events): return {(a['pic'], a['variant']) for a in alerts(events) if not a['resolved']}


def score(v): return {'a': 1.0, 'b': 0.0, 'tie': 0.5}.get(v.get('winner'))


def elo(votes, retired=()):
    """Elo ratings by playing the votes in time order (K = 24); retired variants' votes are skipped. -> {variant: rating}."""
    r = {}
    for v in sorted(votes, key=lambda v: v.get('t', 0)):
        s = score(v)
        if s is None or v['a'] in retired or v['b'] in retired: continue
        ra, rb = r.setdefault(v['a'], START), r.setdefault(v['b'], START); ea = 1 / (1 + 10 ** ((rb - ra) / 400)); r[v['a']] = ra + K * (s - ea); r[v['b']] = rb + K * ((1 - s) - (1 - ea))
    return r


def bradley_terry(votes, retired=(), iters=200):
    """Order-independent strengths (Bradley-Terry, ties as half a win each way) on the Elo scale, the mean of the variants 1500. -> {variant: rating}."""
    wins, games, names = {}, {}, set()
    for v in votes:
        s = score(v)
        if s is None or v['a'] in retired or v['b'] in retired: continue
        a, b = v['a'], v['b']; names |= {a, b}; wins[(a, b)] = wins.get((a, b), 0) + s; wins[(b, a)] = wins.get((b, a), 0) + 1 - s; games[frozenset((a, b))] = games.get(frozenset((a, b)), 0) + 1
    if not names: return {}
    p = {n: 1.0 for n in names}
    for _ in range(iters):
        q = {}
        for i in names:
            w = sum(wins.get((i, j), 0) for j in names if j != i) + 0.01
            d = sum(games.get(frozenset((i, j)), 0) / (p[i] + p[j]) for j in names if j != i) + 1e-9
            q[i] = w / d
        g = math.exp(sum(math.log(x) for x in q.values()) / len(q)); p = {n: x / g for n, x in q.items()}
    return {n: START + 400 * math.log10(x) for n, x in p.items()}


def bootstrap(votes, retired=(), n=200, seed=0):
    """The 5th and 95th percentile of each variant's Bradley-Terry rating over resamples of the votes: how far to trust a gap. -> {variant: (lo, hi)}."""
    rng = random.Random(seed); runs = []
    for _ in range(n):
        runs.append(bradley_terry([votes[rng.randrange(len(votes))] for _ in votes], retired, iters=60)) if votes else None
    out = {}
    for name in bradley_terry(votes, retired, iters=20):
        xs = sorted(r[name] for r in runs if name in r)
        if xs: out[name] = (xs[int(0.05 * len(xs))], xs[min(len(xs) - 1, int(0.95 * len(xs)))])
    return out


def head_to_head(votes, retired=()):
    """{(a, b): [wins of a, wins of b, ties]} over unordered pairs (the names sorted)."""
    t = {}
    for v in votes:
        s = score(v)
        if s is None or v['a'] in retired or v['b'] in retired: continue
        a, b = sorted((v['a'], v['b'])); row = t.setdefault((a, b), [0, 0, 0]); first = v['a'] == a
        if s == 0.5: row[2] += 1
        elif (s == 1.0) == first: row[0] += 1
        else: row[1] += 1
    return t


def pick_pair(pool, votes, rng, retired=(), want=None, blocked=()):
    """Choose what to show next: a picture and two of its variants (-> (picture, left, right), left and right already shuffled), or None when nothing can be offered. Pictures with the fewest votes come first (spreading the votes over the whole pool), then among a picture's pairs the one whose variants have been compared the least in total, ties broken at random; `want` (a set of variant names) limits pairs to those containing one of them (a new variant being tested)."""
    seen = {}; pool = {pic: {n: e for n, e in vs.items() if (pic, n) not in blocked} for pic, vs in pool.items()}                  # a variant with an open alert is not offered
    for v in votes: seen[v['pic']] = seen.get(v['pic'], 0) + 1
    pair_n = {}
    for v in votes: k = frozenset((v['a'], v['b'])); pair_n[k] = pair_n.get(k, 0) + 1
    pics = []
    for pic, vs in pool.items():
        names = [n for n in vs if n not in retired]; pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if vs[a]['hash'] != vs[b]['hash'] and (want is None or a in want or b in want)]
        if pairs: pics.append((pic, pairs))
    if not pics: return None
    low = min(seen.get(p, 0) for p, _ in pics); cand = [(p, ps) for p, ps in pics if seen.get(p, 0) == low]; pic, pairs = cand[rng.randrange(len(cand))]
    best = min(pair_n.get(frozenset(p), 0) for p in pairs); a, b = rng.choice([p for p in pairs if pair_n.get(frozenset(p), 0) == best])
    return (pic, a, b) if rng.random() < 0.5 else (pic, b, a)
