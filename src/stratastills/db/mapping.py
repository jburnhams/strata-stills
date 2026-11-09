"""Match the arena's picture names ('2023-01-11_DSC00010', a Lightroom export's folder date + the shot's stem) to the photo whose raw was hashed, and fill `arena_pic.photo_id`.

Rules, in this order; the first that gives exactly ONE raw wins, anything else stays unmapped and is listed (never guessed):
1. exact: the raw sits in a folder `<root>/<year>/<date>[ title]/<stem>.<ext>` whose date and stem are the name's;
2. catalogue: the Lightroom catalogue pairs an export in a folder `<date> - title` with a raw; the arena names that pair `<folder date>_<the RAW's stem>` (scripts/build_arena.py::order), and the raw was hashed;
3. window: exactly one raw with that stem has a folder date within 3 days of the name's (a capture date that is not the import folder's).
Two raws for one name at a step = ambiguous (reported, not mapped). `plan()` needs no database; `write()` fills only NULL foreign keys of existing rows in one transaction, verifies, and never overwrites a link that is already set (a different one is reported as a conflict)."""

import datetime as D
import os
import posixpath
import re
from collections import defaultdict

from sqlalchemy import select

from .hashes import _chunks
from .models import ArenaPic, Photo

WINDOW_DAYS = 3
NAME = re.compile(r"^(\d{4}-\d{2}-\d{2})_(.+)$")


class MappingMismatch(Exception):
    pass


def arena_name(final, raw):
    """The name the arena gives a catalogue pair (scripts/build_arena.py::order): the first 10 characters of the export's folder name + the RAW's stem (not the export's: 'DSC08525-Edit.jpg' still names '..._DSC08525'), spaces removed. Separators of either kind (the catalogue is built on Windows too)."""
    if not final or not raw:
        return None
    folder = posixpath.basename(posixpath.dirname(final.replace("\\", "/")))
    stem = posixpath.splitext(posixpath.basename(raw.replace("\\", "/")))[0]
    return f"{folder[:10]}_{stem}".replace(" ", "") if NAME.match(f"{folder[:10]}_{stem}") else None


def build_index(records, catalogue, roots):
    """{sha: {raw path: sha256}, by_name, by_final, by_stem} from the hashed raws (records: dicts with path, sha256) and the catalogue rows (dicts with raw, final)."""
    hashed = {r["path"]: r["sha256"] for r in records}
    by_name, by_stem, by_final = defaultdict(set), defaultdict(list), defaultdict(set)
    for path in hashed:
        for root in roots:
            root = os.path.normpath(os.path.expanduser(root))
            if path.startswith(root + os.sep):
                parts = os.path.relpath(path, root).replace(os.sep, "/").split("/")
                if len(parts) >= 3 and re.match(r"\d{4}-\d{2}-\d{2}", parts[1]):
                    stem = os.path.splitext(parts[-1])[0]
                    by_name[f"{parts[1][:10]}_{stem}"].add(path)
                    try:
                        by_stem[stem].append((D.date.fromisoformat(parts[1][:10]), path))
                    except ValueError:
                        pass
                break
    for r in catalogue:
        key = arena_name(r.get("final"), r.get("raw"))
        if key and r.get("raw") in hashed:
            by_final[key].add(r["raw"])
    return dict(hashed=hashed, by_name=by_name, by_stem=by_stem, by_final=by_final)


def match(name, idx):
    """(tier, raw path) for a name; tier is 'exact' | 'catalogue' | 'window' | 'ambiguous' | 'none' (path None for the last two)."""
    m = NAME.match(name)
    if not m:
        return "none", None
    amb = False
    for tier, cands in (
        ("exact", idx["by_name"].get(name, set())),
        ("catalogue", idx["by_final"].get(name, set())),
        ("window", None),
    ):
        if cands is None:
            try:
                d = D.date.fromisoformat(m.group(1))
            except ValueError:
                break
            cands = {p for x, p in idx["by_stem"].get(m.group(2), []) if abs((x - d).days) <= WINDOW_DAYS}
        if len(cands) == 1:
            return tier, next(iter(cands))
        if len(cands) > 1:
            amb = True
    return ("ambiguous" if amb else "none"), None


def plan(names, records, catalogue, roots):
    """{rows: [{name, sha256, tier, raw}], unmapped: [{name, why}], counts}. Reads only."""
    idx = build_index(records, catalogue, roots)
    rows, unmapped, counts = [], [], defaultdict(int)
    for n in dict.fromkeys(names):
        tier, raw = match(n, idx)
        counts[tier] += 1
        if raw:
            rows.append(dict(name=n, sha256=idx["hashed"][raw], tier=tier, raw=raw))
        else:
            unmapped.append(dict(name=n, why=tier))
    return dict(
        rows=rows,
        unmapped=unmapped,
        counts=dict(counts, mapped=len(rows), unmapped=len(unmapped), names=len(rows) + len(unmapped)),
    )


def write(session, p):
    """Fill NULL arena_pic.photo_id for the planned names; returns {linked, already, missing_pic, conflicts}. Rolls back and raises MappingMismatch if a link is not what was planned afterwards, or a hash has no photo."""
    planned = {r["name"]: r["sha256"] for r in p["rows"]}
    ids = {}
    for c in _chunks(set(planned.values())):
        ids.update({s: i for s, i in session.execute(select(Photo.sha256, Photo.id).where(Photo.sha256.in_(c)))})
    absent = sorted(n for n, s in planned.items() if s not in ids)
    if absent:
        session.rollback()
        raise MappingMismatch(
            f"{len(absent)} planned hash(es) have no photo row, e.g. {absent[:3]}; load the hashes first"
        )
    linked = already = 0
    conflicts = set()
    pics = {}
    for c in _chunks(planned):
        pics.update({x.name: x for x in session.scalars(select(ArenaPic).where(ArenaPic.name.in_(c)))})
    for n, s in planned.items():
        pic = pics.get(n)
        if pic is None:
            continue
        if pic.photo_id is None:
            pic.photo_id = ids[s]
            linked += 1
        elif pic.photo_id == ids[s]:
            already += 1
        else:
            conflicts.add(n)
    session.flush()
    bad = []
    for n, s in planned.items():
        pic = pics.get(n)
        if pic is not None and n not in conflicts and pic.photo_id != ids[s]:
            bad.append(n)
    if bad:
        session.rollback()
        raise MappingMismatch(f"{len(bad)} link(s) differ from the plan after the write, e.g. {bad[:3]}")
    session.commit()
    return dict(
        linked=linked, already=already, missing_pic=sum(1 for n in planned if n not in pics), conflicts=len(conflicts)
    )
