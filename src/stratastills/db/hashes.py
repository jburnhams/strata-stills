"""Raw files' SHA-256 into schema `photos`: a photo is identified by the hash of its original (`Photo.sha256`), the places it was seen are `File` rows (root label, root-relative path, size, mtime_ns, role 'raw').

Two steps like the arena importer: `plan()` reads records (path, sha256, size, mtime_ns; the JSONL of the hashing job, or the batch `hash_new` just made) and needs no database; `write()` puts the plan into the session's transaction, skips what is there, verifies and commits (rollback + HashMismatch if a planned row is missing afterwards).
Rules: a file is identified by (root, relpath, size, mtime_ns), so a rerun adds nothing. A path whose bytes changed (another size/mtime/hash) is a NEW file row and the older rows of that path become `current` = False; they stay. Two paths with the same hash are one photo with two file rows and are reported as duplicates. The root label is the folder name of the source root that holds the path (`~/raw` -> 'raw')."""

import json
import os
from datetime import UTC, datetime

from sqlalchemy import func, select

from .models import File, Photo

CHUNK = 500


class HashMismatch(Exception):
    pass


def root_labels(roots):
    """[(label, abs root)] longest root first; a label is the folder's name and must be unique among the roots."""
    out = [(os.path.basename(os.path.normpath(r)) or r, os.path.normpath(os.path.expanduser(r))) for r in roots]
    names = [n for n, _ in out]
    if len(set(names)) != len(names):
        raise ValueError(f"source roots share a folder name: {names}")
    return sorted(out, key=lambda x: -len(x[1]))


def read_jsonl(path, problems):
    """Records of a hashing job's JSONL; a line that does not parse (the job may be mid-write) or lacks a field is a problem and skipped; an unreadable file is FATAL."""
    recs = []
    try:
        f = open(path, encoding="utf8")
    except OSError as e:
        problems.append(f"FATAL unreadable {path}: {e}")
        return recs
    with f:
        for i, ln in enumerate(f, 1):
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
                recs.append(dict(path=r["path"], sha256=r["sha256"], size=int(r["size"]), mtime_ns=int(r["mtime_ns"])))
            except (ValueError, KeyError, TypeError):
                problems.append(f"line {i}: unparsable or incomplete, skipped")
    return recs


def plan(records, roots, problems=None):
    """{files: [row dicts], photos: {sha256: stem}, duplicates: {sha256: [paths]}, problems, counts}. Reads only."""
    problems = list(problems or [])
    labels = root_labels(roots)
    files, photos, paths, seen = [], {}, {}, {}  # seen: key -> sha256 of the first record with that key
    for r in records:
        sha = str(r["sha256"]).lower()
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            problems.append(f"{r['path']}: not a SHA-256")
            continue
        ap = os.path.normpath(r["path"])
        hit = next(
            ((n, os.path.relpath(ap, root)) for n, root in labels if ap == root or ap.startswith(root + os.sep)), None
        )
        if hit is None:
            problems.append(f"{r['path']}: under none of the source roots")
            continue
        root, rel = hit[0], hit[1].replace(os.sep, "/")
        key = (root, rel, r["size"], r["mtime_ns"])
        if key in seen:
            if seen[key] != sha:
                problems.append(f"{r['path']}: two different hashes for the same size and mtime, first one kept")
            continue
        seen[key] = sha
        files.append(dict(root=root, relpath=rel, size=r["size"], mtime_ns=r["mtime_ns"], sha256=sha, role="raw"))
        photos.setdefault(sha, os.path.splitext(os.path.basename(rel))[0])
        paths.setdefault(sha, set()).add((root, rel))
    dups = {s: sorted(f"{a}:{b}" for a, b in ps) for s, ps in paths.items() if len(ps) > 1}
    counts = dict(
        files=len(files),
        photos=len(photos),
        duplicate_hashes=len(dups),
        duplicate_files=sum(len(v) for v in dups.values()),
        roots=sorted({f["root"] for f in files}),
    )
    return dict(files=files, photos=photos, duplicates=dups, problems=problems, counts=counts)


def _chunks(xs, n=CHUNK):
    xs = list(xs)
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def write(session, p):
    """Write a plan in the session's transaction, then verify and commit. Returns {photo, file, superseded}: rows added / older rows of a changed path marked not current."""
    fatal = [x for x in p["problems"] if x.startswith("FATAL")]
    if fatal:
        raise HashMismatch("refusing to write a plan with an unreadable source: " + "; ".join(fatal))
    have_photo = {}
    for c in _chunks(p["photos"]):
        have_photo.update({s: i for s, i in session.execute(select(Photo.sha256, Photo.id).where(Photo.sha256.in_(c)))})
    new_photos = [Photo(sha256=s, stem=stem, kind="still") for s, stem in p["photos"].items() if s not in have_photo]
    session.add_all(new_photos)
    session.flush()
    ids = dict(have_photo)
    ids.update({x.sha256: x.id for x in new_photos})
    have_files = set()
    by_path = {}
    for c in _chunks({f["relpath"] for f in p["files"]}):
        for f in session.scalars(select(File).where(File.relpath.in_(c))):
            have_files.add((f.root, f.relpath, f.size, f.mtime_ns))
            by_path.setdefault((f.root, f.relpath), []).append(f)
    new_files, now = [], datetime.now(UTC).replace(tzinfo=None)
    for f in p["files"]:
        if (f["root"], f["relpath"], f["size"], f["mtime_ns"]) in have_files:
            continue
        new_files.append(
            File(
                root=f["root"],
                relpath=f["relpath"],
                size=f["size"],
                mtime_ns=f["mtime_ns"],
                photo_id=ids[f["sha256"]],
                role="raw",
                sha256=f["sha256"],
                current=True,
                last_seen=now,
            )
        )
    session.add_all(new_files)
    session.flush()
    superseded = 0  # per path exactly one version is current: the newest (mtime_ns, then id) across the rows already there AND the new ones, the others stay as history
    added = {}  # new rows grouped by path once (a scan of new_files per path is quadratic: 7 minutes for 21k files)
    for f in new_files:
        added.setdefault((f.root, f.relpath), []).append(f)
    for k, mine in added.items():
        rows = by_path.get(k, []) + mine
        keep = max(rows, key=lambda x: (x.mtime_ns, x.id or 0))
        for f in rows:
            if f is keep:
                f.current = True
            elif f.current:
                f.current = False
                superseded += 1
    session.flush()
    bad = []
    for c in _chunks(p["photos"]):
        got = {s for (s,) in session.execute(select(Photo.sha256).where(Photo.sha256.in_(c)))}
        if got != set(c):
            bad.append(f"photo: {len(set(c) - got)} planned hash(es) missing after the write")
    keys = {}  # relpath -> {(root, relpath, size, mtime_ns): sha256}; each chunk checks only its own paths
    for f in p["files"]:
        keys.setdefault(f["relpath"], {})[(f["root"], f["relpath"], f["size"], f["mtime_ns"])] = f["sha256"]
    for c in _chunks(keys):
        got = {
            (f.root, f.relpath, f.size, f.mtime_ns): f.sha256
            for f in session.scalars(select(File).where(File.relpath.in_(c)))
        }
        for rp in c:
            for k, sha in keys[rp].items():
                if got.get(k) != sha:
                    bad.append(f"file {k[0]}:{k[1]}: row missing or hash differs after the write")
    n = session.scalar(select(func.count()).select_from(Photo))
    if n < len(p["photos"]):
        bad.append(f"photo: {n} rows for {len(p['photos'])} planned")
    if bad:
        session.rollback()
        raise HashMismatch("; ".join(bad[:20]))
    session.commit()
    return dict(photo=len(new_photos), file=len(new_files), superseded=superseded)


def keep_newest_current(rows):
    """Of the versions of ONE path (File objects, old and new) exactly one stays current: the newest (mtime_ns, then id); the others are history. Returns how many were switched off."""
    keep = max(rows, key=lambda x: (x.mtime_ns, x.id or 0))
    off = 0
    for f in rows:
        if f is keep:
            f.current = True
        elif f.current:
            f.current = False
            off += 1
    return off
