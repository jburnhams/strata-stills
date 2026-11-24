"""Tag the archive's photos through strata-models (stratastills.tagging): every photo with a raw and no `vision_tags` row of this version yet.
Writes photos.analysis (needs STRATA_DB_LIVE=1 STRATA_DB_DATABASE=omarchy and the strata-models service running). Resumable: rerun to go on.

    .venv/bin/python scripts/tag_photos.py [--limit N] [--photos 12 34 ...] [--batch 8] [--dry-run]

Photos the nudity filter flags are skipped and never sent. The caption call is a Claude API call (about $0.0004 a photo, from the strata-models
Claude budget); the fields call runs on the GPU queue at batch priority."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import strata_models_client as sm
from sqlalchemy import select
from strata_python.db import connect, session

from stratastills import browser, nsfw, settings, tagging
from stratastills.db.models import Analysis, File, PhotoMeta

APP = "strata-stills-tags"


def todo(s, limit, only):
    done = set(
        s.scalars(
            select(Analysis.photo_id).where(
                Analysis.kind == tagging.KIND,
                Analysis.producer == tagging.PRODUCER,
                Analysis.producer_version == tagging.VERSION,
            )
        )
    )
    roots = dict(browser.roots())
    q = (
        select(File.photo_id, File.root, File.relpath)
        .where(File.role == "raw", File.current.is_(True), File.photo_id.is_not(None))
        .order_by(File.photo_id)
    )
    seen, out = set(), []
    for pid, root, rel in s.execute(q):
        if pid in done or pid in seen or (only and pid not in only) or root not in roots:
            continue
        p = os.path.join(roots[root], rel)
        if os.path.exists(p):
            seen.add(pid)
            out.append((pid, p))
        if limit and len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--photos", type=int, nargs="*")
    ap.add_argument("--batch", type=int, default=8, help="photos in flight at once")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not sm.available():
        sys.exit(f"strata-models is not running at {sm.url()}")
    folder = os.path.join(settings.home(), "previews", str(tagging.PX))
    url = settings.database_url()
    if not url:
        sys.exit(
            "no database: set database_url in the settings (STRATA_DB_LIVE=1 STRATA_DB_DATABASE=omarchy for the live one)"
        )
    eng = connect(url)
    with session(eng) as s:
        work = todo(s, a.limit, set(a.photos or []))
        metas = {
            m.photo_id: m for m in s.scalars(select(PhotoMeta).where(PhotoMeta.photo_id.in_([p for p, _ in work])))
        }
    print(f"{len(work)} photos to tag", flush=True)
    if a.dry_run:
        return
    ok = skipped = failed = 0
    t0 = time.time()
    for i in range(0, len(work), a.batch):
        flight = []
        for pid, raw in work[i : i + a.batch]:
            try:
                if (
                    nsfw.score(raw)[0] >= nsfw.threshold() and nsfw._key(raw) not in nsfw._allowed()
                ):  # checked even with the filter off: the caption call leaves this machine
                    skipped += 1
                    continue
                img, key = tagging.preview(raw, folder)
                m = metas.get(pid)
                ctx = tagging.context(
                    {
                        c: getattr(m, c)
                        for c in (
                            "taken_at",
                            "model",
                            "lens",
                            "iso",
                            "exposure_s",
                            "f_number",
                            "focal_mm",
                            "gps_lat",
                            "gps_lon",
                        )
                    }
                    if m
                    else None
                )
                flight.append((pid, key, tagging.submit(sm, img, ctx, APP)))
            except (tagging.TagError, OSError) as e:
                failed += 1
                print(f"  p{pid}: {e}", flush=True)
        rows = []
        for pid, key, jobs in flight:
            try:
                rows.append(Analysis(**tagging.row(pid, key, tagging.collect(sm, jobs))))
            except tagging.TagError as e:
                failed += 1
                print(f"  p{pid}: {e}", flush=True)
        with session(eng) as s:
            s.add_all(rows)
            s.commit()
        ok += len(rows)
        print(
            f"{min(i + a.batch, len(work))}/{len(work)}: tagged {ok}, skipped (nudity) {skipped}, failed {failed}, {time.time() - t0:.0f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
