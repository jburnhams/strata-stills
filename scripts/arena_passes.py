"""Rebuild the arena one version at a time, in the order given (the newest idea first), so each version's pictures replace the old ones as
soon as that pass ends and the served pool picks them up (import_to_db --write after each pass): votes start on the new version while the
others are still being remade.
  OMARCHY_DB_LIVE=1 python scripts/arena_passes.py [--order auto7,auto5,...] [-- build_arena arguments]
The first pass makes the neutral develops that every version starts from (cached in out/cache/report-auto-neutral, so later passes reuse
them); a pass with nothing to do ends at once. Progress goes to stdout, one line per pass; a failed pass stops the queue."""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = "auto7,auto5,auto6,auto4,auto3,auto2,auto1,neutral,camera,lightroom"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--order", default=ORDER)
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="passed on to build_arena.py (after --)")
    a = ap.parse_args()
    rest = [x for x in a.rest if x != "--"]
    for v in a.order.split(","):
        t = time.time()
        print(f"pass {v}: building", flush=True)
        r = subprocess.run([sys.executable, os.path.join(HERE, "build_arena.py"), "--versions", v, *rest])
        if r.returncode:
            print(f"pass {v}: build_arena failed ({r.returncode}); stopping", flush=True)
            return r.returncode
        r = subprocess.run([sys.executable, os.path.join(HERE, "import_to_db.py"), "--write"])
        print(
            f"pass {v}: done in {(time.time() - t) / 60:.0f} min"
            + ("" if r.returncode == 0 else f" (import failed: {r.returncode})"),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
