"""Read-only parity check of the report flags and the pipeline presets on the real data (src/stratastills/flags_store.py, presets_store.py): for each report, the flags the file store reads against what the database holds ALONE (every field, the times as written), and the saved presets the same way. Nothing is written. Exit status 0 only when everything is equal and at least one flag was compared.
python scripts/parity_flags_presets.py [--flags ~/code/github/strata-stills/out/flags] [--url postgresql+psycopg:///omarchy?options=-csearch_path%3Dphotos,public]"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stratastills import flags_store, presets_store, settings
from stratastills.db.flags import DbStore as FlagDb
from stratastills.db.presets import DbStore as PresetDb


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=settings.database_url() or "photos")
    ap.add_argument(
        "--flags",
        help="the report flags folder (default: out/flags beside the code this runs from; the running service uses ~/code/github/strata-stills/out/flags)",
    )
    a = ap.parse_args()
    if a.flags:
        os.environ["STRATASTILLS_REPORT_FLAGS"] = os.path.abspath(os.path.expanduser(a.flags))
    ff, fd, pf, pd = flags_store.FileStore(), FlagDb(a.url), presets_store.FileStore(), PresetDb(a.url)
    rows = []
    ok = True
    seen = 0
    for rep in flags_store.REPORTS:
        f, d = ff.read(rep), fd.read_db(rep)
        only_f, only_d = sorted(set(f) - set(d)), sorted(set(d) - set(f))
        diff = sorted(k for k in set(f) & set(d) if f[k] != d[k])
        seen += len(f)
        rows.append(
            dict(
                report=rep,
                file=len(f),
                database=len(d),
                equal=len(set(f) & set(d)) - len(diff),
                only_in_file=only_f[:3],
                only_in_database=only_d[:3],
                different=diff[:3],
            )
        )
        ok &= not (only_f or only_d or diff)
    pa, pb = {p["name"]: p for p in pf.list()}, {p["name"]: p for p in pd.list_db()}
    pdiff = sorted(k for k in set(pa) & set(pb) if pa[k] != pb[k])
    ok &= set(pa) == set(pb) and not pdiff
    print(json.dumps(dict(flags=rows, presets=dict(file=len(pa), database=len(pb), different=pdiff)), indent=1))
    print("PARITY OK" if ok and seen else "PARITY DIFFERS")
    return 0 if ok and seen else 1


if __name__ == "__main__":
    sys.exit(main())
