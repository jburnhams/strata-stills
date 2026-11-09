"""Read the full EXIF/XMP/MakerNote metadata of the hashed raws with exiftool and keep it in a resumable JSONL (the input of scripts/import_meta.py). Reads only the file headers (the raws are on a slow disk), at nice 15 and idle I/O priority.
  python scripts/read_meta.py [--hashes ~/work/strata-stills/hash/raw_sha256.jsonl] [--out ~/work/strata-stills/hash/raw_meta.jsonl] [--batch 100] [--limit N]
One line per file: {path, sha256, size, mtime_ns, tags: {"Group:Tag": value}}. Options `-G1 -a -n -struct`: family-1 group names, duplicates kept, numbers unformatted (GPS signed, exposure in seconds), XMP structures as objects. Unknown (undecoded) tags are NOT kept (-u would be 100 KB per Sony raw), and the volatile System:File{Access,InodeChange}Date and FilePermissions are dropped. Needs exiftool on the PATH. Resumable: paths already in --out are skipped."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

VOLATILE = ("System:FileAccessDate", "System:FileInodeChangeDate", "System:FilePermissions", "System:Directory")
ARGS = ["-j", "-G1", "-a", "-n", "-struct"]


def lower_priority():
    try:
        if hasattr(os, "nice"):
            os.nice(15)
    except (OSError, AttributeError):
        pass
    try:
        if shutil.which("ionice"):
            subprocess.run(["ionice", "-c3", "-p", str(os.getpid())], check=False, capture_output=True)
    except OSError:
        pass


def is_error(tags):
    return any(k.endswith(":Error") or k == "Error" for k in tags)


def run_exiftool(paths):
    """{path: {tag: value}} for a batch; a file exiftool could not read is missing from the result."""
    with tempfile.NamedTemporaryFile("w", suffix=".args", delete=False, encoding="utf8") as f:
        f.write("\n".join(paths) + "\n")
        name = f.name
    try:
        out = subprocess.run(
            ["exiftool", *ARGS, "-charset", "filename=utf8", "-@", name],
            capture_output=True,
            text=True,
            encoding="utf8",
            check=False,
        ).stdout
    finally:
        os.unlink(name)
    try:
        rows = json.loads(out) if out.strip() else []
    except ValueError:
        rows = []
    return {r["SourceFile"]: {k: v for k, v in r.items() if k != "SourceFile" and k not in VOLATILE} for r in rows}


def main():
    h = os.path.expanduser("~/work/strata-stills/hash")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hashes", default=f"{h}/raw_sha256.jsonl")
    ap.add_argument("--out", default=f"{h}/raw_meta.jsonl")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if not shutil.which("exiftool"):
        print("exiftool not found", file=sys.stderr)
        return 2
    lower_priority()
    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out, encoding="utf8"):
            try:
                r = json.loads(ln)
                if not is_error(r.get("tags") or {}):
                    done.add(
                        r["path"]
                    )  # an error record from an earlier run is retried (the file is only ever appended to)
            except (ValueError, KeyError):
                pass
    todo = []
    for ln in open(a.hashes, encoding="utf8"):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r["path"] not in done:
            todo.append(r)
    if a.limit:
        todo = todo[: a.limit]
    n = bad = 0
    with open(a.out, "a", encoding="utf8") as o:
        for i in range(0, len(todo), a.batch):
            chunk = todo[i : i + a.batch]
            tags = run_exiftool([r["path"] for r in chunk])
            for r in chunk:
                t = tags.get(r["path"])
                if t is None or is_error(t):
                    bad += 1
                    print("no metadata:", r["path"], (t or {}).get("ExifTool:Error", ""), file=sys.stderr)
                    continue
                o.write(
                    json.dumps(
                        dict(path=r["path"], sha256=r["sha256"], size=r["size"], mtime_ns=r["mtime_ns"], tags=t),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
            o.flush()
    print(f"read {n}, unreadable {bad}, already done {len(done)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
