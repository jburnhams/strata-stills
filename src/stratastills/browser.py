"""The Browser screen: the source roots as a tree of folders, the pictures of one folder as thumbnails, and adding pictures or whole folders to a project. Originals are only read.

A folder is named by a KEY: the root's label (its folder name, as the database labels roots: `raw`, `drone`) and the path below it, with forward slashes (`raw/2019/2019-03-02`). Keys never leave the roots (`locate` refuses `..` and links out).

Thumbnails come from the best picture there is for a shot, in this order (`sources`): a develop the app made in a project that has the folder as a source (newest first), the person's Lightroom export (the database's `lightroom_edit`, when the database is switched on), the preview the camera embedded in the raw, the camera JPEG, and last a develop of the raw itself (slow). The listing names each picture's sources, and the web app asks for them in order (strata-react's `Thumbnail` tries the next when one fails), so a request names its source and the server never re-reads the folder to answer it.
Each thumbnail is made once and kept in `settings.thumbs_dir()` under a key of the source file's path, size and time stamp, the size asked for and `VERSION`: a changed file or a new recipe is a new key. Making them is IO-bound on a spinning archive disk, so at most `SLOTS` are made at a time, and the embedded preview is read by its offset (`images.embedded_preview`: the headers and the JPEG, a few hundred KB, not the 25 MB raw)."""

import datetime as dt
import hashlib
import io
import json
import os
import threading

from stratastills import images, ingest, settings
from stratastills import projects as P

VERSION = 1  # of the thumbnail recipe: bump when what a thumbnail looks like changes
SIZES = (160, 320, 640)  # the sizes made (a request is rounded up to one, so the cache holds few copies)
SLOTS = threading.BoundedSemaphore(4)
MAX_FOLDERS = 2000  # the most folders one 'add with subfolders' adds
PICTURE_EXTS = ingest.RAW_EXTS + ingest.STILL_EXTS


class BrowseError(ValueError):
    pass


_scans = {}  # folder -> (dir mtime_ns, time read, shots): every thumbnail request needs its picture's files
SCAN_TTL = 120.0
_checking = set()  # folders whose pictures the nudity filter is looking at in the background
_check_lock = threading.Lock()
_check_errors = {}  # folder -> why its background check stopped (the detector is not installed), told once at the next listing


def _scan(path):
    """ingest.scan of a folder WITHOUT the nudity filter (see `_visible`), remembered for SCAN_TTL seconds while the folder is unchanged."""
    import time

    m = os.stat(path).st_mtime_ns
    hit = _scans.get(path)
    if hit and hit[0] == m and time.monotonic() - hit[1] < SCAN_TTL:
        return hit[2]
    shots = ingest.scan(path, filter=False)
    _scans[path] = (m, time.monotonic(), shots)
    return shots


def _check_later(path, shots):
    """Run the nudity filter on the pictures of a folder it has not seen, in one background thread per folder (on a new folder that is a decode per picture, minutes on the archive disk: the listing must not wait for it)."""
    from stratastills import nsfw

    with _check_lock:
        if path in _checking:
            return
        _checking.add(path)

    def run():
        try:
            for sh in shots:
                try:
                    nsfw.shot_flagged(sh)
                except nsfw.NsfwUnavailable as e:
                    _check_errors[path] = str(
                        e
                    )  # no detector: say so at the next listing instead of 'still being checked' for ever
                    return
                except Exception:
                    pass  # an unreadable picture scores as flagged inside nsfw; anything else is retried at the next listing
        finally:
            with _check_lock:
                _checking.discard(path)

    threading.Thread(target=run, daemon=True, name=f"browser-nsfw {os.path.basename(path)}").start()


def _visible(path):
    """(shots the nudity filter lets through, how many it has not looked at yet). Unchecked ones are left out and checked in the background, so they appear at a later listing; flagged ones never appear."""
    from stratastills import nsfw

    shots = _scan(path)
    if path in _check_errors:
        raise BrowseError(_check_errors.pop(path))
    pending = [sh for sh in shots if not nsfw.shot_known(sh)]
    if pending:
        _check_later(path, pending)
    waiting = {id(sh) for sh in pending}
    return [sh for sh in shots if id(sh) not in waiting and not nsfw.shot_flagged(sh)], len(pending)


def roots():
    """[(label, absolute root)] in settings order; a label is the root's folder name (as `db/hashes.root_labels`)."""
    out, seen = [], set()
    for r in settings.source_roots():
        name = os.path.basename(os.path.normpath(r)) or r
        if name in seen:
            raise BrowseError(f"source roots share a folder name: {name}")
        seen.add(name)
        out.append((name, os.path.normpath(r)))
    return out


def locate(key):
    """The absolute folder of a key; BrowseError when the root is unknown, the key leaves it or it is not a folder."""
    parts = [x for x in (key or "").replace("\\", "/").split("/") if x]
    if not parts:
        raise BrowseError("no folder given")
    if any(x in (".", "..") for x in parts):
        raise BrowseError(f"bad folder {key!r}")
    root = dict(roots()).get(parts[0])
    if root is None:
        raise BrowseError(f"no source root called {parts[0]!r}")
    p = os.path.join(root, *parts[1:])
    real_root, real = os.path.realpath(root), os.path.realpath(p)
    if os.path.commonpath([real_root, real]) != real_root:
        raise BrowseError(f"bad folder {key!r}")
    if not os.path.isdir(p):
        raise BrowseError(f"not a folder: {key}")
    return p


def key_of(path):
    """The key of an absolute folder under a root, else None."""
    ap = os.path.normpath(path)
    for name, root in sorted(roots(), key=lambda x: -len(x[1])):
        if ap == root:
            return name
        if ap.startswith(root + os.sep):
            return name + "/" + os.path.relpath(ap, root).replace(os.sep, "/")
    return None


# ---- the tree ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def _count(path):
    """(sub-folders, pictures) in a folder, from one listing: pictures are distinct stems of raw or still files (a raw and its JPEG are one)."""
    subs, stems = 0, set()
    try:
        with os.scandir(path) as it:
            for e in it:
                if e.name.startswith("."):
                    continue
                try:
                    if e.is_dir():
                        subs += 1
                        continue
                except OSError:
                    continue
                stem, _tag, ext = ingest.split_name(e.name)
                if ext in PICTURE_EXTS:
                    stems.add(stem)
    except OSError:
        return 0, 0
    return subs, len(stems)


def folders(key=None):
    """The folders in `key` (the roots when None): [{key, name, folders, pictures}] sorted by name, each with how many sub-folders and pictures it holds."""
    if not key:
        return [dict(key=n, name=n, **dict(zip(("folders", "pictures"), _count(r)))) for n, r in roots()]
    base = locate(key)
    out = []
    with os.scandir(base) as it:
        for e in sorted(it, key=lambda e: e.name.lower()):
            if e.name.startswith(".") or not e.is_dir():
                continue
            subs, pics = _count(e.path)
            out.append(dict(key=f"{key.rstrip('/')}/{e.name}", name=e.name, folders=subs, pictures=pics))
    return out


# ---- the pictures of a folder ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def _projects_with(path):
    """[(project id, its source dict)] of every project with `path` as a source."""
    np = os.path.normcase(os.path.normpath(path))
    out = []
    for p in P.list_projects():
        for s in p["sources"]:
            try:
                where = P.resolve(s)
            except Exception:
                continue
            if os.path.normcase(os.path.normpath(where)) == np:
                out.append((p["id"], s))
    return out


def _lightroom(path):
    """{stem: absolute path of the Lightroom export} for the raws of a folder, from the database; {} when it is off or does not know the folder."""
    if not settings.database_url():
        return {}
    try:
        from stratastills.db.browser import lightroom_finals

        return lightroom_finals(path)
    except Exception:
        return {}  # the database is an extra here: the screen works without it


def _iso(ts):
    return dt.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def pictures(key):
    """The pictures of a folder: {pictures: [{stem, kind, files, taken, bytes, sources: [{id, label}], projects: [id]}] by name, checking: how many the nudity filter is still looking at (they are listed once it has)}. `sources` is best first (see the module docstring); `projects` lists the projects that already hold the picture. Videos are left out."""
    path = locate(key)
    shots, checking = _visible(path)
    shots = [s for s in shots if s.kind != "video"]
    projs = _projects_with(path)
    lr = _lightroom(path) if any(s.raw for s in shots) else {}
    out = []
    for sh in shots:
        files = [os.path.basename(f) for f in (sh.raw, sh.jpeg) if f] + [os.path.basename(f) for _t, f in sh.derived]
        main = sh.raw or sh.jpeg or (sh.derived[0][1] if sh.derived else None)
        try:
            st = os.stat(main) if main else None
        except OSError:
            st = None
        srcs, held = [], []
        for pid, s in projs:
            only = s.get("stems")
            if only is not None and sh.stem not in only:
                continue
            held.append(pid)
            iid = P.item_id(P.source_id(s["path"]), sh.stem)
            try:
                gen = P.generated_variants(pid, iid)
            except Exception:
                gen = []
            for v in reversed(gen):
                srcs.append(dict(id=f"gen:{pid}:{v['key']}", label=f"Developed in {pid}"))
        if sh.stem in lr:
            srcs.append(dict(id="lightroom", label="Lightroom export"))
        if sh.raw:
            srcs.append(dict(id="embedded", label="Camera preview"))
        if sh.jpeg:
            srcs.append(dict(id="jpeg", label="Camera JPEG"))
        for tag, _f in sh.derived:
            srcs.append(dict(id=f"derived:{tag}", label=tag))
        if sh.raw:
            srcs.append(dict(id="raw", label="Raw"))
        out.append(
            dict(
                stem=sh.stem,
                kind=sh.kind,
                files=files,
                taken=_iso(st.st_mtime) if st else None,
                bytes=st.st_size if st else None,
                sources=srcs,
                projects=held,
            )
        )
    return dict(pictures=out, checking=checking)


# ---- thumbnails --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def _size(px):
    px = int(px)
    return next((s for s in SIZES if s >= px), SIZES[-1])


def _resolve(key, stem, source):
    """(path of the file, how to read it) of one thumbnail source of a picture."""
    from stratastills import nsfw

    path = locate(key)
    sh = next((s for s in _scan(path) if s.stem == stem), None)
    if sh is None or nsfw.shot_flagged(sh):
        raise BrowseError(f"no picture {stem!r} in {key}")
    if source == "embedded" and sh.raw:
        return sh.raw, "embedded"
    if source == "raw" and sh.raw:
        return sh.raw, "develop"
    if source == "jpeg" and sh.jpeg:
        return sh.jpeg, "picture"
    if source.startswith("derived:"):
        tag = source.split(":", 1)[1]
        f = next((f for t, f in sh.derived if t == tag), None)
        if f:
            return f, "develop" if os.path.splitext(f)[1].lower() in ingest.RAW_EXTS else "picture"
    if source == "lightroom":
        f = _lightroom(path).get(stem)
        if f and os.path.isfile(f):
            return f, "picture"
    if source.startswith("gen:"):
        _g, pid, gkey = source.split(":", 2)
        for p, s in _projects_with(path):
            if p != pid:
                continue
            for v in P.generated_variants(pid, P.item_id(P.source_id(s["path"]), stem)):
                if v["key"] == gkey and os.path.isfile(v["path"]):
                    return v["path"], "develop" if v["ext"] in (".tif", ".tiff") else "picture"
    raise BrowseError(f"picture {stem!r} has no {source!r}")


_ORIENT = {3: 180, 6: 270, 8: 90}  # EXIF orientation -> PIL rotate angle (counter-clockwise)


def _make(path, how, px):
    """The thumbnail as JPEG bytes, long side px."""
    from PIL import Image, ImageOps

    if how == "embedded":
        got = images.embedded_preview(path)
        if got is None:
            how = "develop"
        else:
            data, orient = got
            im = Image.open(io.BytesIO(data))
            im.draft("RGB", (px, px))
            im = im.convert("RGB")
            if orient in _ORIENT:
                im = im.rotate(_ORIENT[orient], expand=True)
    if how == "picture":
        try:
            im = Image.open(path)
            im.draft("RGB", (px, px))
            im = ImageOps.exif_transpose(im).convert("RGB")
        except Exception:
            how = "develop"  # a picture Pillow cannot open (a 16-bit RGB TIFF): images.load has the readers for it
    if how == "develop":
        import cv2

        bgr = images.load(path, px)
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    im.thumbnail((px, px), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85, optimize=True)
    return buf.getvalue()


def thumb(key, stem, source="embedded", px=320):
    """The path of the cached thumbnail of one picture from one of its sources, made if it is not there yet."""
    px = _size(px)
    path, how = _resolve(key, stem, source)
    st = os.stat(path)
    k = hashlib.sha1(
        json.dumps([os.path.normcase(os.path.abspath(path)), st.st_size, st.st_mtime_ns, how, px, VERSION]).encode()
    ).hexdigest()
    out = os.path.join(settings.thumbs_dir(), k[:2], k + ".jpg")
    if os.path.exists(out):
        return out
    with SLOTS:
        if os.path.exists(out):  # made while this request waited
            return out
        try:
            data = _make(path, how, px)
        except Exception as e:
            raise BrowseError(f"cannot make a thumbnail of {os.path.basename(path)}: {e}")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = f"{out}.{threading.get_ident()}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, out)
    return out


# ---- adding to a project -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def _with_pictures(path, limit=MAX_FOLDERS):
    """`path` and every folder below it that holds pictures, in walk order (at most `limit`)."""
    out = []
    for d, subs, names in os.walk(path):
        subs[:] = sorted(s for s in subs if not s.startswith("."))
        if any(ingest.split_name(n)[2] in PICTURE_EXTS for n in names):
            out.append(d)
            if len(out) > limit:
                raise BrowseError(f"more than {limit} folders with pictures: choose a smaller part of the tree")
    return out


def add(project=None, name=None, group=None, folders=(), subfolders=False, pictures=None):
    """Add whole folders (with every folder below that holds pictures when `subfolders`) and single pictures ({folder key: [stems]}) to `project`, or to a new project called `name`. Returns {project, sources, pictures, unchanged}: how many folders were added whole, how many pictures were added on their own, and how many asked for were there already."""
    if not project and not (name or "").strip():
        raise BrowseError("choose a project or name a new one")
    if not project:
        project = P.create(name, group=group)["id"]
    else:
        P.load(project)
    whole, picked, same = 0, 0, 0
    for k in folders or ():
        base = locate(k)
        for d in _with_pictures(base) if subfolders else [base]:
            n = P.include(project, d)
            whole += n > 0
            same += n == 0
    for k, stems in (pictures or {}).items():
        n = P.include(project, locate(k), stems=list(stems))
        picked += n
        same += len(set(stems)) - n
    return dict(project=project, sources=whole, pictures=picked, unchanged=same)
