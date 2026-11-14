"""The stratastills server: `./stratastills serve` opens on the list of projects to choose from.

    ./stratastills serve [--host 127.0.0.1] [--port 8361]

API (JSON): GET /api/report (is there a visual report) and GET /report/ (the report's own pages, from settings.report_dir()); GET /api/settings; PUT /api/settings {projects_dir}; GET /api/projects; POST /api/projects {name, sources?, group?};
GET /api/project?id=; GET /api/project/sources?id=; GET /api/project/items?id=&source=; GET /api/project/item?id=&item=; PUT /api/project/primary?id=&item= {variant}; GET /api/project/image?id=&item=&variant=&transform=&px=; GET /api/project/thumb?id=&item=&variant=&px= (the grid's thumbnail: the primary variant, a raw's embedded preview, cached with the Browser's); GET /api/project/crop?id=&item=&variant=&transform=&x=&y=&size=&zoom= (the loupe: the same scene area of any variant and transform at full resolution; 202 while the variant's full-size copy is being made); GET /api/project/fullres?id=&item=&variant= (starts making it and reports {state: ready|running|failed|idle, text}); GET /api/project/transform?id=&item=&variant=&transform=; GET /api/project/detail?id=&item=&variant=; GET /api/pipelines, PUT/DELETE /api/pipeline?name= (presets); POST /api/project/generate/check?id=&item= and POST /api/project/generate?id=&item= {variant, preset, options, stages} (a new generated variant; runs in the background), GET /api/project/generate?id=&item= (progress), GET /api/project/generated/file?id=&item=&variant=; GET/PUT /api/project/rules?id=, POST (optional &workers=: how many pictures at once, default by machine) /GET/DELETE /api/project/apply?id= (apply the rules to the sources); GET /api/project/shots?id=; POST /api/project/sources?id= {path}; DELETE /api/project/sources?id=&path=;
the Browser (browser.py): GET /api/browse/folders?folder= (sub-folders of a folder key `<root label>/<path>`, the roots without one), GET /api/browse/pictures?folder=, GET /api/browse/thumb?folder=&stem=&source=&px= (cached), POST /api/browse/add {project | name, group, folders, subfolders, pictures}.
Project ids contain slashes, so they are query parameters. The page at / is a plain single-file stand-in until the React app (web/) exists. Binds to 127.0.0.1 by default: the person using it can name any folder as a source (read-only)."""

import hashlib
import os
import random
import re
import secrets
import threading
import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from stratastills import (
    arena as AR,
)
from stratastills import (
    arena_store,
    browser,
    flags_store,
    ingest,
    nsfw,
    preview,
    settings,
)
from stratastills import (
    export as X,
)
from stratastills import (
    pipeline as PL,
)
from stratastills import (
    projects as P,
)

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
LANDING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "landing.html")


def app_page(route: str, title: str) -> str:
    """A page that moved into the React app (web/src/routes.tsx): send the browser to its route there, keeping the query (the arena's ?set= and ?want=).
    Relative, so it works at the server's root and behind Caddy at /strata-stills/ alike; without a build, it says how to make one."""
    if not os.path.exists(os.path.join(STATIC, "index.html")):
        return f"<!doctype html><meta charset=utf-8><title>{title}</title><p>{title} is part of the web app, which is not built: <code>cd web; npm run build</code>.</p>"
    return f"<!doctype html><meta charset=utf-8><title>{title}</title><script>location.replace('../#/{route}'+location.search)</script><p><a href='../#/{route}'>{title}</a></p>"


class NewProject(BaseModel):
    name: str
    sources: list[str] = []
    group: str | None = None


class SourcePath(BaseModel):
    path: str


class PrimaryIn(BaseModel):
    variant: str


class GenerateIn(BaseModel):
    """What to make: a preset (default 'best') with the Advanced panel's flat options laid over it, or the full `stages` of a custom pipeline."""

    variant: str | None = None  # the source variant to start from (default: the item's primary)
    preset: str | None = None
    options: dict | None = None  # size, format, quality, level, sharpen, sharpen_amount, highlight_contrast
    stages: list[dict] | None = None


class PresetIn(BaseModel):
    label: str | None = None
    stages: list[dict]


class RulesIn(BaseModel):
    rules: list[dict]


class FlagIn(BaseModel):
    flag: bool = True
    rating: str = ""  # worst | bad | medium | good | perfect | '' (not rated)
    reasons: list[str] = []
    note: str = ""
    status: str = "open"  # open | fixable | deferred | fixed


class VoteIn(BaseModel):
    token: str
    choice: str  # left | right | tie | skip
    ms: int = 0


class FlaggedIn(BaseModel):
    path: str
    allow: bool = True


class FlaggedScanIn(BaseModel):
    folder: str


class AlertIn(BaseModel):
    token: str
    side: str  # left | right: the picture with something wrong
    note: str = ""
    ms: int = 0


class AlertStatusIn(BaseModel):
    id: str
    resolved: bool | None = None  # older clients: True = fixed, False = open
    status: str | None = None  # open | investigating | fixed | wontfix
    note: str | None = None  # the reason, written afterwards


class BrowseAddIn(BaseModel):
    """What the Browser adds: to `project`, or to a new project called `name` (in `group`); whole `folders` (and every folder below that holds pictures when `subfolders`) and single `pictures` ({folder key: [stems]})."""

    project: str | None = None
    name: str | None = None
    group: str | None = None
    folders: list[str] = []
    subfolders: bool = False
    pictures: dict[str, list[str]] = {}


class SettingsIn(BaseModel):
    projects_dir: str | None = None


def create_app():
    app = FastAPI(title="strata-stills", docs_url="/api/docs", openapi_url="/api/openapi.json")
    jobs = {}  # (project, item) -> {state: running|done|failed, text, result, error}: generating takes a minute or two, so it runs in a thread and is polled
    batches = {}  # project -> {state, done, total, text, failed, stop}: applying the rules to a whole source

    def guard(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except P.ProjectError as e:
            raise HTTPException(400, str(e))

    @app.get("/", response_class=HTMLResponse)
    def index():
        built = os.path.join(
            STATIC, "index.html"
        )  # the React app when it has been built (cd web; npm run build), else the plain stand-in page
        return open(built if os.path.exists(built) else LANDING, encoding="utf8").read()

    @app.get("/flagged")
    def flagged_slash():
        return RedirectResponse("/flagged/")

    @app.get("/flagged/", response_class=HTMLResponse)
    def flagged_page():
        return app_page("flagged", "Flagged pictures")

    @app.get("/api/flagged")
    def flagged_list():
        return dict(threshold=nsfw.threshold(), enabled=nsfw.enabled(), items=nsfw.review())

    def flagged_known(path):
        if path not in {r["path"] for r in nsfw.review()}:
            raise HTTPException(
                404, "not a flagged picture"
            )  # only what the filter has flagged is served: no way to read other files through here

    @app.get("/api/flagged/image")
    def flagged_image(path: str, px: int = 900):
        flagged_known(path)
        import cv2

        from stratastills import images

        ok, buf = cv2.imencode(".jpg", images.load(path, max(100, min(px, 3000))), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/flagged/decide")
    def flagged_decide(body: FlaggedIn):
        flagged_known(body.path)
        nsfw.allow(body.path, body.allow)
        return dict(ok=True)

    @app.post("/api/flagged/scan")
    def flagged_scan(body: FlaggedScanIn):
        """Look at every picture of a folder (and below) so that it shows up here when flagged; the filter's skip is applied by ingest.scan itself."""
        if not os.path.isdir(body.folder):
            raise HTTPException(400, f"not a folder: {body.folder}")
        n = len(ingest.scan(body.folder, recursive=True))
        return dict(kept=n, flagged=len(nsfw.review()))

    @app.get("/api/report")
    def report_info():
        root = settings.report_dir()
        f = os.path.join(
            root, "index.html"
        )  # the visual reports (scripts/build_report.py -> neutral/, scripts/build_lr_report.py -> auto/): static files, looked up per request so one built after the server started shows up
        peers = {d: os.path.exists(os.path.join(root, d, "index.html")) for d in ("neutral", "auto")}
        return dict(available=os.path.exists(f), built=os.path.getmtime(f) if os.path.exists(f) else None, **peers)

    pending, arena_lock = {}, threading.Lock()

    def arena_pool():
        return arena_store.get().pool()

    def arena_retired():
        return arena_store.get().retired()

    def arena_votes():
        return arena_store.get().votes()

    def arena_alert_events():
        return arena_store.get().alert_events()  # files or the database (arena_store.py)

    @app.get("/arena")
    def arena_slash():
        return RedirectResponse("/arena/")

    @app.get("/arena/", response_class=HTMLResponse)
    def arena_page():
        return app_page("arena", "Rating arena")

    @app.get("/api/arena/next")
    def arena_next(want: str = "", set_name: str = Query("", alias="set")):
        try:
            only = AR.load_set(settings.arena_dir(), set_name)
        except ValueError as e:
            raise HTTPException(404, str(e))
        pool, retired = arena_pool(), arena_retired()
        pick = AR.pick_pair(
            pool,
            AR.current_votes(arena_votes(), pool),
            random.Random(),
            retired,
            {w for w in want.split(",") if w} or None,
            AR.open_alerts(arena_alert_events()),
            only,
            AR.load_priorities(settings.arena_dir()),
        )
        if not pick:
            raise HTTPException(404, "nothing to rate yet: make the pictures with scripts/build_arena.py")
        pic, a, b = pick
        token = secrets.token_urlsafe(9)
        with arena_lock:
            pending[token] = dict(
                pic=pic, a=a, b=b, t0=time.time(), ha=pool[pic][a].get("hash"), hb=pool[pic][b].get("hash")
            )  # the pictures shown: a vote counts for them, and for a remake only while it is judged the same (arena.current_votes)
            while len(pending) > 300:
                pending.pop(next(iter(pending)))
        return dict(
            token=token, left=f"img/{token}/left", right=f"img/{token}/right", pool=len(pool), votes=len(arena_votes())
        )

    @app.get("/arena/img/{token}/{side}")
    def arena_img(token: str, side: str):
        q = pending.get(token)
        if not q or side not in ("left", "right"):
            raise HTTPException(404, "unknown pair")
        rel = arena_pool()[q["pic"]][q["a"] if side == "left" else q["b"]]["file"]
        root = os.path.realpath(settings.arena_dir())
        f = os.path.realpath(os.path.join(root, rel))
        if os.path.commonpath([root, f]) != root or not os.path.isfile(f):
            raise HTTPException(404, "no picture")
        data = open(f, "rb").read()
        with arena_lock:  # the vote is for the bytes served, whatever the pool said at /next (a remake in between, or a database pool not yet re-imported)
            q["ha" if side == "left" else "hb"] = hashlib.sha1(data).hexdigest()[:16]
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/arena/vote")
    def arena_vote(body: VoteIn):
        with arena_lock:
            q = pending.pop(body.token, None)
        if not q:
            raise HTTPException(404, "unknown pair (already answered?)")
        if body.choice == "skip":
            return dict(ok=True, skipped=True)
        if any(a.get("token") == body.token and not a["resolved"] for a in AR.alerts(arena_alert_events())):
            return dict(
                ok=True, skipped=True, alerted=True
            )  # a pair with an open alert is not a vote: one of its pictures is wrong
        if body.choice not in ("left", "right", "tie"):
            raise HTTPException(400, "choice is left, right, tie or skip")
        arena_store.get().add_vote(
            dict(
                pic=q["pic"],
                a=q["a"],
                b=q["b"],
                winner={"left": "a", "right": "b", "tie": "tie"}[body.choice],
                t=round(time.time(), 1),
                ms=body.ms,
                ha=q["ha"],
                hb=q["hb"],
            )
        )
        return dict(ok=True, votes=len(arena_votes()))

    @app.post("/api/arena/alert")
    def arena_alert(body: AlertIn):
        """Something is wrong with one picture of the pair on screen (the vote is not recorded; the picture's variant is no longer offered until the alert is resolved)."""
        if body.side not in ("left", "right"):
            raise HTTPException(400, "side is left or right")
        q = pending.get(body.token)  # the pair stays on screen: the alert button turns red and a second press undoes it
        if not q:
            raise HTTPException(404, "unknown pair (already answered?)")
        mine = [
            a
            for a in AR.alerts(arena_alert_events())
            if a.get("token") == body.token and a["side"] == body.side and not a["resolved"]
        ]
        if mine:
            return dict(ok=True, id=mine[0]["id"])
        var, other = (q["a"], q["b"]) if body.side == "left" else (q["b"], q["a"])
        votes = arena_votes()
        ev = dict(
            id=secrets.token_hex(5),
            token=body.token,
            pic=q["pic"],
            variant=var,
            other=other,
            side=body.side,
            note=body.note.strip()[:500],
            t=round(time.time(), 1),
            ms=body.ms,
            votes_on_pic=sum(v["pic"] == q["pic"] for v in votes),
            votes_on_variant=sum(v["pic"] == q["pic"] and var in (v["a"], v["b"]) for v in votes),
        )
        arena_store.get().add_alert_event(ev)
        return dict(ok=True, id=ev["id"])

    @app.post("/api/arena/alert/clear")
    def arena_alert_clear(body: AlertIn):
        """Undo the alert on one side of the pair on screen (the button was pressed again)."""
        mine = [
            a
            for a in AR.alerts(arena_alert_events())
            if a.get("token") == body.token and a["side"] == body.side and not a["resolved"]
        ]
        for a in mine:
            arena_store.get().add_alert_event(dict(id=a["id"], resolved=True, t=round(time.time(), 1)))
        return dict(ok=True, cleared=len(mine))

    @app.get("/alerts")
    def alerts_slash():
        return RedirectResponse("/alerts/")

    @app.get("/alerts/", response_class=HTMLResponse)
    def alerts_page():
        return app_page("alerts", "Arena flags")

    @app.get("/api/alerts")
    def alerts_list():
        return dict(alerts=list(reversed(AR.alerts(arena_alert_events(), arena_pool()))))

    @app.get("/api/alerts/image/{aid}")
    def alerts_image(aid: str, variant: str | None = None):
        """The newest picture of the flagged version (as in the pool now, so after a rebuild the new one), or with ?variant=lightroom the same picture's other version to compare with."""
        pool = arena_pool()
        a = next((x for x in AR.alerts(arena_alert_events(), pool) if x["id"] == aid), None)
        ent = a and ((pool.get(a["pic"]) or {}).get(variant) if variant else a["pool"])
        if not ent:
            raise HTTPException(404, "no picture")
        root = os.path.realpath(settings.arena_dir())
        f = os.path.realpath(os.path.join(root, ent["file"]))
        if os.path.commonpath([root, f]) != root or not os.path.isfile(f):
            raise HTTPException(404, "no picture")
        return FileResponse(f, headers={"Cache-Control": "no-cache"})

    @app.post("/api/alerts/status")
    def alerts_status(body: AlertStatusIn):
        if body.id not in {a["id"] for a in AR.alerts(arena_alert_events())}:
            raise HTTPException(404, "unknown alert")
        if body.status is not None and body.status not in AR.STATUSES:
            raise HTTPException(400, "status is " + ", ".join(AR.STATUSES))
        ev = dict(id=body.id, t=round(time.time(), 1))
        if body.note is not None:
            ev["note"] = body.note.strip()[:500]
        if body.status is not None:
            ev["status"] = body.status
        elif body.resolved is not None:
            ev["resolved"] = body.resolved
        arena_store.get().add_alert_event(ev)
        return dict(ok=True)

    @app.post("/api/arena/undo")
    def arena_undo():
        with arena_lock:
            r = arena_store.get().undo()
        if r is None:
            raise HTTPException(404, "no vote to undo")
        return dict(undone=r[0], votes=r[1])

    @app.get("/api/arena/ratings")
    def arena_ratings():
        every, retired = arena_votes(), arena_retired()
        pool = arena_pool()
        votes = AR.current_votes(every, pool)  # only the votes on pictures still shown, or remade with a small change
        bt = AR.bradley_terry(votes, retired)
        ci = AR.bootstrap(votes, retired, n=100)
        e = AR.elo(votes, retired)
        n = {}
        for v in votes:
            for k in (v["a"], v["b"]):
                n[k] = n.get(k, 0) + 1
        names = sorted(set(bt) | {k for vs in pool.values() for k in vs} - retired, key=lambda k: -bt.get(k, 0))
        return dict(
            votes=len(votes),
            superseded_votes=len(every) - len(votes),
            pictures=len(pool),
            voted_pictures=len({v["pic"] for v in votes}),
            retired=sorted(retired),
            left_odds=round(
                AR.side_advantage(votes, retired), 3
            ),  # >1: the left picture wins more often between equals (already taken out of the ratings)
            variants=[
                dict(
                    name=k,
                    bt=round(bt[k]) if k in bt else None,
                    elo=round(e[k]) if k in e else None,
                    lo=round(ci[k][0]) if k in ci else None,
                    hi=round(ci[k][1]) if k in ci else None,
                    votes=n.get(k, 0),
                )
                for k in names
            ],
            head_to_head=[
                dict(a=a, b=b, a_wins=w[0], b_wins=w[1], ties=w[2])
                for (a, b), w in sorted(AR.head_to_head(votes, retired).items())
            ],
        )

    FLAG_REPORTS = ("neutral", "auto")
    FLAG_PIC = re.compile(r"^[A-Za-z0-9_.\- ]{1,120}$")
    flag_lock = threading.Lock()

    def flag_report(report):
        if report not in FLAG_REPORTS:
            raise HTTPException(404, "no such report")
        return report

    @app.get("/api/report-flags/{report}")
    def get_flags(report: str):
        return flags_store.get().read(flag_report(report))

    @app.put("/api/report-flags/{report}/{pic}")
    def put_flag(report: str, pic: str, body: FlagIn):
        report = flag_report(report)
        if not FLAG_PIC.match(pic):
            raise HTTPException(400, "bad picture id")
        with (
            flag_lock
        ):  # the flags of a report are one small file: read, change one picture, write whole (through a temporary file)
            st = flags_store.get()
            cur = st.read(report)
            rating = body.rating if body.rating in ("worst", "bad", "medium", "good", "perfect") else ""
            entry = None
            if rating or body.reasons or body.note:
                now = time.strftime("%Y-%m-%dT%H:%M:%S")
                entry = dict(
                    flag=True,
                    rating=rating,
                    reasons=body.reasons[:20],
                    note=body.note[:2000],
                    status=body.status if body.status in ("open", "fixable", "deferred", "fixed") else "open",
                    at=now,
                    first=cur.get(pic, {}).get("first", now),
                )  # `at` is the last change, `first` when it was first flagged: to see which to revisit
            return st.write(report, pic, entry)

    @app.get("/report")
    def report_slash():
        return RedirectResponse("/report/")

    @app.get("/report/{path:path}")
    def report_file(path: str = ""):
        root = os.path.realpath(settings.report_dir())
        f = os.path.realpath(os.path.join(root, path or "index.html"))
        if os.path.isdir(f):
            f = os.path.join(f, "index.html")
        if os.path.commonpath([root, f]) != root or not os.path.isfile(f):
            raise HTTPException(
                404,
                "no report here: build it with scripts/build_report.py (see the README), or set STRATASTILLS_REPORT to its folder",
            )
        return FileResponse(f, headers={"Cache-Control": "no-cache"})

    # ---- the Browser: folders of the source roots, their pictures as cached thumbnails, adding them to a project (browser.py) --------------------------------------------------
    def bguard(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except (browser.BrowseError, P.ProjectError) as e:
            raise HTTPException(400, str(e))
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))

    @app.get("/api/browse/folders")
    def browse_folders(folder: str | None = None):
        return bguard(browser.folders, folder)

    @app.get("/api/browse/pictures")
    def browse_pictures(folder: str):
        return bguard(browser.pictures, folder)

    @app.get("/api/browse/thumb")
    def browse_thumb(folder: str, stem: str, source: str = "embedded", px: int = 320):
        path = bguard(browser.thumb, folder, stem, source, px)
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/project/thumb")
    def project_thumb(id: str, item: str, variant: str | None = None, px: int = 320):
        path = bguard(browser.item_thumb, id, item, px, variant)
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @app.post("/api/browse/add")
    def browse_add(b: BrowseAddIn):
        return bguard(browser.add, b.project, b.name, b.group, b.folders, b.subfolders, b.pictures)

    @app.get("/api/settings")
    def get_settings():
        return settings.all_settings()

    @app.put("/api/settings")
    def put_settings(s: SettingsIn):
        if s.projects_dir is not None and not os.path.isabs(os.path.expanduser(s.projects_dir)):
            raise HTTPException(400, "the projects folder must be an absolute path")
        settings.save(dict(projects_dir=s.projects_dir))
        return settings.all_settings()

    @app.get("/api/projects")
    def list_projects():
        return P.list_projects()

    @app.post("/api/projects")
    def new_project(b: NewProject):
        return guard(P.create, b.name, b.sources, b.group)

    @app.get("/api/project")
    def get_project(id: str):
        return guard(P.load, id)

    @app.get("/api/project/shots")
    def get_shots(id: str):
        out = []
        for src, sh in guard(P.shots, id):
            out.append(dict(source=src, unreachable=True) if sh is None else dict(sh.to_json(), source=src))
        return out

    @app.get("/api/project/sources")
    def get_sources(id: str):
        return guard(P.source_summaries, id)

    @app.get("/api/project/items")
    def get_items(id: str, source: str):
        return guard(P.source_items, id, source)

    @app.get("/api/project/item")
    def get_item(id: str, item: str):
        return guard(P.get_item, id, item)

    @app.put("/api/project/primary")
    def put_primary(id: str, item: str, b: PrimaryIn):
        return guard(P.set_primary, id, item, b.variant)

    @app.get("/api/project/image")
    def get_image(id: str, item: str, variant: str | None = None, transform: str = "original", px: int = 1600):
        path, _ = guard(preview.render, id, item, variant, transform, px)
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/project/crop")
    def get_crop(
        id: str,
        item: str,
        variant: str | None = None,
        transform: str = "original",
        x: float = 0.5,
        y: float = 0.5,
        size: int = 200,
        zoom: int = 2,
    ):
        try:
            return Response(
                guard(preview.crop, id, item, variant, transform, x, y, size, zoom), media_type="image/jpeg"
            )
        except preview.NotReady as e:
            return JSONResponse(
                dict(e.status, detail="the full-resolution copy of this picture is still being made"), status_code=202
            )

    @app.get("/api/project/fullres")
    def get_fullres(id: str, item: str, variant: str | None = None):
        return guard(preview.fullres_status, id, item, variant)

    @app.get("/api/project/transform")
    def get_transform(id: str, item: str, variant: str | None = None, transform: str = "original"):
        _, info = guard(preview.render, id, item, variant, transform, 1600)
        return info

    @app.get("/api/project/detail")
    def get_detail(id: str, item: str, variant: str | None = None):
        it = guard(P.get_item, id, item)
        v = next((x for x in it["variants"] if x["id"] == (variant or it["primary"])), None)
        if v is None:
            raise HTTPException(400, "no such variant")
        try:
            return X.measure_detail(id, v["path"])
        except Exception as e:
            raise HTTPException(400, str(e))

    # ---- pipelines: presets, generating a variant, rules for a whole source -------------------------------------------------------------------------------------------------------------------------------------
    @app.get("/api/pipelines")
    def get_pipelines():
        return PL.list_presets()

    @app.put("/api/pipeline")
    def put_pipeline(name: str, b: PresetIn):
        return guard(PL.save_preset, name, b.label, b.stages)

    @app.delete("/api/pipeline")
    def delete_pipeline(name: str):
        guard(PL.delete_preset, name)
        return PL.list_presets()

    @app.get("/api/look")
    def look_versions():
        """The look's versions with their arena rating, and which one 'best' is now (the Generate panel's Look list)."""
        from stratastills import look

        r = look.ratings()
        return dict(
            best=look.winner(),
            versions=[
                dict(name=v, rating=r.get(v, {}).get("rating"), votes=r.get(v, {}).get("votes", 0))
                for v in look.VARIANTS
            ],
        )

    @app.post("/api/project/generate/check")
    def check_generate(id: str, item: str, b: GenerateIn):
        key, have = guard(PL.find_existing, id, item, b.variant, b.stages, b.preset, b.options)
        st = guard(PL.resolve, b.preset, b.options, b.stages)
        return dict(
            key=key, exists=have is not None, variant=have["id"] if have else None, stages=st, summary=PL.summary(st)
        )

    @app.post("/api/project/generate")
    def start_generate(id: str, item: str, b: GenerateIn):
        key, have = guard(PL.find_existing, id, item, b.variant, b.stages, b.preset, b.options)
        if have is not None:
            return dict(state="exists", variant=have["id"])
        if jobs.get((id, item), {}).get("state") == "running":
            raise HTTPException(409, "a picture of this item is already being generated")
        job = jobs[(id, item)] = dict(state="running", text="starting", result=None, error=None)

        def run():
            try:
                job["result"] = PL.generate(
                    id, item, b.variant, b.preset, b.options, b.stages, progress=lambda t: job.update(text=t)
                )
                job["state"] = "done"
            except Exception as e:
                job.update(state="failed", error=str(e))

        threading.Thread(target=run, daemon=True).start()
        return job

    @app.get("/api/project/generate")
    def generate_status(id: str, item: str):
        return jobs.get((id, item), dict(state="idle"))

    @app.get("/api/project/generated/file")
    def generated_file(id: str, item: str, variant: str):
        v = next(
            (x for x in guard(P.get_item, id, item)["variants"] if x["id"] == variant and x["kind"] == "generated"),
            None,
        )
        if v is None or not os.path.isfile(v["path"]):
            raise HTTPException(404, "no such generated picture")
        return FileResponse(v["path"], filename=f"{item.split('~')[-1]}-{v['output'][0]}x{v['output'][1]}{v['ext']}")

    @app.get("/api/project/rules")
    def get_rules(id: str):
        return dict(rules=guard(PL.rules, id), pending=len(guard(PL.pending, id)))

    @app.put("/api/project/rules")
    def put_rules(id: str, b: RulesIn):
        guard(PL.set_rules, id, b.rules)
        return dict(rules=PL.rules(id), pending=len(PL.pending(id)))

    @app.post("/api/project/apply")
    def start_apply(id: str, workers: int | None = None):
        guard(PL.rules, id)
        if batches.get(id, {}).get("state") == "running":
            raise HTTPException(409, "this project is already being processed")
        job = batches[id] = dict(state="running", done=0, total=0, text="starting", failed=[], stop=False)

        def run():
            def prog(i, n, t):
                job.update(done=i, total=n, text=t)

            try:
                r = PL.apply_rules(id, prog, stop=lambda: job["stop"], workers=workers)
                job.update(
                    state="stopped" if job["stop"] else "done",
                    done=r["done"] + len(r["failed"]),
                    total=r["total"],
                    failed=r["failed"],
                    workers=r["workers"],
                )
            except Exception as e:
                job.update(state="failed", text=str(e))

        threading.Thread(target=run, daemon=True).start()
        return job

    @app.get("/api/project/apply")
    def apply_status(id: str):
        return batches.get(id, dict(state="idle"))

    @app.delete("/api/project/apply")
    def stop_apply(id: str):
        if id in batches:
            batches[id]["stop"] = True
        return batches.get(id, dict(state="idle"))

    @app.post("/api/project/sources")
    def add_source(id: str, b: SourcePath):
        return guard(P.add_source, id, b.path)

    @app.delete("/api/project/sources")
    def remove_source(id: str, path: str):
        return guard(P.remove_source, id, path)

    if os.path.isdir(os.path.join(STATIC, "assets")):
        app.mount("/assets", StaticFiles(directory=os.path.join(STATIC, "assets")), name="assets")
    return app


def main(argv=None):
    import argparse

    import uvicorn

    a = argparse.ArgumentParser()
    a.add_argument("--host", default="127.0.0.1")
    a.add_argument("--port", type=int, default=8361)
    a = a.parse_args(argv)
    print(f"projects folder: {settings.projects_dir()}\nhttp://{a.host}:{a.port}/")
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")
