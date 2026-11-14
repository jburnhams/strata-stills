import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from stratastills.server.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATASTILLS_PROJECTS", str(tmp_path / "projects"))
    monkeypatch.setenv("STRATASTILLS_HOME", str(tmp_path / "home"))
    return TestClient(create_app())


def test_landing_page_and_empty_list(client):
    assert "strata-stills" in client.get("/").text and client.get("/api/projects").json() == []


def test_project_lifecycle(client, tmp_path):
    src = tmp_path / "s"
    src.mkdir()
    (src / "DJI_1_0001_D.JPG").write_bytes(b"x")
    p = client.post("/api/projects", json=dict(name="Demo", sources=[str(src)])).json()
    assert p["id"] == "demo"
    assert client.post("/api/projects", json=dict(name="Demo")).status_code == 400
    assert [s["stem"] for s in client.get("/api/project/shots", params=dict(id="demo")).json()] == ["DJI_1_0001_D"]
    assert (
        client.post(
            "/api/project/sources", params=dict(id="demo"), json=dict(path=str(tmp_path / "missing"))
        ).status_code
        == 400
    )
    assert client.delete("/api/project/sources", params=dict(id="demo", path=str(src))).json()["sources"] == []
    assert client.get("/api/project", params=dict(id="nope")).status_code == 400


def test_sources_items_primary_and_image(client, tmp_path):
    from PIL import Image

    src = tmp_path / "s"
    src.mkdir()
    Image.new("RGB", (80, 60), "gray").save(src / "X_1.JPG")
    Image.new("RGB", (80, 60)).save(src / "X_1-Enhanced.jpg")
    client.post("/api/projects", json=dict(name="Demo", sources=[str(src)]))
    (s,) = client.get("/api/project/sources", params=dict(id="demo")).json()
    assert s["stills"] == 1
    (it,) = client.get("/api/project/items", params=dict(id="demo", source=s["id"])).json()
    assert it["primary"] == "jpeg"
    assert (
        client.put(
            "/api/project/primary", params=dict(id="demo", item=it["id"]), json=dict(variant="enhanced.jpg")
        ).json()["primary"]
        == "enhanced.jpg"
    )
    assert (
        client.put("/api/project/primary", params=dict(id="demo", item=it["id"]), json=dict(variant="zzz")).status_code
        == 400
    )
    r = client.get("/api/project/image", params=dict(id="demo", item=it["id"], transform="level", px=64))
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert (
        client.get("/api/project/transform", params=dict(id="demo", item=it["id"], transform="level")).json()["applied"]
        is False
    )
    import time

    cq = dict(id="demo", item=it["id"], x=0.3, y=0.6, size=32, zoom=3)
    c = client.get("/api/project/crop", params=cq)
    assert c.status_code in (200, 202) and (
        c.status_code == 200 or c.json()["state"] in ("running", "ready")
    )  # 202 while the full-size copy is being made
    for _ in range(100):
        if client.get("/api/project/fullres", params=dict(id="demo", item=it["id"])).json()["state"] == "ready":
            break
        time.sleep(0.05)
    c = client.get("/api/project/crop", params=cq)
    assert c.status_code == 200 and c.headers["content-type"] == "image/jpeg"
    assert client.get("/api/project/crop", params=dict(id="demo", item="nope~x")).status_code == 400
    assert client.get("/api/project/image", params=dict(id="demo", item="nope~x")).status_code == 400
    assert client.get("/api/project/item", params=dict(id="demo", item=it["id"])).json()["stem"] == "X_1"


def test_settings_roundtrip(client, tmp_path):
    assert client.put("/api/settings", json=dict(projects_dir="relative")).status_code == 400
    assert client.get("/api/settings").json()["projects_dir"] == str(tmp_path / "projects")


def test_the_visual_report_is_served_from_its_folder(client, tmp_path, monkeypatch):
    rep = tmp_path / "rep"
    (rep / "assets").mkdir(parents=True)
    (rep / "index.html").write_text("<h1>report</h1>")
    (rep / "assets" / "a.jpg").write_bytes(b"jpg")
    (tmp_path / "secret.txt").write_text("no")
    monkeypatch.setenv("STRATASTILLS_REPORT", str(tmp_path / "none"))
    assert (
        client.get("/api/report").json() == dict(available=False, built=None, neutral=False, auto=False)
        and client.get("/report/").status_code == 404
    )
    (rep / "auto").mkdir()
    (rep / "auto" / "index.html").write_text("a")
    monkeypatch.setenv("STRATASTILLS_REPORT", str(rep))
    j = client.get("/api/report").json()
    assert j["available"] is True and j["auto"] is True and j["neutral"] is False
    assert (
        client.get("/report/").text == "<h1>report</h1>"
        and client.get("/report/assets/a.jpg").content == b"jpg"
        and client.get("/report", follow_redirects=False).status_code in (302, 307)
    )
    assert (
        client.get("/report/../secret.txt").status_code == 404
        and client.get("/report/%2e%2e/secret.txt").status_code == 404
    )


def test_report_flags_are_kept_per_report_and_picture(client, tmp_path, monkeypatch):
    monkeypatch.setenv("STRATASTILLS_REPORT_FLAGS", str(tmp_path / "flags"))
    assert (
        client.get("/api/report-flags/auto").json() == {} and client.get("/api/report-flags/other").status_code == 404
    )
    r = client.put(
        "/api/report-flags/auto/2023-12-16_DSC05595",
        json=dict(rating="bad", reasons=["too red / warm"], note="skin ok, room red", status="deferred"),
    )
    assert (
        r.status_code == 200
        and r.json()["status"] == "deferred"
        and r.json()["rating"] == "bad"
        and r.json()["first"]
        and r.json()["at"]
    )
    assert (
        client.get("/api/report-flags/auto").json()["2023-12-16_DSC05595"]["reasons"] == ["too red / warm"]
        and client.get("/api/report-flags/neutral").json() == {}
    )
    assert client.put("/api/report-flags/auto/..%2Fx", json=dict(flag=True)).status_code in (400, 404, 405)
    assert client.put("/api/report-flags/auto/2023-12-16_DSC05595", json=dict(rating="nonsense")).json() == {}
    assert client.get("/api/report-flags/auto").json() == {}


def test_the_arena_offers_blind_pairs_records_votes_and_rates(client, tmp_path, monkeypatch):
    arena = tmp_path / "arena"
    (arena / "img").mkdir(parents=True)
    monkeypatch.setenv("STRATASTILLS_ARENA", str(arena))
    assert client.get("/api/arena/next").status_code == 404 and client.get("/arena/").status_code == 200
    for n in ("lr", "auto1"):
        (arena / "img" / f"{n}.jpg").write_bytes(b"jpg-" + n.encode())
    h = {n: hashlib.sha1(b"jpg-" + n.encode()).hexdigest()[:16] for n in ("lr", "auto1")}  # as build_arena hashes them
    pool = {"p1": {n: {"file": f"img/{n}.jpg", "hash": h[n]} for n in h}}
    (arena / "pool.json").write_text(json.dumps(pool))
    p = client.get("/api/arena/next").json()
    assert "lr" not in str(p) and "auto1" not in str(p)
    got = {client.get("/arena/" + p["left"]).content, client.get("/arena/" + p["right"]).content}
    assert got == {b"jpg-lr", b"jpg-auto1"}
    assert (
        client.post("/api/arena/vote", json=dict(token=p["token"], choice="left", ms=900)).json()["votes"] == 1
        and client.post("/api/arena/vote", json=dict(token=p["token"], choice="left")).status_code == 404
    )
    q = client.get("/api/arena/next").json()
    assert client.post("/api/arena/vote", json=dict(token=q["token"], choice="sideways")).status_code == 400
    r = client.get("/api/arena/ratings").json()
    assert (
        r["votes"] == 1 and {v["name"] for v in r["variants"]} == {"lr", "auto1"} and r["head_to_head"][0]["ties"] == 0
    )
    assert client.post("/api/arena/undo").json()["votes"] == 0 and client.post("/api/arena/undo").status_code == 404
    (arena / "sets").mkdir()
    (arena / "sets" / "hard.json").write_text('["p1"]')
    (arena / "sets" / "none.json").write_text('["p9"]')
    assert (
        client.get("/api/arena/next?set=hard").status_code == 200
        and client.get("/api/arena/next?set=none").status_code == 404
        and client.get("/api/arena/next?set=missing").status_code == 404
    )


def test_a_vote_remembers_the_pictures_shown_and_stops_counting_when_one_is_remade(client, tmp_path, monkeypatch):
    arena = tmp_path / "arena"
    (arena / "img").mkdir(parents=True)
    monkeypatch.setenv("STRATASTILLS_ARENA", str(arena))
    for n in ("lr", "auto1"):
        (arena / "img" / f"{n}.jpg").write_bytes(b"jpg-" + n.encode())
    pool = {"p1": {"lr": {"file": "img/lr.jpg", "hash": "a"}, "auto1": {"file": "img/auto1.jpg", "hash": "b"}}}
    (arena / "pool.json").write_text(json.dumps(pool))
    p = client.get("/api/arena/next").json()
    client.post("/api/arena/vote", json=dict(token=p["token"], choice="left"))
    v = json.loads((arena / "votes.jsonl").read_text())
    assert {v["ha"], v["hb"]} == {"a", "b"} and pool["p1"][v["a"]]["hash"] == v["ha"]
    pool["p1"]["auto1"].update(hash="c", same=["b"])  # remade, judged the same picture
    (arena / "pool.json").write_text(json.dumps(pool))
    assert client.get("/api/arena/ratings").json()["votes"] == 1
    pool["p1"]["auto1"].update(hash="d", same=[], changed=1.0)  # remade with a real change
    (arena / "pool.json").write_text(json.dumps(pool))
    r = client.get("/api/arena/ratings").json()
    assert r["votes"] == 0 and r["superseded_votes"] == 1


def test_a_vote_is_for_the_picture_served_even_when_it_was_remade_after_the_pair_was_drawn(
    client, tmp_path, monkeypatch
):
    arena = tmp_path / "arena"
    (arena / "img").mkdir(parents=True)
    monkeypatch.setenv("STRATASTILLS_ARENA", str(arena))
    for n in ("lr", "auto1"):
        (arena / "img" / f"{n}.jpg").write_bytes(b"jpg-" + n.encode())
    pool = {"p1": {"lr": {"file": "img/lr.jpg", "hash": "a"}, "auto1": {"file": "img/auto1.jpg", "hash": "b"}}}
    (arena / "pool.json").write_text(json.dumps(pool))
    p = client.get("/api/arena/next").json()
    (arena / "img" / "auto1.jpg").write_bytes(b"remade")  # rebuilt between /next and the image request
    for side in ("left", "right"):
        client.get("/arena/" + p[side])
    client.post("/api/arena/vote", json=dict(token=p["token"], choice="tie"))
    v = json.loads((arena / "votes.jsonl").read_text())
    shown = {v["a"]: v["ha"], v["b"]: v["hb"]}
    assert (
        shown["auto1"] == hashlib.sha1(b"remade").hexdigest()[:16]
        and shown["lr"] == hashlib.sha1(b"jpg-lr").hexdigest()[:16]
    )


def test_arena_alerts_hide_a_version_until_resolved_and_are_listed_with_metadata(client, tmp_path, monkeypatch):
    arena = tmp_path / "arena"
    (arena / "img").mkdir(parents=True)
    monkeypatch.setenv("STRATASTILLS_ARENA", str(arena))
    for n in ("lr", "auto1"):
        (arena / "img" / f"{n}.jpg").write_bytes(b"jpg-" + n.encode())
    (arena / "pool.json").write_text(
        '{"p1": {"lr": {"file": "img/lr.jpg", "hash": "a"}, "auto1": {"file": "img/auto1.jpg", "hash": "b", "key": "k1"}}}'
    )
    p = client.get("/api/arena/next").json()
    assert client.post("/api/arena/alert", json=dict(token=p["token"], side="sideways")).status_code == 400
    assert client.get("/arena/" + p["left"]).content in (b"jpg-lr", b"jpg-auto1")
    a = client.post("/api/arena/alert", json=dict(token=p["token"], side="left", note=" banding ", ms=1200)).json()
    assert (
        client.post("/api/arena/alert", json=dict(token=p["token"], side="left")).json()["id"] == a["id"]
        and client.get("/api/arena/next").status_code == 404
    )  # pressing again does not double it; only one version left: no pair
    assert (
        client.post("/api/arena/vote", json=dict(token=p["token"], choice="left")).json()["alerted"]
        and client.get("/api/arena/ratings").json()["votes"] == 0
    )  # a pair with an open alert is not a vote
    row = client.get("/api/alerts").json()["alerts"][0]
    assert (
        row["id"] == a["id"]
        and row["note"] == "banding"
        and row["pic"] == "p1"
        and row["ms"] == 1200
        and not row["resolved"]
        and row["pool"]["file"].startswith("img/")
    )
    assert (
        client.get("/api/alerts/image/" + a["id"]).status_code == 200 and "Arena flags" in client.get("/alerts/").text
    )
    assert client.post("/api/alerts/status", json=dict(id="nope")).status_code == 404
    client.post("/api/alerts/status", json=dict(id=a["id"], resolved=True))
    assert (
        client.get("/api/alerts").json()["alerts"][0]["resolved"] and client.get("/api/arena/next").status_code == 200
    )
    p = client.get("/api/arena/next").json()
    b = client.post("/api/arena/alert", json=dict(token=p["token"], side="right")).json()
    assert client.get("/api/arena/next").status_code == 404
    assert (
        client.post("/api/arena/alert/clear", json=dict(token=p["token"], side="right")).json()["cleared"] == 1
        and client.get("/api/arena/next").status_code == 200
    )  # pressed again: undone, the version is offered again


def test_browser_endpoints(client, tmp_path, monkeypatch):
    from PIL import Image

    day = tmp_path / "raw" / "2019-03-02"
    day.mkdir(parents=True)
    Image.new("RGB", (400, 300), (10, 200, 10)).save(day / "IMG_0001.JPG")
    Image.new("RGB", (400, 300), (10, 20, 200)).save(day / "IMG_0002.JPG")
    monkeypatch.setenv("STRATASTILLS_SOURCES", str(tmp_path / "raw"))
    monkeypatch.setenv("STRATASTILLS_THUMBS", str(tmp_path / "thumbs"))
    assert client.get("/api/browse/folders").json()[0]["key"] == "raw"
    assert client.get("/api/browse/folders", params=dict(folder="raw")).json()[0]["pictures"] == 2
    assert client.get("/api/browse/folders", params=dict(folder="raw/..")).status_code == 400
    pics = client.get("/api/browse/pictures", params=dict(folder="raw/2019-03-02")).json()["pictures"]
    assert [p["stem"] for p in pics] == ["IMG_0001", "IMG_0002"]
    r = client.get("/api/browse/thumb", params=dict(folder="raw/2019-03-02", stem="IMG_0001", source="jpeg", px=160))
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get("/api/browse/thumb", params=dict(folder="raw/2019-03-02", stem="x")).status_code == 400
    r = client.post("/api/browse/add", json=dict(name="Picks", pictures={"raw/2019-03-02": ["IMG_0002"]})).json()
    assert r == dict(project="picks", sources=0, pictures=1, unchanged=0)
    src = client.get("/api/project/sources", params=dict(id="picks")).json()[0]
    assert src["stills"] == 1 and src["picked"] == 1
    assert client.post("/api/browse/add", json=dict(folders=["raw/2019-03-02"])).status_code == 400
