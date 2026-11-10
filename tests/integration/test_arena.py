import json
import os

from stratastills import arena_store


def test_arena_integration(client, tmp_path, monkeypatch):
    """Integration test for the arena API endpoints."""
    # Set up the arena directory
    arena_dir = tmp_path / "arena"
    arena_dir.mkdir()
    monkeypatch.setenv("STRATASTILLS_ARENA", str(arena_dir))

    # Test that next fails when empty
    res = client.get("/api/arena/next")
    assert res.status_code == 404
    assert "nothing to rate yet" in res.json()["detail"]

    # Add a mock pool
    pool = {
        "pic1": {
            "variant1": {"file": "variant1.jpg", "hash": "h1"},
            "variant2": {"file": "variant2.jpg", "hash": "h2"}
        }
    }
    with open(arena_dir / "pool.json", "w") as f:
        json.dump(pool, f)

    # Create the picture files
    with open(arena_dir / "variant1.jpg", "wb") as f:
        f.write(b"pic1")
    with open(arena_dir / "variant2.jpg", "wb") as f:
        f.write(b"pic2")

    # Test /arena/ next endpoint returns token
    res = client.get("/api/arena/next")
    assert res.status_code == 200
    j = res.json()
    token = j["token"]
    assert token

    # Test image retrieval
    res_left = client.get(f"/arena/img/{token}/left")
    assert res_left.status_code == 200
    res_right = client.get(f"/arena/img/{token}/right")
    assert res_right.status_code == 200

    # Test vote
    res_vote = client.post("/api/arena/vote", json={"token": token, "choice": "left", "ms": 100})
    assert res_vote.status_code == 200
    assert res_vote.json()["votes"] == 1

    # Verify the vote was recorded
    votes = arena_store.get().votes()
    assert len(votes) == 1
    assert votes[0]["winner"] == "a" or votes[0]["winner"] == "b"

    # Test undo
    res_undo = client.post("/api/arena/undo")
    assert res_undo.status_code == 200
    assert len(arena_store.get().votes()) == 0

def test_arena_alerts_and_ratings(client, tmp_path, monkeypatch):
    """Test arena alerts and ratings API."""
    arena_dir = tmp_path / "arena"
    arena_dir.mkdir()
    monkeypatch.setenv("STRATASTILLS_ARENA", str(arena_dir))

    pool = {
        "pic1": {
            "variant1": {"file": "variant1.jpg", "hash": "h1"},
            "variant2": {"file": "variant2.jpg", "hash": "h2"}
        }
    }
    with open(arena_dir / "pool.json", "w") as f:
        json.dump(pool, f)

    with open(arena_dir / "variant1.jpg", "wb") as f:
        f.write(b"pic1")
    with open(arena_dir / "variant2.jpg", "wb") as f:
        f.write(b"pic2")

    # Start a pair
    res = client.get("/api/arena/next")
    assert res.status_code == 200
    token = res.json()["token"]

    # Issue an alert
    alert_res = client.post("/api/arena/alert", json={"token": token, "side": "left", "note": "Too dark", "ms": 100})
    assert alert_res.status_code == 200
    aid = alert_res.json()["id"]

    # Read alerts list
    list_res = client.get("/api/alerts")
    assert list_res.status_code == 200
    alerts = list_res.json()["alerts"]
    assert len(alerts) == 1
    assert alerts[0]["id"] == aid

    # Try fetching the image
    img_res = client.get(f"/api/alerts/image/{aid}")
    assert img_res.status_code == 200

    # Try fetching with variant override
    img_res = client.get(f"/api/alerts/image/{aid}?variant=variant2")
    assert img_res.status_code == 200

    # Try setting status
    status_res = client.post("/api/alerts/status", json={"id": aid, "status": "investigating"})
    assert status_res.status_code == 200

    # Try setting status note and resolving
    status_res = client.post("/api/alerts/status", json={"id": aid, "note": "updated note", "resolved": True})
    assert status_res.status_code == 200

    # Test clearing the alert
    clear_res = client.post("/api/arena/alert/clear", json={"token": token, "side": "left"})
    assert clear_res.status_code == 200

    # Test vote with skip and check alerts again
    res = client.get("/api/arena/next")
    token2 = res.json()["token"]

    skip_res = client.post("/api/arena/vote", json={"token": token2, "choice": "skip", "ms": 10})
    assert skip_res.status_code == 200
    assert skip_res.json()["skipped"]

    # Test ratings
    ratings_res = client.get("/api/arena/ratings")
    assert ratings_res.status_code == 200
    assert "variants" in ratings_res.json()

def test_arena_pages(client, tmp_path, monkeypatch):
    """Test HTML pages of arena."""
    arena_dir = tmp_path / "arena"
    arena_dir.mkdir()
    monkeypatch.setenv("STRATASTILLS_ARENA", str(arena_dir))

    # Test arena page
    assert client.get("/arena", follow_redirects=False).status_code == 307
    res = client.get("/arena/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]

    # Test alerts page
    assert client.get("/alerts", follow_redirects=False).status_code == 307
    res = client.get("/alerts/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]

    # Test flagged page
    assert client.get("/flagged", follow_redirects=False).status_code == 307
    res = client.get("/flagged/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]

def test_report_flags(client, tmp_path, monkeypatch):
    """Test report flags."""
    flags_dir = tmp_path / "flags"
    flags_dir.mkdir()
    monkeypatch.setenv("STRATASTILLS_REPORT_FLAGS", str(flags_dir))

    # Test reading missing flags
    res = client.get("/api/report-flags/neutral")
    assert res.status_code == 200
    assert res.json() == {}

    # Test bad report
    assert client.get("/api/report-flags/badreport").status_code == 404

    # Test put flag
    res = client.put("/api/report-flags/neutral/mypic.jpg", json={
        "rating": "bad",
        "reasons": ["dark"],
        "note": "Too dark",
        "status": "open"
    })
    assert res.status_code == 200

    # Read back flag
    res = client.get("/api/report-flags/neutral")
    assert res.status_code == 200
    j = res.json()
    assert "mypic.jpg" in j
    assert j["mypic.jpg"]["rating"] == "bad"
    assert j["mypic.jpg"]["reasons"] == ["dark"]
    assert j["mypic.jpg"]["note"] == "Too dark"
    assert j["mypic.jpg"]["status"] == "open"
    assert "first" in j["mypic.jpg"]
    assert "at" in j["mypic.jpg"]

    # Test invalid pic name
    res = client.put("/api/report-flags/neutral/bad!@#.jpg", json={"rating": "good"})
    assert res.status_code == 400

def test_report_flags_removal(client, tmp_path, monkeypatch):
    """Test report flags removal."""
    flags_dir = tmp_path / "flags"
    flags_dir.mkdir()
    monkeypatch.setenv("STRATASTILLS_REPORT_FLAGS", str(flags_dir))

    # Test put flag with empty values to remove
    client.put("/api/report-flags/neutral/mypic.jpg", json={
        "rating": "bad",
        "reasons": ["dark"],
        "note": "Too dark",
        "status": "open"
    })

    # Clear it
    res = client.put("/api/report-flags/neutral/mypic.jpg", json={
        "rating": "",
        "reasons": [],
        "note": "",
        "status": ""
    })
    assert res.status_code == 200

    # Read back flag
    res = client.get("/api/report-flags/neutral")
    assert res.status_code == 200
    j = res.json()
    assert "mypic.jpg" not in j

def test_arena_edge_cases(client, tmp_path, monkeypatch):
    """Test arena edge cases."""
    arena_dir = tmp_path / "arena"
    arena_dir.mkdir()
    monkeypatch.setenv("STRATASTILLS_ARENA", str(arena_dir))

    pool = {
        "pic1": {
            "variant1": {"file": "variant1.jpg", "hash": "h1"},
            "variant2": {"file": "variant2.jpg", "hash": "h2"}
        },
        "pic2": {
            "variant1": {"file": "variant1.jpg", "hash": "h1"},
            "variant2": {"file": "variant2.jpg", "hash": "h2"}
        }
    }
    with open(arena_dir / "pool.json", "w") as f:
        json.dump(pool, f)

    with open(arena_dir / "variant1.jpg", "wb") as f:
        f.write(b"pic1")
    with open(arena_dir / "variant2.jpg", "wb") as f:
        f.write(b"pic2")

    res = client.get("/api/arena/next")
    token = res.json()["token"]

    # Image not found edge case
    assert client.get(f"/arena/img/{token}/badside").status_code == 404

    # Image retrieval bad token
    assert client.get("/arena/img/badtoken/left").status_code == 404

    # Alert on bad token
    assert client.post("/api/arena/alert", json={"token": "badtoken", "side": "left", "note": "Too dark", "ms": 100}).status_code == 404

    # Alert on bad side
    assert client.post("/api/arena/alert", json={"token": token, "side": "badside", "note": "Too dark", "ms": 100}).status_code == 400

    # Issue alert to create an alerted state
    alert_res = client.post("/api/arena/alert", json={"token": token, "side": "left", "note": "Too dark", "ms": 100})
    aid = alert_res.json()["id"]

    # Second alert on same token and side should return same id
    alert_res_2 = client.post("/api/arena/alert", json={"token": token, "side": "left", "note": "Too dark", "ms": 100})
    assert alert_res_2.json()["id"] == aid

    # Try voting on a pair that has an open alert
    vote_res = client.post("/api/arena/vote", json={"token": token, "choice": "left", "ms": 100})
    assert vote_res.status_code == 200
    assert vote_res.json().get("alerted") is True

    # Missing alert
    assert client.post("/api/alerts/status", json={"id": "badid", "status": "investigating"}).status_code == 404

    # Bad status
    assert client.post("/api/alerts/status", json={"id": aid, "status": "badstatus"}).status_code == 400

    # No vote to undo
    assert client.post("/api/arena/undo").status_code == 404

    # Bad vote token
    assert client.post("/api/arena/vote", json={"token": "badtoken", "choice": "left", "ms": 100}).status_code == 404

    # Bad vote choice (must get a fresh pair as previous was alerted)
    res2 = client.get("/api/arena/next")
    token2 = res2.json()["token"]
    assert client.post("/api/arena/vote", json={"token": token2, "choice": "badchoice", "ms": 100}).status_code == 400

    # Image retrieval missing file
    os.remove(arena_dir / "variant1.jpg")
    assert client.get(f"/arena/img/{token2}/left").status_code == 404

    # Alert image retrieval bad id
    assert client.get("/api/alerts/image/badid").status_code == 404

    # Alert image retrieval missing file for one of the variants in alert
    with open(arena_dir / "pool.json", "w") as f:
        json.dump({"pic1": {"variant1": {"file": "missing.jpg", "hash": "h1"}, "variant2": {"file": "variant2.jpg", "hash": "h2"}}}, f)

    assert client.get(f"/api/alerts/image/{aid}?variant=variant1").status_code == 404

def test_arena_priorities_and_sets(client, tmp_path, monkeypatch):
    """Test arena priorities and sets."""
    arena_dir = tmp_path / "arena"
    arena_dir.mkdir()
    monkeypatch.setenv("STRATASTILLS_ARENA", str(arena_dir))

    pool = {
        "pic1": {
            "variant1": {"file": "variant1.jpg", "hash": "h1"},
            "variant2": {"file": "variant2.jpg", "hash": "h2"}
        }
    }
    with open(arena_dir / "pool.json", "w") as f:
        json.dump(pool, f)

    with open(arena_dir / "variant1.jpg", "wb") as f:
        f.write(b"pic1")
    with open(arena_dir / "variant2.jpg", "wb") as f:
        f.write(b"pic2")

    # Add priorities.json
    priorities = {
        "interest": {"variant1": 2.0},
        "faces": ["pic1"],
        "dist": {"pic1": {"variant1|variant2": 10.0}}
    }
    with open(arena_dir / "priorities.json", "w") as f:
        json.dump(priorities, f)

    # Add set
    sets_dir = arena_dir / "sets"
    sets_dir.mkdir()
    with open(sets_dir / "myset.json", "w") as f:
        json.dump(["pic1"], f)

    # Test /arena/ next endpoint returns token
    res = client.get("/api/arena/next?set=myset")
    assert res.status_code == 200
    token = res.json()["token"]
    assert token

    # Test invalid set
    res = client.get("/api/arena/next?set=badset")
    assert res.status_code == 404

    # Test invalid set name
    res = client.get("/api/arena/next?set=bad!@#set")
    assert res.status_code == 404

    # Test priorities read with bad format
    with open(arena_dir / "priorities.json", "w") as f:
        json.dump([], f)
    res = client.get("/api/arena/next?set=myset")
    assert res.status_code == 200

def test_arena_priorities_dist_list(client, tmp_path, monkeypatch):
    """Test arena priorities with dist as a list."""
    arena_dir = tmp_path / "arena"
    arena_dir.mkdir()
    monkeypatch.setenv("STRATASTILLS_ARENA", str(arena_dir))

    pool = {
        "pic1": {
            "variant1": {"file": "variant1.jpg", "hash": "h1"},
            "variant2": {"file": "variant2.jpg", "hash": "h2"}
        }
    }
    with open(arena_dir / "pool.json", "w") as f:
        json.dump(pool, f)

    with open(arena_dir / "variant1.jpg", "wb") as f:
        f.write(b"pic1")
    with open(arena_dir / "variant2.jpg", "wb") as f:
        f.write(b"pic2")

    # Add priorities.json
    priorities = {
        "dist": {"pic1": {"variant1|variant2": [5.0, ["h1", "h2"]]}}
    }
    with open(arena_dir / "priorities.json", "w") as f:
        json.dump(priorities, f)

    # Test /arena/ next endpoint returns token
    res = client.get("/api/arena/next")
    assert res.status_code == 200
    token = res.json()["token"]
    assert token

def test_arena_ratings_ties_and_head_to_head(client, tmp_path, monkeypatch):
    """Test arena ratings with ties and head-to-head records."""
    arena_dir = tmp_path / "arena"
    arena_dir.mkdir()
    monkeypatch.setenv("STRATASTILLS_ARENA", str(arena_dir))

    pool = {
        "pic1": {
            "variant1": {"file": "variant1.jpg", "hash": "h1"},
            "variant2": {"file": "variant2.jpg", "hash": "h2"},
            "variant3": {"file": "variant3.jpg", "hash": "h3"},
        }
    }
    with open(arena_dir / "pool.json", "w") as f:
        json.dump(pool, f)

    with open(arena_dir / "variant1.jpg", "wb") as f:
        f.write(b"pic1")
    with open(arena_dir / "variant2.jpg", "wb") as f:
        f.write(b"pic2")
    with open(arena_dir / "variant3.jpg", "wb") as f:
        f.write(b"pic3")

    for i in range(5):
        res = client.get("/api/arena/next")
        token = res.json()["token"]
        client.post("/api/arena/vote", json={"token": token, "choice": "tie", "ms": 100})

    for i in range(5):
        res = client.get("/api/arena/next")
        token = res.json()["token"]
        client.post("/api/arena/vote", json={"token": token, "choice": "right", "ms": 100})

    ratings_res = client.get("/api/arena/ratings")
    assert ratings_res.status_code == 200
    j = ratings_res.json()
    assert "variants" in j
    assert "head_to_head" in j
    assert len(j["head_to_head"]) > 0

def test_arena_ratings_math(client, tmp_path, monkeypatch):
    """Test arena ratings math endpoints with more votes."""
    arena_dir = tmp_path / "arena"
    arena_dir.mkdir()
    monkeypatch.setenv("STRATASTILLS_ARENA", str(arena_dir))

    pool = {
        "pic1": {
            "variant1": {"file": "variant1.jpg", "hash": "h1"},
            "variant2": {"file": "variant2.jpg", "hash": "h2"},
            "variant3": {"file": "variant3.jpg", "hash": "h3"},
        }
    }
    with open(arena_dir / "pool.json", "w") as f:
        json.dump(pool, f)

    with open(arena_dir / "variant1.jpg", "wb") as f:
        f.write(b"pic1")
    with open(arena_dir / "variant2.jpg", "wb") as f:
        f.write(b"pic2")
    with open(arena_dir / "variant3.jpg", "wb") as f:
        f.write(b"pic3")

    for i in range(10):
        res = client.get("/api/arena/next")
        token = res.json()["token"]
        client.post("/api/arena/vote", json={"token": token, "choice": "left", "ms": 100})

    ratings_res = client.get("/api/arena/ratings")
    assert ratings_res.status_code == 200
    assert "variants" in ratings_res.json()
