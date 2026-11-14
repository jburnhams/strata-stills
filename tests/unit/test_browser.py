import io
import os
import struct

import pytest
from PIL import Image

from stratastills import browser as B
from stratastills import images
from stratastills import projects as P


def fake_raw(path, w=400, h=300, orientation=1, colour=(200, 40, 40)):
    """A TIFF-based 'raw' (an .ARW name): IFD0 with Orientation and a JPEGInterchangeFormat preview, as Sony writes PreviewImage; no sensor data (browser and nudity probe only read the preview)."""
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, "JPEG", quality=90)
    jpeg = buf.getvalue()
    tags = [(256, 4, 1, w), (257, 4, 1, h), (274, 3, 1, orientation), (513, 4, 1, 0), (514, 4, 1, len(jpeg))]
    ifd_size = 2 + 12 * len(tags) + 4
    data_off = 8 + ifd_size
    out = b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", len(tags))
    for tag, typ, n, val in tags:
        val = data_off if tag == 513 else val
        out += struct.pack("<HHI", tag, typ, n) + (struct.pack("<HH", val, 0) if typ == 3 else struct.pack("<I", val))
    out += struct.pack("<I", 0) + jpeg
    with open(path, "wb") as f:
        f.write(out)
    return path


@pytest.fixture
def roots(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    day = raw / "2019" / "2019-03-02"
    day.mkdir(parents=True)
    (raw / "2019" / "2019-03-03" / "deeper").mkdir(parents=True)
    fake_raw(str(day / "DSC00001.ARW"))
    Image.new("RGB", (400, 300), (10, 200, 10)).save(day / "DSC00001.JPG")
    fake_raw(str(day / "DSC00002.ARW"), orientation=6)
    Image.new("RGB", (64, 48)).save(raw / "2019" / "2019-03-03" / "deeper" / "IMG_1.JPG")
    (day / "clip.MP4").write_bytes(b"x")
    monkeypatch.setenv("STRATASTILLS_SOURCES", str(raw))
    monkeypatch.setenv("STRATASTILLS_THUMBS", str(tmp_path / "thumbs"))
    B._scans.clear()
    return raw


def test_folder_keys_stay_inside_the_roots(roots):
    assert B.locate("raw/2019") == str(roots / "2019")
    assert B.key_of(str(roots / "2019" / "2019-03-02")) == "raw/2019/2019-03-02" and B.key_of("/elsewhere") is None
    for bad in ("", "raw/../..", "nope/2019", "raw/2019/missing"):
        with pytest.raises(B.BrowseError):
            B.locate(bad)


def test_tree_counts_folders_and_pictures(roots):
    assert B.folders() == [dict(key="raw", name="raw", folders=1, pictures=0)]
    assert B.folders("raw/2019") == [
        dict(key="raw/2019/2019-03-02", name="2019-03-02", folders=0, pictures=2),  # a raw and its JPEG are one
        dict(key="raw/2019/2019-03-03", name="2019-03-03", folders=1, pictures=0),
    ]


def test_pictures_list_sources_best_first_and_leave_out_videos(roots):
    r = B.pictures("raw/2019/2019-03-02")
    assert r["checking"] == 0
    a, b = r["pictures"]
    assert a["stem"] == "DSC00001" and a["files"] == ["DSC00001.ARW", "DSC00001.JPG"] and a["projects"] == []
    assert [s["id"] for s in a["sources"]] == ["embedded", "jpeg", "raw"]
    assert [s["id"] for s in b["sources"]] == ["embedded", "raw"]


def test_embedded_preview_is_read_by_offset_and_turned_upright(roots):
    data, orient = images.embedded_preview(str(roots / "2019" / "2019-03-02" / "DSC00002.ARW"))
    assert data[:2] == b"\xff\xd8" and orient == 6
    assert images.preview_bgr(str(roots / "2019" / "2019-03-02" / "DSC00002.ARW")).shape[:2] == (400, 300)
    assert images.embedded_preview(str(roots / "2019" / "2019-03-03" / "deeper" / "IMG_1.JPG")) is None


def test_thumbnails_are_made_once_and_cached(roots, monkeypatch):
    k = "raw/2019/2019-03-02"
    p = B.thumb(k, "DSC00002", "embedded", 300)
    with Image.open(p) as im:
        assert im.size == (240, 320)  # rounded up to 320, portrait after the turn
    assert p.startswith(os.environ["STRATASTILLS_THUMBS"])
    with Image.open(B.thumb(k, "DSC00001", "jpeg", 100)) as im:
        assert im.size == (160, 120) and im.getpixel((80, 60))[1] > 150  # the camera JPEG, not the red preview
    monkeypatch.setattr(B, "_make", lambda *a: pytest.fail("made again"))
    assert B.thumb(k, "DSC00002", "embedded", 320) == p
    with pytest.raises(B.BrowseError):
        B.thumb(k, "DSC00002", "jpeg", 320)
    with pytest.raises(B.BrowseError):
        B.thumb(k, "NOPE", "embedded", 320)


def test_add_pictures_then_whole_folders_to_a_new_and_an_existing_project(roots):
    k = "raw/2019/2019-03-02"
    r = B.add(name="Trip", pictures={k: ["DSC00002"]})
    assert r == dict(project="trip", sources=0, pictures=1, unchanged=0)
    src = P.load("trip")["sources"][0]
    assert src["stems"] == ["DSC00002"] and src["rel"] == "2019/2019-03-02"
    assert [i["stem"] for i in P.source_items("trip", P.source_id(src["path"]))] == ["DSC00002"]
    assert B.pictures(k)["pictures"][1]["projects"] == ["trip"] and B.pictures(k)["pictures"][0]["projects"] == []
    assert B.add(project="trip", pictures={k: ["DSC00002", "DSC00001"]})["pictures"] == 1
    assert B.add(project="trip", folders=[k]) == dict(project="trip", sources=1, pictures=0, unchanged=0)
    assert "stems" not in P.load("trip")["sources"][0]  # the whole folder now
    assert B.add(project="trip", pictures={k: ["DSC00001"]})["unchanged"] == 1
    r = B.add(project="trip", folders=["raw/2019"], subfolders=True)
    assert r["sources"] == 1 and r["unchanged"] == 1  # 2019-03-03/deeper added, 2019-03-02 already whole
    with pytest.raises(B.BrowseError):
        B.add(folders=[k])
    with pytest.raises(P.ProjectError):
        B.add(project="nope", folders=[k])


def test_unchecked_pictures_wait_for_the_nudity_filter(roots, monkeypatch):
    from stratastills import nsfw

    monkeypatch.setenv("STRATASTILLS_NSFW", "1")
    nsfw.reset()
    started = []
    monkeypatch.setattr(B, "_check_later", lambda path, shots: started.append(len(shots)))
    r = B.pictures("raw/2019/2019-03-02")
    assert r == dict(pictures=[], checking=2) and started == [2]
    flagged = {"DSC00001"}
    monkeypatch.setattr(nsfw, "shot_known", lambda sh: True)
    monkeypatch.setattr(nsfw, "shot_flagged", lambda sh: {"path": sh.raw} if sh.stem in flagged else None)
    assert [p["stem"] for p in B.pictures("raw/2019/2019-03-02")["pictures"]] == ["DSC00002"]
    with pytest.raises(B.BrowseError):
        B.thumb("raw/2019/2019-03-02", "DSC00001", "embedded", 320)


def test_a_missing_detector_is_told_not_checked_for_ever(roots, monkeypatch):
    from stratastills import nsfw

    monkeypatch.setenv("STRATASTILLS_NSFW", "1")
    nsfw.reset()
    monkeypatch.setattr(nsfw, "shot_known", lambda sh: False)

    def unavailable(sh):
        raise nsfw.NsfwUnavailable("the nudity filter needs nudenet")

    monkeypatch.setattr(nsfw, "shot_flagged", unavailable)
    B._check_errors.clear()
    B._checking.clear()
    monkeypatch.setattr(B.threading, "Thread", lambda target, **kw: type("T", (), {"start": lambda self: target()})())
    assert B.pictures("raw/2019/2019-03-02")["checking"] == 3  # the video too: shot_known is faked
    with pytest.raises(B.BrowseError, match="nudenet"):
        B.pictures("raw/2019/2019-03-02")


def test_a_picture_pillow_cannot_open_is_read_by_the_app(roots, monkeypatch):
    import numpy as np

    day = roots / "2019" / "2019-03-02"
    Image.new("RGB", (40, 30)).save(day / "PANO_1.JPG")
    B._scans.clear()
    from PIL import Image as PI

    real = PI.open
    monkeypatch.setattr(
        PI,
        "open",
        lambda p, *a, **k: (
            (_ for _ in ()).throw(OSError("16 bit")) if str(p).endswith("PANO_1.JPG") else real(p, *a, **k)
        ),
    )
    monkeypatch.setattr(images, "load", lambda path, px: np.full((30, 40, 3), 200, np.uint8))
    with PI.open(B.thumb("raw/2019/2019-03-02", "PANO_1", "jpeg", 160)) as im:
        assert im.size == (40, 30)


def test_a_project_item_thumbnail_comes_from_the_shared_cache(roots, monkeypatch):
    k = "raw/2019/2019-03-02"
    B.add(name="Trip", folders=[k])
    src = P.load("trip")["sources"][0]
    iid = P.item_id(P.source_id(src["path"]), "DSC00002")
    made = B.thumb(k, "DSC00002", "embedded", 320)
    real = B._make
    monkeypatch.setattr(B, "_make", lambda *a: pytest.fail("made again"))
    assert B.item_thumb("trip", iid, 300) == made  # the Browser's thumbnail, reused
    P.set_primary("trip", P.item_id(P.source_id(src["path"]), "DSC00001"), "jpeg")
    monkeypatch.setattr(B, "_make", real)
    with Image.open(B.item_thumb("trip", P.item_id(P.source_id(src["path"]), "DSC00001"), 160)) as im:
        assert im.getpixel((80, 60))[1] > 150  # the chosen JPEG, not the raw's red preview
    assert B.item_thumb("trip", iid, 300, "raw") == made
    with pytest.raises(B.BrowseError):
        B.item_thumb("trip", iid, 300, "jpeg")  # DSC00002 has no JPEG
    with pytest.raises(P.ProjectError):
        B.item_thumb("trip", "nope~x")
