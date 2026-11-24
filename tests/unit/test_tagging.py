"""stratastills.tagging: the preview it sends, the context, the two calls and the merged analysis row (a fake strata-models client)."""

import cv2
import numpy as np
import pytest

from stratastills import tagging


def _jpeg(path, w=2000, h=1000):
    cv2.imwrite(str(path), np.full((h, w, 3), 128, np.uint8))


def test_preview_uses_the_camera_jpeg_and_caches(tmp_path):
    raw = tmp_path / "DSC1.ARW"
    raw.write_bytes(b"not a raw")
    _jpeg(tmp_path / "DSC1.JPG")
    p, key = tagging.preview(str(raw), str(tmp_path / "prev"))
    assert cv2.imread(p).shape[:2] == (512, 1024) and len(key) == 16
    assert tagging.preview(str(raw), str(tmp_path / "prev")) == (p, key)


def test_preview_without_a_picture_fails(tmp_path, monkeypatch):
    raw = tmp_path / "DSC2.ARW"
    raw.write_bytes(b"x")
    monkeypatch.setattr(tagging.images, "preview_bgr", lambda path, px: None)
    with pytest.raises(tagging.TagError, match="no camera picture"):
        tagging.preview(str(raw), str(tmp_path / "prev"))


def test_context_rounds_gps_and_handles_no_meta():
    assert tagging.context(None) == {}
    c = tagging.context(dict(model="NEX-5N", iso=100, gps_lat=51.123456, gps_lon=-0.98765, taken_at=None))
    assert c["camera"] == "NEX-5N" and c["gps"] == [51.1235, -0.9877] and c["taken"] is None
    assert tagging.context(dict(gps_lat=1.0, gps_lon=None))["gps"] is None


class FakeSM:
    class Unavailable(RuntimeError):
        pass

    def __init__(self, results):
        self.results, self.sent = results, []

    def submit(self, role, params, app, priority):
        self.sent.append((role, params, app, priority))
        return dict(id=len(self.sent))

    def wait_job(self, jid):
        return self.results[jid - 1]


def test_two_calls_merge_into_one_row():
    sm = FakeSM(
        [
            dict(status="done", result=dict(json=dict(indoor_outdoor="outdoor", aesthetic=4), model="qwen3.5-9b-vl")),
            dict(
                status="done",
                result=dict(json=dict(caption="A dog on a beach.", tags=["dog"]), model="claude-haiku-5-5-api"),
            ),
        ]
    )
    jobs = tagging.submit(sm, "/p.jpg", dict(camera="X"), "app")
    assert [s[0] for s in sm.sent] == ["photo-fields", "photo-caption"]
    assert (
        sm.sent[0][1] == dict(image="/p.jpg", preset="photo-fields-v2", context=dict(camera="X"), max_tokens=600)
        and sm.sent[0][3] == "batch"
    )
    r = tagging.row(7, "k" * 16, tagging.collect(sm, jobs))
    assert (r["kind"], r["producer"], r["producer_version"], r["label"], r["score"]) == (
        "vision_tags",
        "strata-models",
        tagging.VERSION,
        "A dog on a beach.",
        4.0,
    )
    assert (
        r["data"]["_models"] == {"photo-fields": "qwen3.5-9b-vl", "photo-caption": "claude-haiku-5-5-api"}
        and r["data"]["indoor_outdoor"] == "outdoor"
    )


def test_a_failed_call_or_an_answer_without_json_fails_the_photo():
    for bad in (dict(status="failed", error="no JSON answer"), dict(status="done", result=dict(text="hi"))):
        sm = FakeSM([dict(status="done", result=dict(json={})), bad])
        with pytest.raises(tagging.TagError):
            tagging.collect(sm, tagging.submit(sm, "/p.jpg", {}, "app"))
    assert tagging.row(1, "k", dict(caption="", aesthetic=True))["score"] is None


def test_retry_waits_out_a_restart_and_gives_up_in_the_end():
    sm, calls, slept = FakeSM([]), [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise sm.Unavailable("connection refused")
        return "ok"

    assert tagging.retry(sm, flaky, wait_s=5, sleep=slept.append) == "ok" and slept == [5, 5]

    def down():
        raise sm.Unavailable("down")

    with pytest.raises(sm.Unavailable):
        tagging.retry(sm, down, wait_s=10, give_up_s=20, sleep=lambda s: None)


def test_a_corrupt_camera_jpeg_falls_back_to_the_raw_preview_and_a_bad_raw_fails_only_its_photo(tmp_path, monkeypatch):
    raw = tmp_path / "DSC3.ARW"
    raw.write_bytes(b"x")
    (tmp_path / "DSC3.JPG").write_bytes(b"not a jpeg")
    monkeypatch.setattr(tagging.images, "preview_bgr", lambda path, px: np.full((10, 20, 3), 50, np.uint8))
    assert tagging.camera_picture(str(raw)).shape == (10, 20, 3)

    def boom(path, px):
        raise ValueError("bad raw")

    monkeypatch.setattr(tagging.images, "preview_bgr", boom)
    with pytest.raises(tagging.TagError, match="bad raw"):
        tagging.preview(str(raw), str(tmp_path / "prev"))
