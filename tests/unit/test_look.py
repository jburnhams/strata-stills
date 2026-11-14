"""stratastills.look: the feature row of a new picture, the frozen bundle, the versions it renders, and the pipeline's `look` stage (synthetic pictures and stub models: the real bundle is private)."""

import json
import os
import pickle

import numpy as np
import pytest

from stratastills import look
from stratastills import pipeline as PL
from stratastills.look import apply as A
from stratastills.look import face_first as FF
from stratastills.look import features as F


class Const:
    """A stand-in for a trained regressor: predicts the same value for every row."""

    def __init__(self, v):
        self.v = v

    def predict(self, X):
        return np.full(len(np.atleast_2d(X)), self.v, np.float32)


RECS = [
    {
        "IFD0:Model": "ILCE-7C",
        "ExifIFD:ISO": 100,
        "ExifIFD:DateTimeOriginal": "2020:05:01 10:30:00",
        "Composite:LensID": "E 16mm",
    },
    {
        "IFD0:Model": "ILCE-7C",
        "ExifIFD:ISO": 800,
        "ExifIFD:DateTimeOriginal": "2021:06:02 20:00:00",
        "Composite:LensID": "FE 28-60",
    },
    {
        "IFD0:Model": "NEX-5N",
        "ExifIFD:ISO": 200,
        "ExifIFD:DateTimeOriginal": "2012:03:11 13:12:00",
        "Composite:LensID": "E 16mm",
        "ExifIFD:LensInfo": "16 16 2.8 2.8",
    },
]


def test_meta_encoder_ranks_text_and_encodes_a_new_picture():
    enc = F.meta_encoder(RECS)
    assert "IFD0:Model" in enc["names"] and enc["ranks"]["IFD0:Model"] == {"ILCE-7C": 0, "NEX-5N": 1}
    assert "when:hour" in enc["names"] and "ExifIFD:LensInfo[0]" not in enc["names"]  # one value only: dropped
    row = F.meta_row({"IFD0:Model": "NEX-5N", "ExifIFD:ISO": 400, "Composite:LensID": "unknown lens"}, enc)
    named = dict(zip(enc["names"], row))
    assert named["IFD0:Model"] == 1 and named["ExifIFD:ISO"] == 400 and np.isnan(named["Composite:LensID"])


def test_clip_row_projects_and_softmaxes_each_group():
    rng = np.random.default_rng(1)
    mu, vt = rng.normal(size=8), np.linalg.qr(rng.normal(size=(8, 8)))[0][:3]
    basis = dict(mu=mu, vt=vt, texts=[("where", ["in", "out"], rng.normal(size=(2, 8)))])
    e = rng.normal(size=8)
    r = F.clip_row(e, basis)
    assert len(r) == 5 and np.allclose(r[:3], (e - mu) @ vt.T, atol=1e-5) and np.isclose(r[3:].sum(), 1)


def test_main_face_is_the_largest_real_one():
    fl = [
        dict(box=[0, 0, 0.3, 0.3], real=0.1, score=0.9),
        dict(box=[0, 0, 0.2, 0.2], real=0.9, score=0.9),
        dict(box=[0, 0, 0.1, 0.1], real=0.9, score=0.9),
    ]
    assert FF.main_face(fl)["box"][2] == 0.2 and FF.main_face([]) is None


@pytest.fixture
def stub_bundle(tmp_path, monkeypatch):
    """A bundle folder whose models are constants: the look model predicts the picture's own percentiles (so auto1 leaves the tone near as it is)."""
    names = ["a", "b"]
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(names=names, base_names=names, n_meta=0, n_clip=0, files={"x": "1"}))
    )
    pred = [40.0] * len(F.PCTS) + [40.0] + [0, 0, 20, 30, 20, 40] + [d for _, d in A.SLIDERS]
    pickle.dump([Const(v) for v in pred], open(tmp_path / "look.pkl", "wb"))
    pickle.dump([Const(0.0)] * 19, open(tmp_path / "tmodel.pkl", "wb"))
    pickle.dump([Const(0.0)] * 3, open(tmp_path / "ftarget.pkl", "wb"))
    monkeypatch.setenv("STRATASTILLS_LOOK", str(tmp_path))
    look._bundles.clear()
    yield look.bundle()
    look._bundles.clear()


def test_no_bundle_is_a_clear_refusal_and_its_own_version(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATASTILLS_LOOK", str(tmp_path / "none"))
    look._bundles.clear()
    with pytest.raises(look.LookUnavailable):
        look.bundle()
    assert look.version().endswith("-none")


def test_versions_render_on_a_picture(stub_bundle):
    rng = np.random.default_rng(0)
    neu = (rng.random((60, 80, 3)) * 200 + 20).astype(np.uint8)
    x = np.zeros(2, np.float32)
    for v in ("auto1", "auto2", "auto3"):
        out, did = look.render(v, neu, x, [], stub_bundle)
        assert out.shape == neu.shape and out.dtype == np.uint8 and "curve" in did
    fl = [
        dict(box=[0.3, 0.3, 0.3, 0.3], score=0.9, cam=[60.0, 10.0, 15.0], lm=[], pose=[0, 0, 0], real=0.9, kind="real")
    ]
    out, did = look.render("auto3", neu, x, fl, stub_bundle)
    assert "face_dL" in did
    with pytest.raises(ValueError):
        look.render("auto9", neu, x, [], stub_bundle)
    assert look.version() != f"{look.VERSION}-none"


def test_the_pipeline_has_a_look_stage_set_to_the_arena_leader(monkeypatch):
    monkeypatch.setattr(
        look, "ratings", lambda: {"auto5": dict(rating=1647, votes=241), "auto7": dict(rating=1700, votes=40)}
    )
    st = PL.normalize(PL.BEST)
    assert [s["stage"] for s in st][-2:] == ["look", "encode"] and st[-2]["variant"] == "auto7"
    monkeypatch.setattr(look, "ratings", lambda: {})
    assert PL.normalize(PL.BEST)[-2]["variant"] == look.FALLBACK == "auto5"
    assert "look" not in [s["stage"] for s in PL.get_preset("neutral")["stages"]]
    assert "look" not in [s["stage"] for s in PL.apply_options(st, dict(look="none"))]
    assert PL.apply_options(PL.get_preset("neutral")["stages"], dict(look="auto5"))[-2] == dict(
        stage="look", variant="auto5"
    )
    with pytest.raises(PL.P.ProjectError):
        PL.normalize([dict(stage="develop"), dict(stage="look", variant="auto9"), dict(stage="encode")])


def test_without_a_bundle_the_stage_leaves_the_picture_and_says_why(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATASTILLS_LOOK", str(tmp_path / "none"))
    look._bundles.clear()

    class St:
        rec = dict(steps=[])
        path = os.path.join(str(tmp_path), "x.jpg")
        say = staticmethod(lambda t: None)

    st = St()
    PL.run_look(st, dict(variant="auto7"))
    assert st.rec["look"]["applied"] is False and "no look bundle" in st.rec["steps"][0]


def _face_lm(cx=0.5, cy=0.5, r=0.15):
    """68 landmarks roughly where a face's are (jaw round the bottom, brows, eyes, nose, mouth), as fractions."""
    t = np.linspace(np.pi * 0.1, np.pi * 0.9, 17)
    jaw = np.stack([cx - r * np.cos(t), cy + r * np.sin(t) * 0.8], 1)
    brows = np.stack([np.linspace(cx - r * 0.8, cx + r * 0.8, 10), np.full(10, cy - r * 0.5)], 1)
    rest = np.stack(
        [np.linspace(cx - r * 0.6, cx + r * 0.6, 41), cy + np.tile([-0.25, -0.2, 0.0, 0.3], 11)[:41] * r], 1
    )
    return np.concatenate([jaw, brows, rest]).tolist()


class FakeSM:
    """strata_models_client as the look uses it: one face, its kind, and a CLIP embedding."""

    def __init__(self, emb):
        self.emb = emb

    def available(self):
        return True

    def faces(self, path, app, priority="batch", **kw):
        from PIL import Image

        w, h = Image.open(path).size
        lm = [[x * w, y * h, 0.0] for x, y in _face_lm()]
        return dict(
            faces=[
                dict(bbox=[0.3 * w, 0.3 * h, 0.7 * w, 0.75 * h], det_score=0.9, landmark_3d_68=lm, pose=[1.0, 2.0, 0.0])
            ]
        )

    def run(self, role, params, app, priority="batch", model=None):
        if role == "face-kind":
            return dict(faces=[dict(real=0.95, kind="real") for _ in params["boxes"]])
        np.savez(params["out"], images=self.emb[None])
        return {}


@pytest.fixture
def full_bundle(stub_bundle, monkeypatch):
    """The stub bundle with metadata and CLIP blocks, a stand-in network and skin models, and a fake strata-models."""
    import sys

    import torch

    from stratastills.look import transform as T

    d = stub_bundle.path
    base = list(F.img_feats(np.full((40, 60, 3), 128, np.uint8))) + list(F.raw_feats({}))
    enc = F.meta_encoder(RECS)
    rng = np.random.default_rng(3)
    np.savez(
        os.path.join(d, "clip-basis.npz"),
        mu=rng.normal(size=8),
        vt=rng.normal(size=(3, 8)),
        text_where=rng.normal(size=(2, 8)),
    )
    json.dump([["where", ["in", "out"]]], open(os.path.join(d, "clip-prompts.json"), "w"))
    json.dump(enc, open(os.path.join(d, "meta-encoder.json"), "w"))
    names = base + ["meta:" + n for n in enc["names"]] + ["clip:pc0", "clip:pc1", "clip:pc2", "clip:in", "clip:out"]
    man = dict(names=names, base_names=base, n_meta=len(enc["names"]), n_clip=5, files={"x": "2"})
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"))
    pickle.dump([Const(0.0)] * 3, open(os.path.join(d, "skin.pkl"), "wb"))
    look._bundles.clear()
    b = look.bundle()

    class Net(torch.nn.Module):
        def forward(self, img, x):
            return T.init_params(len(img), "cpu")

    ident = T.init_params(1, "cpu")[0].numpy()
    b._m["net"] = dict(
        net=Net(), mu=np.zeros(len(names)), sd=np.ones(len(names)), dev="cpu", lo=ident - 1, hi=ident + 1, typical=ident
    )
    monkeypatch.setitem(sys.modules, "strata_models_client", FakeSM(rng.normal(size=8)))
    return b


def test_a_new_picture_gets_its_features_and_faces_and_every_version(full_bundle, monkeypatch):
    rng = np.random.default_rng(5)
    pv = (rng.random((300, 400, 3)) * 120 + 60).astype(np.uint8)
    pv[90:225, 120:280] = (110, 140, 190)  # a skin-coloured face area (BGR)
    ex = {"ISO": 400, "ExposureTime": 0.01, "Model": "ILCE-7C", "DateTimeOriginal": "2020:05:01 10:00:00"}
    emb = look.clip_embedding(pv)
    x = look.feature_row(pv, ex, RECS[0], emb, full_bundle)
    assert len(x) == len(full_bundle.names) and np.isfinite(x[-5:]).all()
    fl = look.faces(pv)
    assert len(fl) == 1 and fl[0]["real"] == 0.95 and len(fl[0]["lm"]) == 68 and 0.29 < fl[0]["box"][0] < 0.31
    neu = pv.copy()
    for v in look.VARIANTS:
        out, did = look.render(v, neu, x, fl, full_bundle)
        assert out.shape == neu.shape, v
    assert "face_target" in look.render("auto7", neu, x, fl, full_bundle)[1]
    x2 = look.feature_row(pv, ex, RECS[0], None, full_bundle)
    assert np.isnan(x2[-5:]).all()  # no strata-models: CLIP columns missing, as the models were trained to cope with


def test_the_stage_runs_the_look_on_the_picture(full_bundle, monkeypatch, tmp_path):
    import cv2

    p = str(tmp_path / "shot.jpg")
    rng = np.random.default_rng(6)
    cv2.imwrite(p, (rng.random((120, 160, 3)) * 200 + 20).astype(np.uint8))
    monkeypatch.setattr(look, "exif", lambda path: ({"ISO": 100}, RECS[1]))

    class St:
        rec = dict(steps=[])
        path = p
        say = staticmethod(lambda t: None)
        rgb = rng.random((120, 160, 3)).astype(np.float32)

    st = St()
    PL.run_look(st, dict(variant="auto5"))
    assert st.rec["look"]["applied"] and st.rec["look"]["clip"] and st.rec["look"]["faces"] == 1
    assert st.rgb.shape == (120, 160, 3) and st.rgb.dtype == np.float32
    PL.run_look(st, dict(variant="none"))  # nothing to do


def test_exif_reads_both_records_with_exiftool(monkeypatch):
    import subprocess

    calls = []

    class R:
        def __init__(self, out):
            self.stdout = out

    def run(cmd, **kw):
        calls.append(cmd)
        return R(json.dumps([{"SourceFile": cmd[-1], "IFD0:Model" if "-G1" in cmd else "Model": "X"}]))

    monkeypatch.setattr(subprocess, "run", run)
    ex, meta = look.exif("/x/a.ARW")
    assert ex["Model"] == "X" and meta["IFD0:Model"] == "X" and len(calls) == 2


def test_ratings_come_from_the_arena_votes(monkeypatch):
    from stratastills import arena as AR
    from stratastills import arena_store

    class Store:
        def retired(self):
            return {"auto2"}

        def votes(self):
            return []

        def pool(self):
            return {}

    monkeypatch.setattr(arena_store, "get", lambda: Store())
    monkeypatch.setattr(AR, "current_votes", lambda v, p: [dict(a="auto5", b="auto7")] * 3)
    monkeypatch.setattr(
        AR, "bradley_terry", lambda v, r: {"auto5": 1650.4, "auto7": 1500.0, "auto2": 1700.0, "lightroom": 1800.0}
    )
    look._ratings.clear()
    assert look.ratings() == {"auto5": dict(rating=1650, votes=3), "auto7": dict(rating=1500, votes=3)}
    assert look.winner() == "auto5"
    look._ratings.clear()


def test_faces_follow_the_crop_and_the_turn_into_the_finished_picture():
    f = dict(box=[0.35, 0.4, 0.1, 0.1], lm=[[0.4, 0.45]], real=0.9)
    (c,) = look.map_faces([f], [("centre", 0.5)])
    assert np.allclose(c["box"], [0.2, 0.3, 0.2, 0.2]) and np.allclose(c["lm"], [[0.3, 0.4]])
    assert look.map_faces([dict(f, box=[0.0, 0.0, 0.1, 0.1])], [("centre", 0.5)]) == []  # cropped away
    (r,) = look.map_faces([dict(f, box=[0.45, 0.45, 0.1, 0.1], lm=[[0.5, 0.5]])], [("rotate", 3.0, 400, 300)])
    assert np.allclose(r["lm"], [[0.5, 0.5]]) and r["box"][2] > 0.1  # the centre stays put; a turned box grows


def test_no_service_calls_while_the_whole_card_is_leased(monkeypatch, full_bundle):
    from stratastills import gpu

    monkeypatch.setattr(gpu, "lease_is_exclusive", lambda: True)
    pv = np.full((100, 100, 3), 120, np.uint8)
    assert look.faces(pv) == [] and look.clip_embedding(pv) is None


def test_a_sticky_cuda_error_restarts_the_process_and_nothing_else_does(monkeypatch):
    import threading

    from stratastills import gpu

    started = []

    class T:
        def __init__(self, delay, fn):
            started.append(fn)

        def start(self):
            pass

    monkeypatch.setattr(threading, "Timer", T)
    assert not gpu.exit_if_cuda_broken(RuntimeError("CUDA out of memory")) and started == []
    assert gpu.cuda_broken(RuntimeError("CUDA error: device-side assert triggered"))
    assert (
        gpu.exit_if_cuda_broken(RuntimeError("CUDA error: an illegal memory access was encountered"))
        and len(started) == 1
    )


def test_a_batch_stops_starting_pictures_after_a_sticky_cuda_error(monkeypatch):
    calls = []

    def gen(pid, iid, *a, **k):
        calls.append(iid)
        raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setattr(PL, "pending", lambda pid: [(f"s~{i}", None, "best") for i in range(20)])
    monkeypatch.setattr(PL, "generate", gen)
    with pytest.raises(RuntimeError, match="device-side"):
        PL.apply_rules("p", workers=1)
    assert len(calls) == 1
