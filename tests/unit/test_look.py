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
