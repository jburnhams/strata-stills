import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from stratastills import develop as D, lensfundb as F, lenscorr as L

XML = """<lensdatabase version="2"><lens><maker>Sony</maker><model>E 50mm f/1.8 OSS</model><model lang="de">E 50mm 1,8</model><mount>Sony E</mount><cropfactor>1.534</cropfactor>
<calibration><distortion model="poly3" focal="50" k1="0.01"/><tca model="linear" focal="50" kr="1.0002" kb="0.9998"/>
<vignetting model="pa" focal="50" aperture="1.8" distance="1000" k1="-0.5" k2="-0.2" k3="0"/><vignetting model="pa" focal="50" aperture="2.8" distance="1000" k1="-0.3" k2="-0.1" k3="0"/></calibration></lens>
<lens><maker>Sony</maker><model>FE 28-60mm f/4-5.6</model><mount>Sony E</mount><cropfactor>1.0</cropfactor><aspect-ratio>3:2</aspect-ratio>
<calibration><distortion model="ptlens" focal="28" a="0.0" b="-0.05" c="0.0"/><distortion model="ptlens" focal="60" a="0.0" b="0.0" c="0.0"/></calibration></lens></lensdatabase>"""


@pytest.fixture
def lenses(tmp_path):
    p = tmp_path / 'mil-sony.xml'; p.write_text(XML); return F.load(str(tmp_path))


def test_the_lens_is_found_from_the_exif_name_by_its_words_and_numbers(lenses):
    assert F.find('E 50mm F1.8 OSS', lenses)['model'] == 'E 50mm f/1.8 OSS' and F.find('FE 28-60mm F4-5.6', lenses)['model'] == 'FE 28-60mm f/4-5.6'
    assert F.find('SAMYANG AF 50mm F1.4 II', lenses) is None and F.find('E 50mm F1.8', lenses) is None and F.find('', lenses) is None            # a name needs all its tokens: no partial match


def test_models_follow_lensfuns_formulas(lenses):
    e50, fe = lenses; r = np.array([0.0, 0.5, 1.0])
    assert np.allclose(F.distortion_ratio(e50, 50, r), [0.99, 0.01 * 0.25 + 0.99, 1.0]) and np.allclose(F.distortion_ratio(fe, 28, r), -0.05 * r * r + 1 + 0.05)                          # poly3: k r^2 + 1 - k; ptlens: a r^3 + b r^2 + c r + 1 - a - b - c
    assert np.allclose(F.distortion_ratio(fe, 44, r), 0.5 * (-0.05 * r * r + 1.05) + 0.5 * 1.0) and F.tca_ratio(e50, 50, 0) == 1.0002 and F.tca_ratio(e50, 50, 2) == 0.9998
    att = F.vignetting_attenuation(e50, 50, 1.8, None, np.array([0.0, 1.0])); assert np.allclose(att, [1.0, 1 - 0.7]) and np.allclose(F.vignetting_attenuation(e50, 50, 2.8, None, np.array([1.0])), 1 - 0.4)
    mid = F.vignetting_attenuation(e50, 50, 2.24, None, np.array([1.0])); assert 0.3 < mid[0] < 0.6                                                                                                # between the two apertures, in stops: not the lower one, not the higher


def test_the_correction_is_handed_to_the_development_as_opcodes(lenses):
    e50, _ = lenses; gain, warp = F.corrections(e50, 600, 400, 50, 1.8, None, camera_crop=1.534)
    assert gain(np.array([1.0]))[0] == pytest.approx(1 / 0.3) and warp.shape == (3, 6)
    ops = D.parse_opcodes(L.opcodes_from(gain, warp, 600, 400)); assert [i for i, _ in ops] == [9, 1]
    m = D.gain_map(ops[0][1])[0]; assert m.shape == (65, 65, 1) and m[32, 32, 0] == pytest.approx(1.0, abs=1e-3) and m[0, 0, 0] == pytest.approx(1 / 0.3, rel=1e-3)
    g2, w2 = F.corrections(e50, 600, 400, 50, 1.8, None, camera_crop=1.0); assert g2(np.array([1.0]))[0] == pytest.approx(1 / 0.3)                                                   # a lens for a smaller sensor on a bigger one: the corner's value is held beyond the calibrated circle
    assert L.opcodes_from(None, None, 10, 10) is None and [i for i, _ in D.parse_opcodes(L.opcodes_from(None, warp, 10, 10))] == [1]


def test_no_database_no_correction(tmp_path, monkeypatch):
    monkeypatch.setenv(F.ENV, str(tmp_path / 'nothing')); F._db.clear(); assert F.load() == [] and F.find('E 50mm F1.8 OSS') is None


def test_an_aps_c_lens_on_a_full_frame_body_is_cropped_to_its_image_circle(lenses, monkeypatch):
    shots = {'e50': ('E 50mm F1.8 OSS', 50.0, 1.8, None, 1.0), 'fe': ('FE 28-60mm F4-5.6', 28.0, 4.0, None, 1.0), 'e50_on_aps_c': ('E 50mm F1.8 OSS', 50.0, 1.8, None, 1.534), 'unknown': ('SAMYANG AF 50mm F1.4 II', 50.0, 1.4, None, 1.0), 'no_sensor': ('E 50mm F1.8 OSS', 50.0, 1.8, None, None)}
    monkeypatch.setattr(F, 'shot_of', lambda path: shots[path]); find = F.find; monkeypatch.setattr(F, 'find', lambda name: find(name, lenses))                                                                  # the fixture's database instead of models/lensfun
    assert F.image_circle('e50') == pytest.approx(1 / 1.534, abs=1e-4)
    assert [F.image_circle(k) for k in ('fe', 'e50_on_aps_c', 'unknown', 'no_sensor')] == [1.0] * 4              # made for this sensor, the sensor the lens was made for, no profile, no crop factor of the sensor: the whole picture


def test_the_crop_is_centred_and_keeps_the_shape(monkeypatch):
    from stratastills import pipeline as PL
    st = type('S', (), {})(); st.path = 'x'; st.rgb = np.zeros((400, 600, 3), np.float32); st.analysis = np.zeros((200, 300), np.float32); st.rec = {}
    monkeypatch.setattr(F, 'image_circle', lambda path: 0.65); PL.crop_to_image_circle(st, 'full'); assert st.rgb.shape == (400, 600, 3) and not st.rec
    PL.crop_to_image_circle(st, 'auto'); assert st.rgb.shape == (260, 390, 3) and st.analysis.shape == (130, 195) and st.rec['frame']['fraction'] == 0.65
