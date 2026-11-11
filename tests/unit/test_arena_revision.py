import numpy as np

from stratastills import arena_revision as R


def picture(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (300, 400, 3), dtype=np.uint8)


def test_change_is_the_share_of_pixels_a_person_could_tell_apart():
    a = picture()
    b = a.copy()
    b[:, :40] = 255 - b[:, :40]  # a tenth of the picture turned inside out
    assert R.change(a, a) == 0.0 and 0.09 < R.change(a, b) < 0.11
    assert R.change(a, np.clip(a.astype(int) + 1, 0, 255).astype(np.uint8)) == 0.0  # below a just noticeable difference
    assert R.change(a, a[:200]) == 1.0 and R.change(None, a) == 1.0  # a new crop, or no old picture


def test_revise_keeps_the_votes_through_small_changes_and_drops_them_after_a_real_one():
    assert R.revise(None, "h1", 1.0, 5) == {}
    e1 = dict(hash="h1")
    e2 = dict(e1, hash="h2", **R.revise(e1, "h2", 0.01, 10))
    assert e2["same"] == ["h1"] and "changed" not in e2
    e3 = dict(e2, hash="h3", **R.revise(e2, "h3", 0.015, 20))
    assert e3["same"] == ["h1", "h2"]
    assert R.revise(e3, "h3", 0.0, 30) == {"same": ["h1", "h2"], "change": 0.015}
    e4 = R.revise(e3, "h4", 0.3, 40)
    assert e4 == {"same": [], "changed": 40, "change": 0.3}
