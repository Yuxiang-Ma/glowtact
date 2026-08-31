"""Normal-force estimation: features, fitting, honesty of the stored stats."""
import numpy as np
import pytest

from glowtact import force


def _dark(area_px=1000, level=50, shape=(120, 160)):
    d = np.zeros(shape, np.uint8)
    side = int(np.sqrt(area_px))
    d[10:10 + side, 10:10 + side] = level
    return d


def test_features_scale_with_press_depth_and_area():
    shallow = force.extract_features(_dark(level=30))
    deep = force.extract_features(_dark(level=90))
    wide = force.extract_features(_dark(area_px=4000, level=30))
    assert deep["dark_sum"] == pytest.approx(3 * shallow["dark_sum"])
    assert deep["contact_area"] == shallow["contact_area"]
    assert wide["contact_area"] > 3 * shallow["contact_area"]
    assert deep["density"] > shallow["density"]


def test_darkening_from_signed_matches_convention():
    signed = np.full((8, 8), 128, np.uint8)
    signed[2, 2] = 88                                # 40 levels darker
    dark = force.darkening_from_signed(signed)
    assert dark[2, 2] == 40 and dark[0, 0] == 0


def test_fit_recovers_a_linear_gel_and_reports_loo(tmp_path):
    rng = np.random.default_rng(0)
    feats, newtons = [], []
    for _ in range(20):
        level = rng.integers(20, 120)
        d = _dark(level=int(level))
        feats.append(force.extract_features(d))
        newtons.append(2.5 * feats[-1]["dark_sum"] + rng.normal(0, 0.02))
    cal = force.fit_calibration(feats, newtons, ("dark_sum",), source="synthetic")
    assert cal.coef[0] == pytest.approx(2.5, rel=0.05)
    assert cal.loo_rms_n < 0.1                       # honest noise level
    est = cal.estimate(feats[0])
    assert est == pytest.approx(newtons[0], abs=0.1)

    path = tmp_path / "cal.npz"
    cal.save(str(path))
    back = force.NormalForceCalibration.load(str(path))
    assert back.estimate(feats[0]) == pytest.approx(est)
    assert back.force_range_n == cal.force_range_n
    assert back.source == "synthetic"


def test_log_fit_never_returns_negative_force():
    feats = [force.extract_features(_dark(level=l)) for l in (20, 40, 80, 120)]
    cal = force.fit_calibration(feats, [0.5, 1.0, 2.0, 3.0],
                                ("dark_sum", "contact_area"), log=True)
    tiny = force.extract_features(_dark(area_px=4, level=9))
    assert cal.estimate(tiny) >= 0.0


def test_pairs_from_index_drops_presses_beside_the_load_cell(tmp_path):
    """0 N with a big imprint = the press missed the cell; it must not
    become a zero anchor (observed on the 2026-07-27 screw sessions)."""
    import csv

    cv2 = pytest.importorskip("cv2")
    folder = tmp_path
    rows = []
    for i, (newton, level) in enumerate([(0.0, 0), (0.0, 60), (2.0, 60)]):
        signed = np.full((40, 40, 3), 128, np.uint8)
        if level:
            signed[10:30, 10:30] = 128 - level
        name = f"{i:06d}_diff.png"
        cv2.imwrite(str(folder / name), signed)
        rows.append({"force_newtons": f"{newton}", "diff_file": name,
                     "diff_mode": "signed", "name": f"s{i}"})
    with open(folder / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    feats, newtons, names = force.pairs_from_index(str(folder / "index.csv"))
    assert list(newtons) == [0.0, 2.0]               # contradictory row dropped
    assert names == ["s0", "s2"]
