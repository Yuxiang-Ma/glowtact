import sys

import numpy as np
import pytest


from glowtact import enhance_diff as ed  # noqa: E402


def test_center_crop_keeps_fraction_and_is_centred():
    img = np.arange(100 * 200).reshape(100, 200).astype(np.uint8)
    c = ed.center_crop(img, 0.4)
    assert c.shape == (40, 80)                       # 40% of each dimension
    # symmetric margins
    assert c[0, 0] == img[30, 60]


def test_center_crop_rejects_bad_fraction():
    with pytest.raises(ValueError):
        ed.center_crop(np.zeros((10, 10)), 0.0)


def _speckly_threads(seed=0):
    """A repetitive ridge pattern buried in per-pixel speckle."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:80, 0:80]
    threads = 128 + 18 * np.sin(xs * 0.9)            # ~7 px pitch ridges
    speckle = rng.normal(0, 12, (80, 80))
    return np.clip(threads + speckle, 0, 255).astype(np.uint8)


def test_denoise_reduces_speckle_and_keeps_threads():
    pytest.importorskip("cv2")
    img = _speckly_threads()
    out = ed.denoise(img, h=8)
    assert ed.speckle_level(out) < ed.speckle_level(img) * 0.6   # speckle down
    assert ed.structure_corr(img, out) > 0.85                    # threads kept


def test_auto_h_stops_before_breaking_the_pattern():
    pytest.importorskip("cv2")
    img = _speckly_threads()
    h = ed.auto_h(img, candidates=(3, 5, 7, 9, 11, 21), min_corr=0.90)
    assert h > 0                                     # some denoising chosen
    assert ed.structure_corr(img, ed.denoise(img, h)) >= 0.90
    # h=0 when the floor is impossible (no denoise preserves it perfectly).
    assert ed.auto_h(img, candidates=(21,), min_corr=0.999) == 0


def test_process_folder_writes_enhanced_and_skips_derived(tmp_path):
    cv2 = pytest.importorskip("cv2")
    img = _speckly_threads()
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(tmp_path / "a_diff.png"), bgr)
    cv2.imwrite(str(tmp_path / "a_diff_x3.png"), bgr)          # must be skipped
    cv2.imwrite(str(tmp_path / "a_diff_clean.png"), bgr)      # must be skipped

    rows = ed.process_folder(str(tmp_path), out_dir=str(tmp_path / "out"),
                             crop=0.6, h=6)

    assert [s for s, _ in rows] == ["a"]
    produced = {p.name for p in (tmp_path / "out").iterdir()}
    assert "a_diff_enh.png" in produced
    assert "a_diff_enh_x3.png" in produced
    enh = cv2.imread(str(tmp_path / "out" / "a_diff_enh.png"),
                     cv2.IMREAD_GRAYSCALE)
    assert enh.shape == (48, 48)                      # 60% center crop of 80x80
