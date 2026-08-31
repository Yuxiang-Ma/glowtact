import sys

import numpy as np
import pytest


from glowtact import clean_diff as cd  # noqa: E402


def test_recover_signed_undoes_the_128_centring():
    stored = np.array([[128, 138, 118]], dtype=np.uint8)
    assert cd.recover_signed(stored).tolist() == [[0, 10, -10]]


def test_dog_bandpass_removes_a_smooth_global_shift():
    pytest.importorskip("cv2")
    ys, xs = np.mgrid[0:120, 0:120]
    smooth = (xs + ys).astype(np.float32)          # a large-scale ramp = the shift
    ridges = 20 * np.sin(xs.astype(np.float32) * 0.9)  # ~7 px thread pitch
    inner = (slice(30, 90), slice(30, 90))         # away from Gaussian edge bleed
    ramp_only = cd.dog_bandpass(smooth, 1.2, 6.0)  # a linear ramp has ~zero DoG
    with_ridges = cd.dog_bandpass(smooth + ridges, 1.2, 6.0)
    # The ridge texture survives, orders of magnitude above the leftover ramp.
    assert np.abs(with_ridges[inner]).max() > 5
    assert np.abs(ramp_only[inner]).max() < 0.2
    assert (np.abs(with_ridges[inner]).max()
            > 20 * np.abs(ramp_only[inner]).max())


def test_local_median_compensate_removes_nonuniform_background():
    """A curved finger's artifact is a spatially-VARYING smooth field; the
    local median must flatten it while a robust median ignores the sharp
    thread outliers (unlike a mean, which they would drag)."""
    pytest.importorskip("cv2")
    ys, xs = np.mgrid[0:120, 0:120].astype(np.float32)
    # non-uniform smooth background (a curved bulge), plus sharp ridges
    background = 20 * np.sin(xs / 40.0) * np.cos(ys / 30.0)
    ridges = np.zeros((120, 120), np.float32)
    ridges[55:65, 50:70] = 25 * np.sign(np.sin(xs[55:65, 50:70]))
    comp = cd.local_median_compensate(background + ridges, bg_kernel=21,
                                      denoise_sigma=0)
    # Background flattened away from the ridges; ridge contrast preserved.
    assert np.abs(comp[10:30, 10:30]).max() < 4.0
    assert np.abs(comp[55:65, 50:70]).max() > 12.0


def test_compensate_dispatches_and_rejects_unknown_method():
    pytest.importorskip("cv2")
    img = np.zeros((30, 30), np.float32)
    img[10:20, 10:20] = 20
    assert cd.compensate(img, "dog").shape == img.shape
    assert cd.compensate(img, "local-median", bg_kernel=7).shape == img.shape
    with pytest.raises(ValueError):
        cd.compensate(img, "nope")


def test_local_energy_favours_dense_over_sparse():
    pytest.importorskip("cv2")
    field = np.zeros((100, 100), np.float32)
    field[40:60, 40:60] = 10.0                     # a dense patch (the contact)
    field[10, 80] = 10.0                           # a lone spike (a crease)
    m = cd.local_energy(field, sigma=6.0, gamma=1.5)
    assert m[50, 50] > 0.5                          # dense region kept
    assert m[10, 80] < 0.2                          # isolated spike suppressed


def test_fov_mask_is_one_at_centre_and_zero_outside():
    m = cd.fov_mask((100, 100), radius_frac=0.8, softness=0.05)
    assert m[50, 50] == pytest.approx(1.0, abs=1e-3)
    assert m[0, 0] == 0.0                            # far corner is outside the disc
    assert 0.0 <= m.min() and m.max() <= 1.0


def test_autoscale_gain_maps_percentile_to_target():
    cleaned = np.zeros((100, 100), np.float32)
    cleaned[:50] = 4.0                              # 99.5-pct magnitude ~ 4
    g = cd.autoscale_gain(cleaned, target=100.0)
    assert g == pytest.approx(25.0, rel=0.1)


def _synthetic_signed(shift=0.4):
    """A diff with a non-uniform background shift plus a compact ridge patch."""
    ys, xs = np.mgrid[0:120, 0:120].astype(np.float32)
    bg = 15 * np.sin(xs / 35.0) * np.cos(ys / 28.0) + shift * (xs - 60)
    signed = bg.copy()
    signed[55:70, 52:72] += 28 * np.sign(np.sin(xs[55:70, 52:72] * 0.9))
    return np.repeat(signed[..., None], 3, 2).astype(np.int16)


def test_evaluate_ranks_clean_below_raw(tmp_path):
    pytest.importorskip("cv2")
    signed = _synthetic_signed()
    fov = cd.fov_mask(signed.shape[:2], 0.95)
    roi = cd.reference_roi(cd.clean(signed), fov)
    raw = cd.to_signal(signed, True)
    cleaned = cd.clean(signed)

    raw_obj = cd.evaluate(signed, raw, roi, fov)["objective"]
    clean_obj = cd.evaluate(signed, cleaned, roi, fov)["objective"]
    assert clean_obj < raw_obj                        # cleaning lowers artifact
    assert 0.0 <= cd.evaluate(signed, cleaned, roi, fov)["bg_ratio"]


def test_reference_roi_is_a_small_compact_region():
    pytest.importorskip("cv2")
    signed = _synthetic_signed()
    fov = cd.fov_mask(signed.shape[:2], 0.95)
    roi = cd.reference_roi(cd.clean(signed), fov)
    # The ROI localises the ridge patch, not the whole frame.
    assert 0.0 < roi.mean() < 0.25
    assert roi[55:70, 52:72].mean() > 0.5             # covers the ridge patch


def test_process_folder_writes_clean_and_skips_derived(tmp_path):
    cv2 = pytest.importorskip("cv2")
    # canonical diff with a thread-like ridge patch on a shifted background
    ys, xs = np.mgrid[0:120, 0:120]
    signed = ((xs - 60) * 0.3).astype(np.float32)   # smooth shift
    signed[50:70, 50:70] += 30 * np.sin(xs[50:70, 50:70] * 2.0)
    stored = np.clip(signed[..., None] + 128, 0, 255).astype(np.uint8).repeat(3, 2)
    cv2.imwrite(str(tmp_path / "a_diff.png"), stored)
    cv2.imwrite(str(tmp_path / "a_diff_x3.png"), stored)      # must be skipped
    cv2.imwrite(str(tmp_path / "a_diff_autoscale.png"), stored)  # must be skipped

    written = cd.process_folder(str(tmp_path), out_dir=str(tmp_path / "out"))

    assert [n for n, _ in written] == ["a_diff_clean.png"]
    clean = cv2.imread(str(tmp_path / "out" / "a_diff_clean.png"),
                       cv2.IMREAD_GRAYSCALE)
    # The ridge patch stands out from the (now flat) background.
    assert clean[60, 50:70].std() > clean[10, 10:30].std() + 10
