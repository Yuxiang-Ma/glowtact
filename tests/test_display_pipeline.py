"""The despeckle/display pipeline in glowtact.imgfilt (docs/pipeline_params.md)."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from glowtact import imgfilt as F  # noqa: E402


def _press(shape=(200, 200), depth=80.0, center=(100, 100), r=25):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    d2 = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    return np.where(d2 < r * r, depth * (1 - d2 / (r * r)), 0).astype(np.float32)


def test_signed_difference_keeps_brightening():
    ref = np.full((10, 10), 100, np.uint8)
    cur = ref.copy()
    cur[2, 2], cur[5, 5] = 60, 140
    s = F.signed_difference(ref, cur)
    assert s[2, 2] == 40 and s[5, 5] == -40 and s.dtype == np.float32


def test_signed_difference_accepts_bgr():
    ref = np.full((8, 8, 3), 100, np.uint8)
    assert F.signed_difference(ref, ref).shape == (8, 8)


def test_remove_drift_kills_ramp_but_keeps_imprint():
    yy, xx = np.mgrid[:200, :200].astype(np.float32)
    ramp = 8.0 * xx / 200.0                      # smooth illumination drift
    s = _press() + ramp
    out = F.remove_drift(s)
    far = np.zeros_like(s, bool)
    far[:, :30] = far[:, -30:] = True
    assert np.abs(out[far]).mean() < 1.0         # background flattened
    assert out[100, 100] > 70                    # imprint survives


def test_refine_drops_small_and_faint_components():
    s = _press()
    s[10:13, 10:13] = 50                         # 9 px blob: below min_area
    s[150:170, 20:40] = 6                        # faint patch: below min_peak
    r = F.refine_signed(s)
    unfiltered = F.refine_signed(s, min_area=0, min_peak=0)
    # gains are soft-edged (blurred keep mask): small blobs are attenuated
    assert np.abs(r[11, 11]) < 0.7 * np.abs(unfiltered[11, 11])
    assert np.abs(r[160, 30]) < 1
    assert r[100, 100] > 60


def test_refine_preserves_sign():
    r = F.refine_signed(-_press())
    assert r[100, 100] < -60


def test_render_bidirectional_levels():
    s = np.array([[0.0, 160.0, 1000.0, -80.0]], np.float32)
    img = F.render_bidirectional(s)
    assert img.dtype == np.uint8
    assert list(img[0]) == [225, 0, 0, 255]


def test_display_pipeline_removes_speckle_keeps_press():
    rng = np.random.default_rng(0)
    ref = np.clip(150 + rng.normal(0, 6, (200, 200)), 0, 255).astype(np.uint8)
    cur = np.clip(ref.astype(np.float32) - _press()
                  + rng.normal(0, 6, (200, 200)), 0, 255).astype(np.uint8)
    img = F.display_pipeline(ref, cur)
    assert img.shape == (200, 200)
    assert img[100, 100] < 140                   # press is dark (bg 225)
    bg = img[:, :40]
    assert np.abs(bg.astype(int) - 225).mean() < 3   # speckle gone
    raw = F.render_bidirectional(F.signed_difference(ref, cur))
    assert np.abs(raw[:, :40].astype(int) - 225).mean() > \
        np.abs(bg.astype(int) - 225).mean()


def test_lab_transfer_matches_target_stats():
    rng = np.random.default_rng(1)
    src = rng.integers(60, 120, (64, 64, 3), dtype=np.uint8)
    tgt = rng.integers(120, 200, (64, 64, 3), dtype=np.uint8)
    out = F.apply_lab_affine(src, F.lab_transfer_affine(src, F.lab_stats(tgt)))
    for (mo, _), (mt, _) in zip(F.lab_stats(out), F.lab_stats(tgt)):
        assert abs(mo - mt) < 2.0
