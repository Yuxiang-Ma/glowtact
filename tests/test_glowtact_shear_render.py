import sys

import numpy as np
import pytest


pytest.importorskip("cv2")

from glowtact.tools import shear_field_render as sfr  # noqa: E402
from glowtact.shear import ShearField  # noqa: E402


def _demo_field(rows=7, cols=7, spacing=26):
    ys, xs = np.mgrid[0:rows, 0:cols]
    nodes = np.stack([(xs.ravel() + 1) * spacing,
                      (ys.ravel() + 1) * spacing], axis=1).astype(np.float32)
    vecs = np.stack([np.full(rows * cols, 1.2), np.full(rows * cols, -0.7)],
                    axis=1).astype(np.float32)
    conf = np.full(rows * cols, 0.8, np.float32)
    return ShearField(nodes, vecs, conf, (rows, cols))


def _panel(bg_denoise, bg_fade, seed=0):
    rng = np.random.default_rng(seed)
    gray = rng.integers(60, 200, (240, 240), dtype=np.uint8)
    contact = np.zeros((240, 240), bool)
    contact[100:140, 100:140] = True
    return sfr._draw_field(gray, _demo_field(), contact, scale=8.0,
                           min_conf=0.10, cap=34, slip=0.5,
                           bg_denoise=bg_denoise, bg_fade=bg_fade)


def test_background_treatment_does_not_touch_the_readout():
    """The two published video variants must differ ONLY in the backdrop.

    The readout banner is the measurement; if a display setting could move it,
    the two folders would be two different results rather than two views of
    one. (Encoded mp4s do differ slightly there -- that is H.264 allocating
    bits over a different frame, not a different render.)
    """
    tuned = _panel(True, 0.55)
    plain = _panel(False, 0.0)
    assert tuned.shape == plain.shape
    banner_h = 34
    assert np.array_equal(tuned[-banner_h:], plain[-banner_h:])


def test_fade_lightens_the_gel_without_moving_the_arrows():
    tuned = _panel(True, 0.55)[:-34]
    plain = _panel(False, 0.0)[:-34]
    # the backdrop is lighter and flatter
    assert tuned.mean() > plain.mean() + 20
    assert tuned.std() < plain.std()
    # ...but the drawn overlay still lands in the same places: the amber arrows
    # are the strongly non-grey pixels, and they survive the treatment.
    def coloured(img):
        mx = img.max(axis=2).astype(int)
        mn = img.min(axis=2).astype(int)
        return (mx - mn) > 40
    assert coloured(tuned).sum() > 0
    overlap = (coloured(tuned) & coloured(plain)).sum()
    assert overlap > 0.7 * coloured(plain).sum()


def test_zero_fade_and_no_denoise_leaves_the_gel_alone():
    rng = np.random.default_rng(3)
    gray = rng.integers(60, 200, (80, 80), dtype=np.uint8)
    out = sfr._quiet_background(gray, denoise=False, fade=0.0)
    assert np.array_equal(out[..., 0], gray)
    assert np.array_equal(out[..., 1], gray)
    assert np.array_equal(out[..., 2], gray)


def _tiny_clip(path, n=6):
    """A few frames of speckle with a textured square pressing then sliding."""
    import cv2
    rng = np.random.default_rng(3)
    gel = cv2.GaussianBlur(rng.integers(0, 255, (160, 160), dtype=np.uint8), (0, 0), 1.2)
    obj = cv2.GaussianBlur(rng.integers(0, 255, (50, 50), dtype=np.uint8), (0, 0), 1.2)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 160))
    for i in range(n):
        f = gel.copy()
        if i >= 1:
            x = 55 + 2 * max(0, i - 2)                  # still, then sliding
            f[55:105, x:x + 50] = (obj * 0.6).astype(np.uint8)  # darkening imprint
        vw.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    vw.release()


@pytest.mark.parametrize("mode", ["temporal", "spatial"])
def test_render_writes_one_panel_per_frame_in_every_slip_mode(tmp_path, mode):
    """The panel loop zips several per-frame lists; a branch that forgets to
    append to one of them silently rendered ZERO panels (temporal mode did)."""
    clip = str(tmp_path / "clip.avi")
    _tiny_clip(clip, n=6)
    res = sfr.render(clip, str(tmp_path / "out.mp4"), spacing=20, win=21,
                     min_conf=0.05, slip_mode=mode,
                     panel_dir=str(tmp_path / "panels"))
    assert res["n_panels"] == 5
    assert len(list((tmp_path / "panels").glob("*.png"))) == 5
    assert len(res["stick_ratio"]) == 5 and res["slip_mode"] == mode
    assert (tmp_path / "out.mp4").exists()
