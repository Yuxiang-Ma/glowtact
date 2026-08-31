import sys

import numpy as np
import pytest


from glowtact import render as tr  # noqa: E402


def test_save_gif_writes_a_multi_frame_animation(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    frames = [np.full((16, 24, 3), v, np.uint8) for v in (0, 80, 160, 240)]
    path = tmp_path / "spin.gif"

    tr.save_gif(str(path), frames, fps=10)

    im = Image.open(str(path))
    assert im.n_frames == 4
    assert im.size == (24, 16)                       # PIL reports (w, h)


def test_grid_xy_is_constant_and_scaled():
    x, y = tr.grid_xy((40, 60), stride=4, mm_per_px=0.05)
    rows, cols = len(range(0, 40, 4)), len(range(0, 60, 4))
    assert x.shape == y.shape == (rows * cols,)
    assert x.max() == pytest.approx((cols - 1) * 4 * 0.05)   # last col in mm
    assert y.max() == pytest.approx((rows - 1) * 4 * 0.05)
