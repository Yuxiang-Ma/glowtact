import sys

import numpy as np
import pytest


from glowtact import realtime as rt  # noqa: E402
from glowtact import render as tr  # noqa: E402


def test_latest_frame_is_a_drop_old_double_buffer():
    slot = rt.LatestFrame()
    assert slot.get(-1) is None                     # nothing published yet
    slot.publish("a")
    assert slot.get(-1) == (1, "a")                 # first reader sees it
    assert slot.get(1) is None                      # same version -> no redraw
    slot.publish("b")                               # overwrite (never queued)
    assert slot.get(1) == (2, "b")                  # newest wins, old dropped


def test_frame_geometry_keeps_xy_constant_and_tracks_z():
    h1 = np.zeros((40, 60), np.float32)
    h2 = h1.copy()
    h2[10:20, 10:20] = 0.5                          # a bump in the second frame
    v1, shade1, grid = rt.frame_geometry(h1, stride=4, mm_per_px=0.05, z_scale=3.0)
    v2, _s2, grid2 = rt.frame_geometry(h2, stride=4, mm_per_px=0.05, z_scale=3.0)

    rows, cols = grid
    assert grid == grid2
    assert v1.shape == (rows * cols, 3)
    # x/y columns are the constant grid; only z changes between frames.
    assert np.array_equal(v1[:, :2], v2[:, :2])
    assert not np.array_equal(v1[:, 2], v2[:, 2])
    assert v2[:, 2].max() == pytest.approx(0.5 * 3.0)   # z exaggeration applied
    assert len(shade1) == rows * cols


def test_constant_triangles_match_the_grid():
    rows, cols = 12, 15
    tris = tr.height_grid_triangles(rows, cols)
    assert len(tris) == 2 * (rows - 1) * (cols - 1)     # two per cell
    # stable across calls (safe to compute once and reuse each frame)
    assert np.array_equal(tris, tr.height_grid_triangles(rows, cols))
    assert tris.max() == rows * cols - 1


def test_reconstruct_once_lights_only_the_contact():
    pytest.importorskip("cv2")
    ref = np.full((60, 60, 3), 128, np.uint8)
    frame = ref.copy()
    frame[25:35, 25:35] = 60                         # a darker (pressed) patch
    lut = np.linspace(0, 2.55, 256)                  # 0.01 mm per grey level
    height = rt.reconstruct_once(frame, ref, lut, lighting_threshold=8,
                                 kernels=(1,))
    assert height[30, 30] > 0.2                       # contact has depth
    assert height[5, 5] == 0.0                        # untouched stays flat


class _FakeReader:
    """Hands the worker one preset frame per wait_for_new call, then None."""
    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def wait_for_new(self, last_seq, timeout=1.0):
        if self.i >= len(self.frames):
            return None
        f = self.frames[self.i]
        self.i += 1
        return (self.i, f, 0, 0)


def _worker(frames):
    reader = _FakeReader(frames)
    lut = np.linspace(0, 2.55, 256)
    stats = {"lighting_threshold": 8, "mm_per_px": 0.05, "max_depth_mm": 2.0}
    return rt.ReconstructionWorker(reader, lut, stats, rt.LatestFrame(),
                                   stride=4, z_scale=3.0, kernels=(1,))


def test_worker_step_reconstructs_against_first_frame_as_reference():
    pytest.importorskip("cv2")
    flat = np.full((60, 60, 3), 128, np.uint8)
    pressed = flat.copy()
    pressed[25:35, 25:35] = 60
    w = _worker([flat, pressed])

    first = w._step()                                # sets reference = flat
    assert first.max_mm == pytest.approx(0.0, abs=1e-6)
    second = w._step()                               # pressed vs flat -> contact
    assert second.max_mm > 0.2
    assert second.version == 2
    rows, cols = second.grid
    assert second.vertices.shape == (rows * cols, 3)


def test_worker_retare_adopts_the_next_frame_as_baseline():
    pytest.importorskip("cv2")
    flat = np.full((60, 60, 3), 128, np.uint8)
    pressed = flat.copy()
    pressed[25:35, 25:35] = 60
    # frames: flat (ref), pressed (contact), pressed again (new ref), pressed
    w = _worker([flat, pressed, pressed, pressed])
    w._step()                                        # ref = flat
    assert w._step().max_mm > 0.2                     # contact vs flat
    w.request_retare()
    assert w._step().max_mm == pytest.approx(0.0, abs=1e-6)  # new ref = pressed
    assert w._step().max_mm == pytest.approx(0.0, abs=1e-6)  # pressed vs pressed
