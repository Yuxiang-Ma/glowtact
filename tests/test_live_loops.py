"""Headless drives of the live CLIs: a fake camera opener + a stubbed cv2 GUI.

The live loops (`undistort calibrate`, `reconstruct live`) were the one part
of the pipeline no test exercised, and that is exactly where an extracted
import (`mc.signed_difference`) went stale without anyone noticing. These
tests push a scripted key sequence through each loop with synthetic frames, so
every statement in the loop body runs on CI with no camera and no display.
"""
import types

import numpy as np
import pytest

from glowtact import capture, depth, reconstruct, undistort


class _ScriptedReader:
    """wait_for_new() hands out the given frames in order, then None."""

    def __init__(self, frames):
        self._frames = list(frames)
        self._seq = 0
        self.stopped = False

    def wait_for_new(self, last_seq, timeout=1.0):
        if self._seq >= len(self._frames):
            return None
        frame = self._frames[self._seq]
        self._seq += 1
        return self._seq, frame, 0, 0

    def stop(self):
        self.stopped = True


@pytest.fixture
def fake_camera(monkeypatch):
    """Register an opener returning a scripted reader; reset afterwards."""
    state = {}

    def install(frames):
        reader = _ScriptedReader(frames)
        live = capture.LiveCamera("fake_cam", object(), reader,
                                  lambda: state.__setitem__("closed", True))
        capture.register_camera_opener(
            lambda camera, config, **kw: state.setdefault("live", live))
        state["reader"] = reader
        return state

    yield install
    capture.register_camera_opener(None)


@pytest.fixture
def scripted_gui(monkeypatch):
    """Silence the cv2 window calls and script what waitKey returns."""
    cv2 = pytest.importorskip("cv2")

    def install(keys):
        pressed = iter(keys)
        monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
        monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
        monkeypatch.setattr(cv2, "destroyWindow", lambda *a, **k: None)
        monkeypatch.setattr(cv2, "setMouseCallback", lambda *a, **k: None)
        monkeypatch.setattr(cv2, "waitKey", lambda *a, **k: ord(next(pressed)))
    return install


def _ball_frame(rows=6, cols=9, width=640, height=480):
    """A pressed ball array the robust detector locks onto (see test_undistort)."""
    cv2 = pytest.importorskip("cv2")
    ideal = undistort.ideal_lattice(rows, cols, width, height, 0.62)
    ys, xs = np.mgrid[0:height, 0:width]
    image = np.full((height, width), 150.0)
    for x, y in ideal:
        image[((xs - x) ** 2 + (ys - y) ** 2) < 25] -= 45
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def test_undistort_calibrate_runs_baseline_fit_save_quit(tmp_path, fake_camera,
                                                         scripted_gui):
    frame = _ball_frame()
    state = fake_camera([frame] * 4)
    scripted_gui("bfsq")               # baseline, fit, save, quit -- one per frame

    args = types.SimpleNamespace(
        camera="fake_cam", config=None, pattern="circles", rows=6, cols=9,
        fill=0.8, asymmetric=False, roi=None, min_area=10, max_area=8000,
        legacy_detector=False, diff_gain=3.0, detect_every=1, degree=5,
        model="radial", keep_tilt=False, no_auto_dump=True,
        dump_dir=str(tmp_path / "dump"), out=str(tmp_path / "map.yaml"))

    assert undistort.cmd_calibrate(args) == 0

    saved = undistort.load_map(str(tmp_path / "map.yaml"))
    assert saved["rows"] == 6 and saved["cols"] == 9
    assert saved["rms_residual_px"] < 3.0            # a clean synthetic press
    assert state["closed"] and state["reader"].stopped   # camera torn down


def test_reconstruct_live_runs_retare_save_quit(tmp_path, fake_camera,
                                                scripted_gui):
    pytest.importorskip("cv2")
    lut = np.linspace(0, 2.55, 256)
    calib = tmp_path / "calib.npz"
    depth.save_calibration(str(calib), lut, np.ones(256),
                           {"lighting_threshold": 8, "max_depth_mm": 2.0,
                            "spacing_mm": 7.4, "mm_per_px": 0.043})
    reference = np.full((60, 80, 3), 150, np.uint8)
    pressed = reference.copy()
    pressed[20:40, 30:50] = 60                       # a darker contact patch
    state = fake_camera([reference, reference, pressed, pressed])
    scripted_gui("bbsq")               # retare twice, save the pressed frame, quit

    args = types.SimpleNamespace(
        camera="fake_cam", config=None, calib=str(calib), crop_bounds=None,
        kernels=(1,), median=0, min_component=0, out=str(tmp_path / "out"))

    assert reconstruct.cmd_live(args) == 0

    height = np.load(tmp_path / "out" / "depth_0001.npy")
    assert height[30, 40] > 0.5                      # the contact has depth
    assert height[5, 5] == 0.0                       # untouched stays flat
    assert state["closed"]
