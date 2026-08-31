"""Headless tests for glowtact.camera -- no hardware, no display.

The camera class is driven through a fake read()-compatible source via
GlowTactCamera.from_capture, so the reader/diff/record logic is exercised
without a GlowTact plugged in. Discovery's sysfs parser is checked against a
synthetic /sys tree.
"""
import os
import sys
import threading
import time

import numpy as np
import pytest


from glowtact.camera import diff, render_difference, signed_peak      # noqa: E402
from glowtact.camera.camera import GlowTactCamera                     # noqa: E402


class FakeCapture:
    """A cv2.VideoCapture stand-in: hands out a counter-tinted frame each read."""

    def __init__(self, shape=(48, 64, 3), fps=200.0):
        self.shape = shape
        self._period = 1.0 / fps
        self._n = 0
        self._released = False

    def read(self):
        time.sleep(self._period)
        self._n += 1
        frame = np.full(self.shape, self._n % 256, dtype=np.uint8)
        return True, frame

    def release(self):
        self._released = True


# ------------------------------------------------------------------- diff --

def test_diff_modes_convention():
    base = np.full((4, 4, 3), 100, np.uint8)
    frame = np.full((4, 4, 3), 60, np.uint8)          # darker than baseline
    signed = render_difference(frame, base, "signed")
    dark = render_difference(frame, base, "dark")
    bright = render_difference(frame, base, "bright")
    assert np.all(signed == 128 - 40)                 # frame-base+128
    assert np.all(dark == 40)                          # base-frame, contact bright
    assert np.all(bright == 0)                          # frame-base clipped at 0
    assert signed_peak(frame, base) == 40


def test_diff_gain_and_clip():
    base = np.zeros((2, 2, 3), np.uint8)
    frame = np.full((2, 2, 3), 200, np.uint8)
    assert np.all(render_difference(frame, base, "bright", gain=2.0) == 255)


def test_diff_rejects_bad_mode():
    z = np.zeros((2, 2, 3), np.uint8)
    with pytest.raises(ValueError):
        render_difference(z, z, "nonsense")


# ------------------------------------------------------------- reader/API --

def test_reader_hands_out_fresh_frames():
    cam = GlowTactCamera.from_capture(FakeCapture())
    try:
        a = cam.read()
        b = cam.read()
        assert a is not None and b is not None
        assert a.shape == (48, 64, 3)
        # read() never replays: the counter tint advances.
        assert int(a[0, 0, 0]) != int(b[0, 0, 0])
    finally:
        cam.close()


def test_reader_is_drop_old_under_slow_consumer():
    cam = GlowTactCamera.from_capture(FakeCapture(fps=500.0))
    try:
        seqs = []
        last = -1
        for _ in range(3):
            time.sleep(0.05)                       # let many frames pile up
            seq, _f, _t = cam._reader.wait_for_new(last, timeout=1.0)
            seqs.append(seq)
            last = seq
        # Each wait jumps forward by many frames -> old ones were dropped.
        assert seqs[1] - seqs[0] > 1
        assert seqs[2] - seqs[1] > 1
    finally:
        cam.close()


def test_baseline_and_difference():
    cam = GlowTactCamera.from_capture(FakeCapture())
    try:
        cam.set_baseline()
        assert cam.baseline is not None
        d = cam.difference(cam.read(), mode="dark")
        assert d.shape == (48, 64, 3) and d.dtype == np.uint8
    finally:
        cam.close()


def test_difference_without_baseline_raises():
    cam = GlowTactCamera.from_capture(FakeCapture())
    try:
        with pytest.raises(RuntimeError):
            cam.difference(cam.read(), mode="dark")
    finally:
        cam.close()


def test_frames_generator_stops_on_close():
    cam = GlowTactCamera.from_capture(FakeCapture())
    got = []

    def consume():
        for frame in cam.frames():
            got.append(frame)
            if len(got) >= 3:
                cam.close()                        # generator must then end

    t = threading.Thread(target=consume)
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert len(got) >= 3


# --------------------------------------------------------------- discovery --

# --------------------------------------------------------- capture session --

def test_session_snapshot_writes_files_and_index(tmp_path):
    import csv as _csv

    from glowtact.camera.interactive import CaptureSession

    cam = GlowTactCamera.from_capture(FakeCapture())
    try:
        cam.set_baseline()
        session = CaptureSession(cam, tmp_path)
        raw, diff_path = session.save_snapshot(cam.read(), diff_mode="dark")
        assert os.path.isfile(raw)
        assert os.path.isfile(diff_path)          # baseline set -> diff written
        with open(session.index_path, newline="") as f:
            rows = list(_csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["type"] == "snapshot"
        assert rows[0]["diff_mode"] == "dark"
    finally:
        cam.close()


def test_session_snapshot_without_baseline_has_no_diff(tmp_path):
    from glowtact.camera.interactive import CaptureSession

    cam = GlowTactCamera.from_capture(FakeCapture())
    try:
        session = CaptureSession(cam, tmp_path)
        raw, diff_path = session.save_snapshot(cam.read(), diff_mode="dark")
        assert os.path.isfile(raw)
        assert diff_path == ""                     # no baseline -> raw only
    finally:
        cam.close()


def test_session_records_video(tmp_path):
    from glowtact.camera.interactive import CaptureSession

    cam = GlowTactCamera.from_capture(FakeCapture(fps=200.0))
    try:
        session = CaptureSession(cam, tmp_path)
        path = session.start_recording(fps=30.0, get_diff_mode=lambda: None)
        assert session.is_recording
        time.sleep(0.4)
        result = session.stop_recording(fps=30.0)
        assert result is not None
        vpath, n = result
        assert os.path.isfile(vpath) and os.path.getsize(vpath) > 0
        assert n > 0
        assert not session.is_recording
    finally:
        cam.close()


def test_sysfs_ids(tmp_path, monkeypatch):
    from glowtact.camera import discovery

    # Build a fake /sys tree: video9 -> usb device with idVendor/idProduct
    usb_dev = tmp_path / "usb" / "1-3"
    (usb_dev / "1-3:1.0").mkdir(parents=True)
    (usb_dev / "idVendor").write_text("1e45\n")
    (usb_dev / "idProduct").write_text("8022\n")
    vids = tmp_path / "sys" / "class" / "video4linux" / "video9"
    vids.mkdir(parents=True)
    os.symlink(usb_dev / "1-3:1.0", vids / "device")

    real_realpath = os.path.realpath

    def fake_realpath(p):
        if str(p) == f"/sys/class/video4linux/video9/device":
            return str(usb_dev / "1-3:1.0")
        return real_realpath(p)

    monkeypatch.setattr(discovery.os.path, "realpath", fake_realpath)
    assert discovery._sysfs_ids("/dev/video9") == ("1e45", "8022")
