"""GlowTact camera: open, stream, snapshot, record -- no force sensor needed.

Self-contained: depends only on OpenCV + NumPy. A background reader thread
drains the camera continuously and always hands out the NEWEST frame (dropping
stale ones), so a slow consumer -- a depth reconstruction, a JPEG encoder, a
browser -- can only drop frames, never throttle capture or accumulate lag.

    from glowtact.camera import GlowTactCamera

    with GlowTactCamera() as cam:          # auto-finds the 180 deg unit by USB id
        cam.set_baseline()                 # capture untouched reference
        for frame in cam.frames():         # newest-frame generator
            diff = cam.difference(frame, mode="dark")
            ...

The reader accepts anything with a cv2.VideoCapture-style ``read() -> (ok,
frame)``, so it also wraps an already-open capture (integration / testing)
without any hardware.
"""
import threading
import time

import numpy as np

from . import diff as _diff
from .discovery import (GLOWTACT_180_PID, GLOWTACT_180_VID, find_glowtact)

DEFAULT_RESOLUTION = (640, 480)      # 4:3 full sensor FOV; a 16:9 mode crops it
DEFAULT_FOURCC = "MJPG"


class _LatestFrameReader(threading.Thread):
    """Drain a capture on a thread; always expose the newest (seq, frame).

    Mirrors the parent project's CameraReader contract in a standalone form.
    ``capture`` is any object with ``read() -> (ok, ndarray)``.
    """

    def __init__(self, capture):
        super().__init__(daemon=True)
        self._cap = capture
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._t_ns = 0
        self._stop = threading.Event()
        self._error = None

    def run(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._error = RuntimeError("frame grab failed")
                if self._stop.wait(0.05):
                    break
                continue
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._t_ns = time.monotonic_ns()
                self._cond.notify_all()

    def latest(self):
        with self._cond:
            if self._frame is None:
                return None
            return self._seq, self._frame, self._t_ns

    def wait_for_new(self, last_seq, timeout=2.0):
        with self._cond:
            if not self._cond.wait_for(
                    lambda: self._frame is not None and self._seq > last_seq,
                    timeout):
                return None
            return self._seq, self._frame, self._t_ns

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self.is_alive():
            self.join(timeout=3.0)


class GlowTactCamera:
    """A live GlowTact camera. Force sensor NOT required for anything here.

    Args:
        index: explicit /dev/videoN index. If None, auto-discover by USB id
            (Linux only) -- see ``vendor_id``/``product_id``.
        vendor_id, product_id: USB ids for auto-discovery; default = 180 deg
            GlowTact. For the 120 deg builds pass 1bcf:28c4 or 337b:090c.
        resolution: (w, h) requested capture size. 640x480 keeps the full FOV.
        fourcc: pixel format four-char code; MJPG is fast over USB.
        which: pick among multiple identical units (0 = first).
    """

    def __init__(self, index=None, *, vendor_id=GLOWTACT_180_VID,
                 product_id=GLOWTACT_180_PID, resolution=DEFAULT_RESOLUTION,
                 fourcc=DEFAULT_FOURCC, which=0, warmup=5):
        import cv2

        if index is None:
            index = find_glowtact(vendor_id, product_id, which=which)
        self.index = index
        self._cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open /dev/video{index}")
        # BUFFERSIZE=1 + a draining reader = freshest frame, no backlog lag.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        w, h = resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.resolution = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._reader = _LatestFrameReader(self._cap)
        self._reader.start()
        self.baseline = None
        self._await_first(warmup)

    # -- construction helpers -------------------------------------------------

    @classmethod
    def from_capture(cls, capture):
        """Wrap an already-open cv2.VideoCapture (or a read()-compatible stub).

        Skips discovery and camera config entirely -- for embedding in another
        pipeline, or driving the reader from a fake source in tests.
        """
        self = cls.__new__(cls)
        self.index = None
        self._cap = capture
        self.baseline = None
        self._reader = _LatestFrameReader(capture)
        self._reader.start()
        # Resolution isn't queryable on an arbitrary source, so take it from
        # the first frame (recording needs a fixed (w, h)).
        first = self._reader.wait_for_new(-1, timeout=2.0)
        if first is None:
            self._reader.stop()
            raise RuntimeError("wrapped capture produced no frames")
        h, w = first[1].shape[:2]
        self.resolution = (w, h)
        return self

    def _await_first(self, warmup):
        for _ in range(max(1, warmup)):
            if self._reader.wait_for_new(-1, timeout=1.0) is not None:
                return
        raise RuntimeError("camera produced no frames during warm-up")

    # -- frame access ---------------------------------------------------------

    def read(self, timeout=2.0):
        """Newest frame as a copy, or None on timeout. Never replays a frame."""
        item = self._reader.wait_for_new(getattr(self, "_last_seq", -1), timeout)
        if item is None:
            return None
        self._last_seq, frame, _ = item
        return frame.copy()

    def frames(self, timeout=2.0):
        """Generator yielding each new frame until the camera is closed."""
        last = -1
        while not self._reader._stop.is_set():
            item = self._reader.wait_for_new(last, timeout)
            if item is None:
                continue
            last, frame, _ = item
            yield frame.copy()

    def measure_fps(self, seconds=1.5):
        """Observed delivery rate (Hz), or None if nothing arrived."""
        first = self._reader.wait_for_new(-1, timeout=5.0)
        if first is None:
            return None
        base = first[0]
        time.sleep(seconds)
        latest = self._reader.latest()
        produced = latest[0] - base if latest else 0
        return produced / seconds if produced > 0 else None

    # -- difference imaging ---------------------------------------------------

    def set_baseline(self, frame=None):
        """Adopt an untouched reference for difference imaging (the 'tare').

        Uses ``frame`` if given, else grabs the current one. Call it with the
        gel NOT in contact.
        """
        if frame is None:
            frame = self.read()
            if frame is None:
                raise RuntimeError("no frame available to set as baseline")
        self.baseline = frame.copy()
        return self.baseline

    def difference(self, frame, mode="dark", gain=1.0):
        """Diff of ``frame`` against the stored baseline (set_baseline first)."""
        if self.baseline is None:
            raise RuntimeError("call set_baseline() before difference()")
        return _diff.render_difference(frame, self.baseline, mode, gain)

    # -- saving ---------------------------------------------------------------

    def snapshot(self, path, *, diff_mode=None, gain=1.0):
        """Save one newest frame to ``path``. Returns the frame (BGR ndarray).

        With ``diff_mode`` set, saves the difference image instead of the raw
        frame (needs a baseline). PNG keeps it lossless; .jpg is smaller.
        """
        import cv2

        frame = self.read()
        if frame is None:
            raise RuntimeError("no frame to snapshot")
        out = (self.difference(frame, diff_mode, gain)
               if diff_mode else frame)
        if not cv2.imwrite(str(path), out):
            raise RuntimeError(f"cv2 could not write {path}")
        return frame

    def record(self, path, duration_s, *, fps=None, fourcc="MJPG",
               diff_mode=None, gain=1.0, on_frame=None):
        """Record a clip to ``path`` for ``duration_s`` seconds.

        ``fps`` defaults to the measured live rate so playback is real-time.
        With ``diff_mode`` the difference stream is written (needs a baseline).
        ``on_frame(raw_frame)`` is called per written frame (e.g. progress).
        Returns the number of frames written.
        """
        import cv2

        if fps is None:
            fps = self.measure_fps() or 30.0
        first = self.read()
        if first is None:
            raise RuntimeError("no frame to record")
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc),
                                 float(fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"cv2 could not open VideoWriter for {path}")
        written = 0
        end = time.monotonic() + duration_s
        try:
            for frame in self.frames():
                out = (self.difference(frame, diff_mode, gain)
                       if diff_mode else frame)
                writer.write(out)
                written += 1
                if on_frame is not None:
                    on_frame(frame)
                if time.monotonic() >= end:
                    break
        finally:
            writer.release()
        return written

    # -- lifecycle ------------------------------------------------------------

    def close(self):
        self._reader.stop()
        if getattr(self, "_cap", None) is not None:
            try:
                self._cap.release()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
