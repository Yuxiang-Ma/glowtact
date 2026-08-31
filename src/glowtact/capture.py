"""Threaded newest-frame capture, and the pluggable live-camera opener.

`CameraReader` drains any camera exposing ``capture_frame()`` and always hands
out the NEWEST frame (dropping stale ones), so a slow consumer never throttles
capture. `open_live` is what the live CLIs (``reconstruct live/realtime``,
``undistort calibrate``) call to get a camera; by default it opens a
:class:`glowtact.camera.GlowTactCamera` from a ``/dev/videoN`` index or a
``vid:pid`` USB id. A host application with its own device registry (ProbingPi
and its ``config/pc_server.yaml``) plugs in with :func:`register_camera_opener`
so the same CLIs accept its camera keys instead.

cv2 is imported lazily; importing this module is cheap and cycle-free.
"""
import dataclasses
import threading
import time


class CameraReader(threading.Thread):
    """Continuously drain the camera so the pipeline never idles.

    Reading inline costs a whole frame period per iteration: `cap.grab()` waits
    for a frame, then `cap.retrieve()` spends ~25 ms decoding, and with the
    driver queue only one deep the frame that arrives during that decode has
    nowhere to go and is discarded -- so the next grab waits again. Measured on
    the GelSight Mini that halves throughput (9.4 fps vs 18.5).

    A dedicated reader plus a deeper driver queue (see `set_capture_buffer`)
    fixes it: decode of frame N overlaps the transfer of N+1. Staleness -- the
    reason the queue was one deep -- is handled by always handing out the
    NEWEST frame and never replaying one, so a deeper queue buys rate without
    buying lag.

    Frames are stamped here, at capture, rather than after the consumer picks
    them up, which also removes the consumer's latency from the timestamps.
    """

    def __init__(self, camera, to_bgr=None):
        super().__init__(daemon=True)
        self._camera = camera
        self._to_bgr = to_bgr
        self._lock = threading.Condition()
        self._frame = None
        self._seq = 0
        self._t_monotonic_ns = 0
        self._t_epoch_ns = 0
        self._stop_event = threading.Event()
        self._error = None

    def run(self):
        while not self._stop_event.is_set():
            try:
                frame = self._camera.capture_frame()
                if self._to_bgr is not None:
                    frame = self._to_bgr(frame)
            except Exception as exc:
                self._error = exc
                if self._stop_event.wait(0.05):
                    break
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1
                self._t_monotonic_ns = time.monotonic_ns()
                self._t_epoch_ns = time.time_ns()
                self._lock.notify_all()

    def latest(self):
        """(seq, frame, t_monotonic_ns, t_epoch_ns), or None before first frame."""
        with self._lock:
            if self._frame is None:
                return None
            return (self._seq, self._frame, self._t_monotonic_ns,
                    self._t_epoch_ns)

    def wait_for_new(self, last_seq, timeout=1.0):
        """Block until a frame newer than `last_seq` exists; None on timeout."""
        with self._lock:
            if not self._lock.wait_for(
                    lambda: self._frame is not None and self._seq > last_seq,
                    timeout):
                return None
            return (self._seq, self._frame, self._t_monotonic_ns,
                    self._t_epoch_ns)

    def measure_fps(self, seconds=1.5):
        """Observed delivery rate of this reader, or None if it produced nothing."""
        first = self.wait_for_new(-1, timeout=5.0)
        if first is None:
            return None
        base_seq = first[0]
        start = time.monotonic()
        if self._stop_event.wait(seconds):
            return None
        elapsed = time.monotonic() - start
        with self._lock:
            produced = self._seq - base_seq
        return (produced / elapsed) if produced and elapsed > 0 else None

    def stop(self):
        self._stop_event.set()
        with self._lock:
            self._lock.notify_all()
        if self.is_alive():                 # never-started readers can't be joined
            self.join(timeout=3.0)


def set_capture_buffer(camera, frames):
    """Deepen the driver queue so decode can overlap the next transfer.

    Only safe together with CameraReader, which always yields the newest frame;
    without it a deeper queue just serves stale frames.
    Returns the previous value, or None if the backend exposes no VideoCapture.
    """
    cap = getattr(camera, "cap", None)
    if cap is None:
        return None
    import cv2

    previous = cap.get(cv2.CAP_PROP_BUFFERSIZE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, int(frames))
    return previous


@dataclasses.dataclass
class LiveCamera:
    """What `open_live` returns: a reader plus how to tear the camera down.

    `reader` exposes ``wait_for_new(last_seq, timeout) -> (seq, frame,
    t_monotonic_ns, t_epoch_ns) | None`` and ``stop()``; that is the whole
    contract the live loops rely on.
    """
    camera_id: str
    camera: object
    reader: object
    closer: object = None

    def close(self):
        self.reader.stop()
        if self.closer is not None:
            self.closer()


class _ReaderAdapter:
    """Present a GlowTactCamera's internal reader with the 4-tuple contract."""

    def __init__(self, inner):
        self._inner = inner

    def _widen(self, item):
        if item is None:
            return None
        seq, frame, t_ns = item
        return seq, frame, t_ns, time.time_ns()

    def latest(self):
        return self._widen(self._inner.latest())

    def wait_for_new(self, last_seq, timeout=1.0):
        return self._widen(self._inner.wait_for_new(last_seq, timeout))

    def is_alive(self):
        return self._inner.is_alive()

    def stop(self):
        # The camera owns its reader thread; close() stops it. Nothing to do.
        return None


def parse_camera_spec(spec):
    """Turn a CLI camera spec into GlowTactCamera keyword arguments.

    ``None`` / ``""``  -> auto-discover the default (180 deg) unit
    ``"2"`` / ``2``    -> explicit /dev/video2
    ``"1bcf:28c4"``    -> auto-discover by USB vendor:product (hex)
    ``"1bcf:28c4:1"``  -> ... the second such unit
    """
    if spec is None or spec == "":
        return {}
    if isinstance(spec, int):
        return {"index": spec}
    text = str(spec).strip()
    if text.isdigit():
        return {"index": int(text)}
    if text.lower().startswith("/dev/video") and text[10:].isdigit():
        return {"index": int(text[10:])}
    parts = text.split(":")
    if len(parts) in (2, 3) and all(parts[:2]):
        kw = {"vendor_id": parts[0].lower(), "product_id": parts[1].lower()}
        if len(parts) == 3:
            kw["which"] = int(parts[2])
        return kw
    raise ValueError(f"camera spec {spec!r}: expected an index, /dev/videoN, "
                     f"or vendor:product[:which]")


def open_glowtact(spec=None, *, start=True, **camera_kw):
    """Open a GlowTactCamera for `spec` (see `parse_camera_spec`) as a LiveCamera."""
    from .camera import GlowTactCamera

    cam = GlowTactCamera(**parse_camera_spec(spec), **camera_kw)
    camera_id = (f"/dev/video{cam.index}" if cam.index is not None
                 else "glowtact")
    return LiveCamera(camera_id, cam, _ReaderAdapter(cam._reader), cam.close)


_OPENER = None


def register_camera_opener(opener):
    """Install a host-specific opener used by `open_live` (None resets).

    ``opener(camera, config, *, buffer, fresh_reads, start) -> LiveCamera``.
    The host decides what `camera` means (ProbingPi: a key in its
    pc_server.yaml); `config` is passed through from ``-c/--config``.
    """
    global _OPENER
    _OPENER = opener
    return opener


def open_live(camera=None, config=None, *, buffer=4, fresh_reads=1, start=True):
    """Bring up one camera for a live loop, through the registered opener.

    Without a registered opener `camera` is a GlowTact spec (index or USB id)
    and `config` is ignored.
    """
    if _OPENER is not None:
        return _OPENER(camera, config, buffer=buffer, fresh_reads=fresh_reads,
                       start=start)
    return open_glowtact(camera, start=start)
