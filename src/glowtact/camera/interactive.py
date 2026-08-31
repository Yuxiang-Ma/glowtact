"""Interactive keyboard-driven capture session for a GlowTact camera.

A live OpenCV window with on-screen status and single-key controls:

    s  save a snapshot (raw PNG + diff PNG when a baseline is set)
    r  start / stop video recording (clean frames, no overlay burned in)
    b  set the baseline / re-tare (do this with the gel untouched)
    m  cycle the difference view: off -> dark -> signed -> bright
    h  toggle the help overlay
    q / Esc  quit (stops an in-progress recording first)

Everything lands under a timestamped session directory below the folder you
pass in, so repeated runs never collide and an index.csv records every file:

    <data_dir>/session_YYYYMMDD_HHMMSS/
        snapshots/  snapshot_0001.png, snapshot_0001_diff.png, ...
        videos/     video_0001.avi
        index.csv

No force sensor is involved. Video is recorded on its own thread pulling the
newest frame, so display rate and record rate are independent and neither
throttles capture.
"""
import csv
import datetime as _dt
import os
import threading
import time

import numpy as np

from .diff import MODES

# Difference-view cycle: None = raw frame, then the three diff modes.
_DIFF_CYCLE = (None,) + MODES


class _VideoRecorder(threading.Thread):
    """Write the newest frames to a clip on a background thread until stopped.

    Frames are pulled straight from the camera's drop-old reader (not the
    display path), so the saved video carries clean frames with no overlay and
    is unaffected by the UI's redraw rate. The diff mode is sampled per frame,
    so re-taring mid-clip is honoured.
    """

    def __init__(self, cam, path, fps, get_diff_mode, gain, fourcc="MJPG"):
        super().__init__(daemon=True)
        import cv2

        self.cam = cam
        self.path = path
        self.get_diff_mode = get_diff_mode
        self.gain = gain
        w, h = cam.resolution
        self._writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h))
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open VideoWriter for {path}")
        self.frames_written = 0
        self.start_monotonic = time.monotonic()
        self._stop = threading.Event()

    def run(self):
        last = -1
        while not self._stop.is_set():
            item = self.cam._reader.wait_for_new(last, timeout=1.0)
            if item is None:
                continue
            last, frame, _ = item
            mode = self.get_diff_mode()
            if mode and self.cam.baseline is not None:
                out = self.cam.difference(frame, mode, self.gain)
            else:
                out = frame
            self._writer.write(out)
            self.frames_written += 1

    @property
    def elapsed(self):
        return time.monotonic() - self.start_monotonic

    def stop(self):
        self._stop.set()
        if self.is_alive():
            self.join(timeout=3.0)
        self._writer.release()


class CaptureSession:
    """Owns the session directory, the index.csv and the save/record actions.

    Split from the render loop so it can be unit-tested headless: a test drives
    ``save_snapshot`` / ``start_recording`` with a fake camera and asserts on the
    files and index rows, no window required.
    """

    def __init__(self, cam, data_dir, *, gain=1.0):
        self.cam = cam
        self.gain = gain
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(os.path.expanduser(str(data_dir)),
                                        f"session_{stamp}")
        self.snap_dir = os.path.join(self.session_dir, "snapshots")
        self.video_dir = os.path.join(self.session_dir, "videos")
        os.makedirs(self.snap_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        self.index_path = os.path.join(self.session_dir, "index.csv")
        if not os.path.exists(self.index_path):
            with open(self.index_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["id", "type", "iso_time", "file", "diff_file",
                     "diff_mode", "peak", "frames", "fps"])
        self._snap_n = 0
        self._video_n = 0
        self.recorder = None

    def _append_index(self, row):
        with open(self.index_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def save_snapshot(self, frame, diff_mode=None):
        """Write a raw PNG (+ a diff PNG when a baseline exists). Returns paths."""
        import cv2

        self._snap_n += 1
        stem = f"snapshot_{self._snap_n:04d}"
        raw_path = os.path.join(self.snap_dir, stem + ".png")
        cv2.imwrite(raw_path, frame)
        diff_path = ""
        peak = ""
        if self.cam.baseline is not None:
            from .diff import signed_peak
            peak = signed_peak(frame, self.cam.baseline)
            if diff_mode:
                diff_path = os.path.join(self.snap_dir, stem + "_diff.png")
                cv2.imwrite(diff_path,
                            self.cam.difference(frame, diff_mode, self.gain))
        self._append_index([stem, "snapshot",
                            _dt.datetime.now().isoformat(timespec="seconds"),
                            raw_path, diff_path, diff_mode or "", peak, "", ""])
        return raw_path, diff_path

    def start_recording(self, fps, get_diff_mode):
        if self.recorder is not None:
            return None
        self._video_n += 1
        path = os.path.join(self.video_dir, f"video_{self._video_n:04d}.avi")
        self.recorder = _VideoRecorder(self.cam, path, fps, get_diff_mode,
                                       self.gain)
        self.recorder.start()
        return path

    def stop_recording(self, fps):
        if self.recorder is None:
            return None
        rec = self.recorder
        rec.stop()
        self.recorder = None
        self._append_index(
            [f"video_{self._video_n:04d}", "video",
             _dt.datetime.now().isoformat(timespec="seconds"),
             str(rec.path), "", rec.get_diff_mode() or "", "",
             rec.frames_written, f"{fps:.1f}"])
        return rec.path, rec.frames_written

    @property
    def is_recording(self):
        return self.recorder is not None

    def close(self):
        if self.recorder is not None:
            self.recorder.stop()
            self.recorder = None


# ------------------------------------------------------------- render loop --

def _draw_overlay(view, lines, recording, show_help, help_lines):
    import cv2

    y = 22
    for text in lines:
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    if recording:
        cv2.circle(view, (view.shape[1] - 24, 24), 9, (0, 0, 255), -1)
    if show_help:
        hy = view.shape[0] - 12 - 20 * len(help_lines)
        for text in help_lines:
            cv2.putText(view, text, (10, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(view, text, (10, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 255, 200), 1, cv2.LINE_AA)
            hy += 20
    return view


def run_session(cam, data_dir, *, diff_mode="dark", gain=1.0, fps=None,
                window="GlowTact capture"):
    """Blocking interactive capture loop. Returns the session directory.

    ``diff_mode`` is the initial difference view (None/"" for raw); cycle it
    live with 'm'. ``fps`` defaults to the measured live rate (used for video
    playback timing).
    """
    import cv2

    if fps is None:
        fps = cam.measure_fps() or 30.0
    session = CaptureSession(cam, data_dir, gain=gain)
    if cam.baseline is None:
        cam.set_baseline()                       # first frame = untouched ref

    diff_idx = _DIFF_CYCLE.index(diff_mode) if diff_mode in _DIFF_CYCLE else 1
    show_help = True
    help_lines = ["s=snapshot  r=rec  b=baseline  m=diff-view  h=help  q=quit"]

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    print(f"session -> {session.session_dir}")
    print("keys: s=snapshot  r=record  b=baseline  m=diff-view  h=help  q=quit")

    # FPS meter over the reader's sequence numbers.
    last_seq = -1
    fps_t0 = time.monotonic()
    fps_base = 0
    disp_fps = 0.0
    try:
        while True:
            item = cam._reader.wait_for_new(last_seq, timeout=1.0)
            if item is None:
                if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                    break
                continue
            last_seq, frame, _ = item

            now = time.monotonic()
            if now - fps_t0 >= 0.5:
                disp_fps = (last_seq - fps_base) / (now - fps_t0)
                fps_t0, fps_base = now, last_seq

            mode = _DIFF_CYCLE[diff_idx]
            if mode and cam.baseline is not None:
                view = cam.difference(frame, mode, gain).copy()
                mode_label = f"diff:{mode}"
            else:
                view = frame.copy()
                mode_label = "raw"

            status = [f"{mode_label}   {disp_fps:.0f} fps   "
                      f"snaps:{session._snap_n}"]
            if session.is_recording:
                rec = session.recorder
                status.append(f"REC {rec.elapsed:5.1f}s  "
                              f"{rec.frames_written} frames")
            _draw_overlay(view, status, session.is_recording, show_help,
                          help_lines)
            cv2.imshow(window, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                raw, diff = session.save_snapshot(frame, _DIFF_CYCLE[diff_idx])
                print(f"snapshot -> {raw}" + (f"  (+ {diff})" if diff else ""))
            elif key == ord("r"):
                if session.is_recording:
                    path, n = session.stop_recording(fps)
                    print(f"recorded {n} frames -> {path}")
                else:
                    path = session.start_recording(
                        fps, lambda: _DIFF_CYCLE[diff_idx])
                    print(f"recording -> {path}  (press r again to stop)")
            elif key == ord("b"):
                cam.set_baseline(frame)
                print("baseline set")
            elif key == ord("m"):
                diff_idx = (diff_idx + 1) % len(_DIFF_CYCLE)
                print(f"diff view: {_DIFF_CYCLE[diff_idx] or 'raw'}")
            elif key == ord("h"):
                show_help = not show_help
    finally:
        session.close()
        try:
            cv2.destroyWindow(window)
        except Exception:
            pass
    return session.session_dir
