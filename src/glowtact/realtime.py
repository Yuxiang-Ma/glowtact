"""Realtime GlowTact 3D reconstruction that never blocks the camera framerate.

Three decoupled stages:

  capture thread  (CameraReader)  -- drains the camera, hands out the NEWEST
                                     frame, drops stale ones.
  reconstruct thread (ReconstructionWorker) -- newest frame -> darkening -> LUT
                                     depth -> mesh vertices; publishes a snapshot.
  render loop (MAIN thread)        -- Open3D window (GUI must be main-thread),
                                     or the OpenCV relief fallback; pulls the
                                     latest snapshot and draws it.

Because each stage runs independently and always works on the freshest data
(never a queue), a slow reconstruction or render can only DROP frames -- it can
never throttle capture. The triangle topology is constant for a fixed grid, so
per frame the worker rebuilds only the vertex z-column + shading (cheap).

Open3D is imported lazily and only in the render loop; the worker and all the
pure helpers here run without it (and without a display), so they are unit
tested headless.
"""
import dataclasses
import threading
import time

import numpy as np

from . import depth as _depth
from . import imgfilt as _imgfilt
from . import render as _render


@dataclasses.dataclass
class MeshFrame:
    """One reconstructed snapshot handed from the worker to the renderer."""
    version: int
    height: np.ndarray            # (H, W) depth map, mm
    vertices: np.ndarray          # (rows*cols, 3) strided surface, mm
    shade: np.ndarray             # (rows*cols,) 9DTact relief shade, for CV
    grid: tuple                   # (rows, cols) of the strided vertex grid
    max_mm: float


class LatestFrame:
    """A 1-slot, lock-guarded double buffer: the reader always gets the newest.

    Mirrors CameraReader's contract for meshes. `publish` overwrites; `get`
    returns the current snapshot only if it is newer than the caller's version
    (so the render loop redraws only on change). Never queues -> drop-old.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._version = 0

    def publish(self, frame):
        with self._lock:
            self._frame = frame
            self._version += 1

    def get(self, since_version):
        with self._lock:
            if self._frame is None or self._version <= since_version:
                return None
            return self._version, self._frame


def frame_geometry(height, stride, mm_per_px, z_scale):
    """height -> (vertices Nx3 mm, shade N, (rows,cols)). Pure numpy.

    x/y come from the constant grid; only z varies frame to frame. Split from
    the worker so a test can assert x/y stay fixed while z tracks the height.
    """
    x_const, y_const = _render.grid_xy(height.shape, stride, mm_per_px)
    z = height[::stride, ::stride]
    rows, cols = z.shape
    vertices = np.column_stack([x_const, y_const, z.ravel() * z_scale])
    shade = _render.surface_shading(height)[::stride, ::stride].ravel()
    return vertices, shade, (rows, cols)


def reconstruct_once(frame, reference, lut, lighting_threshold,
                     kernels=_depth.DEFAULT_KERNELS, median=0,
                     min_component_px=0):
    """frame + reference -> depth map (mm). The per-frame reconstruction step."""
    diff = _depth.darkening(frame, reference)
    return _depth.depth_from_diff(diff, lut, lighting_threshold, kernels,
                                  median=median,
                                  min_component_px=min_component_px)


class ReconstructionWorker(threading.Thread):
    """Reconstruct newest frames into MeshFrames on a background thread."""

    def __init__(self, reader, lut, stats, latest, *, crop=None,
                 kernels=_depth.DEFAULT_KERNELS, median=0, min_component_px=0,
                 stride=3, z_scale=3.0, mm_per_px=None, timeout=1.0):
        super().__init__(daemon=True)
        self.reader = reader
        self.lut = lut
        self.stats = stats
        self.latest = latest
        self.crop = crop
        self.kernels = kernels
        self.median = median
        self.min_component_px = min_component_px
        self.stride = stride
        self.z_scale = z_scale
        self.mm_per_px = (mm_per_px if mm_per_px is not None
                          else stats.get("mm_per_px", 1.0))
        self.timeout = timeout
        self._last_seq = -1
        self._reference = None
        self._version = 0
        self._retare = threading.Event()
        self._stop = threading.Event()

    def request_retare(self):
        """Ask the worker to adopt the next frame as the untouched baseline."""
        self._retare.set()

    def _step(self):
        """Pull one frame and reconstruct it. Returns a MeshFrame or None.

        Thread-free: a test drives this directly with a fake reader.
        """
        item = self.reader.wait_for_new(self._last_seq, timeout=self.timeout)
        if item is None:
            return None
        self._last_seq, frame = item[0], item[1]
        frame = _imgfilt.apply_crop(frame, self.crop)
        if self._reference is None or self._retare.is_set():
            self._reference = frame.copy()
            self._retare.clear()
        height = reconstruct_once(
            frame, self._reference, self.lut,
            self.stats.get("lighting_threshold", _depth.DEFAULT_LIGHTING_THRESHOLD),
            self.kernels, self.median, self.min_component_px)
        vertices, shade, grid = frame_geometry(
            height, self.stride, self.mm_per_px, self.z_scale)
        self._version += 1
        return MeshFrame(self._version, height, vertices, shade, grid,
                         float(height.max()))

    def run(self):
        while not self._stop.is_set():
            frame = self._step()
            if frame is not None:
                self.latest.publish(frame)

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------- render loops --

def _run_open3d(worker, latest, stop_event, *, grey=0.72, window=1280):
    """Non-blocking Open3D loop on the MAIN thread; True if it actually ran."""
    try:
        import open3d as o3d
    except ImportError:
        return False

    # Wait for the first snapshot so the mesh has geometry to add.
    first = None
    while first is None and not stop_event.is_set():
        got = latest.get(-1)
        first = got[1] if got else None
        if first is None:
            time.sleep(0.02)
    if first is None:
        return True

    mesh = o3d.geometry.TriangleMesh()
    rows, cols = first.grid
    mesh.triangles = o3d.utility.Vector3iVector(
        _render.height_grid_triangles(rows, cols))            # constant: set once
    mesh.vertices = o3d.utility.Vector3dVector(first.vertices)
    mesh.paint_uniform_color([grey, grey, grey])
    mesh.compute_vertex_normals()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("GlowTact realtime", width=window, height=int(window * 0.75))
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.light_on = True                     # real normal-based lighting, live
    opt.background_color = np.array([0.92, 0.92, 0.92])
    _render.apply_o3d_view_preset(vis, mesh.get_center())

    vis.register_key_callback(ord("B"), lambda _v: worker.request_retare() or False)
    vis.register_key_callback(ord("Q"), lambda _v: stop_event.set() or False)

    print('realtime: Open3D backend. keys: b=re-tare (sensor untouched)  '
          'q=quit  mouse=orbit')
    seen = first.version
    last_report = time.monotonic()
    frames = 0
    try:
        while not stop_event.is_set():
            got = latest.get(seen)
            if got is not None:
                seen, snap = got
                mesh.vertices = o3d.utility.Vector3dVector(snap.vertices)
                mesh.compute_vertex_normals()
                vis.update_geometry(mesh)
                frames += 1
            if not vis.poll_events():        # window closed
                break
            vis.update_renderer()
            now = time.monotonic()
            if now - last_report >= 2.0:
                print(f"realtime: render {frames / (now - last_report):.0f} fps, "
                      f"max {snap.max_mm:.2f} mm" if frames else "realtime: idle")
                frames = 0
                last_report = now
    finally:
        vis.destroy_window()
    return True


def _run_cv(worker, latest, stop_event, stats):
    """OpenCV relief fallback on the MAIN thread (no Open3D)."""
    import cv2

    renderer = _render.ReliefRenderer(width=720, height=560)
    window = "GlowTact realtime (cv)"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    renderer.attach_mouse(window)
    max_mm = stats.get("max_depth_mm", 2.0)
    print('realtime: OpenCV fallback. keys: b=re-tare  q=quit  drag=orbit')
    seen = -1
    try:
        while not stop_event.is_set():
            got = latest.get(seen)
            if got is None:
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
                continue
            seen, snap = got
            relief = renderer.render(snap.vertices, snap.shade,
                                     label=f"max {snap.max_mm:.2f} mm")
            depth_view = _depth.colorize_depth(snap.height, max_mm)
            depth_view = cv2.resize(depth_view, (relief.shape[1], relief.shape[0]))
            cv2.imshow(window, np.hstack([relief, depth_view]))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):
                worker.request_retare()
    finally:
        try:
            cv2.destroyWindow(window)
        except Exception:
            pass


def run_realtime(args):
    """Bring up the camera + worker and run the chosen render backend."""
    from .capture import open_live

    lut, stats = _depth.load_calibration(args.calib)
    crop = None
    if getattr(args, "crop_bounds", None):
        from .crop import bounds_from_file
        crop = bounds_from_file(args.crop_bounds)

    live = open_live(args.camera, getattr(args, "config", None),
                     buffer=4, fresh_reads=1)
    latest = LatestFrame()
    stop_event = threading.Event()
    worker = ReconstructionWorker(
        live.reader, lut, stats, latest, crop=crop,
        kernels=getattr(args, "kernels", _depth.DEFAULT_KERNELS),
        median=getattr(args, "median", 0),
        min_component_px=getattr(args, "min_component", 0),
        stride=getattr(args, "stride", 3),
        z_scale=getattr(args, "z_scale", 3.0))
    worker.start()

    backend = getattr(args, "backend", "auto")
    try:
        if backend in ("auto", "open3d"):
            ran = _run_open3d(worker, latest, stop_event)
            if not ran:
                if backend == "open3d":
                    raise SystemExit(
                        "Open3D not importable. Run inside the 'open3d' conda "
                        "env (and `pip install pyudev` there), or use "
                        "--backend cv.")
                _run_cv(worker, latest, stop_event, stats)
        else:
            _run_cv(worker, latest, stop_event, stats)
    finally:
        stop_event.set()
        worker.stop()
        worker.join(timeout=2.0)
        live.close()
    return 0
