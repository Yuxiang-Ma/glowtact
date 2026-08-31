"""Generic 180-deg fisheye correction + centre crop for GlowTact clips.

The GlowTact 180-deg lens gives a barrel-distorted circular view with black
corners and a bright aperture rim. We have no per-unit calibration grid for the
recorded sessions, and the repo's ``undistort_map.yaml`` was fit on a DIFFERENT
unit (its rectification lands rotated/off-centre here), so instead of a
calibrated warp this module uses a parameter-light analytic model:

  1. auto-detect the illuminated gel disc  -> centre (cx, cy), radius R;
  2. treat the lens as an equidistant fisheye (r_img = f * theta), with the
     disc edge R at the sensor half-FOV, so f = R / radians(sensor_fov/2);
  3. reproject the CENTRAL field to a pinhole (straight lines) and keep only
     ``keep_fov`` degrees, so the square output sits fully inside the disc --
     the fisheye is flattened AND the black rim/corners are gone in one remap.

Keeping a modest central FOV (default 90 deg) avoids the pinhole singularity at
the 90-deg edge of a true 180-deg fisheye and matches "just the centre part".
"""
import os

import numpy as np

# Per-unit calibrations ship with the package (glowtact/calib/<unit>/).
_CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib")
DEFAULT_CALIB_180 = os.path.join(_CALIB_DIR, "glowtact180",
                                 "opencv_fisheye_glowtact180.yaml")


def make_rectifier(sample_gray, calib=None, undistort=False, keep_fov=100.0,
                   out_size=480, balance=0.0, crop="auto"):
    """Pick a corrector: calibrated K/D if given, else the generic disc model.

    Returns a callable frame->frame, or None if no correction was requested.
    ``calib`` is a path to an opencv_fisheye yaml (metric, preferred);
    ``undistort`` without ``calib`` falls back to the calibration-free
    :class:`Rectifier`.
    """
    if calib:
        return CalibratedRectifier.from_yaml(calib, sample_gray=sample_gray,
                                             balance=balance, crop=crop)
    if undistort:
        return Rectifier(sample_gray, out_size=out_size, keep_fov_deg=keep_fov)
    return None


def detect_disc(gray):
    """Illuminated-gel disc as (cx, cy, R) from a (mean) grayscale frame.

    The gel is bright; the corners past the aperture are near-black. Otsu picks
    the gel, and the min-enclosing circle of the largest bright blob gives the
    aperture centre/radius. Falls back to the frame centre if detection fails.
    """
    import cv2

    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    if not cnts:
        return w / 2.0, h / 2.0, min(h, w) / 2.0
    (cx, cy), R = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    return float(cx), float(cy), float(R)


def mean_frame(video_path, n=30):
    """Average up to ``n`` frames -> a stable grayscale image for disc detection."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
    idxs = np.linspace(0, max(total - 1, 0), min(n, total)).astype(int)
    acc, cnt = None, 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        acc = g if acc is None else acc + g
        cnt += 1
    cap.release()
    if acc is None:
        raise RuntimeError(f"no frames in {video_path}")
    return (acc / cnt).astype(np.uint8)


def build_maps(cx, cy, R, out_size=360, keep_fov_deg=90.0,
               sensor_fov_deg=180.0):
    """Dense (map_x, map_y) reprojecting the central fisheye field to a pinhole.

    ``keep_fov_deg`` is the full horizontal FOV retained in the square output.
    A corner is at angle atan(sqrt(2)*tan(keep_fov/2)); as long as that stays
    below sensor_fov/2 the whole output samples inside the disc (no black
    edges). Default 90 deg on a 180-deg lens keeps corners at ~55 deg < 90 deg.
    """
    f_fish = R / np.radians(sensor_fov_deg / 2.0)
    half = np.radians(keep_fov_deg / 2.0)
    f_out = (out_size / 2.0) / np.tan(half)
    j, i = np.meshgrid(np.arange(out_size), np.arange(out_size))
    x = (j - out_size / 2.0) / f_out
    y = (i - out_size / 2.0) / f_out
    r = np.hypot(x, y)
    theta = np.arctan(r)                       # pinhole ray angle
    psi = np.arctan2(y, x)
    r_src = f_fish * theta                      # equidistant fisheye
    map_x = (cx + r_src * np.cos(psi)).astype(np.float32)
    map_y = (cy + r_src * np.sin(psi)).astype(np.float32)
    return map_x, map_y


def _gel_crop_box(gray, max_trim=0.14, dark_frac=0.30, safety=0.01):
    """Largest gel-square crop, trimming an edge inward only if it's on a band.

    After a calibrated undistort the gel is an axis-aligned bright square with
    dark holder bands at some sides. Otsu + largest-contour bounding-rect gives
    the bright extent; we then keep each edge AT that boundary and only pull it
    inward while the edge line is mostly dark (a holder band), up to
    ``max_trim`` of the side. So clean edges (usually top/bottom/left) stay wide
    and only a band edge (usually the right) is trimmed -- no uniform inset that
    needlessly shrinks the good sides. A tiny ``safety`` inset trims the last
    rim pixel. ``dark_frac`` = fraction of an edge line below the Otsu threshold
    that counts as "on a band".
    """
    import cv2

    thr, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    if not cnts:
        return 0, 0, w, h
    x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    x0, y0, x1, y1 = x, y, x + bw, y + bh
    dark = gray < thr
    mtx, mty = int(bw * max_trim), int(bh * max_trim)

    lim = x0 + mtx
    while x0 < lim and dark[y0:y1, x0].mean() > dark_frac:
        x0 += 1
    lim = x1 - mtx
    while x1 > lim and dark[y0:y1, x1 - 1].mean() > dark_frac:
        x1 -= 1
    lim = y0 + mty
    while y0 < lim and dark[y0, x0:x1].mean() > dark_frac:
        y0 += 1
    lim = y1 - mty
    while y1 > lim and dark[y1 - 1, x0:x1].mean() > dark_frac:
        y1 -= 1

    sx, sy = int((x1 - x0) * safety), int((y1 - y0) * safety)
    return (x0 + sx, y0 + sy, x1 - sx, y1 - sy)


class CalibratedRectifier:
    """Metric fisheye undistort from a saved OpenCV Kannala-Brandt K/D, cropped.

    Uses the per-unit calibration (e.g. glowtact/calib/glowtact180/
    opencv_fisheye_glowtact180.yaml, RMS ~1.6 px) so straight edges come back
    straight -- unlike the calibration-free :class:`Rectifier`. ``balance`` in
    [0,1] trades FOV vs black borders (0 = no black). After undistort the frame
    is centre-cropped to the detected gel square (``crop='auto'``), a fractional
    box, or left full (``crop=None``).
    """

    def __init__(self, K, D, size, sample_gray=None, balance=0.0, crop="auto"):
        import cv2

        self.K = np.asarray(K, np.float64)
        self.D = np.asarray(D, np.float64).reshape(4, 1)
        w, h = size
        new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, (w, h), np.eye(3), balance=float(balance))
        self.map_x, self.map_y = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), new_k, (w, h), cv2.CV_32FC1)
        self.box = None
        if crop == "auto" and sample_gray is not None:
            rect = cv2.remap(sample_gray, self.map_x, self.map_y,
                             cv2.INTER_LINEAR)
            self.box = _gel_crop_box(rect)
        elif isinstance(crop, (tuple, list)):
            self.box = tuple(int(v) for v in crop)

    def __call__(self, frame):
        import cv2

        rect = cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR)
        if self.box:
            x0, y0, x1, y1 = self.box
            rect = rect[y0:y1, x0:x1]
        return rect

    @classmethod
    def from_yaml(cls, path, sample_gray=None, balance=None, crop="auto"):
        """Load an ``opencv_fisheye`` map yaml (K, D, size) and build the maps."""
        import yaml

        with open(path) as f:
            m = yaml.safe_load(f)
        if m.get("backend") != "opencv_fisheye":
            raise ValueError(f"{path} is not an opencv_fisheye calibration")
        bal = balance if balance is not None else m.get("balance", 0.0)
        return cls(m["K"], m["D"], (m["source_width"], m["source_height"]),
                   sample_gray=sample_gray, balance=bal, crop=crop)


class Rectifier:
    """Reusable frame corrector: detect the disc once, remap every frame."""

    def __init__(self, sample_gray, out_size=360, keep_fov_deg=90.0,
                 sensor_fov_deg=180.0, disc=None):
        self.cx, self.cy, self.R = disc or detect_disc(sample_gray)
        self.out_size = out_size
        self.map_x, self.map_y = build_maps(
            self.cx, self.cy, self.R, out_size, keep_fov_deg, sensor_fov_deg)

    def __call__(self, frame):
        import cv2

        return cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR)

    @classmethod
    def from_video(cls, video_path, **kw):
        return cls(mean_frame(video_path), **kw)
