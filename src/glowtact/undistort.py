"""Rectify a fisheye tactile view by pressing a known grid into the gel.

Everything of interest lies on ONE plane -- the gel surface -- and the camera is
rigidly fixed to it. That makes full fisheye intrinsics unnecessary: a single
2D->2D warp from the distorted image of that plane to a rectified view is both
sufficient and far easier to fit, and it absorbs lens distortion and viewing
obliquity together.

The warp is a polynomial fitted to correspondences between detected grid points
and the regular lattice they are known to form. No camera model is assumed, so
it copes with a 180-degree lens that a pinhole+k1..k4 model fits poorly.

    # 1. press a rigid grid into the gel, capture it, then fit:
    glowtact-undistort fit shot.png --rows 7 --cols 7
    #    add --pattern chessboard if you pressed a checkerboard
    #    add --show to eyeball the detection and the result

    # 2. apply to every image in a folder:
    glowtact-undistort apply ~/tactile_data/glowtact/sessions/session_.../snapshots

Accuracy is reported as an RMS residual in output pixels. This targets visual
consistency; it is not a metric calibration, and no millimetre scale is claimed.
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml


DEFAULT_MAP_NAME = "undistort_map.yaml"
# Degree 5, not 3. A fisheye's radial correction expands in ODD powers of r, so
# going 3 -> 4 buys nothing (measured identical residuals) while 3 -> 5 cuts the
# residual ~3x on a synthetic 180-degree lens (4.9 px -> 1.5 px). Degree 7 is
# better still (0.4 px) but needs 36 coefficients per axis, so only use it with
# a dense grid.
DEGREE = 5


def num_coefficients(degree):
    return (degree + 1) * (degree + 2) // 2


def poly_features(x, y, degree=DEGREE):
    """Basis [1, y, y^2, ..., x, xy, ..., x^degree] for 2D polynomial fitting."""
    terms = [(x ** i) * (y ** j)
             for i in range(degree + 1)
             for j in range(degree + 1 - i)]
    return np.stack(terms, axis=-1)


def fit_warp(dst_points, src_points, degree=DEGREE):
    """Fit dst -> src, the direction cv2.remap needs (it samples the source).

    Coordinates are normalised to roughly [-1, 1] first; fitting a cubic on raw
    pixel values is badly conditioned and the high-order terms come out noise.
    """
    dst = np.asarray(dst_points, dtype=np.float64)
    src = np.asarray(src_points, dtype=np.float64)
    needed = num_coefficients(degree)
    if len(dst) < needed:
        raise ValueError(
            f"degree {degree} needs >= {needed} points, got {len(dst)}; "
            f"press a denser grid or lower --degree")
    centre = dst.mean(axis=0)
    scale = np.abs(dst - centre).max()
    if scale <= 0:
        raise ValueError("degenerate destination points")
    normalised = (dst - centre) / scale
    basis = poly_features(normalised[:, 0], normalised[:, 1], degree)
    coef_x, *_ = np.linalg.lstsq(basis, src[:, 0], rcond=None)
    coef_y, *_ = np.linalg.lstsq(basis, src[:, 1], rcond=None)
    return {"centre": centre.tolist(), "scale": float(scale),
            "degree": int(degree),
            "coef_x": coef_x.tolist(), "coef_y": coef_y.tolist()}


def _apply_radial(warp, xs, ys):
    """Rectified plane -> homography -> radial distortion -> image."""
    matrix = np.asarray(warp["homography"], dtype=np.float64).reshape(3, 3)
    pts = np.stack([np.asarray(xs, dtype=np.float64),
                    np.asarray(ys, dtype=np.float64),
                    np.ones(np.size(xs))], axis=-1)
    projected = pts @ matrix.T
    projected = projected[:, :2] / projected[:, 2:3]
    cx, cy = warp["centre"]
    k1, k2, k3 = warp["k"]
    delta = projected - np.array([cx, cy])
    r2 = (delta ** 2).sum(axis=1)
    scale = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    distorted = np.array([cx, cy]) + delta * scale[:, None]
    return distorted[:, 0], distorted[:, 1]


def apply_warp(warp, xs, ys):
    """Map output coordinates to source coordinates."""
    pre = warp.get("pre_transform")
    if pre is not None:
        matrix = np.asarray(pre, dtype=np.float64).reshape(3, 3)
        pts = np.stack([np.asarray(xs, dtype=np.float64),
                        np.asarray(ys, dtype=np.float64),
                        np.ones(np.size(xs))], axis=-1) @ matrix.T
        xs, ys = pts[:, 0] / pts[:, 2], pts[:, 1] / pts[:, 2]
    if warp.get("model") == "fisheye":
        return _apply_fisheye(warp, xs, ys)
    if warp.get("model") == "radial":
        return _apply_radial(warp, xs, ys)
    centre = np.asarray(warp["centre"], dtype=np.float64)
    basis = poly_features((xs - centre[0]) / warp["scale"],
                          (ys - centre[1]) / warp["scale"], warp["degree"])
    return (basis @ np.asarray(warp["coef_x"]),
            basis @ np.asarray(warp["coef_y"]))


def fit_radial(dst_points, src_points, width, height):
    """Fit plane-homography + radial distortion -- the physical camera model.

    Preferred over the free polynomial because it EXTRAPOLATES. Radial
    distortion is a smooth monotonic function of radius with three coefficients,
    so it stays sensible outside the calibrated area; a 21-coefficient 2D
    polynomial has no such constraint and folds over. Measured on real data
    (54 balls, 180-degree lens): the polynomial fitted the grid slightly better
    (2.82 px vs 3.47 px) but mapped the top-left image corner to source x=975
    and the bottom-left to x=-470 -- a fold, not a correction -- while the
    radial model kept all four corners correctly ordered.
    """
    from scipy.optimize import least_squares

    dst = np.asarray(dst_points, dtype=np.float64)
    src = np.asarray(src_points, dtype=np.float64)

    def unpack(params):
        return {"model": "radial",
                "centre": [float(params[0]), float(params[1])],
                "k": [float(params[2]), float(params[3]), float(params[4])],
                "homography": np.append(params[5:], 1.0).tolist()}

    def residual(params):
        px, py = _apply_radial(unpack(params), dst[:, 0], dst[:, 1])
        return np.concatenate([px - src[:, 0], py - src[:, 1]])

    start = np.r_[width / 2.0, height / 2.0, 0.0, 0.0, 0.0,
                  np.eye(3).ravel()[:8]]
    solution = least_squares(residual, start, method="lm", max_nfev=20000)
    return unpack(solution.x)


def _apply_fisheye(warp, xs, ys):
    """Rectified plane -> homography -> Kannala-Brandt fisheye -> image.

    theta = atan(r) then theta_d = theta*(1 + k1*t^2 + k2*t^4 + k3*t^6 + k4*t^8).
    Because the angle, not the radius, carries the polynomial, the map stays
    well behaved out to 180 degrees -- which is the whole reason this model is
    the standard for wide lenses and a pinhole+radial series is not.
    """
    matrix = np.asarray(warp["homography"], dtype=np.float64).reshape(3, 3)
    pts = np.stack([np.asarray(xs, dtype=np.float64),
                    np.asarray(ys, dtype=np.float64),
                    np.ones(np.size(xs))], axis=-1)
    projected = pts @ matrix.T
    nx = projected[:, 0] / projected[:, 2]
    ny = projected[:, 1] / projected[:, 2]

    r = np.hypot(nx, ny)
    theta = np.arctan(r)
    k1, k2, k3, k4 = warp["k"]
    t2 = theta * theta
    theta_d = theta * (1.0 + k1 * t2 + k2 * t2 ** 2
                       + k3 * t2 ** 3 + k4 * t2 ** 4)
    # r -> 0 is the optical axis; the scale factor tends to 1 there.
    scale = np.where(r > 1e-12, theta_d / np.maximum(r, 1e-12), 1.0)
    focal, cx, cy = warp["focal"], warp["centre"][0], warp["centre"][1]
    return cx + focal * scale * nx, cy + focal * scale * ny


def fit_fisheye_multi(observations, width, height):
    """Joint Kannala-Brandt fit: shared fisheye, per-press in-plane homography.

    cv2.fisheye.calibrate cannot be used here. It needs the target seen at
    different ANGLES to separate intrinsics from extrinsics, but this camera is
    fixed inside the sensor looking at one plane, so every press is the same
    pose translated -- a degenerate configuration that makes its extrinsic
    initialisation fail outright ("fabs(norm_u1) > 0").

    Parameterising the in-plane motion as a homography per press sidesteps
    that: the geometry that is genuinely shared (focal, centre, k1..k4) is
    fitted jointly, and each press only contributes where it landed.
    """
    from scipy.optimize import least_squares

    pairs = [(np.asarray(i, dtype=np.float64), np.asarray(d, dtype=np.float64))
             for i, d in observations]
    count = len(pairs)

    def unpack(params, index):
        return {"model": "fisheye",
                "focal": float(params[0]),
                "centre": [float(params[1]), float(params[2])],
                "k": [float(params[3]), float(params[4]),
                      float(params[5]), float(params[6])],
                "homography": np.append(
                    params[7 + 8 * index: 15 + 8 * index], 1.0).tolist()}

    def residual(params):
        out = []
        for index, (ideal, detected) in enumerate(pairs):
            px, py = _apply_fisheye(unpack(params, index),
                                    ideal[:, 0], ideal[:, 1])
            out.append(px - detected[:, 0])
            out.append(py - detected[:, 1])
        return np.concatenate(out)

    # Start from a plausible focal and an identity in-plane mapping scaled so
    # the lattice lands near the image centre.
    span = max(width, height)
    start_homography = np.array([1.0 / span, 0, -0.5, 0, 1.0 / span, -0.5,
                                 0, 0])
    start = np.r_[span / 2.0, width / 2.0, height / 2.0, 0.0, 0.0, 0.0, 0.0,
                  np.tile(start_homography, count)]
    solution = least_squares(residual, start, method="lm", max_nfev=200000)

    per_press = [warp_residual(unpack(solution.x, i), ideal, detected)
                 for i, (ideal, detected) in enumerate(pairs)]
    centres = [d.mean(axis=0) for _i, d in pairs]
    middle = int(np.argmin([np.hypot(c[0] - width / 2, c[1] - height / 2)
                            for c in centres]))
    return unpack(solution.x, middle), per_press


def fit_radial_multi(observations, width, height):
    """Joint fit over several presses: shared distortion, per-press homography.

    Pressing the array once covers only part of the field, and outside that the
    map is extrapolation. Pressing it at several positions extends the coverage
    -- but each press sits at an unknown place on the gel, so the point sets
    cannot simply be pooled against one lattice.

    The structure mirrors standard camera calibration: the distortion centre
    and k1..k3 are physical properties of the lens and are SHARED across every
    press, while each press gets its own homography to absorb where it landed.
    That is what lets presses at different positions reinforce one estimate of
    the distortion instead of fighting each other.

    `observations` is a list of (ideal_points, detected_points).
    Returns (warp, per_press_rms). The returned warp carries the shared
    distortion with the most central press's homography, which only sets the
    output framing.
    """
    from scipy.optimize import least_squares

    pairs = [(np.asarray(i, dtype=np.float64), np.asarray(d, dtype=np.float64))
             for i, d in observations]
    if not pairs:
        raise ValueError("no observations to fit")
    count = len(pairs)

    def unpack(params, index):
        return {"model": "radial",
                "centre": [float(params[0]), float(params[1])],
                "k": [float(params[2]), float(params[3]), float(params[4])],
                "homography": np.append(
                    params[5 + 8 * index: 13 + 8 * index], 1.0).tolist()}

    def residual(params):
        out = []
        for index, (ideal, detected) in enumerate(pairs):
            px, py = _apply_radial(unpack(params, index),
                                   ideal[:, 0], ideal[:, 1])
            out.append(px - detected[:, 0])
            out.append(py - detected[:, 1])
        return np.concatenate(out)

    start = np.r_[width / 2.0, height / 2.0, 0.0, 0.0, 0.0,
                  np.tile(np.eye(3).ravel()[:8], count)]
    solution = least_squares(residual, start, method="lm", max_nfev=100000)

    per_press = []
    for index, (ideal, detected) in enumerate(pairs):
        warp = unpack(solution.x, index)
        per_press.append(warp_residual(warp, ideal, detected))

    # Frame the output on whichever press sat closest to the image centre.
    centres = [d.mean(axis=0) for _i, d in pairs]
    middle = np.argmin([np.hypot(c[0] - width / 2, c[1] - height / 2)
                        for c in centres])
    return unpack(solution.x, int(middle)), per_press


def coverage_fraction(all_points, width, height):
    """Fraction of the frame inside the convex hull of the calibration points.

    The honest measure of a calibration's reach. Counting cells that contain a
    point undercounts badly -- the points are a sparse lattice by construction,
    so even perfect coverage would score under 20%. What matters is the region
    the points ENCLOSE, because inside the hull the fit interpolates and
    outside it extrapolates.
    """
    import cv2

    pts = np.asarray(all_points, dtype=np.float32)
    if len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    # Clip to the frame: points can sit outside it and that area is not covered.
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(canvas, hull.astype(np.int32), 1)
    return float(canvas.mean())


def warp_residual(warp, dst_points, src_points):
    """RMS distance, in source pixels, between fitted and observed points."""
    dst = np.asarray(dst_points, dtype=np.float64)
    px, py = apply_warp(warp, dst[:, 0], dst[:, 1])
    src = np.asarray(src_points, dtype=np.float64)
    return float(np.sqrt(((px - src[:, 0]) ** 2 + (py - src[:, 1]) ** 2).mean()))


def ideal_lattice(rows, cols, width, height, fill=0.8, asymmetric=False):
    """The regular grid the pressed pattern is known to form, centred output.

    Point order must match what findCirclesGrid returns, or the correspondence
    is scrambled and the fitted warp is nonsense. For an ASYMMETRIC grid that
    means a staggered lattice: consecutive rows are offset half a step and the
    row pitch is half the column pitch -- not the rectangular grid a symmetric
    pattern produces.
    """
    span = min(width, height) * fill
    if asymmetric:
        step = span / max(cols - 1, (rows - 1) / 2.0)
        points = []
        for row in range(rows):
            offset = (row % 2) * step / 2.0
            for col in range(cols):
                points.append((
                    (col - (cols - 1) / 2) * step + offset + width / 2,
                    (row - (rows - 1) / 2) * step / 2.0 + height / 2))
        return np.asarray(points, dtype=np.float64)
    step = span / max(rows - 1, cols - 1)
    xs = (np.arange(cols) - (cols - 1) / 2) * step + width / 2
    ys = (np.arange(rows) - (rows - 1) / 2) * step + height / 2
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return grid.astype(np.float64)


def canonicalize_grid(points, rows, cols):
    """Reorder detected points to a fixed top-to-bottom, left-to-right order.

    A SYMMETRIC grid is invariant under 180-degree rotation and mirroring, so
    findCirclesGrid has no way to pick a canonical corner -- it can and does
    return a different orientation from frame to frame. Fitting that directly
    produces a warp that mirrors the image on some frames and not others, with
    an identical residual either way (observed live: +0.93 on one press, -0.92
    on the next).

    Reordering here makes the fit deterministic regardless of what the detector
    happened to return. Asymmetric grids are not ambiguous, but normalising them
    the same way is harmless.
    """
    grid = np.asarray(points, dtype=np.float64).reshape(rows, cols, 2)
    if grid[0, :, 1].mean() > grid[-1, :, 1].mean():
        grid = grid[::-1]                       # rows ran bottom-to-top
    if grid[:, 0, 0].mean() > grid[:, -1, 0].mean():
        grid = grid[:, ::-1]                    # columns ran right-to-left
    return grid.reshape(-1, 2)


def point_residuals(warp, dst_points, src_points):
    """Per-point fit error, for spotting a few mislocated detections."""
    dst = np.asarray(dst_points, dtype=np.float64)
    src = np.asarray(src_points, dtype=np.float64)
    px, py = apply_warp(warp, dst[:, 0], dst[:, 1])
    return np.hypot(px - src[:, 0], py - src[:, 1])


def fit_warp_robust(dst_points, src_points, degree=DEGREE, reject_sigma=4.0,
                    iterations=2, screen_degree=1, model="radial",
                    width=None, height=None):
    """Fit, drop points whose error is a clear outlier, refit.

    A blob detector occasionally locks onto the wrong feature -- on this sensor,
    the bright ring around the gel. Rejection is MEDIAN + k*MAD, not mean+k*std:
    the outliers inflate the standard deviation that is supposed to expose them,
    so a mean-based threshold widens to accommodate the very points it should
    reject. Observed on real data: two ring artefacts survived mean+3*std, and
    were removed immediately by the median/MAD rule.

    Screening happens at `screen_degree`, NOT at the final degree. A degree-5
    polynomial has 21 coefficients and simply bends through a bad point, so it
    reports no outlier at all while quietly corrupting the whole map. A rigid
    low-order model leaves the bad points nowhere to hide. Measured on real
    data with two ring artefacts among 54 balls: degree 1 flagged exactly those
    two, degree 3 flagged both, degree 5 flagged none.

    Returns (warp, keep_mask).
    """
    dst = np.asarray(dst_points, dtype=np.float64)
    src = np.asarray(src_points, dtype=np.float64)
    keep = np.ones(len(dst), dtype=bool)
    screen = max(1, min(screen_degree, degree))

    for _ in range(max(1, iterations)):
        coarse = fit_warp(dst[keep], src[keep], screen)
        errors = point_residuals(coarse, dst, src)
        median = np.median(errors[keep])
        mad = np.median(np.abs(errors[keep] - median))
        # 1.4826 makes MAD a consistent estimator of sigma for normal errors.
        threshold = median + reject_sigma * 1.4826 * mad
        candidate = errors <= max(threshold, 1e-6)
        if candidate.sum() < num_coefficients(degree):
            break                       # refusing to fit on too few points
        if np.array_equal(candidate, keep):
            break                       # converged
        keep = candidate
    if model == "radial":
        return fit_radial(dst[keep], src[keep], width, height), keep
    return fit_warp(dst[keep], src[keep], degree), keep


def warp_rotation(warp, width, height, step=50.0):
    """Rotation, in degrees, that the fit bakes into its output.

    The fit maps a lattice that is axis-aligned BY CONSTRUCTION onto whatever
    orientation the pattern happened to be pressed at, so pressing the array
    tilted rotates every rectified image by that tilt. Measured by walking one
    step along the output x-axis and reading the direction that moves in the
    source.
    """
    cx, cy = width / 2.0, height / 2.0
    x0, y0 = apply_warp(warp, np.array([cx]), np.array([cy]))
    x1, y1 = apply_warp(warp, np.array([cx + step]), np.array([cy]))
    return float(np.degrees(np.arctan2(y1[0] - y0[0], x1[0] - x0[0])))


def align_warp(warp, width, height):
    """Cancel the press rotation so the output is axis-aligned with the input.

    Without this the rectified orientation depends on how the operator happened
    to lay the array down -- press it 30 degrees off and every rectified frame
    comes out 30 degrees off, and two calibration sessions disagree. Composing
    the inverse rotation into the map makes the result reproducible.

    The correction is modulo 180 degrees, and that limit is fundamental rather
    than an oversight: a SYMMETRIC grid rotated 180 degrees is identical to
    itself, so nothing in the observations can say which way is up. The output
    is guaranteed axis-aligned but may be upside down. Use an asymmetric
    pattern if the absolute orientation has to be pinned down.
    """
    # NEGATIVE: the model already rotates by +angle, so the pre-transform must
    # rotate by -angle for the composition to come out axis-aligned. Using
    # +angle here doubles the tilt instead of cancelling it.
    measured = warp_rotation(warp, width, height)
    # Fold into (-90, 90]: a 90 degree press is just the grid's own symmetry,
    # and rotating the output by a quarter turn to "fix" it would needlessly
    # swap the image's width and height.
    measured = (measured + 90.0) % 180.0 - 90.0
    angle = -np.deg2rad(measured)
    cos, sin = np.cos(angle), np.sin(angle)
    cx, cy = width / 2.0, height / 2.0
    # Rotate output coordinates about the image centre before the model runs.
    rotation = np.array([[cos, -sin, cx - cos * cx + sin * cy],
                         [sin, cos, cy - sin * cx - cos * cy],
                         [0.0, 0.0, 1.0]])
    existing = warp.get("pre_transform")
    if existing is not None:
        rotation = np.asarray(existing, dtype=np.float64).reshape(3, 3) @ rotation
    aligned = dict(warp)
    aligned["pre_transform"] = rotation.tolist()
    return aligned


def zoom_warp(warp, width, height, zoom):
    """Scale the output about the image centre. zoom < 1 shows MORE.

    Correcting barrel distortion magnifies the centre and pushes the periphery
    past the frame edge, so a rectified image looks cropped even when it is
    sampling the whole source. Scaling the output coordinates outward brings
    that periphery back into view, at the cost of the subject being smaller --
    the same trade-off OpenCV exposes as the `balance` of
    estimateNewCameraMatrixForUndistortRectify.
    """
    if zoom <= 0:
        raise ValueError("zoom must be > 0")
    factor = 1.0 / float(zoom)
    cx, cy = width / 2.0, height / 2.0
    scale = np.array([[factor, 0.0, cx * (1 - factor)],
                      [0.0, factor, cy * (1 - factor)],
                      [0.0, 0.0, 1.0]])
    existing = warp.get("pre_transform")
    if existing is not None:
        scale = np.asarray(existing, dtype=np.float64).reshape(3, 3) @ scale
    zoomed = dict(warp)
    zoomed["pre_transform"] = scale.tolist()
    return zoomed


def auto_zoom(warp, width, height, margin=1.0, steps=60):
    """Largest zoom whose output still shows the whole source frame.

    Searches rather than solves: the map is non-linear, so the scale at which
    the source corners come into view has no closed form. Returns 1.0 if the
    full frame is already visible.
    """
    def visible(zoom):
        map_x, map_y = build_maps(zoom_warp(warp, width, height, zoom),
                                  width, height, mask_folds=False)
        # Does the output reach every edge of the source?
        return (map_x.min() <= 0 and map_y.min() <= 0
                and map_x.max() >= width - 1 and map_y.max() >= height - 1)

    if visible(1.0):
        return 1.0
    low, high = 0.2, 1.0
    for _ in range(steps):
        middle = (low + high) / 2
        if visible(middle):
            low = middle
        else:
            high = middle
        if high - low < 0.002:
            break
    return round(low / margin, 3)


def warp_orientation(warp, width, height):
    """Sign of the warp's Jacobian determinant at the image centre.

    Negative means the fit mirrors the image -- which a lens never does. It is
    the signature of a scrambled point correspondence (wrong grid type, or rows
    read in the opposite order), and it is worth catching because such a fit can
    still report a small residual while producing a visibly wrong picture.
    """
    cx, cy = width / 2.0, height / 2.0
    h = 1.0
    xs = np.array([cx - h, cx + h, cx, cx])
    ys = np.array([cy, cy, cy - h, cy + h])
    sx, sy = apply_warp(warp, xs, ys)
    dxdu, dydu = (sx[1] - sx[0]) / (2 * h), (sy[1] - sy[0]) / (2 * h)
    dxdv, dydv = (sx[3] - sx[2]) / (2 * h), (sy[3] - sy[2]) / (2 * h)
    return float(dxdu * dydv - dxdv * dydu)


def build_maps(warp, width, height, mask_folds=True):
    """Dense (map_x, map_y) for cv2.remap.

    Every model used here can fold far outside the fitted region: the radial
    one needs a negative k1 to describe barrel distortion, and such a series
    eventually turns back on itself. Folded pixels are the swirling garbage in
    a rectified frame -- the mapping there is not merely inaccurate, it is
    multi-valued.

    `mask_folds` sends those pixels to an out-of-range coordinate so remap
    fills them with the border value (black) instead of inventing texture. The
    test is the sign of the local Jacobian: it matches the sign at the image
    centre wherever the mapping is well behaved, and flips across a fold.
    """
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    sx, sy = apply_warp(warp, xs.ravel(), ys.ravel())
    map_x = sx.reshape(height, width)
    map_y = sy.reshape(height, width)

    if mask_folds:
        # Finite differences of the map itself -- no model-specific maths.
        dxdu = np.gradient(map_x, axis=1)
        dydu = np.gradient(map_y, axis=1)
        dxdv = np.gradient(map_x, axis=0)
        dydv = np.gradient(map_y, axis=0)
        jacobian = dxdu * dydv - dxdv * dydu
        reference = np.sign(jacobian[height // 2, width // 2])
        if reference != 0:
            folded = np.sign(jacobian) != reference
            map_x = np.where(folded, -1.0, map_x)
            map_y = np.where(folded, -1.0, map_y)
    return map_x.astype(np.float32), map_y.astype(np.float32)


# ---------------------------------------------------------------------------
# OpenCV fisheye (Kannala-Brandt) backend.
#
# The `warp` models above fit our OWN least-squares over per-press homographies.
# This backend instead calls stock cv2.fisheye.calibrate -- the same maths, but
# the standard, widely-tested OpenCV implementation, returning intrinsics K and
# distortion D that plug straight into cv2.fisheye.initUndistortRectifyMap.
#
# It works here despite the "fixed camera on one plane" geometry that the docs
# warn is degenerate, because uneven hand presses tilt each board slightly and
# that small angular spread is enough to constrain the fit. Two gotchas it hides
# from callers: cv2.fisheye.calibrate wants every view to hold the SAME number
# of points (so we intersect to the grid cells seen in ALL views), each shaped
# (1, N, 3)/(1, N, 2) or it raises a -209 size error.
# ---------------------------------------------------------------------------
def fit_fisheye_opencv(observations_ij, width, height, spacing=1.0,
                       min_points=10, reject_sigma=2.5, labels=None):
    """Stock cv2.fisheye.calibrate over ball presses, with outlier rejection.

    observations_ij: list of (ij, detected) where ij[k] = (row, col) integer
    lattice index of detected[k] = (x, y) pixel. spacing sets the object units
    (only scales the returned focal; the distortion and the undistort map are
    scale-independent). `labels` optionally names each observation (for the
    dropped-press report). Returns (K, D, rms, n_views, n_common, dropped).
    """
    import cv2

    entries = [(idx, {(int(r), int(c)): (float(x), float(y))
                      for (r, c), (x, y) in zip(ij, det)})
               for idx, (ij, det) in enumerate(observations_ij)
               if len(ij) >= min_points]
    if len(entries) < 4:
        raise ValueError(f"need >=4 usable presses, got {len(entries)}")
    # cv2.fisheye.calibrate needs the SAME point count in every view, so a view
    # subset is scored on the grid cells seen in ALL of them. A sparse press
    # shrinks that common set toward collinearity, which trips OpenCV's extrinsic
    # init (fabs(norm_u1) > 0); densest-first keeps the common set healthy.
    entries.sort(key=lambda e: len(e[1]), reverse=True)
    order = [e[0] for e in entries]     # original obs index, densest-first
    views = [e[1] for e in entries]

    def label_of(pos):
        return labels[order[pos]] if labels else order[pos]

    def calibrate(sel):
        """Fit on positions `sel`; return a result dict with per-view RMS."""
        common = sorted(set.intersection(*[set(views[p]) for p in sel]))
        if len(common) < 10:
            return None
        n = len(common)
        objp = np.array([[c * spacing, r * spacing, 0.0] for (r, c) in common],
                        np.float64).reshape(1, n, 3)
        obj = [objp.copy() for _ in sel]
        img = [np.array([views[p][c] for c in common], np.float64).reshape(1, n, 2)
               for p in sel]
        K = np.eye(3)
        D = np.zeros((4, 1))
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                obj, img, (width, height), K, D,
                flags=cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                          300, 1e-9))
        except cv2.error:
            return None
        mx, my = build_maps_opencv(K, D, width, height, balance=0.0)
        vf = float(((mx >= 0) & (mx < width) & (my >= 0) & (my < height)).mean())
        per = []
        for k in range(len(sel)):
            proj, _ = cv2.fisheye.projectPoints(obj[k], rvecs[k], tvecs[k], K, D)
            d = proj.reshape(-1, 2) - img[k].reshape(-1, 2)
            per.append(float(np.sqrt(np.mean((d ** 2).sum(axis=1)))))
        return {"rms": float(rms), "K": K, "D": D, "common": common,
                "vf": vf, "per": per, "sel": list(sel)}

    # Phase 1 -- find the densest-first prefix whose undistort map stays IN the
    # frame (a diverged/folding fit sends most pixels outside; valid_fraction is
    # low) with the lowest reprojection RMS. This geometry is ill-conditioned, so
    # no single scalar suffices -- the in-frame gate plus RMS together do.
    best = None
    for keep in range(4, len(views) + 1):
        common = sorted(set.intersection(*[set(views[p]) for p in range(keep)]))
        if len(common) < 10:
            break                       # monotonic: larger keep only shrinks it
        r = calibrate(range(keep))
        if r is None or r["vf"] < 0.6:
            continue
        if best is None or r["rms"] < best["rms"]:
            best = r
    if best is None:
        raise ValueError(
            "cv2.fisheye.calibrate could not fit a usable model: no view subset "
            "produced an in-frame undistortion. The presses cover too little of "
            "the frame / are too fronto-parallel. Recapture ball presses that "
            "reach all four CORNERS and vary the board tilt, then refit.")

    # Phase 2 -- iteratively drop the single highest-error press while it is a
    # robust outlier (median + reject_sigma * MAD, with a 1 px floor so a set
    # that is already uniformly good is left alone). A drop is only KEPT if the
    # refit stays in-frame and does not inflate the RMS: with so few views a bad
    # drop can destabilise the fit, and we must never return a worse map than we
    # already have.
    sel = list(best["sel"])
    dropped = []
    while len(best["sel"]) > 6:
        per = np.array(best["per"])
        med = np.median(per)
        mad = np.median(np.abs(per - med))
        threshold = max(med + reject_sigma * 1.4826 * mad, 1.0)
        worst = int(per.argmax())
        if per[worst] <= threshold:
            break                       # no outlier left
        trial = sel[:worst] + sel[worst + 1:]
        r = calibrate(trial)
        if r is None or r["vf"] < 0.6 or r["rms"] > best["rms"] + 1.0:
            break                       # dropping it did not help -- keep best
        dropped.append((label_of(sel[worst]), round(per[worst], 2)))
        sel, best = trial, r
    return best["K"], best["D"], best["rms"], len(best["sel"]), \
        len(best["common"]), dropped


def build_maps_opencv(K, D, width, height, balance=0.0):
    """Dense (map_x, map_y) for cv2.remap from OpenCV fisheye K, D.

    `balance` in [0, 1] trades off field of view against black borders: 0 keeps
    only the fully-covered rectangle (no black), 1 keeps the whole fisheye FOV
    (black corners). cv2.fisheye handles the fold-over internally, so no manual
    Jacobian masking is needed.
    """
    import cv2

    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64).reshape(4, 1)
    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (width, height), np.eye(3), balance=float(balance))
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_k, (width, height), cv2.CV_32FC1)
    return map_x, map_y


def save_opencv_map(path, K, D, width, height, rms, balance, source=""):
    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64).ravel()
    payload = {
        "backend": "opencv_fisheye",
        "model": "fisheye_kannala_brandt",
        "source_width": int(width), "source_height": int(height),
        "balance": float(balance),
        "rms_residual_px": round(float(rms), 3),
        "source_image": source,
        "K": K.tolist(),
        "D": D.tolist(),
        "note": ("stock cv2.fisheye.calibrate intrinsics; apply with "
                 "cv2.fisheye.initUndistortRectifyMap (see build_maps_opencv)"),
    }
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return payload


def load_opencv_map(path):
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if data.get("backend") != "opencv_fisheye":
        raise ValueError(f"{path} is not an opencv_fisheye map")
    return data


# -------------------------------------------------------- robust detection --
# cv2.findCirclesGrid assumes a near-planar projective grid, so it cannot match
# the topology of a 120/180-degree fisheye impression even when every ball is
# found -- and it needs ALL rows*cols present. The functions below detect ball
# centres by flat-field + threshold + contours (which catches faint impressions
# a SimpleBlobDetector misses) and assign them to lattice indices by clustering,
# which is invariant to fisheye curvature and tolerates missing balls.

def detect_ball_centres(grey, roi=None, min_area=30, max_area=8000,
                        min_circularity=0.55, ffield_sigma=25):
    """Candidate ball centres (Nx2, FULL-image coords), unordered.

    A single ball array is often pressed slightly non-parallel, so the near
    rows image far darker than the far ones. Dividing by a heavily blurred copy
    (flat-fielding) removes that broad shading and the bright surround gradient
    together, so ONE threshold catches faint and strong impressions alike --
    the failure mode of the plain SimpleBlobDetector on this sensor. Both
    polarities are tried (an impression reads dark or bright with the lighting);
    the one yielding more candidates wins.
    """
    import cv2

    ox, oy = 0, 0
    if roi:
        x0, y0, x1, y1 = roi
        grey = grey[y0:y1, x0:x1]
        ox, oy = x0, y0
    grey = grey.astype(np.float32)
    background = cv2.GaussianBlur(grey, (0, 0), ffield_sigma)
    flat = np.clip(grey / (background + 1e-3) * 128.0, 0, 255).astype(np.uint8)

    best = np.empty((0, 2))
    for invert in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        mask = cv2.adaptiveThreshold(flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     invert, 41, 10)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        pts = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            if 4 * np.pi * area / (perimeter * perimeter) < min_circularity:
                continue
            moments = cv2.moments(contour)
            pts.append((moments["m10"] / moments["m00"],
                        moments["m01"] / moments["m00"]))
        if len(pts) > len(best):
            best = np.asarray(pts, dtype=np.float64)
    if len(best):
        best = best + (ox, oy)
    return best


def _grid_tilt(points):
    """Median rotation of the lattice from axis-aligned, folded into +-45 deg.

    Estimated from each point's vector to its nearest neighbour -- those run
    along grid rows/columns, so their common angle IS the press tilt. Folding
    by 90 degrees removes the row-vs-column ambiguity.
    """
    pts = np.asarray(points, dtype=np.float64)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
    np.fill_diagonal(dist, np.inf)
    nearest = dist.argmin(axis=1)
    vectors = pts[nearest] - pts
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angles = (angles + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return float(np.median(angles))


def _kmeans_1d(values, k, iters=100):
    """1-D k-means; returns labels remapped so 0 is the smallest-centre group.

    Centres are seeded on an EVENLY SPACED ladder (min..max), not on quantiles:
    a regular grid's rows are equally spaced, but quantile seeding chases point
    density -- when the near rows image darker and yield more detections, it
    packs seeds there and starves a sparse row to an empty cluster (observed:
    quantile seeding left the middle row of a real press with zero points).
    """
    v = np.asarray(values, dtype=np.float64)
    centres = np.linspace(v.min(), v.max(), k)
    labels = np.zeros(len(v), dtype=int)
    for _ in range(iters):
        labels = np.abs(v[:, None] - centres[None, :]).argmin(axis=1)
        moved = np.array([v[labels == j].mean() if (labels == j).any()
                          else centres[j] for j in range(k)])
        if np.allclose(moved, centres):
            break
        centres = moved
    order = np.argsort(centres)
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return remap[labels], centres[order]


def _walk_grid(pts, rows, cols):
    """Propagate integer (row, col) labels across the points by local steps.

    A global x/y binning fails under a 180-degree lens: the outer rows are
    compressed so far that they overlap the inner ones in absolute coordinate.
    Growing the labels from a seed using each point's OWN neighbour spacing is
    invariant to that -- it only ever asks "which point is one step to my
    right / down from here", which stays well-defined however the lens warps
    the global picture. Returns {point index: (i, j)} in un-normalised indices.
    """
    n = len(pts)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
    np.fill_diagonal(dist, np.inf)
    local_step = dist.min(axis=1)                 # this point's own grid pitch

    angle = _grid_tilt(pts)
    right = np.array([np.cos(angle), np.sin(angle)])
    down = np.array([-np.sin(angle), np.cos(angle)])
    directions = [(0, 1, right), (0, -1, -right), (1, 0, down), (-1, 0, -down)]

    # Seed from the point with the MOST grid-spaced neighbours, not a corner:
    # a false positive on the bright surround ring is usually an extreme corner,
    # and seeding the walk there starts it on an island it can never leave. An
    # interior grid point has neighbours on all four sides and always
    # propagates.
    step = float(np.median(local_step))
    neighbourly = ((dist > 0.55 * step) & (dist < 1.6 * step)).sum(axis=1)
    seed = int(neighbourly.argmax())
    coord = {seed: (0, 0)}
    frontier = [seed]
    while frontier:
        p = frontier.pop()
        i, j = coord[p]
        vectors = pts - pts[p]
        step = local_step[p]
        for di, dj, axis in directions:
            perp = np.array([-axis[1], axis[0]])
            along = vectors @ axis
            across = vectors @ perp
            # A neighbour one cell away: ~step along the axis, little sideways.
            ok = ((along > 0.45 * step) & (along < 1.7 * step)
                  & (np.abs(across) < 0.5 * step))
            if not ok.any():
                continue
            error = np.hypot(along - step, across)
            error[~ok] = np.inf
            q = int(error.argmin())
            if q not in coord:
                coord[q] = (i + di, j + dj)
                frontier.append(q)
    return coord


def assign_lattice(points, rows, cols):
    """Assign scattered grid points to (row, col) indices. Nx2 int ij, Nx2 xy.

    Labels grow from a corner seed by local neighbour steps (see `_walk_grid`),
    which reads local structure rather than global position or projective
    topology -- so it survives the fisheye curvature that defeats
    findCirclesGrid AND the extreme outer-row compression of a 180-degree lens,
    and simply fills fewer cells when balls are missing. The walked indices are
    shifted so they start at (0, 0) and flipped to a canonical top-to-bottom,
    left-to-right order. Returns None if too few points connect into a grid.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < max(rows, cols):
        return None
    coord = _walk_grid(pts, rows, cols)
    if len(coord) < max(rows, cols):
        return None
    idx = np.array(list(coord.keys()))
    ij = np.array([coord[k] for k in idx], dtype=int)
    # A false positive on the surround ring can extend the walk one cell past
    # the true grid. Keep the densest rows x cols window of labels rather than
    # rejecting the whole press, so those strays are dropped, not fatal.
    ij, keep = _densest_window(ij, rows, cols)
    idx = idx[keep]
    if len(idx) < max(rows, cols):
        return None
    xy = pts[idx]

    # Canonicalise orientation: seed corner is arbitrary under the grid's
    # symmetry, so force rows top->bottom (small i = small y) and cols
    # left->right (small j = small x).
    angle = _grid_tilt(pts)
    right = np.array([np.cos(angle), np.sin(angle)])
    down = np.array([-np.sin(angle), np.cos(angle)])
    if _axis_sign(ij[:, 0], xy @ down) < 0:
        ij[:, 0] = ij[:, 0].max() - ij[:, 0]
    if _axis_sign(ij[:, 1], xy @ right) < 0:
        ij[:, 1] = ij[:, 1].max() - ij[:, 1]

    # Model refine: the walk labels the interior perfectly but can be off by a
    # cell at the extreme corners of a near-180-degree press, where the row
    # curves so hard the "right" and "down-right" neighbours blur. Fitting a
    # smooth (row, col) -> (x, y) polynomial on the walked labels (dominated by
    # the correct majority) and re-snapping every point to its nearest
    # predicted cell centre repairs those corners. `xy` (the walked points, in
    # ij order), not `pts`, is what pairs with these labels.
    return _refine_by_model(xy, ij, rows, cols)


def _densest_window(ij, rows, cols):
    """Shift labels to (0,0) and keep the rows x cols window holding the most.

    Returns (windowed_ij, keep_mask). A stray point one step outside the true
    grid extends the label range; the window that captures the most points is
    the real grid, and the strays fall outside it.
    """
    shifted = ij - ij.min(axis=0)

    def best_offset(values, size):
        span = values.max() + 1
        if span <= size:
            return 0
        counts = [(np.sum((values >= o) & (values < o + size)), o)
                  for o in range(span - size + 1)]
        return max(counts)[1]

    r0 = best_offset(shifted[:, 0], rows)
    c0 = best_offset(shifted[:, 1], cols)
    keep = ((shifted[:, 0] >= r0) & (shifted[:, 0] < r0 + rows)
            & (shifted[:, 1] >= c0) & (shifted[:, 1] < c0 + cols))
    return shifted[keep] - (r0, c0), keep


def _lattice_features(ij, degree, rows=1, cols=1):
    """Polynomial features of (row, col) indices, normalised to ~[0, 1].

    A degree-3 term in raw indices is j^3 up to ~500, which makes the normal
    equations badly conditioned and the fitted surface oscillate; scaling the
    indices by their span keeps every term O(1).
    """
    i = ij[:, 0].astype(np.float64) / max(rows - 1, 1)
    j = ij[:, 1].astype(np.float64) / max(cols - 1, 1)
    return np.stack([(i ** a) * (j ** b)
                     for a in range(degree + 1)
                     for b in range(degree + 1 - a)], axis=-1)


def _refine_by_model(pts, ij, rows, cols):
    """Fit (row, col) -> (x, y), predict all cell centres, snap points to them."""
    degree = 3 if len(pts) >= 20 else 2
    if len(pts) < _lattice_features(np.zeros((1, 2), int), degree).shape[1]:
        order = np.lexsort((ij[:, 1], ij[:, 0]))     # too few to model
        return ij[order], pts[order]
    features = _lattice_features(ij, degree, rows, cols)
    coef_x, *_ = np.linalg.lstsq(features, pts[:, 0], rcond=None)
    coef_y, *_ = np.linalg.lstsq(features, pts[:, 1], rcond=None)
    cells = np.array([(i, j) for i in range(rows) for j in range(cols)])
    cell_feat = _lattice_features(cells, degree, rows, cols)
    centres = np.stack([cell_feat @ coef_x, cell_feat @ coef_y], axis=1)

    dist = np.hypot(pts[:, None, 0] - centres[None, :, 0],
                    pts[:, None, 1] - centres[None, :, 1])
    nearest = dist.argmin(axis=1)
    best = {}
    for n, cell in enumerate(nearest):
        if cell not in best or dist[n, cell] < dist[best[cell], cell]:
            best[cell] = n
    order = sorted(best.items(), key=lambda kv: kv[0])
    out_ij = np.array([cells[c] for c, _ in order])
    out_xy = np.array([pts[n] for _, n in order])
    return out_ij, out_xy


def _axis_sign(index, coordinate):
    """Sign of the correlation between an index axis and a screen coordinate."""
    index = index - index.mean()
    coordinate = coordinate - coordinate.mean()
    return float(np.sign(np.dot(index, coordinate)))


def detect_grid_robust(image, rows, cols, roi=None, min_area=30, max_area=8000,
                       min_fraction=0.5, debug_path=None):
    """(ideal_ij, detected_xy) for a circle grid, or None. Missing-ball tolerant.

    `min_fraction` of rows*cols must be found for the result to be trusted;
    below that the assignment is guessing. `debug_path` writes the candidate
    overlay so a failure shows whether nothing was found or the grid did not
    resolve.
    """
    import cv2

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    centres = detect_ball_centres(grey, roi, min_area, max_area)
    result = assign_lattice(centres, rows, cols) if len(centres) else None
    if debug_path:
        matched = None if result is None else result[1]
        _write_debug(debug_path, grey,
                     [type("K", (), {"pt": tuple(p), "size": 10})()
                      for p in centres], matched)
    if result is None:
        return None
    ij, xy = result
    if len(xy) < min_fraction * rows * cols:
        return None
    return ij, xy


def blob_detector(min_area=30, max_area=8000, min_circularity=0.55,
                  dark=True):
    """SimpleBlobDetector tuned for ball impressions in a gel.

    OpenCV's defaults reject these: the area window is far too narrow for
    impressions that change size with press depth, and the circularity floor is
    too strict for a dent whose rim the gel rounds off.
    """
    import cv2

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0 if dark else 255
    params.filterByArea = True
    params.minArea = float(min_area)
    params.maxArea = float(max_area)
    params.filterByCircularity = True
    params.minCircularity = float(min_circularity)
    params.filterByConvexity = True
    params.minConvexity = 0.6
    params.filterByInertia = True
    params.minInertiaRatio = 0.05
    params.minDistBetweenBlobs = 8.0
    return cv2.SimpleBlobDetector_create(params)


def detect_grid(image, rows, cols, pattern="circles", roi=None,
                debug_path=None, min_area=30, max_area=8000,
                asymmetric=False):
    """Locate the pressed pattern. Returns Nx2 points row-major, or None.

    `roi` (x0, y0, x1, y1) restricts the search, which matters on this sensor:
    the bright illumination ring around the gel produces strong edges that
    dominate blob detection. Returned coordinates are always in FULL-image
    space, so a ROI never has to be undone by the caller.

    `debug_path` writes an overlay of every blob considered. Detection failures
    are otherwise opaque -- this shows whether nothing was found, or plenty was
    found but it did not form the requested grid.
    """
    import cv2

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ox, oy = 0, 0
    if roi:
        x0, y0, x1, y1 = roi
        grey = grey[y0:y1, x0:x1]
        ox, oy = x0, y0
    equalised = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(grey)
    size = (cols, rows)

    if pattern == "chessboard":
        ok, corners = cv2.findChessboardCorners(
            equalised, size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            cv2.cornerSubPix(
                equalised, corners, (5, 5), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            if debug_path:
                _write_debug(debug_path, grey, [], corners.reshape(-1, 2))
            return corners.reshape(-1, 2).astype(np.float64) + (ox, oy)
        return None

    # ONLY the requested grid type is tried. Falling back to the other one used
    # to look helpful, but findCirclesGrid orders asymmetric results in a
    # staggered sequence that does not correspond to a rectangular lattice --
    # matching them up scrambles the correspondence and fits a warp that twists
    # the picture while still reporting a plausible residual.
    flags = (cv2.CALIB_CB_ASYMMETRIC_GRID if asymmetric
             else cv2.CALIB_CB_SYMMETRIC_GRID)
    # Polarity depends on the illumination: an impression can read dark or
    # bright depending on how the gel is lit.
    best_blobs = []
    for source in (equalised, grey):
        for dark in (True, False):
            detector = blob_detector(min_area, max_area, dark=dark)
            keypoints = detector.detect(source)
            if len(keypoints) > len(best_blobs):
                best_blobs = keypoints
            ok, centres = cv2.findCirclesGrid(
                source, size, flags=flags, blobDetector=detector)
            if ok:
                if debug_path:
                    _write_debug(debug_path, grey, keypoints,
                                 centres.reshape(-1, 2))
                return centres.reshape(-1, 2).astype(np.float64) + (ox, oy)
    if debug_path:
        _write_debug(debug_path, grey, best_blobs, None)
    return None


def _write_debug(path, grey, keypoints, matched):
    import cv2

    vis = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    for kp in keypoints:
        cv2.circle(vis, (int(kp.pt[0]), int(kp.pt[1])),
                   max(3, int(kp.size / 2)), (0, 0, 255), 2)
    if matched is not None:
        for x, y in matched:
            cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)
    cv2.putText(vis, f"blobs={len(keypoints)}"
                + ("" if matched is None else f"  grid={len(matched)}"),
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if matched is not None else (0, 0, 255), 2)
    cv2.imwrite(path, vis)


def grid_correspondences(image, rows, cols, pattern, width, height, fill,
                         asymmetric, roi, min_area, max_area, robust=True,
                         debug_path=None):
    """Fit-ready (ideal_dst, detected_src) point pairs, or None.

    Unifies both detectors behind one contract: whatever is found, the ideal
    lattice point returned for each detection is the SAME index it was assigned,
    so the correspondence is never scrambled. The robust circle detector may
    return fewer than rows*cols pairs (missing balls dropped on both sides); the
    findCirclesGrid and chessboard paths always return the full grid.
    """
    ideal_full = ideal_lattice(rows, cols, width, height, fill,
                               asymmetric=asymmetric)
    if pattern == "circles" and robust and not asymmetric:
        result = detect_grid_robust(image, rows, cols, roi=roi,
                                    min_area=min_area, max_area=max_area,
                                    debug_path=debug_path)
        if result is None:
            return None
        ij, detected = result
        ideal = ideal_full[ij[:, 0] * cols + ij[:, 1]]
        return ideal, detected
    detected = detect_grid(image, rows, cols, pattern, roi=roi,
                           debug_path=debug_path, min_area=min_area,
                           max_area=max_area, asymmetric=asymmetric)
    if detected is None:
        return None
    return ideal_full, canonicalize_grid(detected, rows, cols)


def valid_region(ideal_points, width, height, margin=0.5):
    """Output rectangle the fit is actually supported over.

    A polynomial is only meaningful inside the span of the points that
    constrained it. Outside, degree 5 diverges violently -- the smeared,
    swirling edges of a rectified frame are extrapolation, not lens correction.
    `margin` extends the box by that fraction of the grid pitch, so the outer
    ring of balls is not sliced off.
    """
    pts = np.asarray(ideal_points, dtype=np.float64)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    pad_x = margin * (x1 - x0) / max(1, len(np.unique(pts[:, 0].round(3))) - 1)
    pad_y = margin * (y1 - y0) / max(1, len(np.unique(pts[:, 1].round(3))) - 1)
    return (int(max(0, np.floor(x0 - pad_x))),
            int(max(0, np.floor(y0 - pad_y))),
            int(min(width, np.ceil(x1 + pad_x))),
            int(min(height, np.ceil(y1 + pad_y))))


def save_map(path, warp, rows, cols, width, height, residual, source="",
             valid=None):
    payload = {
        "warp": warp,
        "rows": rows, "cols": cols,
        "source_width": int(width), "source_height": int(height),
        "rms_residual_px": round(residual, 3),
        "source_image": source,
        "note": ("polynomial image->gel-plane rectification; visual "
                 "consistency only, not a metric calibration"),
    }
    if valid:
        payload["valid_region"] = {"x0": int(valid[0]), "y0": int(valid[1]),
                                   "x1": int(valid[2]), "y1": int(valid[3])}
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return payload


def load_map(path):
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if "warp" not in data:
        raise ValueError(f"{path} has no 'warp' section")
    return data


def _roi_arg(value):
    """--roi x0,y0,x1,y1 -> tuple of ints."""
    try:
        x0, y0, x1, y1 = (int(v) for v in str(value).split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("--roi must be x0,y0,x1,y1")
    if x1 <= x0 or y1 <= y0:
        raise argparse.ArgumentTypeError("--roi needs x1>x0 and y1>y0")
    return (x0, y0, x1, y1)


def cmd_fit(args):
    import cv2

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    height, width = image.shape[:2]

    pairs = grid_correspondences(
        image, args.rows, args.cols, args.pattern, width, height, args.fill,
        args.asymmetric, args.roi, args.min_area, args.max_area,
        robust=not args.legacy_detector, debug_path=args.debug)
    if pairs is None:
        raise SystemExit(
            f"could not find a {args.rows}x{args.cols} {args.pattern} grid.\n"
            "  - press EVENLY: a tilted array leaves the far rows too faint to "
            "detect -- all rows need real depth\n"
            "  - check --rows/--cols count the ball rows/cols (INNER corners "
            "for a chessboard, not squares)\n"
            "  - restrict --roi to the pad, excluding the bright surround ring\n"
            "  - pass --debug out.png to see which balls were found")
    ideal, detected = pairs
    warp, keep = fit_warp_robust(ideal, detected, degree=args.degree,
                                 model=args.model,
                                 width=width, height=height)
    press_tilt = warp_rotation(warp, width, height)
    if not args.keep_tilt:
        warp = align_warp(warp, width, height)
        print(f"[info] cancelled {press_tilt:+.1f} deg of press tilt")
    residual = warp_residual(warp, ideal[keep], detected[keep])
    dropped = int((~keep).sum())
    if dropped:
        print(f"[info] dropped {dropped} outlier point(s) before the final fit")
    orientation = warp_orientation(warp, width, height)
    if orientation <= 0:
        print("[warn] the fit MIRRORS the image -- correspondence is scrambled",
              file=sys.stderr)
    needed = num_coefficients(args.degree)
    if len(detected) < 2 * needed:
        print(f"[warn] {len(detected)} points for {needed} coefficients per axis "
              f"-- the fit may be following noise rather than the lens. A low "
              f"residual here does not prove accuracy; use a denser grid or "
              f"--degree {max(3, args.degree - 2)}.")

    out_path = args.out or os.path.join(os.path.dirname(args.image) or ".",
                                        DEFAULT_MAP_NAME)
    valid = valid_region(ideal, width, height)
    save_map(out_path, warp, args.rows, args.cols, width, height, residual,
             os.path.basename(args.image), valid=valid)
    print(f"detected {len(detected)} grid points")
    print(f"RMS residual : {residual:.2f} px")
    print(f"saved map    : {out_path}")

    if args.show:
        map_x, map_y = build_maps(warp, width, height)
        corrected = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        marked = image.copy()
        for x, y in detected:
            cv2.circle(marked, (int(x), int(y)), 3, (0, 220, 255), -1)
        for x, y in ideal:
            cv2.circle(corrected, (int(x), int(y)), 3, (0, 220, 255), 1)
        cv2.imshow("detected (left)  |  rectified (right)",
                   np.hstack([marked, corrected]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def _dump_debug(dump_dir, frame, baseline, view, detected, fitted, args):
    """Write everything needed to reproduce the current state offline.

    A live window shows that something is wrong but not why. This saves the raw
    frame, the baseline it was differenced against, the detection overlay and
    the current rectification, plus the numbers -- so a bad fit can be picked
    apart without the rig.
    """
    import cv2

    os.makedirs(dump_dir, exist_ok=True)
    existing = [n for n in os.listdir(dump_dir) if n.endswith("_frame.png")]
    stem = os.path.join(dump_dir, f"{len(existing) + 1:03d}")

    written = {"frame": f"{stem}_frame.png"}
    cv2.imwrite(written["frame"], frame)
    if baseline is not None:
        written["baseline"] = f"{stem}_baseline.png"
        cv2.imwrite(written["baseline"], baseline)
    written["diff"] = f"{stem}_diff.png"
    cv2.imwrite(written["diff"], view)

    overlay = view.copy()
    if detected is not None:
        # Numbered in detection order: if the numbers do not run along the rows
        # the way the physical grid does, the correspondence is the problem.
        for i, (x, y) in enumerate(detected):
            cv2.circle(overlay, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.putText(overlay, str(i), (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1,
                        cv2.LINE_AA)
    written["detection"] = f"{stem}_detection.png"
    cv2.imwrite(written["detection"], overlay)

    meta = {"rows": args.rows, "cols": args.cols, "pattern": args.pattern,
            "asymmetric": bool(args.asymmetric), "roi": args.roi,
            "degree": args.degree, "fill": args.fill,
            "min_area": args.min_area, "max_area": args.max_area,
            "points_detected": 0 if detected is None else len(detected)}
    if detected is not None:
        meta["detected_points"] = np.asarray(detected).round(2).tolist()
    if fitted is not None:
        height, width = frame.shape[:2]
        map_x, map_y = build_maps(fitted["warp"], width, height)
        rectified = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        written["rectified"] = f"{stem}_rectified.png"
        cv2.imwrite(written["rectified"], rectified)          # uncropped
        valid = fitted.get("valid")
        if valid:
            vx0, vy0, vx1, vy1 = valid
            written["rectified_valid"] = f"{stem}_rectified_valid.png"
            cv2.imwrite(written["rectified_valid"],
                        rectified[vy0:vy1, vx0:vx1])
            meta["valid_region"] = [int(v) for v in valid]
        meta["rms_residual_px"] = round(fitted["residual"], 3)
        meta["orientation"] = round(
            warp_orientation(fitted["warp"], width, height), 4)
        meta["warp"] = fitted["warp"]
    with open(f"{stem}_meta.yaml", "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    written["meta"] = f"{stem}_meta.yaml"
    return written


def cmd_calibrate(args):
    """Live calibration: press the board, watch detection, fit when it locks on.

    Fitting from a saved PNG means discovering a failed detection long after the
    board has been lifted. Here the detection overlay updates live, so the board
    can be re-seated until every corner is found, and only then fitted.

    Detection is tried on the RAW frame first and only then on the amplified
    difference. Measured on synthetic impressions the raw frame wins: the
    detector's own CLAHE already removes the illumination gradient, whereas
    differencing multiplies the baseline noise along with the signal, so it
    loses a board the raw frame still finds (12 grey levels: raw ok, diff not).
    The difference is still shown, because it is what makes a weak press
    obvious to the eye.
    """
    import cv2

    from .capture import open_live

    live = open_live(args.camera, args.config, buffer=4, fresh_reads=1)
    camera_id, reader = live.camera_id, live.reader

    window = "undistort calibrate"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    baseline = None
    detected = None
    correspondence = None
    fitted = None
    frame_no = 0
    print(json.dumps({
        "camera": camera_id, "pattern": args.pattern,
        "inner_corners": f"{args.cols}x{args.rows}",
        "controls": {"b": "set unloaded baseline",
                     "f": "fit from current detection",
                     "w": "write debug bundle (frame+diff+detection+fit)",
                     "s": "save fitted map", "q": "quit"},
    }))

    try:
        last_seq = -1
        while True:
            item = reader.wait_for_new(last_seq, timeout=1.0)
            if item is None:
                continue
            last_seq, frame, _t_mono, _t_epoch = item
            frame_no += 1
            height, width = frame.shape[:2]
            if baseline is None:
                baseline = frame.copy()

            signed = mc.signed_difference(frame, baseline)
            view = (mc.render_difference(signed, args.diff_gain)
                    if signed is not None else frame.copy())

            # Detection is expensive; every Nth frame keeps the view fluid.
            # It runs on the RAW frame -- the robust detector flat-fields
            # internally, so the raw view has cleaner ball edges than the
            # baseline-differenced one (differencing multiplies noise). The
            # diff is shown alongside because it makes a weak press obvious to
            # the eye, but it is not what the detector reads.
            if frame_no % max(1, args.detect_every) == 0:
                correspondence = grid_correspondences(
                    frame, args.rows, args.cols, args.pattern, width, height,
                    args.fill, args.asymmetric, args.roi, args.min_area,
                    args.max_area, robust=not args.legacy_detector)
                source = "raw"
                if correspondence is None and signed is not None:
                    correspondence = grid_correspondences(
                        view, args.rows, args.cols, args.pattern, width,
                        height, args.fill, args.asymmetric, args.roi,
                        args.min_area, args.max_area,
                        robust=not args.legacy_detector)
                    source = "diff"
                detected = None if correspondence is None else correspondence[1]

            total = args.rows * args.cols
            # Detection overlay goes on the ORIGINAL frame: the operator asked
            # to see the raw image, and the green dots on it show exactly what
            # the fit will use.
            raw_panel = frame.copy()
            if detected is not None:
                for x, y in detected:
                    cv2.circle(raw_panel, (int(x), int(y)), 4, (0, 255, 0), -1)
                status = f"LOCKED {len(detected)}/{total} pts ({source})"
                colour = (0, 200, 0)
            else:
                status, colour = "searching...", (0, 180, 255)
            cv2.putText(raw_panel, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, colour, 2, cv2.LINE_AA)
            cv2.putText(raw_panel, "original", (12, height - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA)

            diff_panel = view.copy()
            cv2.putText(diff_panel, f"difference (x{args.diff_gain:g})",
                        (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(diff_panel,
                        "b=baseline  f=fit  w=debug  s=save  q=quit"
                        + ("   [fitted]" if fitted else ""),
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)

            panes = [raw_panel, diff_panel]
            if fitted is not None:
                map_x, map_y = build_maps(fitted["warp"], width, height)
                rect = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
                # Full frame, with the supported region outlined. Everything
                # OUTSIDE the box is extrapolated -- the polynomial has no data
                # there and the swirling is meaningless -- but it is shown so
                # the whole warp can be judged before deciding where to crop.
                vx0, vy0, vx1, vy1 = fitted["valid"]
                cv2.rectangle(rect, (vx0, vy0), (vx1 - 1, vy1 - 1),
                              (0, 220, 255), 1)
                cv2.putText(rect, "rectified", (vx0 + 4, vy0 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1,
                            cv2.LINE_AA)
                panes.append(rect)
            cv2.imshow(window, np.hstack(panes))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):
                baseline = frame.copy()
                detected = None
                correspondence = None
                print(json.dumps({"event": "baseline_set"}))
            elif key == ord("f"):
                if correspondence is None:
                    print("no detection to fit -- reseat the board", file=sys.stderr)
                    continue
                # The correspondence already pairs each detected ball with its
                # own lattice index (robust path drops missing balls on both
                # sides), so no re-detection or re-ordering here.
                ideal, detected = correspondence
                warp, keep = fit_warp_robust(ideal, detected,
                                             degree=args.degree,
                                             model=args.model,
                                             width=width,
                                             height=height)
                press_tilt = warp_rotation(warp, width, height)
                if not args.keep_tilt:
                    warp = align_warp(warp, width, height)
                residual = warp_residual(warp, ideal[keep], detected[keep])
                orientation = warp_orientation(warp, width, height)
                dropped = int((~keep).sum())
                if dropped:
                    print(f"[info] dropped {dropped} outlier point(s) before "
                          f"the final fit", file=sys.stderr)
                fitted = {"warp": warp, "residual": residual,
                          "width": width, "height": height,
                          "ideal": ideal, "detected": detected,
                          "valid": valid_region(ideal, width, height)}
                print(json.dumps({"event": "fitted",
                                  "points": len(detected),
                                  "rms_residual_px": round(residual, 3),
                                  "orientation": round(orientation, 4),
                                  "press_tilt_deg": round(press_tilt, 1)}))
                needed = num_coefficients(args.degree)
                if len(detected) < 2 * needed:
                    print(f"[warn] {len(detected)} points for {needed} "
                          f"coefficients -- fit may be following noise",
                          file=sys.stderr)
                if orientation <= 0:
                    print("[warn] the fit MIRRORS the image (negative Jacobian). "
                          "A lens cannot do that, so the point correspondence is "
                          "scrambled -- usually the wrong grid type. Try adding "
                          "or removing --asymmetric. Do NOT trust this map.",
                          file=sys.stderr)
                if not args.no_auto_dump:
                    # Written automatically: a fit is exactly the moment worth
                    # keeping, and relying on the operator to also press 'w'
                    # loses the evidence for the fits that looked wrong.
                    dumped = _dump_debug(args.dump_dir, frame, baseline, view,
                                         detected, fitted, args)
                    print(json.dumps({"event": "dumped", "trigger": "fit",
                                      **dumped}))
            elif key == ord("w"):
                try:
                    dumped = _dump_debug(args.dump_dir, frame, baseline, view,
                                         detected, fitted, args)
                    print(json.dumps({"event": "dumped", "trigger": "key",
                                      **dumped}))
                except Exception as exc:
                    print(f"[error] debug dump failed: {exc}", file=sys.stderr)
            elif key == ord("s"):
                if fitted is None:
                    print("nothing fitted yet -- press 'f' first", file=sys.stderr)
                    continue
                out_path = args.out or DEFAULT_MAP_NAME
                save_map(out_path, fitted["warp"], args.rows, args.cols,
                         fitted["width"], fitted["height"], fitted["residual"],
                         f"live:{camera_id}", valid=fitted["valid"])
                print(json.dumps({"event": "saved", "path": out_path,
                                  "rms_residual_px": round(fitted["residual"], 3)}))
                if not args.no_auto_dump:
                    dumped = _dump_debug(args.dump_dir, frame, baseline, view,
                                         detected, fitted, args)
                    print(json.dumps({"event": "dumped", "trigger": "save",
                                      **dumped}))
    finally:
        try:
            cv2.destroyWindow(window)
        except Exception:
            pass
        live.close()
    return 0


def cmd_fit_multi(args):
    """Offline joint fit over every press found in a folder."""
    import cv2

    patterns = sorted(n for n in os.listdir(args.folder)
                      if n.lower().endswith((".png", ".jpg")))
    # Debug bundles hold several renderings of the same press; only the raw
    # frame is a fresh observation, so counting the rest would double-weight it.
    frames = [n for n in patterns if n.endswith("_frame.png")] or patterns
    if not frames:
        raise SystemExit(f"no images in {args.folder}")

    observations, used, all_points = [], [], []
    width = height = None
    for name in frames:
        image = cv2.imread(os.path.join(args.folder, name))
        if image is None:
            continue
        height, width = image.shape[:2]
        pairs = grid_correspondences(
            image, args.rows, args.cols, args.pattern, width, height,
            args.fill, args.asymmetric, args.roi, args.min_area, args.max_area,
            robust=not args.legacy_detector)
        if pairs is None:
            print(f"  {name}: no grid found -- skipped")
            continue
        ideal, detected = pairs
        observations.append((ideal, detected))
        all_points.extend(detected.tolist())
        used.append(name)
        print(f"  {name}: {len(detected)} points")

    if not observations:
        raise SystemExit("no press produced a usable detection")

    if getattr(args, "model", "radial") == "fisheye":
        warp, per_press = fit_fisheye_multi(observations, width, height)
    else:
        warp, per_press = fit_radial_multi(observations, width, height)
    press_tilt = warp_rotation(warp, width, height)
    if not args.keep_tilt:
        warp = align_warp(warp, width, height)
    covered = coverage_fraction(all_points, width, height)
    combined = float(np.sqrt(np.mean(np.square(per_press))))

    print(f"\npresses used   : {len(observations)}")
    print(f"points total   : {len(all_points)}")
    print(f"field covered  : {100 * covered:.0f}%  "
          f"(the rest is extrapolated)")
    print(f"RMS residual   : {combined:.3f} px  "
          f"(per press: {', '.join(f'{r:.2f}' for r in per_press)})")
    print(f"press tilt     : {press_tilt:+.1f} deg"
          f"{'' if args.keep_tilt else ' (cancelled)'}")

    ideal = observations[0][0]
    out_path = args.out or os.path.join(args.folder, DEFAULT_MAP_NAME)
    save_map(out_path, warp, args.rows, args.cols, width, height, combined,
             f"multi:{len(observations)} presses",
             valid=valid_region(np.asarray(all_points), width, height))
    print(f"saved map      : {out_path}")
    return 0


def cmd_fit_opencv(args):
    """Calibrate with stock cv2.fisheye.calibrate over a folder of presses."""
    import cv2

    names = sorted(n for n in os.listdir(args.folder)
                   if n.lower().endswith((".png", ".jpg")))
    frames = [n for n in names if n.endswith("_frame.png")] or names
    if not frames:
        raise SystemExit(f"no images in {args.folder}")

    observations, labels, width, height = [], [], None, None
    for name in frames:
        image = cv2.imread(os.path.join(args.folder, name))
        if image is None:
            continue
        height, width = image.shape[:2]
        res = detect_grid_robust(image, args.rows, args.cols, roi=args.roi,
                                 min_area=args.min_area, max_area=args.max_area)
        if res is None:
            print(f"  {name}: no grid found -- skipped")
            continue
        ij, det = res
        observations.append((ij, det))
        labels.append(name)
        print(f"  {name}: {len(ij)} points")
    if not observations:
        raise SystemExit("no press produced a usable detection")

    K, D, rms, nviews, ncommon, dropped = fit_fisheye_opencv(
        observations, width, height, spacing=args.spacing,
        min_points=args.min_points, reject_sigma=args.reject_sigma,
        labels=labels)
    if dropped:
        print("\ndropped (high reprojection error):")
        for name, err in dropped:
            print(f"  {name}: {err} px")
    print(f"\nviews used     : {nviews}")
    print(f"common cells   : {ncommon}  (points per view)")
    print(f"RMS residual   : {rms:.3f} px")
    print(f"K              : f={K[0, 0]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}")
    print(f"D (k1..k4)     : {np.round(D.ravel(), 4).tolist()}")
    out_path = args.out or os.path.join(args.folder, "opencv_fisheye_map.yaml")
    save_opencv_map(out_path, K, D, width, height, rms, args.balance,
                    f"opencv:{nviews} views x {ncommon} cells, "
                    f"{len(dropped)} dropped")
    print(f"saved map      : {out_path}")
    return 0


def cmd_apply(args):
    import cv2

    map_file = args.map or os.path.join(args.folder, DEFAULT_MAP_NAME)
    data = load_map(map_file)
    expect = (data.get("source_width"), data.get("source_height"))
    valid = data.get("valid_region")

    out_dir = args.out or os.path.join(args.folder, "undistorted")
    if os.path.abspath(out_dir) == os.path.abspath(args.folder):
        raise SystemExit("--out must differ from the source folder")
    os.makedirs(out_dir, exist_ok=True)

    zoom = args.zoom
    if args.auto_zoom:
        zoom = auto_zoom(data["warp"], expect[0] or 640, expect[1] or 480)
        print(f"auto zoom: {zoom} (whole source frame brought into view)")
    if zoom != 1.0:
        data = dict(data)
        data["warp"] = zoom_warp(data["warp"], expect[0] or 640,
                                 expect[1] or 480, zoom)

    maps_cache = {}
    written, skipped = 0, []
    for name in sorted(os.listdir(args.folder)):
        if not name.lower().endswith(".png"):
            continue
        image = cv2.imread(os.path.join(args.folder, name))
        if image is None:
            continue
        height, width = image.shape[:2]
        if all(expect) and (width, height) != tuple(expect):
            skipped.append(name)
            continue
        if (width, height) not in maps_cache:
            maps_cache[(width, height)] = build_maps(data["warp"], width, height)
        map_x, map_y = maps_cache[(width, height)]
        rectified = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        if valid and args.crop_valid:
            rectified = rectified[valid["y0"]:valid["y1"],
                                  valid["x0"]:valid["x1"]]
        cv2.imwrite(os.path.join(out_dir, name), rectified)
        written += 1

    print(f"undistorted {written} images -> {out_dir}")
    if skipped:
        print(f"[warn] skipped {len(skipped)} image(s) not {expect[0]}x{expect[1]}: "
              f"{skipped[:3]}{'...' if len(skipped) > 3 else ''}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit a rectification from a pressed grid")
    f.add_argument("image", help="image of the grid pressed into the gel")
    f.add_argument("--rows", type=int, required=True, help="INNER points per column")
    f.add_argument("--cols", type=int, required=True, help="INNER points per row")
    f.add_argument("--pattern", choices=["circles", "chessboard"], default="circles")
    f.add_argument("--fill", type=float, default=0.8,
                   help="fraction of the frame the rectified grid spans")
    f.add_argument("--degree", type=int, default=DEGREE,
                   help=f"polynomial degree (default {DEGREE}); 7 fits a strong "
                        f"fisheye better but needs a dense grid")
    f.add_argument("--roi", type=_roi_arg, default=None,
                   help="x0,y0,x1,y1 search region -- exclude the bright ring "
                        "around the gel, which dominates blob detection")
    f.add_argument("--model", choices=["radial", "poly"],
                   default="radial",
                   help="radial (default): homography + radial distortion, "
                        "the physical model -- extrapolates sanely past the "
                        "grid. poly: free 2D polynomial, fits the grid a "
                        "little better but folds over outside it")
    f.add_argument("--keep-tilt", action="store_true",
                    help="keep the rotation the pattern was pressed at. "
                         "By default it is cancelled so the rectified "
                         "orientation does not depend on how the array "
                         "happened to be laid down")
    f.add_argument("--asymmetric", action="store_true",
                   help="the balls are STAGGERED (rows offset by half a "
                        "step), not a rectangular lattice")
    f.add_argument("--min-area", type=int, default=30,
                   help="smallest blob accepted, px^2 (default 30). A fisheye squeezes the outer balls, so they image far smaller than the centre ones")
    f.add_argument("--max-area", type=int, default=8000,
                   help="largest blob accepted, px^2 (default 8000)")
    f.add_argument("--legacy-detector", action="store_true",
                   help="use cv2.findCirclesGrid instead of the robust "
                        "contour+lattice detector. The legacy path needs ALL "
                        "rows*cols balls and cannot match a strong fisheye; the "
                        "robust one tolerates missing balls and lens curvature")
    f.add_argument("--debug", default=None,
                   help="write a blob overlay here to diagnose failures")
    f.add_argument("--out", default=None)
    f.add_argument("--show", action="store_true", help="preview the result")
    f.set_defaults(func=cmd_fit)

    c = sub.add_parser("calibrate",
                       help="live: press the board and fit while watching")
    c.add_argument("--camera", required=True,
                   help="/dev/video index, USB vendor:product id, or a "
                        "host camera key")
    c.add_argument("-c", "--config", default=None,
                   help="host device config (only with a registered opener)")
    c.add_argument("--rows", type=int, required=True,
                   help="INNER corners per column (squares - 1)")
    c.add_argument("--cols", type=int, required=True,
                   help="INNER corners per row (squares - 1)")
    c.add_argument("--pattern", choices=["circles", "chessboard"],
                   default="chessboard")
    c.add_argument("--degree", type=int, default=DEGREE)
    c.add_argument("--fill", type=float, default=0.8)
    c.add_argument("--diff-gain", type=float, default=3.0,
                   help="amplification of the difference the detector runs on")
    c.add_argument("--detect-every", type=int, default=3,
                   help="run detection every Nth frame (detection is slow)")
    c.add_argument("--roi", type=_roi_arg, default=None,
                   help="x0,y0,x1,y1 search region -- exclude the bright ring "
                        "around the gel, which dominates blob detection")
    c.add_argument("--model", choices=["radial", "poly"],
                   default="radial",
                   help="radial (default): homography + radial distortion, "
                        "the physical model -- extrapolates sanely past the "
                        "grid. poly: free 2D polynomial, fits the grid a "
                        "little better but folds over outside it")
    c.add_argument("--keep-tilt", action="store_true",
                    help="keep the rotation the pattern was pressed at. "
                         "By default it is cancelled so the rectified "
                         "orientation does not depend on how the array "
                         "happened to be laid down")
    c.add_argument("--asymmetric", action="store_true",
                   help="the balls are STAGGERED (rows offset by half a "
                        "step), not a rectangular lattice")
    c.add_argument("--min-area", type=int, default=30,
                   help="smallest blob accepted, px^2 (default 30). A fisheye squeezes the outer balls, so they image far smaller than the centre ones")
    c.add_argument("--max-area", type=int, default=8000,
                   help="largest blob accepted, px^2 (default 8000)")
    c.add_argument("--legacy-detector", action="store_true",
                   help="use cv2.findCirclesGrid instead of the robust "
                        "contour+lattice detector (needs every ball; no "
                        "fisheye tolerance)")
    c.add_argument("--no-auto-dump", action="store_true",
                   help="do not write a debug bundle on every fit/save")
    c.add_argument("--dump-dir", default="undistort_debug",
                   help="where 'w' writes debug bundles (default ./undistort_debug)")
    c.add_argument("--out", default=None,
                   help=f"where to save (default ./{DEFAULT_MAP_NAME})")
    c.set_defaults(func=cmd_calibrate)

    fm = sub.add_parser("fit-multi",
                        help="offline joint fit over several saved presses")
    fm.add_argument("folder", help="folder of frames / debug bundles")
    fm.add_argument("--rows", type=int, required=True)
    fm.add_argument("--cols", type=int, required=True)
    fm.add_argument("--pattern", choices=["circles", "chessboard"],
                    default="circles")
    fm.add_argument("--model", choices=["radial", "fisheye"], default="radial",
                    help="distortion model to fit. 'radial' (pinhole + k1..k3) "
                         "suits moderate lenses; 'fisheye' (Kannala-Brandt, "
                         "angle-based) is the correct choice for a wide/180-deg "
                         "lens -- the radial series folds over past ~120 deg.")
    fm.add_argument("--keep-tilt", action="store_true",
                    help="keep the rotation the pattern was pressed at. "
                         "By default it is cancelled so the rectified "
                         "orientation does not depend on how the array "
                         "happened to be laid down")
    fm.add_argument("--asymmetric", action="store_true")
    fm.add_argument("--roi", type=_roi_arg, default=None)
    fm.add_argument("--min-area", type=int, default=30)
    fm.add_argument("--max-area", type=int, default=8000)
    fm.add_argument("--legacy-detector", action="store_true",
                    help="use cv2.findCirclesGrid instead of the robust "
                         "contour+lattice detector")
    fm.add_argument("--fill", type=float, default=0.8)
    fm.add_argument("--out", default=None)
    fm.set_defaults(func=cmd_fit_multi)

    fo = sub.add_parser(
        "fit-opencv",
        help="calibrate with stock cv2.fisheye.calibrate over saved presses "
             "(Kannala-Brandt K, D -- the OpenCV-standard fisheye model)")
    fo.add_argument("folder", help="folder of frames / debug bundles")
    fo.add_argument("--rows", type=int, required=True)
    fo.add_argument("--cols", type=int, required=True)
    fo.add_argument("--spacing", type=float, default=1.0,
                    help="ball centre-to-centre spacing in mm (optional; only "
                         "scales the reported focal, not the correction)")
    fo.add_argument("--roi", type=_roi_arg, default=None)
    fo.add_argument("--min-area", type=int, default=30)
    fo.add_argument("--max-area", type=int, default=8000)
    fo.add_argument("--min-points", type=int, default=10,
                    help="skip presses with fewer than this many detected balls")
    fo.add_argument("--reject-sigma", type=float, default=2.5,
                    help="drop presses whose reprojection error exceeds "
                         "median + this*MAD and refit (lower = stricter; "
                         "default 2.5). Set very high to disable rejection.")
    fo.add_argument("--balance", type=float, default=0.0,
                    help="undistort FOV vs black borders, 0..1 (0 = crop to the "
                         "no-black rectangle, 1 = keep the whole fisheye FOV)")
    fo.add_argument("--out", default=None)
    fo.set_defaults(func=cmd_fit_opencv)

    a = sub.add_parser("apply", help="apply a saved rectification to a folder")
    a.add_argument("folder")
    a.add_argument("--map", default=None,
                   help=f"map file (default <folder>/{DEFAULT_MAP_NAME})")
    a.add_argument("--out", default=None,
                   help="output folder (default <folder>/undistorted)")
    a.add_argument("--zoom", type=float, default=1.0,
                   help="output scale; BELOW 1 shows more of the frame. "
                        "Undistortion magnifies the centre, so a rectified "
                        "image looks cropped even when it samples everything")
    a.add_argument("--auto-zoom", action="store_true",
                   help="pick the largest zoom that still shows the whole "
                        "source frame")
    a.add_argument("--crop-valid", action="store_true",
                   help="crop output to the calibrated region. Off by default: "
                        "the full frame is kept so the whole warp stays "
                        "visible, but note the area outside that region is "
                        "extrapolated and not trustworthy")
    a.set_defaults(func=cmd_apply)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
