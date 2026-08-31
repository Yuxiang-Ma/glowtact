"""Pressed-grid fisheye rectification: fitting, detection, orientation.

Moved from ProbingPi's tests/test_manual_collect.py when undistort became part
of the glowtact library.
"""
import numpy as np
import pytest

from glowtact import undistort


def _fisheye_points(ideal, focal, width=640, height=480):
    """Equidistant-fisheye image positions for ideal (rectified) points."""
    centre = np.array([width / 2, height / 2])
    delta = ideal - centre
    radius = np.hypot(delta[:, 0], delta[:, 1])
    warped = focal * np.arctan(radius / focal)
    scale = np.where(radius > 0, warped / np.maximum(radius, 1e-9), 1.0)
    return centre + delta * scale[:, None]


def test_undistort_warp_recovers_a_fisheye():
    """Fitted warp must map rectified points back onto their distorted ones."""

    ideal = undistort.ideal_lattice(9, 9, 640, 480, fill=0.85)
    distorted = _fisheye_points(ideal, focal=110.0)

    uncorrected = float(np.hypot(*(distorted - ideal).T).max())
    assert uncorrected > 100          # the lens really does bend things

    warp = undistort.fit_warp(ideal, distorted, degree=5)
    residual = undistort.warp_residual(warp, ideal, distorted)
    assert residual < 3.0             # ~2 orders of magnitude better


def test_undistort_higher_degree_fits_a_fisheye_better():
    """A fisheye's radial term is odd, so degree 5 beats 3; 4 does not beat 3."""

    ideal = undistort.ideal_lattice(11, 11, 640, 480, fill=0.85)
    distorted = _fisheye_points(ideal, focal=110.0)

    r3, r4, r5 = (undistort.warp_residual(
        undistort.fit_warp(ideal, distorted, degree=d), ideal, distorted)
        for d in (3, 4, 5))
    assert r5 < r3 / 2
    assert abs(r4 - r3) < 0.05        # even terms buy nothing for a radial lens


def test_undistort_refuses_an_underdetermined_fit():

    ideal = undistort.ideal_lattice(3, 3, 640, 480)      # 9 points
    with pytest.raises(ValueError, match="needs >="):
        undistort.fit_warp(ideal, ideal, degree=5)       # needs 21


def test_undistort_map_round_trips_through_a_file(tmp_path):

    ideal = undistort.ideal_lattice(9, 9, 640, 480, fill=0.85)
    warp = undistort.fit_warp(ideal, _fisheye_points(ideal, 110.0), degree=5)
    path = tmp_path / "undistort_map.yaml"
    undistort.save_map(str(path), warp, 9, 9, 640, 480, 1.23, "shot.png")

    loaded = undistort.load_map(str(path))
    assert loaded["rms_residual_px"] == 1.23
    assert loaded["source_width"] == 640
    xs = np.array([100.0, 320.0])
    ys = np.array([100.0, 240.0])
    a = undistort.apply_warp(warp, xs, ys)
    b = undistort.apply_warp(loaded["warp"], xs, ys)
    assert np.allclose(a, b)


def test_warp_orientation_catches_a_mirrored_correspondence():
    """A flipped point order fits with an IDENTICAL residual, so the residual
    cannot detect it -- only the Jacobian sign can."""

    ideal = undistort.ideal_lattice(6, 9, 640, 480, 0.80)
    detected = _fisheye_points(ideal, focal=110.0)

    good = undistort.fit_warp(ideal, detected, degree=5)
    flipped_cols = detected.reshape(6, 9, 2)[:, ::-1].reshape(-1, 2)
    bad = undistort.fit_warp(ideal, flipped_cols, degree=5)

    # Same quality of fit -- the residual is blind to this failure.
    assert abs(undistort.warp_residual(good, ideal, detected)
               - undistort.warp_residual(bad, ideal, flipped_cols)) < 0.01
    # But the orientation flips sign, which a real lens can never do.
    assert undistort.warp_orientation(good, 640, 480) > 0
    assert undistort.warp_orientation(bad, 640, 480) < 0


def test_asymmetric_lattice_is_staggered_not_rectangular():
    """Asymmetric point order must match findCirclesGrid's, or the fit is junk."""

    rect = undistort.ideal_lattice(4, 5, 640, 480, 0.8, asymmetric=False)
    stag = undistort.ideal_lattice(4, 5, 640, 480, 0.8, asymmetric=True)
    assert len(rect) == len(stag) == 20

    # Rectangular: every row shares one y, and rows share the same x values.
    assert len(set(np.round(rect[:5, 1], 3))) == 1
    assert np.allclose(rect[:5, 0], rect[5:10, 0])
    # Staggered: consecutive rows are offset by half the column step.
    step = stag[1, 0] - stag[0, 0]
    assert np.allclose(stag[5:10, 0] - stag[:5, 0], step / 2)


def test_radial_model_extrapolates_where_the_polynomial_folds():
    """The physical model stays ordered outside the calibrated grid.

    A free 2D polynomial fits the grid slightly better but has no constraint
    beyond it: on real data it mapped the top-left image corner to source
    x=975 and the bottom-left to x=-470 -- the mapping folds over. Radial
    distortion is monotonic in radius, so corners stay on their own side.
    """

    width, height = 640, 480
    ideal = undistort.ideal_lattice(6, 9, width, height, 0.80)
    observed = _fisheye_points(ideal, focal=110.0)

    radial = undistort.fit_radial(ideal, observed, width, height)
    poly = undistort.fit_warp(ideal, observed, degree=5)

    # Both must fit the grid itself well.
    assert undistort.warp_residual(radial, ideal, observed) < 5.0
    assert undistort.warp_residual(poly, ideal, observed) < 5.0

    # Both models can fold far outside the fitted grid, so what matters is
    # that build_maps DETECTS the fold and masks it rather than sampling
    # through it. Masked pixels are sent out of range and render as border.
    for warp in (radial, poly):
        map_x, map_y = undistort.build_maps(warp, width, height,
                                            mask_folds=True)
        raw_x, raw_y = undistort.build_maps(warp, width, height,
                                            mask_folds=False)
        masked = (map_x < 0) & (map_y < 0)
        # The calibrated centre must survive masking.
        assert not masked[height // 2, width // 2]
        # Masking only ever removes pixels, never invents new mapping.
        assert np.all(map_x[~masked] == raw_x[~masked])


def test_apply_warp_dispatches_on_model():

    width, height = 640, 480
    ideal = undistort.ideal_lattice(6, 9, width, height, 0.80)
    observed = _fisheye_points(ideal, focal=110.0)
    radial = undistort.fit_radial(ideal, observed, width, height)

    assert radial["model"] == "radial"
    assert len(radial["k"]) == 3
    assert len(radial["homography"]) == 9
    # Round-trips through the generic entry point.
    a = undistort.apply_warp(radial, ideal[:, 0], ideal[:, 1])
    b = undistort._apply_radial(radial, ideal[:, 0], ideal[:, 1])
    assert np.allclose(a, b)


def _chessboard_image(rows=6, cols=9, width=640, height=480):
    """Synthetic chessboard with `rows` x `cols` inner corners."""
    cv2 = pytest.importorskip("cv2")

    square = 42
    board_w, board_h = (cols + 1) * square, (rows + 1) * square
    x0, y0 = (width - board_w) // 2, (height - board_h) // 2
    image = np.full((height, width), 180, dtype=np.uint8)
    for row in range(rows + 1):
        for col in range(cols + 1):
            value = 35 if (row + col) % 2 == 0 else 235
            image[y0 + row * square:y0 + (row + 1) * square,
                  x0 + col * square:x0 + (col + 1) * square] = value
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_chessboard_detection_writes_debug_overlay(tmp_path):
    """Offline fit debugging should leave proof images for chessboards too."""
    pytest.importorskip("cv2")

    debug = tmp_path / "chess_debug.png"
    found = undistort.detect_grid(_chessboard_image(), 6, 9, "chessboard",
                                  debug_path=str(debug))

    assert found is not None
    assert len(found) == 54
    assert debug.exists()


def _ball_image(rotation_deg, rows=6, cols=9, width=640, height=480):
    """Synthetic pressed ball grid, rotated by `rotation_deg`."""
    cv2 = pytest.importorskip("cv2")

    ideal = undistort.ideal_lattice(rows, cols, width, height, 0.62)
    centre = np.array([width / 2, height / 2])
    theta = np.deg2rad(rotation_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    points = centre + (ideal - centre) @ rot.T

    ys, xs = np.mgrid[0:height, 0:width]
    image = np.full((height, width), 150.0)
    for x, y in points:
        image[((xs - x) ** 2 + (ys - y) ** 2) < 25] -= 45
    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_grid_detection_survives_any_press_rotation():
    """The array will not be laid down perfectly square; detection must cope."""
    pytest.importorskip("cv2")

    for rotation in (0, 15, 30, 45, 60, 75, 90, 180):
        image = _ball_image(rotation)
        found = undistort.detect_grid(image, 6, 9, "circles", min_area=10)
        assert found is not None, f"detection failed at {rotation} deg"
        assert len(found) == 54


def test_assign_lattice_recovers_indices_under_fisheye():
    """A fisheye scrambles spacing but not order; assignment must survive it."""

    ideal = undistort.ideal_lattice(6, 9, 640, 480, 0.85)
    distorted = _fisheye_points(ideal, focal=120.0)          # strong barrel
    assert np.hypot(*(distorted - ideal).T).max() > 100      # really bends

    # Shuffle: the detector returns points in arbitrary order.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(distorted))
    ij, xy = undistort.assign_lattice(distorted[perm], 6, 9)

    assert len(ij) == 54                                     # every cell filled
    # Each returned point must carry the (row, col) of the ideal lattice cell
    # it actually came from -- reconstruct the original index and compare.
    for (i, j), point in zip(ij, xy):
        original = np.argmin(np.hypot(*(distorted - point).T))
        assert (original // 9, original % 9) == (i, j)


def test_assign_lattice_tolerates_missing_balls():
    """A non-parallel press leaves some balls untouched; the rest must still map."""

    ideal = undistort.ideal_lattice(6, 9, 640, 480, 0.85)
    distorted = _fisheye_points(ideal, focal=120.0)
    drop = {0, 1, 9, 17, 26, 53}                             # scattered gaps
    kept = np.array([p for n, p in enumerate(distorted) if n not in drop])

    ij, xy = undistort.assign_lattice(kept, 6, 9)

    assert len(ij) == len(kept)                              # no spurious cells
    got = {(int(i), int(j)) for i, j in ij}
    assert (0, 0) not in got and (5, 8) not in got           # dropped corners
    assert (3, 4) in got                                     # centre still there


def test_robust_detector_reads_a_faint_uneven_press():
    """The near rows image far darker than the far ones on a tilted array.

    A single-threshold blob detector loses the faint rows; the flat-fielded
    robust detector must recover the whole grid.
    """
    cv2 = pytest.importorskip("cv2")

    # Ball grid whose contrast fades from 60 grey levels (bottom) to 8 (top),
    # plus a broad illumination gradient -- the real failure mode.
    width, height = 640, 480
    ideal = undistort.ideal_lattice(6, 9, width, height, 0.62)
    ys, xs = np.mgrid[0:height, 0:width]
    image = (150 + 40 * (xs / width)).astype(np.float64)     # lighting gradient
    for n, (x, y) in enumerate(ideal):
        depth = 8 + 52 * (y / height)                        # faint at the top
        image[((xs - x) ** 2 + (ys - y) ** 2) < 25] -= depth
    image = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8),
                         cv2.COLOR_GRAY2BGR)

    result = undistort.detect_grid_robust(image, 6, 9, min_area=10)
    assert result is not None
    ij, xy = result
    assert len(xy) >= 48                                     # nearly all 54
    # The top (faint) row must be represented, not just the dark bottom.
    assert (ij[:, 0] == 0).any()


def test_grid_correspondences_pairs_each_ball_with_its_ideal_cell():
    """The returned ideal point for each detection is that cell's lattice point."""
    pytest.importorskip("cv2")

    image = _ball_image(12)                                  # tilted press
    ideal, detected = undistort.grid_correspondences(
        image, 6, 9, "circles", 640, 480, 0.8, False, None, 10, 8000)

    assert len(ideal) == len(detected)
    full = undistort.ideal_lattice(6, 9, 640, 480, 0.8)
    # Every returned ideal point is one of the true lattice points.
    for point in ideal:
        assert np.min(np.hypot(*(full - point).T)) < 1e-6


def test_press_tilt_is_cancelled_so_output_orientation_is_reproducible():
    """Pressing the array tilted must not tilt every rectified frame.

    The fitted map takes an axis-aligned lattice onto however the pattern was
    laid down, so without correction a 30-degree press rotates all output by
    30 degrees and two calibration sessions disagree.
    """
    pytest.importorskip("cv2")

    width, height = 640, 480
    ideal = undistort.ideal_lattice(6, 9, width, height, 0.62)

    for rotation in (0, 15, 30, 45, 80):
        found = undistort.detect_grid(_ball_image(rotation), 6, 9, "circles",
                                      min_area=10)
        ordered = undistort.canonicalize_grid(found, 6, 9)
        warp, _keep = undistort.fit_warp_robust(
            ideal, ordered, degree=5, model="radial",
            width=width, height=height)
        assert undistort.warp_orientation(warp, width, height) > 0

        aligned = undistort.align_warp(warp, width, height)
        residual_tilt = undistort.warp_rotation(aligned, width, height)
        # Axis-aligned to within a degree. 180 is allowed and is NOT a defect:
        # a symmetric grid is identical under a half turn, so the observations
        # cannot say which way is up.
        assert min(abs(residual_tilt), abs(abs(residual_tilt) - 180)) < 1.0
        assert undistort.warp_orientation(aligned, width, height) > 0


def test_exact_quarter_turn_press_is_caught_as_mirrored():
    """A 6x9 grid pressed at exactly 90 deg reads as a transposed 9x6.

    Transposition is a reflection, so the correspondence mirrors and the fit is
    unusable. canonicalize_grid cannot repair it -- sorting rows and columns
    cannot undo a transpose -- so the value of the Jacobian check is that it
    refuses the result instead of silently producing a flipped map. Measured
    boundary: 0-80 deg fit correctly, only exactly 90 mirrors.
    """
    pytest.importorskip("cv2")

    width, height = 640, 480
    ideal = undistort.ideal_lattice(6, 9, width, height, 0.62)
    found = undistort.detect_grid(_ball_image(90), 6, 9, "circles", min_area=10)
    assert found is not None                      # detection itself still works

    ordered = undistort.canonicalize_grid(found, 6, 9)
    warp, _keep = undistort.fit_warp_robust(
        ideal, ordered, degree=5, model="radial", width=width, height=height)
    assert undistort.warp_orientation(warp, width, height) < 0
