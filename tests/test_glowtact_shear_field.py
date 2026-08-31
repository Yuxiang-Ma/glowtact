import sys

import numpy as np
import pytest


from glowtact.shear import ShearField  # noqa: E402


def _field(vecs, conf=None, grid=(9, 9)):
    rows, cols = grid
    ys, xs = np.mgrid[0:rows, 0:cols]
    nodes = np.stack([xs.ravel() * 26.0, ys.ravel() * 26.0], axis=1)
    vecs = np.asarray(vecs, float).reshape(-1, 2)
    conf = np.ones(len(nodes)) if conf is None else np.asarray(conf, float)
    return ShearField(nodes, vecs, conf, grid)


def test_untouched_gel_reads_zero_twist():
    """The reported bug: flow jitter must not register as rotation.

    Sub-pixel jitter at the measured no-contact level (mean speed ~0.2 px)
    is below the motion floor, so the answer is exactly zero -- not "whatever
    ratio the noise happened to form". Magnitude weighting ALONE would make
    this worse, since the loudest nodes here are simply the noisiest.
    """
    rng = np.random.default_rng(0)
    jitter = rng.normal(0, 0.15, (81, 2))
    f = _field(jitter)
    assert f.curl() == 0.0
    assert f.divergence() == 0.0
    # the unweighted mean is what used to be reported, and it is not zero
    assert abs(f._deriv("curl", 0.15, weighted=False)) > 0


def test_real_rotation_still_measured():
    """A solid-body rotation must come back with the right sign and scale."""
    rows, cols = 9, 9
    ys, xs = np.mgrid[0:rows, 0:cols].astype(float)
    cx = cy = 4.0
    k = 0.5                                   # px of motion per grid step
    vecs = np.stack([(-(ys - cy) * k).ravel(), ((xs - cx) * k).ravel()], 1)
    f = _field(vecs)
    # curl of (-y, x)*k on a unit-step grid is 2k
    assert f.curl() == pytest.approx(2 * k, rel=0.05)
    assert abs(f.divergence()) < 1e-6


def test_pure_expansion_is_spread_not_twist():
    rows, cols = 9, 9
    ys, xs = np.mgrid[0:rows, 0:cols].astype(float)
    k = 0.5
    vecs = np.stack([((xs - 4) * k).ravel(), ((ys - 4) * k).ravel()], 1)
    f = _field(vecs)
    assert f.divergence() == pytest.approx(2 * k, rel=0.05)
    assert abs(f.curl()) < 1e-6


def test_magnitude_weighting_follows_the_moving_region():
    """Rotation in a moving corner must survive a quiet majority.

    Unweighted, 77 still nodes outvote 4 rotating ones and the reading
    collapses toward zero -- which is how a real twist got lost in the average.
    """
    rows, cols = 9, 9
    vecs = np.zeros((rows * cols, 2))
    ys, xs = np.mgrid[0:rows, 0:cols]
    spin = (ys < 5) & (xs < 5)          # a press-sized patch, not a corner pixel
    k = 1.5
    vecs[spin.ravel(), 0] = -(ys[spin] - 2) * k
    vecs[spin.ravel(), 1] = (xs[spin] - 2) * k
    f = _field(vecs)
    weighted = f.curl()
    plain = f._deriv("curl", 0.15, weighted=False)
    truth = 2 * k                       # curl of (-y, x)*k
    # The weighted reading recovers most of the real rotation; the plain mean
    # is diluted by the still majority down to a small fraction of it.
    assert weighted > 0.5 * truth
    assert plain < 0.2 * truth
    assert weighted > 3 * abs(plain)


def test_low_confidence_nodes_do_not_manufacture_rotation():
    """The contact hole used to be pasted with the field mean.

    np.gradient across that seam is a step, and a step differentiates into a
    large fake rotation exactly at the rim. Untrusted nodes must be dropped.
    """
    rows, cols = 9, 9
    vecs = np.tile([2.0, 0.0], (rows * cols, 1))     # uniform slide, no curl
    conf = np.ones(rows * cols)
    hole = np.zeros((rows, cols), bool)
    hole[3:6, 3:6] = True
    conf[hole.ravel()] = 0.0
    vecs[hole.ravel()] = [-4.0, 3.0]                 # garbage under the object
    f = _field(vecs, conf)
    assert abs(f.curl()) < 1e-9
    assert abs(f.divergence()) < 1e-9


def test_mean_is_confidence_weighted_and_ignores_the_hole():
    vecs = np.tile([1.0, -2.0], (81, 1))
    conf = np.ones(81)
    conf[:10] = 0.0
    vecs[:10] = [50.0, 50.0]
    f = _field(vecs, conf)
    assert f.mean() == pytest.approx((1.0, -2.0))


# -- slip area / friction cone ------------------------------------------------

def test_smooth_affine_deformation_is_all_grip():
    """A press that shifts, twists and stretches the gel as one body slips nowhere."""
    rows, cols = 9, 9
    ys, xs = np.mgrid[0:rows, 0:cols].astype(float)
    dx = 1.5 + 0.08 * (xs - 4) - 0.05 * (ys - 4)      # shift + stretch + twist
    dy = -0.6 + 0.05 * (xs - 4) + 0.08 * (ys - 4)
    f = _field(np.stack([dx.ravel(), dy.ravel()], 1))
    mask, _ = f.slip_mask()
    assert not mask.any()
    stats = f.slip_area()
    assert stats["stick_ratio"] == 1.0 and stats["n_slip"] == 0
    assert np.isnan(stats["annularity"])


def test_rim_patch_sliding_against_the_bulk_is_the_slip_area():
    """Cattaneo-Mindlin: nodes at the rim moving relative to the body are slip."""
    rows, cols = 11, 11
    ys, xs = np.mgrid[0:rows, 0:cols].astype(float)
    n = rows * cols
    vec = np.stack([np.full(n, 1.0), np.full(n, 0.0)], 1)     # body: rigid shift
    r = np.hypot(xs - 5, ys - 5)
    # Slip begins on ONE edge of the contact (the trailing side under a
    # tangential load): a rim patch on the right, a minority of the nodes.
    rim = ((r >= 3.5) & (xs > 7)).ravel()
    assert 0.1 < rim.mean() < 0.35
    vec[rim, 0] += 2.0                                       # patch slides on
    f = _field(vec, grid=(rows, cols))
    mask, mag = f.slip_mask()
    assert mask[rim].mean() > 0.9 and mask[~rim].mean() < 0.05
    stats = f.slip_area()
    assert abs(stats["slip_frac"] - rim.mean()) < 0.05
    assert stats["annularity"] > 0.7                         # sits at the rim
    # The plain least-squares affine is pulled by the patch and misfires:
    rx, _ry, _ = f.residual_field(robust=False)
    assert np.abs(rx[~rim]).mean() > 0.3                     # body contaminated


def test_slip_area_only_counts_nodes_on_the_contact():
    """A stray outlier far from the imprint must not register as slip."""
    vec = np.zeros((81, 2)); vec[0] = (4.0, 0.0)              # lone corner node
    f = _field(vec)
    contact = np.zeros((240, 240), bool)
    contact[90:130, 90:130] = True                           # centre imprint
    assert f.slip_mask()[0][0]                               # no mask: flagged
    assert not f.slip_mask(contact=contact)[0][0]            # gated: ignored
    assert f.slip_area(contact=contact)["n_contact"] < 81


def test_traction_ratio_divides_shear_by_local_pressure():
    """|shear| / pressure per node, NaN where there is nothing pressing."""
    pytest.importorskip("cv2")
    vec = np.stack([np.full(81, 2.0), np.zeros(81)], 1)
    f = _field(vec)
    pressure = np.zeros((240, 240), np.float32)
    pressure[:, :120] = 50.0                                 # left half loaded
    ratio = f.traction_ratio(pressure, win=5)
    left = f.nodes[:, 0] < 100
    right = f.nodes[:, 0] > 140
    assert np.allclose(ratio[left], 2.0 / 50.0, rtol=1e-3)
    assert np.isnan(ratio[right]).all()
    cone, _ = f.friction_cone_mask(pressure, mu=0.03, win=5)
    assert cone[left].all() and not cone[right].any()
