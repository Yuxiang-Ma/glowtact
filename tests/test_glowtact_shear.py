"""Headless correctness tests for glowtact.shear.

Each tracker is fed a synthetic speckle image shifted by a KNOWN amount, so we
can assert it recovers the shift -- no hardware, no ground-truth file. Also
checks the contact mask isolates a dark blob and that the mask keeps a blob from
polluting the shear estimate.
"""
import os
import sys

import numpy as np
import pytest


from glowtact import shear                                           # noqa: E402


def _speckle(h=240, w=320, seed=0):
    """A low-contrast speckle field on a bright background, like the GlowTact."""
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 185, np.float32)
    noise = rng.normal(0, 18, (h, w)).astype(np.float32)
    import cv2
    noise = cv2.GaussianBlur(noise, (0, 0), 1.1)      # speckle grain size
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _shift(img, dx, dy):
    import cv2

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          borderMode=cv2.BORDER_REFLECT)


@pytest.mark.parametrize("name", list(shear.TRACKERS))
@pytest.mark.parametrize("dx,dy", [(3.0, 0.0), (0.0, -2.0), (2.0, 2.0)])
def test_tracker_recovers_known_shift(name, dx, dy):
    base = _speckle()
    shifted = _shift(base, dx, dy)
    tr = shear.TRACKERS[name]()
    try:
        tr.reset(base)
    except Exception as exc:                    # e.g. DIS not in this cv2 build
        pytest.skip(f"{name} unavailable: {exc}")
    no_contact = np.zeros(base.shape, bool)
    r = tr.estimate(shifted, no_contact)
    # a warpAffine shift of the scene by +dx means features move by +dx;
    # every tracker reports that motion (sign consistent across trackers).
    assert abs(r.dx - dx) < 1.0, f"{name} dx {r.dx:.2f} vs {dx}"
    assert abs(r.dy - dy) < 1.0, f"{name} dy {r.dy:.2f} vs {dy}"


def test_contact_mask_isolates_blob():
    import cv2

    base = _speckle()
    frame = base.copy()
    cv2.circle(frame, (160, 120), 40, 40, -1)         # dark indentation
    cm = shear.contact_mask(frame, base)
    assert cm[120, 160]                                # centre flagged
    assert not cm[10, 10]                               # far corner clear
    assert 0.02 < cm.mean() < 0.5                       # a blob, not everything


def test_phasecorr_resists_pure_press():
    """The recommended tracker reports ~0 shear for a pure normal press.

    A dark blob is added without moving the speckle; phasecorr (global, over the
    masked speckle) must not turn that into shear. This is the robustness the
    real-data eval rewards -- phasecorr has the lowest indentation intrusion.
    """
    import cv2

    base = _speckle()
    frame = base.copy()
    cv2.circle(frame, (160, 120), 45, 30, -1)          # pure normal press
    cm = shear.contact_mask(frame, base)
    tr = shear.PhaseCorrTracker()
    tr.reset(base)
    r = tr.estimate(frame, cm)
    assert abs(r.dx) < 0.8 and abs(r.dy) < 0.8         # ~0 shear despite blob




def test_evaluate_scores_present(tmp_path):
    """evaluate_video runs end to end on a tiny synthetic clip and scores."""
    import cv2

    base = _speckle()
    path = str(tmp_path / "clip.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
                         (base.shape[1], base.shape[0]))
    for i in range(12):
        f = _shift(base, 0.1 * i, 0.0)                 # gentle drift = shear
        vw.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    vw.release()
    out = shear.evaluate_video(path, tracker_names=["lk", "phasecorr"])
    for n in ("lk", "phasecorr"):
        assert "score" in out[n]
        assert 0.0 <= out[n]["score"] <= 1.0


# ------------------------------------------------ coherence / slip / fill --

def _grid_field(rows=7, cols=9, spacing=20):
    """A synthetic ShearField on a regular grid with a uniform shear."""
    ys, xs = np.mgrid[0:rows, 0:cols]
    nodes = np.stack([xs.ravel() * spacing + 30, ys.ravel() * spacing + 30],
                     1).astype(np.float32)
    vecs = np.tile(np.array([1.5, -0.5], np.float32), (rows * cols, 1))
    conf = np.ones(rows * cols, np.float32)
    return shear.ShearField(nodes, vecs, conf, (rows, cols))


def test_coherence_rejects_lone_outlier():
    f = _grid_field()
    f.vecs[25] = [12.0, 9.0]                         # one wild node
    filt = f.filter_coherence(min_conf=0.5, max_dev=1.5)
    assert filt.conf[25] == 0.0                       # rejected
    assert (filt.conf >= 0.5).sum() == len(f.conf) - 1  # only that one


def test_slip_index_zero_for_affine_field():
    """A pure affine field (shift+stretch) has no slip."""
    f = _grid_field()
    c = f.nodes.mean(0)
    f.vecs = ((f.nodes - c) * 0.05 + np.array([2.0, 1.0])).astype(np.float32)
    assert f.slip_index(0.5) < 1e-3


def test_slip_index_positive_for_local_discontinuity():
    f = _grid_field()
    f.vecs[30:33] += np.array([4.0, 4.0], np.float32)   # a slipping patch
    assert f.slip_index(0.5) > 0.5


def test_fill_fills_enclosed_hole_only():
    f = _grid_field()
    # punch a hole in the interior (enclosed) and one on the border edge
    f.conf[31] = 0.0            # interior (row3,col4) -> enclosed
    f.conf[0] = 0.0             # corner -> not enclosed
    filled, mask = f.fill(min_conf=0.5)
    assert mask[31] and not mask[0]
    # filled interior node recovers ~the uniform shear
    assert np.allclose(filled.vecs[31], [1.5, -0.5], atol=0.2)


def test_flow_quality_near_perfect_on_pure_shift():
    """A clean translated speckle scores ideal: FB≈0, photometric≪1."""
    import cv2
    base = _speckle()
    cur = _shift(base, 2.0, -1.0)
    q = shear.flow_quality(base, cur)
    assert q["fb_error"] < 0.1
    assert q["photometric"] < 0.4          # flow explains the motion
    assert q["reliable_frac"] > 0.8


def test_flow_score_ranks_clean_above_noisy():
    import cv2
    base = _speckle()
    cur = _shift(base, 2.0, -1.0)
    noisy = np.clip(cur.astype(np.float32) +
                    np.random.default_rng(1).normal(0, 40, cur.shape), 0, 255
                    ).astype(np.uint8)
    clean = shear.flow_score(base, cur)
    dirty = shear.flow_score(base, noisy)
    assert clean["score"] > 90 and clean["grade"] == "excellent"
    assert dirty["score"] < clean["score"]
    assert 0 <= dirty["score"] <= 100


# -- temporal slip: imprint moving relative to the gel ------------------------

def _speckle(seed=0, size=(240, 240)):
    import cv2
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 255, size, dtype=np.uint8)
    return cv2.GaussianBlur(g, (0, 0), 1.2)


def _pair(object_shift, gel_shift=(0, 0)):
    """prev/cur frames: a textured 'imprint' square that moves by object_shift
    while the surrounding gel speckle moves by gel_shift."""
    prev = _speckle(0)
    cur = np.roll(np.roll(prev, gel_shift[1], 0), gel_shift[0], 1)
    imprint = _speckle(1)[80:160, 80:160]
    prev = prev.copy(); prev[80:160, 80:160] = imprint
    cur = cur.copy()
    ox, oy = object_shift
    cur[80 + oy:160 + oy, 80 + ox:160 + ox] = imprint
    contact = np.zeros(prev.shape, bool); contact[84:156, 84:156] = True
    return prev, cur, contact


def test_object_sliding_over_still_gel_is_slip():
    pytest.importorskip("cv2")
    prev, cur, contact = _pair(object_shift=(3, 0))
    r = shear.relative_slip(prev, cur, contact, spacing=20, win=31)
    assert r["n_contact"] >= 4
    assert r["slip_frac"] > 0.8                       # the whole imprint slides
    assert abs(r["v_obj"][0] - 3.0) < 0.8 and abs(r["v_gel"][0]) < 0.4
    assert not r["slip"][~r["inside"]].any()          # never flagged off-imprint


def test_object_and_gel_moving_together_is_grip():
    """The imprint carried along with the gel is exactly what gripping looks like."""
    pytest.importorskip("cv2")
    prev, cur, contact = _pair(object_shift=(3, 0), gel_shift=(3, 0))
    r = shear.relative_slip(prev, cur, contact, spacing=20, win=31)
    assert r["n_contact"] >= 4
    assert r["slip_frac"] < 0.2
    assert r["stick_ratio"] > 0.8


def test_nothing_moving_is_not_slip():
    pytest.importorskip("cv2")
    prev, cur, contact = _pair(object_shift=(0, 0))
    r = shear.relative_slip(prev, cur, contact, spacing=20, win=31)
    assert r["n_slip"] == 0 and r["slip_area_px"] == 0.0


def test_despeckle_removes_noise_but_keeps_ridges():
    """The measured contract: speckle-scale noise drops hard, a ridge pattern
    (fingerprint/thread scale, ~6 px period) survives nearly untouched."""
    cv2 = pytest.importorskip("cv2")
    from glowtact.imgfilt import despeckle
    rng = np.random.default_rng(0)
    ys, xs = np.mgrid[0:200, 0:200]
    ridges = 40 + 30 * np.sin(2 * np.pi * xs / 6.0)          # 6 px pitch band
    noisy = np.clip(ridges + rng.normal(0, 8, ridges.shape), 0, 255)
    out = despeckle(noisy).astype(np.float32)
    band = lambda a: float(np.mean((cv2.GaussianBlur(a, (0, 0), 1.5)
                                    - cv2.GaussianBlur(a, (0, 0), 2.7)) ** 2))
    keep = band(out) / band(np.float32(ridges))
    flat = np.clip(40 + rng.normal(0, 8, ridges.shape), 0, 255)
    resid = despeckle(flat).astype(np.float32).std()
    assert keep > 0.7                                        # ridges survive
    assert resid < 0.5 * np.float32(flat).std()              # noise halved+
