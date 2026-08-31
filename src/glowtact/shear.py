"""Marker-free shear tracking from the GlowTact's natural speckle texture.

The GlowTact gel has no printed tracking dots (unlike the GelSight Mini), but
its surface carries a faint, fixed speckle pattern. When the contact shears the
gel tangentially, that whole speckle field translates/warps; tracking it gives a
marker-free shear estimate.

Two problems this module is built around:

1. **The speckle is low-contrast.** CLAHE (and, for some trackers, a high-pass)
   lifts it into a trackable signal -- ~2000 corners per frame after CLAHE.
2. **Object indentation breaks tracking.** A press paints a large dark blob into
   the image that is *normal* deformation, not shear; naive flow locks onto its
   edges and reports huge bogus motion. Every tracker here therefore excludes a
   **contact mask** (plus the bright aperture rim) and aggregates shear only
   from the surrounding speckle -- so a straight-down press reads ~0 shear.

Four trackers share one interface (:class:`ShearTracker`), all anchored to a
rest baseline so the output is absolute shear from the untouched state:

  * ``lk``        -- sparse Lucas-Kanade on speckle corners (naturally robust:
                     corners that land in the blob are dropped).
  * ``farneback`` -- dense Farneback flow, masked-median aggregation.
  * ``dis``       -- dense DIS optical flow (faster, edge-preserving).
  * ``phasecorr`` -- global phase correlation over the valid speckle region.

Because there is no force ground truth, quality is judged by four
self-supervised metrics (:func:`evaluate_video`): forward-backward warp
residual, no-contact noise floor, indentation intrusion, and directional
coherence. Lower is better for the first three, higher for the last.

Pure OpenCV + NumPy; matplotlib only for the optional evaluation plots.
"""
import dataclasses

import numpy as np


# --------------------------------------------------------------- preprocessing --

# Optional id-keyed cache, warmed by evaluate_video so the same frame is
# CLAHE-equalised once instead of per tracker and per warp-residual call. Off
# (None) for normal library use, where each frame is enhanced on demand.
_ENH_CACHE = None


def enhance(gray, clip=3.0, tiles=8):
    """CLAHE-equalised grayscale -- lifts the faint speckle for tracking."""
    import cv2

    if _ENH_CACHE is not None:
        hit = _ENH_CACHE.get(id(gray))
        if hit is not None:
            return hit
    out = cv2.createCLAHE(clip, (tiles, tiles)).apply(gray)
    if _ENH_CACHE is not None:
        _ENH_CACHE[id(gray)] = out
    return out


def local_contrast_norm(gray, sigma=15.0):
    """Divide out local illumination: (I - localMean) / localStd.

    Flattens the GlowTact's uneven lighting (the bright hotspot) so the speckle
    is trackable everywhere -- the piece CLAHE alone can't do. Verified to cut
    the dense flow's worst-case FB error ~20x on hard/deep presses.
    """
    import cv2

    f = gray.astype(np.float32)
    m = cv2.GaussianBlur(f, (0, 0), sigma)
    sd = np.sqrt(cv2.GaussianBlur((f - m) ** 2, (0, 0), sigma)) + 3.0
    return cv2.normalize((f - m) / sd, None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)


def flow_enhance(gray, preprocess="lcn_clahe", clip=8.0, tiles=16):
    """Preprocessing for the dense-flow method: LCN+CLAHE (default) or CLAHE.

    ``lcn_clahe`` (local-contrast-normalise, then CLAHE) is the evidence-based
    optimum for the flow -- robust to the illumination hotspot and nearly
    parameter-insensitive; ``clahe`` reproduces the older behaviour.
    """
    import cv2

    src = local_contrast_norm(gray) if preprocess == "lcn_clahe" else gray
    return cv2.createCLAHE(clip, (tiles, tiles)).apply(src)


def highpass(gray, sigma=9.0):
    """Remove smooth shading (and the blob's low-freq dip), keep speckle."""
    import cv2

    g = enhance(gray)
    hp = cv2.subtract(g, cv2.GaussianBlur(g, (0, 0), sigma))
    return cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def fov_mask(shape, radius_frac=0.82):
    """Boolean disc that drops the bright circular aperture rim near the edge."""
    h, w = shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r0 = radius_frac * min(h, w) / 2.0
    ys, xs = np.mgrid[0:h, 0:w]
    return (np.hypot(xs - cx, ys - cy) <= r0)


def contact_mask(gray, baseline_gray, thresh=22, dilate=9):
    """Boolean mask of the indentation blob (where the gel darkened vs rest).

    The object press darkens the gel; thresholding baseline-minus-current and
    dilating gives the region to EXCLUDE from shear aggregation. Returns an
    all-False mask when nothing is touching.
    """
    import cv2

    dark = baseline_gray.astype(np.int16) - gray.astype(np.int16)
    m = (dark > thresh).astype(np.uint8)
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if dilate:
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    return m.astype(bool)


# ------------------------------------------------------------------- result --

@dataclasses.dataclass
class ShearResult:
    dx: float                     # median tangential shift (px), +x = right
    dy: float                     # +y = down
    coherence: float              # 0..1, agreement of the flow vectors
    n: int                        # samples backing the estimate
    field: object = None          # optional (points, vectors) for viz


# ----------------------------------------------------------------- trackers --

class ShearTracker:
    """Common interface: reset() on a rest baseline, then estimate() per frame."""

    name = "base"

    def reset(self, baseline_gray):
        raise NotImplementedError

    def estimate(self, gray, contact):
        raise NotImplementedError

    @staticmethod
    def _coherence(vecs):
        """Fraction of vectors within ~30 deg of the median direction."""
        if len(vecs) == 0:
            return 0.0
        med = np.median(vecs, axis=0)
        n = np.linalg.norm(med)
        if n < 1e-6:
            return 0.0
        md = med / n
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        unit = vecs / np.maximum(norms, 1e-6)
        return float((unit @ md > 0.866).mean())


class LKTracker(ShearTracker):
    """Sparse Lucas-Kanade on speckle corners, anchored to the baseline.

    Corners are found once on the rest frame; each frame they are tracked from
    rest. Corners that fail, drift too far, or land inside the contact blob are
    rejected, so the shear is the median motion of the intact surround.
    """

    name = "lk"

    def __init__(self, max_corners=1500, quality=0.01, min_dist=6,
                 win=21, max_err=20.0):
        self.max_corners = max_corners
        self.quality = quality
        self.min_dist = min_dist
        self.win = win
        self.max_err = max_err

    def reset(self, baseline_gray):
        import cv2

        self.base = enhance(baseline_gray)
        m = fov_mask(baseline_gray.shape).astype(np.uint8) * 255
        self.p0 = cv2.goodFeaturesToTrack(
            self.base, self.max_corners, self.quality, self.min_dist, mask=m)

    def estimate(self, gray, contact):
        import cv2

        cur = enhance(gray)
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.base, cur, self.p0, None,
            winSize=(self.win, self.win), maxLevel=3)
        st = st.ravel().astype(bool)
        err = err.ravel()
        p0 = self.p0.reshape(-1, 2)
        p1r = p1.reshape(-1, 2)
        good = st & (err < self.max_err)
        # drop features whose current position sits in the indentation blob
        xi = np.clip(p1r[:, 0].astype(int), 0, contact.shape[1] - 1)
        yi = np.clip(p1r[:, 1].astype(int), 0, contact.shape[0] - 1)
        good &= ~contact[yi, xi]
        vecs = (p1r - p0)[good]
        if len(vecs) < 8:
            return ShearResult(0.0, 0.0, 0.0, 0)
        med = np.median(vecs, axis=0)
        return ShearResult(float(med[0]), float(med[1]),
                           self._coherence(vecs), int(len(vecs)),
                           field=(p0[good], vecs))


class _DenseTracker(ShearTracker):
    """Shared masked-median aggregation for dense-flow trackers."""

    def reset(self, baseline_gray):
        self.base = enhance(baseline_gray)
        self.fov = fov_mask(baseline_gray.shape)

    def _flow(self, base, cur):
        raise NotImplementedError

    def estimate(self, gray, contact):
        flow = self._flow(self.base, enhance(gray))
        valid = self.fov & ~contact
        fx = flow[..., 0][valid]
        fy = flow[..., 1][valid]
        if fx.size < 50:
            return ShearResult(0.0, 0.0, 0.0, 0)
        med = (float(np.median(fx)), float(np.median(fy)))
        vecs = np.stack([fx, fy], axis=1)
        # subsample coherence for speed
        sub = vecs[:: max(1, len(vecs) // 4000)]
        return ShearResult(med[0], med[1], self._coherence(sub), int(fx.size),
                           field=("dense", flow, valid))


class FarnebackTracker(_DenseTracker):
    name = "farneback"

    def _flow(self, base, cur):
        import cv2

        return cv2.calcOpticalFlowFarneback(
            base, cur, None, 0.5, 3, 25, 3, 5, 1.2, 0)


class DISTracker(_DenseTracker):
    name = "dis"

    def __init__(self):
        self._dis = None

    def _flow(self, base, cur):
        import cv2

        if self._dis is None:
            self._dis = cv2.DISOpticalFlow_create(
                cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        return self._dis.calc(base, cur, None)


class PhaseCorrTracker(ShearTracker):
    """Global phase correlation over the valid speckle region (rest-anchored)."""

    name = "phasecorr"

    def reset(self, baseline_gray):
        import cv2

        self.fov = fov_mask(baseline_gray.shape)
        self.base = highpass(baseline_gray).astype(np.float32)
        self.base *= self.fov
        self.win = cv2.createHanningWindow(
            (baseline_gray.shape[1], baseline_gray.shape[0]), cv2.CV_32F)

    def estimate(self, gray, contact):
        import cv2

        keep = (self.fov & ~contact).astype(np.float32)
        cur = highpass(gray).astype(np.float32) * keep
        base = self.base * keep
        (sx, sy), resp = cv2.phaseCorrelate(base, cur, self.win)
        # phaseCorrelate returns the shift of `cur` relative to `base`
        return ShearResult(float(sx), float(sy), float(resp), int(keep.sum()))


def flow_quality(base_gray, cur_gray, flow=None, mask=None,
                 preprocess="lcn_clahe", clip=8.0, tiles=16):
    """No-ground-truth quality metrics for a dense flow of base->cur.

    We have no labelled flow, so quality is judged self-supervised:

    * ``fb_error``   -- forward-backward consistency: flow forward then the
      backward flow at the landed point should return to the start;
      ``|f_fwd(x) + f_bwd(x+f_fwd)|``. The single most telling no-GT metric --
      it spikes on occlusion, aperture ambiguity and texture-less regions.
      Reported as the median over ``mask`` (px). Lower is better.
    * ``photometric`` -- warp the enhanced current frame back along the flow and
      compare to the enhanced base over ``mask``, as a fraction of the no-warp
      residual. <1 means the flow explains the pixel motion; lower is better.
    * ``reliable_frac`` -- fraction of ``mask`` where the structure tensor is
      well-conditioned (min-eigenvalue above a floor): the flow is only
      trustworthy where there is 2-D texture, so this is the trustable area.

    ``mask`` (bool, defaults to all) should exclude the contact blob -- flow in
    the occluded indentation is meaningless. Farneback is used if ``flow`` is
    None (forward); the backward flow is always computed here.

    ``preprocess``/``clip``/``tiles`` must match whatever produced ``flow``.
    This used to be hardcoded to the legacy ``enhance`` (CLAHE 3.0/8), so a
    flow computed with LCN+CLAHE was scored against CLAHE-only images: the
    backward flow disagreed with the forward one for no reason but the
    mismatch, and the structure-tensor reliability could not see the
    preprocessing difference at all -- which is precisely the difference the
    term exists to measure.
    """
    import cv2

    be = flow_enhance(base_gray, preprocess, clip, tiles).astype(np.float32)
    ce = flow_enhance(cur_gray, preprocess, clip, tiles).astype(np.float32)
    fb_par = dict(pyr_scale=0.5, levels=3, winsize=25, iterations=3,
                  poly_n=5, poly_sigma=1.2, flags=0)
    if flow is None:
        flow = cv2.calcOpticalFlowFarneback(be, ce, None, **fb_par)
    back = cv2.calcOpticalFlowFarneback(ce, be, None, **fb_par)
    h, w = base_gray.shape
    if mask is None:
        mask = np.ones((h, w), bool)

    # forward-backward consistency
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    mx = (xs + flow[..., 0]).clip(0, w - 1)
    my = (ys + flow[..., 1]).clip(0, h - 1)
    bwx = cv2.remap(back[..., 0], mx, my, cv2.INTER_LINEAR)
    bwy = cv2.remap(back[..., 1], mx, my, cv2.INTER_LINEAR)
    fb = np.hypot(flow[..., 0] + bwx, flow[..., 1] + bwy)

    # photometric warp residual (ratio vs no-warp)
    warped = cv2.remap(ce, mx, my, cv2.INTER_LINEAR)
    m = mask
    base_res = np.abs(ce[m] - be[m]).mean()
    warp_res = np.abs(warped[m] - be[m]).mean()

    # structure-tensor reliability (min eigenvalue of the windowed gradient tensor)
    gx = cv2.Sobel(be, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(be, cv2.CV_32F, 0, 1, ksize=3)
    k = (0, 0)
    Axx = cv2.GaussianBlur(gx * gx, k, 3)
    Ayy = cv2.GaussianBlur(gy * gy, k, 3)
    Axy = cv2.GaussianBlur(gx * gy, k, 3)
    tmp = np.sqrt(np.maximum((Axx - Ayy) ** 2 + 4 * Axy ** 2, 0))
    min_eig = 0.5 * (Axx + Ayy - tmp)
    floor = 0.01 * float(min_eig.max())
    return {
        "fb_error": float(np.median(fb[m])),
        "fb_error_p90": float(np.percentile(fb[m], 90)),
        "photometric": float(warp_res / max(base_res, 1e-6)),
        "reliable_frac": float((min_eig[m] > floor).mean()),
        "motion": float(np.median(np.hypot(flow[..., 0], flow[..., 1])[m])),
    }                                    # median flow magnitude = real motion


def flow_score(base_gray, cur_gray, flow=None, mask=None, motion_min=0.3,
               preprocess="lcn_clahe", clip=8.0, tiles=16):
    """A single 0-100 quality score for a dense flow, no ground truth needed.

    There is no canonical no-GT flow score, so this is a transparent composite
    of the :func:`flow_quality` metrics, each mapped to 0..1 and blended:

    * ``fb``   -- forward-backward consistency (weight 0.5, the primary signal):
      ``1/(1+(fb/0.5)^2)`` so 0 px -> 1, 0.5 px -> 0.5, 2 px -> 0.06.
    * ``photo`` -- photometric residual ratio (weight 0.3): ``(0.95-r)/0.6``
      clipped, so r=0.35 -> 1, r=0.95 -> 0. **Motion-gated**: when there is
      little motion (``motion < motion_min``) the ratio is uninformative (there
      is nothing to explain), so this term is dropped and the weight
      redistributed -- a still, correct flow is not punished.
    * ``rel``  -- structure-tensor reliable fraction (weight 0.2): ``rel/0.8``.

    Bands: >=85 excellent, 70-85 good, 55-70 fair, <55 poor. Anchored so a clean
    synthetic translation scores ~100. Returns the score, grade, sub-scores and
    the raw metrics -- treat it as a reliability indicator, not a substitute for
    endpoint error against real ground truth.
    """
    q = flow_quality(base_gray, cur_gray, flow=flow, mask=mask,
                     preprocess=preprocess, clip=clip, tiles=tiles)
    # blend the FB median (typical accuracy) and the p90 tail (worst-case
    # reliability -- where preprocessing choices like LCN actually differ).
    s_fb = (0.6 / (1.0 + (q["fb_error"] / 0.4) ** 2)
            + 0.4 / (1.0 + (q["fb_error_p90"] / 1.5) ** 2))
    s_rel = min(1.0, q["reliable_frac"] / 0.8)
    terms = [(0.5, s_fb), (0.2, s_rel)]
    if q["motion"] >= motion_min:
        s_photo = min(1.0, max(0.0, (0.95 - q["photometric"]) / 0.6))
        terms.append((0.3, s_photo))
    else:
        s_photo = None
    wsum = sum(w for w, _ in terms)
    score = 100.0 * sum(w * s for w, s in terms) / wsum
    grade = ("excellent" if score >= 85 else "good" if score >= 70
             else "fair" if score >= 55 else "poor")
    return {
        "score": round(score, 1), "grade": grade,
        "sub": {"fb": round(s_fb, 3),
                "photometric": None if s_photo is None else round(s_photo, 3),
                "reliable": round(s_rel, 3)},
        "metrics": q,
    }


TRACKERS = {
    "lk": LKTracker,
    "farneback": FarnebackTracker,
    "dis": DISTracker,
    "phasecorr": PhaseCorrTracker,
}


# ------------------------------------------------------- spatial shear field --

@dataclasses.dataclass
class ShearField:
    """A grid of per-node displacement vectors -- the marker-flow-style field.

    nodes:  (N, 2) rest positions (x, y) of the virtual grid.
    vecs:   (N, 2) displacement (dx, dy) px of each node vs the rest baseline.
    conf:   (N,)   0..1 confidence (phase-corr response / LK validity); nodes
                   where the indentation wiped out the speckle come back low.
    grid:   (rows, cols) so the field can be reshaped for curl/divergence.
    """
    nodes: np.ndarray
    vecs: np.ndarray
    conf: np.ndarray
    grid: tuple

    def mean(self, min_conf=0.15):
        """Confidence-weighted global shear (dx, dy) over trusted nodes."""
        w = (self.conf >= min_conf).astype(float) * self.conf
        if w.sum() < 1e-6:
            return 0.0, 0.0
        m = (self.vecs * w[:, None]).sum(0) / w.sum()
        return float(m[0]), float(m[1])

    #: The field must be moving at least this fast on average (px) before any
    #: rotation or divergence is reported at all. Measured on this sensor: with
    #: nothing touching the gel the confidence-weighted mean speed sits at
    #: 0.17-0.24 px of flow jitter, while the lightest real press runs
    #: 0.65-1.24 px -- so 0.35 clears the noise by 1.4x and the weakest signal
    #: by 1.9x. Same idea as ``flow_score``'s ``motion_min``.
    MOTION_FLOOR_PX = 0.35

    def curl(self, min_conf=0.15, weighted=True, motion_floor=None):
        """Rotation (torsion) of the field, per grid step, x1 (banner shows x1e3).

        ``dvy/dx - dvx/dy``; positive = counter-clockwise gel twist. Averaged
        over the grid weighted by each node's confidence AND its displacement
        magnitude -- see :meth:`_deriv` for why the unweighted mean was unusable.
        """
        return self._deriv("curl", min_conf, weighted, motion_floor)

    def divergence(self, min_conf=0.15, weighted=True, motion_floor=None):
        """Divergence (bulk push-out) of the field, same weighting as curl.

        ``dvx/dx + dvy/dy``; positive = gel driven radially outward, which is
        what a normal press looks like from directly above.
        """
        return self._deriv("div", min_conf, weighted, motion_floor)

    def _deriv(self, which, min_conf, weighted=True, motion_floor=None):
        """Weighted spatial mean of curl / divergence over the trusted grid.

        The plain mean this used to take made ``rot`` twitch on a gel nobody
        was touching, for three compounding reasons, all addressed here:

        * **Untrusted nodes were pasted with the field mean.** That put a step
          discontinuity around the contact hole, and ``np.gradient`` across a
          step returns a large derivative -- manufacturing rotation exactly at
          the rim, where the gel is most occluded. They are NaN now, and any
          derivative whose stencil touches one is dropped instead.
        * **Every node voted equally.** A node jittering by 0.05 px counted as
          much as one carrying 3 px of real shear, so a real twist confined to
          the contact patch was averaged away by the still majority around it.
          Nodes are now weighted by ``conf * |v|``, so the number describes the
          part of the gel that is actually moving.
        * **Nothing checked whether the gel was moving at all.** A normalised
          average returns whatever ratio the jitter forms however small the
          vectors are -- and magnitude weighting on its own makes that WORSE,
          because with no contact the loudest nodes are simply the noisiest.
          So the reading is gated on the field's own mean speed
          (``motion_floor``): below it, there is no deformation to describe and
          the answer is exactly 0.0.

        Units are per grid step (nodes are ``spacing`` px apart), unchanged
        from before so the x1e3 banner scale still means the same thing.
        """
        rows, cols = self.grid
        floor = self.MOTION_FLOOR_PX if motion_floor is None else motion_floor
        vx = self.vecs[:, 0].reshape(rows, cols).astype(float)
        vy = self.vecs[:, 1].reshape(rows, cols).astype(float)
        conf = self.conf.reshape(rows, cols).astype(float)
        good = conf >= min_conf
        mag = np.hypot(vx, vy)

        vx = np.where(good, vx, np.nan)
        vy = np.where(good, vy, np.nan)
        gyx, gxx = np.gradient(vx)
        gyy, gxy = np.gradient(vy)
        field = (gxy - gyx) if which == "curl" else (gxx + gyy)
        valid = np.isfinite(field)
        if not valid.any():
            return 0.0
        if not weighted:
            return float(np.mean(field[valid]))

        w_speed = conf * good
        if w_speed.sum() > 1e-9 and (mag * w_speed).sum() / w_speed.sum() < floor:
            return 0.0                      # gel at rest: nothing to report
        w = np.where(valid & good, conf * mag, 0.0)
        if w.sum() < 1e-9:
            return 0.0
        return float((np.where(valid, field, 0.0) * w).sum() / w.sum())

    # -- neighbour coherence (GelSight-style spring regularisation) -----------

    def neighbour_dev(self, min_conf=0.12, connectivity=8):
        """Per-node deviation (px) of its flow from the confident-neighbour median.

        Mirrors GelSight's K2 spring term: a marker should move like its
        lattice neighbours. Returns (dev, (mnx, mny)) where dev is NaN for nodes
        with no confident neighbour.
        """
        rows, cols = self.grid
        vx = self.vecs[:, 0].reshape(rows, cols).astype(float)
        vy = self.vecs[:, 1].reshape(rows, cols).astype(float)
        good = (self.conf >= min_conf).reshape(rows, cols)
        gx = np.where(good, vx, np.nan)
        gy = np.where(good, vy, np.nan)
        offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 8:
            offs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        def shift(a, dy, dx):
            p = np.full((rows + 2, cols + 2), np.nan)
            p[1:-1, 1:-1] = a
            return p[1 + dy:1 + dy + rows, 1 + dx:1 + dx + cols]

        nx = np.stack([shift(gx, dy, dx) for dy, dx in offs])
        ny = np.stack([shift(gy, dy, dx) for dy, dx in offs])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slices
            mnx = np.nanmedian(nx, axis=0)
            mny = np.nanmedian(ny, axis=0)
        dev = np.hypot(vx - mnx, vy - mny)
        return dev.ravel(), (mnx.ravel(), mny.ravel())

    def filter_coherence(self, min_conf=0.12, max_dev=1.5, connectivity=8):
        """Reject nodes whose flow disagrees with neighbours (spring outliers).

        Returns a copy with the confidence of incoherent confident nodes zeroed
        -- the field's own clean-up, more principled than a blind median blur.
        """
        dev, _ = self.neighbour_dev(min_conf, connectivity)
        conf = self.conf.copy()
        bad = (self.conf >= min_conf) & np.isfinite(dev) & (dev > max_dev)
        conf[bad] = 0.0
        return ShearField(self.nodes, self.vecs.copy(), conf, self.grid)

    # -- rigid fit & slip -----------------------------------------------------

    #: Residual (px) within which a node counts as consistent with a candidate
    #: body motion in the consensus fit: the no-contact residual jitter is
    #: 0.17-0.24 px, so 0.35 clears it while still separating a patch that
    #: slid by half a pixel.
    CONSENSUS_INLIER_PX = 0.35

    def residual_field(self, min_conf=0.12, robust=True, samples=300, seed=0,
                       population=None):
        """Per-node flow minus the best-fit AFFINE (translation+rotation+stretch).

        A press deforms the gel smoothly (bulk shift, rotation, radial stretch)
        -- all affine, and none of it is slip. What remains after removing the
        body's affine map is the local NON-smooth motion: a patch that grips vs
        a patch that slides. Returns (res_x, res_y, good).

        The body is found by CONSENSUS (RANSAC over 3-node samples, inliers
        within CONSENSUS_INLIER_PX, then least squares on the inlier set), not
        by a plain least-squares fit over every node. Measured on a synthetic
        press where a rim patch on one side slid 2 px relative to the body:
        the plain fit explained the patch as a stretch term (dvx/dx) and left
        ~0.5 px of residual uniformly on BOTH populations, so no node stood out
        -- and an IRLS/Huber polish could not escape that basin either, since
        it starts from the contaminated solution. Maximising the number of
        nodes explained to within noise picks the body (the coherent majority)
        and leaves the patch as the residual.

        ``population`` (bool per node) restricts the nodes the body is fitted
        on; residuals are still returned for every node. Slip detection fits
        on the contact patch alone: the elastic field AROUND an imprint decays
        with distance (Boussinesq), which no single affine map describes, so a
        whole-frame fit takes the still far gel as the body and reports the
        entire deforming neighbourhood as slip (measured on a plain normal
        press of a coin: 73-100% of the rim nodes flagged, none of it slip).
        """
        good = self.conf >= min_conf
        if population is not None:
            good = good & np.asarray(population, bool)
        if good.sum() < 6:
            z = np.zeros(len(self.nodes))
            return z, z, good
        p = self.nodes[good]
        c = self.nodes.mean(0)
        px, py = (p - c).T
        A = np.column_stack([px, py, np.ones(len(p))])       # affine basis
        vx, vy = self.vecs[good, 0], self.vecs[good, 1]

        def fit(rows):
            cx = np.linalg.lstsq(A[rows], vx[rows], rcond=None)[0]
            cy = np.linalg.lstsq(A[rows], vy[rows], rcond=None)[0]
            return cx, cy

        cx, cy = fit(slice(None))
        if robust and len(p) >= 12:
            rng = np.random.default_rng(seed)
            best = -1
            for _ in range(samples):
                idx = rng.choice(len(p), 3, replace=False)
                if abs(np.linalg.det(A[idx])) < 1e-6:        # collinear
                    continue
                sx, sy = fit(idx)
                r = np.hypot(vx - A @ sx, vy - A @ sy)
                inliers = r <= self.CONSENSUS_INLIER_PX
                n_in = int(inliers.sum())
                if n_in > best:
                    best, best_in = n_in, inliers
            if best >= 6:
                cx, cy = fit(best_in)
                # one re-selection on the refined model tightens the set
                r = np.hypot(vx - A @ cx, vy - A @ cy)
                inliers = r <= self.CONSENSUS_INLIER_PX
                if inliers.sum() >= 6:
                    cx, cy = fit(inliers)
        allpx, allpy = (self.nodes - c).T
        allA = np.column_stack([allpx, allpy, np.ones(len(self.nodes))])
        res_x = np.zeros(len(self.nodes))
        res_y = np.zeros(len(self.nodes))
        res_x[good] = self.vecs[good, 0] - (allA @ cx)[good]
        res_y[good] = self.vecs[good, 1] - (allA @ cy)[good]
        return res_x, res_y, good

    def slip_index(self, min_conf=0.12):
        """RMS non-affine residual (px): 0 = smooth deform, higher = local slip.

        A single scalar flagging incipient/partial slip, like GelSight's
        slip_index. Because the smooth affine part (shift, rotation, stretch of
        a normal press) is removed first, a clean grip reads ~0 and only a
        genuinely slipping/discontinuous contact drives it up.
        """
        rx, ry, good = self.residual_field(min_conf)
        if good.sum() < 6:
            return 0.0
        return float(np.sqrt(np.mean(rx[good] ** 2 + ry[good] ** 2)))

    # -- slip area: where the contact slides vs where it grips ----------------

    #: Non-affine residual (px) a node must exceed to count as slipping. The
    #: no-contact residual jitter on this sensor is 0.17-0.24 px (see
    #: MOTION_FLOOR_PX); a grid node must move 2x that away from the smooth
    #: bulk deformation before it is called slip, not noise.
    SLIP_RESIDUAL_PX = 0.5

    def _node_contact(self, contact, dilate_cells=1.0):
        """Bool per node: inside the (pixel) contact mask, grown by cells.

        The speckle is wiped out INSIDE the imprint, so the nodes that carry
        the slip signal sit on its rim; growing the mask by about one grid
        step keeps that ring in the population being judged.
        """
        if contact is None:
            return np.ones(len(self.nodes), bool)
        import cv2

        rows, cols = self.grid
        step = float(np.median(np.diff(np.unique(self.nodes[:, 0])))) \
            if cols > 1 else 1.0
        k = int(max(1, round(dilate_cells * step)))
        grown = cv2.dilate(contact.astype(np.uint8),
                           np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        ni = np.clip(self.nodes.astype(int), 0, None)
        ni[:, 0] = np.minimum(ni[:, 0], grown.shape[1] - 1)
        ni[:, 1] = np.minimum(ni[:, 1], grown.shape[0] - 1)
        return grown[ni[:, 1], ni[:, 0]] > 0

    def slip_mask(self, min_conf=0.12, thresh=None, contact=None):
        """Per-node slip flags from the non-affine residual. -> (mask, res_mag)

        Cattaneo-Mindlin partial slip: the smooth (affine) part of the field is
        the gel deforming as one body -- grip. What is left is a patch moving
        relative to that body -- the sliding region. A node is flagged when its
        residual magnitude exceeds ``thresh`` (default SLIP_RESIDUAL_PX); with a
        pixel ``contact`` mask only nodes on / next to the imprint are eligible,
        so texture noise far from the object cannot register as slip.
        """
        eligible_pop = self._node_contact(contact)
        rx, ry, good = self.residual_field(min_conf, population=eligible_pop)
        mag = np.hypot(rx, ry)
        eligible = good & eligible_pop
        if thresh is None:
            # Adaptive: the gripping majority sets the noise scale (robust
            # MAD), never below the sensor's own residual floor.
            r = mag[eligible]
            scale = 1.4826 * float(np.median(r)) if r.size else 0.0
            thresh = max(self.SLIP_RESIDUAL_PX, 2.5 * scale)
        return (eligible & (mag > float(thresh))), mag

    def slip_area(self, min_conf=0.12, thresh=None, contact=None):
        """Slipping fraction of the contact and its Cattaneo annularity.

        Returns a dict:
          n_contact    trusted nodes in / around the imprint
          n_slip       of those, flagged as slipping
          slip_frac    n_slip / n_contact (0 = full grip, 1 = gross slip)
          stick_ratio  1 - slip_frac -- the HCI/robotics "stick ratio" whose
                       collapse toward 0 is the incipient-slip predictor
          annularity   mean normalised radius (0 centre .. 1 rim) of the
                       slipping nodes within the contact population. Partial
                       slip physically starts at the RIM and grows inward, so
                       a value well above 0.5 says the flagged nodes form the
                       expected annulus rather than scattered noise; NaN when
                       nothing slips.
        """
        mask, _mag = self.slip_mask(min_conf, thresh, contact)
        good = (self.conf >= min_conf) & self._node_contact(contact)
        n_contact = int(good.sum())
        n_slip = int(mask.sum())
        frac = n_slip / n_contact if n_contact else 0.0
        annularity = float("nan")
        if n_slip and n_contact >= 3:
            pts = self.nodes[good]
            centre = pts.mean(0)
            r = np.hypot(*(self.nodes - centre).T)
            r_max = max(float(r[good].max()), 1e-6)
            annularity = float((r[mask] / r_max).mean())
        return {"n_contact": n_contact, "n_slip": n_slip,
                "slip_frac": float(frac), "stick_ratio": float(1.0 - frac),
                "annularity": annularity}

    # -- friction cone: tangential / normal per node ----------------------------

    def traction_ratio(self, pressure, min_conf=0.12, win=None, floor=None):
        """Per-node |shear| / pressure -- the local friction-cone loading.

        ``pressure`` is a per-pixel map of the normal loading: reconstructed
        depth (mm) when a darkness->depth LUT exists for the unit, otherwise
        the darkening image itself (baseline - frame, >= 0), which is a
        monotonic proxy. It is averaged over a ``win`` x ``win`` box around
        each node. Nodes where the pressure is below ``floor`` (default: 2% of
        the map's 99th percentile) have no normal load to divide by and come
        back NaN -- there the ratio is undefined, not infinite. Coulomb slip
        occurs where the ratio exceeds the friction coefficient; the ratio's
        units are px per pressure-unit, so ``mu`` must be calibrated on this
        sensor from a known gross-slip event.
        """
        import cv2

        p = np.asarray(pressure, dtype=np.float32)
        rows, cols = self.grid
        step = float(np.median(np.diff(np.unique(self.nodes[:, 0])))) \
            if cols > 1 else 41.0
        k = int(win if win is not None else step)
        k = max(3, k | 1)
        local = cv2.blur(p, (k, k))
        ni = self.nodes.astype(int)
        ni[:, 0] = np.clip(ni[:, 0], 0, p.shape[1] - 1)
        ni[:, 1] = np.clip(ni[:, 1], 0, p.shape[0] - 1)
        pn = local[ni[:, 1], ni[:, 0]]
        if floor is None:
            floor = 0.02 * float(np.percentile(p, 99)) if p.size else 0.0
        mag = np.hypot(self.vecs[:, 0], self.vecs[:, 1])
        ratio = np.full(len(self.nodes), np.nan)
        ok = (self.conf >= min_conf) & (pn > max(floor, 1e-9))
        ratio[ok] = mag[ok] / pn[ok]
        return ratio

    def friction_cone_mask(self, pressure, mu, min_conf=0.12, win=None,
                           floor=None):
        """Nodes whose traction ratio exceeds ``mu``: predicted Coulomb slip."""
        ratio = self.traction_ratio(pressure, min_conf, win, floor)
        return np.isfinite(ratio) & (ratio > mu), ratio

    # -- fill the contact hole ------------------------------------------------

    def fill(self, min_conf=0.12, iters=150):
        """Interpolate flow across the ENCLOSED contact hole from its ring.

        GelSight has markers inside the contact; our speckle is wiped out there,
        leaving a hole. Gel shear is smooth, so we diffuse the surrounding
        confident vectors inward -- but only into nodes fully enclosed by
        confident neighbours (the contact blob), never the un-textured border,
        which would be invention. Returns (filled_field, filled_mask).
        """
        rows, cols = self.grid
        known = (self.conf >= min_conf).reshape(rows, cols)
        if known.sum() < 4:
            return ShearField(self.nodes, self.vecs.copy(), self.conf.copy(),
                              self.grid), np.zeros(len(self.nodes), bool)

        # enclosed = unknown with a known node on both sides along row AND column
        left = np.pad(np.maximum.accumulate(known, 1), ((0, 0), (1, 0)),
                      constant_values=False)[:, :-1]
        right = np.pad(np.maximum.accumulate(known[:, ::-1], 1)[:, ::-1],
                       ((0, 0), (0, 1)), constant_values=False)[:, 1:]
        up = np.pad(np.maximum.accumulate(known, 0), ((1, 0), (0, 0)),
                    constant_values=False)[:-1, :]
        down = np.pad(np.maximum.accumulate(known[::-1], 0)[::-1],
                      ((0, 1), (0, 0)), constant_values=False)[1:, :]
        enclosed = (~known) & left & right & up & down

        vx = self.vecs[:, 0].reshape(rows, cols).astype(float).copy()
        vy = self.vecs[:, 1].reshape(rows, cols).astype(float).copy()
        mx, my = self.mean(min_conf)
        for comp, seed in ((vx, mx), (vy, my)):
            comp[enclosed] = seed
            for _ in range(iters):
                nb = np.zeros_like(comp)
                nb[1:] += comp[:-1]; nb[:-1] += comp[1:]
                nb[:, 1:] += comp[:, :-1]; nb[:, :-1] += comp[:, 1:]
                cnt = np.zeros_like(comp)
                cnt[1:] += 1; cnt[:-1] += 1; cnt[:, 1:] += 1; cnt[:, :-1] += 1
                upd = nb / np.maximum(cnt, 1)
                comp[enclosed] = upd[enclosed]

        vecs = np.stack([vx.ravel(), vy.ravel()], 1).astype(np.float32)
        conf = self.conf.copy()
        fill_mask = enclosed.ravel()
        conf[fill_mask] = np.maximum(conf[fill_mask], min_conf)  # drawn, flagged
        return ShearField(self.nodes, vecs, conf, self.grid), fill_mask


# -- temporal slip: the imprint moving relative to the gel --------------------

#: Frame-to-frame relative speed (px/frame) between a point of the imprint and
#: the surrounding gel above which that point is sliding. Measured on this
#: sensor: frames with the object at rest read 0.04-0.2 px of relative jitter,
#: frames where it slid read 1.0-2.8 px.
SLIP_VELOCITY_PX = 0.5


def relative_slip(prev_gray, cur_gray, contact, spacing=26, win=41,
                  fov=None, min_conf=0.10, thresh=None, ring_px=45,
                  method="flow"):
    """Slip as MOTION of the imprint relative to the gel -- per node, per frame.

    Why temporal, not spatial: on a marker-less gel the interior of the
    contact carries the OBJECT's own texture (coin relief, fingerprint
    ridges), so a frame-to-frame flow measured there is the object's
    velocity, while the speckle just outside the imprint gives the gel's.
    Gripping means the two agree -- the object and the gel it holds move as
    one; a point of the imprint moving relative to the neighbouring gel is,
    by definition, sliding over it (Dong et al. 2019). This needs no model
    of how the gel deforms, which is what defeats the spatial (non-affine
    residual) view: a soft fingertip deforms non-affinely without slipping.

    Args:
      prev_gray, cur_gray: consecutive frames.
      contact: bool pixel mask of the imprint (shear.contact_mask).
      ring_px: the gel reference is the median velocity of confident nodes
        within this distance outside the imprint.
      thresh: relative speed (px/frame) flagging slip; default
        SLIP_VELOCITY_PX.

    Returns a dict:
      field        the frame-to-frame ShearField (object texture tracked too)
      inside       bool per node: node lies on the imprint
      rel          per-node |v_node - v_gel| (px/frame), NaN off the imprint
      slip         bool per node: inside, confident, rel > thresh
      v_obj        median velocity of the confident imprint nodes (dx, dy)
      v_gel        median velocity of the ring nodes (dx, dy)
      n_contact    confident imprint nodes
      n_slip       of those, sliding
      slip_frac / stick_ratio
      slip_area_px  approx sliding area (cells x spacing^2), clipped to the
                    imprint's pixel area
    """
    import cv2

    thresh = SLIP_VELOCITY_PX if thresh is None else float(thresh)
    # smooth=False: the 3x3 grid median would blend imprint nodes with the
    # gel across the contact boundary -- exactly the contrast this measures.
    field = shear_field(prev_gray, cur_gray, spacing=spacing, win=win,
                        method=method, fov=fov, smooth=False)
    contact = np.asarray(contact, bool)
    ni = field.nodes.astype(int)
    ni[:, 0] = np.clip(ni[:, 0], 0, contact.shape[1] - 1)
    ni[:, 1] = np.clip(ni[:, 1], 0, contact.shape[0] - 1)
    inside = contact[ni[:, 1], ni[:, 0]]
    dist = cv2.distanceTransform((~contact).astype(np.uint8), cv2.DIST_L2, 3)
    ring = (~inside) & (dist[ni[:, 1], ni[:, 0]] < ring_px)
    good = field.conf >= min_conf

    nan2 = np.array([np.nan, np.nan])
    v_gel = (np.median(field.vecs[ring & good], axis=0)
             if (ring & good).sum() >= 3 else nan2)
    # The object's velocity from nodes whose window lies wholly on the
    # imprint: an edge node's window is half gel and reads a blend.
    k = max(1, int(win // 2))
    core = cv2.erode(contact.astype(np.uint8),
                     np.ones((2 * k + 1, 2 * k + 1), np.uint8))[ni[:, 1], ni[:, 0]] > 0
    core_good = core & good if (core & good).any() else inside & good
    v_obj = (np.median(field.vecs[core_good], axis=0)
             if core_good.sum() >= 1 else nan2)
    rel = np.full(len(field.nodes), np.nan)
    if np.isfinite(v_gel).all():
        d = field.vecs - v_gel
        rel[inside] = np.hypot(d[inside, 0], d[inside, 1])
    slip = inside & good & np.isfinite(rel) & (rel > thresh)
    n_contact = int((inside & good).sum())
    n_slip = int(slip.sum())
    frac = n_slip / n_contact if n_contact else 0.0
    area = min(float(n_slip * spacing * spacing), float(contact.sum()))
    return {"field": field, "inside": inside, "rel": rel, "slip": slip,
            "v_obj": v_obj, "v_gel": v_gel, "n_contact": n_contact,
            "n_slip": n_slip, "slip_frac": float(frac),
            "stick_ratio": float(1.0 - frac), "slip_area_px": area}


def _grid_median(vecs, grid, k=1):
    """3x3 (k=1) spatial median of a vector field on the node grid.

    Independent per-node estimates are noisy on the faint speckle; neighbours
    should agree, so a small spatial median removes lone outliers and gives the
    clean marker-flow look without blurring genuine gradients much.
    """
    rows, cols = grid
    vx = vecs[:, 0].reshape(rows, cols)
    vy = vecs[:, 1].reshape(rows, cols)
    out = np.empty_like(vecs)
    for c, comp in enumerate((vx, vy)):
        pad = np.pad(comp, k, mode="edge")
        stack = [pad[i:i + rows, j:j + cols]
                 for i in range(2 * k + 1) for j in range(2 * k + 1)]
        out[:, c] = np.median(np.stack(stack, axis=0), axis=0).ravel()
    return out


def _texture_conf(base_gray, nodes, win):
    """Per-node baseline speckle energy, 0..1 -- weak texture => low trust."""
    import cv2

    hp = highpass(base_gray).astype(np.float32)
    energy = cv2.GaussianBlur(hp * hp, (0, 0), max(3.0, win / 3.0))
    ni = nodes.astype(int)
    e = energy[ni[:, 1], ni[:, 0]]
    ref = np.percentile(e, 85) if e.size else 1.0
    return np.clip(e / max(ref, 1e-6), 0, 1).astype(np.float32)


def shear_field(base_gray, cur_gray, spacing=28, win=41, method="flow",
                fov=None, smooth=True, max_px=8.0, flow_downscale=1,
                preprocess="lcn_clahe"):
    """Per-node displacement field of ``cur`` vs the rest ``base``.

    A regular ``spacing``-px grid of virtual markers over the FOV; each node's
    local shift is measured in a ``win`` x ``win`` window. This is the
    marker-flow field (Yuan et al.) reproduced on the GlowTact's natural
    speckle -- no printed dots.

    Because that speckle is far fainter than printed markers, three cleanups
    make the field usable: per-node **texture confidence** (weak-speckle nodes
    fade), a spatial **median smooth** across the grid (``smooth``), and a
    **magnitude clamp** (``max_px`` -- real shear is a few px, so larger
    estimates are noise and are dropped to zero confidence).

    ``method``:
      * ``flow`` (default) -- sample dense Farneback flow at the nodes. Its
        built-in spatial regularisation gives by far the cleanest FIELD on the
        faint speckle (independent per-node estimates are too noisy).
      * ``phasecorr`` -- local phase correlation on a high-pass window; this is
        the best GLOBAL scalar estimator, but sparse as a field.
      * ``lk``   -- Lucas-Kanade of the grid points from base to current.

    ``flow_downscale`` (>1, ``flow`` method only) computes the dense flow at
    1/scale resolution and upsamples -- roughly ``scale**2`` faster (the flow is
    the pipeline's bottleneck). It costs accuracy: at scale 2 the per-node
    vectors shift ~0.5 px vs full-res, which is a real fraction of a 1-3 px
    shear signal, so keep it at 1 for measurement and use 2 only for fast
    preview / live display.

    Returns a :class:`ShearField`.
    """
    import cv2

    h, w = base_gray.shape
    if fov is None:
        fov = fov_mask(base_gray.shape)
    half = win // 2
    xs = np.arange(half, w - half, spacing)
    ys = np.arange(half, h - half, spacing)
    gx, gy = np.meshgrid(xs, ys)
    nodes = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    grid = (len(ys), len(xs))

    if method == "phasecorr":
        be = highpass(base_gray).astype(np.float32)
        ce = highpass(cur_gray).astype(np.float32)
        hann = cv2.createHanningWindow((win, win), cv2.CV_32F)
        vecs = np.zeros((len(nodes), 2), np.float32)
        conf = np.zeros(len(nodes), np.float32)
        for i, (x, y) in enumerate(nodes.astype(int)):
            if not fov[y, x]:
                continue
            a = be[y - half:y + half + 1, x - half:x + half + 1]
            b = ce[y - half:y + half + 1, x - half:x + half + 1]
            if a.shape != (win, win):
                continue
            (dx, dy), resp = cv2.phaseCorrelate(a, b, hann)
            vecs[i] = (dx, dy)
            conf[i] = resp
    elif method == "lk":
        be, ce = enhance(base_gray), enhance(cur_gray)
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            be, ce, nodes.reshape(-1, 1, 2), None,
            winSize=(win, win), maxLevel=3)
        p1 = p1.reshape(-1, 2)
        vecs = (p1 - nodes).astype(np.float32)
        st = st.ravel().astype(bool)
        conf = (st & (err.ravel() < 20)).astype(np.float32)
    elif method == "flow":
        be = flow_enhance(base_gray, preprocess)
        ce = flow_enhance(cur_gray, preprocess)
        if flow_downscale > 1:
            s = int(flow_downscale)
            b2 = cv2.resize(be, (w // s, h // s))
            c2 = cv2.resize(ce, (w // s, h // s))
            fl = cv2.calcOpticalFlowFarneback(b2, c2, None, 0.5, 3, 25, 3, 5,
                                              1.2, 0)
            flow = cv2.resize(fl, (w, h)) * s
        else:
            flow = cv2.calcOpticalFlowFarneback(be, ce, None, 0.5, 3, 25, 3, 5,
                                                1.2, 0)
        ni = nodes.astype(int)
        vecs = flow[ni[:, 1], ni[:, 0]].astype(np.float32)
        conf = np.ones(len(nodes), np.float32)
    else:
        raise ValueError(f"unknown method {method!r}")

    # weak-speckle nodes are untrustworthy regardless of the raw response
    conf = conf * _texture_conf(base_gray, nodes, win)
    conf = conf * fov[nodes[:, 1].astype(int), nodes[:, 0].astype(int)]
    # physically implausible jumps (> max_px) are tracking failures
    if max_px:
        conf[np.hypot(vecs[:, 0], vecs[:, 1]) > max_px] = 0.0
    if smooth:
        vecs = _grid_median(vecs, grid)
    return ShearField(nodes, vecs, conf, grid)


# --------------------------------------------------------------- evaluation --

def _warp_residual(base_gray, cur_gray, dx, dy, valid):
    """Masked mean |diff| after shifting cur by -(dx,dy) toward base.

    If the shear vector explains the speckle motion, the shifted current frame
    matches the baseline over the valid (non-contact) region, so the residual
    drops. Reported as a fraction of the no-shift residual: <1 means the
    estimate helped, and smaller is better.
    """
    import cv2

    be = enhance(base_gray).astype(np.float32)
    ce = enhance(cur_gray).astype(np.float32)
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    shifted = cv2.warpAffine(ce, M, (ce.shape[1], ce.shape[0]),
                             borderMode=cv2.BORDER_REFLECT)
    # erode valid to avoid warp border artefacts leaking in
    v = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    if v.sum() < 50:
        return 1.0
    base_res = np.abs(ce[v] - be[v]).mean()
    warp_res = np.abs(shifted[v] - be[v]).mean()
    return float(warp_res / max(base_res, 1e-6))


def evaluate_video(path, tracker_names=None, baseline_idx=0, max_frames=None,
                   stride=1):
    """Run each tracker over a video and return per-tracker quality metrics.

    Returns ``{name: {metrics}, "series": {name: {t,dx,dy,contact_area}}}``.

    Metrics (all ground-truth-free):
      * ``warp_residual``   mean masked warp residual ratio (lower better).
      * ``noise_floor``     std of |shear| on no-contact frames, px (lower).
      * ``indent_intrusion``|corr(contact area, |shear|)| (lower better -- a
                            straight press must not masquerade as shear).
      * ``coherence``       mean flow-vector agreement, 0..1 (higher better).
      * ``score``           composite; higher is better.
    """
    import cv2

    names = tracker_names or list(TRACKERS)
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if len(frames) < 3:
        raise ValueError(f"{path}: too few frames")

    base = frames[baseline_idx]
    idxs = list(range(baseline_idx + 1, len(frames), stride))
    fov = fov_mask(base.shape)

    # Warm the enhance cache: every frame is CLAHE'd once, not per tracker.
    global _ENH_CACHE
    _ENH_CACHE = {}
    for f in frames:
        enhance(f)

    contact_area = []
    contacts = []
    for i in idxs:
        cm = contact_mask(frames[i], base)
        contacts.append(cm)
        contact_area.append(float((cm & fov).sum() / max(fov.sum(), 1)))
    contact_area = np.array(contact_area)
    no_contact = contact_area < 0.005

    out = {"series": {}, "contact_area": contact_area.tolist()}
    for name in names:
        tr = TRACKERS[name]()
        try:
            tr.reset(base)
        except Exception as exc:               # tracker unavailable (e.g. DIS)
            out[name] = {"error": str(exc)}
            continue
        dxs, dys, cohs, resids = [], [], [], []
        for j, i in enumerate(idxs):
            r = tr.estimate(frames[i], contacts[j])
            dxs.append(r.dx)
            dys.append(r.dy)
            cohs.append(r.coherence)
            resids.append(_warp_residual(base, frames[i], r.dx, r.dy,
                                         fov & ~contacts[j]))
        dxs, dys = np.array(dxs), np.array(dys)
        mag = np.hypot(dxs, dys)
        cohs, resids = np.array(cohs), np.array(resids)

        noise = float(mag[no_contact].std()) if no_contact.sum() > 2 else \
            float("nan")
        if contact_area.std() > 1e-6 and mag.std() > 1e-6:
            intrusion = abs(float(np.corrcoef(contact_area, mag)[0, 1]))
        else:
            intrusion = 0.0
        metrics = {
            "warp_residual": float(np.nanmean(resids)),
            "noise_floor": noise,
            "indent_intrusion": intrusion,
            "coherence": float(np.nanmean(cohs)),
            "max_shear_px": float(np.nanmax(mag)),
        }
        out[name] = metrics
        out["series"][name] = {"dx": dxs.tolist(), "dy": dys.tolist()}

    _add_scores(out, names)
    _ENH_CACHE = None
    return out


def _add_scores(out, names):
    """Composite score: rank-free normalised blend across the metrics."""
    valid = [n for n in names if "error" not in out.get(n, {})]
    if not valid:
        return

    def col(key):
        return np.array([out[n][key] for n in valid], dtype=float)

    def norm_lower(v):        # smaller -> higher score
        v = np.nan_to_num(v, nan=np.nanmax(v) if np.isfinite(v).any() else 1.0)
        rng = v.max() - v.min()
        return 1.0 - (v - v.min()) / rng if rng > 1e-9 else np.ones_like(v)

    def norm_higher(v):
        v = np.nan_to_num(v, nan=0.0)
        rng = v.max() - v.min()
        return (v - v.min()) / rng if rng > 1e-9 else np.ones_like(v)

    s = (0.35 * norm_lower(col("warp_residual"))
         + 0.25 * norm_lower(col("noise_floor"))
         + 0.25 * norm_lower(col("indent_intrusion"))
         + 0.15 * norm_higher(col("coherence")))
    for n, sc in zip(valid, s):
        out[n]["score"] = float(round(sc, 4))
