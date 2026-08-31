"""GlowTact darkness -> LUT -> depth core (9DTact-style). numpy + lazy cv2.

NOTE: this ``build_lut`` maps grey-darkening -> depth. It is a DIFFERENT function
from photometric_stereo.build_lut (colour -> gradient); do not merge them.
"""
import json

import numpy as np

DEFAULT_LIGHTING_THRESHOLD = 8      # grey levels of darkening treated as noise
DEFAULT_KERNELS = (5, 5)            # iterative Gaussian smoothing, as in 9DTact


def build_lut(gray_values, depth_values,
              lighting_threshold=DEFAULT_LIGHTING_THRESHOLD,
              min_samples=50):
    """Global grey-darkening -> depth (mm) table, 9DTact style, hardened.

    Two guards on top of the plain per-grey mean:

    * Grey levels backed by fewer than `min_samples` pixels are treated as
      unobserved. Measured on real data the last few grey values had <10
      samples and a garbage mean -- and a clamp anchored there dragged the
      whole deep end of the table down (1.03 mm -> 0.67 mm cliff).
    * The table is forced non-decreasing (cumulative max). Depth grows with
      darkening -- that is the sensor premise this whole method rests on, and
      it was verified on this gel (6/6 monotonic ball profiles) -- so a dip in
      the fitted table is estimation noise, not physics.

    Below the lighting threshold the table is zero: that darkening is noise.
    """
    gray = np.clip(np.asarray(gray_values).astype(int), 0, 255)
    depth = np.asarray(depth_values, dtype=np.float64)
    sums = np.bincount(gray, weights=depth, minlength=256)
    counts = np.bincount(gray, minlength=256)
    lut = np.zeros(256)
    seen = counts >= max(1, min_samples)
    if not seen.any():
        seen = counts > 0                       # tiny datasets: use everything
    if not seen.any():
        raise ValueError("no calibration samples at all")
    lut[seen] = sums[seen] / counts[seen]
    seen_idx = np.flatnonzero(seen)
    interp_range = np.arange(seen_idx[0], seen_idx[-1] + 1)
    lut[interp_range] = np.interp(interp_range, seen_idx, lut[seen_idx])
    lut[seen_idx[-1]:] = lut[seen_idx[-1]]          # clamp past reliable data
    lut[:lighting_threshold] = 0.0
    lut = np.maximum.accumulate(lut)                # enforce the physics
    return lut, counts


def depth_from_diff(diff, lut, lighting_threshold=DEFAULT_LIGHTING_THRESHOLD,
                    kernels=DEFAULT_KERNELS, median=0, min_component_px=0):
    """Darkening image -> smoothed depth map in mm (9DTact's runtime step).

    `median` (odd, e.g. 3 or 5) runs a median filter on the darkening image
    BEFORE the LUT: hot pixels / dust specks are isolated outliers in the
    diff, and the LUT amplifies them into tall spikes that Gaussian blur only
    spreads out instead of removing. 9DTact itself has no median stage -- it
    relies on two Gaussian passes (kernel_list [7, 7]) and a cleaner optical
    path; 0 keeps that behaviour.

    `min_component_px` zeroes above-threshold connected components smaller
    than this many pixels. Dust specks and sensor blemishes survive a small
    median (they are several pixels wide) but no real contact is ever a
    lone 100-px island, so area is the discriminator that finally kills
    them without touching genuine contact edges.
    """
    import cv2

    idx = np.clip(diff, 0, 255).astype(np.uint8)
    if median >= 3:
        idx = cv2.medianBlur(idx, median)
    if min_component_px > 0:
        mask = (idx >= lighting_threshold).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < min_component_px)
        keep_off = np.isin(labels, small[small > 0])
        idx = np.where(keep_off, 0, idx)
    height = lut[idx] - lut[lighting_threshold]
    height[height < 0] = 0.0
    height = height.astype(np.float32)
    for kernel in kernels:
        height = cv2.GaussianBlur(height, (kernel, kernel), 0)
    return height


def darkening(img_bgr, ref_bgr):
    """ref - img in grayscale, clipped at zero: how much darker than untouched."""
    import cv2

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    ref = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    return np.clip(ref - img, 0, 255)


def colorize_depth(height_map, max_depth_mm):
    """Depth (mm) -> a JET-style colormap image; black = no contact."""
    import cv2

    norm = np.clip(height_map / max(max_depth_mm, 1e-6), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[height_map <= 0] = 0
    return img


def save_calibration(path, lut, counts, stats):
    np.savez(path, pixel_to_depth=lut, samples_per_gray=counts,
             meta=json.dumps(stats))
    return path


def load_calibration(path):
    data = np.load(path, allow_pickle=False)
    stats = json.loads(str(data["meta"]))
    return data["pixel_to_depth"], stats
