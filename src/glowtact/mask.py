"""GlowTact frame geometry: the circular gel aperture."""
import numpy as np


def fov_mask(shape, radius_frac=0.95, softness=0.06):
    """Soft circular mask (1 inside, ->0 past the FOV) to kill the rim ring.

    The GlowTact images the gel through a circular aperture whose bright border
    is a strong edge the band-pass would otherwise light up. A cosine-feathered
    disc centred on the frame removes it without a hard cutoff. radius_frac is
    in units of the half-min-dimension; 1.0 touches the nearer edges.
    """
    h, w = shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r0 = radius_frac * min(h, w) / 2.0
    ys, xs = np.mgrid[0:h, 0:w]
    r = np.hypot(xs - cx, ys - cy)
    feather = max(softness * min(h, w) / 2.0, 1e-6)
    m = 0.5 * (1.0 - np.cos(np.pi * np.clip((r0 + feather - r) / feather, 0, 1)))
    return m.astype(np.float32)
