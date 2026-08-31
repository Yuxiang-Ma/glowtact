"""Sensor-agnostic image filters: band-pass, energy, crops. Lazy cv2.

Anything whose geometry is specific to one sensor lives beside that sensor
instead -- the GlowTact aperture mask in ``glowtact.mask``; a GelSight's
gsrobotics border crop belongs to that sensor's own package.
"""
import numpy as np


def dog_bandpass(signal, sigma_lo=1.2, sigma_hi=6.0):
    """Difference-of-Gaussians: keep [sigma_lo, sigma_hi]-scale structure.

    sigma_lo blurs away pixel noise; subtracting the sigma_hi blur removes the
    smooth large-scale membrane shift. What survives is the thread-pitch band.
    The sigma_hi Gaussian is a local MEAN, so the strong contact leaks into the
    background estimate -- see clean_diff.local_median_compensate for the robust
    variant.
    """
    import cv2

    lo = cv2.GaussianBlur(signal, (0, 0), sigma_lo)
    hi = cv2.GaussianBlur(signal, (0, 0), sigma_hi)
    return lo - hi


def local_energy(bandpassed, sigma=9.0, gamma=1.5):
    """A 0..1 map that is high on DENSE ridges and low on sparse crease lines.

    Blurred magnitude of the band-pass, normalised and gamma-shaped. Used as a
    multiplicative weight so a compact thread patch survives while lone creases
    (little sustained energy) fade toward zero.
    """
    import cv2

    e = cv2.GaussianBlur(np.abs(bandpassed), (0, 0), sigma)
    peak = float(e.max())
    if peak <= 0:
        return np.zeros_like(e)
    return np.power(e / peak, gamma)


def center_crop(img, frac=0.40):
    """Keep the central `frac` of each dimension (0 < frac <= 1)."""
    if not 0 < frac <= 1:
        raise ValueError("crop fraction must be in (0, 1]")
    h, w = img.shape[:2]
    ch, cw = max(1, int(round(h * frac))), max(1, int(round(w * frac)))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def apply_crop(image, crop):
    """Rotation-then-crop from a crop_gui bounds tuple (bounds, w, h, rotation).

    `crop` is what glowtact.crop.bounds_from_file returns; None passes through.
    """
    if crop is None:
        return image
    from .crop import rotate_image
    bounds, _w, _h, rotation = crop
    x0, y0, x1, y1 = bounds
    return rotate_image(image, rotation)[y0:y1, x0:x1]
