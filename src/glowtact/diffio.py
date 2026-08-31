"""Signed-difference codec, canonical globber, and viewable-render helpers.

The repo stores a canonical ``*_diff.png`` at gain 1, 128-centred
(``frame - baseline + 128``), lossless while ``|diff| <= 127``. These helpers
are the one place that convention lives.
"""
import glob
import os

import numpy as np

DEFAULT_DIFF_GAIN = 3.0             # contact signal is subtle; amplify to see it
DIFF_MODES = ("signed", "dark", "bright")

# Files that are DERIVED from a canonical *_diff.png and must never be treated as
# inputs themselves (amplified copies, autoscale, cleaned, grayscale, enhanced).
DERIVED_DIFF_MARKERS = ("_diff_x", "_autoscale", "_clean", "_grey", "_gray",
                        "_enh")


def recover_signed(stored_bgr):
    """Recover the signed difference from a gain-1, 128-centred PNG (BGR)."""
    return stored_bgr.astype(np.int16) - 128


def canonical_diff_paths(folder, pattern="*_diff.png",
                         skip=DERIVED_DIFF_MARKERS):
    """Sorted canonical ``*_diff.png`` paths, excluding derived copies.

    Replaces the ~7 copy-pasted globbers; the skip set is a superset of each
    script's own (they variously dropped `_diff_x`/`_autoscale`/`_clean`).
    """
    return sorted(p for p in glob.glob(os.path.join(folder, pattern))
                  if not any(m in os.path.basename(p) for m in skip))


def load_signed_diff(path, crop=False, scale_255=False, bgr_to_rgb=True):
    """Read a 128-centred ``*_diff.png`` -> float signed difference.

    Unifies the two former loaders:
      * photometric_stereo: ``crop=True, scale_255=False`` -> RGB in +-127.
      * touchnet: ``crop=False, scale_255=True``  -> RGB in [-0.5, 0.5].
    ``crop`` is a callable applied to the float image (e.g. the gsrobotics
    border crop for a GelSight Mini); ``False``/``None`` leaves it whole.
    """
    import cv2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    if bgr_to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = img.astype(np.float32) - 128.0
    if scale_255:
        out = out / 255.0
    if callable(crop):
        out = crop(out)
    elif crop:
        raise TypeError("crop must be a callable (the sensor's crop "
                        "function) or None")
    return out


def signed_difference(frame_bgr, baseline_bgr):
    """Raw signed frame-minus-baseline, in ORIGINAL intensity units (no gain).

    Kept separate from the rendered view so the true magnitude stays available
    for labelling -- once gain is applied and clipped, it cannot be recovered.
    Returns None when there is no usable baseline yet.
    """
    if baseline_bgr is None or baseline_bgr.shape != frame_bgr.shape:
        return None
    return frame_bgr.astype(np.int16) - baseline_bgr.astype(np.int16)


def render_difference(signed_diff, gain=DEFAULT_DIFF_GAIN, mode="signed"):
    """Render a signed difference (frame - baseline) as a viewable image.

    Three representations, because the right one depends on the sensor's optics:

    * ``signed`` -- amplify and centre on mid grey (128). No-contact is grey;
      brighter and darker both show, in opposite directions. Correct for a
      GelSight, whose directional RGB lighting makes a contact push each channel
      either way depending on the surface normal. Wastes half the 8-bit range
      per direction, so it clips once |diff|*gain reaches 127.

    * ``dark`` -- ``baseline - current`` (reference minus current) clipped to
      [0, 255], BLACK background. For a sensor whose contact darkens the image
      (measured: this GlowTact darkens on 86-94% of contact pixels), this is
      both more intuitive -- black = untouched, bright = contact -- and uses the
      FULL 0-255 range for the signal instead of half. This is the 9DTact-style
      representation.

    * ``bright`` -- ``current - baseline`` clipped to [0, 255], black
      background, for a sensor whose contact brightens the image.
    """
    if mode == "dark":
        return np.clip(-signed_diff * gain, 0, 255).astype(np.uint8)
    if mode == "bright":
        return np.clip(signed_diff * gain, 0, 255).astype(np.uint8)
    return np.clip(signed_diff * gain + 128, 0, 255).astype(np.uint8)


def gain_tag(gain):
    """Filename-safe gain tag: 3 -> '3', 1.5 -> '1p5' (the repo convention)."""
    return f"{gain:g}".replace(".", "p")
