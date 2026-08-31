"""Difference imaging for GlowTact streams -- baseline subtraction, 3 modes.

Byte-for-byte the same convention the parent ProbingPi project writes, so a
live stream and an offline capture produce comparable diff images:

  * ``signed`` -- ``frame - baseline`` centred on mid-grey (128); flat gel is
    grey, both push directions visible. Lossless for |diff| <= 127 at gain 1.
  * ``dark``   -- ``baseline - frame`` clipped at 0; for a sensor whose contact
    DARKENS the image (GlowTact), so contact shows up bright. 9DTact-style.
    This is the mode the 3drecon depth pipelines consume.
  * ``bright`` -- ``frame - baseline`` clipped at 0; contact brightens.

All operate on BGR/RGB uint8 frames and return uint8 the same shape.
"""
import numpy as np

from ..diffio import DIFF_MODES as MODES, render_difference as _render


def render_difference(frame, baseline, mode="dark", gain=1.0):
    """One diff image from a frame and its untouched baseline.

    ``gain`` amplifies the (subtle) contact signal for display; keep it at 1
    when the output feeds a calibrated depth pipeline that expects true units.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    signed = frame.astype(np.int16) - baseline.astype(np.int16)
    return _render(signed, gain, mode)


def signed_peak(frame, baseline):
    """Largest absolute per-channel deviation from baseline (contact strength).

    Handy as a cheap 'is anything touching?' scalar and to detect clipping
    (a peak of 127 in signed gain-1 means the diff saturated)."""
    return int(np.abs(frame.astype(np.int16) - baseline.astype(np.int16)).max())
