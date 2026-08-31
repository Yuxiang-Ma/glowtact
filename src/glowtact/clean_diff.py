"""Clean the global-shift artifacts out of soft-finger difference images.

The GlowTact omnifinger's gel is soft: pressing an object shifts and stretches
the WHOLE membrane, not just the contact. A plain ``frame - baseline`` diff
therefore carries the entire membrane's motion -- creases, texture, a big smooth
warp -- swamping the feature you actually want (e.g. screw threads).

What did NOT work (measured on this data, kept here so it is not re-tried):

* High-pass / large-sigma unsharp: the membrane texture is high-frequency too,
  so subtracting a blur leaves most of it.
* Global homography / dense optical-flow registration of baseline->frame: the
  shift is a large non-rigid warp; registration only partly aligns it and adds
  its own smearing, and it tries to "align away" the rigid object too.

A curved soft finger does NOT shift uniformly, and much of the artifact is
photometric anyway (GlowTact's directional lighting means a background point
that merely TILTS under load changes brightness without moving) -- so no single
global warp, and no geometric registration at all, can remove it. The fix is a
LOCAL, spatially-adaptive background estimate subtracted per-region. Two methods
(``--method``) differ only in how that local background is estimated:

* ``local-median`` (default) -- subtract a large-window MEDIAN of the diff. The
  median is robust to the sharp thread outliers, so the estimated background is
  the smooth membrane field alone; subtracting it leaves the threads with little
  of the halo a mean-based estimate rings with. This is the local compensation.
* ``dog`` -- a Difference-of-Gaussians band-pass (subtract a Gaussian MEAN
  background). Cheaper, but the strong contact leaks into the mean, ringing the
  threads a little.

Both are then sharpened by the same two structure filters:

1. Local-energy weighting -- multiply by a blurred |compensated| map: a dense
   ridge field (the contact) has sustained energy; isolated creases do not, so
   they fade while the threads stay.
2. A soft circular FOV mask suppresses the sensor's bright rim ring.

    glowtact-clean-diff <snapshots-folder>
    glowtact-clean-diff <folder> --method dog --band 1.2,6
    glowtact-clean-diff <folder> --method local-median --bg-kernel 41
"""
import argparse
import glob
import os
import sys

import numpy as np

from .diffio import recover_signed  # noqa: F401
from .imgfilt import dog_bandpass, local_energy  # noqa: F401
from .mask import fov_mask  # noqa: F401


def to_signal(signed, grayscale=True):
    """Reduce a signed BGR diff to the array the filters run on."""
    s = signed.astype(np.float32)
    if grayscale and s.ndim == 3:
        return s.mean(axis=2)
    return s


def local_median_compensate(signal, bg_kernel=41, denoise_sigma=1.0):
    """Subtract a large-window local MEDIAN: robust spatially-adaptive comp.

    A curved soft finger deforms non-uniformly, so the background artifact is a
    LOCAL, spatially-varying field, not one global shift. A median over a window
    wider than the thread pitch estimates that field per-region while ignoring
    the sharp thread/edge outliers (a mean would be dragged by them and ring the
    contact). Subtracting it flattens the membrane shift -- geometric AND the
    photometric tilt-brightening -- and keeps the sub-window features.
    """
    import cv2

    k = int(bg_kernel) | 1                              # medianBlur needs odd k
    offset = np.clip(signal + 128.0, 0, 255).astype(np.uint8)
    background = cv2.medianBlur(offset, k).astype(np.float32) - 128.0
    out = signal - background
    if denoise_sigma:
        out = cv2.GaussianBlur(out, (0, 0), denoise_sigma)
    return out


def compensate(signal, method="local-median", sigma_lo=1.2, sigma_hi=6.0,
               bg_kernel=41):
    """Remove the local background artifact, keeping the sharp features."""
    if method == "local-median":
        return local_median_compensate(signal, bg_kernel, denoise_sigma=sigma_lo)
    if method == "dog":
        return dog_bandpass(signal, sigma_lo, sigma_hi)
    raise ValueError(f"unknown method {method!r} (use 'local-median' or 'dog')")


def clean(signed, method="local-median", sigma_lo=1.2, sigma_hi=6.0,
          bg_kernel=41, mask_sigma=9.0, mask_gamma=1.5, use_mask=True,
          radius_frac=0.95, grayscale=True):
    """signed diff -> cleaned float signal (0-centred), before display scaling."""
    sig = to_signal(signed, grayscale)
    comp = compensate(sig, method, sigma_lo, sigma_hi, bg_kernel)
    out = comp
    if use_mask:
        out = out * local_energy(comp, mask_sigma, mask_gamma)
    if radius_frac:
        out = out * fov_mask(sig.shape, radius_frac)
    return out


def autoscale_gain(cleaned, target=100.0, pct=99.5):
    """Gain that maps the cleaned signal's 99.5th-pct magnitude to `target`."""
    ref = float(np.percentile(np.abs(cleaned), pct))
    return (target / ref) if ref > 1e-6 else 1.0


def render(cleaned, gain):
    """Center the cleaned signal on mid grey for viewing."""
    return np.clip(cleaned * gain + 128.0, 0, 255).astype(np.uint8)


def reference_roi(ref_clean, fov, energy_sigma=9.0, pct=97.0, dilate=15):
    """Localise the feature (thread) region from a reference cleaned image.

    Frozen once per image (from the default clean) so every candidate parameter
    set is scored against the SAME region -- otherwise a metric that also picks
    its own ROI could reward garbage. The largest compact high-energy blob is
    the contact; the smooth membrane artifact never forms such a blob.
    """
    import cv2

    inside = fov > 0.5
    e = cv2.GaussianBlur(np.abs(ref_clean), (0, 0), energy_sigma) * (inside)
    thr = np.percentile(e[inside], pct)
    m = (e > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return m.astype(bool)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    roi = cv2.dilate((lab == big).astype(np.uint8),
                     np.ones((dilate, dilate), np.uint8))
    return roi.astype(bool)


def evaluate(signed, cleaned, roi, fov=None):
    """Artifact scores for one cleaned image against a frozen feature ROI.

    Returns dict:
      bg_ratio -- background residual / feature signal (THE artifact measure;
                  lower is cleaner). Scale-invariant, so display gain is moot.
      detail   -- correlation of the cleaned ROI with the raw band-passed
                  threads (thread fidelity; higher keeps the feature real).
      objective-- bg_ratio + 0.4*(1 - detail); lower is better overall.
    """
    import cv2

    if fov is None:
        fov = fov_mask(cleaned.shape, 0.95)
    raw_band = dog_bandpass(to_signal(signed, True), 1.2, 6.0)
    bg = (fov > 0.5) & ~roi
    feat = float(np.percentile(np.abs(cleaned[roi]), 90)) + 1e-6
    art = float(np.percentile(np.abs(cleaned[bg]), 90))
    a, b = cleaned[roi], raw_band[roi]
    detail = (float(np.corrcoef(a, b)[0, 1])
              if a.std() > 1e-6 and b.std() > 1e-6 else 0.0)
    bg_ratio = art / feat
    return {"bg_ratio": bg_ratio, "detail": detail,
            "objective": bg_ratio + 0.4 * (1 - max(detail, 0.0))}


def process_folder(folder, out_dir=None, gain="auto", **kw):
    import cv2

    out_dir = out_dir or folder
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in glob.glob(os.path.join(folder, "*_diff.png"))
                   if "_diff_x" not in os.path.basename(n)
                   and "_autoscale" not in os.path.basename(n)
                   and "_clean" not in os.path.basename(n))
    if not names:
        raise SystemExit(f"no '*_diff.png' files in {folder}")

    written = []
    for path in names:
        stored = cv2.imread(path)
        if stored is None:
            continue
        cleaned = clean(recover_signed(stored), **kw)
        g = autoscale_gain(cleaned) if gain == "auto" else float(gain)
        stem = os.path.basename(path)[: -len("_diff.png")]
        out = os.path.join(out_dir, f"{stem}_diff_clean.png")
        cv2.imwrite(out, render(cleaned, g))
        written.append((os.path.basename(out), g))
    return written


def _band_arg(value):
    lo, hi = (float(v) for v in value.split(","))
    if not 0 < lo < hi:
        raise argparse.ArgumentTypeError("--band needs 0 < lo < hi, e.g. 1.2,6")
    return lo, hi


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder", help="snapshots folder with *_diff.png (signed mode)")
    p.add_argument("--out", default=None, help="output folder (default: in place)")
    p.add_argument("--method", choices=["local-median", "dog"],
                   default="local-median",
                   help="local background estimate: 'local-median' (robust, "
                        "default) or 'dog' (Gaussian band-pass)")
    p.add_argument("--bg-kernel", type=int, default=41,
                   help="local-median window in px (> thread pitch); default 41")
    p.add_argument("--band", type=_band_arg, default=(1.2, 6.0),
                   help="DoG sigmas 'lo,hi' in px (thread-pitch band); default 1.2,6")
    p.add_argument("--mask-sigma", type=float, default=9.0,
                   help="local-energy blur radius; larger keeps a wider contact")
    p.add_argument("--mask-gamma", type=float, default=1.5,
                   help="energy-mask contrast; higher = harsher crease removal")
    p.add_argument("--no-mask", action="store_true",
                   help="skip the local-energy weighting (band-pass only)")
    p.add_argument("--fov-radius", type=float, default=0.95,
                   help="soft circular FOV mask radius (0 disables the rim cut)")
    p.add_argument("--color", action="store_true",
                   help="process per-channel instead of grayscale")
    p.add_argument("--gain", default="auto",
                   help="display gain, or 'auto' (99.5th-pct autoscale)")
    args = p.parse_args(argv)

    written = process_folder(
        args.folder, args.out, gain=args.gain, method=args.method,
        sigma_lo=args.band[0], sigma_hi=args.band[1], bg_kernel=args.bg_kernel,
        mask_sigma=args.mask_sigma, mask_gamma=args.mask_gamma,
        use_mask=not args.no_mask, radius_frac=args.fov_radius,
        grayscale=not args.color)
    for name, g in written:
        print(f"{name}  (gain x{g:.1f})")
    print(f"cleaned {len(written)} images -> {args.out or args.folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
