"""Directly improve tactile difference images: center-crop + speckle denoise.

Two cheap, artifact-safe steps applied to the raw ``*_diff.png`` (unlike
clean_diff.py, which removes the membrane-shift background):

1. Center crop -- the feature sits in the middle; the periphery is vignette and
   rim noise. Keeping the central ``--crop`` fraction of each dimension (0.40 by
   default) zooms in and drops most of the noisy border.

2. Grayscale Non-Local-Means denoise -- speckle is random per pixel, while a
   thread pattern is REPETITIVE, so NLM (which averages self-similar patches)
   suppresses the speckle and reinforces the ridges. Grayscale on purpose: the
   thread signal is luminance, and colour NLM leaves chroma blotches (new
   artifacts). ``--auto-h`` picks the strongest denoise that still keeps the
   thread structure correlated >= ``--min-corr`` with the raw crop, so it can
   never smooth the pattern away.

    glowtact-enhance-diff ~/tactile_data/glowtact/sessions/session_.../snapshots
    glowtact-enhance-diff <folder> --crop 0.4 --h 6 --upscale 2
"""
import argparse
import glob
import os
import sys

import numpy as np

from .imgfilt import center_crop, dog_bandpass  # noqa: F401


def speckle_level(gray):
    """High-frequency noise: std of (image - its 3x3 median)."""
    import cv2

    g = gray.astype(np.float32)
    return float((g - cv2.medianBlur(gray, 3).astype(np.float32)).std())


def band_pass(gray, lo=1.0, hi=6.0):
    # Thin wrapper over the shared DoG; keeps enhance_diff's gentler 1.0/6.0
    # defaults (structure_corr relies on them).
    return dog_bandpass(gray.astype(np.float32), lo, hi)


def structure_corr(a_gray, b_gray):
    """Correlation of the two images' thread-band structure (0..1).

    Measured on the band-pass so it reflects the ridge PATTERN, not the DC
    level. Used to guarantee denoising keeps the threads (corr near 1) rather
    than blurring them into the background (corr falls).
    """
    a, b = band_pass(a_gray).ravel(), band_pass(b_gray).ravel()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def denoise(gray, h=6, template=7, search=21):
    """Grayscale Non-Local-Means. h=0 returns the input unchanged."""
    import cv2

    if h <= 0:
        return gray
    return cv2.fastNlMeansDenoising(gray, None, float(h), template, search)


def auto_h(gray, candidates=(3, 5, 7, 9, 11), min_corr=0.90):
    """Strongest denoise (largest h) that keeps thread structure >= min_corr.

    Enforces the "do not break the thread pattern" requirement directly: try
    heavier denoising, stop before the ridge correlation drops below the floor.
    """
    best_h = 0
    for h in sorted(candidates):
        if structure_corr(gray, denoise(gray, h)) >= min_corr:
            best_h = h
        else:
            break
    return best_h


def enhance(stored_bgr, crop=0.40, h="auto", min_corr=0.90, upscale=1.0):
    """Raw 128-centred diff -> (enhanced 128-centred grayscale diff, info)."""
    import cv2

    cropped = center_crop(stored_bgr, crop)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    chosen = auto_h(gray, min_corr=min_corr) if h == "auto" else int(h)
    out = denoise(gray, chosen)
    info = {"h": chosen,
            "speckle_before": speckle_level(gray),
            "speckle_after": speckle_level(out),
            "structure_corr": structure_corr(gray, out)}
    if upscale and upscale != 1.0:
        out = cv2.resize(out, None, fx=upscale, fy=upscale,
                         interpolation=cv2.INTER_CUBIC)
    return out, info


def amplify(gray_diff, gain=3.0):
    """128-centred grayscale diff -> amplified viewable copy."""
    signed = gray_diff.astype(np.int16) - 128
    return np.clip(signed * gain + 128, 0, 255).astype(np.uint8)


def process_folder(folder, out_dir=None, crop=0.40, h="auto", min_corr=0.90,
                   upscale=1.0, gain=3.0):
    import cv2

    out_dir = out_dir or os.path.join(folder, "enhanced")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in glob.glob(os.path.join(folder, "*_diff.png"))
                   if "_diff_x" not in os.path.basename(n)
                   and "_autoscale" not in os.path.basename(n)
                   and "_clean" not in os.path.basename(n)
                   and "_enh" not in os.path.basename(n))
    if not names:
        raise SystemExit(f"no '*_diff.png' in {folder}")

    rows = []
    for path in names:
        stored = cv2.imread(path)
        if stored is None:
            continue
        enhanced, info = enhance(stored, crop, h, min_corr, upscale)
        stem = os.path.basename(path)[: -len("_diff.png")]
        cv2.imwrite(os.path.join(out_dir, f"{stem}_diff_enh.png"), enhanced)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_diff_enh_x{gain:g}.png"),
                    amplify(enhanced, gain))
        rows.append((stem, info))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder", help="snapshots folder with *_diff.png")
    p.add_argument("--out", default=None, help="output folder (default <folder>/enhanced)")
    p.add_argument("--crop", type=float, default=0.40,
                   help="central fraction of each dimension to keep (default 0.40)")
    p.add_argument("--h", default="auto",
                   help="NLM strength, or 'auto' (strongest that preserves threads)")
    p.add_argument("--min-corr", type=float, default=0.90,
                   help="auto-h thread-structure correlation floor (default 0.90)")
    p.add_argument("--upscale", type=float, default=1.0,
                   help="resize the crop up for a zoomed view (default 1.0 = none)")
    p.add_argument("--gain", type=float, default=3.0,
                   help="amplification for the *_enh_x N viewable copy")
    args = p.parse_args(argv)

    rows = process_folder(args.folder, args.out, args.crop, args.h,
                          args.min_corr, args.upscale, args.gain)
    sb = np.mean([r[1]["speckle_before"] for r in rows])
    sa = np.mean([r[1]["speckle_after"] for r in rows])
    for stem, info in rows:
        print(f"{stem:22s} h{info['h']:<2}  speckle {info['speckle_before']:.2f}"
              f" -> {info['speckle_after']:.2f}  thread-corr {info['structure_corr']:.3f}")
    print(f"mean speckle {sb:.2f} -> {sa:.2f} "
          f"({100 * (sb - sa) / sb:.0f}% less)  -> "
          f"{args.out or os.path.join(args.folder, 'enhanced')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
