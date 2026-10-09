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


def despeckle(darkening, strength=7, template=5, search=21):
    """Remove the gel's speckle noise from a darkening image, keeping detail.

    Measured on the CNC controlled-loading set (13 candidate filters, scored
    on no-contact residual noise, low-force SNR, localisation, Hertz profile
    and screw-thread band energy): linear low-pass is a bad trade -- the
    darkening image's SNR is positive at ALL spatial frequencies (+7 dB even
    at the M1-thread pitch), so a Gaussian removes signal and speckle in equal
    measure (sigma=2: thread detail 23%, speckle 21%). Non-local means
    exploits the speckle's spatial statistics instead and keeps 99% of the
    thread-band detail while dropping speckle-band noise to 16%, raising
    low-force SNR 16.9 -> 21.8 and the noise p99 11.4 -> 8.4.

    ~150 ms per 640x480 frame -- offline/figure/depth pipelines. For live
    loops use cv2.bilateralFilter(d, 9, 25, 5) (5 ms; keeps 65% of detail at
    21% speckle) or leave the image alone.
    """
    import cv2

    d = np.clip(np.asarray(darkening), 0, 255).astype(np.uint8)
    return cv2.fastNlMeansDenoising(d, None, strength, template, search)


# ------------------------------------------- display pipeline (frozen 2026-09-08)
# Steps 1-6 of docs/pipeline_params.md. Signed differences throughout: press
# darkens (s > 0), Philips slots / gaps brighten (s < 0). Display/depth paths
# only -- the shear pipeline consumes the raw speckle and must never see this.

def signed_difference(reference, frame):
    """Step 1: ``reference - frame`` as float32 grey, NOT clipped."""
    import cv2

    def grey(im):
        im = np.asarray(im)
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return im.astype(np.float32)

    return grey(reference) - grey(frame)


def remove_drift(s, sigma=25.0, thresh=10.0, dilate=15):
    """Step 2: subtract press-induced illumination drift (the "plaques").

    The low-frequency field is estimated from NON-imprint pixels only: pixels
    with ``|s| > thresh`` (dilated ``dilate`` px) are filled with the
    background median before the ``sigma`` Gaussian, so the imprint itself
    does not leak into its own background. Re-zeroed by the median.
    """
    import cv2

    s = np.asarray(s, dtype=np.float32)
    m = cv2.dilate((np.abs(s) > thresh).astype(np.uint8),
                   np.ones((dilate, dilate), np.uint8)).astype(bool)
    filled = s.copy()
    if (~m).sum() > 100:
        filled[m] = np.median(s[~m])
    out = s - cv2.GaussianBlur(filled, (0, 0), sigma)
    return out - np.median(out)


def despeckle_signed(s, strength=7, template=5, search=21):
    """Step 3: :func:`despeckle` for a SIGNED difference (NLM on ``s + 128``)."""
    import cv2

    u = np.clip(np.asarray(s, dtype=np.float32) + 128.0, 0, 255).astype(np.uint8)
    out = cv2.fastNlMeansDenoising(u, None, strength, template, search)
    return out.astype(np.float32) - 128.0


def refine_signed(s, deadzone=2.0, min_peak=12.0, min_area=12,
                  valley_frac=0.15, valley_radius=7, valley_width=3.0,
                  gain_sigma=2.0):
    """Step 4: soft-edged gains computed on ``|s|`` and applied to ``s``.

    * deadzone: ``|s| - deadzone`` (NLM leaves a <=2-grey haze that the gamma
      render shows as mottling);
    * component filter: drop connected components whose p98 is below
      ``min_peak`` grey or whose area is below ``min_area`` px;
    * valley suppression: sigmoid of the sigma-1.5 field against
      ``valley_frac`` x its local max (ellipse radius ``valley_radius``);
    * every gain is Gaussian-blurred (``gain_sigma``) so edges stay soft.
    """
    import cv2

    s = np.asarray(s, dtype=np.float32)
    mag = np.clip(np.abs(s) - deadzone, 0, None)
    keep = np.ones_like(mag)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        (mag > 0.5).astype(np.uint8))
    for i in range(1, n):
        sel = lab == i
        if stats[i, cv2.CC_STAT_AREA] < min_area or \
                np.percentile(mag[sel], 98) < min_peak:
            keep[sel] = 0
    gain = cv2.GaussianBlur(keep, (0, 0), gain_sigma)
    base = cv2.GaussianBlur(mag * gain, (0, 0), 1.5)
    k = 2 * valley_radius + 1
    lm = cv2.dilate(base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    g = 1.0 / (1.0 + np.exp(-(base - valley_frac * lm) / valley_width))
    gain = gain * cv2.GaussianBlur(g.astype(np.float32), (0, 0), gain_sigma)
    return np.sign(s) * mag * gain


def sharpen_finish(r, blur_sigma=0.6, blur_ref=20.0, unsharp=0.5,
                   unsharp_sigma=1.0, unsharp_ref=15.0):
    """Step 5: smooth weak regions, sharpen strong ones.

    Magnitude-weighted blur ``blur(r)*(1-w) + r*w`` with
    ``w = clip(|r|/blur_ref, 0, 1)``, then an unsharp mask weighted by
    ``clip(|r|/unsharp_ref, 0, 1)``. Edge sharpness +25~50 % vs a plain
    sigma-0.6 blur at the same mid-tone smoothness (sweep 2026-09-08).
    """
    import cv2

    r = np.asarray(r, dtype=np.float32)
    w = np.clip(np.abs(r) / blur_ref, 0, 1)
    r = cv2.GaussianBlur(r, (0, 0), blur_sigma) * (1 - w) + r * w
    detail = r - cv2.GaussianBlur(r, (0, 0), unsharp_sigma)
    return r + unsharp * detail * np.clip(np.abs(r) / unsharp_ref, 0, 1)


def render_bidirectional(s, gamma=0.6, background=225.0, press_ref=160.0,
                         bright_ref=80.0):
    """Step 6: signed difference -> uint8 grey display image.

    Press (``s > 0``) darkens from the ``background`` grey toward black,
    saturating at ``press_ref``; brightening (``s < 0``) lifts toward white,
    saturating at ``bright_ref``. Both gamma-shaped.
    """
    s = np.asarray(s, dtype=np.float32)
    pos = np.power(np.clip(s, 0, press_ref) / press_ref, gamma)
    neg = np.power(np.clip(-s, 0, bright_ref) / bright_ref, gamma)
    r = background * (1.0 - pos) + (255.0 - background) * neg
    return np.clip(r, 0, 255).astype(np.uint8)


def clean_signed(s, drift=True):
    """Steps 2-5 on a signed difference; returns the cleaned float32 field."""
    if drift:
        s = remove_drift(s)
    return sharpen_finish(refine_signed(despeckle_signed(s)))


def display_pipeline(reference, frame, drift=True):
    """Steps 1-6: reference + current frame -> despeckled display image.

    ~150 ms per 640x480 frame (NLM dominates): an offline/figure tier.
    """
    return render_bidirectional(
        clean_signed(signed_difference(reference, frame), drift=drift))


# ------------------------------------------------ display-side colour transfer

def lab_stats(image, mask=None):
    """Per-channel (mean, std) of a BGR image in LAB, optionally masked."""
    import cv2

    lab = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2LAB).astype(np.float32)
    px = lab[mask] if mask is not None else lab.reshape(-1, 3)
    return [(float(px[:, c].mean()), float(px[:, c].std())) for c in range(3)]


def lab_transfer_affine(source, target_stats, mask=None):
    """Reinhard LAB transfer as a per-channel (gain, bias) affine.

    Fit once (e.g. on a pre-contact frame, or on background pixels only via
    ``mask``) and reuse it on every frame, so the colour does not flicker.
    """
    out = []
    for (ms, ss), (mt, st) in zip(lab_stats(source, mask), target_stats):
        g = st / (ss + 1e-6)
        out.append((g, mt - ms * g))
    return out


def apply_lab_affine(image, affine):
    """Apply a :func:`lab_transfer_affine` result to a BGR image."""
    import cv2

    lab = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2LAB).astype(np.float32)
    for c, (g, b) in enumerate(affine):
        lab[..., c] = lab[..., c] * g + b
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8),
                        cv2.COLOR_LAB2BGR)
