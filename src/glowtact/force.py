"""Normal-force estimation from GlowTact darkening images.

The gel darkens where it is indented, monotonically with depth (the premise
the depth LUT rests on), so image-integrated darkening tracks the indentation
volume and, through the gel's stiffness, the normal force. Measured on the
2026-07-23 pad's snapshot pairs (n=18, 0.3-19 N, three press types):

* WITHIN one press type a single feature -- the darkening sum -- is linear
  with R^2 0.90-0.96. A per-condition calibration is genuinely useful.
* ACROSS press types the slope varies 8x (0.5 N/1e6 grey for a compliant
  fingertip to 4.0 for a rigid screw head): the same gel indentation carries
  very different force depending on what pressed. The best cross-object
  model found (log-domain, sum + contact area) leaves a leave-one-out median
  error of ~0.7 N and RMS ~4 N over 0.3-19 N.

So: calibrate per pad AND per object class when you can; treat a pooled
calibration as an order-of-magnitude readout. Every calibration stores its
leave-one-out statistics so the uncertainty travels with the file.

    from glowtact import force
    feats = [force.extract_features(diff) for diff in diffs]      # darkening
    calib = force.fit_calibration(feats, newtons, log=True)
    calib.save("force_calib.npz")
    print(calib.estimate(force.extract_features(new_diff)))       # newtons

``diff`` is a gain-1 darkening image (baseline - frame, clipped at 0) -- use
``diffio.render_difference(signed, 1.0, "dark")`` or pass ``signed=`` to
convert a 128-centred stored diff.
"""
import dataclasses
import json
import os

import numpy as np

#: Grey levels of darkening below which a pixel is illumination noise, not
#: contact (matches the depth pipeline's lighting threshold).
AREA_THRESHOLD = 8

FEATURE_NAMES = ("dark_sum", "contact_area", "depth_sum", "p995", "density")


def darkening_from_signed(stored_bgr_or_gray):
    """128-centred stored diff -> gain-1 darkening (baseline - frame, >= 0)."""
    a = np.asarray(stored_bgr_or_gray)
    if a.ndim == 3:
        import cv2

        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    return np.clip(128 - a.astype(np.int16), 0, 255).astype(np.uint8)


def extract_features(darkening, lut=None):
    """Feature dict for one gain-1 darkening image.

    dark_sum      total darkening / 1e6 (the indentation-volume proxy)
    contact_area  fraction of pixels darker than AREA_THRESHOLD
    depth_sum     LUT-linearised depth sum / 1e3 (mm-ish), 0 without a `lut`
    p995          99.5th percentile darkening (peak pressure proxy)
    density       dark_sum / contact_area (how concentrated the press is)
    """
    dark = np.asarray(darkening)
    if dark.ndim == 3:
        import cv2

        dark = cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY)
    dark = np.clip(dark, 0, 255).astype(np.uint8)
    s = float(dark.sum()) / 1e6
    area = float((dark > AREA_THRESHOLD).mean())
    depth = float(lut[dark].sum()) / 1e3 if lut is not None else 0.0
    return {"dark_sum": s, "contact_area": area, "depth_sum": depth,
            "p995": float(np.percentile(dark, 99.5)),
            "density": s / max(area, 1e-6)}


def _design(features, names):
    rows = [[f[n] for n in names] + [1.0] for f in features]
    return np.asarray(rows, dtype=np.float64)


@dataclasses.dataclass
class NormalForceCalibration:
    """A fitted feature -> newtons map plus its honest error statistics."""
    coef: np.ndarray                  # per feature + intercept
    features: tuple                   # names, in coef order
    log: bool                         # fitted in log-force space
    loo_rms_n: float                  # leave-one-out RMS error, newtons
    loo_median_n: float               # leave-one-out median |error|
    force_range_n: tuple              # (min, max) seen in calibration
    n_samples: int
    source: str = ""

    def estimate(self, feats):
        """Newtons for one feature dict (or a list of them)."""
        single = isinstance(feats, dict)
        A = _design([feats] if single else feats, self.features)
        y = A @ self.coef
        out = np.exp(y) if self.log else y
        out = np.maximum(out, 0.0)
        return float(out[0]) if single else out

    def save(self, path):
        np.savez(path, coef=self.coef,
                 meta=json.dumps({
                     "features": list(self.features), "log": self.log,
                     "loo_rms_n": self.loo_rms_n,
                     "loo_median_n": self.loo_median_n,
                     "force_range_n": list(self.force_range_n),
                     "n_samples": self.n_samples, "source": self.source}))
        return path

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        m = json.loads(str(d["meta"]))
        return cls(coef=d["coef"], features=tuple(m["features"]),
                   log=bool(m["log"]), loo_rms_n=m["loo_rms_n"],
                   loo_median_n=m["loo_median_n"],
                   force_range_n=tuple(m["force_range_n"]),
                   n_samples=m["n_samples"], source=m.get("source", ""))


def fit_calibration(features, newtons, feature_names=("dark_sum",),
                    log=False, source=""):
    """Least-squares fit + leave-one-out error, so the file carries its truth.

    ``log=True`` fits log(F): multiplicative errors, never negative force --
    the better pooled model measured on the 2026-07-23 pairs; plain linear is
    the better within-series model.
    """
    newtons = np.asarray(newtons, dtype=np.float64)
    if log and (newtons <= 0).any():
        raise ValueError("log fit needs strictly positive forces")
    A = _design(features, feature_names)
    if len(newtons) <= A.shape[1]:
        raise ValueError(f"{len(newtons)} samples cannot fit "
                         f"{A.shape[1]} coefficients")
    y = np.log(newtons) if log else newtons
    coef = np.linalg.lstsq(A, y, rcond=None)[0]

    errs = []
    for i in range(len(newtons)):
        m = np.ones(len(newtons), bool)
        m[i] = False
        c = np.linalg.lstsq(A[m], y[m], rcond=None)[0]
        pred = A[i] @ c
        errs.append((np.exp(pred) if log else pred) - newtons[i])
    errs = np.asarray(errs)
    return NormalForceCalibration(
        coef=coef, features=tuple(feature_names), log=log,
        loo_rms_n=float(np.sqrt(np.mean(errs ** 2))),
        loo_median_n=float(np.median(np.abs(errs))),
        force_range_n=(float(newtons.min()), float(newtons.max())),
        n_samples=int(len(newtons)), source=source)


def pairs_from_index(index_csv, resolve=None, min_force_n=0.2,
                     require_area=0.02):
    """(features, newtons, names) from a manual_collect snapshot index.csv.

    Reads each row's gain-1 diff (``diff_mode`` column decides signed vs
    dark decoding). Rows are DROPPED when force and image contradict each
    other -- |F| < `min_force_n` while the darkened area exceeds
    `require_area` means the press missed the load cell (observed in the
    2026-07-27 sessions: 9-20% contact area at 0.00 N), and such a pair
    would poison any fit. ``resolve(path) -> path`` may redirect to a moved
    copy (the tactile_data dedupe/reorganise manifests).
    """
    import csv

    import cv2

    folder = os.path.dirname(os.path.abspath(index_csv))
    feats, forces, names = [], [], []
    for r in csv.DictReader(open(index_csv)):
        if not r.get("force_newtons") or not r.get("diff_file"):
            continue
        path = os.path.join(folder, r["diff_file"])
        if resolve is not None:
            path = resolve(path) or path
        img = cv2.imread(path)
        if img is None:
            continue
        if r.get("diff_mode", "signed") == "signed":
            dark = darkening_from_signed(img)
        else:
            import cv2 as _cv2
            dark = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
        f = extract_features(dark)
        newton = float(r["force_newtons"])
        if abs(newton) < min_force_n and f["contact_area"] > require_area:
            continue                    # pressed beside the load cell
        feats.append(f)
        forces.append(newton)
        names.append(r.get("name", ""))
    return feats, np.asarray(forces), names
