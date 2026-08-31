# Marker-free shear tracking on the GlowTact

The GelSight Mini prints tracking dots to measure shear; the GlowTact has none.
But its gel carries a faint **natural speckle** texture, and when contact shears
the gel tangentially that whole speckle field translates. Tracking the speckle
gives a **marker-free shear estimate** — implemented in
`glowtact.shear` with four algorithms
and a ground-truth-free evaluation.

## The two hard problems

1. **The speckle is low-contrast.** CLAHE lifts it to ~2000 trackable corners
   per frame; a high-pass variant additionally removes the smooth shading.
2. **Object indentation breaks tracking.** A press paints a large dark blob that
   is *normal* deformation, not shear — naive flow locks onto its moving edges
   and reports huge bogus shear. Every tracker therefore builds a **contact
   mask** (`baseline − current > threshold`, morph-cleaned and dilated) plus a
   circular **FOV mask** for the bright aperture rim, and aggregates shear
   **only from the surrounding speckle**. A straight-down press then reads ≈0.

## The four algorithms

| key | method | how shear is aggregated |
|-----|--------|-------------------------|
| `lk` | sparse Lucas–Kanade on speckle corners, anchored to the rest frame | median of surviving corners (those failing/landing in the blob are dropped) |
| `farneback` | dense Farnebäck flow | masked median over FOV ∩ ¬contact |
| `dis` | dense DIS optical flow (faster) | masked median |
| `phasecorr` | global phase correlation over the masked speckle region | single global translation |

All are **rest-anchored** (baseline = first frame), so the output is absolute
shear from the untouched state — the quantity that tracks shear force.

## Automatic evaluation (no ground truth)

There is no force label, so quality is judged by four self-supervised metrics in
`glowtact-shear-eval` (`glowtact.tools.shear_eval`):

- **`warp_residual`** — warp the current frame by the estimated shear and
  compare to baseline over the speckle region, as a fraction of the no-shift
  residual. Lower = the estimate explains the motion. *(Weakly discriminating
  when true shear is small.)*
- **`noise_floor`** — std of |shear| on no-contact frames; should be ≈0. Lower better.
- **`indent_intrusion`** — |correlation(contact area, |shear|)|. A pure press
  must not masquerade as shear. **Lower better** — this is the metric that
  captures the user's core worry.
- **`coherence`** — fraction of flow vectors agreeing in direction. Higher better.

A composite `score` blends them (0.35 warp + 0.25 noise + 0.25 intrusion + 0.15
coherence, each normalised across trackers).

## Results on `session_20260809_022437` (5 clips)

```
                score   warp_resid  noise_floor  intrusion  coherence
  phasecorr     0.788     0.939       0.034        0.490      0.543     ← best
  lk            0.420     0.942       0.032        0.615      0.291
  dis           0.390     0.936       0.056        0.660      0.307
  farneback     0.366     0.944       0.030        0.587      0.298
```

**Phase correlation wins**, decisively on the two metrics that matter here:
lowest **indentation intrusion** (0.49 vs 0.59–0.66) and highest **coherence**
(0.54 vs ~0.30). Its noise floor (0.03 px) ties the best.

Why: `phasecorr` operates on a **high-pass** image (local, so a big dark blob
barely perturbs it) and estimates one **global** translation over the masked
speckle. The CLAHE-based trackers (`lk`, `farneback`, `dis`) are more easily
biased — CLAHE re-equalises the whole frame when a large indentation appears,
shifting the enhanced speckle everywhere and injecting apparent motion. The
contact mask removes features *under* the blob but not this global CLAHE effect.

All four trackers nonetheless **agree** on a temporally-smooth signal that sits
at ≈0 when untouched and rises during contact (see `*_series.png`), and each
recovers a synthetic known shift to within 1 px (`tests/test_glowtact_shear.py`).

### Caveat — these clips are mostly normal presses

The recorded session has little deliberate lateral drag, so true shear is small
(1–3 px) and `warp_residual` can't discriminate much. The ranking is therefore
driven by robustness-to-indentation and coherence. **To validate shear
magnitude**, record a clip that presses *and* drags laterally (and, ideally,
capture the HX711 force so shear can be regressed against measured tangential
force). The evaluation harness will then exercise `warp_residual` meaningfully.

## Usage

```bash
# evaluate every clip in a session -> metrics.json + comparison/series/overlay PNGs
glowtact-shear-eval ~/glowtact_captures/session_YYYYMMDD_HHMMSS

# faster (every other frame), or a subset of trackers / explicit videos
glowtact-shear-eval <session> --stride 2 --trackers phasecorr lk
glowtact-shear-eval --videos a.avi b.avi --out shear_out
```

Library use (the recommended tracker on a live/recorded stream):

```python
import cv2
from glowtact import shear

cap = cv2.VideoCapture("clip.avi")
_, f0 = cap.read()
base = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
tr = shear.PhaseCorrTracker(); tr.reset(base)

while True:
    ok, f = cap.read()
    if not ok: break
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    cm = shear.contact_mask(g, base)          # exclude the indentation
    r = tr.estimate(g, cm)
    print(f"shear = ({r.dx:+.2f}, {r.dy:+.2f}) px   coherence {r.coherence:.2f}")
```

## Spatial shear field (GelSight marker-flow style)

For the arrow-field visualisation (à la Yuan et al.) rather than one global
vector, `shear.shear_field()` lays a virtual marker grid over the speckle and
estimates each node's local displacement:

```bash
glowtact-shear-field <session_dir>   # arrow-field MP4s
```

```python
field = shear.shear_field(base_gray, cur_gray, spacing=26, method="flow")
mx, my = field.mean()          # global shear (confidence-weighted)
rot    = field.curl()          # rotation / torsion sense
div    = field.divergence()    # in/out bulk motion (normal-press signature)
```

Three cleanups make the faint speckle usable as a field: **texture confidence**
(weak-speckle nodes fade), **grid median smoothing** (neighbours agree), and a
**magnitude clamp** (real shear is a few px). Method note: for the *field*,
`flow` (dense Farneback, spatially regularised) is the default and by far the
cleanest; `phasecorr` remains best for the single *global* vector but is too
sparse per-node. The field cleanly separates contact physics — **radial/
divergent** = normal press, **uniform-directional** = shear, **curl** = torsion.

## 180° fisheye correction + centre crop

The GlowTact's 180° lens barrel-distorts the view and frames it in a black rim.
`glowtact.fisheye` flattens it without a calibration grid (the repo's
`undistort_map.yaml` was fit on a *different* unit and lands rotated/off-centre
here): it auto-detects the illuminated gel disc, treats the lens as equidistant
(`r_img = f·θ`), and reprojects the central field to a pinhole, keeping ~100° so
the square output sits fully inside the disc — flattening the distortion and
removing the rim/corners in one remap.

```bash
glowtact-undistort-video <session_dir>          # plain corrected clips
glowtact-shear-field <session_dir> --undistort   # corrected field
glowtact-shear-render<session_dir> --undistort   # corrected global
```

`--keep-fov` (default 100°) sets the crop tightness; when `--undistort` is on the
render scripts use a full-frame FOV since the rim is already gone.

## Next steps to turn shear *tracking* into shear *force*

1. Record press-and-drag clips **with synchronised HX711 force** (use
   `manual_collect.py`, which logs force alongside video).
2. Fit shear-px → tangential-force (linear is usually enough for small shear).
3. Consider a **spatial shear field** (per-cell phase correlation on a grid)
   rather than one global vector, to capture rotation/slip incipience.
4. Switch the sparse/dense trackers to the high-pass front-end to cut their
   indentation intrusion, if you need a dense field rather than a global vector.

See also: [`docs/glowtact_sensor.md`](glowtact_sensor.md) (capturing data),
the library README (package API).
