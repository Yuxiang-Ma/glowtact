# Pipeline parameters — despeckling & shear tracking (frozen 2026-09-08)

The most recent, swept-and-verified parameters. Reference implementation for
despeckling: `~/tactile_data/glowtact/analysis/despeckle/process_videos.py`
(functions `remove_drift`, `refine`, `tact`); the same values live in
`references/consistency/make_pipeline_grids.py` (`P` dict) and
`analysis/despeckle/sample_grid.py`. Shear tracking: `glowtact.shear`.

## Despeckling / display pipeline (per frame)

| Step | Parameters | Why (sweep evidence) |
|---|---|---|
| 1. Difference | `s = reference − frame`, **bi-directional** (signed; no clip). Stored `dark`-mode diffs lost the brightening at capture; `signed`-mode files decode as `median(border) − img`. | press darkens, but philips slots / gaps brighten |
| 2. Drift removal | mask-aware low-pass: mask `|s|>10` dilated 15 px, filled with background median, subtract `GaussianBlur(σ=25)`, re-zero by median | removes press-induced illumination plaques around imprints |
| 3. NLM despeckle | `fastNlMeansDenoising(h=7, template=5, search=21)` on `s+128` (u8), shift back | despeckle eval winner: 99 % detail kept, 84 % speckle removed, low-force SNR 16.9→21.8 dB |
| 4. Refinement (soft-edged, gains from `|s|`, applied to `s`) | deadzone 2; component filter: drop components with `p98 < 12` grey **or** area `< 12` px; valley suppression: sigmoid vs `0.15 × local max` (dilate r=7 on σ1.5-smoothed field, w=3); every gain `GaussianBlur(σ=2)` before applying | plaque score →0 at unchanged keep (opt it4, `plaque_opt.json`); soft edges: mid-tone edge energy −25~40 % (no "puddles") |
| 5. Sharpness finish | magnitude-weighted blur: `blur(σ0.6)·(1−w)+r·w`, `w=clip(|r|/20,0,1)`; then unsharp `r += 0.5·(r−blur(σ1))·clip(|r|/15,0,1)` | edge sharpness +25~50 % vs plain σ0.6 blur, smoothness kept |
| 6. Render | `225·(1−(clip(s,0,160)/160)^0.6)` dark for press; brightening adds `(255−225)·(clip(−s,0,80)/80)^0.6` | γ/dref sweep: deep press p5=14 (near-black), 0 % saturation, light-press contrast 63 |

Display-side colour: Reinhard LAB transfer to
`analysis/sensitive_pad/cropped/000011_mms_0_tactile.png` (imprint images:
affine fitted on background pixels only, `|L−median|<18`). Original-panel
sharpness matched down to the set median (Laplacian-variance bisection).

Library (`glowtact.imgfilt`, bit-exact against `process_videos.py`):
`signed_difference` (1), `remove_drift` (2), `despeckle_signed` (3; the
unsigned `despeckle` remains for darkening-only images), `refine_signed` (4),
`sharpen_finish` (5), `render_bidirectional` (6); `display_pipeline(reference,
frame)` chains all six. Colour: `lab_stats` / `lab_transfer_affine` /
`apply_lab_affine`. Note `process_videos.py` runs NLM twice (once on the
positive-clipped diff before `refine`); the library follows the single signed
pass specified above.

## Shear tracking (`glowtact.shear`, unchanged since the 28-round sweep)

| Component | Parameters |
|---|---|
| Flow preprocessing | `flow_enhance(preprocess="lcn_clahe", clip=8.0, tiles=16)`; LCN `σ=15`, CLAHE on top |
| High-pass (speckle features) | `highpass(σ=9)` — **speckle is the signal here: never despeckle the shear path** |
| FOV mask | `fov_mask(radius_frac=0.82)` |
| Contact mask | darkening `thresh=22`, open/close 5×5, `dilate=9` |
| Tracker | `max_corners=1500, quality=0.01, min_dist=6` (Shi-Tomasi + LK) |
| Slip modality | temporal (imprint motion relative to gel), from the 28-iteration sweep — ground truth still pending (CNC tangential rig) |

## Rules of thumb

- Despeckle feeds tactile-image/depth/display paths only; the shear pipeline
  consumes the raw speckle.
- Rectification maps and flat-field/sensitivity maps are **per mount / per
  pad** — refit after any remount.
- Runtime: full despeckle ≈150 ms/frame (offline tier); guided filter
  `r=4, ε=5²` ≈13 ms/frame is the realtime substitute (detail 0.93).
