# GlowTact evaluation roadmap

What the paper (GlowTact.pdf, 8 pp.) already evaluates, what comparable VBTS
papers evaluate, and what to add -- ordered by value per unit of effort.
Each item says what it needs: **now** = the data on disk is enough,
**CNC day** = one session on the existing CNC + HX711 rig, **hardware** =
something we do not have yet.

## What the paper covers today

| section | evidence |
|---|---|
| contact geometry | 9DTact reconstruction pipeline on screws / M1 thread / ball array -- qualitative |
| contact sensitivity | SNR vs force, minimum detectable force (0.12 N vs Mini 0.23 N), passive 1 g objects |
| normal-force estimation | ResNet-18 on 13,116 controlled + 1,600 everyday-object pairs; RMSE 0.276 N vs Mini 0.285 N |

The title-level claim -- *raw darkness is a direct image of contact pressure*
-- is argued from the optical principle and shown qualitatively, but never
measured against a pressure reference.

## Tier 1 -- support the core claim

### 1. Pressure-distribution ground truth  (hardware: Prescale film ~$100)
Fujifilm Prescale (0.006-300 MPa, permanent microcapsule print) between probe
and membrane during one press gives a pressure map and a darkening map of the
same contact. Report pixel-wise correlation and the radial profile against
Hertz theory for the spherical probe. Alternative/complement: FEM-generated
distributions (Sferrazza & D'Andrea 2019). **A partial version is possible
now**: for the round probe, darkening radial profile vs the Hertz profile
p(r) = p0 sqrt(1 - r^2/a^2), and contact radius a vs F^(1/3).

### 2. Spatial resolution, quantified  (CNC day; localization: now)
Line-pair gratings / pin arrays at decreasing pitch -> MTF or minimum resolved
pitch, the standard TacEva-style metric, instead of "M1 threads are visible".
Contact-localisation error (CNC xy vs darkening centroid) is computable now
from the controlled-loading filenames.

### 3. Temporal behaviour  (CNC day)
Step response rise/recovery time; membrane-gel *decoupling* time after
release (unique to the air-gap principle); hysteresis loop (load/unload at the
same location -- the current dataset only has loading steps); creep under
constant force; repeatability over N presses at one spot.

### 4. Durability  (CNC day, unattended)
10^4 loading cycles; track SNR, minimum detectable force and the
darkening-force slope vs cycle count; membrane replacement consistency (does a
new membrane need a new calibration?).

## Tier 2 -- capabilities the library is halfway to

### 5. 6-D force / torque  (hardware: 6-axis F/T sensor)
9DTact trains 6-D force regression on ~100k pairs / 175 objects and
generalises to unseen objects. `glowtact.shear` already yields the image
features for Fx, Fy, Mz (mean shear, curl); missing is labelled data. Even
3-D (Fx, Fy, Fz) allows a head-to-head with 9DTact's table.

### 6. Slip detection, quantified  (CNC day -- same session as 5)
GelSlim 3.0 treats shape / force / slip as the standard triple. The temporal
slip modality exists (`glowtact.shear.relative_slip`); it needs ground truth:
commanded tangential displacement under known normal load -> detection
latency and F1 for incipient and gross slip.

### 7. Curved fingertip, quantified  (CNC day)
The omnidirectional and humanoid fingertips are shown only in photographs.
Repeat SNR / minimum detectable force / force regression on the fingertip;
`glowtact.undistort` and `glowtact.fisheye` handle the optics.

### 8. Head-to-head with 9DTact / DTact  (hardware: open-source 9DTact build)
Same probe ladder on a 9DTact: SNR, minimum detectable force, reconstruction
error (ball-array sphere fit via `eval_sphere_fit`). Same "darkness" family,
so a fairer comparison than GelSight Mini alone.

## Tier 3 -- downstream tasks and data

### 9. Cross-sensor benchmark tasks  (now for GlowTact side; Mini data exists)
TacVerse / DigiTac style: shape classification and grating-pitch
classification on the same objects across GlowTact and GelSight Mini;
supports the claim that an un-calibrated darkening image is easier to learn from.

### 10. Tasks that exploit the low-force advantage  (robot + gripper)
Fragile / light-object grasping (egg, paper cup, tofu) success rate under a
force ceiling; tactile servoing (edge / surface following as in DigiTac).

### 11. Release the dataset  (now)
13,116 controlled + 1,600 everyday pairs already exist; with 6-D force and
slip labels added it becomes the citable asset.

## Tier 4 -- cheap sensor characterisations reviewers ask for

* Ambient light / temperature robustness (the black membrane should make this
  a strength -- measure it).
* Design-space sweeps: sandpaper grit (P2500 vs others), membrane tension,
  membrane thickness -> sensitivity and hysteresis.
* Multi-contact: two probes at once -- separability and cross-talk.

## If only three

1 (pressure ground truth), 3+4 combined (one CNC day: dynamics, hysteresis,
durability), 5+6 combined (one F/T-sensor session labels both 6-D force and
slip; the library code is in place).

## Done from the data on disk (2026-09-01)

`~/tactile_data/glowtact/analysis/cnc_characterization/` — 18 logged
iterations; final: contact localisation RMS 0.30 mm (round probe, p95 0.52,
25 px/mm); sensitivity map R² 0.994 per location with a systematic 3x
centre-to-rim slope variation (perimeter LEDs); Hertz contact-radius exponent
0.322 per location vs the theoretical 1/3, radial darkening profile within RMS
0.063 of the Hertz pressure profile. Undistortion with the shipped map made
localisation worse -- refit it per unit before use.

## Start now with the data on disk

`~/tactile_data/glowtact/cnc/<unit>_<pad>_<set>/...` encodes
`probe | index | (x, y, z) | force` in every filename with an `initial.jpg`
reference per probe (unit1_pad0_shapes: 7,118 frames, round / star / triangle).

| analysis | needs | serves item |
|---|---|---|
| contact-localisation error: darkening centroid vs CNC xy, per probe, over the pad | filenames | 2 |
| sensitivity map: darkening-per-newton slope at every xy (spatial uniformity) | filenames | 3, paper Fig. 10 |
| Hertz check on the round probe: contact radius vs F^(1/3); radial darkening vs Hertz p(r) | round probe frames | 1 (partial) |
| gel stiffness curve: CNC z vs force (force-displacement per probe) | filenames | 3 |
| contact area vs force per probe shape (round / star / triangle) | frames | 2 |
| spatial robustness of the force model: error vs distance from pad centre | existing ResNet or `glowtact.force` | paper Table I |
| shape classification across the three probes (and vs Mini, whose CNC set exists) | frames | 9 |

Not possible from the current data: hysteresis (no unloading frames),
repeatability (locations are random, not repeated), anything temporal
(stills, not video), shear/slip truth, pressure truth.

## Sources

9DTact (Lin et al. 2023, arXiv:2308.14277) · GelSlim 3.0 (Taylor et al. 2022,
arXiv:2103.12269) · TacEva (arXiv:2509.19037) · TacO benchmark
(arXiv:2605.21976) · TacVerse (arXiv:2606.25877) · DigiTac (arXiv:2206.13657)
· FEM ground-truth force distributions (arXiv:1909.04000) · Fujifilm Prescale.
