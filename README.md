# glowtact

Everything for the **GlowTact** tactile sensor in one installable library:
camera driver and streaming, difference imaging, 9DTact-style darkness→depth
reconstruction (offline and realtime 3D), pressed-grid fisheye undistortion,
shear-field tracking, and the mesh/figure rendering used in the papers.

Extracted from the ProbingPi data-collection project so it can be used from
any codebase. Depends on NumPy + OpenCV + PyYAML; heavier things (Open3D,
matplotlib, pyudev) are optional and imported lazily.

## Install

```bash
pip install -e .                 # library + console scripts
pip install -e '.[render,udev]'  # + Open3D/matplotlib figures, USB auto-discovery
pip install -e '.[dev]' && pytest
```

## Library map

| module | what |
|---|---|
| `glowtact.camera` | `GlowTactCamera` (auto-discovers by USB id, newest-frame reader), `serve_mjpeg`, interactive capture session |
| `glowtact.capture` | `CameraReader` (threaded, drop-old), `open_live` + `register_camera_opener` — the hook a host uses to plug in its own device registry |
| `glowtact.diffio` | the signed/dark/bright difference codec (`signed_difference`, `render_difference`, `load_signed_diff`, `recover_signed`) |
| `glowtact.depth` | darkness→depth lookup table: `build_lut`, `depth_from_diff`, calibration I/O |
| `glowtact.reconstruct` | `calibrate` (from ball-array presses) · `depth` · `live` · `view` · `spin` · `realtime` — also `glowtact-reconstruct` |
| `glowtact.realtime` | non-blocking capture → reconstruct → Open3D/OpenCV render pipeline |
| `glowtact.undistort` | fit a rectification from a pressed ball grid / chessboard (robust fisheye-tolerant detector), apply it, live `calibrate` — `glowtact-undistort` |
| `glowtact.fisheye` | analytic 180° fisheye correction + per-unit OpenCV fisheye models for clips |
| `glowtact.shear` | shear-field trackers (LK, Farneback, DIS, phase-correlation), `ShearField`, quality scores |
| `glowtact.clean_diff`, `glowtact.enhance_diff` | membrane-shift suppression; display band-pass — `glowtact-clean-diff`, `glowtact-enhance-diff` |
| `glowtact.render`, `glowtact.figure` | height-map meshes, baked relief shading, turntables, paper-style figure panels |
| `glowtact.crop`, `glowtact.imgfilt`, `glowtact.mask` | pad crops / rotation bounds, DoG filters, the circular aperture mask |
| `glowtact.tools` | `glowtact-undistort-video`, `glowtact-shear-render`, `glowtact-shear-field`, `glowtact-shear-eval` |
| `glowtact.calib` | shipped per-unit undistortion maps (`glowtact.calib_path("glowtact180", ...)`) |

## Quick start

```python
from glowtact.camera import GlowTactCamera
from glowtact import depth, diffio

with GlowTactCamera() as cam:                 # or GlowTactCamera(index=2)
    cam.set_baseline()                        # gel untouched
    frame = cam.read()
    dark = cam.difference(frame, mode="dark") # contact shows bright on black

lut, stats = depth.load_calibration("glowtact_depth_calib.npz")
height_mm = depth.depth_from_diff(dark, lut, stats["lighting_threshold"])
```

```bash
glowtact-camera serve --port 8000 --diff dark          # browser stream
glowtact-undistort calibrate --camera 1bcf:28c4 --rows 6 --cols 9   # live
glowtact-reconstruct calibrate undistort_debug --ball-radius-mm 2.35 --spacing-mm 7.4
glowtact-reconstruct depth snapshots/ --from-diff --calib glowtact_depth_calib.npz
glowtact-reconstruct realtime --camera 1bcf:28c4 --calib glowtact_depth_calib.npz
```

`--camera` on the live commands is a `/dev/video` index or a USB
`vendor:product[:which]` id. A host application with its own camera registry
registers an opener once and then passes its own keys:

```python
from glowtact.capture import register_camera_opener
register_camera_opener(lambda camera, config, **kw: my_open(config, camera, **kw))
```

(ProbingPi does exactly this in its `scripts/tactile/common/capture.py`.)

## Docs

- `docs/glowtact_sensor.md` — capturing data with the sensor
- `docs/glowtact_shear.md` — the shear-tracking pipeline and its evaluation
