# Using the GlowTact sensor

How to view, snapshot, and record a **GlowTact** tactile camera. Everything
here is **camera-only — no force sensor is required**. (Force is only needed by
`scripts/manual_collect.py`, the combined tactile+force recorder.)

All commands use the self-contained `glowtact.camera` (the `glowtact-camera` console script)
package, which depends on only OpenCV + NumPy and works both inside this project
and copied elsewhere. See the library README
for the library API.

- [1. Connect and find the sensor](#1-connect-and-find-the-sensor)
- [2. Live view](#2-live-view)
- [3. Interactive capture — keys + a data folder](#3-interactive-capture--keys--a-data-folder)
- [4. One-shot snapshot / timed video](#4-one-shot-snapshot--timed-video)
- [5. Stream over the network (browser)](#5-stream-over-the-network-browser)
- [6. Difference modes](#6-difference-modes)
- [7. 3D reconstruction](#7-3d-reconstruction)
- [Troubleshooting](#troubleshooting)

---

## 1. Connect and find the sensor

Plug the GlowTact into USB, then:

```bash
cd TheProbe/ProbingPi
glowtact-camera list
```

Expected:

```
GlowTact 1e45:8022 candidate nodes: /dev/video2, /dev/video3
-> capture node that delivers frames: /dev/video2
```

The sensor is matched by its **USB id** (`1e45:8022` for the 180° unit), so the
port does not matter. A UVC camera exposes several `/dev/videoN` nodes but only
one delivers frames; `list` probes and reports which.

- **120° builds** use different ids — pass them explicitly:
  `--vendor 1bcf --product 28c4` (120° #1) or `--vendor 337b --product 090c`
  (120° #2, autofocus).
- **Skip discovery** entirely (e.g. macOS/Windows, where auto-find is
  unavailable): add `--index N` to any command.

---

## 2. Live view

A live window (needs a display; run locally, not over plain SSH):

```bash
glowtact-camera view --diff dark
```

Keys: `b` set baseline (untouched), `s` save a frame to the current directory,
`q` quit. Drop `--diff dark` to see the raw frame.

---

## 3. Interactive capture — keys + a data folder

The main workflow: a live window whose keys save snapshots and video into a
timestamped session folder you choose.

```bash
glowtact-camera capture -d ~/glowtact_captures
```

| key | action |
|-----|--------|
| `s` | save a snapshot (raw PNG + a diff PNG when a baseline is set) |
| `r` | start / stop video recording (toggle) |
| `b` | set baseline / re-tare — do this with the gel **untouched** |
| `m` | cycle the difference view: raw → dark → signed → bright |
| `h` | toggle the help overlay |
| `q` / Esc | quit (stops recording first) |

The window shows a status line (view mode, fps, snapshot count) and, while
recording, a red ● with elapsed time and frame count.

**What lands on disk** — each run makes a fresh session so runs never collide:

```
~/glowtact_captures/session_YYYYMMDD_HHMMSS/
    snapshots/   snapshot_0001.png, snapshot_0001_diff.png, ...
    videos/      video_0001.avi
    index.csv    id, type, iso_time, file, diff_file, diff_mode, peak, frames, fps
```

Notes:
- Video frames are pulled on a separate thread, so the saved clip has **clean
  frames with no overlay burned in** and the display rate never affects it.
- `index.csv` records a `peak` per snapshot (largest deviation from baseline) —
  a quick contact-strength indicator; a value of 127 in `signed` at gain 1
  means the diff saturated.

Useful flags: `--diff signed` (initial view; also cycle with `m`),
`--fps N` (fix video playback rate; default = measured live rate),
`--gain G` (amplify the diff for display).

---

## 4. One-shot snapshot / timed video

Non-interactive, for scripting:

```bash
# a single raw frame
glowtact-camera snapshot -o shot.png

# a single dark-diff frame: captures a baseline, waits --delay, then shoots
glowtact-camera snapshot -o contact.png --diff dark --delay 3

# a fixed-length clip (default 10 s, real-time fps)
glowtact-camera record -o clip.avi --duration 10
glowtact-camera record -o clip.avi --duration 10 --diff dark
```

---

## 5. Stream over the network (browser)

Headless-friendly — no display needed. Serves MJPEG over HTTP; one camera fans
out to many viewers.

```bash
glowtact-camera serve --port 8000 --diff dark
```

Open `http://<host>:8000/` in any browser. `Ctrl-C` to stop.

> Only one program can open the camera at a time. Stop `serve` (or any other
> capture) before running `capture`/`view`, and vice-versa.

---

## 6. Difference modes

The GlowTact's contact signal is subtle; a **difference against an untouched
baseline** makes it visible. These match the rest of the project byte-for-byte,
so a live diff and an offline capture are comparable.

| mode     | formula          | use                                             |
|----------|------------------|-------------------------------------------------|
| `signed` | `frame-base+128` | grey background, both push directions visible   |
| `dark`   | `base-frame` ≥0  | **GlowTact default** — contact darkens → shows bright |
| `bright` | `frame-base` ≥0  | contact brightens                               |

Always set a baseline first (`b` in the window, or `--diff` auto-captures one in
the one-shot commands) with the gel not in contact.

---

## 7. 3D reconstruction

Saved GlowTact frames/diffs feed the depth pipelines. The GlowTact uses the
darkness-LUT method (`scripts/reconstruct_3d.py`), calibrated from ball presses:

```bash
# live depth view straight from the sensor (uses the pc_server.yaml camera key)
python scripts/reconstruct_3d.py live --camera glowtact_cam \
    --calib glowtact_depth_calib.npz

# offline: a folder of *_diff.png (as written above) -> depth maps
python scripts/reconstruct_3d.py depth <session>/snapshots \
    --from-diff --pattern '*_diff.png' --calib glowtact_depth_calib.npz
```

(For **GelSight Mini** captures the learned TouchNet / vanilla photometric-stereo
pipelines live in `scripts/touchnet_reconstruct.py` and
`scripts/photometric_stereo.py`.)

---

## Troubleshooting

| symptom | fix |
|---------|-----|
| `no GlowTact (1e45:8022) found` | check the cable; `glowtact-camera list`; for a 120° unit pass `--vendor/--product`; on macOS/Windows use `--index N` |
| `found … but none delivered frames` | the camera is **busy** — another `serve`/`view`/`capture`/server has it open. Close that first. (`fuser /dev/video2` shows who holds it.) |
| window doesn't open over SSH | `view`/`capture` need a display. Use `serve` (browser) or the one-shot `snapshot`/`record` instead. |
| diff is all grey / black | set a baseline with the gel untouched (`b`), then touch the gel |
| USB auto-discovery unavailable | Linux-only; pass `--index N`. `pip install pyudev` makes discovery more robust but a `/sys` fallback works without it. |

See also: the library README (
API), [`docs/data_format.md`](data_format.md) (dataset layout).
