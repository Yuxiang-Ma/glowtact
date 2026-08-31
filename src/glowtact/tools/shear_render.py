"""Render shear-tracking overlay videos for remote inspection / debugging.

For each input clip, writes a browser-playable H.264 MP4 with, per frame:

  * the grayscale gel frame,
  * the contact blob outlined in red (this region is EXCLUDED from shear),
  * one arrow per tracker from the centre = its shear vector (scaled),
  * a live readout (frame #, each tracker's shear px, contact area %),
  * a scrolling strip chart of shear-y so temporal behaviour is visible.

Encoded with ffmpeg libx264 (+faststart) so it plays inline on a Hugging Face
repo page. Falls back to OpenCV mp4v if ffmpeg is missing (may not play in all
browsers).

    glowtact-shear-render ~/glowtact_captures/session_XXduration \
        --out shear_videos
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from .. import fisheye, shear

# One BGR colour per tracker for its shear arrow / strip trace.
COLORS = {"lk": (255, 160, 0), "farneback": (0, 200, 255),
          "dis": (0, 255, 0), "phasecorr": (60, 60, 255)}
ARROW_SCALE = 40.0          # px on screen per px of shear (shear is small)


def _draw(frame_gray, i, contact, results, area, strip, tnames):
    import cv2

    h, w = frame_gray.shape
    vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    cnts, _ = cv2.findContours(contact.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 0, 255), 2)

    cx, cy = w // 2, h // 2
    cv2.circle(vis, (cx, cy), 3, (255, 255, 255), -1)
    for name in tnames:
        r = results[name]
        col = COLORS[name]
        tip = (int(cx + r.dx * ARROW_SCALE), int(cy + r.dy * ARROW_SCALE))
        cv2.arrowedLine(vis, (cx, cy), tip, col, 2, tipLength=0.3)

    # text panel
    cv2.rectangle(vis, (0, 0), (232, 20 + 20 * (len(tnames) + 1)),
                  (0, 0, 0), -1)
    cv2.putText(vis, f"frame {i}  contact {area * 100:4.1f}%", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    y = 36
    for name in tnames:
        r = results[name]
        cv2.putText(vis, f"{name:9s} ({r.dx:+.2f},{r.dy:+.2f}) c{r.coherence:.2f}",
                    (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[name], 1,
                    cv2.LINE_AA)
        y += 20

    # strip chart of shear-y (bottom 90 px)
    sh = 90
    chart = np.full((sh, w, 3), 245, np.uint8)
    if len(strip[tnames[0]]) > 1:
        allv = np.concatenate([np.array(strip[n]) for n in tnames])
        vmax = max(1.0, float(np.abs(allv).max()))
        mid = sh // 2
        cv2.line(chart, (0, mid), (w, mid), (200, 200, 200), 1)
        for name in tnames:
            pts = strip[name]
            xs = np.linspace(0, w - 1, len(pts)).astype(int)
            ys = (mid - np.array(pts) / vmax * (sh // 2 - 4)).astype(int)
            for k in range(1, len(pts)):
                cv2.line(chart, (xs[k - 1], ys[k - 1]), (xs[k], ys[k]),
                         COLORS[name], 1)
        cv2.putText(chart, f"shear-y (px), full-scale +/-{vmax:.1f}", (6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([vis, chart])


def render_video(path, out_mp4, tnames=None, fps=None, undistort=False,
                 keep_fov=100.0, out_size=480, calib=None):
    import cv2

    tnames = tnames or list(shear.TRACKERS)
    cap = cv2.VideoCapture(path)
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 20.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()

    if undistort or calib:
        mean = np.mean(np.stack(frames), axis=0).astype(np.uint8)
        rect = fisheye.make_rectifier(mean, calib=calib, undistort=undistort,
                                      keep_fov=keep_fov, out_size=out_size)
        frames = [rect(f) for f in frames]
    base = frames[0]
    fov = shear.fov_mask(base.shape)

    trackers = {}
    for n in tnames:
        t = shear.TRACKERS[n]()
        try:
            t.reset(base)
            trackers[n] = t
        except Exception as exc:
            print(f"  {n} unavailable: {exc}")
    tnames = [n for n in tnames if n in trackers]

    strip = {n: [] for n in tnames}
    tmp = tempfile.mkdtemp(prefix="shear_render_")
    try:
        for i, g in enumerate(frames):
            if i == 0:
                continue
            cm = shear.contact_mask(g, base)
            area = float((cm & fov).sum() / max(fov.sum(), 1))
            results = {n: trackers[n].estimate(g, cm) for n in tnames}
            for n in tnames:
                strip[n].append(results[n].dy)
            panel = _draw(g, i, cm, results, area, strip, tnames)
            cv2.imwrite(os.path.join(tmp, f"{i:05d}.png"), panel)
        _encode(tmp, out_mp4, fps)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_mp4


def _encode(frame_dir, out_mp4, fps):
    """PNG frames -> H.264 MP4 via ffmpeg; OpenCV mp4v fallback."""
    import cv2

    pngs = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if not pngs:
        raise RuntimeError("no frames rendered")
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-framerate", f"{fps:.3f}",
               "-pattern_type", "glob", "-i",
               os.path.join(frame_dir, "*.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", out_mp4]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0:
            return
        print("  ffmpeg failed, falling back to mp4v:",
              r.stderr.decode()[-300:])
    frame0 = cv2.imread(pngs[0])
    h, w = frame0.shape[:2]
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in pngs:
        vw.write(cv2.imread(p))
    vw.release()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", nargs="?")
    ap.add_argument("--videos", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--trackers", nargs="+", default=None,
                    choices=list(shear.TRACKERS))
    ap.add_argument("--undistort", action="store_true",
                    help="calibration-free 180-deg correction + centre crop")
    ap.add_argument("--calib", nargs="?", default=None, const="__default180__",
                    help="saved OpenCV-fisheye calibration yaml; bare --calib "
                         "uses the repo's glowtact180 calibration")
    ap.add_argument("--keep-fov", type=float, default=100.0)
    ap.add_argument("--out-size", type=int, default=480)
    args = ap.parse_args(argv)
    if args.calib == "__default180__":
        args.calib = fisheye.DEFAULT_CALIB_180

    if args.videos:
        videos = args.videos
        out_dir = args.out or "shear_videos"
    elif args.session:
        videos = sorted(glob.glob(os.path.join(args.session, "videos",
                                               "*.avi")))
        out_dir = args.out or os.path.join(args.session, "shear_videos")
    else:
        ap.error("give a session dir or --videos")
    os.makedirs(out_dir, exist_ok=True)
    for v in videos:
        name = os.path.splitext(os.path.basename(v))[0]
        out = os.path.join(out_dir, f"{name}_shear.mp4")
        mode = "calib" if args.calib else ("undistort" if args.undistort
                                           else "raw")
        print(f"[{name}] rendering ({mode}) -> {out}", flush=True)
        render_video(v, out, tnames=args.trackers, undistort=args.undistort,
                     keep_fov=args.keep_fov, out_size=args.out_size,
                     calib=args.calib)
    print(f"outputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
