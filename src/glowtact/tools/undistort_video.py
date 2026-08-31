"""Write 180-deg-corrected, centre-cropped copies of GlowTact clips.

Applies the generic fisheye rectification in ``fisheye.py`` (auto-detected disc,
equidistant model, central pinhole reprojection) so the barrel distortion is
flattened and the black rim/corners are gone, then encodes browser-playable
H.264. The disc is detected once per clip from its mean frame.

    glowtact-undistort-video <session_dir>
    glowtact-undistort-video --videos a.avi --keep-fov 90
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from .. import fisheye


def undistort_clip(path, out_mp4, keep_fov=100.0, out_size=480, fps=None,
                   calib=None):
    import cv2

    cap = cv2.VideoCapture(path)
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 20.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    gray_mean = np.mean(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                                  for f in frames]), axis=0).astype(np.uint8)
    rect = fisheye.make_rectifier(gray_mean, calib=calib, undistort=calib is None,
                                  keep_fov=keep_fov, out_size=out_size)

    tmp = tempfile.mkdtemp(prefix="undistort_")
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"{i:05d}.png"), rect(f))
        _encode(tmp, out_mp4, fps)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out_mp4


def _encode(frame_dir, out_mp4, fps):
    import cv2

    pngs = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-framerate", f"{fps:.3f}", "-pattern_type",
               "glob", "-i", os.path.join(frame_dir, "*.png"), "-c:v",
               "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               out_mp4]
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            return
    f0 = cv2.imread(pngs[0])
    vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (f0.shape[1], f0.shape[0]))
    for p in pngs:
        vw.write(cv2.imread(p))
    vw.release()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", nargs="?")
    ap.add_argument("--videos", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-fov", type=float, default=100.0)
    ap.add_argument("--out-size", type=int, default=480)
    ap.add_argument("--calib", nargs="?", default=None, const="__default180__",
                    help="saved OpenCV-fisheye calibration yaml (metric); bare "
                         "--calib uses the repo's glowtact180 calibration")
    args = ap.parse_args(argv)
    if args.calib == "__default180__":
        args.calib = fisheye.DEFAULT_CALIB_180

    if args.videos:
        videos, out_dir = args.videos, (args.out or "undistorted")
    elif args.session:
        videos = sorted(glob.glob(os.path.join(args.session, "videos",
                                               "*.avi")))
        out_dir = args.out or os.path.join(args.session, "videos_undistorted")
    else:
        ap.error("give a session dir or --videos")
    os.makedirs(out_dir, exist_ok=True)
    for v in videos:
        name = os.path.splitext(os.path.basename(v))[0]
        out = os.path.join(out_dir, f"{name}_undistorted.mp4")
        print(f"[{name}] undistort ({'calib' if args.calib else 'generic'})"
              f" -> {out}", flush=True)
        undistort_clip(v, out, keep_fov=args.keep_fov, out_size=args.out_size,
                       calib=args.calib)
    print(f"outputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
