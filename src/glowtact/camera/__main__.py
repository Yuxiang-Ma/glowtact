"""Command-line entry: list / view / snapshot / record / serve.

    glowtact-camera list                    # or: python -m glowtact.camera list
    glowtact-camera capture -d ~/glowtact_captures   # interactive, keys
    glowtact-camera view --diff dark        # needs a display
    glowtact-camera snapshot -o shot.png --diff dark
    glowtact-camera record -o clip.avi --duration 10
    glowtact-camera serve --port 8000 --diff dark   # browser stream

Every subcommand accepts --index N to skip USB auto-discovery, and
--vendor/--product to target a non-180 GlowTact (e.g. 1bcf:28c4 for the 120).
No force sensor is involved in any of these.
"""
import argparse
import sys
import time

from . import __version__
from .camera import GlowTactCamera
from .diff import MODES
from .discovery import (GLOWTACT_180_PID, GLOWTACT_180_VID, find_glowtact,
                        list_candidates)


def _open(args):
    return GlowTactCamera(index=args.index, vendor_id=args.vendor,
                          product_id=args.product, which=args.which)


def _add_common(p):
    p.add_argument("--index", type=int, default=None,
                   help="explicit /dev/videoN index (skip USB discovery)")
    p.add_argument("--vendor", default=GLOWTACT_180_VID,
                   help=f"USB vendor id (default {GLOWTACT_180_VID}=GlowTact180)")
    p.add_argument("--product", default=GLOWTACT_180_PID,
                   help=f"USB product id (default {GLOWTACT_180_PID})")
    p.add_argument("--which", type=int, default=0,
                   help="pick among multiple identical units (0=first)")


def cmd_list(args):
    nodes = list_candidates(args.vendor, args.product)
    if not nodes:
        print(f"no GlowTact ({args.vendor}:{args.product}) found")
        return 1
    print(f"GlowTact {args.vendor}:{args.product} candidate nodes: "
          f"{', '.join(nodes)}")
    try:
        idx = find_glowtact(args.vendor, args.product, which=args.which)
        print(f"-> capture node that delivers frames: /dev/video{idx}")
    except Exception as exc:
        print(f"-> none delivered frames: {exc}")
        return 1
    return 0


def cmd_capture(args):
    from .interactive import run_session

    with _open(args) as cam:
        print(f"/dev/video{cam.index}  {cam.resolution}  "
              f"~{cam.measure_fps() or 0:.0f} fps")
        session_dir = run_session(cam, args.data_dir, diff_mode=args.diff,
                                  gain=args.gain, fps=args.fps)
    print(f"session saved under {session_dir}")
    return 0


def cmd_view(args):
    import cv2

    with _open(args) as cam:
        print(f"/dev/video{cam.index}  {cam.resolution}  "
              f"~{cam.measure_fps() or 0:.0f} fps")
        print("keys: b=set baseline (untouched)  s=save frame  q=quit")
        cam.set_baseline()
        saved = 0
        for frame in cam.frames():
            view = cam.difference(frame, args.diff) if args.diff else frame
            cv2.imshow("GlowTact", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):
                cam.set_baseline(frame)
                print("baseline set")
            elif key == ord("s"):
                saved += 1
                name = f"glowtact_{saved:04d}.png"
                cv2.imwrite(name, view)
                print(f"saved {name}")
        cv2.destroyAllWindows()
    return 0


def cmd_snapshot(args):
    with _open(args) as cam:
        if args.diff:
            cam.set_baseline()
            print("baseline captured; place object now, snapshot in "
                  f"{args.delay:.0f}s")
            time.sleep(args.delay)
        cam.snapshot(args.output, diff_mode=args.diff, gain=args.gain)
        print(f"saved {args.output}")
    return 0


def cmd_record(args):
    with _open(args) as cam:
        if args.diff:
            cam.set_baseline()
        n = cam.record(args.output, args.duration, fps=args.fps,
                       diff_mode=args.diff, gain=args.gain)
        print(f"wrote {n} frames -> {args.output}")
    return 0


def cmd_serve(args):
    from .mjpeg import serve_mjpeg

    with _open(args) as cam:
        if args.diff:
            cam.set_baseline()
        print(f"/dev/video{cam.index}  serving MJPEG at "
              f"http://{args.host}:{args.port}/   (Ctrl-C to stop)")
        serve_mjpeg(cam, host=args.host, port=args.port, quality=args.quality,
                    diff_mode=args.diff, gain=args.gain)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="glowtact-camera",
        description="Stream / snapshot / record a GlowTact camera "
                    "(no force sensor needed).")
    parser.add_argument("--version", action="version",
                        version=f"glowtact {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list matching cameras")
    _add_common(p)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser(
        "capture",
        help="interactive session: keys save snapshots/video to a data folder")
    _add_common(p)
    p.add_argument("-d", "--data-dir", default="~/glowtact_captures",
                   help="folder for the timestamped session dir "
                        "(default ~/glowtact_captures)")
    p.add_argument("--diff", choices=MODES, default="dark",
                   help="initial difference view; cycle live with 'm' "
                        "(default dark)")
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--fps", type=float, default=None,
                   help="video playback rate (default: measured live rate)")
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("view", help="live OpenCV window (needs a display)")
    _add_common(p)
    p.add_argument("--diff", choices=MODES, default=None,
                   help="show a difference image instead of the raw frame")
    p.set_defaults(func=cmd_view)

    p = sub.add_parser("snapshot", help="save one frame")
    _add_common(p)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--diff", choices=MODES, default=None)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--delay", type=float, default=3.0,
                   help="seconds between baseline and shot (--diff only)")
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("record", help="record a video clip")
    _add_common(p)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--fps", type=float, default=None,
                   help="default: measured live rate")
    p.add_argument("--diff", choices=MODES, default=None)
    p.add_argument("--gain", type=float, default=1.0)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("serve", help="stream over HTTP (MJPEG, view in browser)")
    _add_common(p)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--quality", type=int, default=80)
    p.add_argument("--diff", choices=MODES, default=None)
    p.add_argument("--gain", type=float, default=1.0)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
