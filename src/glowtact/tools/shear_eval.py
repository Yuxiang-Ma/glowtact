"""Evaluate marker-free shear trackers on recorded GlowTact sessions.

Runs every tracker in ``glowtact.shear`` over each video under a session
(or an explicit list of videos), computes the ground-truth-free quality metrics,
and writes:

  <out>/metrics.json                 all metrics, per video + aggregated
  <out>/comparison.png               metric bars + per-video winner
  <out>/<video>_series.png           shear x/y + contact-area time series
  <out>/<video>_overlay.png          annotated frames (flow + contact mask)

    glowtact-shear-eval ~/glowtact_captures/session_YYYYMMDD_HHMMSS
    glowtact-shear-eval --videos a.avi b.avi --out shear_out
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

from .. import shear

METRIC_KEYS = ["warp_residual", "noise_floor", "indent_intrusion",
               "coherence", "score"]
LOWER_BETTER = {"warp_residual", "noise_floor", "indent_intrusion"}


def _overlay_frames(video, tracker_name, out_path, n_panels=3):
    """Annotated frames: contact mask outline + shear flow arrows."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    base = frames[0]
    tr = shear.TRACKERS[tracker_name]()
    tr.reset(base)
    # pick frames with the most contact (most interesting for shear)
    areas = []
    fov = shear.fov_mask(base.shape)
    for f in frames[1:]:
        cm = shear.contact_mask(f, base)
        areas.append((cm & fov).sum())
    order = np.argsort(areas)[::-1]
    picks = sorted(order[:n_panels] + 1)

    fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 5))
    if len(picks) == 1:
        axes = [axes]
    for ax, i in zip(axes, picks):
        cm = shear.contact_mask(frames[i], base)
        r = tr.estimate(frames[i], cm)
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2RGB)
        # red contact outline
        cnts, _ = cv2.findContours(cm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, cnts, -1, (255, 0, 0), 2)
        ax.imshow(rgb)
        # sparse flow arrows if available
        if r.field is not None and r.field[0] is not None \
                and not isinstance(r.field[0], str):
            pts, vecs = r.field
            step = max(1, len(pts) // 200)
            for (x, y), (u, v) in zip(pts[::step], vecs[::step]):
                ax.arrow(x, y, u * 8, v * 8, color="lime", width=0.4,
                         head_width=4, length_includes_head=True)
        # global shear arrow at centre
        h, w = frames[i].shape
        ax.arrow(w / 2, h / 2, r.dx * 25, r.dy * 25, color="yellow", width=2,
                 head_width=12, length_includes_head=True)
        ax.set_title(f"frame {i}: shear=({r.dx:+.2f},{r.dy:+.2f})px "
                     f"coh={r.coherence:.2f}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{os.path.basename(video)} -- {tracker_name} "
                 f"(red=contact excluded, green=speckle flow x8, "
                 f"yellow=shear x25)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _series_plot(result, video_name, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    area = np.array(result["contact_area"])
    names = list(result["series"])
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for name in names:
        s = result["series"][name]
        axes[0].plot(s["dx"], label=name)
        axes[1].plot(s["dy"], label=name)
    axes[0].set_ylabel("shear x (px)"); axes[0].legend(fontsize=8, ncol=4)
    axes[0].grid(alpha=0.3); axes[0].axhline(0, color="k", lw=0.5)
    axes[1].set_ylabel("shear y (px)"); axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[2].fill_between(range(len(area)), area * 100, color="0.7")
    axes[2].set_ylabel("contact area (%)"); axes[2].set_xlabel("frame")
    axes[2].grid(alpha=0.3)
    fig.suptitle(f"{video_name}: shear vs contact "
                 "(shear should stay flat where contact is a pure press)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _comparison_plot(agg, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(agg)
    fig, axes = plt.subplots(1, len(METRIC_KEYS), figsize=(4 * len(METRIC_KEYS),
                                                           4))
    for ax, key in zip(axes, METRIC_KEYS):
        vals = [agg[n].get(key, np.nan) for n in names]
        colors = ["C2" if ((key not in LOWER_BETTER) ==
                           (v == (max(vals) if key not in LOWER_BETTER
                                  else min(vals)))) else "C0"
                  for v in vals]
        ax.bar(names, vals, color=colors)
        ax.set_title(key + ("  (lower=better)" if key in LOWER_BETTER
                            else "  (higher=better)"), fontsize=9)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Shear tracker comparison (mean over videos)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", nargs="?",
                    help="session dir (uses its videos/*.avi)")
    ap.add_argument("--videos", nargs="+", help="explicit video paths")
    ap.add_argument("--out", default=None,
                    help="output dir (default <session>/shear_eval)")
    ap.add_argument("--stride", type=int, default=1,
                    help="frame stride (2 = every other frame, faster)")
    ap.add_argument("--trackers", nargs="+", default=None,
                    choices=list(shear.TRACKERS))
    args = ap.parse_args(argv)

    if args.videos:
        videos = args.videos
        out_dir = args.out or "shear_eval_out"
    elif args.session:
        videos = sorted(glob.glob(os.path.join(args.session, "videos",
                                               "*.avi")))
        out_dir = args.out or os.path.join(args.session, "shear_eval")
    else:
        ap.error("give a session dir or --videos")
    if not videos:
        ap.error("no videos found")
    os.makedirs(out_dir, exist_ok=True)

    per_video = {}
    for v in videos:
        name = os.path.splitext(os.path.basename(v))[0]
        print(f"[{name}] evaluating ...", flush=True)
        res = shear.evaluate_video(v, tracker_names=args.trackers,
                                   stride=args.stride)
        per_video[name] = {k: res[k] for k in res if k != "series"}
        _series_plot(res, name, os.path.join(out_dir, f"{name}_series.png"))
        # overlay for the best tracker on this video
        best = max((n for n in shear.TRACKERS if isinstance(res.get(n), dict)
                    and "score" in res[n]),
                   key=lambda n: res[n]["score"], default="lk")
        try:
            _overlay_frames(v, best, os.path.join(out_dir, f"{name}_overlay.png"))
        except Exception as exc:
            print(f"  overlay failed: {exc}")
        for n in shear.TRACKERS:
            if isinstance(res.get(n), dict) and "score" in res[n]:
                print(f"    {n:10s} score={res[n]['score']:.3f} "
                      f"resid={res[n]['warp_residual']:.3f} "
                      f"noise={res[n]['noise_floor']:.3f} "
                      f"intrusion={res[n]['indent_intrusion']:.3f} "
                      f"coh={res[n]['coherence']:.3f}")

    # aggregate across videos
    trackers = [n for n in shear.TRACKERS
                if any("score" in per_video[v].get(n, {}) for v in per_video)]
    agg = {}
    for n in trackers:
        rows = [per_video[v][n] for v in per_video
                if "score" in per_video[v].get(n, {})]
        agg[n] = {k: float(np.nanmean([r[k] for r in rows]))
                  for k in METRIC_KEYS if k in rows[0]}
    ranked = sorted(agg, key=lambda n: agg[n]["score"], reverse=True)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"per_video": per_video, "aggregate": agg,
                   "ranking": ranked}, f, indent=2)
    _comparison_plot(agg, os.path.join(out_dir, "comparison.png"))

    print("\n=== AGGREGATE RANKING (best first) ===")
    for n in ranked:
        m = agg[n]
        print(f"  {n:10s} score={m['score']:.3f}  resid={m['warp_residual']:.3f}"
              f"  noise={m['noise_floor']:.4f}  intrusion={m['indent_intrusion']:.3f}"
              f"  coh={m['coherence']:.3f}")
    print(f"\noutputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
