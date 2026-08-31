"""Render a GelSight-style shear ARROW FIELD over the GlowTact speckle.

Reproduces the marker-flow visualisation (Yuan et al., "Measurement of shear and
slip ...") without printed markers: a virtual grid of nodes over the natural
speckle, each drawn as an arrow from its rest position to its tracked position.
Arrow colour/opacity follows per-node confidence, so the texture-less contact
hole fades instead of showing garbage. Overlays the global shear, curl
(rotation/torsion) and divergence derived from the field.

Browser-playable H.264 (ffmpeg libx264), for viewing on a Hugging Face page.

    glowtact-shear-field <session_dir>
    glowtact-shear-field --videos clip.avi \
        --method phasecorr --spacing 26 --scale 12
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


# Node-confidence arrow ramp, identical to the live pipeline page's legend
# ("arrow colour = node confidence"): pale straw at low confidence, saturated
# amber at high. RGB there, BGR here.
CONF_LOW_BGR = (170, 240, 255)          # rgb(255,240,170)
CONF_HIGH_BGR = (40, 180, 255)          # rgb(255,180,40)
FILLED_BGR = (221, 208, 43)             # #2bd0dd -- interpolated into the hole
REJECTED_BGR = (148, 143, 138)          # #8a8f94 -- below min confidence


def _color(conf):
    """Confidence -> BGR along the page's straw->amber ramp.

    Previously COLORMAP_INFERNO, which runs black -> purple -> orange: the
    low-confidence end came out near-black and read as a *direction* hue rather
    than a trust level, so the field looked like a rainbow quiver that
    disagreed with the site. A single-hue ramp keeps lightness monotonic in
    confidence, which is the only thing the colour is supposed to encode.
    """
    c = float(np.clip(conf, 0, 1))
    return tuple(int(round(lo + (hi - lo) * c))
                 for lo, hi in zip(CONF_LOW_BGR, CONF_HIGH_BGR))


def _quiet_background(gray, denoise=True, fade=0.55):
    """Calm the gel so the arrows are the figure and the speckle is ground.

    The speckle is the measurement signal, but as a backdrop it is visually
    louder than the vectors drawn on top of it -- every frame reads as static.
    Two cheap steps fix that without touching anything the tracker sees (this
    is display only, computed after the field):

    * non-local-means denoise, which suppresses the fine grain while keeping
      the object's imprint edge, and
    * a fade toward white, which drops the remaining contrast so amber arrows
      sit clearly on top.
    """
    import cv2

    g = gray
    if denoise:
        g = cv2.fastNlMeansDenoising(g, None, h=7, templateWindowSize=7,
                                     searchWindowSize=21)
    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).astype(np.float32)
    if fade:
        vis = vis * (1.0 - fade) + 255.0 * fade
    return vis.astype(np.uint8)


SLIP_BGR = (40, 40, 230)          # slipping node + its cell: red
CONE_BGR = (0, 140, 255)          # traction ratio above mu: orange ring


def _draw_field(gray, field, contact, scale, min_conf, cap=None,
                filled_mask=None, slip=None, bg_denoise=True, bg_fade=0.55,
                slip_nodes=None, slip_stats=None, cone_nodes=None):
    """One panel. ``slip_nodes`` (bool per node) shades each slipping node's
    grid cell red -- the slippage AREA -- and ``cone_nodes`` rings the nodes
    whose traction ratio exceeds mu; ``slip_stats`` (from
    ShearField.slip_area) goes on the banner as the stick ratio."""
    import cv2

    vis = _quiet_background(gray, bg_denoise, bg_fade)
    cnts, _ = cv2.findContours(contact.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 0, 255), 1)

    if slip_nodes is not None and slip_nodes.any():
        # Shade the slipping nodes' cells so the sliding REGION reads as an
        # area, not a scatter of red dots; translucent so the imprint shows.
        xs = np.unique(field.nodes[:, 0])
        half = int(round((np.median(np.diff(xs)) if len(xs) > 1 else 26) / 2))
        layer = vis.copy()
        for (x, y) in field.nodes[slip_nodes]:
            cv2.rectangle(layer, (int(x) - half, int(y) - half),
                          (int(x) + half, int(y) + half), SLIP_BGR, -1)
        vis = cv2.addWeighted(layer, 0.35, vis, 0.65, 0)

    if filled_mask is None:
        filled_mask = np.zeros(len(field.nodes), bool)
    for i, ((x, y), (dx, dy), c) in enumerate(
            zip(field.nodes, field.vecs, field.conf)):
        if c < min_conf:
            cv2.circle(vis, (int(x), int(y)), 1, REJECTED_BGR, -1)
            continue
        ax, ay = dx * scale, dy * scale
        if cap:                                  # clamp only the drawn length
            L = np.hypot(ax, ay)
            if L > cap:
                ax, ay = ax * cap / L, ay * cap / L
        tip = (int(x + ax), int(y + ay))
        colour = FILLED_BGR if filled_mask[i] else _color(c)
        if slip_nodes is not None and slip_nodes[i]:
            colour = SLIP_BGR
        cv2.arrowedLine(vis, (int(x), int(y)), tip, colour, 1,
                        tipLength=0.35, line_type=cv2.LINE_AA)
        if cone_nodes is not None and cone_nodes[i]:
            cv2.circle(vis, (int(x), int(y)), 4, CONE_BGR, 1, cv2.LINE_AA)

    # Metrics go on a banner strip BELOW the frame, never over the gel view.
    mx, my = field.mean()
    curl = field.curl() * 1000
    div = field.divergence() * 1000
    # Each number gets its physical meaning spelled out -- "rot" alone told a
    # viewer nothing about what was rotating or in which sense.
    banner = np.zeros((34, vis.shape[1], 3), np.uint8)
    F = cv2.FONT_HERSHEY_SIMPLEX
    rows = [(f"slide ({mx:+.2f},{my:+.2f})px", (0, 255, 255)),
            (f"twist {curl:+.1f}", (255, 200, 0)),
            (f"spread {div:+.1f}", (0, 200, 120))]
    if slip is not None:
        rows.append((f"slip {slip:.2f}px", (80, 200, 255)))
    if slip_stats is not None and slip_stats.get("n_contact", 0):
        rows.append((f"stick {100 * slip_stats['stick_ratio']:.0f}%"
                     f" ({slip_stats['n_slip']}/{slip_stats['n_contact']})",
                     SLIP_BGR))
        if "slid_px" in slip_stats:
            rows.append((f"slid {slip_stats['slid_px']:.1f}px", SLIP_BGR))
    x = 8
    for text, colour in rows:
        cv2.putText(banner, text, (x, 15), F, 0.42, colour, 1, cv2.LINE_AA)
        x += cv2.getTextSize(text, F, 0.42, 1)[0][0] + 16
    # Fit the legend to the frame instead of trusting one hardcoded size: at
    # 0.32 the long form is 512 px wide and a 494 px frame silently clipped
    # "slip=non-affine residual" off the right edge of every rendered video.
    legends = ["slide=mean tangential shift  twist=rotation x1e3  "
               "spread=push-out x1e3  slip=non-affine residual  "
               "stick=imprint moving with the gel (red cells slide over it)",
               "slide=mean tangential shift  twist=rotation x1e3  "
               "spread=push-out x1e3  slip=non-affine residual",
               "slide=mean shift  twist=rotation x1e3  "
               "spread=push-out x1e3  slip=non-affine",
               "slide=shift  twist=rotation  spread=push-out  "
               "slip=non-affine  (x1e3)"]
    avail = vis.shape[1] - 16
    for legend in legends:
        for fs in (0.32, 0.30, 0.28, 0.26):
            if cv2.getTextSize(legend, F, fs, 1)[0][0] <= avail:
                cv2.putText(banner, legend, (8, 29), F, fs, (150, 150, 150), 1,
                            cv2.LINE_AA)
                break
        else:
            continue
        break
    return np.vstack([vis, banner])


def render(path, out_mp4, method="flow", spacing=26, win=41, scale=None,
           min_conf=0.10, fps=None, undistort=False, keep_fov=100.0,
           out_size=480, calib=None, coherence=True, max_dev=1.5, fill=True,
           slip=True, panel_dir=None, variants=None, slip_thresh=None,
           mu=None, slip_area=True, slip_mode="temporal"):
    """``scale=None`` auto-sizes arrows so the 90th-pct one is ~0.9 grid cells.

    Real shear is only 1-3 px, so arrows must be magnified to be seen; auto
    scaling keeps them legible without the overlap that a fixed factor causes.
    """
    import cv2

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
        fov = np.ones(frames[0].shape, bool)     # edges already removed
    else:
        fov = shear.fov_mask(frames[0].shape)
    base = frames[0]

    # compute every field once, so auto-scale can see the magnitude spread.
    # Apply the GelSight-style clean-up: reject spring-incoherent nodes, then
    # fill the enclosed contact hole; keep the slip index per frame.
    fields, masks, fills, slips = [], [], [], []
    slip_nodes, slip_stats, cone_nodes, traction = [], [], [], []
    for i, g in enumerate(frames):
        if i == 0:
            continue
        cm = shear.contact_mask(g, base)
        masks.append(cm)
        fld = shear.shear_field(base, g, spacing=spacing, win=win,
                                method=method, fov=fov)
        if coherence:
            fld = fld.filter_coherence(min_conf, max_dev=max_dev)
        slips.append(fld.slip_index(min_conf) if slip else None)
        if slip_area and slip_mode == "temporal":
            # Slip AREA as the imprint moving relative to the gel between
            # consecutive frames (shear.relative_slip) -- the definition that
            # survives a soft object deforming without sliding.
            rs = shear.relative_slip(frames[i - 1], g, cm, spacing=spacing,
                                     win=win, fov=fov, min_conf=min_conf,
                                     thresh=slip_thresh, method=method)
            slip_nodes.append(rs["slip"])
            slid = (slip_stats[-1]["slid_px"] if slip_stats and slip_stats[-1]
                    else 0.0) + (rs["rel_speed"] if rs["sliding"] else 0.0)
            slip_stats.append({k: rs[k] for k in ("n_contact", "n_slip",
                                                  "slip_frac", "stick_ratio",
                                                  "slip_area_px", "rel_speed",
                                                  "sliding", "noise")}
                              | {"slid_px": slid})
        elif slip_area:
            # Spatial alternative: non-affine residual on the contact patch.
            sm, _ = fld.slip_mask(min_conf, slip_thresh, contact=cm)
            slip_nodes.append(sm)
            slip_stats.append(fld.slip_area(min_conf, slip_thresh, contact=cm))
        else:
            slip_nodes.append(None); slip_stats.append(None)
        if slip_area:
            # Normal-load proxy: darkening of the gel (baseline - frame).
            # Appended in BOTH slip modes: the panel loop zips these lists
            # with the fields, and a short list silently drops every panel.
            pressure = np.clip(base.astype(np.int16) - g.astype(np.int16),
                               0, 255).astype(np.float32)
            ratio = fld.traction_ratio(pressure, min_conf, win=win)
            finite = ratio[np.isfinite(ratio)]
            traction.append(float(np.percentile(finite, 90))
                            if finite.size else float("nan"))
            cone_nodes.append(np.isfinite(ratio) & (ratio > mu)
                              if mu is not None else None)
        else:
            cone_nodes.append(None); traction.append(None)
        if fill:
            fld, fmask = fld.fill(min_conf)
        else:
            fmask = np.zeros(len(fld.nodes), bool)
        fields.append(fld)
        fills.append(fmask)
    if scale is None:
        mags = np.concatenate([np.hypot(*f.vecs[f.conf >= min_conf].T)
                               for f in fields
                               if (f.conf >= min_conf).any()] or [np.array([1.0])])
        p90 = float(np.percentile(mags, 90)) if mags.size else 1.0
        scale = (0.9 * spacing) / max(p90, 0.3)
        print(f"  auto arrow scale x{scale:.1f} (p90 shear {p90:.2f} px)")
    cap_len = 1.3 * spacing                        # no arrow spans >1.3 cells

    # The dense flow is the whole cost of this function and it does not depend
    # on how the background is drawn, so every background variant is rendered
    # from the SAME computed fields rather than by running the pipeline again.
    if variants is None:
        variants = [{"name": "default", "out": out_mp4,
                     "denoise": True, "fade": 0.55}]
    # Panels are kept for the FIRST variant only: they feed the GIF and the
    # peak-contact still, which should match the video a caller treats as primary.
    keep = panel_dir is not None
    written = []
    for i, v in enumerate(variants):
        first = i == 0
        tmp = (panel_dir if (keep and first)
               else tempfile.mkdtemp(prefix="shear_field_"))
        os.makedirs(tmp, exist_ok=True)
        try:
            series = (fields, masks, fills, slips, slip_nodes, slip_stats,
                      cone_nodes)
            assert len({len(s) for s in series}) == 1, \
                [len(s) for s in series]          # one entry per rendered frame
            for k, (field, cm, fmask, sl, sn, ss, cn) in enumerate(
                    zip(*series), start=1):
                panel = _draw_field(frames[k], field, cm, scale, min_conf,
                                    cap=cap_len, filled_mask=fmask, slip=sl,
                                    bg_denoise=v.get("denoise", True),
                                    bg_fade=v.get("fade", 0.55),
                                    slip_nodes=sn, slip_stats=ss,
                                    cone_nodes=cn)
                cv2.imwrite(os.path.join(tmp, f"{k:05d}.png"), panel)
            os.makedirs(os.path.dirname(v["out"]) or ".", exist_ok=True)
            _encode(tmp, v["out"], fps)
            written.append({"name": v.get("name", "default"), "file": v["out"],
                            "denoise": bool(v.get("denoise", True)),
                            "fade": float(v.get("fade", 0.55))})
            print(f"  [{v.get('name','default')}] -> {v['out']}")
        finally:
            if not (keep and first):
                shutil.rmtree(tmp, ignore_errors=True)
    return {
        "video": variants[0]["out"],
        "variants": written,
        "fps": float(fps),
        "panels": tmp if keep else None,
        "arrow_scale": float(scale),
        "n_panels": len(fields),
        "contact_area": [float(m.mean()) for m in masks],
        "slip": [None if s_ is None else float(s_) for s_ in slips],
        "shear_px": [[float(v) for v in f.mean()] for f in fields],
        "stick_ratio": [None if s is None else s["stick_ratio"]
                        for s in slip_stats],
        "slip_frac": [None if s is None else s["slip_frac"]
                      for s in slip_stats],
        "annularity": [None if s is None else s.get("annularity")
                       for s in slip_stats],
        "slid_px": [None if s is None else s.get("slid_px")
                    for s in slip_stats],
        "rel_speed": [None if s is None else s.get("rel_speed")
                      for s in slip_stats],
        "slip_mode": slip_mode,
        "traction_p90": traction,
    }


def _encode(frame_dir, out_mp4, fps):
    import cv2

    pngs = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if not pngs:
        raise RuntimeError("no frames")
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-framerate", f"{fps:.3f}",
               "-pattern_type", "glob", "-i", os.path.join(frame_dir, "*.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", out_mp4]
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
    ap.add_argument("--method", default="flow",
                    choices=["phasecorr", "lk", "flow"])
    ap.add_argument("--spacing", type=int, default=26)
    ap.add_argument("--win", type=int, default=41)
    ap.add_argument("--scale", type=float, default=None,
                    help="arrow length px per px of shear "
                         "(default: auto, ~0.9 grid cell at the 90th pct)")
    ap.add_argument("--min-conf", type=float, default=0.10,
                    help="node confidence gate; 0.10 is the live pipeline "
                         "page's evidence-based default")
    ap.add_argument("--undistort", action="store_true",
                    help="calibration-free 180-deg correction + centre crop")
    ap.add_argument("--calib", nargs="?", default=None,
                    const="__default180__",
                    help="use a saved OpenCV-fisheye calibration yaml (metric); "
                         "bare --calib uses the repo's glowtact180 calibration")
    ap.add_argument("--keep-fov", type=float, default=100.0,
                    help="output field-of-view in deg when --undistort (smaller "
                         "= tighter centre crop)")
    ap.add_argument("--out-size", type=int, default=480)
    ap.add_argument("--no-coherence", action="store_true",
                    help="disable neighbour-coherence outlier rejection")
    ap.add_argument("--max-dev", type=float, default=1.5,
                    help="reject a node whose flow deviates > this (px) from "
                         "its neighbour median")
    ap.add_argument("--no-fill", action="store_true",
                    help="do not interpolate across the contact hole")
    ap.add_argument("--no-slip", action="store_true",
                    help="do not overlay the slip index")
    ap.add_argument("--no-slip-area", action="store_true",
                    help="do not shade the slipping region / stick ratio")
    ap.add_argument("--slip-thresh", type=float, default=None,
                    help="non-affine residual (px) that flags a node as "
                         "slipping (default ShearField.SLIP_RESIDUAL_PX)")
    ap.add_argument("--slip-mode", choices=["temporal", "spatial"],
                    default="temporal",
                    help="temporal: imprint moving relative to the gel "
                         "between frames (default); spatial: non-affine "
                         "residual on the contact patch")
    ap.add_argument("--mu", type=float, default=None,
                    help="friction-cone threshold on |shear|/darkening; ring "
                         "the nodes above it (needs calibrating per sensor)")
    args = ap.parse_args(argv)
    if args.calib == "__default180__":
        args.calib = fisheye.DEFAULT_CALIB_180

    if args.videos:
        videos, out_dir = args.videos, (args.out or "shear_field")
    elif args.session:
        videos = sorted(glob.glob(os.path.join(args.session, "videos",
                                               "*.avi")))
        out_dir = args.out or os.path.join(args.session, "shear_field")
    else:
        ap.error("give a session dir or --videos")
    os.makedirs(out_dir, exist_ok=True)
    for v in videos:
        name = os.path.splitext(os.path.basename(v))[0]
        out = os.path.join(out_dir, f"{name}_field.mp4")
        mode = "calib" if args.calib else ("undistort" if args.undistort
                                           else "raw")
        print(f"[{name}] rendering field ({args.method}, {mode}) -> {out}",
              flush=True)
        render(v, out, method=args.method, spacing=args.spacing, win=args.win,
               scale=args.scale, min_conf=args.min_conf,
               undistort=args.undistort, keep_fov=args.keep_fov,
               out_size=args.out_size, calib=args.calib,
               coherence=not args.no_coherence, max_dev=args.max_dev,
               fill=not args.no_fill, slip=not args.no_slip,
               slip_area=not args.no_slip_area, slip_thresh=args.slip_thresh,
               mu=args.mu, slip_mode=args.slip_mode)
    print(f"outputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
