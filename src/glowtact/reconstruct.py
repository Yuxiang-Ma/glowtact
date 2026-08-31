"""Vanilla 9DTact-style 3D reconstruction for the GlowTact sensors.

The principle (verified on this sensor's own ball presses: 6/6 impressions show
monotonic radial darkness decay, ~84 grey levels of range): pixel DARKENING
relative to an untouched reference maps monotonically to indentation depth, so
one global lookup table turns a darkness image into a depth map. No photometric
stereo, no network.

Pipeline, after 9DTact's ``_2_Sensor_Calibration.py`` / ``sensor.py``:

* ``calibrate`` -- consume ball-array press bundles (the ``undistort_debug``
  dumps: frame + untouched baseline + detected ball centres). Every pixel
  inside a contact circle gets a geometric ground-truth depth from the sphere::

      depth(d) = sqrt(R^2 - d_mm^2) - sqrt(R^2 - r_contact_mm^2)

  and is paired with its darkening (ref_gray - img_gray). Averaging depth per
  grey level gives the ``pixel_to_depth`` table. One array press contributes 54
  known spheres, where 9DTact presses a single ball repeatedly.

* ``depth`` -- offline: darkening -> LUT -> depth map (mm), Gaussian-smoothed,
  written as a colormapped PNG + .npy + optional .ply point cloud.

* ``live`` -- the same against a live camera, with ``b`` re-taking the
  reference and ``s`` saving a depth snapshot.

Geometry is deliberately "vanilla": no undistortion. The fisheye makes the
mm-per-pixel scale position-dependent, so the calibration derives a LOCAL scale
per ball from its own grid-neighbour spacing rather than assuming one global
value. Cropping the vignette corners is supported at runtime via a
crop_bounds.yaml from crop_gui.py; calibration needs no crop because it only
ever samples pixels inside detected contact circles.

    glowtact-reconstruct calibrate undistort_debug \\
        --ball-radius-mm 2.35 --spacing-mm 7.4
    glowtact-reconstruct depth IMG --ref REF --calib glowtact_depth_calib.npz
    glowtact-reconstruct live --camera 1bcf:28c4 --calib ...

``--camera`` is a /dev/video index or USB ``vendor:product`` id; a host that
registers its own opener (glowtact.capture.register_camera_opener) may accept
its own camera keys instead.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import yaml

from .depth import (  # noqa: F401
    DEFAULT_KERNELS, DEFAULT_LIGHTING_THRESHOLD, build_lut, colorize_depth,
    darkening, depth_from_diff, load_calibration, save_calibration)
from .render import (  # noqa: F401
    ReliefRenderer, _cloud_colors, apply_o3d_view_preset, bake_shading,
    height_grid_triangles, height_to_cloud, height_to_o3d_mesh, light_vector,
    render_turntable, save_gif, save_ply, surface_shading)

DEFAULT_CALIB = "glowtact_depth_calib.npz"
CONTACT_EDGE_THRESHOLD = 6.0        # radial profile below this = contact edge


# ---------------------------------------------------------------- geometry --

def grid_local_scale(points, rows, cols):
    """Pixel distance to grid neighbours, per ball -- a LOCAL px-per-unit scale.

    Under the fisheye a single global mm-per-pixel is wrong by design: the
    periphery is compressed ~3x relative to the centre. Each ball's own
    neighbour spacing captures the scale where that ball actually sits.
    Returns an (rows, cols) array of mean neighbour distance in pixels.
    """
    grid = np.asarray(points, dtype=np.float64).reshape(rows, cols, 2)
    scale = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            dists = []
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    dists.append(np.linalg.norm(grid[r, c] - grid[rr, cc]))
            scale[r, c] = float(np.mean(dists))
    return scale


def radial_profile(diff, centre, max_radius, bin_px=1.0):
    """Mean darkening per radial bin around `centre`."""
    height, width = diff.shape
    x0 = max(0, int(centre[0] - max_radius)); x1 = min(width, int(centre[0] + max_radius) + 1)
    y0 = max(0, int(centre[1] - max_radius)); y1 = min(height, int(centre[1] + max_radius) + 1)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(xs - centre[0], ys - centre[1])
    values = diff[y0:y1, x0:x1]
    edges = np.arange(0, max_radius + bin_px, bin_px)
    profile = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if m.any():
            profile[i] = float(values[m].mean())
    return profile


def contact_radius_px(diff, centre, max_radius,
                      threshold=CONTACT_EDGE_THRESHOLD):
    """Radius where the darkening falls to the noise floor.

    Scanned OUTWARD from the centre so a neighbouring ball's dark rim beyond
    the gap cannot be mistaken for this contact. None if the centre itself is
    at noise level (a ball that never touched).
    """
    profile = radial_profile(diff, centre, max_radius)
    if np.isnan(profile[0]) or profile[0] < threshold:
        return None
    for i, value in enumerate(profile):
        if np.isnan(value) or value < threshold:
            return float(i)          # first bin at/below noise
    return float(len(profile))


def sphere_depth_map(shape, centre, contact_px, ball_radius_mm, mm_per_px):
    """Ground-truth indentation depth (mm) for one spherical contact.

    depth(d) = sqrt(R^2 - d^2) - sqrt(R^2 - r_c^2), inside the contact circle.
    Returns (depth_map, valid_mask). The contact radius is clamped just under R
    -- a detected radius beyond the ball's own radius is a detection artefact,
    not physics.
    """
    r_c_mm = min(contact_px * mm_per_px, 0.98 * ball_radius_mm)
    height, width = shape
    x0 = max(0, int(centre[0] - contact_px)); x1 = min(width, int(centre[0] + contact_px) + 1)
    y0 = max(0, int(centre[1] - contact_px)); y1 = min(height, int(centre[1] + contact_px) + 1)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d_mm = np.hypot(xs - centre[0], ys - centre[1]) * mm_per_px
    inside = d_mm <= r_c_mm
    depth = np.zeros(shape)
    valid = np.zeros(shape, dtype=bool)
    depth_local = (np.sqrt(np.maximum(ball_radius_mm ** 2 - d_mm ** 2, 0.0))
                   - np.sqrt(ball_radius_mm ** 2 - r_c_mm ** 2))
    depth[y0:y1, x0:x1] = np.where(inside, depth_local, 0.0)
    valid[y0:y1, x0:x1] = inside
    return depth, valid


# --------------------------------------------------------------- calibrate --

def calibrate_from_bundles(folder, ball_radius_mm, spacing_mm,
                           lighting_threshold=DEFAULT_LIGHTING_THRESHOLD,
                           min_center_darkening=20):
    """Build the LUT from every usable press bundle in `folder`.

    Balls whose centre never darkened past `min_center_darkening` are skipped:
    they did not really touch (the array is never perfectly parallel), and
    pairing their noise with a geometric depth would poison the table.
    """
    import cv2

    from .undistort import canonicalize_grid

    gray_samples, depth_samples = [], []
    used_bundles = balls_used = balls_skipped = 0
    max_depth_seen = 0.0

    for meta_path in sorted(glob.glob(os.path.join(folder, "*_meta.yaml"))):
        meta = yaml.safe_load(open(meta_path))
        if not meta.get("detected_points"):
            continue
        stem = meta_path[: -len("_meta.yaml")]
        frame = cv2.imread(stem + "_frame.png")
        ref = cv2.imread(stem + "_baseline.png")
        if frame is None or ref is None:
            continue
        rows, cols = meta["rows"], meta["cols"]
        points = canonicalize_grid(np.array(meta["detected_points"]), rows, cols)
        diff = darkening(frame, ref)
        scale_px = grid_local_scale(points, rows, cols)     # px per spacing_mm
        used_bundles += 1

        grid = points.reshape(rows, cols, 2)
        for r in range(rows):
            for c in range(cols):
                centre = grid[r, c]
                mm_per_px = spacing_mm / scale_px[r, c]
                # Stay inside this ball's own cell so a deep neighbour's rim
                # is never sampled as part of this contact.
                max_radius = 0.45 * scale_px[r, c]
                contact = contact_radius_px(diff, centre, max_radius)
                if contact is None or diff[int(centre[1]), int(centre[0])] < \
                        min_center_darkening:
                    balls_skipped += 1
                    continue
                depth_map, valid = sphere_depth_map(
                    diff.shape, centre, contact, ball_radius_mm, mm_per_px)
                mask = valid & (diff >= lighting_threshold)
                gray_samples.append(diff[mask])
                depth_samples.append(depth_map[mask])
                max_depth_seen = max(max_depth_seen, float(depth_map.max()))
                balls_used += 1

    if not gray_samples:
        raise SystemExit("no usable ball presses found -- check the folder")
    gray = np.concatenate(gray_samples)
    depth = np.concatenate(depth_samples)
    lut, counts = build_lut(gray, depth, lighting_threshold)
    stats = {
        "bundles": used_bundles,
        "balls_used": balls_used,
        "balls_skipped": balls_skipped,
        "samples": int(len(gray)),
        "max_gray": int(gray.max()),
        "max_depth_mm": round(max_depth_seen, 3),
        "ball_radius_mm": ball_radius_mm,
        "spacing_mm": spacing_mm,
        "lighting_threshold": lighting_threshold,
    }
    return lut, counts, stats


# ------------------------------------------------------------------ output --


def _apply_crop(image, crop):
    """Rotation-then-crop from a crop_gui bounds file (see glowtact.crop)."""
    if crop is None:
        return image
    bounds, _w, _h, rotation = crop
    from .crop import rotate_image
    x0, y0, x1, y1 = bounds
    return rotate_image(image, rotation)[y0:y1, x0:x1]


# -------------------------------------------------------------------- CLIs --

def cmd_calibrate(args):
    lut, counts, stats = calibrate_from_bundles(
        args.folder, args.ball_radius_mm, args.spacing_mm,
        args.lighting_threshold)
    save_calibration(args.out, lut, counts, stats)
    print(json.dumps(stats, indent=2))
    nonzero = np.flatnonzero(lut)
    increases = np.diff(lut[stats["lighting_threshold"]:stats["max_gray"]])
    print(f"LUT: depth {lut.max():.3f} mm at gray {int(np.argmax(lut))}; "
          f"monotonic rises {100 * (increases >= 0).mean():.0f}% of steps")
    print(f"saved -> {args.out}")

    # A picture of the mapping is the fastest sanity check there is.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot(lut); ax[0].set_xlabel("darkening (gray)")
        ax[0].set_ylabel("depth (mm)"); ax[0].set_title("pixel_to_depth")
        ax[1].semilogy(np.maximum(counts, 1)); ax[1].set_xlabel("darkening (gray)")
        ax[1].set_ylabel("samples"); ax[1].set_title("calibration coverage")
        fig.tight_layout()
        plot = os.path.splitext(args.out)[0] + ".png"
        fig.savefig(plot, dpi=110)
        print(f"curve  -> {plot}")
    except Exception as exc:                        # pragma: no cover
        print(f"[warn] could not plot the curve: {exc}", file=sys.stderr)
    return 0


def cmd_depth(args):
    import cv2

    lut, stats = load_calibration(args.calib)
    crop = None
    if args.crop_bounds:
        from .crop import bounds_from_file
        crop = bounds_from_file(args.crop_bounds)

    ref = None
    if not args.from_diff:
        if not args.ref:
            raise SystemExit("--ref is required unless --from-diff is used")
        ref = cv2.imread(args.ref)
        if ref is None:
            raise SystemExit(f"cannot read reference {args.ref}")
        ref = _apply_crop(ref, crop)

    pattern = args.pattern or "*.png"
    targets = ([args.image] if os.path.isfile(args.image) else
               sorted(glob.glob(os.path.join(args.image, pattern))))
    if not targets:
        raise SystemExit(f"nothing matches {pattern} in {args.image}")
    os.makedirs(args.out, exist_ok=True)
    # PLY scale: the calibration knows its own mm-per-pixel; a CLI value only
    # overrides it for data captured at a different scale.
    nominal_scale = (stats["spacing_mm"] / args.nominal_spacing_px
                     if args.nominal_spacing_px
                     else stats.get("mm_per_px", 0.05))
    for path in targets:
        img = cv2.imread(path)
        if img is None:
            continue
        img = _apply_crop(img, crop)
        if args.from_diff:
            # The input already IS a diff map -- no reference needed.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
            if args.diff_mode == "signed":
                # 128-centred (frame - baseline + 128): contact reads DARKER
                # than the untouched baseline, so darkening = 128 - gray.
                # This is the format manual_collect records as diff_mode
                # `signed` (e.g. the fingerprint captures).
                diff = np.clip(128.0 - gray, 0, 255)
            else:
                # Darkening map already (background ~0, contact bright), e.g.
                # the ball-array pad *_diff.png.
                diff = gray
        else:
            if img.shape != ref.shape:
                print(f"[warn] {os.path.basename(path)}: size differs from "
                      f"the reference -- skipped")
                continue
            diff = darkening(img, ref)
        height = depth_from_diff(diff, lut,
                                 stats["lighting_threshold"], args.kernels,
                                 median=args.median,
                                 min_component_px=args.min_component)
        stem = os.path.join(args.out,
                            os.path.splitext(os.path.basename(path))[0])
        np.save(stem + "_depth.npy", height)
        cv2.imwrite(stem + "_depth.png",
                    colorize_depth(height, stats["max_depth_mm"]))
        line = (f"{os.path.basename(path)}: max {height.max():.3f} mm, "
                f"contact {100 * (height > 0).mean():.1f}% of pixels")
        if args.ply:
            n = save_ply(stem + ".ply", height, nominal_scale)
            line += f", {n} pts -> {os.path.basename(stem)}.ply"
        print(line)
    return 0


def cmd_live(args):
    import cv2

    from .capture import open_live

    lut, stats = load_calibration(args.calib)
    crop = None
    if args.crop_bounds:
        from .crop import bounds_from_file
        crop = bounds_from_file(args.crop_bounds)

    live = open_live(args.camera, args.config, buffer=4, fresh_reads=1)
    camera_id, reader = live.camera_id, live.reader

    window = "reconstruct_3d"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    # The live 3D relief updates every frame, exactly 9DTact's run loop
    # (update points -> poll events); dragging the window orbits it.
    renderer = ReliefRenderer(width=640, height=480)
    renderer.attach_mouse(window)
    mm_per_px = stats.get("mm_per_px", 1.0)
    print(json.dumps({"camera": camera_id, "calib": args.calib,
                      "controls": {"b": "re-take reference (sensor untouched)",
                                   "drag": "orbit the 3D view",
                                   "s": "save depth snapshot", "q": "quit"}}))
    reference = None
    saved = 0
    os.makedirs(args.out, exist_ok=True)
    try:
        last_seq = -1
        while True:
            item = reader.wait_for_new(last_seq, timeout=1.0)
            if item is None:
                continue
            last_seq, frame, _tm, _te = item
            frame = _apply_crop(frame, crop)
            if reference is None:
                reference = frame.copy()
            height = depth_from_diff(darkening(frame, reference), lut,
                                     stats["lighting_threshold"], args.kernels,
                                     median=args.median,
                                     min_component_px=args.min_component)
            depth_view = colorize_depth(height, stats["max_depth_mm"])
            cv2.putText(depth_view, f"max {height.max():.2f} mm", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                        cv2.LINE_AA)
            points, shade = height_to_cloud(height, mm_per_px, stride=3,
                                            z_scale=3.0)
            relief = renderer.render(points, shade, label="3D (drag to orbit)")
            panels = [frame, depth_view,
                      cv2.resize(relief, (frame.shape[1], frame.shape[0]))]
            cv2.imshow(window, np.hstack(panels))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):
                reference = frame.copy()
                print(json.dumps({"event": "reference_set"}))
            elif key == ord("s"):
                saved += 1
                stem = os.path.join(args.out, f"depth_{saved:04d}")
                np.save(stem + ".npy", height)
                cv2.imwrite(stem + ".png", depth_view)
                cv2.imwrite(stem + "_frame.png", frame)
                print(json.dumps({"event": "saved", "stem": stem,
                                  "max_depth_mm": round(float(height.max()), 3)}))
    finally:
        try:
            cv2.destroyWindow(window)
        except Exception:
            pass
        live.close()
    return 0


# ---------------------------------------------- 9DTact-style visualisation --


def _view_open3d(files, mm_per_px, args):
    """The genuine 9DTact viewer: an Open3D window updated in place.

    Mirrors their Visualizer: one PointCloud whose points/colours are swapped
    per press of n/p, an oblique preset view, orbit/zoom native to Open3D.
    """
    import open3d as o3d

    state = {"idx": args.start % len(files), "mode": args.render,
             "z_scale": args.z_scale, "ambient": args.ambient,
             "azimuth": args.light_azimuth, "elevation": args.light_elevation}
    mesh = o3d.geometry.TriangleMesh()
    pcd = o3d.geometry.PointCloud()

    def refresh(vis=None):
        height = np.load(files[state["idx"]])
        if state["mode"] == "mesh":
            built = height_to_o3d_mesh(height, mm_per_px, args.stride,
                                       state["z_scale"],
                                       light_azimuth_deg=state["azimuth"],
                                       light_elevation_deg=state["elevation"],
                                       ambient=state["ambient"])
            mesh.vertices = built.vertices
            mesh.triangles = built.triangles
            mesh.vertex_colors = built.vertex_colors
            mesh.vertex_normals = built.vertex_normals
        else:
            points, colors = _cloud_colors(height, mm_per_px, args.stride,
                                           state["z_scale"], "depth")
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        print(f"[{state['idx'] + 1}/{len(files)}] "
              f"{os.path.basename(files[state['idx']])}  "
              f"max {height.max():.3f} mm  ({state['mode']}, "
              f"z x{state['z_scale']:g}, ambient {state['ambient']:.2f}, "
              f"light az {state['azimuth']:.0f} el {state['elevation']:.0f})")
        if vis is not None:
            vis.update_geometry(mesh if state["mode"] == "mesh" else pcd)
        return False

    def step(delta):
        def cb(vis):
            state["idx"] = (state["idx"] + delta) % len(files)
            return refresh(vis)
        return cb

    def toggle(vis):
        old = mesh if state["mode"] == "mesh" else pcd
        state["mode"] = "points" if state["mode"] == "mesh" else "mesh"
        refresh()
        vis.remove_geometry(old, reset_bounding_box=False)
        vis.add_geometry(mesh if state["mode"] == "mesh" else pcd,
                         reset_bounding_box=False)
        _style(vis)
        return False

    def _style(vis):
        option = vis.get_render_option()
        if state["mode"] == "mesh":
            # The published figures: matte grey clay on a light background.
            # Lighting is BAKED into the vertex colours (see height_to_o3d_mesh)
            # -- the viewer's own headlight would wash the relief out.
            option.background_color = np.array([0.92, 0.92, 0.92])
            option.light_on = False
            option.mesh_show_back_face = True
        else:
            option.background_color = np.array([0.05, 0.05, 0.05])
            option.point_size = 2.0

    refresh()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("GlowTact-Shape_Reconstruction", width=1280, height=960)
    vis.add_geometry(mesh if state["mode"] == "mesh" else pcd)
    _style(vis)
    ctr = vis.get_view_control()
    ctr.set_lookat(mesh.get_center() if state["mode"] == "mesh"
                   else pcd.get_center())
    ctr.set_front([0.0, 0.55, 0.84])      # oblique from above (+z is depth)
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_zoom(0.7)
    def tune(field, factor=None, delta=None, lo=None, hi=None):
        def cb(vis):
            value = state[field] * factor if factor else state[field] + delta
            if lo is not None:
                value = max(lo, value)
            if hi is not None:
                value = min(hi, value)
            state[field] = value
            return refresh(vis)
        return cb

    for key, cb in ((ord("N"), step(+1)), (262, step(+1)),     # n / right
                    (ord("P"), step(-1)), (263, step(-1)),     # p / left
                    (ord("M"), toggle),
                    (93, tune("z_scale", factor=1.25, hi=20)),   # ]
                    (91, tune("z_scale", factor=0.8, lo=0.5)),   # [
                    (46, tune("azimuth", delta=30)),             # .
                    (44, tune("azimuth", delta=-30)),            # ,
                    (48, tune("ambient", delta=0.1, hi=1.0)),    # 0
                    (57, tune("ambient", delta=-0.1, lo=0.0)),   # 9
                    (39, tune("elevation", delta=10, hi=85)),    # '
                    (59, tune("elevation", delta=-10, lo=10))):  # ;
        vis.register_key_callback(key, cb)
    print("keys: n/p=switch  m=mesh/points  [ ]=z-scale  , .=light azimuth  "
          "; '=light elevation  9 0=ambient  mouse=orbit  q=quit")

    if args.screenshot:
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(args.screenshot, do_render=True)
        vis.destroy_window()
        print(f"screenshot -> {args.screenshot}")
        return 0
    vis.run()
    vis.destroy_window()
    return 0


def cmd_view(args):
    """Step through reconstructions in the 9DTact style.

    Uses the real Open3D viewer when the package is importable (the `open3d`
    conda env), and the built-in OpenCV relief renderer everywhere else --
    same shading, same data, different window toolkit.
    """
    import cv2

    files = sorted(glob.glob(os.path.join(args.folder, "*_depth.npy")))
    if not files:
        raise SystemExit(f"no *_depth.npy in {args.folder} -- run `depth` first")

    mm_per_px = args.mm_per_px
    if mm_per_px is None and args.calib and os.path.exists(args.calib):
        _lut, stats = load_calibration(args.calib)
        mm_per_px = stats.get("mm_per_px")
    if mm_per_px is None:
        mm_per_px = 1.0
        print("[warn] no mm-per-px known -- x/y are in PIXEL units")

    if args.backend != "cv":
        try:
            import open3d                              # noqa: F401
            return _view_open3d(files, mm_per_px, args)
        except ImportError:
            if args.backend == "open3d":
                raise SystemExit(
                    "open3d is not importable here. Run inside the open3d "
                    "conda env:\n  conda run -n open3d python "
                    "glowtact-reconstruct view ...")
            # auto: fall through to the OpenCV relief renderer

    renderer = ReliefRenderer()
    state = {"idx": args.start % len(files), "mode": "relief"}

    def load(index):
        height = np.load(files[index])
        points, shade = height_to_cloud(height, mm_per_px, args.stride,
                                        args.z_scale)
        depth = height[::args.stride, ::args.stride].ravel()
        return height, points, shade, depth

    height, points, shade, depth = load(state["idx"])
    if args.screenshot:
        frame = renderer.render(points, shade, depth, state["mode"],
                                label=os.path.basename(files[state["idx"]]))
        cv2.imwrite(args.screenshot, frame)
        print(f"screenshot -> {args.screenshot}")
        return 0

    window = "reconstruct_3d view"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    renderer.attach_mouse(window)
    while True:
        name = os.path.basename(files[state["idx"]])
        label = (f"[{state['idx'] + 1}/{len(files)}] {name}   "
                 f"max {height.max():.3f} mm   z x{args.z_scale:g}   "
                 f"[n/p=switch drag=orbit +/-=zoom c=colour r=reset q=quit]")
        cv2.imshow(window, renderer.render(points, shade, depth,
                                           state["mode"], label))
        key = cv2.waitKey(16) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("n"), 83):
            state["idx"] = (state["idx"] + 1) % len(files)
            height, points, shade, depth = load(state["idx"])
        elif key in (ord("p"), 81):
            state["idx"] = (state["idx"] - 1) % len(files)
            height, points, shade, depth = load(state["idx"])
        elif key == ord("c"):
            state["mode"] = "depth" if state["mode"] == "relief" else "relief"
        elif key == ord("r"):
            renderer.reset()
        elif key in (ord("+"), ord("=")):
            renderer.zoom = min(renderer.zoom * 1.2, 8.0)
        elif key == ord("-"):
            renderer.zoom = max(renderer.zoom / 1.2, 0.2)
    cv2.destroyWindow(window)
    return 0


def cmd_spin(args):
    """Render each reconstruction as a 360 deg rotating GIF (needs Open3D)."""
    try:
        import open3d                                   # noqa: F401
    except ImportError:
        raise SystemExit(
            "spin needs Open3D -- run inside the `open3d` conda env, e.g. "
            "`conda run -n open3d glowtact-reconstruct spin ...`")
    from .figure import (
        build_mesh, center_depth_contact, content_bbox, union_bbox)

    if args.folder.endswith("_depth.npy"):
        files = [args.folder]
    else:
        files = sorted(glob.glob(os.path.join(args.folder, "*_depth.npy")))
    if not files:
        raise SystemExit(f"no *_depth.npy in {args.folder} -- run `depth` first")

    mm_per_px = args.mm_per_px
    if mm_per_px is None and args.calib and os.path.exists(args.calib):
        _lut, stats = load_calibration(args.calib)
        mm_per_px = stats.get("mm_per_px")
    if mm_per_px is None:
        mm_per_px = 1.0
        print("[warn] no mm-per-px known -- x/y are in PIXEL units")

    os.makedirs(args.out, exist_ok=True)
    for path in files:
        height = np.load(path)
        if args.center:
            height = center_depth_contact(height, target_x=args.center_x,
                                          target_y=args.center_y)
        if args.figure_look:
            # Same mesh the published stills are built from, so a turntable and
            # the paper figure of one press are the same object under the same
            # light -- only the camera moves.
            mesh = build_mesh(height, mm_per_px, smooth_px=args.smooth,
                              shade_contrast=args.shade_contrast,
                              shade_smooth=args.shade_smooth,
                              z_scale=args.z_scale, stride=args.stride)
        else:
            mesh = height_to_o3d_mesh(
                height, mm_per_px, args.stride, args.z_scale,
                light_azimuth_deg=args.light_azimuth,
                light_elevation_deg=args.light_elevation, ambient=args.ambient)
        frames = render_turntable(mesh, n_frames=args.frames,
                                  width=args.size, height=args.size,
                                  zoom=args.zoom, bg=args.bg,
                                  front=args.front, up=[0.0, -1.0, 0.0])
        if args.crop:
            threshold = int(round(args.bg * 255)) - 3
            box = union_bbox(content_bbox(f, margin=args.crop_margin,
                                          threshold=threshold)
                             for f in frames)
            x0, y0, x1, y1 = box
            frames = [f[y0:y1, x0:x1] for f in frames]
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.endswith("_depth"):
            stem = stem[:-len("_depth")]
        out = os.path.join(args.out, stem + "_spin.gif")
        save_gif(out, frames, fps=args.fps, colors=args.colors)
        secs = args.frames / max(args.fps, 1)
        print(f"{os.path.basename(path)}  ->  {out}  "
              f"({args.frames} frames @ {args.fps} fps = {secs:.1f} s/turn, "
              f"{frames[0].shape[1]}x{frames[0].shape[0]}, "
              f"max {height.max():.2f} mm, "
              f"{os.path.getsize(out) / 1e6:.1f} MB)")
    return 0


def cmd_realtime(args):
    """Non-blocking live 3D mesh: capture/reconstruct/render on separate stages."""
    from .realtime import run_realtime
    return run_realtime(args)


def _kernels_arg(value):
    kernels = tuple(int(v) for v in str(value).split(",") if v.strip())
    if any(k < 1 or k % 2 == 0 for k in kernels):
        raise argparse.ArgumentTypeError("kernels must be odd and >= 1")
    return kernels


def _median_arg(value):
    k = int(value)
    if k != 0 and (k < 3 or k % 2 == 0):
        raise argparse.ArgumentTypeError("median must be 0 (off) or odd >= 3")
    return k


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="fit darkening->depth from ball presses")
    c.add_argument("folder", help="undistort_debug-style bundle folder")
    c.add_argument("--ball-radius-mm", type=float, required=True,
                   help="radius of the printed calibration balls")
    c.add_argument("--spacing-mm", type=float, required=True,
                   help="centre-to-centre ball spacing on the printed array")
    c.add_argument("--lighting-threshold", type=int,
                   default=DEFAULT_LIGHTING_THRESHOLD)
    c.add_argument("--out", default=DEFAULT_CALIB)
    c.set_defaults(func=cmd_calibrate)

    d = sub.add_parser("depth", help="offline image(s) -> depth map")
    d.add_argument("image", help="one image or a folder")
    d.add_argument("--ref", default=None, help="untouched reference image")
    d.add_argument("--from-diff", action="store_true",
                   help="inputs are already diff maps -- no reference needed")
    d.add_argument("--diff-mode", choices=["darkening", "signed"],
                   default="darkening",
                   help="how a --from-diff image encodes contact: 'darkening' "
                        "(background ~0, contact bright, the ball-pad format) "
                        "or 'signed' (128-centred frame-baseline+128, contact "
                        "darker; the fingerprint captures' diff_mode)")
    d.add_argument("--pattern", default=None,
                   help="glob within a folder, e.g. '*_diff.png'")
    d.add_argument("--calib", default=DEFAULT_CALIB)
    d.add_argument("--crop-bounds", default=None,
                   help="crop_bounds.yaml from crop_gui.py (cuts the corners)")
    d.add_argument("--kernels", type=_kernels_arg, default=DEFAULT_KERNELS)
    d.add_argument("--median", type=_median_arg, default=0,
                   help="median-filter the darkening image before the LUT "
                        "(odd kernel, e.g. 3 or 5) to kill hot-pixel spikes; "
                        "0 = off (9DTact's own pipeline has no median stage)")
    d.add_argument("--min-component", type=int, default=0,
                   help="zero above-threshold darkening islands smaller than "
                        "this many pixels (dust/blemish killer); 0 = off")
    d.add_argument("--ply", action="store_true", help="also write point clouds")
    d.add_argument("--nominal-spacing-px", type=float, default=None,
                   help="observed ball spacing in px, to scale the PLY x/y")
    d.add_argument("--spacing-mm", type=float, default=None, help=argparse.SUPPRESS)
    d.add_argument("--out", default="depth_out")
    d.set_defaults(func=cmd_depth)

    l = sub.add_parser("live", help="live depth view from a camera")
    l.add_argument("--camera", required=True,
                   help="/dev/video index, USB vendor:product id, or a host camera key")
    l.add_argument("-c", "--config", default=None,
                   help="host device config (only with a registered opener)")
    l.add_argument("--calib", default=DEFAULT_CALIB)
    l.add_argument("--crop-bounds", default=None)
    l.add_argument("--kernels", type=_kernels_arg, default=DEFAULT_KERNELS)
    l.add_argument("--median", type=_median_arg, default=0,
                   help="median-filter the darkening image before the LUT "
                        "(odd kernel); 0 = off")
    l.add_argument("--min-component", type=int, default=0,
                   help="zero above-threshold darkening islands smaller than "
                        "this many pixels; 0 = off")
    l.add_argument("--out", default="depth_out")
    l.set_defaults(func=cmd_live)

    v = sub.add_parser("view", help="step through *_depth.npy point clouds")
    v.add_argument("folder", help="folder with *_depth.npy (the `depth` output)")
    v.add_argument("--calib", default=DEFAULT_CALIB,
                   help="calibration file, used for the mm-per-px scale")
    v.add_argument("--mm-per-px", type=float, default=None,
                   help="override the x/y scale")
    v.add_argument("--stride", type=int, default=2,
                   help="subsample factor; 1 = every pixel (slower to rotate)")
    v.add_argument("--z-scale", type=float, default=3.0,
                   help="z exaggeration for visibility (stated in the title); "
                        "1.0 = true geometric scale")
    v.add_argument("--ambient", type=float, default=0.30,
                   help="baked-light ambient floor 0..1; lower = harder "
                        "relief contrast, 1.0 = flat unshaded grey")
    v.add_argument("--light-azimuth", type=float, default=218.0,
                   help="baked light direction in the surface plane, degrees")
    v.add_argument("--light-elevation", type=float, default=55.0,
                   help="baked light elevation, degrees; lower = longer "
                        "shadows / stronger relief")
    v.add_argument("--render", choices=["mesh", "points"], default="mesh",
                   help="mesh (default): dense normal-lit surface like the "
                        "9DTact paper figures; points: depth-coloured cloud")
    v.add_argument("--backend", choices=["auto", "open3d", "cv"],
                   default="auto",
                   help="auto: Open3D when importable (the `open3d` conda "
                        "env), else the built-in OpenCV relief renderer")
    v.add_argument("--start", type=int, default=0)
    v.add_argument("--screenshot", default=None,
                   help="render the first view to a PNG and exit (headless)")
    v.set_defaults(func=cmd_view)

    sp = sub.add_parser("spin",
                        help="render each *_depth.npy as a 360 deg rotating GIF")
    sp.add_argument("folder", help="folder of *_depth.npy, or a single .npy")
    sp.add_argument("--calib", default=DEFAULT_CALIB,
                    help="calibration file, for the mm-per-px scale")
    sp.add_argument("--mm-per-px", type=float, default=None,
                    help="override the x/y scale")
    sp.add_argument("--out", default="spin_gifs", help="output folder for GIFs")
    # 72 frames at 12 fps is one turn every 6 s. The earlier 36 @ 20 spun a
    # revolution in 1.8 s, too fast to read fine relief as it passes.
    sp.add_argument("--frames", type=int, default=72,
                    help="frames per 360 deg turn (more = smoother/bigger)")
    sp.add_argument("--fps", type=int, default=12,
                    help="GIF playback rate; frames/fps = seconds per turn")
    sp.add_argument("--size", type=int, default=900, help="GIF square size (px)")
    sp.add_argument("--colors", type=int, default=256,
                    help="GIF palette size, shared across all frames")
    # Open3D's set_zoom is INVERSE: a larger value pulls the camera back. 0.62
    # (the still-figure value) overflows a square frame once the pad turns
    # through 45 deg and presents its diagonal; 0.80 keeps every rotation
    # inside the render, and the content crop takes the slack back afterwards.
    sp.add_argument("--zoom", type=float, default=0.80,
                    help="camera distance; LARGER pulls back and keeps the "
                         "whole spin in frame")
    sp.add_argument("--stride", type=int, default=1,
                    help="vertex subsample; 1 = every pixel (figure quality)")
    sp.add_argument("--z-scale", type=float, default=1.0,
                    help="z exaggeration; 1 = true geometric scale, as in the "
                         "published figures")
    sp.add_argument("--bg", type=float, default=1.0,
                    help="background grey, 1.0 = white like the figures")
    # Between the figures' steep (0.55/0.84) and mid (0.945/0.327) angles. The
    # grazing figure angles throw the pad's near edge off the bottom of frame
    # once it rotates, which a still never shows.
    sp.add_argument("--front", type=float, nargs=3, default=[0.0, 0.70, 0.71],
                    metavar=("X", "Y", "Z"),
                    help="camera->scene direction; larger Y / smaller Z is a "
                         "lower, more grazing angle")
    # --- figure look (default): the 9DTact baked gradient-grey shading ---
    sp.add_argument("--no-figure-look", dest="figure_look", action="store_false",
                    help="use the normal-lit viewer mesh instead of the "
                         "published figures' baked gradient shading")
    sp.add_argument("--smooth", type=int, default=5,
                    help="odd Gaussian on the depth map before meshing; fine "
                         "threads want 3, ball domes and screw heads 11")
    sp.add_argument("--shade-smooth", type=int, default=0,
                    help="extra odd Gaussian on the SHADING gradients only -- "
                         "kills emboss ripple without melting geometry")
    sp.add_argument("--shade-contrast", type=float, default=-0.65,
                    help="gradient-to-grey gain; negative flips which way the "
                         "relief catches the light")
    sp.add_argument("--center", action="store_true",
                    help="center the contact relief before spinning, so it "
                         "turns in place instead of swinging in an arc")
    sp.add_argument("--center-x", type=float, default=0.5)
    sp.add_argument("--center-y", type=float, default=0.54)
    sp.add_argument("--no-crop", dest="crop", action="store_false",
                    help="keep the full square frame instead of cropping to "
                         "the content (one shared box across all frames)")
    sp.add_argument("--crop-margin", type=int, default=14)
    # --- normal-lit look only ---
    sp.add_argument("--ambient", type=float, default=0.30)
    sp.add_argument("--light-azimuth", type=float, default=218.0)
    sp.add_argument("--light-elevation", type=float, default=55.0)
    sp.set_defaults(func=cmd_spin, figure_look=True, crop=True)

    rt = sub.add_parser("realtime",
                        help="live 3D mesh from a camera, non-blocking "
                             "(capture/reconstruct/render decoupled)")
    rt.add_argument("--camera", required=True,
                    help="/dev/video index, USB vendor:product id, or a host camera key")
    rt.add_argument("-c", "--config", default=None,
                    help="host device config (only with a registered opener)")
    rt.add_argument("--calib", default=DEFAULT_CALIB)
    rt.add_argument("--crop-bounds", default=None)
    rt.add_argument("--kernels", type=_kernels_arg, default=DEFAULT_KERNELS)
    rt.add_argument("--median", type=_median_arg, default=0)
    rt.add_argument("--min-component", type=int, default=0)
    rt.add_argument("--stride", type=int, default=3,
                    help="vertex subsample; higher = fewer triangles = faster")
    rt.add_argument("--z-scale", type=float, default=3.0,
                    help="z exaggeration for visibility")
    rt.add_argument("--backend", choices=["auto", "open3d", "cv"],
                    default="auto",
                    help="auto: Open3D lit mesh when importable (the `open3d` "
                         "conda env), else the OpenCV relief fallback")
    rt.set_defaults(func=cmd_realtime)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
