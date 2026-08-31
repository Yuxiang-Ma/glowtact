"""Relief shading, the CV painter renderer, mesh + Open3D helpers.

numpy at top; cv2 and open3d imported lazily inside functions so the module (and
its pure helpers: height_grid_triangles, light_vector, bake_shading) import
without those deps.
"""
import numpy as np

# The 9DTact-paper camera preset (oblique from above; +z is depth).
O3D_VIEW_PRESET = {"front": [0.0, 0.55, 0.84], "up": [0.0, -1.0, 0.0],
                   "zoom": 0.7}


def surface_shading(height, contact_threshold=0.02):
    """9DTact's relief shading: normalise(dx + dy) -> 0.2..0.8 grey.

    Their Visualizer colours every point by the sum of the surface gradients
    (``np_colors = dx + dy`` scaled into 0.2..0.8, near-zero depth forced
    dark), which is what gives the reconstruction its embossed, side-lit look.
    Reproduced verbatim; only the backend differs (Open3D ships no wheel for
    this machine's Python 3.13).
    """
    dy, dx = np.gradient(height.astype(np.float64))
    shade = dx + dy
    # 9DTact normalises by min/max; on this sensor a few noise spikes own the
    # extremes and flatten everything else, so use robust percentiles instead.
    lo, hi = np.percentile(shade, [2, 98])
    if hi - lo > 1e-9:
        shade = np.clip((shade - lo) / (hi - lo), 0, 1) * 0.6 + 0.2
    else:
        shade = np.full_like(shade, 0.5)
    shade[height <= contact_threshold] = 0.08       # untouched base stays dark
    return shade


def height_to_cloud(height, mm_per_px, stride=2, z_scale=3.0):
    """Full-grid point cloud + relief shading, 9DTact-style.

    The WHOLE surface is exported (not just contact pixels): the flat dark base
    around the contact is part of the look, and the silhouette against it is
    what makes the geometry readable.
    """
    ys, xs = np.mgrid[0:height.shape[0]:stride, 0:height.shape[1]:stride]
    z = height[::stride, ::stride]
    points = np.stack([xs.ravel() * mm_per_px, ys.ravel() * mm_per_px,
                       z.ravel() * z_scale], axis=1)
    shade = surface_shading(height)[::stride, ::stride].ravel()
    return points, shade


class ReliefRenderer:
    """Orthographic painter's-algorithm renderer for the relief point cloud.

    Substitutes for Open3D's viewer: drag to orbit, +/- to zoom, and a preset
    oblique view like 9DTact's (front/lookat/up + rotate in their Visualizer).
    Vectorised end to end, so re-rendering ~45k points per frame is cheap
    enough to run inside the live capture loop.
    """

    def __init__(self, width=900, height=680):
        self.size = (width, height)
        self.reset()

    def reset(self):
        self.yaw, self.pitch, self.zoom = 35.0, 55.0, 1.0

    def render(self, points, shade, depth_mm=None, colormode="relief",
               label=""):
        import cv2

        width, height = self.size
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if len(points) == 0:
            return canvas
        centred = points - points.mean(axis=0)

        yaw, pitch = np.deg2rad(self.yaw), np.deg2rad(self.pitch)
        rot_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        rot_x = np.array([[1, 0, 0],
                          [0, np.cos(pitch), -np.sin(pitch)],
                          [0, np.sin(pitch), np.cos(pitch)]])
        q = centred @ rot_z.T @ rot_x.T

        span = max(np.abs(q[:, :2]).max(), 1e-6)
        scale = self.zoom * 0.46 * min(width, height) / span
        u = (q[:, 0] * scale + width / 2).astype(int)
        v = (q[:, 1] * scale + height / 2).astype(int)

        if colormode == "depth" and depth_mm is not None:
            norm = np.clip(depth_mm / max(depth_mm.max(), 1e-6), 0, 1)
            colors = cv2.applyColorMap((norm * 255).astype(np.uint8),
                                       cv2.COLORMAP_JET).reshape(-1, 3)
            colors[depth_mm <= 0] = 40
        else:
            grey = (shade * 255).astype(np.uint8)
            colors = np.stack([grey, grey, grey], axis=1)

        ok = (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
        # Painter's algorithm: draw far-to-near so near points overwrite.
        order = np.argsort(q[:, 2])[::-1]
        order = order[ok[order]]
        uu, vv, cc = u[order], v[order], colors[order]
        for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):   # 2x2 splat = solid
            canvas[vv + dv, uu + du] = cc
        if label:
            cv2.putText(canvas, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def attach_mouse(self, window):
        """Drag-to-orbit on a cv2 window (Open3D gives this for free)."""
        import cv2

        state = {"drag": None}

        def on_mouse(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:
                state["drag"] = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
                dx, dy = x - state["drag"][0], y - state["drag"][1]
                self.yaw += dx * 0.4
                self.pitch = float(np.clip(self.pitch + dy * 0.4, 0.0, 89.0))
                state["drag"] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                state["drag"] = None

        cv2.setMouseCallback(window, on_mouse)


def _cloud_colors(height, mm_per_px, stride, z_scale, mode):
    """Points + RGB(0..1) colours for either backend's point cloud."""
    import cv2

    points, shade = height_to_cloud(height, mm_per_px, stride, z_scale)
    if mode == "depth":
        depth = height[::stride, ::stride].ravel()
        norm = np.clip(depth / max(depth.max(), 1e-6), 0, 1)
        bgr = cv2.applyColorMap((norm * 255).astype(np.uint8),
                                cv2.COLORMAP_JET).reshape(-1, 3)
        bgr[depth <= 0] = 40
        colors = bgr[:, ::-1].astype(np.float64) / 255.0     # BGR -> RGB
    else:
        colors = np.repeat(shade[:, None], 3, axis=1)
    return points, colors


def height_grid_triangles(rows, cols):
    """Triangle index pairs stitching a rows x cols height grid into a surface.

    Two triangles per cell, vectorised. Winding is chosen so the computed
    normals face the viewer (+z side) and the surface lights up instead of
    rendering self-shadowed black.
    """
    idx = np.arange(rows * cols).reshape(rows, cols)
    a = idx[:-1, :-1].ravel()
    b = idx[:-1, 1:].ravel()
    c = idx[1:, :-1].ravel()
    d = idx[1:, 1:].ravel()
    # (a,b,c): +x edge crossed into +y edge gives +z normals. The first cut of
    # this used (a,c,b), whose -z normals faced away from the camera -- the
    # renderer back-face-culled the whole surface and the baked Lambert term
    # lit the ambient floor only (probe: vertex-colour median 0.186 = exactly
    # grey * ambient, and 99.4% of rendered pixels were background).
    return np.concatenate([np.stack([a, b, c], axis=1),
                           np.stack([b, d, c], axis=1)])


def grid_xy(shape, stride, mm_per_px):
    """Constant x/y (mm) vertex columns for a strided height grid.

    Split out so a realtime loop can precompute x/y ONCE and rebuild only the z
    column per frame (the shape/stride/scale never change mid-stream).
    """
    ys, xs = np.mgrid[0:shape[0]:stride, 0:shape[1]:stride]
    return xs.ravel() * mm_per_px, ys.ravel() * mm_per_px


def light_vector(azimuth_deg, elevation_deg):
    """Unit light direction from azimuth (in the x/y plane) and elevation."""
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    return np.array([np.cos(elevation) * np.cos(azimuth),
                     np.cos(elevation) * np.sin(azimuth),
                     np.sin(elevation)])


def bake_shading(normals, light, ambient=0.30, grey=0.62):
    """Lambert term baked to per-vertex grey. Pure numpy, testable anywhere.

    ambient=1 washes the relief out entirely (uniform grey); ambient=0 lets
    away-facing slopes go fully black. The relief CONTRAST is exactly this
    trade-off, which is why it is exposed as a parameter rather than baked in.
    """
    lambert = np.clip(normals @ light, 0.0, 1.0)
    shade = grey * (ambient + (1.0 - ambient) * lambert)
    return np.repeat(shade[:, None], 3, axis=1)


def height_to_o3d_mesh(height, mm_per_px, stride, z_scale, grey=0.62,
                       light_azimuth_deg=218.0, light_elevation_deg=55.0,
                       ambient=0.30):
    """The 9DTact-paper look: a dense grey TriangleMesh, lit by its normals.

    A point cloud reads sparse at any oblique angle; the published figures are
    a continuous Lambertian surface. Building an actual mesh and letting the
    renderer light it from the vertex normals reproduces that directly.
    """
    import open3d as o3d

    z = height[::stride, ::stride]
    rows, cols = z.shape
    ys, xs = np.mgrid[0:height.shape[0]:stride, 0:height.shape[1]:stride]
    vertices = np.stack([xs.ravel() * mm_per_px, ys.ravel() * mm_per_px,
                         z.ravel() * z_scale], axis=1)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(
        height_grid_triangles(rows, cols))
    mesh.compute_vertex_normals()

    # Bake oblique Lambertian lighting into the vertex colours. The legacy
    # Open3D viewer only offers a camera headlight, which hits this camera-
    # facing surface square on and washes every slope out to the same white;
    # the paper's look needs light raking across the surface so the relief
    # shades. Rendered with option.light_on = False so this baked shading is
    # exactly what appears on screen.
    normals = np.asarray(mesh.vertex_normals)
    light = light_vector(light_azimuth_deg, light_elevation_deg)
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        bake_shading(normals, light, ambient, grey))
    return mesh


def apply_o3d_view_preset(vis, center, preset=O3D_VIEW_PRESET):
    """Set the shared oblique camera preset on an Open3D visualizer."""
    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_front(preset["front"])
    ctr.set_up(preset["up"])
    ctr.set_zoom(preset["zoom"])


def render_turntable(mesh, n_frames=48, width=900, height=900, bg=0.92,
                     zoom=0.62, axis=(0.0, 0.0, 1.0), front=None, up=None,
                     lookat=None):
    """Render a mesh spun 360 deg about `axis` -> list of RGB uint8 frames.

    Offscreen (visible=False). The mesh is rotated about its own centre one
    even step per frame; with the baked-shading mesh (light_on=False) the lit
    look rotates WITH the surface, so the object reads as a solid turntable.
    `axis` is the surface normal (z) by default: an oblique camera then sees
    every side as the relief spins in its plane.

    `front`/`up` override the shared viewer preset, so a turntable can be shot
    from the same camera as a published still figure. `lookat` overrides the
    point the camera aims at, which the mesh centre otherwise supplies; several
    objects only share one camera exactly if they share this too.
    """
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.background_color = np.array([bg, bg, bg])
    opt.light_on = False
    opt.mesh_show_back_face = True
    apply_o3d_view_preset(vis, mesh.get_center() if lookat is None
                          else list(lookat))
    ctr = vis.get_view_control()
    if front is not None:
        ctr.set_front(list(front))
    if up is not None:
        ctr.set_up(list(up))
    vis.get_view_control().set_zoom(zoom)

    centre = mesh.get_center()
    step = mesh.get_rotation_matrix_from_axis_angle(
        np.asarray(axis, float) * (2.0 * np.pi / n_frames))
    frames = []
    try:
        for _ in range(n_frames):
            vis.poll_events()
            vis.update_renderer()
            buf = vis.capture_screen_float_buffer(do_render=True)
            frames.append((np.asarray(buf) * 255.0).astype(np.uint8))
            mesh.rotate(step, center=centre)
            vis.update_geometry(mesh)
    finally:
        vis.destroy_window()
    return frames


def save_gif(path, frames, fps=20, colors=256):
    """Write RGB uint8 frames to an animated GIF (PIL; no imageio needed).

    Every frame is mapped through ONE palette. Letting PIL quantize each frame
    independently is what makes a turntable of a matte grey surface shimmer:
    per-frame palettes drift as the silhouette changes, so a flat pad crawls
    with dither noise that was never in the render. These renders are grey on
    a light background, so a 256-entry palette drawn from the whole sequence
    is effectively lossless and dithering can be turned off entirely.
    """
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames]
    # Build the palette from a strip of every frame, not just the first: the
    # first frame need not contain the darkest slope in the revolution.
    w, h = imgs[0].size
    strip = Image.new("RGB", (w, h * len(imgs)))
    for i, im in enumerate(imgs):
        strip.paste(im, (0, i * h))
    palette = strip.quantize(colors=colors, method=Image.MEDIANCUT)
    quantized = [im.quantize(palette=palette, dither=Image.Dither.NONE)
                 for im in imgs]
    quantized[0].save(path, save_all=True, append_images=quantized[1:],
                      duration=int(round(1000.0 / max(fps, 1))), loop=0,
                      disposal=2, optimize=True)
    return path


def save_ply(path, height_map, mm_per_px, stride=2):
    """Write an ASCII PLY point cloud -- no Open3D dependency needed.

    X/Y in mm via `mm_per_px` (a single nominal scale; fine for eyeballing the
    vanilla output), Z = depth. Only contact pixels are exported.
    """
    ys, xs = np.mgrid[0:height_map.shape[0]:stride, 0:height_map.shape[1]:stride]
    z = height_map[::stride, ::stride]
    keep = z > 0
    pts = np.stack([xs[keep] * mm_per_px, ys[keep] * mm_per_px, z[keep]], axis=1)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n"
                f"element vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "end_header\n")
        for x, y, zz in pts:
            f.write(f"{x:.3f} {y:.3f} {zz:.4f}\n")
    return len(pts)
