"""The published-figure look: gradient-grey relief on white, at true mm scale.

The 9DTact / gelsight_wedge convention -- bake ``dx + dy`` into vertex grey and
render with lighting OFF, so what you see is exactly the shading term. This is
deliberately NOT ``render.height_to_o3d_mesh``, which lights a mesh from its
normals with a raking key light: that one is for the interactive viewer, this
one is for figures and turntables that have to match the paper.

Extracted from ``scripts/render_shapes_o3d.py`` so the still figures and the
spin GIFs build their meshes the same way. Two renderers that disagree about
smoothing produce a figure and an animation of the same press that do not look
like the same sensor.
"""
import numpy as np


def gradient_shade(depth_mm, contrast=0.30, base=0.5, shade_smooth=0):
    """9DTact / wedge shading: grey = base + k * (dx + dy), flat stays base.

    The gradient sum is normalised by its max so every sample uses its full
    grey range; zero slope maps to exactly `base`, reproducing the uniform
    mid-grey pad of the published figures.

    `shade_smooth` blurs the SHADING source beyond the geometry: the emboss
    look amplifies millimetre-scale gradient ripple that is invisible in the
    silhouette, so softening only the shade (never the mesh) is what makes the
    surface read as smooth without melting real detail like screw threads out
    of the geometry. A NEGATIVE `contrast` flips which way the relief appears
    to catch the light.
    """
    z = depth_mm
    if shade_smooth >= 3:
        import cv2

        z = cv2.GaussianBlur(z.astype(np.float32), (shade_smooth,
                                                    shade_smooth), 0)
    dy, dx = np.gradient(z)
    s = dx + dy
    # Normalise by the true max (9DTact normalises by min/max too): a
    # percentile norm saturates the steep flanks into flat black/white blobs
    # with ring-shaped clip boundaries -- the smoothed s has no hot pixels
    # left, so the max is safe here.
    norm = np.abs(s).max()
    if norm < 1e-9:
        return np.full_like(depth_mm, base)
    return np.clip(base + contrast * s / norm, 0.05, 0.98)


def shift_depth_zero(depth_mm, dx, dy):
    """Shift a depth map without wrapping content around the opposite edge."""
    out = np.zeros_like(depth_mm)
    h, w = depth_mm.shape
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)
    if src_x1 > src_x0 and src_y1 > src_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = depth_mm[src_y0:src_y1,
                                                     src_x0:src_x1]
    return out


def center_depth_contact(depth_mm, threshold=0.02, target_x=0.5, target_y=0.5):
    """Move the reconstructed contact relief to the depth-frame center.

    Matters more for a turntable than for a still: an off-centre relief on a
    pad spinning about its own centre swings through a wide arc instead of
    turning in place.
    """
    mask = depth_mm > threshold
    if not mask.any():
        return depth_mm
    ys, xs = np.where(mask)
    cx = (depth_mm.shape[1] - 1) * target_x
    cy = (depth_mm.shape[0] - 1) * target_y
    return shift_depth_zero(depth_mm, int(round(cx - xs.mean())),
                            int(round(cy - ys.mean())))


def build_mesh(depth_mm, mm_per_px, smooth_px=5, shade_contrast=0.30,
               shade_smooth=0, z_scale=1.0, stride=1):
    """Full-resolution grid mesh in mm with baked gradient-grey vertex colour.

    `z_scale` 1 is true geometric scale, which is what the figures use.
    `stride` subsamples vertices; leave at 1 for figure quality -- 2 halves the
    grid in each axis and visibly stair-steps fine threads.
    """
    import cv2
    import open3d as o3d

    z = depth_mm.astype(np.float32)
    if smooth_px >= 3:
        z = cv2.GaussianBlur(z, (smooth_px, smooth_px), 0)
    shade_full = gradient_shade(z, shade_contrast, shade_smooth=shade_smooth)
    if stride > 1:
        z = z[::stride, ::stride]
        shade_full = shade_full[::stride, ::stride]
    h, w = z.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    # x is mirrored because the view (front +y/+z, up -y) renders screen right
    # as -x; the double mirror puts image left/right back where the tactile row
    # has it (verified: un-mirrored, the OREO text renders reversed).
    scale = mm_per_px * stride
    vertices = np.stack([(w - 1 - xs).ravel() * scale,
                         ys.ravel() * scale,
                         z.ravel().astype(np.float64) * z_scale], axis=1)
    idx = np.arange(h * w).reshape(h, w)
    a = idx[:-1, :-1].ravel(); b = idx[:-1, 1:].ravel()
    c = idx[1:, :-1].ravel(); d = idx[1:, 1:].ravel()
    triangles = np.concatenate([np.stack([a, b, c], axis=1),
                                np.stack([b, d, c], axis=1)])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.repeat(shade_full.ravel()[:, None], 3, axis=1))
    mesh.compute_vertex_normals()
    return mesh


def render_mesh(mesh, width=1280, height=960, zoom=0.62,
                front=(0.0, 0.55, 0.84), bg=1.0, lookat=None):
    """Offscreen render: lighting off (baked shade only), oblique paper view.

    `front` is the camera->scene direction; a larger y / smaller z lowers the
    angle (more grazing).

    `lookat` overrides the mesh centre the camera aims at. Pass a shared point
    when several objects must share one camera: the mesh centre carries each
    object's own mean relief height, so aiming at it shifts every object in
    frame by a slightly different amount.
    """
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh)
    option = vis.get_render_option()
    option.background_color = np.array([bg, bg, bg])
    option.light_on = False
    option.mesh_show_back_face = True
    ctr = vis.get_view_control()
    ctr.set_lookat(mesh.get_center() if lookat is None else list(lookat))
    ctr.set_front(list(front))           # camera south of the pad, above it
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_zoom(zoom)
    vis.poll_events()
    vis.update_renderer()
    img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    return (img * 255).astype(np.uint8)


def content_bbox(img, margin=14, threshold=250):
    """Bounding box of the non-background content, plus a small margin."""
    content = (img < threshold).any(axis=2)
    ys, xs = np.where(content)
    if len(ys) == 0:
        return 0, 0, img.shape[1], img.shape[0]
    return (max(0, xs.min() - margin), max(0, ys.min() - margin),
            min(img.shape[1], xs.max() + margin),
            min(img.shape[0], ys.max() + margin))


def crop_white(img, margin=14, threshold=250):
    """Tight crop to the non-white content plus a small margin."""
    x0, y0, x1, y1 = content_bbox(img, margin, threshold)
    return img[y0:y1, x0:x1]


def union_bbox(boxes):
    """Smallest box containing all of `boxes` -- one crop for every frame.

    Cropping a turntable frame-by-frame makes the object jitter and breathe as
    its silhouette changes; one shared box keeps it planted.
    """
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def shape_2d(depth_mm, lo=0.86, hi=0.30):
    """The papers' 2D-shape panel: light-grey pad, deeper = darker grey."""
    z = depth_mm / max(depth_mm.max(), 1e-6)
    grey = lo + (hi - lo) * z
    return (np.repeat(grey[:, :, None], 3, axis=2) * 255).astype(np.uint8)
