import sys

import numpy as np
import pytest


from glowtact import figure  # noqa: E402


def test_flat_depth_shades_to_exactly_base():
    """A pad with no relief must render as the papers' uniform mid-grey."""
    shade = figure.gradient_shade(np.zeros((16, 16)), contrast=0.3, base=0.5)
    assert np.allclose(shade, 0.5)


def test_negative_contrast_flips_the_relief():
    depth = np.zeros((16, 16))
    depth[8:, :] = 1.0
    # Small contrast so neither direction hits the 0.05/0.98 clamp, which
    # would clip the two sides by different amounts.
    pos = figure.gradient_shade(depth, contrast=0.3)
    neg = figure.gradient_shade(depth, contrast=-0.3)
    # Same geometry, mirrored shading about the flat base.
    assert np.allclose(pos - 0.5, -(neg - 0.5))


def test_shift_depth_zero_does_not_wrap():
    depth = np.zeros((6, 6))
    depth[0, 0] = 1.0
    shifted = figure.shift_depth_zero(depth, 2, 2)
    assert shifted[2, 2] == 1.0
    assert shifted.sum() == 1.0            # nothing reappeared on the far edge


def test_center_depth_contact_moves_the_blob_to_the_middle():
    depth = np.zeros((41, 41))
    depth[2:6, 2:6] = 1.0
    centred = figure.center_depth_contact(depth)
    ys, xs = np.where(centred > 0.5)
    assert abs(ys.mean() - 20) <= 1
    assert abs(xs.mean() - 20) <= 1


def test_center_threshold_ignores_sub_threshold_noise():
    """Edge ripple below the threshold must not drag the centroid."""
    depth = np.zeros((41, 41))
    depth[18:23, 18:23] = 1.0              # the real contact, already centred
    depth[40, :] = 0.05                    # a ripple along the bottom edge
    centred = figure.center_depth_contact(depth, threshold=0.15)
    ys, xs = np.where(centred > 0.5)
    assert abs(ys.mean() - 20) <= 1 and abs(xs.mean() - 20) <= 1


def test_content_bbox_and_union():
    img = np.full((40, 40, 3), 255, np.uint8)
    img[10:20, 12:18] = 0
    x0, y0, x1, y1 = figure.content_bbox(img, margin=0)
    assert (x0, y0) == (12, 10) and (x1, y1) == (17, 19)
    other = (5, 5, 30, 30)
    assert figure.union_bbox([(x0, y0, x1, y1), other]) == (5, 5, 30, 30)


def test_shape_2d_maps_deepest_to_darkest():
    depth = np.array([[0.0, 0.5, 1.0]])
    panel = figure.shape_2d(depth)
    assert panel[0, 0, 0] > panel[0, 1, 0] > panel[0, 2, 0]


def test_gif_keeps_a_flat_region_flat_and_stable(tmp_path):
    """Per-frame quantization plus dithering is what makes a matte pad crawl.

    A uniform grey patch must decode to ONE colour, and to the SAME colour in
    every frame -- otherwise the background of a turntable shimmers with noise
    that was never in the render.
    """
    Image = pytest.importorskip("PIL.Image")
    from glowtact.render import save_gif

    rng = np.random.default_rng(0)
    frames = []
    for _ in range(4):
        f = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        f[8:24, 8:24] = 137          # the "pad": flat, awkwardly off-palette
        frames.append(f)
    path = tmp_path / "spin.gif"
    save_gif(str(path), frames, fps=10)

    im = Image.open(str(path))
    assert im.n_frames == 4
    seen = set()
    for i in range(im.n_frames):
        im.seek(i)
        patch = np.array(im.convert("RGB"))[8:24, 8:24]
        assert len(np.unique(patch.reshape(-1, 3), axis=0)) == 1
        seen.add(tuple(patch[0, 0]))
    assert len(seen) == 1
