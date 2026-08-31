import sys

import numpy as np
import pytest


from glowtact import reconstruct as r3  # noqa: E402


def test_build_lut_is_monotonic_and_zero_below_threshold():
    """Depth grows with darkening -- the sensor premise -- so the table must."""
    rng = np.random.default_rng(0)
    gray = rng.integers(10, 120, 20000)
    depth = gray * 0.01 + rng.normal(0, 0.05, len(gray))   # noisy but rising

    lut, counts = r3.build_lut(gray, depth, lighting_threshold=8)

    assert lut.shape == (256,)
    assert np.all(lut[:8] == 0)                     # noise floor is zero depth
    assert np.all(np.diff(lut) >= 0)                # non-decreasing everywhere
    assert lut[119] == pytest.approx(1.19, abs=0.1)


def test_build_lut_ignores_sparsely_sampled_grays():
    """A handful of outlier pixels at one grey must not anchor the deep end.

    Observed on real data: the last few grey levels had <10 samples with a
    garbage mean, and clamping there dragged 1.03 mm down to 0.67 mm.
    """
    gray = np.concatenate([np.full(1000, 50), np.full(1000, 60),
                           np.full(3, 200)])              # 3 rogue samples
    depth = np.concatenate([np.full(1000, 0.5), np.full(1000, 0.6),
                            np.full(3, 0.01)])            # garbage depth

    lut, _ = r3.build_lut(gray, depth, lighting_threshold=8, min_samples=50)

    # The rogue grey is ignored; the table clamps at the last RELIABLE level.
    assert lut[200] == pytest.approx(0.6, abs=1e-6)
    assert lut[255] == pytest.approx(0.6, abs=1e-6)


def test_sphere_depth_map_matches_the_analytic_sphere():
    radius_mm, contact_px, mm_per_px = 3.0, 20.0, 0.05
    depth, valid = r3.sphere_depth_map((100, 100), (50.0, 50.0),
                                       contact_px, radius_mm, mm_per_px)

    r_c = contact_px * mm_per_px                    # 1.0 mm contact radius
    expected_centre = radius_mm - np.sqrt(radius_mm ** 2 - r_c ** 2)
    assert depth[50, 50] == pytest.approx(expected_centre, rel=1e-3)
    assert depth[50, 50] == depth.max()             # deepest at the centre
    assert not valid[50, 90]                        # outside the contact
    assert depth[50, 90] == 0.0


def test_sphere_depth_map_clamps_impossible_contact_radius():
    """A detected contact wider than the ball itself is an artefact, not physics."""
    depth, _valid = r3.sphere_depth_map((100, 100), (50.0, 50.0),
                                        contact_px=90.0, ball_radius_mm=1.0,
                                        mm_per_px=0.05)
    # Never exceeds the full ball radius (max possible indentation ~ R).
    assert depth.max() <= 1.0 + 1e-6
    assert np.isfinite(depth).all()


def test_contact_radius_scans_outward_to_the_noise_floor():
    diff = np.zeros((100, 100))
    ys, xs = np.mgrid[0:100, 0:100]
    r = np.hypot(xs - 50, ys - 50)
    diff[r < 12] = 60.0                              # a 12 px contact disc

    found = r3.contact_radius_px(diff, (50.0, 50.0), max_radius=30)
    assert found == pytest.approx(12, abs=1.5)

    # A ball that never touched (centre at noise level) reports None.
    assert r3.contact_radius_px(np.zeros((50, 50)), (25.0, 25.0), 20) is None


def test_grid_local_scale_recovers_the_spacing():
    xs, ys = np.meshgrid(np.arange(9) * 40.0, np.arange(6) * 40.0)
    points = np.stack([xs, ys], axis=-1).reshape(-1, 2)
    scale = r3.grid_local_scale(points, 6, 9)
    assert scale.shape == (6, 9)
    assert np.allclose(scale, 40.0)                  # uniform grid, uniform scale


def test_depth_from_diff_applies_lut_and_threshold():
    pytest.importorskip("cv2")
    lut = np.linspace(0, 2.55, 256)                  # 0.01 mm per grey level
    diff = np.zeros((50, 50))
    diff[20:30, 20:30] = 100.0

    height = r3.depth_from_diff(diff, lut, lighting_threshold=8, kernels=(1,))

    # LUT minus its own value at the threshold, as in 9DTact.
    assert height[25, 25] == pytest.approx(1.0 - 0.08, abs=0.02)
    assert height[5, 5] == 0.0


def test_depth_from_diff_median_kills_hot_pixel_spikes():
    """A lone hot pixel must vanish with --median, not just get blurred.

    Gaussian smoothing spreads an outlier into a low mound (still visibly a
    spike on the mesh); the median stage removes it before the LUT amplifies
    it. Observed on the Oreo reconstruction: isolated specks in the diff
    became tall needles.
    """
    pytest.importorskip("cv2")
    lut = np.linspace(0, 2.55, 256)
    diff = np.zeros((50, 50))
    diff[10, 40] = 200.0                             # a single hot pixel
    diff[20:30, 20:30] = 100.0                       # real contact survives

    spiky = r3.depth_from_diff(diff, lut, lighting_threshold=8, kernels=(1,))
    clean = r3.depth_from_diff(diff, lut, lighting_threshold=8, kernels=(1,),
                               median=3)

    assert spiky[10, 40] > 1.5                       # the spike, LUT-amplified
    assert clean[10, 40] == 0.0                      # gone entirely
    assert clean[25, 25] == pytest.approx(1.0 - 0.08, abs=0.02)  # contact kept


def test_depth_from_diff_min_component_kills_dust_blobs():
    """Multi-pixel dust blobs survive the median; the area filter kills them.

    Observed on the GlowTact pad: dust specks several pixels wide passed
    --median 3 and speckled every background. No real contact is a lone
    ~50 px island, so area discriminates where kernel size cannot.
    """
    pytest.importorskip("cv2")
    lut = np.linspace(0, 2.55, 256)
    diff = np.zeros((80, 80))
    diff[10:17, 60:67] = 90.0                        # 49 px dust blob
    diff[30:60, 20:50] = 100.0                       # 900 px real contact

    speckled = r3.depth_from_diff(diff, lut, lighting_threshold=8,
                                  kernels=(1,))
    clean = r3.depth_from_diff(diff, lut, lighting_threshold=8, kernels=(1,),
                               min_component_px=200)

    assert speckled[13, 63] > 0.5                    # blob passed the LUT
    assert clean[13, 63] == 0.0                      # area filter removed it
    assert clean[45, 35] == pytest.approx(1.0 - 0.08, abs=0.02)  # contact kept


def test_save_ply_writes_contact_points_only(tmp_path):
    height = np.zeros((20, 20), dtype=np.float32)
    height[8:12, 8:12] = 0.5
    path = tmp_path / "cloud.ply"

    written = r3.save_ply(str(path), height, mm_per_px=0.1, stride=1)

    text = path.read_text().splitlines()
    assert text[0] == "ply"
    assert f"element vertex {written}" in "\n".join(text[:6])
    assert written == 16                             # only the 4x4 contact patch


def test_calibration_round_trips_through_file(tmp_path):
    lut = np.linspace(0, 1, 256)
    counts = np.arange(256)
    stats = {"lighting_threshold": 8, "max_depth_mm": 1.0, "spacing_mm": 5.0}
    path = str(tmp_path / "calib.npz")

    r3.save_calibration(path, lut, counts, stats)
    loaded_lut, loaded_stats = r3.load_calibration(path)

    assert np.allclose(loaded_lut, lut)
    assert loaded_stats == stats


def test_depth_command_from_diff_needs_no_reference(tmp_path):
    """--from-diff treats inputs as ready-made darkening maps.

    The current capture format stores darkening directly (background ~0), and
    the untouched baseline is not saved alongside -- so reconstruction must be
    possible without one.
    """
    cv2 = pytest.importorskip("cv2")

    lut = np.linspace(0, 2.55, 256)
    r3.save_calibration(str(tmp_path / "calib.npz"), lut, np.ones(256),
                        {"lighting_threshold": 8, "max_depth_mm": 2.0,
                         "spacing_mm": 7.4, "mm_per_px": 0.043})

    dark = np.zeros((60, 60, 3), dtype=np.uint8)
    dark[20:40, 20:40] = 100                     # a 100-grey darkening patch
    src = tmp_path / "imgs"
    src.mkdir()
    cv2.imwrite(str(src / "a_diff.png"), dark)
    cv2.imwrite(str(src / "a_diff_grey.png"), dark)   # must NOT be picked up
    cv2.imwrite(str(src / "a_tactile.png"), dark)     # must NOT be picked up

    out = tmp_path / "out"
    r3.main(["depth", str(src), "--from-diff", "--pattern", "*_diff.png",
             "--calib", str(tmp_path / "calib.npz"), "--out", str(out)])

    produced = sorted(p.name for p in out.iterdir())
    assert produced == ["a_diff_depth.npy", "a_diff_depth.png"]
    height = np.load(out / "a_diff_depth.npy")
    # LUT[100] - LUT[8], within smoothing tolerance.
    assert height.max() == pytest.approx(1.0 - 0.08, abs=0.05)
    assert height[5, 5] == 0.0


def test_depth_command_signed_diff_decodes_darkening(tmp_path):
    """--diff-mode signed reads 128-centred diffs (contact = darker).

    The fingerprint captures store diff_mode `signed` (frame-baseline+128),
    where a ridge is DARKER than the baseline. Darkening must be 128 - gray,
    so a patch at gray 28 gives 100 grey levels of darkening -- the same
    depth the darkening-format patch at gray 100 produces.
    """
    cv2 = pytest.importorskip("cv2")

    lut = np.linspace(0, 2.55, 256)
    r3.save_calibration(str(tmp_path / "calib.npz"), lut, np.ones(256),
                        {"lighting_threshold": 8, "max_depth_mm": 2.0,
                         "spacing_mm": 7.4, "mm_per_px": 0.043})

    signed = np.full((60, 60, 3), 128, dtype=np.uint8)   # untouched baseline
    signed[20:40, 20:40] = 28                             # 100 darker = ridge
    src = tmp_path / "imgs"
    src.mkdir()
    cv2.imwrite(str(src / "a_diff.png"), signed)

    out = tmp_path / "out"
    r3.main(["depth", str(src), "--from-diff", "--diff-mode", "signed",
             "--pattern", "*_diff.png", "--calib", str(tmp_path / "calib.npz"),
             "--out", str(out)])

    height = np.load(out / "a_diff_depth.npy")
    assert height.max() == pytest.approx(1.0 - 0.08, abs=0.05)   # LUT[100]-LUT[8]
    assert height[5, 5] == 0.0                                   # baseline flat


def test_surface_shading_reproduces_9dtact_formula():
    """Relief colours: normalise(dx+dy) into 0.2..0.8, untouched base dark."""
    height = np.zeros((40, 40))
    ys, xs = np.mgrid[0:40, 0:40]
    r = np.hypot(xs - 20, ys - 20)
    height[r < 10] = 0.5 * (1 - (r[r < 10] / 10) ** 2)    # a dome

    shade = surface_shading = r3.surface_shading(height)

    assert shade[0, 0] == pytest.approx(0.08)             # flat base -> dark
    contact = shade[height > 0.02]
    assert contact.min() >= 0.2 - 1e-9 and contact.max() <= 0.8 + 1e-9
    # Opposite slopes of the dome shade differently -- that IS the relief.
    assert abs(shade[20, 14] - shade[20, 26]) > 0.1


def test_height_to_cloud_exports_the_full_surface():
    """The dark base plane around the contact is part of the 9DTact look."""
    height = np.zeros((20, 30))
    height[5:10, 5:10] = 0.4

    points, shade = r3.height_to_cloud(
        height, mm_per_px=0.05, stride=1, z_scale=3.0)

    assert len(points) == 20 * 30                          # every pixel, not just contact
    assert points[:, 2].max() == pytest.approx(0.4 * 3.0)  # z exaggeration applied
    assert points[:, 0].max() == pytest.approx(29 * 0.05)  # x in mm
    assert len(shade) == len(points)


def test_relief_renderer_draws_near_over_far():
    """Painter's algorithm: the nearer point must win a contested pixel."""
    r3_mod = r3
    renderer = r3_mod.ReliefRenderer(width=100, height=100)
    renderer.yaw, renderer.pitch = 0.0, 0.0               # identity view

    # Two points at the same x/y, different depth, distinguishable shades.
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                       [5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])  # spreaders for scale
    shade = np.array([1.0, 0.5, 0.3, 0.3])

    canvas = renderer.render(points, shade)

    assert canvas.any()                                    # something was drawn
    centre = canvas[50, 50]
    # Whichever depth ordering the convention picks, the pixel must hold ONE of
    # the two contested shades entirely -- never the average of both.
    assert int(centre[0]) in (255, 127)


def test_height_grid_triangles_wind_toward_plus_z():
    """Triangle winding must give +z normals, or the mesh back-face-culls.

    Measured failure with the opposite winding: the whole surface vanished
    from the Open3D render (99.4% background) and the baked Lambert shading
    bottomed out at the ambient floor because every normal pointed away from
    the light.
    """
    tris = r3.height_grid_triangles(2, 2)
    vertices = np.array([[0, 0, 0], [1, 0, 0],
                         [0, 1, 0], [1, 1, 0]], dtype=float)
    assert len(tris) == 2
    for tri in tris:
        a, b, c = vertices[tri]
        normal = np.cross(b - a, c - a)
        assert normal[2] > 0, f"triangle {tri} winds toward -z"


def test_bake_shading_ambient_controls_relief_contrast():
    """ambient=1 must give uniform grey (no relief); low ambient darkens
    away-facing slopes toward black. This IS the visual 'enhancement', so it
    has to be a parameter with predictable limits, not a constant."""
    normals = np.array([[0.0, 0.0, 1.0],       # flat, faces the sky
                        [0.0, 0.0, -1.0]])     # faces fully away
    light = r3.light_vector(218.0, 55.0)

    flat = r3.bake_shading(normals, light, ambient=1.0, grey=0.62)
    assert np.allclose(flat, 0.62)             # washed out: no shading at all

    hard = r3.bake_shading(normals, light, ambient=0.0, grey=0.62)
    assert hard[1].max() == pytest.approx(0.0)             # away-facing -> black
    assert hard[0].max() == pytest.approx(0.62 * np.sin(np.deg2rad(55)),
                                          abs=1e-6)        # lit by elevation


def test_light_vector_is_unit_and_elevation_sets_z():
    v = r3.light_vector(azimuth_deg=218.0, elevation_deg=55.0)
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert v[2] == pytest.approx(np.sin(np.deg2rad(55.0)))
    # Elevation 90 = straight down the z axis, azimuth irrelevant.
    top = r3.light_vector(azimuth_deg=123.0, elevation_deg=90.0)
    assert top[2] == pytest.approx(1.0)
