"""Headless tests for the generic fisheye correction (glowtact.fisheye)."""
import os

import numpy as np

from glowtact import fisheye                                         # noqa: E402


def test_detect_disc_finds_bright_circle():
    import cv2

    img = np.zeros((480, 640), np.uint8)
    cv2.circle(img, (320, 240), 150, 220, -1)          # bright gel disc
    cx, cy, R = fisheye.detect_disc(img)
    assert abs(cx - 320) < 8 and abs(cy - 240) < 8
    assert abs(R - 150) < 12


def test_build_maps_shape_and_center():
    mx, my = fisheye.build_maps(320, 240, 200, out_size=360, keep_fov_deg=90)
    assert mx.shape == (360, 360) and my.shape == (360, 360)
    # the output centre samples the disc centre (theta=0 -> r_src=0)
    assert abs(mx[180, 180] - 320) < 1.0
    assert abs(my[180, 180] - 240) < 1.0


def test_rectifier_output_size_and_no_black_at_90deg():
    import cv2

    img = np.full((480, 640, 3), 180, np.uint8)
    cv2.circle(img, (320, 240), 356, 200, -1)          # large bright disc
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r = fisheye.Rectifier(gray, out_size=240, keep_fov_deg=90,
                          disc=(320, 240, 356))
    out = r(img)
    assert out.shape == (240, 240, 3)
    assert (out.sum(2) < 10).mean() < 0.01             # corners stay inside disc


def test_calibrated_rectifier_from_yaml():
    """The saved 180 calibration loads and undistorts to a cropped frame."""
    import cv2
    import numpy as np

    calib = fisheye.DEFAULT_CALIB_180
    if not os.path.exists(calib):
        import pytest
        pytest.skip("glowtact180 calibration not present")
    img = np.full((480, 640, 3), 180, np.uint8)
    cv2.circle(img, (320, 240), 180, 210, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r = fisheye.CalibratedRectifier.from_yaml(calib, sample_gray=gray,
                                              crop="auto")
    out = r(img)
    assert out.ndim == 3 and out.shape[0] > 0 and out.shape[1] > 0
    assert out.shape[0] < 480 or out.shape[1] < 640     # crop happened


def test_make_rectifier_prefers_calib():
    import numpy as np

    calib = fisheye.DEFAULT_CALIB_180
    if not os.path.exists(calib):
        import pytest
        pytest.skip("glowtact180 calibration not present")
    gray = np.full((480, 640), 180, np.uint8)
    rect = fisheye.make_rectifier(gray, calib=calib)
    assert isinstance(rect, fisheye.CalibratedRectifier)
