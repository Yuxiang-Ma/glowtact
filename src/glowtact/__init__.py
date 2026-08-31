"""glowtact -- everything GlowTact: camera, diff imaging, depth, shear, optics.

The library is split by what each module assumes about the sensor:

    glowtact.camera      open / stream / snapshot / record a GlowTact camera
    glowtact.capture     threaded newest-frame reader + the pluggable opener
    glowtact.diffio      the signed-difference codec and viewable renders
    glowtact.depth       darkness -> depth LUT (9DTact style)
    glowtact.reconstruct calibrate / depth / live / view / spin / realtime CLI
    glowtact.realtime    non-blocking live 3D mesh pipeline
    glowtact.undistort   pressed-grid fisheye rectification (+ live calibrate)
    glowtact.fisheye     analytic / per-unit fisheye correction for clips
    glowtact.shear       shear-field tracking (LK / Farneback / DIS / phase)
    glowtact.clean_diff, glowtact.enhance_diff
                         membrane-shift suppression, display band-pass
    glowtact.render, glowtact.figure, glowtact.imgfilt, glowtact.crop, glowtact.mask
                         mesh rendering, paper figures, filters, crops, aperture
    glowtact.tools       video / shear renderers and evaluators (console scripts)
    glowtact.calib       shipped per-unit undistortion maps (package data)

Heavy optional dependencies (cv2, open3d, matplotlib, pyudev) are imported
lazily inside functions, so ``import glowtact`` is cheap and the pure helpers
are unit-testable without them.
"""
import os

__version__ = "0.2.0"

CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib")


def calib_path(*parts):
    """Absolute path of a shipped calibration file, e.g.
    ``calib_path("glowtact180", "opencv_fisheye_glowtact180.yaml")``."""
    return os.path.join(CALIB_DIR, *parts)


__all__ = ["__version__", "CALIB_DIR", "calib_path"]
