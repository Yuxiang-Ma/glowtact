"""glowtact.camera -- stream, snapshot and record a GlowTact tactile camera.

Self-contained (OpenCV + NumPy only; pyudev optional for USB auto-discovery)
and force-sensor-free, so it drops into this project or any other. See the
package README for a tour.

    from glowtact.camera import GlowTactCamera, serve_mjpeg

    with GlowTactCamera() as cam:            # auto-find the 180 deg unit
        cam.set_baseline()
        serve_mjpeg(cam, port=8000, diff_mode="dark")
"""
from .camera import GlowTactCamera
from .diff import MODES as DIFF_MODES, render_difference, signed_peak
from .discovery import (DeviceNotFound, find_glowtact, list_candidates,
                        GLOWTACT_180_PID, GLOWTACT_180_VID)
from .mjpeg import serve_mjpeg

from .. import __version__  # noqa: E402
__all__ = [
    "GlowTactCamera",
    "serve_mjpeg",
    "render_difference",
    "signed_peak",
    "DIFF_MODES",
    "find_glowtact",
    "list_candidates",
    "DeviceNotFound",
    "GLOWTACT_180_VID",
    "GLOWTACT_180_PID",
    "__version__",
]
