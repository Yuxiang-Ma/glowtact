"""Find a GlowTact camera's /dev/videoN node by USB vendor:product id.

Port-independent: the sensor is matched by its USB identity, so it is found in
any port. Two resolution backends, tried in order:

  * pyudev, if installed -- the same source the parent project uses.
  * a dependency-free /sys/class/video4linux walk -- so this package runs on a
    bare ``pip install opencv-python numpy`` with nothing else.

A UVC webcam exposes several /dev/videoN nodes but only some deliver frames, so
candidates are probed (open + read) and the first that yields a frame wins.
This module is Linux-only for discovery; on other platforms pass an explicit
device index to :class:`~glowtact.camera.GlowTactCamera` instead.
"""
import glob
import os

# GlowTact 180 deg FOV camera ("USB_camera"). The 120 deg builds use different
# ids (1bcf:28c4, 337b:090c) -- pass them explicitly if that is what you have.
GLOWTACT_180_VID = "1e45"
GLOWTACT_180_PID = "8022"


class DeviceNotFound(Exception):
    pass


def _sysfs_ids(node):
    """(vendor_id, product_id) for /dev/videoN via sysfs, or (None, None).

    The ``device`` symlink of a video node points at the USB *interface*; the
    idVendor/idProduct files live a couple levels up on the USB *device*, so we
    walk parents until we find them.
    """
    name = os.path.basename(node)
    dev = os.path.realpath(f"/sys/class/video4linux/{name}/device")
    while dev and dev != "/":
        vid = os.path.join(dev, "idVendor")
        pid = os.path.join(dev, "idProduct")
        if os.path.isfile(vid) and os.path.isfile(pid):
            with open(vid) as f:
                v = f.read().strip()
            with open(pid) as f:
                p = f.read().strip()
            return v.lower(), p.lower()
        dev = os.path.dirname(dev)
    return None, None


def _candidates_sysfs(vendor_id, product_id):
    nodes = sorted(glob.glob("/dev/video*"),
                   key=lambda n: int("".join(filter(str.isdigit, n)) or 0))
    out = []
    for node in nodes:
        v, p = _sysfs_ids(node)
        if v == vendor_id.lower() and p == product_id.lower():
            out.append(node)
    return out


def _candidates_pyudev(vendor_id, product_id):
    import pyudev

    ctx = pyudev.Context()
    out = []
    for d in ctx.list_devices(subsystem="video4linux"):
        if d.device_node is None:
            continue
        if (d.get("ID_VENDOR_ID", "").lower() == vendor_id.lower()
                and d.get("ID_MODEL_ID", "").lower() == product_id.lower()):
            out.append(d.device_node)
    return sorted(out, key=lambda n: int("".join(filter(str.isdigit, n)) or 0))


def list_candidates(vendor_id=GLOWTACT_180_VID, product_id=GLOWTACT_180_PID):
    """Every /dev/videoN whose USB id matches, newest backend first. May be []."""
    try:
        nodes = _candidates_pyudev(vendor_id, product_id)
        if nodes:
            return nodes
    except Exception:
        pass                                 # pyudev absent or unusable
    return _candidates_sysfs(vendor_id, product_id)


def _yields_frames(index, tries=5):
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            return False
        for _ in range(tries):
            ok, frame = cap.read()
            if ok and frame is not None:
                return True
    finally:
        cap.release()
    return False


def find_glowtact(vendor_id=GLOWTACT_180_VID, product_id=GLOWTACT_180_PID,
                  which=0, probe=True):
    """Return the /dev/videoN index of a matching, frame-yielding GlowTact.

    ``which`` selects among multiple matching *capture* nodes (0 = first),
    useful only if two identical GlowTacts are plugged in at once. With
    ``probe=False`` the first matching node is returned without opening it
    (faster, but may hand back a metadata-only node that delivers no frames).

    Raises :class:`DeviceNotFound` if nothing matches / yields frames.
    """
    nodes = list_candidates(vendor_id, product_id)
    if not nodes:
        raise DeviceNotFound(
            f"no video device with USB id {vendor_id}:{product_id}. "
            f"Plugged in? On non-Linux, pass an explicit index instead.")
    indices = [int("".join(filter(str.isdigit, n)) or 0) for n in nodes]
    if not probe:
        return indices[which]
    good = [i for i in indices if _yields_frames(i)]
    if not good:
        raise DeviceNotFound(
            f"found {vendor_id}:{product_id} at {nodes}, but none delivered "
            f"frames (busy? try unplug/replug or a powered hub)")
    if which >= len(good):
        raise DeviceNotFound(
            f"requested match #{which} but only {len(good)} GlowTact(s) "
            f"deliver frames: {good}")
    return good[which]
