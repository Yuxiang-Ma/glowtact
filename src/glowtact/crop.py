"""Pad crop geometry: fixed-fraction edge trims and rotate-then-crop bounds.

The gel's illumination falls off and the housing intrudes near the frame edge,
so the outer border carries vignetting rather than contact signal. A crop is
either a fixed fraction from each side, or explicit bounds picked in a GUI and
stored in a ``crop_bounds.yaml`` (optionally with an in-plane rotation that
squares up a pad mounted at a slight angle -- applied BEFORE the axis-aligned
crop, so the bounds are in rotated-image coordinates).
"""


def crop_bounds(width, height, fraction):
    """Pixel bounds keeping the central (1 - 2*fraction) of each dimension."""
    if not 0 <= fraction < 0.5:
        raise ValueError("fraction must be in [0, 0.5)")
    dx = int(round(width * fraction))
    dy = int(round(height * fraction))
    return dx, dy, width - dx, height - dy


def crop_image(image, fraction):
    height, width = image.shape[:2]
    x0, y0, x1, y1 = crop_bounds(width, height, fraction)
    return image[y0:y1, x0:x1]


def rotate_image(image, rotation_deg):
    """In-plane rotation about the image centre; black fills the exposed corners.

    Positive is counter-clockwise (OpenCV convention).
    """
    import cv2

    if not rotation_deg:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0),
                                     rotation_deg, 1.0)
    return cv2.warpAffine(image, matrix, (width, height),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def bounds_from_file(path):
    """Read ((x0,y0,x1,y1), source_width, source_height, rotation_deg)."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    missing = [k for k in ("x0", "y0", "x1", "y1") if k not in data]
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    return ((data["x0"], data["y0"], data["x1"], data["y1"]),
            data.get("source_width"), data.get("source_height"),
            float(data.get("rotation_deg", 0.0)))


def apply_crop(image, crop):
    """Rotation-then-crop from a `bounds_from_file` tuple; None passes through."""
    if crop is None:
        return image
    bounds, _w, _h, rotation = crop
    x0, y0, x1, y1 = bounds
    return rotate_image(image, rotation)[y0:y1, x0:x1]
