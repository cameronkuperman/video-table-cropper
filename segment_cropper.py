"""Image crop helpers for segmentation previews."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


class SegmentCropError(ValueError):
    """Raised when a segment crop cannot be generated."""


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _normalize_polygon(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, (list, tuple)):
        return []
    points: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, dict) and {"x", "y"}.issubset(item.keys()):
            points.append((float(item["x"]), float(item["y"])))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            points.append((float(item[0]), float(item[1])))
    return points


def rect_polygon(rect: dict[str, Any]) -> list[tuple[float, float]]:
    polygon = _normalize_polygon(rect.get("polygon"))
    if len(polygon) >= 3:
        return polygon

    corners = _normalize_polygon(rect.get("corners"))
    if len(corners) >= 3:
        return corners

    center = rect.get("center")
    size = rect.get("size")
    angle = rect.get("angle")

    if center is None and {"center_x", "center_y"}.issubset(rect.keys()):
        center = (rect["center_x"], rect["center_y"])
    if size is None and {"width_px", "height_px"}.issubset(rect.keys()):
        size = (rect["width_px"], rect["height_px"])
    if angle is None and "angle_deg" in rect:
        angle = rect["angle_deg"]

    if not center or not size:
        return []

    cx = float(center[0])
    cy = float(center[1])
    width = float(size[0])
    height = float(size[1])
    theta = math.radians(float(angle or 0.0))

    ux = math.cos(theta)
    uy = math.sin(theta)
    vx = -math.sin(theta)
    vy = math.cos(theta)

    half_long = width / 2.0
    half_short = height / 2.0
    corners_local = [
        (-half_long, -half_short),
        (half_long, -half_short),
        (half_long, half_short),
        (-half_long, half_short),
    ]
    return [
        (cx + (long_off * ux) + (short_off * vx), cy + (long_off * uy) + (short_off * vy))
        for long_off, short_off in corners_local
    ]


def crop_with_bbox(image: Image.Image, bbox: dict[str, Any]) -> Image.Image:
    x1 = int(bbox["x1"])
    y1 = int(bbox["y1"])
    x2 = int(bbox["x2"])
    y2 = int(bbox["y2"])

    x1 = _clamp(x1, 0, image.width)
    y1 = _clamp(y1, 0, image.height)
    x2 = _clamp(x2, 0, image.width)
    y2 = _clamp(y2, 0, image.height)

    if x2 <= x1 or y2 <= y1:
        raise SegmentCropError("Invalid bbox after clamping")

    return image.crop((x1, y1, x2, y2))


def crop_with_rotated_bbox(image: Image.Image, rotated_bbox: dict[str, Any]) -> Image.Image:
    center = rotated_bbox.get("center")
    size = rotated_bbox.get("size")
    if not center or not size:
        raise SegmentCropError("rotated_bbox missing center or size")

    cx = float(center[0])
    cy = float(center[1])
    width = int(round(float(size[0])))
    height = int(round(float(size[1])))
    angle = float(rotated_bbox.get("angle", 0.0))

    if width <= 0 or height <= 0:
        raise SegmentCropError("rotated_bbox size must be positive")

    # Rotate around center to align box with image axes, then crop center rectangle.
    aligned = image.rotate(-angle, center=(cx, cy), resample=Image.BICUBIC)

    x1 = int(round(cx - width / 2))
    y1 = int(round(cy - height / 2))
    x2 = x1 + width
    y2 = y1 + height

    x1 = _clamp(x1, 0, aligned.width)
    y1 = _clamp(y1, 0, aligned.height)
    x2 = _clamp(x2, 0, aligned.width)
    y2 = _clamp(y2, 0, aligned.height)

    if x2 <= x1 or y2 <= y1:
        raise SegmentCropError("Invalid rotated crop bounds after clamping")

    return aligned.crop((x1, y1, x2, y2))


def crop_with_polygonal_rect(image: Image.Image, rect: dict[str, Any]) -> Image.Image:
    polygon = rect_polygon(rect)
    if len(polygon) < 3:
        raise SegmentCropError("rect is missing polygon geometry")

    points = np.asarray(polygon, dtype=np.float32)
    x0 = _clamp(int(math.floor(float(np.min(points[:, 0])))), 0, image.width - 1)
    y0 = _clamp(int(math.floor(float(np.min(points[:, 1])))), 0, image.height - 1)
    x1 = _clamp(int(math.ceil(float(np.max(points[:, 0])))), 0, image.width - 1)
    y1 = _clamp(int(math.ceil(float(np.max(points[:, 1])))), 0, image.height - 1)

    if x1 < x0 or y1 < y0:
        raise SegmentCropError("Invalid polygon crop bounds after clamping")

    crop = image.crop((x0, y0, x1 + 1, y1 + 1))
    local_points = [(float(x - x0), float(y - y0)) for x, y in polygon]
    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).polygon(local_points, fill=255)
    masked = Image.new("RGB", crop.size, (0, 0, 0))
    masked.paste(crop.convert("RGB"), mask=mask)
    return masked


def crop_segment_from_path(image_path: Path, segment: dict[str, Any], output_path: Path) -> Path:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if segment.get("bbox"):
            cropped = crop_with_bbox(image, segment["bbox"])
        elif segment.get("rotated_bbox"):
            cropped = crop_with_polygonal_rect(image, segment["rotated_bbox"])
        else:
            raise SegmentCropError("Segment is missing bbox/rotated_bbox")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path, format="JPEG", quality=92)
    return output_path
