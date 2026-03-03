"""Utilities for loading and validating segmentation JSON."""

from __future__ import annotations

import json
from typing import Any


class SegmentParserError(ValueError):
    """Raised when segment JSON is missing required fields."""


def _normalize_bbox(raw_bbox: dict[str, Any]) -> dict[str, int]:
    try:
        x1 = int(raw_bbox["x1"])
        y1 = int(raw_bbox["y1"])
        x2 = int(raw_bbox["x2"])
        y2 = int(raw_bbox["y2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SegmentParserError("bbox must include integer x1, y1, x2, y2") from exc

    if x2 <= x1 or y2 <= y1:
        raise SegmentParserError("bbox must satisfy x2 > x1 and y2 > y1")

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _normalize_rotated_bbox(raw: dict[str, Any]) -> dict[str, Any]:
    try:
        center = [float(raw["center"][0]), float(raw["center"][1])]
        size = [float(raw["size"][0]), float(raw["size"][1])]
        angle = float(raw.get("angle", 0.0))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise SegmentParserError("rotated_bbox requires center, size, and optional angle") from exc

    normalized: dict[str, Any] = {
        "center": center,
        "size": size,
        "angle": angle,
    }

    corners = raw.get("corners")
    if corners and isinstance(corners, list):
        normalized_corners = []
        for corner in corners:
            if not isinstance(corner, list) or len(corner) < 2:
                continue
            normalized_corners.append([float(corner[0]), float(corner[1])])
        if normalized_corners:
            normalized["corners"] = normalized_corners

    return normalized


def parse_segment_json(raw_data: bytes | str | dict[str, Any]) -> list[dict[str, Any]]:
    """Parse segment config from JSON bytes/string/dict into normalized list."""
    if isinstance(raw_data, dict):
        data = raw_data
    else:
        if isinstance(raw_data, bytes):
            raw_data = raw_data.decode("utf-8")
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            raise SegmentParserError(f"Invalid JSON: {exc}") from exc

    entries = data.get("segments") or data.get("tables")
    if not isinstance(entries, list) or not entries:
        raise SegmentParserError("segment.json must contain a non-empty 'segments' or 'tables' array")

    normalized_segments: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        if not entry.get("saved", True):
            continue
        if entry.get("skip_reason"):
            continue

        segment_id = entry.get("id", idx)
        segment: dict[str, Any] = {"segment_id": str(segment_id)}

        if "bbox" in entry and entry["bbox"]:
            segment["bbox"] = _normalize_bbox(entry["bbox"])
        elif "rotated_bbox" in entry and entry["rotated_bbox"]:
            segment["rotated_bbox"] = _normalize_rotated_bbox(entry["rotated_bbox"])
        else:
            raise SegmentParserError(f"Segment {segment_id} is missing bbox/rotated_bbox")

        normalized_segments.append(segment)

    if not normalized_segments:
        raise SegmentParserError("No usable segments found in segment.json")

    return normalized_segments
