"""Shared loader for approved per-camera static table metadata."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from dataset_schema import VECTOR_DIM
from mask_geometry import apply_table_mask_pipeline, compute_mask_geometry, polygon_to_mask, resolve_overlaps

BASE_DIR = Path(__file__).parent
DEFAULT_APPROVED_TABLES_PATHS = (
    BASE_DIR / "approved_table_rectangles.json",
    BASE_DIR / "approved_tables.json",
)

_RAW_CONFIG_CACHE: tuple[str, float, dict[str, dict[str, Any]]] | None = None
_NORMALIZED_CONFIG_CACHE: tuple[str, float, dict[str, dict[str, Any]]] | None = None


def approved_tables_path() -> Path:
    raw_path = os.environ.get("APPROVED_TABLES_JSON_PATH") or os.environ.get("APPROVED_TABLE_RECTANGLES_JSON_PATH")
    if raw_path:
        return Path(raw_path)
    for candidate in DEFAULT_APPROVED_TABLES_PATHS:
        if candidate.exists():
            return candidate
    return DEFAULT_APPROVED_TABLES_PATHS[0]


def canonical_camera_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    match = re.search(r"IPC[\s_-]*(\d+)", text, re.IGNORECASE)
    if match:
        return f"IPC{int(match.group(1))}"

    match = re.fullmatch(r"(\d+)", text)
    if match:
        return f"IPC{int(match.group(1))}"
    return None


def detect_camera_from_filename(filename: str) -> str | None:
    name = Path(filename or "").name
    camera_id = canonical_camera_id(name)
    if camera_id:
        return camera_id

    match = re.match(r"^\s*(\d+)(?:[_-]|$)", name)
    if match:
        return f"IPC{int(match.group(1))}"
    return None


def _first_canonical_camera_id(*values: Any) -> str | None:
    for value in values:
        camera_id = canonical_camera_id(value)
        if camera_id:
            return camera_id
    return None


def _normalize_top_level(raw_payload: Any) -> dict[str, dict[str, Any]]:
    if isinstance(raw_payload, dict):
        if isinstance(raw_payload.get("cameras"), list):
            entries = raw_payload["cameras"]
        elif isinstance(raw_payload.get("cameras"), dict):
            entries = [
                {"camera_id": key, **value}
                for key, value in raw_payload["cameras"].items()
                if isinstance(value, dict)
            ]
        else:
            entries = []
            for key, value in raw_payload.items():
                if canonical_camera_id(key) and isinstance(value, dict):
                    entries.append({"camera_id": key, **value})
        normalized: dict[str, dict[str, Any]] = {}
        for entry in entries:
            camera_id = _first_canonical_camera_id(
                entry.get("camera_id"),
                entry.get("camera"),
                entry.get("id"),
                entry.get("name"),
                entry.get("camera_name"),
                entry.get("source_image"),
                entry.get("camera_number"),
            )
            if not camera_id:
                continue
            payload = dict(entry)
            payload.pop("camera_id", None)
            normalized[camera_id] = payload
        return normalized

    if isinstance(raw_payload, list):
        normalized = {}
        for entry in raw_payload:
            if not isinstance(entry, dict):
                continue
            camera_id = _first_canonical_camera_id(
                entry.get("camera_id"),
                entry.get("camera"),
                entry.get("id"),
                entry.get("name"),
                entry.get("camera_name"),
                entry.get("source_image"),
                entry.get("camera_number"),
            )
            if not camera_id:
                continue
            payload = dict(entry)
            normalized[camera_id] = payload
        return normalized

    return {}


def load_camera_configs(force_reload: bool = False) -> dict[str, dict[str, Any]]:
    global _RAW_CONFIG_CACHE
    path = approved_tables_path()
    if not path.exists():
        return {}

    mtime = path.stat().st_mtime
    path_key = str(path.resolve())
    if not force_reload and _RAW_CONFIG_CACHE and _RAW_CONFIG_CACHE[0] == path_key and _RAW_CONFIG_CACHE[1] == mtime:
        return _RAW_CONFIG_CACHE[2]

    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    configs = _normalize_top_level(raw_payload)
    _RAW_CONFIG_CACHE = (path_key, mtime, configs)
    return configs


def get_camera_config(camera_id: str) -> dict[str, Any] | None:
    canonical_id = canonical_camera_id(camera_id)
    if not canonical_id:
        return None
    return load_camera_configs().get(canonical_id)


def iter_camera_configs() -> list[tuple[str, dict[str, Any]]]:
    return sorted(load_camera_configs().items(), key=lambda item: item[0])


def _as_points(value: Any) -> list[list[float]] | None:
    if isinstance(value, dict):
        polygon = value.get("polygon")
        if polygon is not None:
            points = _as_points(polygon)
            if points:
                return points

        if {"center_x", "center_y", "width_px", "height_px", "angle_deg"}.issubset(value.keys()):
            cx = float(value["center_x"])
            cy = float(value["center_y"])
            width = float(value["width_px"])
            height = float(value["height_px"])
            angle_radians = math.radians(float(value["angle_deg"]))
            cos_theta = math.cos(angle_radians)
            sin_theta = math.sin(angle_radians)
            half_w = width / 2.0
            half_h = height / 2.0
            points = []
            for dx, dy in [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]:
                x = cx + (dx * cos_theta) - (dy * sin_theta)
                y = cy + (dx * sin_theta) + (dy * cos_theta)
                points.append([float(x), float(y)])
            return points

        if {"x1", "y1", "x2", "y2"}.issubset(value.keys()):
            x1 = float(value["x1"])
            y1 = float(value["y1"])
            x2 = float(value["x2"])
            y2 = float(value["y2"])
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return None

    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            x1, y1, x2, y2 = [float(item) for item in value]
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        points: list[list[float]] = []
        for item in value:
            if isinstance(item, dict) and {"x", "y"}.issubset(item.keys()):
                points.append([float(item["x"]), float(item["y"])])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                points.append([float(item[0]), float(item[1])])
        if len(points) >= 3:
            return points
    return None


def _rotated_box_points(rotated_bbox: dict[str, Any]) -> list[list[float]] | None:
    corners = _as_points(rotated_bbox.get("corners"))
    if corners:
        return corners

    center = rotated_bbox.get("center")
    size = rotated_bbox.get("size")
    angle_degrees = rotated_bbox.get("angle")
    if not center or not size or angle_degrees is None:
        return None

    cx, cy = float(center[0]), float(center[1])
    width, height = float(size[0]), float(size[1])
    half_w = width / 2.0
    half_h = height / 2.0
    angle_radians = math.radians(float(angle_degrees))
    cos_theta = math.cos(angle_radians)
    sin_theta = math.sin(angle_radians)

    points = []
    for dx, dy in [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]:
        x = cx + (dx * cos_theta) - (dy * sin_theta)
        y = cy + (dx * sin_theta) + (dy * cos_theta)
        points.append([float(x), float(y)])
    return points


def _extract_table_points(table_payload: dict[str, Any]) -> list[list[float]] | None:
    for key in (
        "tight_rect",
        "rectangle",
        "tight_bbox",
        "bbox",
        "hull_dense_points",
        "hull_ui_points",
        "polygon",
        "points",
        "corners",
        "tight_hull_anchor_polygon",
        "vertices",
        "ideal_blob_polygon",
    ):
        points = _as_points(table_payload.get(key))
        if points:
            return points

    rotated_bbox = table_payload.get("rotated_bbox")
    if isinstance(rotated_bbox, dict):
        points = _rotated_box_points(rotated_bbox)
        if points:
            return points

    for key in ("expanded_bbox",):
        points = _as_points(table_payload.get(key))
        if points:
            return points
    return None


def _extract_expanded_table_points(table_payload: dict[str, Any]) -> list[list[float]] | None:
    for key in (
        "zone_rect",
        "expanded_bbox",
        "expanded_rect",
        "expanded_hull_dense_points",
        "expanded_hull_ui_points",
        "ideal_blob_polygon",
        "ideal_blob_ui_points",
    ):
        points = _as_points(table_payload.get(key))
        if points:
            return points
    return None


def _resize_vector(vector: np.ndarray, target_dim: int = VECTOR_DIM) -> np.ndarray:
    flat = vector.astype(np.float32).reshape(-1)
    if flat.size == target_dim:
        norm = float(np.linalg.norm(flat))
        return flat if norm <= 1e-8 else (flat / norm).astype(np.float32)
    if flat.size == 0:
        return np.zeros((target_dim,), dtype=np.float32)
    old_positions = np.linspace(0.0, 1.0, num=flat.size, dtype=np.float32)
    new_positions = np.linspace(0.0, 1.0, num=target_dim, dtype=np.float32)
    resized = np.interp(new_positions, old_positions, flat).astype(np.float32)
    norm = float(np.linalg.norm(resized))
    return resized if norm <= 1e-8 else (resized / norm).astype(np.float32)


def _extract_table_vector(table_payload: dict[str, Any]) -> tuple[np.ndarray | None, str | None]:
    for key in ("vector", "table_vector", "tight_vector", "table_vec", "table_vec_tight", "vec"):
        value = table_payload.get(key)
        if isinstance(value, dict):
            value = value.get("values") or value.get("data")
        if value is None:
            continue
        try:
            return _resize_vector(np.asarray(value, dtype=np.float32)), f"approved_tables_json:{key}"
        except Exception:
            continue
    return None, None


def _shoelace_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    xs = points[:, 0]
    ys = points[:, 1]
    return float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) * 0.5)


def _geometry_vector(points: list[list[float]], frame_width: int | None, frame_height: int | None) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float32)
    max_x = float(np.max(point_array[:, 0])) if point_array.size else 1.0
    max_y = float(np.max(point_array[:, 1])) if point_array.size else 1.0
    width = max(float(frame_width or 0), max_x, 1.0)
    height = max(float(frame_height or 0), max_y, 1.0)
    normalized_points = np.stack([point_array[:, 0] / width, point_array[:, 1] / height], axis=1).reshape(-1)
    x1 = float(np.min(point_array[:, 0])) if point_array.size else 0.0
    y1 = float(np.min(point_array[:, 1])) if point_array.size else 0.0
    x2 = float(np.max(point_array[:, 0])) if point_array.size else 0.0
    y2 = float(np.max(point_array[:, 1])) if point_array.size else 0.0
    bbox_width = max(x2 - x1, 1.0)
    bbox_height = max(y2 - y1, 1.0)
    centroid_x = float(np.mean(point_array[:, 0])) if point_array.size else 0.0
    centroid_y = float(np.mean(point_array[:, 1])) if point_array.size else 0.0
    bbox_diag = math.hypot(bbox_width, bbox_height)
    frame_diag = max(math.hypot(width, height), 1.0)
    area_ratio = _shoelace_area(point_array) / max(width * height, 1.0)
    stats = np.array(
        [
            centroid_x / width,
            centroid_y / height,
            bbox_width / width,
            bbox_height / height,
            bbox_diag / frame_diag,
            area_ratio,
        ],
        dtype=np.float32,
    )
    return _resize_vector(np.concatenate([normalized_points.astype(np.float32), stats]))


def _extract_rect_params(rect_dict: Any) -> dict[str, Any] | None:
    """Extract rotated-rectangle fields, preserving polygon geometry when available."""
    if not isinstance(rect_dict, dict):
        return None
    required = {"center_x", "center_y", "width_px", "height_px", "angle_deg"}
    if not required.issubset(rect_dict.keys()):
        return None
    rect = {
        "center_x": float(rect_dict["center_x"]),
        "center_y": float(rect_dict["center_y"]),
        "width_px": float(rect_dict["width_px"]),
        "height_px": float(rect_dict["height_px"]),
        "angle_deg": float(rect_dict["angle_deg"]),
    }
    polygon = _as_points(rect_dict.get("polygon"))
    if polygon:
        rect["polygon"] = [[float(x), float(y)] for x, y in polygon]
    return rect


def _extract_bbox_xywh(bbox_value: Any) -> list[float] | None:
    if isinstance(bbox_value, dict):
        if {"x", "y", "width", "height"}.issubset(bbox_value.keys()):
            return [
                float(bbox_value["x"]),
                float(bbox_value["y"]),
                float(bbox_value["width"]),
                float(bbox_value["height"]),
            ]
        if {"x1", "y1", "x2", "y2"}.issubset(bbox_value.keys()):
            x1 = float(bbox_value["x1"])
            y1 = float(bbox_value["y1"])
            x2 = float(bbox_value["x2"])
            y2 = float(bbox_value["y2"])
            return [x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)]
        return None

    if isinstance(bbox_value, (list, tuple)) and len(bbox_value) >= 4 and all(isinstance(item, (int, float)) for item in bbox_value[:4]):
        return [float(bbox_value[0]), float(bbox_value[1]), float(bbox_value[2]), float(bbox_value[3])]

    return None


def _scale_rect_params(
    rect: dict[str, Any],
    scale_x: float,
    scale_y: float,
) -> dict[str, Any]:
    """Scale a parametric rotated rectangle. Angle is scale-invariant."""
    scaled = {
        "center_x": rect["center_x"] * scale_x,
        "center_y": rect["center_y"] * scale_y,
        "width_px": rect["width_px"] * scale_x,
        "height_px": rect["height_px"] * scale_y,
        "angle_deg": rect["angle_deg"],
    }
    polygon = rect.get("polygon")
    if polygon:
        scaled["polygon"] = [[float(x) * scale_x, float(y) * scale_y] for x, y in polygon]
    return scaled


def _scale_bbox_xywh(
    bbox_xywh: list[float] | None,
    scale_x: float,
    scale_y: float,
) -> list[float] | None:
    if not bbox_xywh or len(bbox_xywh) < 4:
        return None
    return [
        float(bbox_xywh[0]) * scale_x,
        float(bbox_xywh[1]) * scale_y,
        float(bbox_xywh[2]) * scale_x,
        float(bbox_xywh[3]) * scale_y,
    ]


def _normalized_table_track_id(table_payload: dict[str, Any], index: int) -> str:
    raw_track_id = table_payload.get("table_track_id")
    if raw_track_id:
        text = str(raw_track_id).strip()
        return text if text.startswith("table_") else f"table_{text}"

    raw_id = table_payload.get("id")
    if raw_id is None:
        raw_id = table_payload.get("mask_id")
    if raw_id is None:
        raw_id = index
    return f"table_{raw_id}"


def _bbox_xyxy_from_points(points: list[list[float]]) -> tuple[int, int, int, int]:
    point_array = np.asarray(points, dtype=np.float32)
    if point_array.size == 0:
        return (0, 0, 0, 0)
    x1 = int(math.floor(float(np.min(point_array[:, 0]))))
    y1 = int(math.floor(float(np.min(point_array[:, 1]))))
    x2 = int(math.ceil(float(np.max(point_array[:, 0])))) + 1
    y2 = int(math.ceil(float(np.max(point_array[:, 1])))) + 1
    return (x1, y1, x2, y2)


def _normalize_camera_config(camera_id: str, camera_payload: dict[str, Any]) -> dict[str, Any]:
    frame_width = camera_payload.get("frame_width") or camera_payload.get("image_width") or camera_payload.get("width")
    frame_height = camera_payload.get("frame_height") or camera_payload.get("image_height") or camera_payload.get("height")
    tables_payload = camera_payload.get("tables") or camera_payload.get("segments") or []

    normalized_tables: list[dict[str, Any]] = []
    for index, table_payload in enumerate(tables_payload):
        if not isinstance(table_payload, dict):
            continue
        points = _extract_table_points(table_payload)
        if not points:
            continue
        track_id = _normalized_table_track_id(table_payload, index)
        vector, vector_source = _extract_table_vector(table_payload)
        if vector is None:
            vector = _geometry_vector(points, frame_width, frame_height)
            vector_source = "geometry_fallback"
        bbox_xywh = _extract_bbox_xywh(table_payload.get("bbox"))
        if bbox_xywh is None:
            x1, y1, x2, y2 = _bbox_xyxy_from_points(points)
            bbox_xywh = [float(x1), float(y1), float(max(x2 - x1, 0)), float(max(y2 - y1, 0))]
        mask_id = table_payload.get("mask_id")
        if mask_id is None:
            mask_id = table_payload.get("id", index)
        normalized_tables.append(
            {
                "id": table_payload.get("id", index),
                "mask_id": mask_id,
                "track_id": track_id,
                "table_label": str(table_payload.get("label") or track_id),
                "bbox_xywh": bbox_xywh,
                "points": [[float(x), float(y)] for x, y in points],
                "expanded_points": _extract_expanded_table_points(table_payload),
                "tight_rect": _extract_rect_params(table_payload.get("tight_rect")),
                "zone_rect": _extract_rect_params(table_payload.get("zone_rect")),
                "vector": vector,
                "vector_source": vector_source,
                "compactness": float(table_payload.get("compactness")) if table_payload.get("compactness") is not None else None,
                "raw": dict(table_payload),
            }
        )

    return {
        "camera_id": camera_id,
        "frame_width": int(frame_width) if frame_width else None,
        "frame_height": int(frame_height) if frame_height else None,
        "video_name": camera_payload.get("video_name"),
        "tables": normalized_tables,
    }


def load_normalized_camera_configs(force_reload: bool = False) -> dict[str, dict[str, Any]]:
    global _NORMALIZED_CONFIG_CACHE
    path = approved_tables_path()
    configs = load_camera_configs(force_reload=force_reload)
    if not path.exists():
        return {}

    mtime = path.stat().st_mtime
    path_key = str(path.resolve())
    if (
        not force_reload
        and _NORMALIZED_CONFIG_CACHE
        and _NORMALIZED_CONFIG_CACHE[0] == path_key
        and _NORMALIZED_CONFIG_CACHE[1] == mtime
    ):
        return _NORMALIZED_CONFIG_CACHE[2]

    normalized = {
        camera_id: _normalize_camera_config(camera_id, payload)
        for camera_id, payload in configs.items()
    }
    _NORMALIZED_CONFIG_CACHE = (path_key, mtime, normalized)
    return normalized


def get_normalized_camera_config(camera_id: str) -> dict[str, Any] | None:
    canonical_id = canonical_camera_id(camera_id)
    if not canonical_id:
        return None
    return load_normalized_camera_configs().get(canonical_id)


def _legacy_table_entry(table: dict[str, Any]) -> dict[str, Any]:
    bbox_xyxy = _bbox_xyxy_from_points(table["points"])
    return {
        "id": table.get("id"),
        "table_track_id": table.get("track_id"),
        "saved": True,
        "bbox": {
            "x1": bbox_xyxy[0],
            "y1": bbox_xyxy[1],
            "x2": bbox_xyxy[2],
            "y2": bbox_xyxy[3],
        },
        "polygon": table["points"],
        "tight_hull_anchor_polygon": table["points"],
    }


def get_legacy_camera_config(camera_id: str) -> dict[str, Any] | None:
    normalized = get_normalized_camera_config(camera_id)
    if not normalized:
        return None
    return {
        "camera_id": normalized["camera_id"],
        "frame_width": normalized.get("frame_width"),
        "frame_height": normalized.get("frame_height"),
        "video_name": normalized.get("video_name"),
        "tables": [_legacy_table_entry(table) for table in normalized.get("tables", [])],
    }


def _scale_points(
    points: list[list[float]],
    *,
    source_width: int | None,
    source_height: int | None,
    target_shape: tuple[int, int],
) -> list[list[float]]:
    target_height, target_width = target_shape
    if not source_width or not source_height:
        return [[float(x), float(y)] for x, y in points]

    scale_x = float(target_width) / float(source_width)
    scale_y = float(target_height) / float(source_height)
    return [[float(x) * scale_x, float(y) * scale_y] for x, y in points]


def build_static_tables_for_frame(
    camera_id: str,
    image_shape: tuple[int, int],
    *,
    expansion_strength: float = 1.0,
) -> list[dict[str, Any]]:
    config = get_normalized_camera_config(camera_id)
    if not config:
        raise KeyError(camera_id)

    target_height, target_width = image_shape
    base_tables: list[dict[str, Any]] = []
    tight_masks: list[np.ndarray] = []

    source_width = config.get("frame_width")
    source_height = config.get("frame_height")
    scale_x = float(target_width) / float(source_width) if source_width else 1.0
    scale_y = float(target_height) / float(source_height) if source_height else 1.0

    for table in config.get("tables", []):
        scaled_points = _scale_points(
            table["points"],
            source_width=source_width,
            source_height=source_height,
            target_shape=image_shape,
        )
        tight_mask = polygon_to_mask(scaled_points, (target_height, target_width)).astype(bool)
        if not tight_mask.any():
            continue
        tight_masks.append(tight_mask)

        tight_rect = table.get("tight_rect")
        if tight_rect:
            tight_rect = _scale_rect_params(tight_rect, scale_x, scale_y)
        zone_rect = table.get("zone_rect")
        if zone_rect:
            zone_rect = _scale_rect_params(zone_rect, scale_x, scale_y)

        base_tables.append(
            {
                "label": "table",
                "mask_id": table.get("mask_id"),
                "table_label": str(table.get("table_label") or table["track_id"]),
                "score": 1.0,
                "vector": table["vector"].astype(np.float32),
                "vector_source": str(table["vector_source"]),
                "track_id": str(table["track_id"]),
                "bbox_xywh": _scale_bbox_xywh(table.get("bbox_xywh"), scale_x, scale_y),
                "compactness_raw": table.get("compactness"),
                "points": scaled_points,
                "expanded_points": _scale_points(
                    table["expanded_points"],
                    source_width=source_width,
                    source_height=source_height,
                    target_shape=image_shape,
                )
                if table.get("expanded_points")
                else None,
                "tight_rect": tight_rect,
                "zone_rect": zone_rect,
            }
        )

    tables: list[dict[str, Any]] = []
    support_masks: list[np.ndarray] = []
    expanded_masks: list[np.ndarray] = []
    any_metadata_expanded = False
    for index, base_table in enumerate(base_tables):
        expanded_points = base_table.get("expanded_points")
        if expanded_points:
            tight_mask = tight_masks[index].astype(bool)
            expanded_mask = polygon_to_mask(expanded_points, (target_height, target_width)).astype(bool)
            geometry = compute_mask_geometry(tight_mask)
            pipeline = {
                "tight_mask": tight_mask,
                "support_mask": expanded_mask.astype(bool),
                "expanded_mask": expanded_mask.astype(bool),
            }
            any_metadata_expanded = True
        else:
            other_masks = tight_masks[:index] + tight_masks[index + 1 :]
            pipeline = apply_table_mask_pipeline(
                tight_masks[index],
                other_anchor_masks=other_masks,
                strength=expansion_strength,
            )
            geometry = compute_mask_geometry(pipeline["tight_mask"])
        table_payload = dict(base_table)
        table_payload.update(
            {
                "bbox_xyxy": geometry.bbox_xyxy,
                "centroid_xy": geometry.centroid_xy,
                "tight_mask": pipeline["tight_mask"].astype(bool),
                "support_mask": pipeline["support_mask"].astype(bool),
                "expanded_mask": pipeline["expanded_mask"].astype(bool),
                "compactness": float(base_table["compactness_raw"])
                if base_table.get("compactness_raw") is not None
                else float(geometry.compactness),
                "hull_points": base_table.get("points") or geometry.hull_points,
                "expanded_bbox_xyxy": geometry.bbox_xyxy,
            }
        )
        tables.append(table_payload)
        support_masks.append(table_payload["support_mask"])
        expanded_masks.append(table_payload["expanded_mask"])

    if expanded_masks and not any_metadata_expanded:
        resolved = resolve_overlaps(expanded_masks, support_masks)
        for table_payload, mask in zip(tables, resolved):
            table_payload["expanded_mask"] = mask.astype(bool)
            ys, xs = np.nonzero(mask)
            if xs.size:
                table_payload["expanded_bbox_xyxy"] = (
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                )
            else:
                table_payload["expanded_bbox_xyxy"] = table_payload["bbox_xyxy"]
    elif expanded_masks:
        for table_payload, mask in zip(tables, expanded_masks):
            ys, xs = np.nonzero(mask)
            if xs.size:
                table_payload["expanded_bbox_xyxy"] = (
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                )
            else:
                table_payload["expanded_bbox_xyxy"] = table_payload["bbox_xyxy"]

    return tables
