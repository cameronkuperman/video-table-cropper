"""Mask cleanup, support, and expansion helpers for table geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull


@dataclass
class MaskGeometry:
    area: int
    centroid_xy: tuple[float, float]
    bbox_xyxy: tuple[int, int, int, int]
    axis_ratio: float
    angle_radians: float
    compactness: float
    hull_points: list[list[float]]


def _disk(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _mask_points(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))


def polygon_to_mask(points: Iterable[tuple[float, float]] | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(image)
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) >= 3:
        draw.polygon(pts, outline=1, fill=1)
    return np.array(image, dtype=bool)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    labeled, count = ndimage.label(mask)
    if count == 0:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, labeled, range(1, count + 1))
    keep = np.zeros(count + 1, dtype=bool)
    for idx, size in enumerate(sizes, start=1):
        if int(size) >= min_area:
            keep[idx] = True
    return keep[labeled]


def fill_holes(mask: np.ndarray) -> np.ndarray:
    return ndimage.binary_fill_holes(mask).astype(bool)


def regularize_mask(mask: np.ndarray, diag: float, close_scale: float = 0.0008, open_scale: float = 0.00045) -> np.ndarray:
    close_radius = max(1, round(diag * close_scale))
    open_radius = max(1, round(diag * open_scale))
    min_area = max(14, round(diag * 0.015))
    cleaned = remove_small_components(mask.astype(bool), min_area)
    cleaned = ndimage.binary_closing(cleaned, structure=_disk(close_radius))
    cleaned = ndimage.binary_opening(cleaned, structure=_disk(open_radius))
    return fill_holes(cleaned)


def principal_axis(mask: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    points = _mask_points(mask)
    if points.shape[0] < 3:
        return 0.0, 1.0, np.array([1.0, 0.0]), np.array([0.0, 1.0])
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    long_axis = eigvecs[:, 0]
    short_axis = eigvecs[:, 1]
    angle = float(math.atan2(long_axis[1], long_axis[0]))
    denom = max(float(eigvals[1]), 1e-6)
    axis_ratio = max(float(math.sqrt(max(eigvals[0], 1e-6) / denom)), 1.0)
    return angle, axis_ratio, long_axis.astype(np.float32), short_axis.astype(np.float32)


def compute_mask_geometry(mask: np.ndarray) -> MaskGeometry:
    area = int(mask.sum())
    centroid = mask_centroid(mask)
    bbox = mask_bbox(mask)
    angle, axis_ratio, _, _ = principal_axis(mask)
    perimeter = float(np.count_nonzero(mask ^ ndimage.binary_erosion(mask)))
    compactness = 0.0 if perimeter <= 0 else float(4 * math.pi * area / max(perimeter * perimeter, 1.0))
    hull_points = convex_hull_points(mask)
    return MaskGeometry(
        area=area,
        centroid_xy=centroid,
        bbox_xyxy=bbox,
        axis_ratio=axis_ratio,
        angle_radians=angle,
        compactness=compactness,
        hull_points=hull_points,
    )


def convex_hull_points(mask: np.ndarray) -> list[list[float]]:
    points = _mask_points(mask)
    if points.shape[0] < 3:
        return [[float(x), float(y)] for x, y in points.tolist()]
    hull = ConvexHull(points)
    return [[float(points[idx, 0]), float(points[idx, 1])] for idx in hull.vertices]


def _project_points(mask: np.ndarray, long_axis: np.ndarray, short_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = _mask_points(mask)
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    long_coords = centered @ long_axis
    short_coords = centered @ short_axis
    return long_coords.astype(np.float32), short_coords.astype(np.float32)


def _reconstruct_band_mask(
    centroid_xy: tuple[float, float],
    long_axis: np.ndarray,
    short_axis: np.ndarray,
    bins: list[tuple[float, float, float]],
    shape: tuple[int, int],
) -> np.ndarray:
    if not bins:
        return np.zeros(shape, dtype=bool)

    polygon: list[tuple[float, float]] = []
    for center, min_short, max_short in bins:
        center_xy = np.array(centroid_xy, dtype=np.float32) + (long_axis * center)
        polygon.append(tuple((center_xy + short_axis * min_short).tolist()))
    for center, min_short, max_short in reversed(bins):
        center_xy = np.array(centroid_xy, dtype=np.float32) + (long_axis * center)
        polygon.append(tuple((center_xy + short_axis * max_short).tolist()))
    return polygon_to_mask(polygon, shape)


def _elongated_support(mask: np.ndarray, diag: float, centroid_xy: tuple[float, float], long_axis: np.ndarray, short_axis: np.ndarray) -> np.ndarray:
    long_coords, short_coords = _project_points(mask, long_axis, short_axis)
    if long_coords.size == 0:
        return mask

    bin_px = max(2, round(diag * 0.0025))
    fill_gap_fraction = 0.10
    bins = np.arange(long_coords.min(), long_coords.max() + bin_px, bin_px)
    occupied: list[tuple[float, float, float]] = []

    for idx in range(len(bins) - 1):
        lo = bins[idx]
        hi = bins[idx + 1]
        in_bin = (long_coords >= lo) & (long_coords < hi)
        if not np.any(in_bin):
            continue
        occupied.append((float((lo + hi) / 2), float(short_coords[in_bin].min()), float(short_coords[in_bin].max())))

    if not occupied:
        return mask

    max_gap = fill_gap_fraction * max(long_coords.max() - long_coords.min(), 1.0)
    filled: list[tuple[float, float, float]] = [occupied[0]]
    for current in occupied[1:]:
        prev = filled[-1]
        gap = current[0] - prev[0]
        if gap <= max_gap:
            fill_center = (prev[0] + current[0]) / 2
            fill_min = min(prev[1], current[1])
            fill_max = max(prev[2], current[2])
            filled.append((fill_center, fill_min, fill_max))
        filled.append(current)

    support = _reconstruct_band_mask(centroid_xy, long_axis, short_axis, filled, mask.shape)
    return support


def _compact_support(mask: np.ndarray) -> np.ndarray:
    hull = convex_hull_points(mask)
    if len(hull) < 3:
        return mask
    return polygon_to_mask(hull, mask.shape)


def _rescue_support(mask: np.ndarray, centroid_xy: tuple[float, float], long_axis: np.ndarray, short_axis: np.ndarray, visible_support: np.ndarray) -> np.ndarray:
    long_coords, short_coords = _project_points(mask, long_axis, short_axis)
    if long_coords.size == 0:
        return visible_support

    visible_long, visible_short = _project_points(visible_support, long_axis, short_axis)
    if visible_long.size == 0:
        return visible_support

    visible_min = float(visible_long.min())
    visible_max = float(visible_long.max())
    visible_span = max(visible_max - visible_min, 1.0)
    short_extent = max(float(np.median(np.abs(short_coords))) * 2.0, 2.0)

    rescue_min = visible_min - 0.22 * visible_span
    rescue_max = visible_max + 0.22 * visible_span
    bins = [
        (rescue_min, -short_extent / 2, short_extent / 2),
        (visible_min, -short_extent / 2, short_extent / 2),
        (visible_max, -short_extent / 2, short_extent / 2),
        (rescue_max, -short_extent / 2, short_extent / 2),
    ]
    rescue_mask = _reconstruct_band_mask(centroid_xy, long_axis, short_axis, bins, mask.shape)
    return fill_holes(rescue_mask | visible_support)


def build_support_mask(anchor_mask: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    anchor_mask = anchor_mask.astype(bool)
    if not anchor_mask.any():
        return anchor_mask, {"axis_ratio": 1.0, "compactness": 0.0}

    diag = math.hypot(anchor_mask.shape[1], anchor_mask.shape[0])
    regularized = regularize_mask(anchor_mask, diag)
    geometry = compute_mask_geometry(regularized)
    _, axis_ratio, long_axis, short_axis = principal_axis(regularized)

    if axis_ratio >= 1.25:
        support = _elongated_support(regularized, diag, geometry.centroid_xy, long_axis, short_axis)
        occupancy = float(regularized.sum() / max(support.sum(), 1))
        if occupancy < 0.55 or support.sum() > 1.65 * max(regularized.sum(), 1):
            support = regularized
    else:
        support = _compact_support(regularized)
        occupancy = float(regularized.sum() / max(support.sum(), 1))
        if occupancy < 0.60 or support.sum() > 1.45 * max(regularized.sum(), 1):
            support = regularized

    rescue_trigger = axis_ratio >= 1.35 and float(regularized.sum() / max(support.sum(), 1)) <= 0.72
    if rescue_trigger:
        rescued = _rescue_support(regularized, geometry.centroid_xy, long_axis, short_axis, support)
        if rescued.sum() <= 2.1 * max(regularized.sum(), 1):
            support = rescued

    support = fill_holes(support | regularized)
    support = regularize_mask(support, diag)
    return support, {
        "axis_ratio": float(axis_ratio),
        "compactness": float(geometry.compactness),
        "anchor_area": float(regularized.sum()),
        "support_area": float(support.sum()),
    }


def compute_expansion_distance(area: float, compactness: float, diag: float, strength: float = 1.0) -> float:
    strength = float(np.clip(strength, 0.2, 6.0))
    r_eq = math.sqrt(max(area, 1.0) / math.pi)
    abs_floor_px = max(1.5, diag * 0.0032)
    floor_ratio = float(np.clip(0.08 + 0.05 * (1.0 - compactness), 0.08, 0.14))
    d_floor = max(abs_floor_px, r_eq * floor_ratio)
    d_cap_global = min(diag * 0.105, max(d_floor * 3.1, r_eq * 1.12))
    d_raw = r_eq * 0.26 * (strength**0.52)
    return float(np.clip(d_raw, d_floor, d_cap_global))


def expand_support_mask(
    support_mask: np.ndarray,
    support_geometry: dict[str, float],
    other_support_masks: list[np.ndarray] | None = None,
    strength: float = 1.0,
) -> np.ndarray:
    support_mask = support_mask.astype(bool)
    if not support_mask.any():
        return support_mask

    other_support_masks = other_support_masks or []
    diag = math.hypot(support_mask.shape[1], support_mask.shape[0])
    d_target = compute_expansion_distance(
        support_geometry.get("anchor_area", float(support_mask.sum())),
        support_geometry.get("compactness", 0.5),
        diag,
        strength=strength,
    )
    dist_to_support = ndimage.distance_transform_edt(~support_mask)

    blocked = np.zeros_like(support_mask, dtype=bool)
    for other in other_support_masks:
        blocked |= other.astype(bool)
    edge_clearance = np.minimum.reduce(
        np.meshgrid(
            np.arange(support_mask.shape[1]),
            np.arange(support_mask.shape[0]),
            indexing="xy",
        )
    )
    del edge_clearance

    if blocked.any():
        dist_to_other = ndimage.distance_transform_edt(~blocked)
    else:
        dist_to_other = np.full_like(dist_to_support, fill_value=d_target + 10.0, dtype=np.float64)

    neighbor_guard_px = max(2.0, diag * 0.0020)
    allowed_padding = np.maximum(dist_to_other - neighbor_guard_px, 0.0)
    expanded = support_mask | ((dist_to_support <= d_target) & (dist_to_support <= allowed_padding))
    expanded = fill_holes(expanded)
    return regularize_mask(expanded, diag)


def resolve_overlaps(expanded_masks: list[np.ndarray], support_masks: list[np.ndarray]) -> list[np.ndarray]:
    if not expanded_masks:
        return []

    owners = np.full(expanded_masks[0].shape, -1, dtype=np.int32)
    support_distances = [ndimage.distance_transform_edt(~mask.astype(bool)) for mask in support_masks]

    for idx, mask in enumerate(expanded_masks):
        mask = mask.astype(bool)
        to_assign = mask & (owners == -1)
        owners[to_assign] = idx

        overlap = mask & (owners != -1) & (owners != idx)
        if not overlap.any():
            continue

        current_distance = support_distances[idx]
        previous_indices = np.unique(owners[overlap])
        for prev_idx in previous_indices:
            prev_idx = int(prev_idx)
            if prev_idx < 0 or prev_idx == idx:
                continue
            choose_current = overlap & (current_distance <= support_distances[prev_idx])
            owners[choose_current] = idx

    return [(owners == idx) for idx in range(len(expanded_masks))]


def apply_table_mask_pipeline(
    anchor_mask: np.ndarray,
    other_anchor_masks: list[np.ndarray] | None = None,
    strength: float = 1.0,
) -> dict[str, object]:
    other_anchor_masks = other_anchor_masks or []
    support_mask, metrics = build_support_mask(anchor_mask)
    other_support_masks = [build_support_mask(mask)[0] for mask in other_anchor_masks if mask.any()]
    expanded_mask = expand_support_mask(support_mask, metrics, other_support_masks, strength=strength)
    geometry = compute_mask_geometry(anchor_mask.astype(bool))
    return {
        "tight_mask": anchor_mask.astype(bool),
        "support_mask": support_mask.astype(bool),
        "expanded_mask": expanded_mask.astype(bool),
        "geometry": geometry,
        "metrics": metrics,
    }
