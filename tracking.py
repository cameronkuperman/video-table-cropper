"""Simple matching utilities for table and person tracks across sampled frames."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from scipy.optimize import linear_sum_assignment


def bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1e-6)
    area_b = max((bx2 - bx1) * (by2 - by1), 1e-6)
    return float(inter / (area_a + area_b - inter))


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    return 0.0 if union <= 0 else intersection / union


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def centroid_distance_norm(
    centroid_a: tuple[float, float],
    centroid_b: tuple[float, float],
    frame_shape: tuple[int, int],
) -> float:
    diag = math.hypot(frame_shape[1], frame_shape[0])
    if diag <= 0:
        return 0.0
    dx = centroid_a[0] - centroid_b[0]
    dy = centroid_a[1] - centroid_b[1]
    return float(math.hypot(dx, dy) / diag)


def _match_rows(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
    score_fn: Callable[[dict[str, Any], dict[str, Any]], float],
    min_score: float,
) -> list[tuple[int, int, float]]:
    if not previous or not current:
        return []

    score_matrix = np.zeros((len(previous), len(current)), dtype=np.float32)
    for i, prev in enumerate(previous):
        for j, cur in enumerate(current):
            score_matrix[i, j] = score_fn(prev, cur)

    cost = 1.0 - score_matrix
    row_indices, col_indices = linear_sum_assignment(cost)
    matches: list[tuple[int, int, float]] = []
    for row_idx, col_idx in zip(row_indices, col_indices):
        score = float(score_matrix[row_idx, col_idx])
        if score >= min_score:
            matches.append((int(row_idx), int(col_idx), score))
    return matches


def assign_track_ids(
    frames: list[list[dict[str, Any]]],
    kind: str,
    frame_shape: tuple[int, int],
) -> None:
    next_track_id = 1
    previous: list[dict[str, Any]] = []

    for detections in frames:
        for detection in detections:
            detection["track_id"] = None

        if kind == "table":
            def score_fn(prev: dict[str, Any], cur: dict[str, Any]) -> float:
                return max(
                    0.0,
                    0.60 * mask_iou(prev["tight_mask"], cur["tight_mask"])
                    + 0.20 * cosine_similarity(prev["vector"], cur["vector"])
                    + 0.20 * math.exp(
                        -centroid_distance_norm(prev["centroid_xy"], cur["centroid_xy"], frame_shape) / 0.10
                    ),
                )

            min_score = 0.45
        else:
            def score_fn(prev: dict[str, Any], cur: dict[str, Any]) -> float:
                return max(
                    0.0,
                    0.80 * cosine_similarity(prev["vector"], cur["vector"])
                    + 0.20 * bbox_iou(prev["bbox_xyxy"], cur["bbox_xyxy"]),
                )

            min_score = 0.45

        matches = _match_rows(previous, detections, score_fn, min_score)
        used_current = set()
        next_previous: list[dict[str, Any]] = []

        for prev_idx, cur_idx, score in matches:
            prev = previous[prev_idx]
            cur = detections[cur_idx]
            cur["track_id"] = prev["track_id"]
            cur["match_score"] = score
            used_current.add(cur_idx)
            next_previous.append(cur)

        for idx, detection in enumerate(detections):
            if idx in used_current:
                continue
            detection["track_id"] = f"{kind}_{next_track_id}"
            next_track_id += 1
            next_previous.append(detection)

        previous = next_previous
