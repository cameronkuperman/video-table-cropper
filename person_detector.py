"""
Person detection, within-triplet tracking, and per-table perception building.

Uses YOLOv8-seg (ultralytics) for person segmentation masks.
Falls back gracefully if ultralytics is not installed — perception.json
is simply not written for that run.

Flow per triplet:
  1. detect_people_in_frame()  — run YOLOv8 on each of the 3 full frames
  2. assign_track_ids()        — match the same person across frames by centroid proximity
  3. build_perception_for_table() — for one table, filter to overlapping people and build JSON
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# ── Model loading ──────────────────────────────────────────────────────────

def load_yolo_model(model_name: str = "yolov8m-seg.pt"):
    """
    Load YOLOv8 segmentation model. Returns None if ultralytics is not installed.
    On first run, downloads the model weights (~52 MB for medium).
    Model options by accuracy/speed tradeoff:
      yolov8n-seg.pt  ~6 MB   fastest, least accurate
      yolov8s-seg.pt  ~22 MB  good balance (default)
      yolov8m-seg.pt  ~52 MB  better for overhead/CCTV angles
    """
    if not _YOLO_AVAILABLE:
        print("  [perception] ultralytics not installed — skipping person detection.")
        print("  [perception] To enable: pip install ultralytics")
        return None
    print(f"  [perception] Loading {model_name} (auto-downloads on first run)...")
    return YOLO(model_name)


# ── Internal helpers ───────────────────────────────────────────────────────

def _rasterize_polygon(points: list[tuple[float, float]], shape: tuple[int, int]) -> np.ndarray:
    """Convert a polygon [(x, y), ...] to a boolean mask of shape (H, W)."""
    h, w = shape
    img = Image.new("L", (w, h), 0)
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) >= 3:
        ImageDraw.Draw(img).polygon(pts, fill=255)
    return np.array(img, dtype=bool)


# ── Step 1: per-frame detection ────────────────────────────────────────────

def _people_from_result(result) -> list[dict[str, Any]]:
    """Convert one ultralytics Results object into our list-of-dicts format."""
    orig_h, orig_w = result.orig_shape
    people: list[dict[str, Any]] = []
    if result.masks is None:
        return people

    for mask_xy, box, conf in zip(result.masks.xy, result.boxes.xyxy, result.boxes.conf):
        if len(mask_xy) < 3:
            continue
        mask = _rasterize_polygon(
            [(float(x), float(y)) for x, y in mask_xy], (orig_h, orig_w)
        )
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue
        people.append({
            "mask": mask,
            "bbox_xyxy": [float(v) for v in box.cpu().tolist()],
            "centroid_norm": (float(xs.mean()) / orig_w, float(ys.mean()) / orig_h),
            "score": float(conf),
            "track_id": None,
        })
    return people


def detect_people_in_frame(frame_path: Path, model) -> list[dict[str, Any]]:
    """
    Run YOLOv8-seg on one frame. Returns one dict per detected person.

    Each dict:
      mask         bool ndarray H×W — pixel-level segmentation (not saved to JSON)
      bbox_xyxy    [x1, y1, x2, y2] in pixel coords of the original image
      centroid_norm  (cx, cy) normalized 0–1 by frame width/height
      score        detection confidence 0–1
      track_id     None — filled in by assign_track_ids()
    """
    results = model(str(frame_path), classes=[0], conf=0.15, verbose=False)
    return _people_from_result(results[0])


def detect_people_batch(frame_paths: list[Path], model) -> list[list[dict[str, Any]]]:
    """
    Run YOLOv8-seg on multiple frames in a single batched forward pass.
    Returns one list-of-people per input frame, in the same order.
    Ultralytics batches natively when passed a list of paths — 2–3× faster on GPU.
    """
    if not frame_paths:
        return []
    results = model([str(p) for p in frame_paths], classes=[0], conf=0.15, verbose=False)
    return [_people_from_result(r) for r in results]


# ── Step 2: within-triplet tracking ───────────────────────────────────────

def assign_track_ids(
    frame_detections: list[list[dict[str, Any]]],
    max_dist: float = 0.2,
) -> None:
    """
    Assign track IDs across frames by nearest-centroid matching. Modifies in place.

    max_dist: max normalized centroid distance (0–1 scale, diagonal = sqrt(2)) to
              count as the same person. 0.2 works well for seated people over ~10s gaps.
              Waiters walking fast will exceed this and get new IDs each frame.
    """
    next_id = 0
    for person in frame_detections[0]:
        person["track_id"] = f"t{next_id}"
        next_id += 1

    for frame_idx in range(1, len(frame_detections)):
        prev = frame_detections[frame_idx - 1]
        curr = frame_detections[frame_idx]
        used_ids: set[str] = set()

        for cp in curr:
            best_id: str | None = None
            best_dist = float("inf")
            for pp in prev:
                if pp["track_id"] in used_ids:
                    continue
                dist = math.hypot(
                    cp["centroid_norm"][0] - pp["centroid_norm"][0],
                    cp["centroid_norm"][1] - pp["centroid_norm"][1],
                )
                if dist < best_dist and dist < max_dist:
                    best_dist = dist
                    best_id = pp["track_id"]

            if best_id is not None:
                cp["track_id"] = best_id
                used_ids.add(best_id)
            else:
                cp["track_id"] = f"t{next_id}"
                next_id += 1


# ── Step 3: per-table perception ──────────────────────────────────────────

def build_perception_for_table(
    frame_detections: list[list[dict[str, Any]]],
    table_polygon: list[tuple[float, float]],
    img_shape: tuple[int, int],
) -> dict[str, Any]:
    """
    Build the perception dict for one specific table across 3 frames.

    Only people whose mask or bounding box overlaps this table's polygon are included.
    The same person appearing in all 3 frames produces 3 rows (one per frame),
    linked by the same track_id.

    Returns a dict ready to be written as perception.json.
    """
    h, w = img_shape
    table_mask = _rasterize_polygon(table_polygon, img_shape)
    table_area = max(int(table_mask.sum()), 1)

    ys, xs = np.nonzero(table_mask)
    table_cx = float(xs.mean()) / w if xs.size > 0 else 0.5
    table_cy = float(ys.mean()) / h if xs.size > 0 else 0.5

    # For displacement: remember where each track was in the previous frame
    prev_centroids: dict[str, tuple[float, float]] = {}
    people_rows: list[dict[str, Any]] = []

    for frame_idx, detections in enumerate(frame_detections):
        for person in detections:
            # Pixel-level overlap between person mask and table mask
            intersection_px = int(np.logical_and(person["mask"], table_mask).sum())

            # If no pixel overlap, check if the bounding box at least touches the table
            if intersection_px == 0:
                x1, y1, x2, y2 = [int(v) for v in person["bbox_xyxy"]]
                y1c, y2c = max(0, y1), min(h, y2)
                x1c, x2c = max(0, x1), min(w, x2)
                if y2c > y1c and x2c > x1c and table_mask[y1c:y2c, x1c:x2c].any():
                    pass  # bbox overlaps — include with overlap_frac = 0
                else:
                    continue  # truly no relationship to this table

            person_area = max(int(person["mask"].sum()), 1)
            overlap_frac_person = intersection_px / person_area  # what % of the person is over the table
            overlap_frac_table  = intersection_px / table_area   # what % of the table the person covers

            dist_norm = math.hypot(
                person["centroid_norm"][0] - table_cx,
                person["centroid_norm"][1] - table_cy,
            )

            track_id = person["track_id"]
            displacement: float | None = None
            if track_id in prev_centroids:
                prev_c = prev_centroids[track_id]
                displacement = math.hypot(
                    person["centroid_norm"][0] - prev_c[0],
                    person["centroid_norm"][1] - prev_c[1],
                )
            prev_centroids[track_id] = person["centroid_norm"]

            people_rows.append({
                "frame_index":           frame_idx,
                "track_id":              track_id,
                "centroid_x":            round(person["centroid_norm"][0], 4),
                "centroid_y":            round(person["centroid_norm"][1], 4),
                "bbox_xyxy":             [round(v, 1) for v in person["bbox_xyxy"]],
                "overlap_frac_of_person": round(overlap_frac_person, 4),
                "overlap_frac_of_table":  round(overlap_frac_table,  4),
                "distance_norm":          round(dist_norm, 4),
                "displacement_from_prev": round(displacement, 4) if displacement is not None else None,
                "score":                  round(person["score"], 3),
            })

    # ── Scalar summaries (one value or list-of-3 per stat) ─────────────────
    person_count = [0, 0, 0]
    max_overlap  = [0.0, 0.0, 0.0]
    overlap_sum  = [0.0, 0.0, 0.0]
    min_dist     = [1.0, 1.0, 1.0]

    for row in people_rows:
        fi = row["frame_index"]
        person_count[fi] += 1
        max_overlap[fi]   = max(max_overlap[fi], row["overlap_frac_of_person"])
        overlap_sum[fi]  += row["overlap_frac_of_person"]
        min_dist[fi]      = min(min_dist[fi], row["distance_norm"])

    # Which tracks appear in which frames → entering / leaving / persistent
    track_frames: dict[str, set[int]] = {}
    for row in people_rows:
        track_frames.setdefault(row["track_id"], set()).add(row["frame_index"])

    entering         = [0, 0]   # [appeared in frame 1 not 0, appeared in frame 2 not 1]
    leaving          = [0, 0]   # [was in frame 0 not 1,      was in frame 1 not 2]
    persistent_count = 0

    for frames in track_frames.values():
        if len(frames) == 3:
            persistent_count += 1
        if 1 in frames and 0 not in frames: entering[0] += 1
        if 2 in frames and 1 not in frames: entering[1] += 1
        if 0 in frames and 1 not in frames: leaving[0]  += 1
        if 1 in frames and 2 not in frames: leaving[1]  += 1

    displacements = [r["displacement_from_prev"] for r in people_rows if r["displacement_from_prev"] is not None]
    mean_disp = round(float(np.mean(displacements)), 4) if displacements else 0.0
    max_disp  = round(float(np.max(displacements)),  4) if displacements else 0.0

    return {
        "people": people_rows,
        "scalars": {
            "person_count":               person_count,
            "max_overlap_frac_of_person": [round(v, 4) for v in max_overlap],
            "overlap_sum":                [round(v, 4) for v in overlap_sum],
            "min_distance_norm":          [round(v, 4) for v in min_dist],
            "mean_displacement":          mean_disp,
            "max_displacement":           max_disp,
            "entering":                   entering,
            "leaving":                    leaving,
            "persistent_count":           persistent_count,
        },
    }
