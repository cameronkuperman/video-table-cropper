"""
Person detection, within-group tracking, and per-table perception building.

Uses YOLO26-seg (ultralytics) for person segmentation masks.
Falls back gracefully if ultralytics is not installed — perception.json
is simply not written for that run.

Flow per group of N frames:
  1. detect_people_in_frame()   — run YOLO on each of the N full frames
  2. assign_track_ids()         — match the same person across frames by
                                  K-frame-lookback centroid + bbox-IoU matching
  3. build_perception_for_table() — for one table, filter to overlapping
                                    people and build a schema-v2 JSON dict

The "triplet" naming is preserved on the wire/Drive protocol; internally
groups can be any N in [MIN, MAX]_FRAMES_PER_GROUP.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


PERCEPTION_SCHEMA_VERSION = 2

# Tenure thresholds — fractions of N. Auto-scale so changing N requires no edits.
PERSISTENT_THRESHOLD_FRAC = 1.0   # strict: present in every frame
MOSTLY_THRESHOLD_FRAC = 0.8       # >= 80% of N (8/10 at N=10)
TRANSIENT_THRESHOLD_FRAC = 0.2    # <= 20% of N (2/10 at N=10)

# Track association tunables.
TRACK_LOOKBACK = 3                # K — match against last K frames, not just K=1
TRACK_MAX_DIST = 0.2              # max normalized centroid distance for a match
TRACK_IOU_BONUS = 0.5             # IoU value above which a candidate is preferred
                                  # over a closer-by-centroid one (tiebreaker)

# Dwell threshold — a person is "dwelling at" the table if at least this much
# of their body overlaps the table in a frame.
DWELL_OVERLAP_THRESHOLD = 0.5

# Two tracks count as "spatially-co-located" (probable ID-swap evidence) when
# their mean centroids are within this distance and their frame sets are disjoint.
DISJOINT_TRACK_DISTANCE_THRESHOLD = 0.1

# How wide a missing-frame run can be before it counts as a real exit/return
# (vs. a single YOLO miss). Used by the gap-filled entering/leaving lists.
MAX_GAP_TO_FILL = 1


# ── Model loading ──────────────────────────────────────────────────────────

DEFAULT_YOLO_MODEL_NAME = "yolo26l-seg.pt"


def load_yolo_model(model_name: str | None = None):
    """
    Load the configured YOLO segmentation model. Returns None if ultralytics is not installed.
    On first run, downloads the configured model weights.

    Override with YOLO_MODEL_NAME. Default: yolo26l-seg.pt.
    """
    if not _YOLO_AVAILABLE:
        print("  [perception] ultralytics not installed — skipping person detection.")
        print("  [perception] To enable: pip install ultralytics")
        return None
    model_name = model_name or os.getenv("YOLO_MODEL_NAME", DEFAULT_YOLO_MODEL_NAME)
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


def _bbox_iou(a: list[float], b: list[float]) -> float:
    """IoU of two [x1, y1, x2, y2] boxes. Returns 0.0 if either is degenerate."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return float(inter / union) if union > 0 else 0.0


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
    Run YOLO segmentation on one frame. Returns one dict per detected person.

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
    Run YOLO segmentation on multiple frames in a single batched forward pass.
    Returns one list-of-people per input frame, in the same order.
    Ultralytics batches natively when passed a list of paths — 2–3× faster on GPU.
    """
    if not frame_paths:
        return []
    results = model([str(p) for p in frame_paths], classes=[0], conf=0.15, verbose=False)
    return [_people_from_result(r) for r in results]


# ── Step 2: within-group tracking ──────────────────────────────────────────

def assign_track_ids(
    frame_detections: list[list[dict[str, Any]]],
    max_dist: float = TRACK_MAX_DIST,
    lookback: int = TRACK_LOOKBACK,
) -> None:
    """
    Assign track IDs across frames. Modifies the per-person dicts in place.

    Matching strategy (per detection in frame i):
      1. Search candidates already-tracked in frames i-1, i-2, ..., i-lookback.
         More-recent frames are preferred (closer in time = more reliable).
      2. Among candidates within `max_dist` (normalized centroid distance), score
         each as: -dist + TRACK_IOU_BONUS * iou(curr_bbox, candidate_bbox).
         A high-IoU candidate one frame further back can beat a closer-centroid
         candidate from the immediately previous frame — this stops customers
         standing close together from swapping IDs.
      3. Each candidate track ID can only be claimed by one current detection
         per frame. If two current detections want the same ID, the one with
         the better score wins.

    The lookback (default 3) makes a single missed YOLO frame (one frame in 10
    where the tracker drops the customer) recoverable: the track resumes on the
    next frame instead of starting a new ID.
    """
    if not frame_detections:
        return

    next_id = 0
    # Seed first frame with fresh IDs.
    for person in frame_detections[0]:
        person["track_id"] = f"t{next_id}"
        next_id += 1

    # Map track_id -> (frame_idx, person_dict) for the most recent frame in
    # which we saw this track. We scan candidates from most-recent backward.
    last_seen: dict[str, tuple[int, dict[str, Any]]] = {}
    for person in frame_detections[0]:
        last_seen[person["track_id"]] = (0, person)

    for frame_idx in range(1, len(frame_detections)):
        curr = frame_detections[frame_idx]

        # Build (track_id, frame_age, candidate_person) tuples, only including
        # tracks whose last sighting is within the lookback window.
        candidates: list[tuple[str, int, dict[str, Any]]] = []
        min_frame = frame_idx - lookback
        for tid, (seen_frame, person) in last_seen.items():
            if seen_frame >= min_frame and seen_frame < frame_idx:
                age = frame_idx - seen_frame
                candidates.append((tid, age, person))

        # Score every (current_detection, candidate) pair.
        # Higher score = better match.
        proposals: list[tuple[float, int, str, dict[str, Any]]] = []
        # tuple = (score, curr_index, candidate_track_id, candidate_person)
        for ci, cp in enumerate(curr):
            for tid, age, pp in candidates:
                dist = math.hypot(
                    cp["centroid_norm"][0] - pp["centroid_norm"][0],
                    cp["centroid_norm"][1] - pp["centroid_norm"][1],
                )
                # Loosen the centroid threshold proportionally to gap age — a
                # customer drifts more in 2-3 frames than in 1.
                if dist > max_dist * age:
                    continue
                iou = _bbox_iou(cp["bbox_xyxy"], pp["bbox_xyxy"])
                # Recency penalty: prefer a frame-1 candidate over a frame-3
                # candidate if both are otherwise equal.
                score = -dist + TRACK_IOU_BONUS * iou - 0.01 * (age - 1)
                proposals.append((score, ci, tid, pp))

        # Greedy 1-to-1 assignment: highest-scoring proposals first; each track
        # ID and each current detection can only be claimed once.
        proposals.sort(key=lambda x: -x[0])
        assigned_curr: set[int] = set()
        used_ids: set[str] = set()
        for score, ci, tid, _pp in proposals:
            if ci in assigned_curr or tid in used_ids:
                continue
            curr[ci]["track_id"] = tid
            assigned_curr.add(ci)
            used_ids.add(tid)

        # Anything unassigned gets a fresh track ID.
        for ci, cp in enumerate(curr):
            if ci in assigned_curr:
                continue
            cp["track_id"] = f"t{next_id}"
            next_id += 1

        # Update last_seen for every current detection (refreshes recency).
        for cp in curr:
            last_seen[cp["track_id"]] = (frame_idx, cp)


# ── Step 3: per-table perception ──────────────────────────────────────────


def _table_polygon_max_y(table_polygon: list[tuple[float, float]]) -> float:
    """The largest y-coordinate of the table polygon (its bottom edge in image
    coords). Used by `bbox_below_table_frac`."""
    if not table_polygon:
        return 0.0
    return float(max(float(y) for _x, y in table_polygon))


def _bbox_below_table_frac(
    bbox_xyxy: list[float],
    table_max_y: float,
) -> float:
    """Fraction of the bbox's vertical extent that lies below the table's
    bottom edge. A seated person's legs go under the table → high value.
    A person walking *behind* the table has their entire bbox above the
    table's bottom edge → 0.
    """
    _x1, y1, _x2, y2 = bbox_xyxy
    height = max(float(y2) - float(y1), 1e-6)
    below_height = max(0.0, float(y2) - float(table_max_y))
    return min(1.0, below_height / height)


def _bbox_aspect_ratio(bbox_xyxy: list[float]) -> float:
    """h / w. Standing ~2.5+, sitting ~1.0–1.5, leaning/bending ~0.5–1.0."""
    x1, y1, x2, y2 = bbox_xyxy
    width = max(float(x2) - float(x1), 1.0)
    height = max(float(y2) - float(y1), 0.0)
    return float(height / width)


def _gap_fill(frames: set[int], n_frames: int, max_gap: int = MAX_GAP_TO_FILL) -> set[int]:
    """Return `frames` with any gap of size <= max_gap between two present
    frames filled in. Used to suppress one-frame YOLO misses in the
    entering/leaving signal.

    Example with max_gap=1: {0, 1, 2, 4, 5} → {0, 1, 2, 3, 4, 5}.
    Example with max_gap=1: {0, 1, 5, 6}    → unchanged (gap of 3 is real).
    """
    if not frames or max_gap < 1:
        return set(frames)
    sorted_frames = sorted(frames)
    filled = set(sorted_frames)
    for a, b in zip(sorted_frames, sorted_frames[1:]):
        if 1 < b - a <= max_gap + 1:
            for between in range(a + 1, b):
                filled.add(between)
    return filled


def _max_consecutive(frames: set[int]) -> int:
    """Longest run of consecutive integers in `frames`."""
    if not frames:
        return 0
    sorted_frames = sorted(frames)
    best = run = 1
    for a, b in zip(sorted_frames, sorted_frames[1:]):
        run = run + 1 if b == a + 1 else 1
        if run > best:
            best = run
    return best


def _gap_count(frames: set[int]) -> int:
    """Number of missing-frame runs strictly between first and last frame
    of the track. A track present in [0,1,2,4,5,7,8,9] has gap_count = 2
    (one gap at frame 3, another at frame 6)."""
    if len(frames) < 2:
        return 0
    sorted_frames = sorted(frames)
    gaps = 0
    for a, b in zip(sorted_frames, sorted_frames[1:]):
        if b - a > 1:
            gaps += 1
    return gaps


def build_perception_for_table(
    frame_detections: list[list[dict[str, Any]]],
    table_polygon: list[tuple[float, float]],
    img_shape: tuple[int, int],
    n_frames: int | None = None,
) -> dict[str, Any]:
    """
    Build the perception dict for one specific table across N frames.

    Only people whose mask or bounding box overlaps this table's polygon are
    included. The same person appearing in K of N frames produces K rows
    (one per frame), linked by the same track_id.

    `n_frames` defaults to len(frame_detections); pass explicitly if the
    detections list is empty for some frames but you still want N-aware
    scalar shapes.

    Returns a dict ready to be serialized as perception.json (schema v2).
    """
    if n_frames is None:
        n_frames = len(frame_detections)
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")

    h, w = img_shape
    table_mask = _rasterize_polygon(table_polygon, img_shape)
    table_area = max(int(table_mask.sum()), 1)
    table_max_y = _table_polygon_max_y(table_polygon)

    ys, xs = np.nonzero(table_mask)
    table_cx = float(xs.mean()) / w if xs.size > 0 else 0.5
    table_cy = float(ys.mean()) / h if xs.size > 0 else 0.5

    # ── Build per-(person × frame) rows ────────────────────────────────────
    prev_centroids: dict[str, tuple[float, float]] = {}
    people_rows: list[dict[str, Any]] = []

    for frame_idx, detections in enumerate(frame_detections):
        if frame_idx >= n_frames:
            break
        for person in detections:
            intersection_px = int(np.logical_and(person["mask"], table_mask).sum())

            if intersection_px == 0:
                x1, y1, x2, y2 = [int(v) for v in person["bbox_xyxy"]]
                y1c, y2c = max(0, y1), min(h, y2)
                x1c, x2c = max(0, x1), min(w, x2)
                if y2c > y1c and x2c > x1c and table_mask[y1c:y2c, x1c:x2c].any():
                    pass  # bbox overlaps — include with overlap_frac = 0
                else:
                    continue  # truly no relationship to this table

            person_area = max(int(person["mask"].sum()), 1)
            overlap_frac_person = intersection_px / person_area
            overlap_frac_table = intersection_px / table_area

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
                "frame_index":            frame_idx,
                "track_id":               track_id,
                "centroid_x":             round(person["centroid_norm"][0], 4),
                "centroid_y":             round(person["centroid_norm"][1], 4),
                "bbox_xyxy":              [round(v, 1) for v in person["bbox_xyxy"]],
                "overlap_frac_of_person": round(overlap_frac_person, 4),
                "overlap_frac_of_table":  round(overlap_frac_table, 4),
                "distance_norm":          round(dist_norm, 4),
                "displacement_from_prev": round(displacement, 4) if displacement is not None else None,
                "score":                  round(person["score"], 3),
                "bbox_below_table_frac":  round(_bbox_below_table_frac(person["bbox_xyxy"], table_max_y), 4),
                "bbox_aspect_ratio":      round(_bbox_aspect_ratio(person["bbox_xyxy"]), 4),
            })

    # ── Per-frame scalar lists (length = n_frames) ────────────────────────
    person_count = [0] * n_frames
    max_overlap = [0.0] * n_frames
    overlap_sum = [0.0] * n_frames
    min_dist = [1.0] * n_frames

    # ── Per-track aggregation (collected during the row scan) ────────────
    track_frames: dict[str, set[int]] = {}
    track_overlaps: dict[str, list[float]] = {}
    track_distances: dict[str, list[float]] = {}
    track_displacements: dict[str, list[float]] = {}
    track_scores: dict[str, list[float]] = {}
    track_centroids: dict[str, list[tuple[float, float]]] = {}
    track_bbox_below: dict[str, list[float]] = {}
    track_bbox_aspect: dict[str, list[float]] = {}

    for row in people_rows:
        fi = row["frame_index"]
        person_count[fi] += 1
        max_overlap[fi] = max(max_overlap[fi], row["overlap_frac_of_person"])
        overlap_sum[fi] += row["overlap_frac_of_person"]
        min_dist[fi] = min(min_dist[fi], row["distance_norm"])

        tid = row["track_id"]
        track_frames.setdefault(tid, set()).add(fi)
        track_overlaps.setdefault(tid, []).append(row["overlap_frac_of_person"])
        track_distances.setdefault(tid, []).append(row["distance_norm"])
        if row["displacement_from_prev"] is not None:
            track_displacements.setdefault(tid, []).append(row["displacement_from_prev"])
        track_scores.setdefault(tid, []).append(row["score"])
        track_centroids.setdefault(tid, []).append((row["centroid_x"], row["centroid_y"]))
        track_bbox_below.setdefault(tid, []).append(row["bbox_below_table_frac"])
        track_bbox_aspect.setdefault(tid, []).append(row["bbox_aspect_ratio"])

    # ── Per-transition lists (length = n_frames - 1) ──────────────────────
    # Raw versions reflect the YOLO output as-is. The non-raw versions
    # gap-fill single-frame misses per track to suppress fake enter/leave
    # caused by a one-frame detection drop.
    n_trans = max(0, n_frames - 1)
    raw_entering = [0] * n_trans
    raw_leaving = [0] * n_trans
    entering = [0] * n_trans
    leaving = [0] * n_trans

    for tid, frames in track_frames.items():
        filled = _gap_fill(frames, n_frames)
        for i in range(n_trans):
            if (i + 1) in frames and i not in frames:
                raw_entering[i] += 1
            if i in frames and (i + 1) not in frames:
                raw_leaving[i] += 1
            if (i + 1) in filled and i not in filled:
                entering[i] += 1
            if i in filled and (i + 1) not in filled:
                leaving[i] += 1

    # ── Tenure thresholds — fractions of N, auto-scale ────────────────────
    persistent_thresh = max(1, math.ceil(PERSISTENT_THRESHOLD_FRAC * n_frames))
    mostly_thresh = max(1, math.ceil(MOSTLY_THRESHOLD_FRAC * n_frames))
    transient_max = max(1, math.floor(TRANSIENT_THRESHOLD_FRAC * n_frames))

    # ── Build tracks[] summary array ──────────────────────────────────────
    tracks_summary: list[dict[str, Any]] = []
    persistent_count = 0
    mostly_persistent_count = 0
    transient_count = 0
    max_track_tenure = 0
    max_consecutive_tenure = 0

    for tid, frames in track_frames.items():
        tenure = len(frames)
        consecutive = _max_consecutive(frames)
        sorted_frames = sorted(frames)

        if tenure >= persistent_thresh:
            persistent_count += 1
        if tenure >= mostly_thresh:
            mostly_persistent_count += 1
        if tenure <= transient_max:
            transient_count += 1
        max_track_tenure = max(max_track_tenure, tenure)
        max_consecutive_tenure = max(max_consecutive_tenure, consecutive)

        ovl = track_overlaps[tid]
        dst = track_distances[tid]
        dsp = track_displacements.get(tid, [])
        scr = track_scores[tid]
        cents = track_centroids[tid]
        bel = track_bbox_below[tid]
        asp = track_bbox_aspect[tid]
        dwell = sum(1 for v in ovl if v > DWELL_OVERLAP_THRESHOLD)

        if cents:
            cx_arr = np.array([c[0] for c in cents], dtype=float)
            cy_arr = np.array([c[1] for c in cents], dtype=float)
            centroid_spread = float(math.hypot(float(cx_arr.std()), float(cy_arr.std())))
        else:
            centroid_spread = 0.0

        tracks_summary.append({
            "track_id":                    tid,
            "frames_present":              tenure,
            "max_consecutive":             consecutive,
            "first_frame":                 sorted_frames[0],
            "last_frame":                  sorted_frames[-1],
            "presence_ratio":              round(tenure / n_frames, 4),
            "consecutive_ratio":           round(consecutive / n_frames, 4),
            "gap_count":                   _gap_count(frames),
            "dwell_frames":                dwell,
            "mean_overlap_frac_of_person": round(float(np.mean(ovl)), 4),
            "max_overlap_frac_of_person":  round(float(np.max(ovl)), 4),
            "overlap_stddev":              round(float(np.std(ovl)), 4),
            "mean_distance_norm":          round(float(np.mean(dst)), 4),
            "min_distance_norm":           round(float(np.min(dst)), 4),
            "centroid_spread":             round(centroid_spread, 4),
            "mean_displacement":           round(float(np.mean(dsp)), 4) if dsp else 0.0,
            "max_displacement":            round(float(np.max(dsp)), 4) if dsp else 0.0,
            "mean_score":                  round(float(np.mean(scr)), 4),
            "min_score":                   round(float(np.min(scr)), 4),
            "mean_bbox_below_table_frac":  round(float(np.mean(bel)), 4),
            "mean_bbox_aspect_ratio":      round(float(np.mean(asp)), 4),
        })

    tracks_summary.sort(key=lambda t: t["track_id"])

    # ── Group-level dwelling signals ──────────────────────────────────────
    frames_with_any_person = sum(1 for c in person_count if c > 0)
    frames_with_dwelling_person = sum(
        1
        for fi in range(n_frames)
        if any(
            r["frame_index"] == fi and r["overlap_frac_of_person"] > DWELL_OVERLAP_THRESHOLD
            for r in people_rows
        )
    )

    primary_track_id: str | None = None
    primary_track_dwell_frames = 0
    if tracks_summary:
        # Highest summed overlap_frac_of_person identifies the primary occupant.
        best_score = -1.0
        for tid, ovl in track_overlaps.items():
            total = float(sum(ovl))
            if total > best_score:
                best_score = total
                primary_track_id = tid
        if primary_track_id is not None:
            for t in tracks_summary:
                if t["track_id"] == primary_track_id:
                    primary_track_dwell_frames = t["dwell_frames"]
                    break

    # ── ID-swap defense: count disjoint-frame, co-located track pairs ────
    disjoint_track_count = 0
    track_ids = list(track_frames.keys())
    track_means: dict[str, tuple[float, float]] = {}
    for tid, cents in track_centroids.items():
        if cents:
            mx = float(np.mean([c[0] for c in cents]))
            my = float(np.mean([c[1] for c in cents]))
            track_means[tid] = (mx, my)
    for i, ta in enumerate(track_ids):
        for tb in track_ids[i + 1:]:
            if track_frames[ta].isdisjoint(track_frames[tb]):
                ma = track_means.get(ta)
                mb = track_means.get(tb)
                if ma is None or mb is None:
                    continue
                if math.hypot(ma[0] - mb[0], ma[1] - mb[1]) <= DISJOINT_TRACK_DISTANCE_THRESHOLD:
                    disjoint_track_count += 1

    # ── Aggregate displacements over all (person × frame) pairs ──────────
    displacements = [r["displacement_from_prev"] for r in people_rows if r["displacement_from_prev"] is not None]
    mean_disp = round(float(np.mean(displacements)), 4) if displacements else 0.0
    max_disp = round(float(np.max(displacements)), 4) if displacements else 0.0

    return {
        "schema_version": PERCEPTION_SCHEMA_VERSION,
        "n_frames": n_frames,
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
            "raw_entering":               raw_entering,
            "raw_leaving":                raw_leaving,
            "persistent_count":           persistent_count,
            "mostly_persistent_count":    mostly_persistent_count,
            "transient_count":            transient_count,
            "max_track_tenure":           max_track_tenure,
            "max_consecutive_tenure":     max_consecutive_tenure,
            "frames_with_any_person":     frames_with_any_person,
            "frames_with_dwelling_person": frames_with_dwelling_person,
            "primary_track_id":           primary_track_id,
            "primary_track_dwell_frames": primary_track_dwell_frames,
            "disjoint_track_count":       disjoint_track_count,
        },
        "tracks": tracks_summary,
    }
