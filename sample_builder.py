"""Helpers for writing review bundles and model-specific sample exports."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dataset_schema import (
    OCCUPANCY_MLP_NPZ_NAME,
    PERCEPTION_NPZ_NAME,
    PREVIEW_ANCHOR_NAME,
    PREVIEW_PREV_1_NAME,
    PREVIEW_PREV_2_NAME,
    SAMPLE_JSON_NAME,
    TEMPORAL_IMAGE_FILES,
    TIGHT_ANCHOR_NAME,
    occupancy_binary_label,
)
from mask_geometry import mask_bbox


def encode_mask_rle(mask: np.ndarray) -> dict[str, Any]:
    flat = mask.astype(np.uint8).ravel(order="F")
    counts: list[int] = []
    count = 0
    last = 0
    for value in flat:
        if int(value) == last:
            count += 1
            continue
        counts.append(count)
        count = 1
        last = int(value)
    counts.append(count)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def bbox_with_padding(bbox_xyxy: tuple[int, int, int, int], shape: tuple[int, int], pad_px: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    return (
        max(0, x1 - pad_px),
        max(0, y1 - pad_px),
        min(shape[1], x2 + pad_px),
        min(shape[0], y2 + pad_px),
    )


def crop_image(image: Image.Image, bbox_xyxy: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = bbox_xyxy
    return image.crop((x1, y1, x2, y2))


def crop_masked_image(image: Image.Image, mask: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> Image.Image:
    base = np.array(image.convert("RGB"))
    masked = np.zeros_like(base)
    masked[mask.astype(bool)] = base[mask.astype(bool)]
    return Image.fromarray(masked).crop(bbox_xyxy)


def preview_padding_for_bbox(bbox_xyxy: tuple[int, int, int, int]) -> int:
    width = max(0, bbox_xyxy[2] - bbox_xyxy[0])
    height = max(0, bbox_xyxy[3] - bbox_xyxy[1])
    return int(np.clip(round(0.05 * max(width, height)), 8, 24))


def npz_bytes(payload: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    return buffer.getvalue()


def update_label(sample_payload: dict[str, Any], human_label: str) -> dict[str, Any]:
    updated = json.loads(json.dumps(sample_payload))
    updated["label"] = {
        "human_label": human_label,
        "occupancy_binary_label": occupancy_binary_label(human_label),
    }
    return updated


def build_manifest_entry(sample_payload: dict[str, Any], label: str, relative_folder: str) -> dict[str, Any]:
    return {
        "sample_id": sample_payload["sample_id"],
        "label": label,
        "relative_folder": relative_folder,
        "source_video_drive_id": sample_payload["source_video"]["drive_file_id"],
        "camera_id": sample_payload["source_video"]["camera_id"],
        "table_track_id": sample_payload["table"]["table_track_id"],
        "anchor_time_seconds": sample_payload["timing"]["anchor_time_seconds"],
    }


def _weighted_mean(vectors: list[np.ndarray], weights: list[float], dim: int) -> np.ndarray:
    if not vectors:
        return np.zeros(dim, dtype=np.float32)
    matrix = np.stack(vectors).astype(np.float32)
    raw_weights = np.array(weights, dtype=np.float32)
    if float(raw_weights.sum()) <= 1e-8:
        raw_weights = np.ones((matrix.shape[0],), dtype=np.float32)
    norm = raw_weights / raw_weights.sum()
    return (matrix * norm[:, None]).sum(axis=0).astype(np.float32)


def build_occupancy_mlp_vector(
    sample_payload: dict[str, Any],
    perception_payload: dict[str, np.ndarray],
) -> np.ndarray:
    table_vecs = perception_payload["table_vecs_tight"].astype(np.float32)
    dim = int(table_vecs.shape[1])

    person_full = perception_payload.get("person_vecs_full", np.zeros((0, dim), dtype=np.float32))
    person_intersection = perception_payload.get("person_vecs_intersection", np.zeros((0, dim), dtype=np.float32))
    person_frames = perception_payload.get("person_frame_index", np.zeros((0,), dtype=np.int32)).astype(np.int32)
    weights = perception_payload.get("person_weight_for_mlp", np.zeros((0,), dtype=np.float32)).astype(np.float32)

    sample_people = sample_payload.get("people", [])
    per_frame_summary: list[np.ndarray] = []
    person_count = [0.0, 0.0, 0.0]
    overlap_sum = [0.0, 0.0, 0.0]
    max_overlap = [0.0, 0.0, 0.0]
    min_distance = [1.0, 1.0, 1.0]

    for frame_idx in range(3):
        indices = np.where(person_frames == frame_idx)[0]
        frame_vectors: list[np.ndarray] = []
        frame_weights: list[float] = []
        person_count[frame_idx] = float(len(indices))
        for idx in indices:
            preferred = person_intersection[idx]
            if float(np.linalg.norm(preferred)) <= 1e-8:
                preferred = person_full[idx]
            frame_vectors.append(preferred.astype(np.float32))
            frame_weights.append(float(weights[idx]) if idx < len(weights) else 1.0)
        per_frame_summary.append(_weighted_mean(frame_vectors, frame_weights, dim))

    for person in sample_people:
        frame_idx = int(person.get("frame_index", 0))
        overlap = float(person.get("overlap_frac_of_person", 0.0))
        overlap_sum[frame_idx] += overlap
        max_overlap[frame_idx] = max(max_overlap[frame_idx], overlap)
        min_distance[frame_idx] = min(min_distance[frame_idx], float(person.get("distance_norm", 1.0)))

    track_presence: dict[str, set[int]] = {}
    displacement_by_transition: dict[int, list[float]] = {1: [], 2: []}
    for person in sample_people:
        track_id = str(person.get("track_id", ""))
        frame_idx = int(person.get("frame_index", 0))
        track_presence.setdefault(track_id, set()).add(frame_idx)
        if frame_idx in {1, 2} and person.get("displacement_from_prev") is not None:
            displacement_by_transition[frame_idx].append(float(person["displacement_from_prev"]))

    entering = [0.0, 0.0]
    leaving = [0.0, 0.0]
    for frames in track_presence.values():
        if 1 in frames and 0 not in frames:
            entering[0] += 1.0
        if 2 in frames and 1 not in frames:
            entering[1] += 1.0
        if 0 in frames and 1 not in frames:
            leaving[0] += 1.0
        if 1 in frames and 2 not in frames:
            leaving[1] += 1.0

    area_ratios = []
    for frame_meta in sample_payload["frames"]:
        tight_area = max(float(frame_meta["tight_area"]), 1.0)
        area_ratios.append(float(frame_meta["expanded_area"]) / tight_area)

    anchor_compactness = float(sample_payload["table"]["compactness_anchor"])
    mean_disp_1 = float(np.mean(displacement_by_transition[1])) if displacement_by_transition[1] else 0.0
    mean_disp_2 = float(np.mean(displacement_by_transition[2])) if displacement_by_transition[2] else 0.0
    max_disp_1 = float(np.max(displacement_by_transition[1])) if displacement_by_transition[1] else 0.0
    max_disp_2 = float(np.max(displacement_by_transition[2])) if displacement_by_transition[2] else 0.0

    scalars = np.array(
        [
            person_count[0],
            person_count[1],
            person_count[2],
            overlap_sum[0],
            overlap_sum[1],
            overlap_sum[2],
            max_overlap[0],
            max_overlap[1],
            max_overlap[2],
            min_distance[0],
            min_distance[1],
            min_distance[2],
            mean_disp_1,
            mean_disp_2,
            max_disp_1,
            max_disp_2,
            entering[0],
            entering[1],
            leaving[0],
            leaving[1],
            area_ratios[0],
            area_ratios[1],
            area_ratios[2],
            anchor_compactness,
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [
            table_vecs[0],
            table_vecs[1],
            table_vecs[2],
            per_frame_summary[0],
            per_frame_summary[1],
            per_frame_summary[2],
            scalars,
        ]
    ).astype(np.float32)


def write_review_bundle(
    sample_dir: Path,
    sample_payload: dict[str, Any],
    review_images: dict[str, Image.Image],
    perception_payload: dict[str, np.ndarray],
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    review_images["anchor"].save(sample_dir / PREVIEW_ANCHOR_NAME, format="JPEG", quality=92)
    review_images["t_minus_10"].save(sample_dir / PREVIEW_PREV_1_NAME, format="JPEG", quality=92)
    review_images["t_minus_20"].save(sample_dir / PREVIEW_PREV_2_NAME, format="JPEG", quality=92)
    review_images["tight_anchor"].save(sample_dir / TIGHT_ANCHOR_NAME, format="JPEG", quality=92)
    (sample_dir / SAMPLE_JSON_NAME).write_text(json.dumps(sample_payload, indent=2, sort_keys=True), encoding="utf-8")
    (sample_dir / PERCEPTION_NPZ_NAME).write_bytes(npz_bytes(perception_payload))


def load_sample_payload(sample_dir: Path) -> dict[str, Any]:
    return json.loads((sample_dir / SAMPLE_JSON_NAME).read_text(encoding="utf-8"))


def load_perception_payload(sample_dir: Path) -> dict[str, np.ndarray]:
    with np.load(sample_dir / PERCEPTION_NPZ_NAME, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def prepare_temporal_export(sample_dir: Path, human_label: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    sample_payload = update_label(load_sample_payload(sample_dir), human_label)
    files = {
        TEMPORAL_IMAGE_FILES[0]: (sample_dir / PREVIEW_PREV_2_NAME).read_bytes(),
        TEMPORAL_IMAGE_FILES[1]: (sample_dir / PREVIEW_PREV_1_NAME).read_bytes(),
        TEMPORAL_IMAGE_FILES[2]: (sample_dir / PREVIEW_ANCHOR_NAME).read_bytes(),
        SAMPLE_JSON_NAME: json.dumps(sample_payload, indent=2, sort_keys=True).encode("utf-8"),
    }
    return sample_payload, files


def prepare_surface_export(sample_dir: Path, human_label: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    sample_payload = update_label(load_sample_payload(sample_dir), human_label)
    files = {
        "surface.jpg": (sample_dir / TIGHT_ANCHOR_NAME).read_bytes(),
        SAMPLE_JSON_NAME: json.dumps(sample_payload, indent=2, sort_keys=True).encode("utf-8"),
    }
    return sample_payload, files


def prepare_occupancy_export(sample_dir: Path, human_label: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    sample_payload = update_label(load_sample_payload(sample_dir), human_label)
    perception_payload = load_perception_payload(sample_dir)
    x = build_occupancy_mlp_vector(sample_payload, perception_payload)
    files = {
        SAMPLE_JSON_NAME: json.dumps(sample_payload, indent=2, sort_keys=True).encode("utf-8"),
        OCCUPANCY_MLP_NPZ_NAME: npz_bytes(
            {
                "x": x,
                "y": np.array(1 if occupancy_binary_label(human_label) == "occupied" else 0, dtype=np.uint8),
            }
        ),
    }
    return sample_payload, files


def prepare_audit_export(sample_dir: Path, human_label: str) -> tuple[dict[str, Any], dict[str, bytes]]:
    sample_payload = update_label(load_sample_payload(sample_dir), human_label)
    files = {
        PREVIEW_ANCHOR_NAME: (sample_dir / PREVIEW_ANCHOR_NAME).read_bytes(),
        PREVIEW_PREV_1_NAME: (sample_dir / PREVIEW_PREV_1_NAME).read_bytes(),
        PREVIEW_PREV_2_NAME: (sample_dir / PREVIEW_PREV_2_NAME).read_bytes(),
        TIGHT_ANCHOR_NAME: (sample_dir / TIGHT_ANCHOR_NAME).read_bytes(),
        SAMPLE_JSON_NAME: json.dumps(sample_payload, indent=2, sort_keys=True).encode("utf-8"),
        PERCEPTION_NPZ_NAME: (sample_dir / PERCEPTION_NPZ_NAME).read_bytes(),
    }
    return sample_payload, files


def preview_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    return mask_bbox(mask)
