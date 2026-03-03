"""Shared schema and routing helpers for the video review pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAMPLE_SCHEMA_VERSION = "sample_v1"
GENERATOR_VERSION = "video_dataset_pipeline_v1"
VECTOR_CONTRACT_VERSION = "derived_region_vector_v1"
VECTOR_DIM = 256
VIDEO_REVIEW_MODE = "video_triplet_review"

SAMPLE_JSON_NAME = "sample.json"
PERCEPTION_NPZ_NAME = "perception.npz"
OCCUPANCY_MLP_NPZ_NAME = "mlp_input.npz"
PREVIEW_ANCHOR_NAME = "preview_anchor.jpg"
PREVIEW_PREV_1_NAME = "preview_t_minus_10.jpg"
PREVIEW_PREV_2_NAME = "preview_t_minus_20.jpg"
TIGHT_ANCHOR_NAME = "tight_anchor.jpg"

PREVIEW_FILE_BY_KIND = {
    "anchor": PREVIEW_ANCHOR_NAME,
    "t_minus_10": PREVIEW_PREV_1_NAME,
    "t_minus_20": PREVIEW_PREV_2_NAME,
    "tight_anchor": TIGHT_ANCHOR_NAME,
}

HUMAN_LABELS = ("occupied", "dirty", "clean")
OCCUPANCY_BINARY_LABELS = ("occupied", "unoccupied")
TASK_ROOT_NAMES = {
    "review": "review_queue",
    "temporal": "temporal_state",
    "surface": "dirty_clean_surface",
    "occupancy": "occupancy_mlp",
    "audit": "sam_audit",
}

TEMPORAL_IMAGE_FILES = ("frame_0.jpg", "frame_1.jpg", "frame_2.jpg")


@dataclass(frozen=True)
class ExportTarget:
    task: str
    label: str


def occupancy_binary_label(human_label: str) -> str:
    return "occupied" if human_label == "occupied" else "unoccupied"


def export_targets_for_label(human_label: str, audit_enabled: bool = False) -> list[ExportTarget]:
    targets = [
        ExportTarget("temporal", human_label),
        ExportTarget("occupancy", occupancy_binary_label(human_label)),
    ]
    if human_label in {"dirty", "clean"}:
        targets.append(ExportTarget("surface", human_label))
    if audit_enabled:
        targets.append(ExportTarget("audit", human_label))
    return targets


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return slug or "item"


def build_sample_id(camera_id: str, video_name: str, table_track_id: str, anchor_time_seconds: int) -> str:
    return "_".join(
        [
            slugify(camera_id),
            slugify(Path(video_name).stem),
            slugify(table_track_id),
            f"t{anchor_time_seconds:06d}",
        ]
    )


def dump_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
