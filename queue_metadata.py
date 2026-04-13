"""Helpers for storing and reading label queue metadata on Drive folders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

QUEUE_SCHEMA_PROPERTY = "autolabel_queue_schema"
QUEUE_SCHEMA_VERSION = "1"
FRAME_PROPERTY_KEYS = {
    "frame_0": "autolabel_frame_0_id",
    "frame_1": "autolabel_frame_1_id",
    "frame_2": "autolabel_frame_2_id",
}


def extract_frame_ids_from_item(item: Mapping[str, Any]) -> dict[str, str | None]:
    app_properties = item.get("appProperties")
    if not isinstance(app_properties, Mapping):
        return {key: None for key in FRAME_PROPERTY_KEYS}

    frames: dict[str, str | None] = {}
    for frame_key, property_key in FRAME_PROPERTY_KEYS.items():
        value = app_properties.get(property_key)
        frames[frame_key] = str(value) if value else None
    return frames


def build_folder_app_properties(frames: Mapping[str, str | None]) -> dict[str, str]:
    properties = {QUEUE_SCHEMA_PROPERTY: QUEUE_SCHEMA_VERSION}
    for frame_key, property_key in FRAME_PROPERTY_KEYS.items():
        file_id = frames.get(frame_key)
        if file_id:
            properties[property_key] = str(file_id)
    return properties


def has_complete_frame_ids(frames: Mapping[str, str | None]) -> bool:
    return all(frames.get(frame_key) for frame_key in FRAME_PROPERTY_KEYS)
