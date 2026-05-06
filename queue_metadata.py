"""Helpers for storing and reading label queue metadata on Drive folders.

Frame slots are keyed `frame_0`..`frame_{MAX_FRAMES_PER_GROUP-1}` to support
groups larger than the legacy 3-frame triplet without breaking old data:
folders written under the old schema only have `frame_0_id`..`frame_2_id`
appProperties; folders written under the new schema may have up to
`frame_{MAX-1}_id`. Readers default to legacy size when no `n_frames` is given.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

QUEUE_SCHEMA_PROPERTY = "autolabel_queue_schema"
QUEUE_SCHEMA_VERSION = "1"

# Upper bound for frames per group. Sized for current 10-frame target plus
# headroom. Increasing this is safe; decreasing breaks old data.
MAX_FRAMES_PER_GROUP = 16
LEGACY_FRAMES_PER_GROUP = 3

FRAME_PROPERTY_KEYS: dict[str, str] = {
    f"frame_{i}": f"autolabel_frame_{i}_id" for i in range(MAX_FRAMES_PER_GROUP)
}


def frame_slot_keys(n_frames: int) -> tuple[str, ...]:
    """Return ('frame_0', 'frame_1', ..., 'frame_{n-1}')."""
    return tuple(f"frame_{i}" for i in range(n_frames))


def extract_frame_ids_from_item(
    item: Mapping[str, Any],
    n_frames: int | None = None,
) -> dict[str, str | None]:
    """Read frame_N_id appProperties off a Drive item.

    With `n_frames` provided: returns exactly that many keys (`frame_0`..
    `frame_{n-1}`), missing values as None. Used when the caller knows the
    expected group size.

    With `n_frames=None`: returns LEGACY_FRAMES_PER_GROUP keys for backwards
    compatibility with code that hasn't been updated to detect N.
    """
    size = n_frames if n_frames is not None else LEGACY_FRAMES_PER_GROUP
    keys = frame_slot_keys(size)

    app_properties = item.get("appProperties")
    if not isinstance(app_properties, Mapping):
        return {key: None for key in keys}

    frames: dict[str, str | None] = {}
    for frame_key in keys:
        property_key = FRAME_PROPERTY_KEYS.get(frame_key)
        value = app_properties.get(property_key) if property_key else None
        frames[frame_key] = str(value) if value else None
    return frames


def build_folder_app_properties(frames: Mapping[str, str | None]) -> dict[str, str]:
    """Build appProperties payload from a frames dict.

    Writes only the slots that have a non-empty file ID, so callers can pass
    a partially-populated dict and the metadata will only carry real IDs.
    """
    properties = {QUEUE_SCHEMA_PROPERTY: QUEUE_SCHEMA_VERSION}
    for frame_key, file_id in frames.items():
        property_key = FRAME_PROPERTY_KEYS.get(frame_key)
        if property_key and file_id:
            properties[property_key] = str(file_id)
    return properties


def has_complete_frame_ids(
    frames: Mapping[str, str | None],
    n_frames: int | None = None,
) -> bool:
    """True iff every expected frame slot has a non-empty ID.

    With `n_frames` given: checks exactly that many slots.
    Without: checks however many slots `frames` contains (so dynamically
    sized inputs validate against themselves).
    """
    if n_frames is not None:
        return all(frames.get(f"frame_{i}") for i in range(n_frames))
    if not frames:
        return False
    return all(value for value in frames.values())
