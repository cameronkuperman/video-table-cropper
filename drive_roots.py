"""Helpers for resolving grouped video workflow Drive roots."""

from __future__ import annotations

from typing import Any

from drive_client import DriveClient, DriveClientError

PROJECT_CHILDREN = {
    "source": "raw_videos",
    "review": "review_queue",
    "temporal": "temporal_state",
    "surface": "dirty_clean_surface",
    "occupancy": "occupancy_mlp",
    "audit": "sam_audit",
}


def resolve_video_pipeline_roots(
    client: DriveClient,
    *,
    project_root_id: str | None,
    source_root_id: str | None,
    review_root_id: str | None,
    temporal_root_id: str | None,
    surface_root_id: str | None,
    occupancy_root_id: str | None,
    audit_root_id: str | None,
) -> dict[str, str | None]:
    """Resolve explicit roots or create/find them beneath a project root."""
    roots: dict[str, str | None] = {
        "project": project_root_id or None,
        "source": source_root_id or None,
        "review": review_root_id or None,
        "temporal": temporal_root_id or None,
        "surface": surface_root_id or None,
        "occupancy": occupancy_root_id or None,
        "audit": audit_root_id or None,
    }

    if not project_root_id:
        return roots

    for key, folder_name in PROJECT_CHILDREN.items():
        if roots.get(key):
            continue
        if key == "audit" and not audit_root_id:
            # Audit stays optional even when a project root is present.
            roots[key] = client.ensure_subfolder(project_root_id, folder_name)
            continue
        roots[key] = client.ensure_subfolder(project_root_id, folder_name)

    return roots
