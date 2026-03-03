"""Service helpers for sample-level Drive review and training export."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from dataset_schema import (
    HUMAN_LABELS,
    PERCEPTION_NPZ_NAME,
    PREVIEW_FILE_BY_KIND,
    SAMPLE_JSON_NAME,
    TASK_ROOT_NAMES,
    TEMPORAL_IMAGE_FILES,
    export_targets_for_label,
    slugify,
)
from drive_client import DriveClient, DriveClientError
from manifest_writer import append_manifest_entry, remove_manifest_entry
from sample_builder import (
    build_manifest_entry,
    load_sample_payload,
    prepare_audit_export,
    prepare_occupancy_export,
    prepare_surface_export,
    prepare_temporal_export,
)
from video_review_store import VideoReviewStore


def ensure_review_roots(client: DriveClient, review_root_id: str) -> dict[str, str]:
    pending_root_id = client.ensure_subfolder(review_root_id, "pending")
    processed_root_id = client.ensure_subfolder(review_root_id, "_processed")
    recycle_root_id = client.ensure_subfolder(review_root_id, "_recycle")
    return {
        "pending": pending_root_id,
        "processed": processed_root_id,
        "recycle": recycle_root_id,
    }


def _bundle_file_ids(client: DriveClient, sample_folder_id: str) -> dict[str, str]:
    files = client.list_files(sample_folder_id)
    lookup = {file_meta.get("name"): file_meta.get("id") for file_meta in files}
    result = {kind: lookup.get(file_name) for kind, file_name in PREVIEW_FILE_BY_KIND.items()}
    result[SAMPLE_JSON_NAME] = lookup.get(SAMPLE_JSON_NAME)
    result[PERCEPTION_NPZ_NAME] = lookup.get(PERCEPTION_NPZ_NAME)
    return {key: value for key, value in result.items() if value}


def index_review_samples(
    client: DriveClient,
    store: VideoReviewStore,
    session_id: str,
    review_root_id: str,
) -> tuple[int, list[str], dict[str, str]]:
    roots = ensure_review_roots(client, review_root_id)
    sample_folders = client.list_folders_recursive(roots["pending"], include_root=False)
    queue_items: list[dict[str, Any]] = []
    errors: list[str] = []

    for folder in sample_folders:
        folder_id = folder["id"]
        file_ids = _bundle_file_ids(client, folder_id)
        sample_json_id = file_ids.get(SAMPLE_JSON_NAME)
        preview_anchor_id = file_ids.get("anchor")
        if not sample_json_id or not preview_anchor_id:
            continue
        try:
            sample_payload = json.loads(client.download_file_content(sample_json_id).decode("utf-8"))
        except (DriveClientError, json.JSONDecodeError) as exc:
            errors.append(f"{folder.get('name', folder_id)}: {exc}")
            continue

        parents = folder.get("parents") or [roots["pending"]]
        queue_items.append(
            {
                "session_id": session_id,
                "review_root_folder_id": review_root_id,
                "source_parent_folder_id": parents[0],
                "sample_folder_id": folder_id,
                "sample_folder_name": folder.get("name", folder_id),
                "sample": sample_payload,
                "preview_anchor_file_id": file_ids.get("anchor"),
                "preview_t_minus_10_file_id": file_ids.get("t_minus_10"),
                "preview_t_minus_20_file_id": file_ids.get("t_minus_20"),
                "tight_anchor_file_id": file_ids.get("tight_anchor"),
                "perception_file_id": file_ids.get(PERCEPTION_NPZ_NAME),
            }
        )

    inserted = store.add_queue_items(queue_items)
    return inserted, errors, roots


def ensure_cached_sample(client: DriveClient, item: dict[str, Any], cache_root: Path) -> Path:
    cache_namespace = str(item.get("session_id") or item.get("claimed_by_session_id") or "shared")
    sample_dir = cache_root / cache_namespace / str(item["id"])
    sample_dir.mkdir(parents=True, exist_ok=True)

    if not (sample_dir / SAMPLE_JSON_NAME).exists():
        (sample_dir / SAMPLE_JSON_NAME).write_text(
            json.dumps(item["sample"], indent=2, sort_keys=True),
            encoding="utf-8",
        )

    for kind, file_name in PREVIEW_FILE_BY_KIND.items():
        file_id = item.get(f"preview_{kind}_file_id") if kind != "tight_anchor" else item.get("tight_anchor_file_id")
        if not file_id:
            continue
        output_path = sample_dir / file_name
        if not output_path.exists():
            client.download_file_to_path(file_id, output_path)

    if item.get("perception_file_id") and not (sample_dir / PERCEPTION_NPZ_NAME).exists():
        client.download_file_to_path(item["perception_file_id"], sample_dir / PERCEPTION_NPZ_NAME)

    return sample_dir


def _task_root_id(task: str, root_ids: dict[str, str | None]) -> str | None:
    if task == "temporal":
        return root_ids.get("temporal")
    if task == "surface":
        return root_ids.get("surface")
    if task == "occupancy":
        return root_ids.get("occupancy")
    if task == "audit":
        return root_ids.get("audit")
    return None


def _prepare_export_target(
    sample_dir: Path,
    human_label: str,
    task: str,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    if task == "temporal":
        return prepare_temporal_export(sample_dir, human_label)
    if task == "surface":
        return prepare_surface_export(sample_dir, human_label)
    if task == "occupancy":
        return prepare_occupancy_export(sample_dir, human_label)
    if task == "audit":
        return prepare_audit_export(sample_dir, human_label)
    raise DriveClientError(f"Unsupported export task: {task}")


def _relative_folder_for_payload(prepared_payload: dict[str, Any], label: str) -> str:
    return "/".join(
        [
            label,
            slugify(prepared_payload["source_video"]["camera_id"]),
            slugify(Path(prepared_payload["source_video"]["video_name"]).stem),
            slugify(prepared_payload["table"]["table_track_id"]),
            prepared_payload["sample_id"],
        ]
    )


def _debug_file_entry(
    file_name: str,
    payload: bytes,
    *,
    source_preview_kind: str | None,
    prepared_payload: dict[str, Any],
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "file_name": file_name,
        "size_bytes": len(payload),
        "source_preview_kind": source_preview_kind,
    }
    if file_name.endswith(".jpg"):
        entry["content_type"] = "image/jpeg"
    elif file_name.endswith(".json"):
        entry["content_type"] = "application/json"
        entry["json_preview"] = {
            "sample_id": prepared_payload["sample_id"],
            "human_label": prepared_payload["label"]["human_label"],
            "occupancy_binary_label": prepared_payload["label"]["occupancy_binary_label"],
        }
    elif file_name.endswith(".npz"):
        entry["content_type"] = "application/octet-stream"
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            summary: dict[str, Any] = {}
            for key in arrays.files:
                value = arrays[key]
                item_summary: dict[str, Any] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                if value.size == 1:
                    item_summary["value"] = value.reshape(-1)[0].item()
                summary[key] = item_summary
        entry["npz_summary"] = summary
    else:
        entry["content_type"] = "application/octet-stream"
    return entry


def describe_sample_exports(
    sample_dir: Path,
    sample_payload: dict[str, Any],
    root_ids: dict[str, str | None],
) -> dict[str, Any]:
    preview_source_by_file = {
        TEMPORAL_IMAGE_FILES[0]: "t_minus_20",
        TEMPORAL_IMAGE_FILES[1]: "t_minus_10",
        TEMPORAL_IMAGE_FILES[2]: "anchor",
        "surface.jpg": "tight_anchor",
        PREVIEW_FILE_BY_KIND["anchor"]: "anchor",
        PREVIEW_FILE_BY_KIND["t_minus_10"]: "t_minus_10",
        PREVIEW_FILE_BY_KIND["t_minus_20"]: "t_minus_20",
        PREVIEW_FILE_BY_KIND["tight_anchor"]: "tight_anchor",
    }

    labels: dict[str, list[dict[str, Any]]] = {}
    audit_enabled = bool(root_ids.get("audit"))
    for human_label in HUMAN_LABELS:
        plans: list[dict[str, Any]] = []
        for target in export_targets_for_label(human_label, audit_enabled=audit_enabled):
            root_id = _task_root_id(target.task, root_ids)
            if not root_id:
                continue

            prepared_payload, files = _prepare_export_target(sample_dir, human_label, target.task)
            plans.append(
                {
                    "task": target.task,
                    "task_root_name": TASK_ROOT_NAMES[target.task],
                    "label": target.label,
                    "root_id": root_id,
                    "relative_folder": _relative_folder_for_payload(prepared_payload, target.label),
                    "files": [
                        _debug_file_entry(
                            file_name,
                            payload,
                            source_preview_kind=preview_source_by_file.get(file_name),
                            prepared_payload=prepared_payload,
                        )
                        for file_name, payload in files.items()
                    ],
                }
            )
        labels[human_label] = plans

    return {
        "sample_id": sample_payload["sample_id"],
        "camera_id": sample_payload["source_video"]["camera_id"],
        "video_name": sample_payload["source_video"]["video_name"],
        "anchor_time_seconds": sample_payload["timing"]["anchor_time_seconds"],
        "table_track_id": sample_payload["table"]["table_track_id"],
        "labels": labels,
    }


def _upload_export_files(client: DriveClient, sample_folder_id: str, files: dict[str, bytes]) -> None:
    for file_name, payload in files.items():
        mime_type = "application/octet-stream"
        if file_name.endswith(".jpg"):
            mime_type = "image/jpeg"
        elif file_name.endswith(".json"):
            mime_type = "application/json"
        client.upsert_bytes(sample_folder_id, file_name, payload, mime_type=mime_type)


def export_sample(
    client: DriveClient,
    sample_dir: Path,
    sample_payload: dict[str, Any],
    human_label: str,
    root_ids: dict[str, str | None],
) -> list[tuple[str, str]]:
    if human_label not in HUMAN_LABELS:
        raise DriveClientError(f"Unsupported human label: {human_label}")

    exported: list[tuple[str, str]] = []
    audit_enabled = bool(root_ids.get("audit"))
    for target in export_targets_for_label(human_label, audit_enabled=audit_enabled):
        root_id = _task_root_id(target.task, root_ids)
        if not root_id:
            continue

        label_folder_id = client.ensure_subfolder(root_id, target.label)
        camera_bucket = client.ensure_subfolder(label_folder_id, slugify(sample_payload["source_video"]["camera_id"]))
        video_bucket = client.ensure_subfolder(camera_bucket, slugify(Path(sample_payload["source_video"]["video_name"]).stem))
        table_bucket = client.ensure_subfolder(video_bucket, slugify(sample_payload["table"]["table_track_id"]))
        sample_folder_id = client.ensure_subfolder(table_bucket, sample_payload["sample_id"])

        prepared_payload, files = _prepare_export_target(sample_dir, human_label, target.task)

        _upload_export_files(client, sample_folder_id, files)
        relative_folder = _relative_folder_for_payload(prepared_payload, target.label)
        append_manifest_entry(client, root_id, build_manifest_entry(prepared_payload, target.label, relative_folder))
        exported.append((target.task, sample_folder_id))

    return exported


def archive_sample(
    client: DriveClient,
    review_roots: dict[str, str],
    sample_payload: dict[str, Any],
    sample_folder_id: str,
    source_parent_folder_id: str,
    label: str,
) -> str:
    label_root = client.ensure_subfolder(review_roots["processed"], label)
    camera_bucket = client.ensure_subfolder(label_root, slugify(sample_payload["source_video"]["camera_id"]))
    video_bucket = client.ensure_subfolder(camera_bucket, slugify(Path(sample_payload["source_video"]["video_name"]).stem))
    table_bucket = client.ensure_subfolder(video_bucket, slugify(sample_payload["table"]["table_track_id"]))
    client.move_file(sample_folder_id, new_parent_id=table_bucket, remove_parent_id=source_parent_folder_id)
    return table_bucket


def recycle_sample(
    client: DriveClient,
    review_roots: dict[str, str],
    sample_payload: dict[str, Any],
    sample_folder_id: str,
    source_parent_folder_id: str,
    session_id: str,
) -> str:
    session_bucket = client.ensure_subfolder(review_roots["recycle"], session_id)
    camera_bucket = client.ensure_subfolder(session_bucket, slugify(sample_payload["source_video"]["camera_id"]))
    video_bucket = client.ensure_subfolder(camera_bucket, slugify(Path(sample_payload["source_video"]["video_name"]).stem))
    table_bucket = client.ensure_subfolder(video_bucket, slugify(sample_payload["table"]["table_track_id"]))
    client.move_file(sample_folder_id, new_parent_id=table_bucket, remove_parent_id=source_parent_folder_id)
    return table_bucket


def undo_export_manifests(
    client: DriveClient,
    sample_payload: dict[str, Any],
    human_label: str,
    root_ids: dict[str, str | None],
) -> None:
    audit_enabled = bool(root_ids.get("audit"))
    for target in export_targets_for_label(human_label, audit_enabled=audit_enabled):
        root_id = _task_root_id(target.task, root_ids)
        if not root_id:
            continue
        remove_manifest_entry(client, root_id, sample_payload["sample_id"])
