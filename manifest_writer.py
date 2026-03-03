"""Helpers for maintaining Drive-backed JSONL manifests."""

from __future__ import annotations

import json
from typing import Any

from drive_client import DriveClient, DriveClientError


def append_manifest_entry(
    client: DriveClient,
    root_folder_id: str,
    entry: dict[str, Any],
    file_name: str = "manifest.jsonl",
) -> dict[str, Any]:
    """Append one entry to a JSONL manifest, deduplicated by sample_id."""
    existing = client.find_file_by_name(root_folder_id, file_name)
    lines: list[str] = []

    if existing:
        try:
            raw = client.download_file_content(existing["id"]).decode("utf-8")
            lines = [line for line in raw.splitlines() if line.strip()]
        except DriveClientError:
            lines = []

    sample_id = str(entry.get("sample_id", "")).strip()
    filtered = []
    for line in lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if sample_id and parsed.get("sample_id") == sample_id:
            continue
        filtered.append(line)

    filtered.append(json.dumps(entry, sort_keys=True))
    payload = ("\n".join(filtered) + "\n").encode("utf-8")
    return client.upsert_bytes(root_folder_id, file_name, payload, mime_type="application/json")


def remove_manifest_entry(
    client: DriveClient,
    root_folder_id: str,
    sample_id: str,
    file_name: str = "manifest.jsonl",
) -> dict[str, Any] | None:
    existing = client.find_file_by_name(root_folder_id, file_name)
    if not existing:
        return None

    try:
        raw = client.download_file_content(existing["id"]).decode("utf-8")
    except DriveClientError:
        return None

    lines = [line for line in raw.splitlines() if line.strip()]
    filtered = []
    for line in lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if parsed.get("sample_id") == sample_id:
            continue
        filtered.append(line)

    payload = ("\n".join(filtered) + ("\n" if filtered else "")).encode("utf-8")
    return client.update_bytes(existing["id"], payload, file_name=file_name, mime_type="application/json")
