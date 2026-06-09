#!/usr/bin/env python3
"""Resumable Google Drive to Azure Blob migration for AutoLabeler."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blob_client import AzureBlobClient  # noqa: E402
from drive_client import DriveClient, FOLDER_MIME  # noqa: E402
from env_loader import load_local_env  # noqa: E402

PROGRESS_VERSION = 1
DEFAULT_PROGRESS_FILE = "/tmp/autolabeler-migration-state/drive_to_blob_progress.json"
FILE_FIELDS = "id,name,mimeType,parents,appProperties,modifiedTime,size,md5Checksum"
FOLDER_FIELDS = "id,name,mimeType,parents,appProperties,modifiedTime"
LABEL_DESTINATIONS = ("clean", "dirty", "occupied", "label_later", "discarded")
ROOT_WORKFLOW_FOLDERS = ("raw_videos", "temp_processing", "unlabeled", *LABEL_DESTINATIONS, "processed_raw")
REOLINK_SITE_ROOTS = ("restaurant-pi-1", "reolink-matthews-01")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_path(value: str) -> str:
    return str(value or "").strip().strip("/")


def _join(*parts: str) -> str:
    cleaned = [_clean_path(part) for part in parts if _clean_path(part)]
    return posixpath.join(*cleaned) if cleaned else ""


def _parent(path: str) -> str:
    parent = posixpath.dirname(_clean_path(path))
    return "" if parent == "." else parent


def _name(path: str) -> str:
    return posixpath.basename(_clean_path(path))


def _human_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


@dataclass(frozen=True)
class MigrationItem:
    source_id: str
    dest_path: str
    rel_path: str
    name: str
    mime_type: str
    is_folder: bool
    parent_source_id: str | None
    size_bytes: int | None = None
    md5_checksum: str | None = None
    app_properties: dict[str, str] | None = None


@dataclass
class Manifest:
    drive_root_id: str
    blob_root_prefix: str
    items: list[MigrationItem]
    duplicate_paths: dict[str, list[str]]

    @property
    def folders(self) -> list[MigrationItem]:
        return [item for item in self.items if item.is_folder]

    @property
    def files(self) -> list[MigrationItem]:
        return [item for item in self.items if not item.is_folder]

    @property
    def known_bytes(self) -> int:
        return sum(item.size_bytes or 0 for item in self.files)

    @property
    def unknown_size_files(self) -> int:
        return sum(1 for item in self.files if item.size_bytes is None)


class MigrationProgress:
    def __init__(self, path: Path, drive_root_id: str, blob_root_prefix: str) -> None:
        self.path = path
        self.drive_root_id = drive_root_id
        self.blob_root_prefix = blob_root_prefix
        self.payload: dict[str, Any] = {
            "version": PROGRESS_VERSION,
            "drive_root_id": drive_root_id,
            "blob_root_prefix": blob_root_prefix,
            "items": {},
            "file_id_to_blob_path": {},
            "updated_at": None,
        }
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if loaded.get("drive_root_id") != drive_root_id or loaded.get("blob_root_prefix") != blob_root_prefix:
                raise RuntimeError(
                    f"Progress file {path} belongs to a different source/destination. "
                    "Pass --progress-file for this migration or delete the old progress file."
                )
            self.payload.update(loaded)
            self.payload.setdefault("items", {})
            self.payload.setdefault("file_id_to_blob_path", {})

    @property
    def file_map(self) -> dict[str, str]:
        return self.payload.setdefault("file_id_to_blob_path", {})

    def item(self, source_id: str) -> dict[str, Any] | None:
        value = self.payload.setdefault("items", {}).get(source_id)
        return value if isinstance(value, dict) else None

    def is_done(self, item: MigrationItem) -> bool:
        record = self.item(item.source_id)
        if not record or record.get("status") != "done":
            return False
        if record.get("dest_path") != item.dest_path:
            return False
        if item.size_bytes is not None and record.get("size_bytes") not in (None, item.size_bytes):
            return False
        return True

    def mark_done(self, item: MigrationItem, *, skipped: bool = False) -> None:
        self.payload.setdefault("items", {})[item.source_id] = {
            "status": "done",
            "kind": "folder" if item.is_folder else "file",
            "dest_path": item.dest_path,
            "size_bytes": item.size_bytes,
            "md5_checksum": item.md5_checksum,
            "skipped_existing": bool(skipped),
            "completed_at": _now(),
        }
        if not item.is_folder:
            self.file_map[item.source_id] = item.dest_path
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.payload["updated_at"] = _now()
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self.payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)


def _parse_size(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _app_properties(item: dict[str, Any]) -> dict[str, str]:
    raw = item.get("appProperties") or {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items() if value is not None}


def build_manifest(
    drive: Any,
    drive_root_id: str,
    blob_root_prefix: str,
    *,
    progress_every: int = 25,
) -> Manifest:
    root = drive.get_file(drive_root_id, fields=FOLDER_FIELDS)
    queue: list[tuple[str, str, str]] = [(drive_root_id, "", _clean_path(blob_root_prefix))]
    items: list[MigrationItem] = [
        MigrationItem(
            source_id=drive_root_id,
            dest_path=_clean_path(blob_root_prefix),
            rel_path="",
            name=str(root.get("name") or _name(blob_root_prefix) or "root"),
            mime_type=FOLDER_MIME,
            is_folder=True,
            parent_source_id=None,
            app_properties=_app_properties(root),
        )
    ]
    seen_dest_paths: dict[str, list[str]] = {items[0].dest_path: [drive_root_id]}
    scanned_folders = 0
    print(f"Scanning Drive tree from root {drive_root_id}...", flush=True)

    while queue:
        folder_id, rel_path, dest_path = queue.pop(0)
        scanned_folders += 1
        if scanned_folders == 1 or (progress_every > 0 and scanned_folders % progress_every == 0):
            print(
                "  scanned "
                f"{scanned_folders} folders, discovered {sum(1 for item in items if item.is_folder)} folders "
                f"and {sum(1 for item in items if not item.is_folder)} files "
                f"(queue={len(queue)})",
                flush=True,
            )
        children = sorted(
            drive.list_files(folder_id, fields=FILE_FIELDS),
            key=lambda child: (str(child.get("name") or ""), str(child.get("id") or "")),
        )
        for child in children:
            child_id = str(child.get("id") or "")
            child_name = str(child.get("name") or child_id)
            if not child_id or not child_name:
                continue
            child_rel = _join(rel_path, child_name)
            child_dest = _join(dest_path, child_name)
            is_folder = child.get("mimeType") == FOLDER_MIME
            item = MigrationItem(
                source_id=child_id,
                dest_path=child_dest,
                rel_path=child_rel,
                name=child_name,
                mime_type=str(child.get("mimeType") or "application/octet-stream"),
                is_folder=is_folder,
                parent_source_id=folder_id,
                size_bytes=None if is_folder else _parse_size(child.get("size")),
                md5_checksum=None if is_folder else (str(child.get("md5Checksum")) if child.get("md5Checksum") else None),
                app_properties=_app_properties(child) if is_folder else None,
            )
            items.append(item)
            seen_dest_paths.setdefault(child_dest, []).append(child_id)
            if is_folder:
                queue.append((child_id, child_rel, child_dest))

    print(
        "Finished Drive scan: "
        f"{sum(1 for item in items if item.is_folder)} folders, "
        f"{sum(1 for item in items if not item.is_folder)} files.",
        flush=True,
    )
    duplicate_paths = {
        path: source_ids
        for path, source_ids in seen_dest_paths.items()
        if len(source_ids) > 1
    }
    return Manifest(drive_root_id=drive_root_id, blob_root_prefix=_clean_path(blob_root_prefix), items=items, duplicate_paths=duplicate_paths)


def resolve_folder_path(drive: Any, drive_root_id: str, rel_path: str) -> str:
    current_id = drive_root_id
    for part in _clean_path(rel_path).split("/"):
        if not part:
            continue
        match = drive.find_file_by_name(current_id, part, mime_type=FOLDER_MIME)
        if not match or not match.get("id"):
            raise RuntimeError(f"Drive folder path not found under root: {rel_path}")
        current_id = str(match["id"])
    return current_id


def build_scoped_manifest(
    drive: Any,
    drive_root_id: str,
    blob_root_prefix: str,
    source_path_prefixes: list[str],
    *,
    progress_every: int = 25,
) -> Manifest:
    root = drive.get_file(drive_root_id, fields=FOLDER_FIELDS)
    blob_root = _clean_path(blob_root_prefix)
    items: list[MigrationItem] = [
        MigrationItem(
            source_id=drive_root_id,
            dest_path=blob_root,
            rel_path="",
            name=str(root.get("name") or _name(blob_root) or "root"),
            mime_type=FOLDER_MIME,
            is_folder=True,
            parent_source_id=None,
            app_properties=_app_properties(root),
        )
    ]
    seen_dest_paths: dict[str, list[str]] = {blob_root: [drive_root_id]}
    queued: set[str] = set()
    queue: list[tuple[str, str, str]] = []

    for prefix in [_clean_path(path) for path in source_path_prefixes if _clean_path(path)]:
        current_source_id = drive_root_id
        current_rel = ""
        current_dest = blob_root
        parent_source_id: str | None = None
        for part in prefix.split("/"):
            current_source_id = resolve_folder_path(drive, current_source_id, part)
            current_rel = _join(current_rel, part)
            current_dest = _join(current_dest, part)
            folder = drive.get_file(current_source_id, fields=FOLDER_FIELDS)
            if current_source_id not in seen_dest_paths.get(current_dest, []):
                items.append(
                    MigrationItem(
                        source_id=current_source_id,
                        dest_path=current_dest,
                        rel_path=current_rel,
                        name=part,
                        mime_type=FOLDER_MIME,
                        is_folder=True,
                        parent_source_id=parent_source_id or drive_root_id,
                        app_properties=_app_properties(folder),
                    )
                )
                seen_dest_paths.setdefault(current_dest, []).append(current_source_id)
            parent_source_id = current_source_id
        if current_source_id not in queued:
            queue.append((current_source_id, current_rel, current_dest))
            queued.add(current_source_id)

    scanned_folders = 0
    print(
        "Scanning selected Drive subtree(s): "
        f"{', '.join(_clean_path(path) for path in source_path_prefixes if _clean_path(path))}",
        flush=True,
    )
    while queue:
        folder_id, rel_path, dest_path = queue.pop(0)
        scanned_folders += 1
        if scanned_folders == 1 or (progress_every > 0 and scanned_folders % progress_every == 0):
            print(
                "  scanned "
                f"{scanned_folders} folders, discovered {sum(1 for item in items if item.is_folder)} folders "
                f"and {sum(1 for item in items if not item.is_folder)} files "
                f"(queue={len(queue)})",
                flush=True,
            )
        children = sorted(
            drive.list_files(folder_id, fields=FILE_FIELDS),
            key=lambda child: (str(child.get("name") or ""), str(child.get("id") or "")),
        )
        for child in children:
            child_id = str(child.get("id") or "")
            child_name = str(child.get("name") or child_id)
            if not child_id or not child_name:
                continue
            child_rel = _join(rel_path, child_name)
            child_dest = _join(dest_path, child_name)
            is_folder = child.get("mimeType") == FOLDER_MIME
            item = MigrationItem(
                source_id=child_id,
                dest_path=child_dest,
                rel_path=child_rel,
                name=child_name,
                mime_type=str(child.get("mimeType") or "application/octet-stream"),
                is_folder=is_folder,
                parent_source_id=folder_id,
                size_bytes=None if is_folder else _parse_size(child.get("size")),
                md5_checksum=None if is_folder else (str(child.get("md5Checksum")) if child.get("md5Checksum") else None),
                app_properties=_app_properties(child) if is_folder else None,
            )
            items.append(item)
            seen_dest_paths.setdefault(child_dest, []).append(child_id)
            if is_folder:
                queue.append((child_id, child_rel, child_dest))

    print(
        "Finished selected Drive scan: "
        f"{sum(1 for item in items if item.is_folder)} folders, "
        f"{sum(1 for item in items if not item.is_folder)} files.",
        flush=True,
    )
    duplicate_paths = {
        path: source_ids
        for path, source_ids in seen_dest_paths.items()
        if len(source_ids) > 1
    }
    return Manifest(drive_root_id=drive_root_id, blob_root_prefix=blob_root, items=items, duplicate_paths=duplicate_paths)


def select_items(
    manifest: Manifest,
    *,
    source_path_prefixes: list[str],
    limit_folders: int | None,
    limit_files: int | None,
) -> list[MigrationItem]:
    prefixes = [_clean_path(prefix) for prefix in source_path_prefixes if _clean_path(prefix)]

    def matches_prefix(item: MigrationItem) -> bool:
        if not prefixes:
            return True
        return any(item.rel_path == prefix or item.rel_path.startswith(f"{prefix}/") for prefix in prefixes)

    by_source_id = {item.source_id: item for item in manifest.items}
    matched_ids = {
        item.source_id
        for item in manifest.items
        if item.rel_path == "" or matches_prefix(item)
    }
    for item in list(manifest.items):
        if item.source_id not in matched_ids:
            continue
        parent_id = item.parent_source_id
        while parent_id and parent_id in by_source_id:
            matched_ids.add(parent_id)
            parent_id = by_source_id[parent_id].parent_source_id

    filtered = [item for item in manifest.items if item.source_id in matched_ids]
    folders = [item for item in filtered if item.is_folder]
    files = [item for item in filtered if not item.is_folder]

    if limit_folders is not None:
        root = [item for item in folders if item.rel_path == ""]
        non_root = [item for item in folders if item.rel_path != ""]
        folders = root + non_root[: max(0, limit_folders)]
        allowed_folder_paths = {item.dest_path for item in folders}
        files = [item for item in files if _parent(item.dest_path) in allowed_folder_paths]

    if limit_files is not None:
        files = files[: max(0, limit_files)]

    selected_by_source_id = {item.source_id: item for item in folders + files}
    return list(selected_by_source_id.values())


def estimate_seconds(bytes_count: int, assume_mbps: float) -> float | None:
    if assume_mbps <= 0:
        return None
    return bytes_count / (assume_mbps * 1024 * 1024 / 8)


def print_summary(manifest: Manifest, selected: list[MigrationItem], assume_mbps: float) -> None:
    selected_files = [item for item in selected if not item.is_folder]
    selected_folders = [item for item in selected if item.is_folder]
    selected_bytes = sum(item.size_bytes or 0 for item in selected_files)
    selected_unknown = sum(1 for item in selected_files if item.size_bytes is None)

    print("Drive -> Azure Blob migration manifest")
    print(f"  Drive root: {manifest.drive_root_id}")
    print(f"  Blob root:  {manifest.blob_root_prefix}")
    print(f"  Full scan:  {len(manifest.folders)} folders, {len(manifest.files)} files, {_human_bytes(manifest.known_bytes)} known bytes")
    if manifest.unknown_size_files:
        print(f"              {manifest.unknown_size_files} files have unknown Drive size")
    print(f"  Selected:   {len(selected_folders)} folders, {len(selected_files)} files, {_human_bytes(selected_bytes)} known bytes")
    if selected_unknown:
        print(f"              {selected_unknown} selected files have unknown Drive size")
    print(f"  Estimate:   full {_duration(estimate_seconds(manifest.known_bytes, assume_mbps))}, selected {_duration(estimate_seconds(selected_bytes, assume_mbps))} at {assume_mbps:g} Mbps")
    if manifest.duplicate_paths:
        print(f"  Conflicts:  {len(manifest.duplicate_paths)} duplicate destination paths detected")


def ensure_blob_folder(blob: Any, dest_path: str) -> str:
    current = ""
    for part in _clean_path(dest_path).split("/"):
        if part:
            current = blob.ensure_subfolder(current, part)
    return current


def ensure_autolabeler_skeleton(blob: Any, blob_root_prefix: str) -> list[str]:
    root = _clean_path(blob_root_prefix)
    created: list[str] = []
    for path in [root, *(_join(root, name) for name in ROOT_WORKFLOW_FOLDERS)]:
        ensure_blob_folder(blob, path)
        created.append(path)

    true_ten_root = _join(root, "10frametrue")
    ensure_blob_folder(blob, true_ten_root)
    created.append(true_ten_root)

    for site in REOLINK_SITE_ROOTS:
        site_root = _join(root, site)
        site_paths = [
            site_root,
            _join(site_root, "unassociated"),
            _join(site_root, "unassociated_zips"),
            _join(site_root, "unlabeled"),
            _join(site_root, "processed_raw"),
            _join(site_root, "3frame"),
            _join(site_root, "3frame", "unlabeled"),
            _join(true_ten_root, site),
        ]
        if site == "reolink-matthews-01":
            site_paths.append(_join(site_root, "crop_configs"))
        for path in site_paths:
            ensure_blob_folder(blob, path)
            created.append(path)
    return created


def destination_size(blob: Any, dest_path: str) -> int | None:
    if hasattr(blob, "blob_size"):
        return blob.blob_size(dest_path)
    return None


def translate_app_properties(app_properties: dict[str, str] | None, file_map: dict[str, str]) -> dict[str, str]:
    translated: dict[str, str] = {}
    for key, value in (app_properties or {}).items():
        translated[str(key)] = file_map.get(str(value), str(value))
    return translated


def item_from_drive(
    drive_item: dict[str, Any],
    *,
    dest_path: str,
    rel_path: str,
    parent_source_id: str | None,
) -> MigrationItem:
    is_folder = drive_item.get("mimeType") == FOLDER_MIME
    return MigrationItem(
        source_id=str(drive_item.get("id") or ""),
        dest_path=dest_path,
        rel_path=rel_path,
        name=str(drive_item.get("name") or _name(dest_path)),
        mime_type=str(drive_item.get("mimeType") or "application/octet-stream"),
        is_folder=is_folder,
        parent_source_id=parent_source_id,
        size_bytes=None if is_folder else _parse_size(drive_item.get("size")),
        md5_checksum=None if is_folder else (str(drive_item.get("md5Checksum")) if drive_item.get("md5Checksum") else None),
        app_properties=_app_properties(drive_item) if is_folder else None,
    )


def migrate_one_file(
    drive: Any,
    blob: Any,
    file_item: MigrationItem,
    progress: MigrationProgress,
    *,
    verify: bool,
    skip_existing: bool,
) -> str:
    existing_size = destination_size(blob, file_item.dest_path) if skip_existing else None
    can_skip_existing = existing_size is not None and (file_item.size_bytes is None or existing_size == file_item.size_bytes)
    can_skip_progress = progress.is_done(file_item) and can_skip_existing
    if can_skip_progress or can_skip_existing:
        progress.mark_done(file_item, skipped=True)
        return "skip"

    data = drive.download_file_content(file_item.source_id)
    if file_item.md5_checksum:
        digest = hashlib.md5(data).hexdigest()
        if digest != file_item.md5_checksum:
            raise RuntimeError(f"MD5 mismatch for {file_item.rel_path}: Drive={file_item.md5_checksum} downloaded={digest}")

    parent_id = ensure_blob_folder(blob, _parent(file_item.dest_path))
    uploaded = blob.upload_bytes(
        data,
        parent_id,
        _name(file_item.dest_path),
        mime_type=file_item.mime_type or "application/octet-stream",
    )
    uploaded_id = str(uploaded.get("id") or file_item.dest_path)
    progress.file_map[file_item.source_id] = uploaded_id
    if verify:
        uploaded_size = destination_size(blob, uploaded_id)
        if file_item.size_bytes is not None and uploaded_size != file_item.size_bytes:
            raise RuntimeError(f"Uploaded size mismatch for {file_item.rel_path}: expected {file_item.size_bytes}, got {uploaded_size}")
    progress.mark_done(file_item)
    return "upload"


def migrate_items(
    drive: Any,
    blob: Any,
    items: list[MigrationItem],
    progress: MigrationProgress,
    *,
    verify: bool,
    skip_existing: bool,
) -> None:
    folders = sorted([item for item in items if item.is_folder], key=lambda item: item.dest_path.count("/"))
    files = sorted([item for item in items if not item.is_folder], key=lambda item: item.dest_path)
    folder_by_dest = {item.dest_path: item for item in folders}

    start = time.time()
    for index, folder in enumerate(folders, start=1):
        if progress.is_done(folder):
            continue
        ensure_blob_folder(blob, folder.dest_path)
        progress.mark_done(folder)
        print(f"[folder {index}/{len(folders)}] {folder.dest_path}")

    for index, file_item in enumerate(files, start=1):
        action = migrate_one_file(
            drive,
            blob,
            file_item,
            progress,
            verify=verify,
            skip_existing=skip_existing,
        )
        elapsed = max(0.001, time.time() - start)
        rate = (sum(item.size_bytes or 0 for item in files[:index]) / elapsed) if elapsed else 0
        print(f"[file {index}/{len(files)}] {action} {file_item.dest_path} ({_human_bytes(file_item.size_bytes)}, {_human_bytes(int(rate))}/s)")

    for index, folder in enumerate(folders, start=1):
        if not folder.app_properties:
            continue
        translated = translate_app_properties(folder.app_properties, progress.file_map)
        blob.update_file_metadata(folder.dest_path, {"appProperties": translated})
        progress.mark_done(folder)
        if folder_by_dest.get(folder.dest_path):
            print(f"[metadata {index}/{len(folders)}] {folder.dest_path}")


def migrate_prefix_without_scan(
    drive: Any,
    blob: Any,
    drive_root_id: str,
    blob_root_prefix: str,
    source_path_prefix: str,
    progress: MigrationProgress,
    *,
    limit_folders: int | None,
    limit_files: int | None,
    verify: bool,
    skip_existing: bool,
) -> None:
    blob_root = _clean_path(blob_root_prefix)
    root = drive.get_file(drive_root_id, fields=FOLDER_FIELDS)
    root_item = item_from_drive(
        root,
        dest_path=blob_root,
        rel_path="",
        parent_source_id=None,
    )
    ensure_blob_folder(blob, root_item.dest_path)
    progress.mark_done(root_item)

    current_source_id = drive_root_id
    current_rel = ""
    current_dest = blob_root
    parent_source_id: str | None = drive_root_id
    prefix = _clean_path(source_path_prefix)
    print(f"Fast migration without pre-scan: {prefix}", flush=True)

    for part in prefix.split("/"):
        if not part:
            continue
        match = drive.find_file_by_name(current_source_id, part, mime_type=FOLDER_MIME)
        if not match or not match.get("id"):
            raise RuntimeError(f"Drive folder path not found under root: {source_path_prefix}")
        current_source_id = str(match["id"])
        current_rel = _join(current_rel, part)
        current_dest = _join(current_dest, part)
        folder = drive.get_file(current_source_id, fields=FOLDER_FIELDS)
        folder_item = item_from_drive(
            folder,
            dest_path=current_dest,
            rel_path=current_rel,
            parent_source_id=parent_source_id,
        )
        ensure_blob_folder(blob, folder_item.dest_path)
        progress.mark_done(folder_item)
        print(f"[folder] {folder_item.dest_path}", flush=True)
        parent_source_id = current_source_id

    folders_seen = 0
    files_seen = 0
    queue: list[MigrationItem] = [
        MigrationItem(
            source_id=current_source_id,
            dest_path=current_dest,
            rel_path=current_rel,
            name=_name(current_dest),
            mime_type=FOLDER_MIME,
            is_folder=True,
            parent_source_id=parent_source_id,
            app_properties={},
        )
    ]
    queued_folder_ids = {current_source_id}

    while queue:
        folder_item = queue.pop(0)
        children = sorted(
            drive.list_files(folder_item.source_id, fields=FILE_FIELDS),
            key=lambda child: (str(child.get("mimeType") != FOLDER_MIME), str(child.get("name") or ""), str(child.get("id") or "")),
        )
        child_folders: list[MigrationItem] = []
        for child in children:
            child_id = str(child.get("id") or "")
            child_name = str(child.get("name") or child_id)
            if not child_id or not child_name:
                continue
            child_item = item_from_drive(
                child,
                dest_path=_join(folder_item.dest_path, child_name),
                rel_path=_join(folder_item.rel_path, child_name),
                parent_source_id=folder_item.source_id,
            )
            if child_item.is_folder:
                child_folders.append(child_item)
                continue

            if limit_files is not None and files_seen >= limit_files:
                continue
            files_seen += 1
            action = migrate_one_file(
                drive,
                blob,
                child_item,
                progress,
                verify=verify,
                skip_existing=skip_existing,
            )
            print(f"[file {files_seen}] {action} {child_item.dest_path} ({_human_bytes(child_item.size_bytes)})", flush=True)

        if folder_item.app_properties:
            translated = translate_app_properties(folder_item.app_properties, progress.file_map)
            blob.update_file_metadata(folder_item.dest_path, {"appProperties": translated})
            progress.mark_done(folder_item)
            print(f"[metadata] {folder_item.dest_path}", flush=True)

        for child_folder in child_folders:
            if child_folder.source_id in queued_folder_ids:
                continue
            if limit_folders is not None and folders_seen >= limit_folders:
                continue
            folders_seen += 1
            ensure_blob_folder(blob, child_folder.dest_path)
            progress.mark_done(child_folder)
            print(f"[folder {folders_seen}] {child_folder.dest_path}", flush=True)
            queue.append(child_folder)
            queued_folder_ids.add(child_folder.source_id)

    print(
        f"Fast migration complete for {prefix}: {folders_seen} descendant folders, {files_seen} files touched.",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root-id", default=os.environ.get("DRIVE_PROJECT_ROOT_FOLDER_ID", "").strip())
    parser.add_argument("--blob-root-prefix", default=os.environ.get("AZURE_PROJECT_ROOT_PREFIX", "project-root").strip())
    parser.add_argument("--progress-file", type=Path, default=Path(os.environ.get("MIGRATION_PROGRESS_FILE", DEFAULT_PROGRESS_FILE)))
    parser.add_argument("--source-path-prefix", action="append", default=[], help="Only migrate a relative Drive path prefix; can be repeated.")
    parser.add_argument("--limit-folders", type=int, default=None, help="Migrate at most this many non-root folders from the selected manifest.")
    parser.add_argument("--limit-files", type=int, default=None, help="Migrate at most this many files from the selected manifest.")
    parser.add_argument("--assume-mbps", type=float, default=float(os.environ.get("MIGRATION_ASSUME_MBPS", "5") or "5"))
    parser.add_argument("--scan-progress-every", type=int, default=25, help="Print Drive scan progress after this many folders; use 0 to disable periodic logs.")
    parser.add_argument("--dry-run", action="store_true", help="Scan/count/estimate only; do not write Azure Blob data.")
    parser.add_argument("--verify", action="store_true", help="Verify uploaded Blob sizes when Drive exposes source sizes.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Reupload files even when matching Blob paths already exist.")
    parser.add_argument("--ignore-name-conflicts", action="store_true", help="Proceed even if multiple Drive items map to the same Blob path.")
    parser.add_argument("--full-scan", action="store_true", help="Scan the whole Drive tree even when --source-path-prefix is provided.")
    parser.add_argument(
        "--dont-scan",
        "--dont_scan",
        action="store_true",
        help="Skip manifest/count scan and immediately migrate the selected source prefix.",
    )
    parser.add_argument(
        "--ensure-autolabeler-skeleton",
        action="store_true",
        help="Create the AutoLabeler root/site workflow folders in Blob before migrating.",
    )
    return parser.parse_args()


def main() -> None:
    load_local_env()
    args = parse_args()
    if not args.drive_root_id:
        raise SystemExit("DRIVE_PROJECT_ROOT_FOLDER_ID or --drive-root-id is required.")
    if not args.blob_root_prefix:
        raise SystemExit("AZURE_PROJECT_ROOT_PREFIX or --blob-root-prefix is required.")

    drive = DriveClient()
    if args.dont_scan:
        if args.dry_run:
            raise SystemExit("--dont-scan is for real migration; use --dry-run without --dont-scan to inspect counts.")
        if not args.source_path_prefix:
            raise SystemExit("--dont-scan requires at least one --source-path-prefix.")
        os.environ["STORAGE_BACKEND"] = "azure"
        os.environ["AZURE_PROJECT_ROOT_PREFIX"] = _clean_path(args.blob_root_prefix)
        progress = MigrationProgress(args.progress_file, args.drive_root_id, _clean_path(args.blob_root_prefix))
        blob = AzureBlobClient()
        if args.ensure_autolabeler_skeleton:
            created = ensure_autolabeler_skeleton(blob, args.blob_root_prefix)
            print(f"Ensured AutoLabeler Blob skeleton: {len(created)} folders")
        for prefix in args.source_path_prefix:
            migrate_prefix_without_scan(
                drive,
                blob,
                args.drive_root_id,
                args.blob_root_prefix,
                prefix,
                progress,
                limit_folders=args.limit_folders,
                limit_files=args.limit_files,
                verify=args.verify,
                skip_existing=not args.no_skip_existing,
            )
        print(f"Migration complete. Progress saved to {args.progress_file}")
        return

    if args.source_path_prefix and not args.full_scan:
        manifest = build_scoped_manifest(
            drive,
            args.drive_root_id,
            args.blob_root_prefix,
            args.source_path_prefix,
            progress_every=args.scan_progress_every,
        )
    else:
        manifest = build_manifest(
            drive,
            args.drive_root_id,
            args.blob_root_prefix,
            progress_every=args.scan_progress_every,
        )
    selected = select_items(
        manifest,
        source_path_prefixes=args.source_path_prefix,
        limit_folders=args.limit_folders,
        limit_files=args.limit_files,
    )
    print_summary(manifest, selected, args.assume_mbps)

    if manifest.duplicate_paths and not args.ignore_name_conflicts:
        for path, source_ids in list(manifest.duplicate_paths.items())[:10]:
            print(f"  duplicate: {path} <- {', '.join(source_ids)}")
        raise SystemExit("Duplicate destination paths found. Resolve Drive duplicates or pass --ignore-name-conflicts.")

    if args.dry_run:
        print("Dry run complete; no Azure Blob writes performed.")
        return

    os.environ["STORAGE_BACKEND"] = "azure"
    os.environ["AZURE_PROJECT_ROOT_PREFIX"] = _clean_path(args.blob_root_prefix)
    progress = MigrationProgress(args.progress_file, args.drive_root_id, _clean_path(args.blob_root_prefix))
    blob = AzureBlobClient()
    if args.ensure_autolabeler_skeleton:
        created = ensure_autolabeler_skeleton(blob, args.blob_root_prefix)
        print(f"Ensured AutoLabeler Blob skeleton: {len(created)} folders")
    migrate_items(
        drive,
        blob,
        selected,
        progress,
        verify=args.verify,
        skip_existing=not args.no_skip_existing,
    )
    print(f"Migration complete. Progress saved to {args.progress_file}")


if __name__ == "__main__":
    main()
