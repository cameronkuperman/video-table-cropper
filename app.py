"""
--label mode: Flask UI that reads unlabeled/ subfolders from Drive,
shows 3 images per folder, and moves the folder on Drive when labeled.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlencode

from flask import Flask, abort, g, jsonify, render_template, request, send_file

from drive_client import DriveClient, DriveClientError, FOLDER_MIME
from env_loader import load_local_env
from queue_metadata import (
    build_folder_app_properties,
    extract_frame_ids_from_item,
    has_complete_frame_ids,
)

load_local_env()

app = Flask(__name__)

QUEUE_BATCH_DEFAULT = max(36, int(os.environ.get("LABEL_QUEUE_BATCH_DEFAULT", "72") or "72"))
QUEUE_BATCH_MAX = max(QUEUE_BATCH_DEFAULT, int(os.environ.get("LABEL_QUEUE_BATCH_MAX", "300") or "300"))
CACHE_CLEANUP_INTERVAL_SECONDS = 300
UNLABELED_LIST_CACHE_SECONDS = max(
    15, int(os.environ.get("LABEL_UNLABELED_CACHE_SECONDS", "300") or "300")
)
HYDRATE_MAX_WORKERS = max(2, int(os.environ.get("LABEL_QUEUE_HYDRATE_WORKERS", "12") or "12"))
PREVIEW_PREWARM_MAX_WORKERS = max(
    2, int(os.environ.get("LABEL_PREVIEW_PREWARM_WORKERS", "24") or "24")
)
FOLDER_PREWARM_MAX_WORKERS = max(
    2, int(os.environ.get("LABEL_FOLDER_PREWARM_WORKERS", "12") or "12")
)
PREWARM_FOLDER_COUNT = max(12, int(os.environ.get("LABEL_PREWARM_FOLDER_COUNT", "180") or "180"))
HYDRATED_FOLDER_CACHE_TTL_SECONDS = max(60, int(os.environ.get("LABEL_HYDRATED_CACHE_TTL_SECONDS", "900") or "900"))
READY_SCAN_MULTIPLIER = max(2, int(os.environ.get("LABEL_READY_SCAN_MULTIPLIER", "12") or "12"))
READY_SCAN_MAX = max(100, int(os.environ.get("LABEL_READY_SCAN_MAX", "720") or "720"))
QUEUE_RETRY_MS = max(100, int(os.environ.get("LABEL_QUEUE_RETRY_MS", "250") or "250"))
TIMING_LOGS_ENABLED = os.environ.get("LABEL_TIMING_LOGS", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
TIMING_LOG_MIN_MS = max(0.0, float(os.environ.get("LABEL_TIMING_LOG_MIN_MS", "0") or "0"))

VIDEO_SOURCE = "video"
REOLINK_SOURCE = "reolink"
LABEL_DESTINATIONS = ("clean", "dirty", "occupied", "label_later")
MATTHEWS_SITE_KEY = "reolink-matthews-01"
CROP_CONFIGS_FOLDER_NAME = "crop_configs"


@dataclass(frozen=True)
class ReolinkSiteConfig:
    site_key: str
    display_name: str
    root_name: str
    root_id: str = ""
    manual_crop_configs: bool = False


@dataclass(frozen=True)
class QueueContext:
    source: str
    site_key: str | None
    queue_key: str
    display_name: str
    input_folder_name: str
    input_folder_id: str
    seed_folder_name: str | None
    seed_folder_id: str | None
    folder_ids: dict[str, str]
    persist_frame_metadata: bool

    def to_payload(self) -> dict[str, str | None]:
        return {
            "source": self.source,
            "site_key": self.site_key,
            "queue_key": self.queue_key,
            "display_name": self.display_name,
            "pending_label": self.input_folder_name,
        }


class CropSetupRequiredError(RuntimeError):
    def __init__(self, site_key: str, site_label: str, missing_channels: list[str]) -> None:
        unique_channels = sorted(set(missing_channels), key=_reolink_channel_sort_key)
        self.site_key = site_key
        self.site_label = site_label
        self.missing_channels = unique_channels

        channel_text = ", ".join(unique_channels)
        plural = "s" if len(unique_channels) != 1 else ""
        super().__init__(
            f"{site_label} needs saved crop config{plural} for {channel_text}. "
            "Open the crop editor and save crops before loading this queue."
        )

    def to_payload(self) -> dict[str, object]:
        first_channel = self.missing_channels[0] if self.missing_channels else None
        return {
            "error": str(self),
            "setup_required": True,
            "site_key": self.site_key,
            "missing_channels": self.missing_channels,
            "setup_url": _crop_editor_url(self.site_key, first_channel),
        }


# Keep this site list in code for now. If a site folder does not live directly
# under DRIVE_PROJECT_ROOT_FOLDER_ID, set its root_id explicitly here.
REOLINK_SITES = (
    ReolinkSiteConfig(
        site_key="reolink-matthews-01",
        display_name="Reolink Matthews 01",
        root_name="reolink-matthews-01",
        manual_crop_configs=True,
    ),
    ReolinkSiteConfig(
        site_key="restaurant-pi-1",
        display_name="Restaurant Pi 1",
        root_name="restaurant-pi-1",
    ),
)
REOLINK_SITES_BY_KEY = {site.site_key: site for site in REOLINK_SITES}


def _default_cache_dir() -> Path:
    configured = os.environ.get("LABEL_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()

    local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
    if local_appdata:
        return Path(local_appdata) / "AutoLabeler" / "label_cache"

    return Path(tempfile.gettempdir()) / "AutoLabeler" / "label_cache"


CACHE_DIR = _default_cache_dir()
CACHE_TTL_HOURS = max(1, int(os.environ.get("LABEL_CACHE_TTL_HOURS", "72") or "72"))
CACHE_MAX_MB = max(64, int(os.environ.get("LABEL_CACHE_MAX_MB", "1024") or "1024"))

# Drive client + cached folder IDs
_source_folder_ids_cache: dict[str, dict[str, str]] = {}
_source_folder_ids_lock = Lock()
_cache_cleanup_lock = Lock()
_last_cache_cleanup_monotonic = 0.0
_listing_cache: dict[str, tuple[float, list[dict[str, str]]]] = {}
_listing_lock = Lock()
_listing_refresh_executor = ThreadPoolExecutor(max_workers=1)
_listing_refresh_inflight: set[str] = set()
_preview_prewarm_executor = ThreadPoolExecutor(max_workers=PREVIEW_PREWARM_MAX_WORKERS)
_preview_prewarm_inflight: set[str] = set()
_preview_prewarm_lock = Lock()
_folder_prewarm_executor = ThreadPoolExecutor(max_workers=FOLDER_PREWARM_MAX_WORKERS)
_folder_prewarm_inflight: set[tuple[str, str]] = set()
_folder_prewarm_lock = Lock()
_hydrated_folder_cache: dict[tuple[str, str], tuple[float, dict | None]] = {}
_hydrated_folder_cache_lock = Lock()
_reolink_generation_lock = Lock()
_yolo_model_lock = Lock()
_yolo_model: Any | None = None
_camera_config_cache: dict[int, dict[str, Any]] | None = None
_camera_config_lock = Lock()
_crop_config_cache: dict[tuple[str, str], dict[str, Any] | None] = {}
_crop_config_lock = Lock()
_CROP_CONFIG_CACHE_MISS = object()


def _log_timing(event: str, **fields: object) -> None:
    if not TIMING_LOGS_ENABLED:
        return

    elapsed_ms = fields.get("total_ms")
    if isinstance(elapsed_ms, (int, float)) and elapsed_ms < TIMING_LOG_MIN_MS:
        return

    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[timing] {event} {details}".rstrip())


def get_client() -> DriveClient:
    client = getattr(g, "_drive_client", None)
    if client is None:
        client = DriveClient()
        g._drive_client = client
    return client


def _ensure_cache_dir() -> Path:
    global CACHE_DIR

    candidates = [CACHE_DIR]
    temp_cache = Path(tempfile.gettempdir()) / "AutoLabeler" / "label_cache"
    repo_cache = Path(__file__).parent / "label_cache"

    for candidate in (temp_cache, repo_cache):
        if candidate not in candidates:
            candidates.append(candidate)

    last_error: OSError | None = None
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            CACHE_DIR = candidate
            return candidate
        except OSError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    return CACHE_DIR


def _root_id() -> str:
    root = os.environ.get("DRIVE_PROJECT_ROOT_FOLDER_ID", "").strip()
    if not root:
        raise RuntimeError("DRIVE_PROJECT_ROOT_FOLDER_ID is not set in .env")
    return root


def _default_reolink_site_key() -> str | None:
    return REOLINK_SITES[0].site_key if REOLINK_SITES else None


def _resolve_site_config(site_key: str) -> ReolinkSiteConfig:
    site = REOLINK_SITES_BY_KEY.get(site_key)
    if site is None:
        raise ValueError(f"Unknown Reolink site: {site_key}")
    return site


def _site_uses_manual_crop_configs(site_key: str | None) -> bool:
    return bool(site_key and _resolve_site_config(site_key).manual_crop_configs)


def _crop_editor_url(site_key: str, channel_code: str | None = None) -> str:
    params = {"site": site_key}
    if channel_code:
        params["channel"] = channel_code
    return f"/crop-editor?{urlencode(params)}"


def _extract_reolink_channel_code(text: str) -> str | None:
    match = re.search(r"CH-CH(\d+)", text, re.IGNORECASE)
    if not match:
        return None
    return f"CH-CH{int(match.group(1)):02d}"


def _normalize_reolink_channel_code(value: str) -> str | None:
    normalized = _extract_reolink_channel_code(value)
    if normalized:
        return normalized

    digits_only = re.fullmatch(r"\s*(\d+)\s*", value or "")
    if digits_only:
        return f"CH-CH{int(digits_only.group(1)):02d}"
    return None


def _reolink_channel_sort_key(channel_code: str) -> tuple[int, str]:
    channel_number = _extract_reolink_channel_number(channel_code)
    return (channel_number if channel_number is not None else 10_000, channel_code)


def _discover_reolink_root_id(client: DriveClient, site: ReolinkSiteConfig) -> str:
    if site.root_id.strip():
        return site.root_id.strip()

    discovered = client.find_file_by_name(_root_id(), site.root_name, mime_type=FOLDER_MIME)
    if discovered and discovered.get("id"):
        return str(discovered["id"])

    raise RuntimeError(
        f"Could not find Reolink site folder '{site.root_name}' under DRIVE_PROJECT_ROOT_FOLDER_ID. "
        f"Set the site root_id in app.py if it lives elsewhere."
    )


def _video_folder_ids(client: DriveClient) -> dict[str, str]:
    queue_key = VIDEO_SOURCE
    with _source_folder_ids_lock:
        cached = _source_folder_ids_cache.get(queue_key)
        if cached is not None:
            return cached

        root = _root_id()
        names = [
            "raw_videos",
            "temp_processing",
            "unlabeled",
            *LABEL_DESTINATIONS,
        ]
        folder_ids = {name: client.ensure_subfolder(root, name) for name in names}
        _source_folder_ids_cache[queue_key] = folder_ids
        return folder_ids


def _reolink_site_folder_ids(client: DriveClient, site_key: str) -> dict[str, str]:
    queue_key = f"{REOLINK_SOURCE}:{site_key}"
    with _source_folder_ids_lock:
        cached = _source_folder_ids_cache.get(queue_key)
        if cached is not None:
            if _site_uses_manual_crop_configs(site_key) and "crop_configs" not in cached:
                cached = dict(cached)
                cached["crop_configs"] = client.ensure_subfolder(cached["root"], CROP_CONFIGS_FOLDER_NAME)
                _source_folder_ids_cache[queue_key] = cached
            return cached

        site = _resolve_site_config(site_key)
        site_root_id = _discover_reolink_root_id(client, site)
        unassociated = client.find_file_by_name(site_root_id, "unassociated", mime_type=FOLDER_MIME)
        if not unassociated or not unassociated.get("id"):
            raise RuntimeError(
                f"Reolink site '{site.display_name}' is missing required folder 'unassociated'."
            )

        folder_ids = {
            "root": site_root_id,
            "unassociated": str(unassociated["id"]),
            "unlabeled": client.ensure_subfolder(site_root_id, "unlabeled"),
        }
        if site.manual_crop_configs:
            folder_ids["crop_configs"] = client.ensure_subfolder(site_root_id, CROP_CONFIGS_FOLDER_NAME)
        for name in LABEL_DESTINATIONS:
            folder_ids[name] = client.ensure_subfolder(site_root_id, name)

        _source_folder_ids_cache[queue_key] = folder_ids
        return folder_ids


def _crop_config_cache_key(site_key: str, channel_code: str) -> tuple[str, str]:
    return (site_key, channel_code)


def _get_cached_crop_config(site_key: str, channel_code: str) -> dict[str, Any] | None | object:
    with _crop_config_lock:
        return _crop_config_cache.get(_crop_config_cache_key(site_key, channel_code), _CROP_CONFIG_CACHE_MISS)


def _set_cached_crop_config(site_key: str, channel_code: str, payload: dict[str, Any] | None) -> None:
    with _crop_config_lock:
        _crop_config_cache[_crop_config_cache_key(site_key, channel_code)] = payload


def _invalidate_crop_config_cache(site_key: str, channel_code: str | None = None) -> None:
    with _crop_config_lock:
        if channel_code is None:
            keys_to_delete = [key for key in _crop_config_cache if key[0] == site_key]
            for key in keys_to_delete:
                _crop_config_cache.pop(key, None)
            return
        _crop_config_cache.pop(_crop_config_cache_key(site_key, channel_code), None)


def _load_saved_crop_config(
    client: DriveClient,
    site_key: str,
    channel_code: str,
) -> dict[str, Any] | None:
    normalized_channel = _normalize_reolink_channel_code(channel_code or "")
    if not normalized_channel:
        raise ValueError("channel must look like CH-CH03")
    if not _site_uses_manual_crop_configs(site_key):
        return None

    cached = _get_cached_crop_config(site_key, normalized_channel)
    if cached is not _CROP_CONFIG_CACHE_MISS:
        return cached

    folder_ids = _reolink_site_folder_ids(client, site_key)
    config_item = client.find_file_by_name(
        folder_ids["crop_configs"],
        f"{normalized_channel}.json",
    )
    if not config_item or not config_item.get("id"):
        _set_cached_crop_config(site_key, normalized_channel, None)
        return None

    raw_bytes = client.download_file_content(str(config_item["id"]))
    payload = json.loads(raw_bytes.decode("utf-8"))
    _set_cached_crop_config(site_key, normalized_channel, payload)
    return payload


def _save_crop_config(
    client: DriveClient,
    site_key: str,
    channel_code: str,
    payload: dict[str, Any],
) -> None:
    normalized_channel = _normalize_reolink_channel_code(channel_code or "")
    if not normalized_channel:
        raise ValueError("channel must look like CH-CH03")
    if not _site_uses_manual_crop_configs(site_key):
        raise ValueError(f"Manual crop configs are not enabled for {site_key}")

    folder_ids = _reolink_site_folder_ids(client, site_key)
    client.upsert_bytes(
        folder_ids["crop_configs"],
        f"{normalized_channel}.json",
        json.dumps(payload, indent=2).encode("utf-8"),
        mime_type="application/json",
    )
    _set_cached_crop_config(site_key, normalized_channel, payload)


def _resolve_queue_context(
    client: DriveClient,
    source: str,
    site_key: str | None,
) -> QueueContext:
    normalized_source = (source or VIDEO_SOURCE).strip().lower() or VIDEO_SOURCE
    if normalized_source == VIDEO_SOURCE:
        folder_ids = _video_folder_ids(client)
        return QueueContext(
            source=VIDEO_SOURCE,
            site_key=None,
            queue_key=VIDEO_SOURCE,
            display_name="Video",
            input_folder_name="unlabeled",
            input_folder_id=folder_ids["unlabeled"],
            seed_folder_name=None,
            seed_folder_id=None,
            folder_ids=folder_ids,
            persist_frame_metadata=True,
        )

    if normalized_source == REOLINK_SOURCE:
        normalized_site_key = (site_key or "").strip()
        if not normalized_site_key:
            raise ValueError("site is required when source=reolink")
        site = _resolve_site_config(normalized_site_key)
        folder_ids = _reolink_site_folder_ids(client, normalized_site_key)
        return QueueContext(
            source=REOLINK_SOURCE,
            site_key=normalized_site_key,
            queue_key=f"{REOLINK_SOURCE}:{normalized_site_key}",
            display_name=site.display_name,
            input_folder_name="unlabeled",
            input_folder_id=folder_ids["unlabeled"],
            seed_folder_name="unassociated",
            seed_folder_id=folder_ids["unassociated"],
            folder_ids=folder_ids,
            persist_frame_metadata=False,
        )

    raise ValueError(f"Unknown source: {source}")


def _request_source_args() -> tuple[str, str | None]:
    source = request.args.get("source", VIDEO_SOURCE)
    site_key = request.args.get("site")
    return source, site_key


def _payload_source_args(data: dict[str, Any]) -> tuple[str, str | None]:
    source = str(data.get("source", VIDEO_SOURCE) or VIDEO_SOURCE)
    site_key_value = data.get("site_key", data.get("site"))
    site_key = str(site_key_value).strip() if site_key_value else None
    return source, site_key


def _table_config_path() -> Path:
    preferred = Path(__file__).parent / "approved_table_rectangles.json"
    if preferred.exists():
        return preferred

    fallback = Path(__file__).parent / "approved_tables.json"
    if fallback.exists():
        return fallback

    raise RuntimeError("No table configuration JSON found in repo root.")


def _camera_configs_by_number() -> dict[int, dict[str, Any]]:
    global _camera_config_cache

    if _camera_config_cache is not None:
        return _camera_config_cache

    with _camera_config_lock:
        if _camera_config_cache is not None:
            return _camera_config_cache

        from processor import _load_tables_json

        cameras = _load_tables_json(_table_config_path())
        _camera_config_cache = {
            int(camera["camera_number"]): camera
            for camera in cameras
            if camera.get("camera_number") is not None
        }
        return _camera_config_cache


def _extract_reolink_channel_number(folder_name: str) -> int | None:
    channel_code = _extract_reolink_channel_code(folder_name)
    if not channel_code:
        return None
    return int(channel_code.split("CH-CH", 1)[1])


def _split_triplet_suffix(folder_name: str) -> tuple[str, str]:
    match = re.search(r"(_t\d+)$", folder_name, re.IGNORECASE)
    if not match:
        return folder_name, ""
    return folder_name[:match.start()], match.group(1)


def _derived_reolink_folder_name(raw_folder_name: str, table_id: str) -> str:
    base, suffix = _split_triplet_suffix(raw_folder_name)
    if suffix:
        return f"{base}_{table_id}{suffix}"
    return f"{raw_folder_name}_{table_id}"


def _ordered_quadrilateral_points(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if len(points) != 4:
        return points

    center_x = sum(point[0] for point in points) / 4.0
    center_y = sum(point[1] for point in points) / 4.0
    ordered = sorted(points, key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))
    top_left_idx = min(range(4), key=lambda idx: ordered[idx][0] + ordered[idx][1])
    return [ordered[(top_left_idx + offset) % 4] for offset in range(4)]


def _build_table_polygons(camera: dict[str, Any]) -> list[tuple[str, list, tuple[int, int, int, int], list]]:
    from processor import bbox_from_polygon, polygon_from_table, zone_polygon_from_table

    table_polygons: list[tuple[str, list, tuple[int, int, int, int], list]] = []
    tables = camera.get("tables", [])
    for idx, table in enumerate(tables):
        tight_poly = polygon_from_table(table)
        if tight_poly is None:
            continue
        table_id = str(table.get("label") or table.get("mask_id") or idx).replace(" ", "_")
        tight_bbox = bbox_from_polygon(tight_poly)
        zone_poly = zone_polygon_from_table(table) or tight_poly
        table_polygons.append((table_id, tight_poly, tight_bbox, zone_poly))
    return table_polygons


def _build_table_polygons_from_crop_config(
    crop_config: dict[str, Any],
) -> list[tuple[str, list[tuple[float, float]], tuple[int, int, int, int], list[tuple[float, float]]]]:
    from processor import bbox_from_polygon

    table_polygons: list[tuple[str, list[tuple[float, float]], tuple[int, int, int, int], list[tuple[float, float]]]] = []
    for idx, crop in enumerate(crop_config.get("crops", [])):
        polygon = _ordered_quadrilateral_points([
            (float(point[0]), float(point[1]))
            for point in crop.get("polygon", [])
            if isinstance(point, (list, tuple)) and len(point) == 2
        ])
        if len(polygon) != 4:
            continue

        name = str(crop.get("name") or f"table_{idx + 1}").strip() or f"table_{idx + 1}"
        table_id = re.sub(r"\s+", "_", name)
        tight_bbox = bbox_from_polygon(polygon)
        table_polygons.append((table_id, polygon, tight_bbox, polygon))
    return table_polygons


def _manual_crop_camera_payload(
    channel_number: int,
    channel_code: str,
    crop_config: dict[str, Any],
) -> dict[str, Any]:
    reference = crop_config.get("reference") or {}
    frame_width = int(reference.get("width") or 0)
    frame_height = int(reference.get("height") or 0)
    return {
        "camera_number": channel_number,
        "camera_id": channel_code,
        "image_width": frame_width,
        "image_height": frame_height,
        "source": "manual_crop_config",
    }


def _mapped_camera_tables_for_reolink_folder(
    raw_folder_name: str,
    *,
    site_key: str | None = None,
    client: DriveClient | None = None,
) -> tuple[int, dict[str, Any], list[tuple[str, list, tuple[int, int, int, int], list]]] | None:
    channel_code = _extract_reolink_channel_code(raw_folder_name)
    channel_number = _extract_reolink_channel_number(raw_folder_name)
    if channel_number is None or channel_code is None:
        return None

    if _site_uses_manual_crop_configs(site_key):
        if client is None:
            raise RuntimeError("Drive client is required for manual Reolink crop configs.")
        crop_config = _load_saved_crop_config(client, site_key or "", channel_code)
        if crop_config is None:
            return None

        table_polygons = _build_table_polygons_from_crop_config(crop_config)
        if not table_polygons:
            return None
        return channel_number, _manual_crop_camera_payload(channel_number, channel_code, crop_config), table_polygons

    camera = _camera_configs_by_number().get(channel_number)
    if camera is None:
        return None

    table_polygons = _build_table_polygons(camera)
    if not table_polygons:
        return None

    return channel_number, camera, table_polygons


def _existing_generated_folder_names(client: DriveClient, context: QueueContext) -> set[str]:
    names: set[str] = set()
    for folder_name in ("unlabeled", *LABEL_DESTINATIONS):
        for item in client.list_folders(context.folder_ids[folder_name], fields="id,name,mimeType,parents,appProperties"):
            name = item.get("name")
            if name:
                names.add(str(name))
    return names


def _list_reolink_raw_folders(
    client: DriveClient,
    context: QueueContext,
) -> list[dict[str, Any]]:
    if not context.seed_folder_id:
        return []
    return sorted(
        client.list_folders(
            context.seed_folder_id,
            fields="id,name,mimeType,parents,appProperties",
        ),
        key=lambda item: str(item.get("name", "")).lower(),
    )


def _missing_manual_crop_channels(
    client: DriveClient,
    context: QueueContext,
) -> list[str]:
    if not context.site_key or not _site_uses_manual_crop_configs(context.site_key):
        return []

    seen_channels: set[str] = set()
    for raw_folder in _list_reolink_raw_folders(client, context):
        channel_code = _extract_reolink_channel_code(str(raw_folder.get("name", "")))
        if channel_code:
            seen_channels.add(channel_code)

    missing_channels: list[str] = []
    for channel_code in sorted(seen_channels, key=_reolink_channel_sort_key):
        if _load_saved_crop_config(client, context.site_key, channel_code) is None:
            missing_channels.append(channel_code)
    return missing_channels


def _assert_manual_crop_setup_ready(client: DriveClient, context: QueueContext) -> None:
    if not context.site_key or not _site_uses_manual_crop_configs(context.site_key):
        return

    missing_channels = _missing_manual_crop_channels(client, context)
    if missing_channels:
        site = _resolve_site_config(context.site_key)
        raise CropSetupRequiredError(context.site_key, site.display_name, missing_channels)


def _find_reolink_reference_frame(
    client: DriveClient,
    site_key: str,
    channel_code: str,
) -> dict[str, Any] | None:
    site = _resolve_site_config(site_key)
    folder_ids = _reolink_site_folder_ids(client, site_key)
    normalized_channel = _normalize_reolink_channel_code(channel_code or "")
    if not normalized_channel:
        raise ValueError("channel must look like CH-CH03")

    raw_folders = sorted(
        client.list_folders(
            folder_ids["unassociated"],
            fields="id,name,mimeType,parents,appProperties",
        ),
        key=lambda item: str(item.get("name", "")).lower(),
        reverse=True,
    )

    fallback_reference: dict[str, Any] | None = None
    for raw_folder in raw_folders:
        if _extract_reolink_channel_code(str(raw_folder.get("name", ""))) != normalized_channel:
            continue
        source_files = {
            item["name"]: item
            for item in client.list_files(
                raw_folder["id"],
                fields="id,name,mimeType,parents,appProperties",
            )
        }
        frame_item = source_files.get("frame_0.jpg")
        if not frame_item or not frame_item.get("id"):
            continue
        reference_payload = {
            "site_key": site_key,
            "site_label": site.display_name,
            "channel_code": normalized_channel,
            "raw_folder_id": str(raw_folder["id"]),
            "raw_folder_name": str(raw_folder.get("name", "")),
            "frame_file_id": str(frame_item["id"]),
            "preview_url": f"/api/preview/{frame_item['id']}",
            "source": "unassociated",
        }
        if all(source_files.get(f"frame_{idx}.jpg") for idx in range(3)):
            return reference_payload
        if fallback_reference is None:
            fallback_reference = reference_payload

    if fallback_reference is not None:
        return fallback_reference

    saved_config = _load_saved_crop_config(client, site_key, normalized_channel)
    reference = (saved_config or {}).get("reference") or {}
    frame_file_id = reference.get("frame_file_id")
    if frame_file_id:
        return {
            "site_key": site_key,
            "site_label": site.display_name,
            "channel_code": normalized_channel,
            "raw_folder_id": reference.get("raw_folder_id"),
            "raw_folder_name": reference.get("raw_folder_name"),
            "frame_file_id": frame_file_id,
            "preview_url": f"/api/preview/{frame_file_id}",
            "source": "saved_config",
            "width": reference.get("width"),
            "height": reference.get("height"),
        }

    return None


def _get_yolo_model() -> Any:
    global _yolo_model

    if _yolo_model is not None:
        return _yolo_model

    with _yolo_model_lock:
        if _yolo_model is not None:
            return _yolo_model

        from person_detector import load_yolo_model

        _yolo_model = load_yolo_model()
        if _yolo_model is None:
            raise RuntimeError("YOLO model could not be loaded for Reolink preprocessing.")
        return _yolo_model


def _copy_optional_json_file(
    client: DriveClient,
    source_files: dict[str, dict[str, Any]],
    source_name: str,
    dest_folder_id: str,
) -> None:
    item = source_files.get(source_name)
    if not item:
        return

    data = client.download_file_content(item["id"])
    client.upsert_bytes(
        dest_folder_id,
        source_name,
        data,
        mime_type=str(item.get("mimeType") or "application/json"),
    )


def _materialize_reolink_table_crops(
    client: DriveClient,
    context: QueueContext,
    raw_folder: dict[str, str],
    missing_table_polygons: list[tuple[str, list, tuple[int, int, int, int], list]],
) -> list[str]:
    from PIL import Image
    from person_detector import assign_track_ids, build_perception_for_table, detect_people_in_frame
    from processor import perspective_crop_polygon, save_jpeg, _scale_table_polygons

    source_files = {
        item["name"]: item
        for item in client.list_files(
            raw_folder["id"],
            fields="id,name,mimeType,parents,appProperties",
        )
    }
    frame_items = [source_files.get(f"frame_{idx}.jpg") for idx in range(3)]
    if any(item is None for item in frame_items):
        return []

    camera_match = _mapped_camera_tables_for_reolink_folder(
        raw_folder["name"],
        site_key=context.site_key,
        client=client,
    )
    if camera_match is None:
        return []
    _, camera, all_table_polygons = camera_match
    table_polygons_by_id = {table_id: (table_id, tight_poly, tight_bbox, zone_poly) for table_id, tight_poly, tight_bbox, zone_poly in all_table_polygons}
    selected_polygons = [
        table_polygons_by_id[table_id]
        for table_id, *_rest in missing_table_polygons
        if table_id in table_polygons_by_id
    ]
    if not selected_polygons:
        return []

    with tempfile.TemporaryDirectory(prefix="reolink_labeler_") as tmpdir:
        tmp = Path(tmpdir)
        frame_paths: list[Path] = []
        for idx, frame_item in enumerate(frame_items):
            output_path = tmp / f"frame_{idx}.jpg"
            client.download_file_to_path(frame_item["id"], output_path)
            frame_paths.append(output_path)

        with Image.open(frame_paths[0]) as image:
            frame_h, frame_w = image.height, image.width
        img_shape = (frame_h, frame_w)

        ref_w = int(camera.get("image_width") or camera.get("frame_width") or camera.get("width") or frame_w)
        ref_h = int(camera.get("image_height") or camera.get("frame_height") or camera.get("height") or frame_h)
        scaled_polygons = selected_polygons
        if ref_w != frame_w or ref_h != frame_h:
            scaled_polygons = _scale_table_polygons(selected_polygons, ref_w, ref_h, frame_w, frame_h)

        yolo_model = _get_yolo_model()
        frame_detections = [detect_people_in_frame(frame_path, yolo_model) for frame_path in frame_paths]
        assign_track_ids(frame_detections)

        generated_names: list[str] = []
        for table_id, tight_poly, _tight_bbox, zone_poly in scaled_polygons:
            derived_name = _derived_reolink_folder_name(raw_folder["name"], table_id)
            dest_folder_id = client.ensure_subfolder(context.input_folder_id, derived_name)
            uploaded_frame_ids: dict[str, str | None] = {
                "frame_0": None,
                "frame_1": None,
                "frame_2": None,
            }

            for frame_idx, frame_path in enumerate(frame_paths):
                cropped = perspective_crop_polygon(frame_path, zone_poly)
                crop_path = tmp / "crops" / f"{derived_name}_f{frame_idx}.jpg"
                save_jpeg(cropped, crop_path)
                uploaded = client.upload_or_update_file(
                    crop_path,
                    dest_folder_id,
                    file_name=f"frame_{frame_idx}.jpg",
                )
                uploaded_frame_ids[f"frame_{frame_idx}"] = str(uploaded["id"])

            client.update_file_metadata(
                dest_folder_id,
                {"appProperties": build_folder_app_properties(uploaded_frame_ids)},
            )

            perception = build_perception_for_table(frame_detections, tight_poly, img_shape)
            perception_path = tmp / "perception" / f"{derived_name}_perception.json"
            perception_path.parent.mkdir(parents=True, exist_ok=True)
            perception_path.write_text(json.dumps(perception, indent=2), encoding="utf-8")
            client.upload_or_update_file(
                perception_path,
                dest_folder_id,
                file_name="perception.json",
                mime_type="application/json",
            )

            _copy_optional_json_file(client, source_files, "metadata.json", dest_folder_id)
            generated_names.append(derived_name)

        return generated_names


def _prepare_reolink_unlabeled_queue(
    client: DriveClient,
    context: QueueContext,
    target_unlabeled_count: int,
) -> None:
    if context.source != REOLINK_SOURCE or not context.seed_folder_id:
        return

    with _reolink_generation_lock:
        _assert_manual_crop_setup_ready(client, context)
        unlabeled_folders = client.list_folders(
            context.input_folder_id,
            fields="id,name,mimeType,parents,appProperties",
        )
        existing_names = _existing_generated_folder_names(client, context)
        unlabeled_count = len(unlabeled_folders)
        if unlabeled_count >= target_unlabeled_count:
            return
        generated_any = False

        raw_folders = _list_reolink_raw_folders(client, context)

        for raw_folder in raw_folders:
            if unlabeled_count >= target_unlabeled_count:
                break

            mapped = _mapped_camera_tables_for_reolink_folder(
                str(raw_folder.get("name", "")),
                site_key=context.site_key,
                client=client,
            )
            if mapped is None:
                continue

            _channel_number, _camera, table_polygons = mapped
            missing_table_polygons = [
                entry
                for entry in table_polygons
                if _derived_reolink_folder_name(str(raw_folder["name"]), entry[0]) not in existing_names
            ]
            if not missing_table_polygons:
                continue

            generated_names = _materialize_reolink_table_crops(
                client,
                context,
                raw_folder,
                missing_table_polygons,
            )
            for name in generated_names:
                existing_names.add(name)
                unlabeled_count += 1
                generated_any = True

        if generated_any:
            _invalidate_listing_cache(context.queue_key)


def _fetch_source_listing(client: DriveClient, context: QueueContext) -> list[dict[str, str]]:
    return sorted(
        client.list_folders(
            context.input_folder_id,
            fields="id,name,mimeType,parents,appProperties",
        ),
        key=lambda item: str(item.get("name", "")).lower(),
    )


def _set_listing_cache(queue_key: str, listing: list[dict[str, str]]) -> None:
    with _listing_lock:
        _listing_cache[queue_key] = (time.monotonic(), listing)


def _invalidate_listing_cache(queue_key: str) -> None:
    with _listing_lock:
        _listing_cache.pop(queue_key, None)


def _remove_folder_from_listing_cache(queue_key: str, folder_id: str) -> None:
    with _listing_lock:
        cached = _listing_cache.get(queue_key)
        if cached is None:
            return
        cached_at, items = cached
        _listing_cache[queue_key] = (
            cached_at,
            [item for item in items if item.get("id") != folder_id],
        )


def _refresh_listing_in_background(context: QueueContext) -> None:
    try:
        listing = _fetch_source_listing(DriveClient(), context)
        _set_listing_cache(context.queue_key, listing)
    except Exception:
        return
    finally:
        with _listing_lock:
            _listing_refresh_inflight.discard(context.queue_key)


def _schedule_listing_refresh(context: QueueContext) -> bool:
    with _listing_lock:
        if context.queue_key in _listing_refresh_inflight:
            return False
        _listing_refresh_inflight.add(context.queue_key)
    _listing_refresh_executor.submit(_refresh_listing_in_background, context)
    return True


def _list_source_subfolders(
    client: DriveClient,
    context: QueueContext,
    force_refresh: bool = False,
) -> list[dict[str, str]]:
    now = time.monotonic()
    with _listing_lock:
        cached = _listing_cache.get(context.queue_key)
        cached_listing = list(cached[1]) if cached is not None else None
        cached_at = cached[0] if cached is not None else 0.0

    cache_is_fresh = cached_listing is not None and (now - cached_at) < UNLABELED_LIST_CACHE_SECONDS
    if cache_is_fresh and not force_refresh:
        return cached_listing

    if cached_listing is not None and not force_refresh:
        _schedule_listing_refresh(context)
        return cached_listing

    listing = _fetch_source_listing(client, context)
    _set_listing_cache(context.queue_key, listing)
    return list(listing)


def _frame_payload_from_files(files: list[dict]) -> dict[str, str | None]:
    file_map = {f["name"]: f["id"] for f in files}
    return {
        "frame_0": file_map.get("frame_0.jpg"),
        "frame_1": file_map.get("frame_1.jpg"),
        "frame_2": file_map.get("frame_2.jpg"),
    }


def _frame_payload_from_folder(folder: dict[str, object]) -> dict[str, str | None]:
    return extract_frame_ids_from_item(folder)


def _build_folder_payload(
    folder: dict[str, str],
    context: QueueContext,
    frames: dict[str, str | None],
) -> dict:
    preview_urls = {
        key: f"/api/preview/{file_id}"
        for key, file_id in frames.items()
        if file_id
    }
    return {
        "folder_id": folder["id"],
        "folder_name": folder["name"],
        "parent_id": context.input_folder_id,
        "source": context.source,
        "site_key": context.site_key,
        "queue_key": context.queue_key,
        "frames": frames,
        "preview_urls": preview_urls,
        "cache_ready": _frames_cache_ready(frames),
    }


def _persist_folder_frame_metadata(
    client: DriveClient,
    folder: dict[str, str],
    frames: dict[str, str | None],
) -> None:
    if not has_complete_frame_ids(frames):
        return
    try:
        metadata = dict(folder.get("appProperties") or {})
        metadata.update(build_folder_app_properties(frames))
        client.update_file_metadata(
            folder["id"],
            {"appProperties": metadata},
            fields="id,appProperties",
        )
        folder["appProperties"] = metadata
    except DriveClientError:
        return


def _cache_path_for_file(file_id: str) -> Path:
    return _ensure_cache_dir() / f"{file_id}.jpg"


def _frames_cache_ready(frames: dict[str, str | None]) -> bool:
    for key in ("frame_0", "frame_1", "frame_2"):
        file_id = frames.get(key)
        if not file_id or not _cache_path_for_file(file_id).exists():
            return False
    return True


def _folder_cache_ready(folder: dict) -> bool:
    return _frames_cache_ready(folder.get("frames", {}))


def _hydrate_folder(client: DriveClient, context: QueueContext, folder: dict[str, str]) -> dict | None:
    frames = _frame_payload_from_folder(folder)
    if has_complete_frame_ids(frames):
        return _build_folder_payload(folder, context, frames)

    frames = _frame_payload_from_files(client.list_files(folder["id"]))
    if not has_complete_frame_ids(frames):
        return None

    if context.persist_frame_metadata:
        _persist_folder_frame_metadata(client, folder, frames)
    return _build_folder_payload(folder, context, frames)


def _hydrated_cache_key(queue_key: str, folder_id: str) -> tuple[str, str]:
    return (queue_key, folder_id)


def _get_cached_hydrated_folder(queue_key: str, folder_id: str) -> dict | None | object:
    now = time.monotonic()
    with _hydrated_folder_cache_lock:
        cached = _hydrated_folder_cache.get(_hydrated_cache_key(queue_key, folder_id))
        if not cached:
            return _MISSING
        cached_at, payload = cached
        if (now - cached_at) > HYDRATED_FOLDER_CACHE_TTL_SECONDS:
            _hydrated_folder_cache.pop(_hydrated_cache_key(queue_key, folder_id), None)
            return _MISSING
        return payload


def _set_cached_hydrated_folder(queue_key: str, folder_id: str, payload: dict | None) -> None:
    with _hydrated_folder_cache_lock:
        _hydrated_folder_cache[_hydrated_cache_key(queue_key, folder_id)] = (
            time.monotonic(),
            payload,
        )


_MISSING = object()


def _warm_file(file_id: str) -> None:
    try:
        cache_path = _cache_path_for_file(file_id)
        if cache_path.exists():
            try:
                os.utime(cache_path, None)
            except OSError:
                pass
            return

        DriveClient().download_file_to_path(file_id, cache_path)
    except Exception:
        return
    finally:
        with _preview_prewarm_lock:
            _preview_prewarm_inflight.discard(file_id)


def _warm_folder_payload(context: QueueContext, folder: dict[str, str]) -> None:
    try:
        payload = _hydrate_folder_with_fresh_client(context, folder)
        _set_cached_hydrated_folder(context.queue_key, folder["id"], payload)
        if payload is not None and not _folder_cache_ready(payload):
            _schedule_preview_prewarm([payload])
    except Exception:
        return
    finally:
        with _folder_prewarm_lock:
            _folder_prewarm_inflight.discard((context.queue_key, folder["id"]))


def _schedule_preview_prewarm(hydrated_folders: list[dict]) -> int:
    if not hydrated_folders:
        return 0

    warm_targets = hydrated_folders[:PREWARM_FOLDER_COUNT]
    scheduled = 0
    for folder in warm_targets:
        frames = folder.get("frames", {})
        for key in ("frame_0", "frame_1", "frame_2"):
            file_id = frames.get(key)
            if not file_id:
                continue
            if _cache_path_for_file(file_id).exists():
                continue
            with _preview_prewarm_lock:
                if file_id in _preview_prewarm_inflight:
                    continue
                _preview_prewarm_inflight.add(file_id)
            _preview_prewarm_executor.submit(_warm_file, file_id)
            scheduled += 1
    return scheduled


def _hydrate_folder_with_fresh_client(
    context: QueueContext,
    folder: dict[str, str],
) -> dict | None:
    # googleapiclient service objects are safer to keep thread-local.
    client = DriveClient()
    return _hydrate_folder(client, context, folder)


def _hydrate_folders_parallel(
    folders: list[dict[str, str]],
    context: QueueContext,
) -> tuple[list[dict | None], dict[str, int]]:
    if not folders:
        return [], {
            "requested": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "workers": 0,
        }

    results: list[dict | None | object] = [_MISSING] * len(folders)
    uncached: list[tuple[int, dict[str, str]]] = []

    for idx, folder in enumerate(folders):
        cached = _get_cached_hydrated_folder(context.queue_key, folder["id"])
        if cached is _MISSING:
            uncached.append((idx, folder))
        else:
            if cached is not None:
                cached["cache_ready"] = _folder_cache_ready(cached)
            results[idx] = cached

    hydrate_stats = {
        "requested": len(folders),
        "cache_hits": len(folders) - len(uncached),
        "cache_misses": len(uncached),
        "workers": 0,
    }

    if uncached:
        if len(uncached) == 1:
            idx, folder = uncached[0]
            payload = _hydrate_folder(get_client(), context, folder)
            if payload is not None:
                payload["cache_ready"] = _folder_cache_ready(payload)
            _set_cached_hydrated_folder(context.queue_key, folder["id"], payload)
            results[idx] = payload
        else:
            max_workers = min(HYDRATE_MAX_WORKERS, len(uncached))
            hydrate_stats["workers"] = max_workers
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                payloads = list(
                    executor.map(
                        lambda item: _hydrate_folder_with_fresh_client(context, item[1]),
                        uncached,
                    )
                )
            for (idx, folder), payload in zip(uncached, payloads):
                if payload is not None:
                    payload["cache_ready"] = _folder_cache_ready(payload)
                _set_cached_hydrated_folder(context.queue_key, folder["id"], payload)
                results[idx] = payload

    return [None if payload is _MISSING else payload for payload in results], hydrate_stats


def _schedule_folder_hydration_prewarm(
    subfolders: list[dict[str, str]],
    start_idx: int,
    context: QueueContext,
) -> int:
    scheduled = 0
    end_idx = min(len(subfolders), start_idx + PREWARM_FOLDER_COUNT)
    for folder in subfolders[start_idx:end_idx]:
        if _get_cached_hydrated_folder(context.queue_key, folder["id"]) is not _MISSING:
            continue
        with _folder_prewarm_lock:
            inflight_key = (context.queue_key, folder["id"])
            if inflight_key in _folder_prewarm_inflight:
                continue
            _folder_prewarm_inflight.add(inflight_key)
        _folder_prewarm_executor.submit(_warm_folder_payload, context, folder)
        scheduled += 1
    return scheduled


def _compute_stats(client: DriveClient, context: QueueContext) -> dict[str, int]:
    stats: dict[str, int] = {
        "unlabeled": len(client.list_folders(context.input_folder_id)),
    }
    for name in LABEL_DESTINATIONS:
        stats[name] = len(client.list_folders(context.folder_ids[name]))
    return stats


def _collect_ready_folders(
    subfolders: list[dict[str, str]],
    context: QueueContext,
    limit: int,
) -> tuple[list[dict], dict[str, int | float]]:
    ready: list[dict] = []
    nonready = 0
    hydrated_valid = 0
    scanned = 0
    hydrate_ms = 0.0
    hydrate_requested = 0
    hydrate_cache_hits = 0
    hydrate_cache_misses = 0
    hydrate_worker_max = 0
    first_unready_idx = len(subfolders)

    target_scan = min(len(subfolders), max(limit * READY_SCAN_MULTIPLIER, PREWARM_FOLDER_COUNT))
    target_scan = min(target_scan, READY_SCAN_MAX)

    while scanned < target_scan and len(ready) < limit:
        remaining_scan = target_scan - scanned
        batch_span = min(
            remaining_scan,
            max(HYDRATE_MAX_WORKERS, (limit - len(ready)) * 2),
        )
        folder_batch = subfolders[scanned:scanned + batch_span]
        if not folder_batch:
            break

        hydrate_started = time.perf_counter()
        payloads, hydrate_stats = _hydrate_folders_parallel(folder_batch, context)
        hydrate_ms += (time.perf_counter() - hydrate_started) * 1000
        hydrate_requested += hydrate_stats["requested"]
        hydrate_cache_hits += hydrate_stats["cache_hits"]
        hydrate_cache_misses += hydrate_stats["cache_misses"]
        hydrate_worker_max = max(hydrate_worker_max, hydrate_stats["workers"])

        for offset, payload in enumerate(payloads):
            absolute_idx = scanned + offset
            if payload is None:
                continue
            hydrated_valid += 1
            payload["cache_ready"] = _folder_cache_ready(payload)
            if payload["cache_ready"]:
                ready.append(payload)
            else:
                nonready += 1
                first_unready_idx = min(first_unready_idx, absolute_idx)
                _schedule_preview_prewarm([payload])
            if len(ready) >= limit:
                break

        scanned += len(folder_batch)

    prewarm_scan_start = first_unready_idx if first_unready_idx < len(subfolders) else scanned
    folder_prewarm_scheduled = _schedule_folder_hydration_prewarm(
        subfolders,
        prewarm_scan_start,
        context,
    )

    return ready, {
        "scanned": scanned,
        "hydrate_ms": hydrate_ms,
        "hydrate_requested": hydrate_requested,
        "hydrate_cache_hits": hydrate_cache_hits,
        "hydrate_cache_misses": hydrate_cache_misses,
        "hydrate_worker_max": hydrate_worker_max,
        "folder_prewarm_scheduled": folder_prewarm_scheduled,
        "prewarm_scan_start": prewarm_scan_start,
        "hydrated_valid": hydrated_valid,
        "nonready": nonready,
    }


def _cleanup_cache_if_needed(force: bool = False) -> None:
    global _last_cache_cleanup_monotonic

    now_monotonic = time.monotonic()
    if not force and (now_monotonic - _last_cache_cleanup_monotonic) < CACHE_CLEANUP_INTERVAL_SECONDS:
        return

    with _cache_cleanup_lock:
        now_monotonic = time.monotonic()
        if not force and (now_monotonic - _last_cache_cleanup_monotonic) < CACHE_CLEANUP_INTERVAL_SECONDS:
            return

        cache_dir = _ensure_cache_dir()
        now_epoch = time.time()
        ttl_seconds = CACHE_TTL_HOURS * 3600
        max_bytes = CACHE_MAX_MB * 1024 * 1024

        files = [path for path in cache_dir.iterdir() if path.is_file()]
        for path in files:
            try:
                age_seconds = now_epoch - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age_seconds > ttl_seconds:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

        file_stats: list[tuple[float, int, Path]] = []
        total_bytes = 0
        for path in cache_dir.iterdir():
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            total_bytes += stat.st_size
            file_stats.append((stat.st_mtime, stat.st_size, path))

        if total_bytes > max_bytes:
            for _, size, path in sorted(file_stats, key=lambda item: item[0]):
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                total_bytes -= size
                if total_bytes <= max_bytes:
                    break

        _last_cache_cleanup_monotonic = now_monotonic


def _manual_crop_status_payload(client: DriveClient, site_key: str) -> dict[str, Any]:
    site = _resolve_site_config(site_key)
    if not site.manual_crop_configs:
        raise ValueError(f"Manual crop configs are not enabled for {site_key}")

    context = _resolve_queue_context(client, REOLINK_SOURCE, site_key)
    channels = sorted(
        {
            channel_code
            for raw_folder in _list_reolink_raw_folders(client, context)
            if (channel_code := _extract_reolink_channel_code(str(raw_folder.get("name", ""))))
        },
        key=_reolink_channel_sort_key,
    )

    channel_payloads: list[dict[str, Any]] = []
    for channel_code in channels:
        config = _load_saved_crop_config(client, site_key, channel_code)
        reference = _find_reolink_reference_frame(client, site_key, channel_code)
        channel_payloads.append(
            {
                "channel_code": channel_code,
                "has_config": config is not None,
                "crop_count": len((config or {}).get("crops", [])),
                "reference_available": reference is not None,
                "setup_url": _crop_editor_url(site_key, channel_code),
            }
        )

    return {
        "site_key": site_key,
        "site_label": site.display_name,
        "channels": channel_payloads,
    }


def _validate_crop_config_payload(data: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    site_key_value = data.get("site_key", data.get("site"))
    site_key = str(site_key_value or "").strip()
    if not site_key:
        raise ValueError("site_key is required")

    site = _resolve_site_config(site_key)
    if not site.manual_crop_configs:
        raise ValueError(f"Manual crop configs are not enabled for {site_key}")

    channel_value = str(data.get("channel_code", data.get("channel", "")) or "").strip()
    channel_code = _normalize_reolink_channel_code(channel_value)
    if not channel_code:
        raise ValueError("channel must look like CH-CH03")

    reference = data.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("reference is required")

    frame_file_id = str(reference.get("frame_file_id") or "").strip()
    raw_folder_name = str(reference.get("raw_folder_name") or "").strip()
    width = int(reference.get("width") or 0)
    height = int(reference.get("height") or 0)
    if not frame_file_id:
        raise ValueError("reference.frame_file_id is required")
    if width <= 0 or height <= 0:
        raise ValueError("reference width and height must be positive")

    raw_crops = data.get("crops")
    if not isinstance(raw_crops, list) or not raw_crops:
        raise ValueError("At least one crop is required")

    normalized_crops: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for idx, crop in enumerate(raw_crops):
        if not isinstance(crop, dict):
            raise ValueError("Each crop must be an object")

        crop_name = str(crop.get("name") or f"table_{idx + 1}").strip() or f"table_{idx + 1}"
        normalized_name = re.sub(r"\s+", "_", crop_name)
        if normalized_name in seen_names:
            raise ValueError(f"Duplicate crop name: {crop_name}")
        seen_names.add(normalized_name)

        polygon = crop.get("polygon")
        if not isinstance(polygon, list) or len(polygon) != 4:
            raise ValueError(f"{crop_name} must have exactly 4 points")

        normalized_points: list[list[float]] = []
        ordered_polygon = _ordered_quadrilateral_points([
            (float(point[0]), float(point[1]))
            for point in polygon
        ])
        for point in ordered_polygon:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError(f"{crop_name} has an invalid point")
            x = float(point[0])
            y = float(point[1])
            normalized_points.append([round(x, 2), round(y, 2)])

        normalized_crops.append(
            {
                "name": crop_name,
                "polygon": normalized_points,
            }
        )

    payload = {
        "version": 1,
        "site_key": site_key,
        "site_label": site.display_name,
        "channel_code": channel_code,
        "channel_number": _extract_reolink_channel_number(channel_code),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "reference": {
            "raw_folder_id": str(reference.get("raw_folder_id") or "").strip() or None,
            "raw_folder_name": raw_folder_name or None,
            "frame_file_id": frame_file_id,
            "width": width,
            "height": height,
        },
        "crops": normalized_crops,
    }
    return site_key, channel_code, payload


@app.route("/")
def index():
    return render_template("label.html")


@app.route("/crop-editor")
def crop_editor():
    site_key = request.args.get("site", MATTHEWS_SITE_KEY)
    channel_code = request.args.get("channel", "").strip() or None
    return render_template(
        "crop_editor.html",
        default_site_key=site_key,
        default_channel_code=channel_code,
    )


@app.route("/api/sources")
def api_sources():
    reolink_sites = [
        {
            "site_key": site.site_key,
            "label": site.display_name,
            "manual_crop_configs": site.manual_crop_configs,
            "crop_editor_url": _crop_editor_url(site.site_key) if site.manual_crop_configs else None,
        }
        for site in REOLINK_SITES
    ]
    return jsonify(
        {
            "sources": [
                {"source": VIDEO_SOURCE, "label": "Video"},
                {"source": REOLINK_SOURCE, "label": "Reolink"},
            ],
            "reolink_sites": reolink_sites,
            "default_source": {
                "source": VIDEO_SOURCE,
                "site_key": _default_reolink_site_key(),
            },
        }
    )


@app.route("/api/reolink/crop-configs/status")
def api_reolink_crop_config_status():
    try:
        client = get_client()
        site_key = request.args.get("site", MATTHEWS_SITE_KEY)
        return jsonify(_manual_crop_status_payload(client, site_key))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (DriveClientError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reolink/crop-config")
def api_reolink_crop_config():
    try:
        client = get_client()
        site_key = request.args.get("site", MATTHEWS_SITE_KEY)
        channel_value = request.args.get("channel", "")
        channel_code = _normalize_reolink_channel_code(channel_value)
        if not channel_code:
            return jsonify({"error": "channel must look like CH-CH03"}), 400

        site = _resolve_site_config(site_key)
        if not site.manual_crop_configs:
            return jsonify({"error": f"Manual crop configs are not enabled for {site_key}"}), 400

        config = _load_saved_crop_config(client, site_key, channel_code)
        reference = _find_reolink_reference_frame(client, site_key, channel_code)
        if reference is None:
            return jsonify({"error": f"No reference frame found for {channel_code} in {site.display_name}."}), 404

        return jsonify(
            {
                "site_key": site_key,
                "site_label": site.display_name,
                "channel_code": channel_code,
                "has_config": config is not None,
                "config": config,
                "reference": reference,
                "setup_url": _crop_editor_url(site_key, channel_code),
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (DriveClientError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reolink/crop-config", methods=["POST"])
def api_save_reolink_crop_config():
    try:
        client = get_client()
        data = request.get_json(force=True)
        site_key, channel_code, payload = _validate_crop_config_payload(data)
        _save_crop_config(client, site_key, channel_code, payload)
        return jsonify(
            {
                "ok": True,
                "site_key": site_key,
                "channel_code": channel_code,
                "config": payload,
                "setup_url": _crop_editor_url(site_key, channel_code),
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (DriveClientError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/folders")
def api_folders():
    """Return list of unlabeled subfolders (names + IDs only, no file listing)."""
    try:
        client = get_client()
        source, site_key = _request_source_args()
        context = _resolve_queue_context(client, source, site_key)
        _prepare_reolink_unlabeled_queue(client, context, target_unlabeled_count=PREWARM_FOLDER_COUNT)
        subfolders = _list_source_subfolders(
            client,
            context,
            force_refresh=request.args.get("refresh", "0") == "1",
        )
        result = [
            {
                "folder_id": f["id"],
                "folder_name": f["name"],
                "parent_id": context.input_folder_id,
                "source": context.source,
                "site_key": context.site_key,
                "queue_key": context.queue_key,
            }
            for f in subfolders
        ]
        return jsonify({"folders": result, "source_context": context.to_payload()})
    except CropSetupRequiredError as e:
        return jsonify(e.to_payload()), 409
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (DriveClientError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/queue")
def api_queue():
    """Return a ready-to-render batch of unlabeled folders whose previews are locally cached."""
    request_started = time.perf_counter()
    try:
        client = get_client()
        source, site_key = _request_source_args()
        context = _resolve_queue_context(client, source, site_key)

        limit = int(request.args.get("limit", str(QUEUE_BATCH_DEFAULT)) or str(QUEUE_BATCH_DEFAULT))
        limit = max(1, min(limit, QUEUE_BATCH_MAX))
        include_stats = request.args.get("include_stats", "0") == "1"
        force_refresh = request.args.get("refresh", "0") == "1"
        target_unlabeled_count = min(
            READY_SCAN_MAX,
            max(limit * READY_SCAN_MULTIPLIER, PREWARM_FOLDER_COUNT),
        )
        _prepare_reolink_unlabeled_queue(client, context, target_unlabeled_count=target_unlabeled_count)

        list_started = time.perf_counter()
        subfolders = _list_source_subfolders(client, context, force_refresh=force_refresh)
        list_ms = (time.perf_counter() - list_started) * 1000
        total_unlabeled = len(subfolders)

        ready_folders, ready_stats = _collect_ready_folders(subfolders, context, limit)

        preview_prewarm_scheduled = _schedule_preview_prewarm(ready_folders)
        ready_buffer_count = len(ready_folders)
        warming_count = int(ready_stats["nonready"])

        response: dict[str, object] = {
            "folders": ready_folders,
            "next_cursor": 0,
            "source_context": context.to_payload(),
            "total_unlabeled": total_unlabeled,
            "has_more": total_unlabeled > ready_buffer_count,
            "ready_buffer_count": ready_buffer_count,
            "warming_count": warming_count,
            "retry_ms": QUEUE_RETRY_MS if total_unlabeled > 0 and ready_buffer_count < limit else 0,
        }
        if include_stats:
            stats_started = time.perf_counter()
            response["stats"] = _compute_stats(client, context)
            stats_ms = (time.perf_counter() - stats_started) * 1000
        else:
            stats_ms = 0.0

        total_ms = (time.perf_counter() - request_started) * 1000
        _log_timing(
            "api_queue",
            total_ms=f"{total_ms:.1f}",
            list_ms=f"{list_ms:.1f}",
            hydrate_ms=f"{ready_stats['hydrate_ms']:.1f}",
            stats_ms=f"{stats_ms:.1f}",
            limit=limit,
            returned=len(ready_folders),
            ready_buffer=ready_buffer_count,
            warming=warming_count,
            scanned=ready_stats["scanned"],
            hydrated_valid=ready_stats["hydrated_valid"],
            cache_hits=ready_stats["hydrate_cache_hits"],
            cache_misses=ready_stats["hydrate_cache_misses"],
            workers=ready_stats["hydrate_worker_max"],
            prewarm_folders=ready_stats["folder_prewarm_scheduled"],
            prewarm_files=preview_prewarm_scheduled,
            total_unlabeled=total_unlabeled,
            include_stats=int(include_stats),
            refresh=int(force_refresh),
            queue=context.queue_key,
        )
        return jsonify(response)
    except CropSetupRequiredError as e:
        return jsonify(e.to_payload()), 409
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (DriveClientError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/folder/<folder_id>/frames")
def api_folder_frames(folder_id: str):
    """Return frame file IDs for a single subfolder."""
    try:
        client = get_client()
        folder = client.get_file(folder_id, fields="id,name,parents,appProperties")
        frames = _frame_payload_from_folder(folder)
        if not has_complete_frame_ids(frames):
            frames = _frame_payload_from_files(client.list_files(folder_id))
        return jsonify(frames)
    except DriveClientError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/preview/<file_id>")
def api_preview(file_id: str):
    """Serve a Drive image file, caching it locally."""
    request_started = time.perf_counter()
    _cleanup_cache_if_needed()

    cache_path = _cache_path_for_file(file_id)
    cache_hit = cache_path.exists()
    download_ms = 0.0
    if not cache_path.exists():
        try:
            client = get_client()
            download_started = time.perf_counter()
            client.download_file_to_path(file_id, cache_path)
            download_ms = (time.perf_counter() - download_started) * 1000
        except DriveClientError as e:
            abort(404, description=str(e))

    try:
        os.utime(cache_path, None)
    except OSError:
        pass

    try:
        size_bytes = cache_path.stat().st_size
    except OSError:
        size_bytes = 0

    total_ms = (time.perf_counter() - request_started) * 1000
    _log_timing(
        "api_preview",
        total_ms=f"{total_ms:.1f}",
        download_ms=f"{download_ms:.1f}",
        cache="hit" if cache_hit else "miss",
        size_kb=f"{size_bytes / 1024:.1f}",
        file_id=file_id,
    )
    return send_file(cache_path, mimetype="image/jpeg", conditional=True, max_age=3600)


@app.route("/api/label", methods=["POST"])
def api_label():
    """Move a folder from unlabeled/ to a label destination."""
    request_started = time.perf_counter()
    data = request.get_json(force=True)
    folder_id = data.get("folder_id", "").strip()
    parent_id = data.get("parent_id", "").strip()
    label = data.get("label", "").strip().lower()
    source, site_key = _payload_source_args(data)

    if not folder_id or not parent_id:
        return jsonify({"error": "folder_id and parent_id required"}), 400
    if label not in LABEL_DESTINATIONS:
        return jsonify({"error": "label must be clean, dirty, occupied, or label_later"}), 400

    try:
        client = get_client()
        context = _resolve_queue_context(client, source, site_key)
        if parent_id != context.input_folder_id:
            return jsonify({"error": "parent_id does not match the active queue"}), 400

        dest_id = context.folder_ids[label]
        move_started = time.perf_counter()
        client.move_file(folder_id, new_parent_id=dest_id, remove_parent_id=parent_id)
        move_ms = (time.perf_counter() - move_started) * 1000
        with _hydrated_folder_cache_lock:
            _hydrated_folder_cache.pop(_hydrated_cache_key(context.queue_key, folder_id), None)
        _remove_folder_from_listing_cache(context.queue_key, folder_id)
        total_ms = (time.perf_counter() - request_started) * 1000
        _log_timing(
            "api_label",
            total_ms=f"{total_ms:.1f}",
            move_ms=f"{move_ms:.1f}",
            label=label,
            folder_id=folder_id,
            queue=context.queue_key,
        )
        return jsonify({"ok": True, "moved_to": label, "source_context": context.to_payload()})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (DriveClientError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    """Return counts of folders in each category."""
    request_started = time.perf_counter()
    try:
        client = get_client()
        source, site_key = _request_source_args()
        context = _resolve_queue_context(client, source, site_key)
        _prepare_reolink_unlabeled_queue(client, context, target_unlabeled_count=PREWARM_FOLDER_COUNT)
        stats = _compute_stats(client, context)
        total_ms = (time.perf_counter() - request_started) * 1000
        _log_timing("api_stats", total_ms=f"{total_ms:.1f}", queue=context.queue_key, **stats)
        return jsonify({**stats, "source_context": context.to_payload()})
    except CropSetupRequiredError as e:
        return jsonify(e.to_payload()), 409
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (DriveClientError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500


def run_label_ui(port: int = 8080) -> None:
    print(f"Starting label UI at http://localhost:{port}")
    _cleanup_cache_if_needed(force=True)
    print(f"Preview cache: {CACHE_DIR}")
    print(
        "Timing logs: "
        f"{'on' if TIMING_LOGS_ENABLED else 'off'}"
        f" (min {TIMING_LOG_MIN_MS:.0f} ms)"
    )
    # Request-scoped Drive clients and locked shared caches make threaded
    # serving practical here, which helps keep the warm queue filled.
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)
