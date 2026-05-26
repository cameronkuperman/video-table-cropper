"""
--label mode: Flask UI that reads unlabeled/ subfolders from Drive,
shows N images per folder, and moves the folder on Drive when labeled.

Group size N is detected per-folder at runtime (typically 10 today, 3 for
legacy data). The wire/Drive protocol still uses "triplet" naming for
backwards compatibility.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
import re
import sys
import hmac
import secrets
import tempfile
import time
import zipfile
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterator
from urllib.parse import unquote, urlencode, urlparse

from flask import (
    Flask,
    abort,
    g,
    has_request_context,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.exceptions import HTTPException

from drive_client import DriveClient, DriveClientError, FOLDER_MIME
from env_loader import load_local_env
from queue_metadata import (
    LEGACY_FRAMES_PER_GROUP,
    MAX_FRAMES_PER_GROUP,
    SPARSE_SAMPLE_FRAME_INDICES,
    build_folder_app_properties,
    extract_frame_ids_from_item,
    frame_slot_keys,
    has_complete_frame_ids,
)

_FRAME_FILENAME_RE = re.compile(r"^frame_(\d+)\.jpg$")


def _detect_n_from_file_names(names: Any) -> int:
    """Count `frame_*.jpg` files (any iterable of names or dict of name→item)
    and return how many distinct frame indices are present, treated as N.

    Returns 0 when no frame files are found.
    """
    if isinstance(names, dict):
        iterable = names.keys()
    else:
        iterable = names
    indices: set[int] = set()
    for name in iterable:
        m = _FRAME_FILENAME_RE.match(str(name))
        if m:
            indices.add(int(m.group(1)))
    return len(indices)


def _detect_n_from_frame_dict(frames: dict[str, Any]) -> int:
    """Number of frame_N keys present in a frames dict, used to recover N
    when reading from Drive metadata that was written with the new schema."""
    indices: set[int] = set()
    for key in frames.keys():
        m = re.match(r"^frame_(\d+)$", str(key))
        if m:
            indices.add(int(m.group(1)))
    return len(indices)


def _frame_index_from_key(key: Any) -> int | None:
    m = re.match(r"^frame_(\d+)$", str(key))
    return int(m.group(1)) if m else None


def _frame_keys_from_client_payload(raw_frames: dict[str, Any]) -> list[str]:
    indices = sorted({
        idx
        for key in raw_frames.keys()
        for idx in [_frame_index_from_key(key)]
        if idx is not None
    })
    if not indices:
        return list(frame_slot_keys(LEGACY_FRAMES_PER_GROUP))
    if indices == [0, 5, 9]:
        return [f"frame_{idx}" for idx in indices]
    if indices == list(range(indices[-1] + 1)):
        return [f"frame_{idx}" for idx in indices]
    return []


def _frames_from_client_payload(raw_frames: dict[str, Any]) -> dict[str, str | None]:
    keys = _frame_keys_from_client_payload(raw_frames)
    return {
        key: str(raw_frames.get(key) or "") or None
        for key in keys
    }


def _frame_ids_belong_to_files(frames: dict[str, str | None], files: list[dict[str, Any]]) -> bool:
    file_ids = {str(file.get("id") or "") for file in files}
    frame_ids = [str(file_id) for file_id in frames.values() if file_id]
    return bool(frame_ids) and all(file_id in file_ids for file_id in frame_ids)


def _ordered_frame_keys(frames: dict[str, Any]) -> list[str]:
    """Return present frame_N keys sorted by N. Use this everywhere that
    iterates a frames dict so the order is deterministic regardless of N."""
    keyed: list[tuple[int, str]] = []
    for key in frames.keys():
        m = re.match(r"^frame_(\d+)$", str(key))
        if m:
            keyed.append((int(m.group(1)), str(key)))
    keyed.sort()
    return [k for _, k in keyed]

load_local_env()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "autolabeler-dev-secret-change-me")


def _read_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or str(default))
    except (TypeError, ValueError):
        return default


def _ready_target_or_legacy_env(name: str, default: int) -> int:
    if os.environ.get("LABEL_READY_TARGET", "").strip():
        return _read_int_env("LABEL_READY_TARGET", default)
    return _read_int_env(name, default)


def _label_ready_target_configured() -> bool:
    return bool(os.environ.get("LABEL_READY_TARGET", "").strip())


LABEL_THROUGHPUT_TARGET_IMAGES = max(
    0,
    _read_int_env("LABEL_THROUGHPUT_TARGET_IMAGES", 4000),
)
LABEL_IMAGES_PER_FOLDER_ESTIMATE = max(
    1,
    _read_int_env("LABEL_IMAGES_PER_FOLDER_ESTIMATE", 3),
)
LABEL_THROUGHPUT_TARGET_FOLDERS = (
    math.ceil(LABEL_THROUGHPUT_TARGET_IMAGES / LABEL_IMAGES_PER_FOLDER_ESTIMATE)
    if LABEL_THROUGHPUT_TARGET_IMAGES
    else 0
)
QUEUE_BATCH_DEFAULT = max(72, int(os.environ.get("LABEL_QUEUE_BATCH_DEFAULT", "180") or "180"))
QUEUE_BATCH_MAX = max(
    QUEUE_BATCH_DEFAULT,
    int(
        os.environ.get(
            "LABEL_QUEUE_BATCH_MAX",
            str(max(600, min(2000, LABEL_THROUGHPUT_TARGET_FOLDERS or 600))),
        )
        or "600"
    ),
)
CACHE_CLEANUP_INTERVAL_SECONDS = 300
INTERACTIVE_PREWARM_FOLDER_CAP = max(
    12,
    int(
        os.environ.get(
            "LABEL_INTERACTIVE_PREWARM_FOLDER_CAP",
            str(max(400, min(2000, LABEL_THROUGHPUT_TARGET_FOLDERS or 400))),
        )
        or "400"
    ),
)
INTERACTIVE_READY_SCAN_CAP = max(
    100,
    int(
        os.environ.get(
            "LABEL_INTERACTIVE_READY_SCAN_CAP",
            str(max(1000, min(3000, (LABEL_THROUGHPUT_TARGET_FOLDERS or 500) * 2))),
        )
        or "1000"
    ),
)
UNLABELED_LIST_CACHE_SECONDS = max(
    15, int(os.environ.get("LABEL_UNLABELED_CACHE_SECONDS", "300") or "300")
)
HYDRATE_MAX_WORKERS = max(2, int(os.environ.get("LABEL_QUEUE_HYDRATE_WORKERS", "20") or "20"))
PREVIEW_PREWARM_MAX_WORKERS = max(
    2, int(os.environ.get("LABEL_PREVIEW_PREWARM_WORKERS", "32") or "32")
)
THUMB_WIDTH = max(128, int(os.environ.get("LABEL_THUMB_WIDTH", "512") or "512"))
THUMB_QUALITY = max(40, min(95, int(os.environ.get("LABEL_THUMB_QUALITY", "82") or "82")))
FOLDER_PREWARM_MAX_WORKERS = max(
    2, min(10, int(os.environ.get("LABEL_FOLDER_PREWARM_WORKERS", "8") or "8"))
)
REOLINK_FRAME_DOWNLOAD_WORKERS = max(
    2, min(10, int(os.environ.get("REOLINK_FRAME_DOWNLOAD_WORKERS", "8") or "8"))
)
REOLINK_TABLE_MATERIALIZE_WORKERS = max(
    1, min(10, int(os.environ.get("REOLINK_TABLE_MATERIALIZE_WORKERS", "8") or "8"))
)
REOLINK_TRUE_TEN_BATCH_SIZE = max(
    1,
    int(os.environ.get("REOLINK_TRUE_TEN_BATCH_SIZE", "4") or "4"),
)
REOLINK_YOLO_BATCH_FRAMES = max(
    1,
    int(os.environ.get("REOLINK_YOLO_BATCH_FRAMES", "40") or "40"),
)
REOLINK_PREPROCESS_MAX_SECONDS = max(
    0.0,
    float(os.environ.get("REOLINK_PREPROCESS_MAX_SECONDS", "0") or "0"),
)
LABEL_READY_TARGET_CONFIGURED = _label_ready_target_configured()
LABEL_READY_TARGET = (
    max(1, _read_int_env("LABEL_READY_TARGET", 1000))
    if LABEL_READY_TARGET_CONFIGURED
    else None
)
PREWARM_FOLDER_COUNT = min(
    INTERACTIVE_PREWARM_FOLDER_CAP,
    max(
        12,
        _ready_target_or_legacy_env("LABEL_PREWARM_FOLDER_COUNT", 400),
    ),
)
REOLINK_PREWARM_TARGET = max(
    PREWARM_FOLDER_COUNT,
    LABEL_THROUGHPUT_TARGET_FOLDERS,
    _ready_target_or_legacy_env("LABEL_REOLINK_PREWARM_TARGET", 1000),
)
INTERACTIVE_REOLINK_PREWARM_TARGET = REOLINK_PREWARM_TARGET
AUTOLABEL_VIDEO_LOW_WATERMARK = max(
    0, _ready_target_or_legacy_env("AUTOLABEL_VIDEO_LOW_WATERMARK", 1000)
)
AUTOLABEL_VIDEO_BATCH_SIZE = max(
    1, int(os.environ.get("AUTOLABEL_VIDEO_BATCH_SIZE", "3") or "3")
)
AUTOLABEL_VIDEO_AUTO_PREPROCESS = os.environ.get(
    "AUTOLABEL_VIDEO_AUTO_PREPROCESS",
    "0",
).strip().lower() in {"1", "true", "yes", "on"}
HYDRATED_FOLDER_CACHE_TTL_SECONDS = max(60, int(os.environ.get("LABEL_HYDRATED_CACHE_TTL_SECONDS", "900") or "900"))
READY_SCAN_MULTIPLIER = max(2, int(os.environ.get("LABEL_READY_SCAN_MULTIPLIER", "8") or "8"))
READY_SCAN_MAX = min(
    INTERACTIVE_READY_SCAN_CAP,
    max(
        100,
        _ready_target_or_legacy_env("LABEL_READY_SCAN_MAX", 180),
    ),
)
QUEUE_HYDRATE_BATCH_SIZE = max(
    1,
    int(
        os.environ.get(
            "LABEL_QUEUE_HYDRATE_BATCH_SIZE",
            str(min(max(HYDRATE_MAX_WORKERS, 1), 24)),
        )
        or "24"
    ),
)
QUEUE_HYDRATE_BUDGET_MS = max(
    250.0,
    float(os.environ.get("LABEL_QUEUE_HYDRATE_BUDGET_MS", "8000") or "8000"),
)
QUEUE_RETRY_MS = max(100, int(os.environ.get("LABEL_QUEUE_RETRY_MS", "250") or "250"))
TIMING_LOGS_ENABLED = os.environ.get("LABEL_TIMING_LOGS", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
TIMING_LOG_MIN_MS = max(0.0, float(os.environ.get("LABEL_TIMING_LOG_MIN_MS", "0") or "0"))
AUTH_REQUIRED = os.environ.get("AUTH_REQUIRED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LABELER_PASSWORD = os.environ.get("LABELER_PASSWORD", "")
MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
if AUTH_REQUIRED and not os.environ.get("FLASK_SECRET_KEY"):
    raise RuntimeError("FLASK_SECRET_KEY must be set when AUTH_REQUIRED=1")

VIDEO_SOURCE = "video"
REOLINK_SOURCE = "reolink"
LABEL_DESTINATIONS = ("clean", "dirty", "occupied", "label_later", "discarded")
MATTHEWS_SITE_KEY = "reolink-matthews-01"
CROP_CONFIGS_FOLDER_NAME = "crop_configs"
PROCESSED_RAW_FOLDER_NAME = "processed_raw"
UNASSOCIATED_ZIPS_FOLDER_NAME = "unassociated_zips"
UNASSOCIATED_ZIPS_MANIFEST_FILE = ".compactor_manifest.jsonl"
UNASSOCIATED_ZIPS_INNER_MANIFEST = "MANIFEST.json"
SCREENRECORD_TRUE_TEN_FOLDER_NAME = "10frametrue"
SCREENRECORD_THREE_FRAME_FOLDER_NAME = "3frame"
SCREENRECORD_THREE_FRAME_UNLABELED_KEY = "screenrecord_3frame_unlabeled"
SCREENRECORD_TRUE_TEN_NODE_KEY = "screenrecord_10frametrue_node"
PERCEPTION_V2_FILE_NAME = "perception_v2.json"
LEGACY_PERCEPTION_FILE_NAMES = ("perception.json", "perception_10frame.json")
PREPROCESS_STATE_SCHEMA_VERSION = 1
PREPROCESS_STATE_FILE_NAME = "preprocess_state.json"
SUPABASE_CROP_CACHE_FILE_NAME = "supabase_crop_cache.json"
SUPABASE_CROP_CACHE_TTL_SECONDS = max(
    30,
    int(os.environ.get("SUPABASE_CROP_CACHE_TTL_SECONDS", "300") or "300"),
)
LABEL_HISTORY_SCHEMA_VERSION = 1
LABEL_HISTORY_FILE_NAME = "label_history.json"
LABEL_JOBS_SCHEMA_VERSION = 1
LABEL_JOBS_FILE_NAME = "label_jobs.json"
LABEL_JOB_RECOVERED_ERROR = "Recovered after label worker fix; retrying Drive push."
LABEL_JOB_ERROR_LIMIT = max(1, int(os.environ.get("LABEL_JOB_ERROR_LIMIT", "25") or "25"))
LABEL_JOB_MAX_ATTEMPTS = max(1, int(os.environ.get("LABEL_JOB_MAX_ATTEMPTS", "100") or "100"))
LABEL_JOB_UNDO_SECONDS = max(0, int(os.environ.get("LABEL_JOB_UNDO_SECONDS", "30") or "30"))
LABEL_JOB_MIN_INTERVAL_SECONDS = max(0.0, float(os.environ.get("LABEL_JOB_MIN_INTERVAL_SECONDS", "0.15") or "0.15"))
LABEL_JOB_JITTER_SECONDS = max(0.0, float(os.environ.get("LABEL_JOB_JITTER_SECONDS", "0.05") or "0.05"))
LABEL_JOB_RATE_LIMIT_COOLDOWN_SECONDS = max(
    1.0,
    float(os.environ.get("LABEL_JOB_RATE_LIMIT_COOLDOWN_SECONDS", "120") or "120"),
)
LABEL_JOB_RATE_LIMIT_MAX_COOLDOWN_SECONDS = max(
    LABEL_JOB_RATE_LIMIT_COOLDOWN_SECONDS,
    float(os.environ.get("LABEL_JOB_RATE_LIMIT_MAX_COOLDOWN_SECONDS", "900") or "900"),
)
LABEL_JOB_PROCESSING_STALE_SECONDS = max(
    30,
    int(os.environ.get("LABEL_JOB_PROCESSING_STALE_SECONDS", "300") or "300"),
)
PROCESSED_RAW_RETENTION_DAYS = max(
    1, int(os.environ.get("PROCESSED_RAW_RETENTION_DAYS", "14") or "14")
)


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


@dataclass(frozen=True)
class LabelSource:
    """Single source of truth for a labeling source.

    Video and restaurant-pi-1 both belong to the Mimosas restaurant and share
    the same folder_prefix, so labeled samples from both end up in the same
    shared destination pool with a `mimosas-` name prefix. Intrinsic folder
    name bodies (`<video_stem>_t0004` vs `Reolink-CH-…`) distinguish capture method.
    """

    source: str
    site_key: str | None
    queue_key: str
    display_name: str
    folder_prefix: str
    manual_crop_configs: bool
    root_name: str | None  # Drive folder name for reolink sources


LABEL_SOURCES: tuple[LabelSource, ...] = (
    LabelSource(
        source="video",
        site_key=None,
        queue_key="video",
        display_name="Mimosas (Video)",
        folder_prefix="mimosas",
        manual_crop_configs=False,
        root_name=None,
    ),
    LabelSource(
        source="reolink",
        site_key="restaurant-pi-1",
        queue_key="reolink:restaurant-pi-1",
        display_name="Mimosas (Photos)",
        folder_prefix="mimosas",
        manual_crop_configs=False,
        root_name="restaurant-pi-1",
    ),
    LabelSource(
        source="reolink",
        site_key="reolink-matthews-01",
        queue_key="reolink:reolink-matthews-01",
        display_name="Matthews",
        folder_prefix="matthews",
        manual_crop_configs=True,
        root_name="reolink-matthews-01",
    ),
)

LABEL_SOURCES_BY_QUEUE_KEY: dict[str, LabelSource] = {
    entry.queue_key: entry for entry in LABEL_SOURCES
}

KNOWN_FOLDER_PREFIXES: tuple[str, ...] = tuple(
    sorted({entry.folder_prefix for entry in LABEL_SOURCES})
)


def _resolve_label_source(source: str | None, site_key: str | None) -> LabelSource:
    normalized_source = (source or VIDEO_SOURCE).strip().lower() or VIDEO_SOURCE
    normalized_site_key = (site_key or "").strip() or None

    if normalized_source == VIDEO_SOURCE:
        queue_key = VIDEO_SOURCE
    elif normalized_source == REOLINK_SOURCE:
        if not normalized_site_key:
            raise ValueError("site is required when source=reolink")
        queue_key = f"{REOLINK_SOURCE}:{normalized_site_key}"
    else:
        raise ValueError(f"Unknown source: {source}")

    entry = LABEL_SOURCES_BY_QUEUE_KEY.get(queue_key)
    if entry is None:
        raise ValueError(f"Unknown labeling source: {queue_key}")
    return entry


def _apply_source_prefix(name: str, source: LabelSource) -> str:
    """Prepend the restaurant prefix to a folder name unless already prefixed.

    Idempotent: safe to call multiple times; never double-prefixes.
    """
    if not name:
        return name
    for known in KNOWN_FOLDER_PREFIXES:
        if name.startswith(f"{known}-"):
            return name
    return f"{source.folder_prefix}-{name}"


# Keep this site list in code for now. If a site folder does not live directly
# under DRIVE_PROJECT_ROOT_FOLDER_ID, set its root_id explicitly here.
# Derived from LABEL_SOURCES so display names/flags stay in one place.
REOLINK_SITES = tuple(
    ReolinkSiteConfig(
        site_key=entry.site_key,  # type: ignore[arg-type]
        display_name=entry.display_name,
        root_name=entry.root_name,  # type: ignore[arg-type]
        manual_crop_configs=entry.manual_crop_configs,
    )
    for entry in LABEL_SOURCES
    if entry.source == "reolink" and entry.site_key and entry.root_name
)
REOLINK_SITES_BY_KEY = {site.site_key: site for site in REOLINK_SITES}


def _default_cache_dir() -> Path:
    configured = os.environ.get("LABEL_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()

    if os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("RAILWAY_PROJECT_ID"):
        return Path("/data/label_cache")

    local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
    if local_appdata:
        return Path(local_appdata) / "AutoLabeler" / "label_cache"

    return Path(tempfile.gettempdir()) / "AutoLabeler" / "label_cache"


CACHE_DIR = _default_cache_dir()
_RAILWAY_ENV = bool(os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("RAILWAY_PROJECT_ID"))
_DEFAULT_CACHE_TTL_HOURS = "336" if _RAILWAY_ENV else "72"
_DEFAULT_CACHE_MAX_MB = "20000" if _RAILWAY_ENV else "1024"
CACHE_TTL_HOURS = max(1, int(os.environ.get("LABEL_CACHE_TTL_HOURS", _DEFAULT_CACHE_TTL_HOURS) or _DEFAULT_CACHE_TTL_HOURS))
CACHE_MAX_MB = max(64, int(os.environ.get("LABEL_CACHE_MAX_MB", _DEFAULT_CACHE_MAX_MB) or _DEFAULT_CACHE_MAX_MB))
CACHE_WARM_ERROR_LIMIT = max(1, int(os.environ.get("LABEL_CACHE_WARM_ERROR_LIMIT", "25") or "25"))
CACHE_WARM_BATCH_SIZE = max(1, int(os.environ.get("LABEL_CACHE_WARM_BATCH_SIZE", "80") or "80"))
CACHE_WARM_BATCH_PAUSE_SECONDS = max(
    0.0,
    float(os.environ.get("LABEL_CACHE_WARM_BATCH_PAUSE_SECONDS", "0") or "0"),
)
CACHE_WARM_LOCK_STALE_SECONDS = max(
    60,
    _read_int_env("LABEL_CACHE_WARM_LOCK_STALE_SECONDS", 3600),
)

# Drive client + cached folder IDs
_source_folder_ids_cache: dict[str, dict[str, str]] = {}
_source_folder_ids_lock = Lock()
_cache_cleanup_lock = Lock()
_last_cache_cleanup_monotonic = 0.0
_label_history_lock = Lock()
_label_jobs_lock = Lock()
_label_job_worker_lock = Lock()
_label_job_worker_inflight = False
_label_job_worker_rerun_requested = False
_label_job_rate_limit_lock = Lock()
_label_job_last_attempt_at: datetime | None = None
_label_job_rate_limit_cooldown_until: datetime | None = None
_label_job_rate_limit_cooldown_seconds = LABEL_JOB_RATE_LIMIT_COOLDOWN_SECONDS
_label_job_last_rate_limit_error: str | None = None
_listing_cache: dict[str, tuple[float, list[dict[str, str]]]] = {}
_listing_lock = Lock()
_listing_refresh_executor = ThreadPoolExecutor(max_workers=1)
_listing_refresh_inflight: set[str] = set()
_preview_prewarm_executor = ThreadPoolExecutor(max_workers=PREVIEW_PREWARM_MAX_WORKERS)
_preview_prewarm_inflight: set[str] = set()
_preview_prewarm_lock = Lock()
_cache_warm_executor = ThreadPoolExecutor(max_workers=1)
_label_job_executor = ThreadPoolExecutor(max_workers=1)
_cache_warm_lock = Lock()
_cache_warm_state: dict[str, Any] = {
    "inflight": False,
    "started_at": None,
    "completed_at": None,
    "requested": {},
    "current_queue": None,
    "queues_total": 0,
    "queues_completed": 0,
    "folders_scanned": 0,
    "folders_hydrated": 0,
    "folders_hot_cached": 0,
    "frames_seen": 0,
    "full_res_cached": 0,
    "thumbs_cached": 0,
    "skipped_full_res": 0,
    "skipped_thumbs": 0,
    "errors": [],
    "last_error": None,
    "stop_requested": False,
    "batch_size": CACHE_WARM_BATCH_SIZE,
    "shared_lock": None,
    "shared_lock_path": None,
}
_folder_prewarm_executor = ThreadPoolExecutor(max_workers=FOLDER_PREWARM_MAX_WORKERS)
_folder_prewarm_inflight: set[tuple[str, str]] = set()
_folder_prewarm_lock = Lock()
_duplicate_cleanup_executor = ThreadPoolExecutor(max_workers=1)
_duplicate_cleanup_inflight: set[tuple[str, str]] = set()
_duplicate_cleanup_lock = Lock()
_hydrated_folder_cache: dict[tuple[str, str], tuple[float, dict | None]] = {}
_hydrated_folder_cache_lock = Lock()
_reolink_generation_lock = Lock()
_yolo_model_lock = Lock()
_video_preprocess_executor = ThreadPoolExecutor(max_workers=1)
_video_preprocess_lock = Lock()
_video_preprocess_state: dict[str, Any] = {
    "inflight": False,
    "last_run_at": None,
    "last_run_videos": 0,
    "last_run_triplets": 0,
    "last_error": None,
}
_reolink_preprocess_executor = ThreadPoolExecutor(max_workers=1)
_reolink_preprocess_lock = Lock()
_reolink_preprocess_inflight: set[str] = set()
_ready_maintainer_executor = ThreadPoolExecutor(max_workers=1)
_ready_maintainer_lock = Lock()
_ready_maintainer_started = False
_READY_MAINTAINER_INTERVAL_SECONDS = 15
READY_MAINTAINER_LOCK_STALE_SECONDS = max(
    300,
    int(os.environ.get("LABEL_READY_MAINTAINER_LOCK_STALE_SECONDS", "3600") or "3600"),
)
_ready_maintainer_state: dict[str, Any] = {
    "inflight": False,
    "started": False,
    "current_queue": None,
    "last_run_at": None,
    "generated": 0,
    "cache_warming": False,
    "last_error": None,
}
_yolo_model: Any | None = None
_camera_config_cache: dict[int, dict[str, Any]] | None = None
_camera_config_lock = Lock()
_crop_config_cache: dict[tuple[str, str], dict[str, Any] | None] = {}
_crop_config_lock = Lock()
_CROP_CONFIG_CACHE_MISS = object()
_supabase_crop_cache: dict[str, dict[str, Any]] = {}
_supabase_crop_cache_lock = Lock()
_supabase_crop_status: dict[str, Any] = {
    "enabled": False,
    "last_error": None,
    "last_lookup_at": None,
    "last_cache_hit": False,
    "last_camera_source_id": None,
    "last_table_count": 0,
}


def _csrf_token() -> str:
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return str(token)


@app.context_processor
def _template_security_context() -> dict[str, str]:
    return {"csrf_token": _csrf_token()}


def _wants_json_response() -> bool:
    return request.path.startswith("/api/") or "application/json" in request.headers.get("Accept", "")


def _auth_public_endpoint() -> bool:
    return request.endpoint in {"login", "healthz", "static"}


@app.before_request
def _require_auth_and_csrf() -> Any | None:
    if request.method == "OPTIONS" or _auth_public_endpoint():
        return None

    if AUTH_REQUIRED and not session.get("authenticated"):
        if _wants_json_response():
            return jsonify({"error": "authentication_required"}), 401
        next_url = request.full_path if request.query_string else request.path
        return redirect(url_for("login", next=next_url))

    if AUTH_REQUIRED and request.method in MUTATING_METHODS:
        expected = session.get("_csrf_token")
        supplied = request.headers.get("X-CSRF-Token", "")
        if not expected or not hmac.compare_digest(str(expected), supplied):
            return jsonify({"error": "csrf_token_invalid"}), 403

    return None


@app.errorhandler(Exception)
def _api_json_error(error: Exception) -> Any:
    if not request.path.startswith("/api/"):
        if isinstance(error, HTTPException):
            return error
        raise error

    if isinstance(error, HTTPException):
        status_code = int(error.code or 500)
        message = str(error.description or error.name or "Request failed")
        code = getattr(error, "name", "http_error").lower().replace(" ", "_")
    else:
        status_code = 500
        message = "Internal server error"
        code = "internal_error"
        app.logger.exception("Unhandled API error on %s", request.path)

    return jsonify({"error": message, "code": code}), status_code


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

    temp_cache = Path(tempfile.gettempdir()) / "AutoLabeler" / "label_cache"
    repo_cache = Path(__file__).parent / "label_cache"

    configured_cache = os.environ.get("LABEL_CACHE_DIR", "").strip()
    if configured_cache:
        candidates = [CACHE_DIR]
    elif repo_cache.exists():
        # Older local runs used repo-local label_cache/. Reuse it when present
        # so cached Drive previews do not get stranded behind a new temp path.
        candidates = [repo_cache, temp_cache]
    else:
        candidates = [CACHE_DIR]

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


def _find_screenrecord_three_frame_unlabeled(client: DriveClient, node_root_id: str) -> str | None:
    three_frame = client.find_file_by_name(
        node_root_id,
        SCREENRECORD_THREE_FRAME_FOLDER_NAME,
        mime_type=FOLDER_MIME,
    )
    if not three_frame or not three_frame.get("id"):
        return None
    unlabeled = client.find_file_by_name(
        str(three_frame["id"]),
        "unlabeled",
        mime_type=FOLDER_MIME,
    )
    return str(unlabeled["id"]) if unlabeled and unlabeled.get("id") else None


def _ensure_screenrecord_three_frame_unlabeled(client: DriveClient, node_root_id: str) -> str:
    three_frame_id = client.ensure_subfolder(node_root_id, SCREENRECORD_THREE_FRAME_FOLDER_NAME)
    return client.ensure_subfolder(three_frame_id, "unlabeled")


def _find_screenrecord_true_ten_node_folder(client: DriveClient, node_root_id: str) -> str | None:
    true_ten_root = client.find_file_by_name(
        _root_id(),
        SCREENRECORD_TRUE_TEN_FOLDER_NAME,
        mime_type=FOLDER_MIME,
    )
    if not true_ten_root or not true_ten_root.get("id"):
        return None

    node_root = client.get_file(node_root_id, fields="id,name,mimeType,parents")
    node_name = str(node_root.get("name") or "").strip()
    if not node_name:
        return None
    node_folder = client.find_file_by_name(
        str(true_ten_root["id"]),
        node_name,
        mime_type=FOLDER_MIME,
    )
    return str(node_folder["id"]) if node_folder and node_folder.get("id") else None


_SHARED_DESTINATIONS_CACHE_KEY = "__shared_destinations__"


def _shared_destination_folder_ids(client: DriveClient) -> dict[str, str]:
    """Return Drive IDs for the shared label destinations under the project root.

    All sources write labeled samples into the same {clean, dirty, occupied,
    label_later, discarded} folders under DRIVE_PROJECT_ROOT_FOLDER_ID. Folder
    names carry a restaurant prefix to preserve source attribution.
    """
    with _source_folder_ids_lock:
        cached = _source_folder_ids_cache.get(_SHARED_DESTINATIONS_CACHE_KEY)
        if cached is not None:
            return cached

        root = _root_id()
        shared = {name: client.ensure_subfolder(root, name) for name in LABEL_DESTINATIONS}
        _source_folder_ids_cache[_SHARED_DESTINATIONS_CACHE_KEY] = shared
        return shared


def _video_folder_ids(client: DriveClient) -> dict[str, str]:
    queue_key = VIDEO_SOURCE
    shared = _shared_destination_folder_ids(client)
    with _source_folder_ids_lock:
        cached = _source_folder_ids_cache.get(queue_key)
        if cached is not None:
            if any(cached.get(name) != shared[name] for name in LABEL_DESTINATIONS):
                cached = {**cached, **shared}
                _source_folder_ids_cache[queue_key] = cached
            return cached

        root = _root_id()
        folder_ids = {
            name: client.ensure_subfolder(root, name)
            for name in ("raw_videos", "temp_processing", "unlabeled")
        }
        folder_ids.update(shared)
        _source_folder_ids_cache[queue_key] = folder_ids
        return folder_ids


def _reolink_site_folder_ids(client: DriveClient, site_key: str) -> dict[str, str]:
    queue_key = f"{REOLINK_SOURCE}:{site_key}"
    shared = _shared_destination_folder_ids(client)
    with _source_folder_ids_lock:
        cached = _source_folder_ids_cache.get(queue_key)
        if cached is not None:
            updated = cached
            if _site_uses_manual_crop_configs(site_key) and "crop_configs" not in updated:
                updated = dict(updated)
                updated["crop_configs"] = client.ensure_subfolder(updated["root"], CROP_CONFIGS_FOLDER_NAME)
            if PROCESSED_RAW_FOLDER_NAME not in updated:
                updated = dict(updated)
                updated[PROCESSED_RAW_FOLDER_NAME] = client.ensure_subfolder(
                    updated["root"],
                    PROCESSED_RAW_FOLDER_NAME,
                )
            if UNASSOCIATED_ZIPS_FOLDER_NAME not in updated:
                existing_zips = client.find_file_by_name(
                    updated["root"], UNASSOCIATED_ZIPS_FOLDER_NAME, mime_type=FOLDER_MIME
                )
                if existing_zips and existing_zips.get("id"):
                    updated = dict(updated)
                    updated[UNASSOCIATED_ZIPS_FOLDER_NAME] = str(existing_zips["id"])
            if SCREENRECORD_THREE_FRAME_UNLABELED_KEY not in updated:
                updated = dict(updated)
                updated[SCREENRECORD_THREE_FRAME_UNLABELED_KEY] = _ensure_screenrecord_three_frame_unlabeled(
                    client,
                    updated["root"],
                )
            if SCREENRECORD_TRUE_TEN_NODE_KEY not in updated:
                screenrecord_true_ten = _find_screenrecord_true_ten_node_folder(client, updated["root"])
                if screenrecord_true_ten:
                    updated = dict(updated)
                    updated[SCREENRECORD_TRUE_TEN_NODE_KEY] = screenrecord_true_ten
            if any(updated.get(name) != shared[name] for name in LABEL_DESTINATIONS):
                updated = {**updated, **shared}
            if updated is not cached:
                _source_folder_ids_cache[queue_key] = updated
            return updated

        site = _resolve_site_config(site_key)
        site_root_id = _discover_reolink_root_id(client, site)
        unassociated = client.find_file_by_name(site_root_id, "unassociated", mime_type=FOLDER_MIME)

        folder_ids = {
            "root": site_root_id,
            "unlabeled": client.ensure_subfolder(site_root_id, "unlabeled"),
            PROCESSED_RAW_FOLDER_NAME: client.ensure_subfolder(
                site_root_id,
                PROCESSED_RAW_FOLDER_NAME,
            ),
            SCREENRECORD_THREE_FRAME_UNLABELED_KEY: _ensure_screenrecord_three_frame_unlabeled(
                client,
                site_root_id,
            ),
        }
        if unassociated and unassociated.get("id"):
            folder_ids["unassociated"] = str(unassociated["id"])
        existing_zips = client.find_file_by_name(
            site_root_id, UNASSOCIATED_ZIPS_FOLDER_NAME, mime_type=FOLDER_MIME
        )
        if existing_zips and existing_zips.get("id"):
            folder_ids[UNASSOCIATED_ZIPS_FOLDER_NAME] = str(existing_zips["id"])
        screenrecord_true_ten = _find_screenrecord_true_ten_node_folder(client, site_root_id)
        if screenrecord_true_ten:
            folder_ids[SCREENRECORD_TRUE_TEN_NODE_KEY] = screenrecord_true_ten
        if site.manual_crop_configs:
            folder_ids["crop_configs"] = client.ensure_subfolder(site_root_id, CROP_CONFIGS_FOLDER_NAME)
        folder_ids.update(shared)

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
    label_source = _resolve_label_source(source, site_key)
    if label_source.source == VIDEO_SOURCE:
        folder_ids = _video_folder_ids(client)
        return QueueContext(
            source=label_source.source,
            site_key=None,
            queue_key=label_source.queue_key,
            display_name=label_source.display_name,
            input_folder_name="unlabeled",
            input_folder_id=folder_ids["unlabeled"],
            seed_folder_name=None,
            seed_folder_id=None,
            folder_ids=folder_ids,
            persist_frame_metadata=True,
        )

    folder_ids = _reolink_site_folder_ids(client, label_source.site_key or "")
    return QueueContext(
        source=label_source.source,
        site_key=label_source.site_key,
        queue_key=label_source.queue_key,
        display_name=label_source.display_name,
        input_folder_name="unlabeled",
        input_folder_id=folder_ids["unlabeled"],
        seed_folder_name="unassociated",
        seed_folder_id=folder_ids.get("unassociated"),
        folder_ids=folder_ids,
        persist_frame_metadata=False,
    )


def _request_source_args() -> tuple[str, str | None]:
    source = request.args.get("source", VIDEO_SOURCE)
    site_key = request.args.get("site")
    return source, site_key


def _payload_source_args(data: dict[str, Any]) -> tuple[str, str | None]:
    source = str(data.get("source", VIDEO_SOURCE) or VIDEO_SOURCE)
    site_key_value = data.get("site_key", data.get("site"))
    site_key = str(site_key_value).strip() if site_key_value else None
    return source, site_key


def _request_json_payload() -> dict[str, Any]:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("JSON object body required")
    return data


REVIEW_UNLABELED_BUCKET = "unlabeled"
REVIEW_SCREENRECORD_BUCKET = "screenrecord_3frame_unlabeled"
REVIEW_BUCKETS = (REVIEW_UNLABELED_BUCKET, REVIEW_SCREENRECORD_BUCKET, *LABEL_DESTINATIONS)
REVIEW_LABELED_DEFAULT_BUCKETS = ("clean", "dirty", "occupied")
REVIEW_LEGACY_DEFAULT_BUCKETS = (
    REVIEW_UNLABELED_BUCKET,
    REVIEW_SCREENRECORD_BUCKET,
    "clean",
    "dirty",
    "occupied",
)
CROP_CLEANUP_DEFAULT_BUCKETS = REVIEW_BUCKETS
CROP_CLEANUP_FALLBACK_KINDS = {"fallback_json", "drive_crop_config"}


def _parse_csv_arg(name: str, default: tuple[str, ...] = ()) -> list[str]:
    values: list[str] = []
    for raw in request.args.getlist(name):
        for part in str(raw).split(","):
            value = part.strip()
            if value:
                values.append(value)
    return values or list(default)


def _review_bucket_parent_ids(context: QueueContext, buckets: list[str]) -> dict[str, str]:
    parent_ids: dict[str, str] = {}
    for bucket in buckets:
        if bucket == REVIEW_UNLABELED_BUCKET:
            parent_ids[bucket] = context.input_folder_id
        elif bucket == REVIEW_SCREENRECORD_BUCKET:
            screenrecord_id = context.folder_ids.get(SCREENRECORD_THREE_FRAME_UNLABELED_KEY)
            if screenrecord_id:
                parent_ids[bucket] = screenrecord_id
        elif bucket in LABEL_DESTINATIONS:
            parent_ids[bucket] = context.folder_ids[bucket]
        else:
            raise ValueError(f"Unknown review bucket: {bucket}")
    return parent_ids


def _review_source_prefix(context: QueueContext) -> str:
    try:
        return _resolve_label_source(context.source, context.site_key).folder_prefix
    except ValueError:
        return ""


def _folder_matches_review_source(folder_name: str, context: QueueContext, bucket: str) -> bool:
    if bucket not in LABEL_DESTINATIONS:
        return True
    prefix = _review_source_prefix(context)
    return not prefix or folder_name.startswith(f"{prefix}-")


def _review_folder_matches_context(context: QueueContext, folder_name: str, app_properties: dict[str, Any], bucket: str) -> bool:
    if bucket not in LABEL_DESTINATIONS:
        return True
    if not _folder_matches_review_source(folder_name, context, bucket):
        return False

    queue_key = str(app_properties.get("autolabel_queue_key") or "").strip()
    if queue_key:
        return queue_key == context.queue_key

    source = str(app_properties.get("autolabel_source") or "").strip()
    if source and source != context.source:
        return False

    site_key = str(app_properties.get("autolabel_site_key") or "").strip()
    if site_key and site_key != (context.site_key or ""):
        return False

    is_reolink_name = "Reolink-" in folder_name
    if context.source == REOLINK_SOURCE:
        return is_reolink_name
    if context.source == VIDEO_SOURCE:
        return not is_reolink_name
    return True


def _review_channel_hint(folder_name: str, metadata: dict[str, Any] | None = None) -> str:
    for value in _camera_id_candidates(folder_name, metadata):
        channel = _extract_reolink_channel_code(value)
        if channel:
            return channel
        ipc = re.search(r"IPC[\s_-]*(\d+)", value, re.IGNORECASE)
        if ipc:
            return f"IPC{int(ipc.group(1))}"
    return ""


def _review_table_hint(folder_name: str, metadata: dict[str, Any] | None = None) -> str:
    if isinstance(metadata, dict):
        table = metadata.get("table")
        if isinstance(table, dict):
            for key in ("label", "id", "table_id"):
                value = str(table.get(key) or "").strip()
                if value:
                    return value
        for key in ("table_label", "supabase_table_id", "table_camera_crops_id"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
    base, _suffix = _split_triplet_suffix(folder_name)
    match = re.search(r"_(table[^_]*(?:_[^_]+)*)$", base, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""


def _review_source_type(
    context: QueueContext,
    bucket: str,
    parent_id: str,
    folder_name: str,
    payload: dict[str, Any],
) -> str:
    screenrecord_id = context.folder_ids.get(SCREENRECORD_THREE_FRAME_UNLABELED_KEY)
    if (
        bucket == REVIEW_SCREENRECORD_BUCKET
        or (screenrecord_id and parent_id == screenrecord_id)
        or payload.get("source_label") == SCREENRECORD_TRUE_TEN_FOLDER_NAME
        or payload.get("perception_file_name") == PERCEPTION_V2_FILE_NAME
    ):
        return "screenrecord"
    if bucket in LABEL_DESTINATIONS:
        return "labeled"
    if context.source == REOLINK_SOURCE and "Reolink-" in folder_name:
        return "legacy"
    return "generated"


def _review_crop_provenance(
    context: QueueContext,
    metadata: dict[str, Any] | None,
    source_type: str,
) -> dict[str, Any]:
    metadata = metadata or {}
    crop_source = str(metadata.get("crop_source") or "").strip()
    table_camera_crops_id = str(metadata.get("table_camera_crops_id") or "").strip()
    supabase_table_id = str(metadata.get("supabase_table_id") or "").strip()
    camera_source_id = str(metadata.get("camera_source_id") or "").strip()
    crop_version = metadata.get("crop_version")

    if table_camera_crops_id or supabase_table_id or crop_source == "supabase_table_camera_crops":
        kind = "supabase"
        label = "Supabase crop"
    elif crop_source == "manual_crop_config" or (
        context.source == REOLINK_SOURCE and context.site_key and _site_uses_manual_crop_configs(context.site_key)
    ):
        kind = "drive_crop_config"
        label = "Drive crop config JSON"
    elif crop_source:
        kind = crop_source
        label = crop_source.replace("_", " ")
    elif source_type in {"legacy", "generated", "labeled", "screenrecord"}:
        kind = "fallback_json"
        label = "Fallback JSON"
    else:
        kind = "unknown"
        label = "Unknown crop source"

    return {
        "kind": kind,
        "label": label,
        "is_supabase": kind == "supabase",
        "is_fallback": kind in {"fallback_json", "drive_crop_config"},
        "crop_source": crop_source or None,
        "table_camera_crops_id": table_camera_crops_id or None,
        "supabase_table_id": supabase_table_id or None,
        "camera_source_id": camera_source_id or None,
        "crop_version": crop_version,
    }


def _review_payload_for_folder(
    client: DriveClient,
    context: QueueContext,
    folder: dict[str, Any],
    bucket: str,
    parent_id: str,
) -> dict[str, Any] | None:
    payload = _hydrate_folder(client, context, folder)
    if payload is None:
        return None
    files = client.list_files(str(folder["id"]))
    files_by_name = _file_by_name(files)
    metadata = _load_json_file_from_drive(client, files_by_name.get("metadata.json"))
    folder_name = str(payload.get("folder_name") or folder.get("name") or "")
    frame_count = len(_ordered_frame_keys(payload.get("frames") or {}))
    source_type = _review_source_type(context, bucket, parent_id, folder_name, payload)
    crop_provenance = _review_crop_provenance(context, metadata, source_type)
    payload.update(
        {
            "bucket": bucket,
            "current_label": bucket if bucket in LABEL_DESTINATIONS else None,
            "review_source_type": source_type,
            "crop_provenance": crop_provenance,
            "crop_source_kind": crop_provenance["kind"],
            "has_supabase_crop": crop_provenance["is_supabase"],
            "is_fallback_crop": crop_provenance["is_fallback"],
            "table_camera_crops_id": crop_provenance["table_camera_crops_id"],
            "supabase_table_id": crop_provenance["supabase_table_id"],
            "camera_source_id": crop_provenance["camera_source_id"],
            "crop_version": crop_provenance["crop_version"],
            "channel_hint": _review_channel_hint(folder_name, metadata),
            "table_hint": _review_table_hint(folder_name, metadata),
            "frame_count": frame_count,
            "modified_time": folder.get("modifiedTime"),
            "metadata_file_id": payload.get("metadata_file_id") or (files_by_name.get("metadata.json") or {}).get("id"),
        }
    )
    return payload


def _review_payload_matches_filters(payload: dict[str, Any], filters: dict[str, str]) -> bool:
    folder_name = str(payload.get("folder_name") or "").lower()
    q = filters.get("q", "").lower()
    if q and q not in folder_name:
        return False
    channel = filters.get("channel", "").lower()
    if channel and channel not in str(payload.get("channel_hint") or "").lower() and channel not in folder_name:
        return False
    table = filters.get("table", "").lower()
    if table and table not in str(payload.get("table_hint") or "").lower() and table not in folder_name:
        return False
    source_type = filters.get("folder_source_type", "").lower()
    if source_type and source_type not in {"all", str(payload.get("review_source_type") or "").lower()}:
        return False
    crop_source_kind = filters.get("crop_source_kind", "").lower()
    if crop_source_kind and crop_source_kind not in {"all", str(payload.get("crop_source_kind") or "").lower()}:
        return False
    frame_count = filters.get("frame_count", "")
    if frame_count:
        try:
            if int(frame_count) != int(payload.get("frame_count") or 0):
                return False
        except ValueError:
            raise ValueError("frame_count must be an integer")
    return True


def _review_list_folders(
    client: DriveClient,
    context: QueueContext,
    buckets: list[str],
    filters: dict[str, str],
    *,
    limit: int,
    cursor: int = 0,
) -> tuple[list[dict[str, Any]], int | None, int]:
    parent_ids = _review_bucket_parent_ids(context, buckets)
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    seen_folder_ids: set[str] = set()
    for bucket, parent_id in parent_ids.items():
        for folder in client.list_folders(parent_id, fields="id,name,mimeType,parents,appProperties,modifiedTime"):
            folder_id = str(folder.get("id") or "")
            folder_name = str(folder.get("name") or "")
            if not folder_id or folder_id in seen_folder_ids:
                continue
            app_properties = dict(folder.get("appProperties") or {})
            if not _review_folder_matches_context(context, folder_name, app_properties, bucket):
                continue
            seen_folder_ids.add(folder_id)
            candidates.append((bucket, parent_id, folder))

    candidates.sort(key=lambda item: str(item[2].get("modifiedTime") or ""), reverse=True)

    results: list[dict[str, Any]] = []
    next_cursor: int | None = None
    index = max(0, cursor)
    while index < len(candidates):
        bucket, parent_id, folder = candidates[index]
        index += 1
        payload = _review_payload_for_folder(client, context, folder, bucket, parent_id)
        if payload is None:
            continue
        if _review_payload_matches_filters(payload, filters):
            results.append(payload)
        if len(results) >= limit:
            next_cursor = index if index < len(candidates) else None
            break

    return results, next_cursor, len(candidates)


def _cleanup_channel_hint_from_camera(camera: dict[str, Any]) -> str:
    config = camera.get("config") if isinstance(camera.get("config"), dict) else {}
    for value in (
        config.get("edge_camera_id"),
        config.get("edge_camera_key"),
        camera.get("name"),
    ):
        channel = _review_channel_hint(str(value or ""))
        if channel:
            if channel.upper().startswith("IPC"):
                match = re.search(r"IPC(\d+)", channel, re.IGNORECASE)
                if match:
                    return f"CH-CH{int(match.group(1)):02d}"
            return channel
    return ""


def _drive_image_dimensions(
    client: DriveClient,
    file_id: str,
    fallback: tuple[int, int] = (0, 0),
) -> tuple[int, int]:
    try:
        from PIL import Image

        with tempfile.TemporaryDirectory(prefix="drive_image_dimensions_") as tmpdir:
            reference_path = Path(tmpdir) / "frame.jpg"
            client.download_file_to_path(file_id, reference_path)
            with Image.open(reference_path) as image:
                return image.width, image.height
    except Exception:
        return fallback


def _reference_payload_for_frame(
    client: DriveClient,
    site_key: str,
    site_label: str,
    channel_code: str,
    raw_folder: dict[str, Any],
    frame_item: dict[str, Any],
    source: str,
    fallback_dimensions: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    frame_file_id = str(frame_item["id"])
    width, height = _drive_image_dimensions(client, frame_file_id, fallback_dimensions)
    return {
        "site_key": site_key,
        "site_label": site_label,
        "channel_code": channel_code,
        "raw_folder_id": str(raw_folder["id"]),
        "raw_folder_name": str(raw_folder.get("name", "")),
        "frame_file_id": frame_file_id,
        "preview_url": f"/api/preview/{frame_file_id}",
        "source": source,
        "width": width,
        "height": height,
    }


def _find_reolink_true_ten_reference_frame(
    client: DriveClient,
    context: QueueContext,
    channel_code: str,
    fallback_dimensions: tuple[int, int] = (0, 0),
) -> dict[str, Any] | None:
    if context.source != REOLINK_SOURCE or not context.site_key:
        return None
    normalized_channel = _normalize_reolink_channel_code(channel_code or "")
    if not normalized_channel:
        return None
    site = _resolve_site_config(context.site_key)
    for raw_folder in sorted(
        _list_screenrecord_true_ten_folders(client, context),
        key=lambda item: str(item.get("name", "")).lower(),
        reverse=True,
    ):
        source_files = {
            item["name"]: item
            for item in client.list_files(
                raw_folder["id"],
                fields="id,name,mimeType,parents,appProperties",
            )
        }
        metadata = _load_json_file_from_drive(client, source_files.get("metadata.json")) or {}
        raw_channel = _screenrecord_channel_code_from_metadata(str(raw_folder.get("name", "")), metadata)
        if raw_channel != normalized_channel:
            continue
        frame_item = source_files.get("frame_0.jpg")
        if not frame_item or not frame_item.get("id"):
            continue
        return _reference_payload_for_frame(
            client,
            context.site_key,
            site.display_name,
            normalized_channel,
            raw_folder,
            frame_item,
            SCREENRECORD_TRUE_TEN_FOLDER_NAME,
            fallback_dimensions,
        )
    return None


def _cleanup_reference_for_channel(
    client: DriveClient,
    context: QueueContext,
    channel_hint: str,
    fallback_dimensions: tuple[int, int] = (0, 0),
) -> dict[str, Any] | None:
    if context.source != REOLINK_SOURCE or not context.site_key or not channel_hint:
        return None
    try:
        channel_code = _normalize_reolink_channel_code(channel_hint) or channel_hint
        true_ten_reference = _find_reolink_true_ten_reference_frame(
            client,
            context,
            channel_code,
            fallback_dimensions,
        )
        if true_ten_reference is not None:
            return true_ten_reference
        return _find_reolink_reference_frame(client, context.site_key, channel_code)
    except Exception:
        return None


def _cleanup_supabase_crop_cards(client: DriveClient, context: QueueContext) -> list[dict[str, Any]]:
    supabase_client = _get_supabase_crop_client()
    if not supabase_client.enabled:
        return []
    try:
        cameras = supabase_client.select(
            "camera_sources",
            {
                "select": "id,name,restaurant_id,is_active,config",
                "limit": "1000",
            },
        )
    except Exception:
        return []

    cards: list[dict[str, Any]] = []
    for camera in cameras:
        if camera.get("is_active") is False or not _camera_source_matches_site_key(camera, context.site_key):
            continue
        camera_source_id = str(camera.get("id") or "").strip()
        if not camera_source_id:
            continue
        crops = _supabase_active_crops_for_camera(supabase_client, camera_source_id)
        if not crops:
            continue
        table_rows = _supabase_table_rows_by_id(
            supabase_client,
            [str(crop.get("table_id") or "") for crop in crops],
        )
        channel_hint = _cleanup_channel_hint_from_camera(camera)
        reference = _cleanup_reference_for_channel(client, context, channel_hint)
        for idx, crop in enumerate(crops):
            raw_polygon = crop.get("polygon")
            polygon: list[list[float]] = []
            if isinstance(raw_polygon, list):
                for point in raw_polygon:
                    if isinstance(point, (list, tuple)) and len(point) == 2:
                        polygon.append([float(point[0]), float(point[1])])
            table_row = table_rows.get(str(crop.get("table_id") or ""))
            label = _supabase_table_label(table_row, crop, idx)
            cards.append(
                {
                    "kind": "supabase",
                    "id": str(crop.get("id") or f"{camera_source_id}:{idx}"),
                    "label": label,
                    "table_hint": _safe_table_slug(label, f"table_{idx + 1}"),
                    "channel_hint": channel_hint,
                    "camera_source_id": camera_source_id,
                    "camera_name": camera.get("name"),
                    "restaurant_id": crop.get("restaurant_id") or camera.get("restaurant_id"),
                    "table_id": crop.get("table_id"),
                    "table_camera_crops_id": crop.get("id"),
                    "crop_version": crop.get("version"),
                    "crop_source": crop.get("source") or "supabase_table_camera_crops",
                    "polygon": polygon,
                    "frame_width": crop.get("frame_width") or (reference or {}).get("width"),
                    "frame_height": crop.get("frame_height") or (reference or {}).get("height"),
                    "reference": reference,
                }
            )
    return cards


def _cleanup_group_key(context: QueueContext, payload: dict[str, Any]) -> str:
    channel = str(payload.get("channel_hint") or "unknown-channel").strip().lower()
    table = str(payload.get("table_hint") or payload.get("folder_name") or "unknown-table").strip().lower()
    crop_kind = str(payload.get("crop_source_kind") or "fallback_json").strip().lower()
    return "|".join([context.queue_key, crop_kind, channel, table])


def _cleanup_manual_crop_visual(
    client: DriveClient,
    context: QueueContext,
    channel_hint: str,
    table_hint: str,
) -> dict[str, Any]:
    if context.source != REOLINK_SOURCE or not context.site_key or not channel_hint:
        return {}
    channel_code = _normalize_reolink_channel_code(channel_hint)
    if not channel_code:
        return {}
    crop_config = _load_saved_crop_config(client, context.site_key, channel_code)
    if not crop_config:
        return {}

    reference = crop_config.get("reference") or {}
    dimensions = (int(reference.get("width") or 0), int(reference.get("height") or 0))
    wanted = _safe_table_slug(table_hint, "").lower()
    polygon: list[list[float]] = []
    if wanted:
        for idx, crop in enumerate(crop_config.get("crops", [])):
            crop_name = str(crop.get("name") or f"table_{idx + 1}").strip() or f"table_{idx + 1}"
            crop_slug = _safe_table_slug(crop_name, f"table_{idx + 1}").lower()
            if wanted not in {crop_name.lower(), crop_slug}:
                continue
            raw_polygon = crop.get("polygon")
            if isinstance(raw_polygon, list):
                for point in raw_polygon:
                    if isinstance(point, (list, tuple)) and len(point) == 2:
                        polygon.append([float(point[0]), float(point[1])])
            break

    return {
        "polygon": polygon,
        "reference_dimensions": dimensions,
    }


def _cleanup_attach_group_visuals(
    client: DriveClient,
    context: QueueContext,
    groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reference_cache: dict[tuple[str, tuple[int, int]], dict[str, Any] | None] = {}
    for group in groups:
        channel_hint = str(group.get("channel_hint") or "").strip()
        table_hint = str(group.get("table_hint") or "").strip()
        manual_visual = _cleanup_manual_crop_visual(client, context, channel_hint, table_hint)
        fallback_dimensions = manual_visual.get("reference_dimensions") or (0, 0)
        reference_key = (
            _normalize_reolink_channel_code(channel_hint) or channel_hint,
            fallback_dimensions,
        )
        if reference_key not in reference_cache:
            reference_cache[reference_key] = _cleanup_reference_for_channel(
                client,
                context,
                channel_hint,
                fallback_dimensions,
            )
        reference = reference_cache[reference_key]
        if reference is not None:
            group["reference"] = reference
        if manual_visual.get("polygon"):
            group["polygon"] = manual_visual["polygon"]
    return groups


def _cleanup_folder_matches_context(context: QueueContext, folder_name: str, app_properties: dict[str, Any], bucket: str) -> bool:
    if not _review_folder_matches_context(context, folder_name, app_properties, bucket):
        return False
    queue_key = str(app_properties.get("autolabel_queue_key") or "").strip()
    if queue_key:
        return queue_key == context.queue_key
    prefix = _review_source_prefix(context)
    if prefix and folder_name.startswith(f"{prefix}-"):
        return True
    is_reolink_name = "Reolink-" in folder_name
    if context.source == REOLINK_SOURCE:
        return is_reolink_name
    if context.source == VIDEO_SOURCE:
        return not is_reolink_name
    return True


def _cleanup_fallback_groups(
    client: DriveClient,
    context: QueueContext,
    buckets: list[str],
    filters: dict[str, str],
) -> list[dict[str, Any]]:
    parent_ids = _review_bucket_parent_ids(context, buckets)
    groups: dict[str, dict[str, Any]] = {}
    seen_folder_ids: set[str] = set()
    for bucket, parent_id in parent_ids.items():
        for folder in client.list_folders(parent_id, fields="id,name,mimeType,parents,appProperties,modifiedTime"):
            folder_id = str(folder.get("id") or "")
            folder_name = str(folder.get("name") or "")
            if not folder_id or folder_id in seen_folder_ids:
                continue
            app_properties = dict(folder.get("appProperties") or {})
            if not _cleanup_folder_matches_context(context, folder_name, app_properties, bucket):
                continue
            seen_folder_ids.add(folder_id)
            payload = _review_payload_for_folder(client, context, folder, bucket, parent_id)
            if payload is None or str(payload.get("crop_source_kind") or "") not in CROP_CLEANUP_FALLBACK_KINDS:
                continue
            if not _review_payload_matches_filters(payload, filters):
                continue
            group_key = _cleanup_group_key(context, payload)
            group = groups.setdefault(
                group_key,
                {
                    "group_id": group_key,
                    "crop_source_kind": payload.get("crop_source_kind"),
                    "crop_label": (payload.get("crop_provenance") or {}).get("label"),
                    "channel_hint": payload.get("channel_hint"),
                    "table_hint": payload.get("table_hint"),
                    "folder_ids": [],
                    "folder_count": 0,
                    "bucket_counts": {},
                    "representative": payload,
                    "folders": [],
                },
            )
            group["folder_ids"].append(folder_id)
            group["folder_count"] += 1
            group["bucket_counts"][bucket] = int(group["bucket_counts"].get(bucket, 0)) + 1
            group["folders"].append(payload)
            if str(folder.get("modifiedTime") or "") > str((group["representative"] or {}).get("modified_time") or ""):
                group["representative"] = payload
    sorted_groups = sorted(
        groups.values(),
        key=lambda group: (
            str(group.get("channel_hint") or ""),
            str(group.get("table_hint") or ""),
            -int(group.get("folder_count") or 0),
        ),
    )
    return _cleanup_attach_group_visuals(client, context, sorted_groups)


def _cleanup_card_matches_filters(card: dict[str, Any], filters: dict[str, str]) -> bool:
    q = filters.get("q", "").lower()
    haystack = " ".join(
        str(card.get(key) or "")
        for key in ("label", "table_hint", "channel_hint", "camera_name", "table_camera_crops_id")
    ).lower()
    if q and q not in haystack:
        return False
    channel = filters.get("channel", "").lower()
    if channel and channel not in str(card.get("channel_hint") or "").lower():
        return False
    table = filters.get("table", "").lower()
    if table and table not in str(card.get("table_hint") or "").lower() and table not in str(card.get("label") or "").lower():
        return False
    return True


def _review_allowed_parent_ids(context: QueueContext) -> set[str]:
    return set(_review_bucket_parent_ids(context, list(REVIEW_BUCKETS)).values())


def _review_current_parent(current: dict[str, Any]) -> str | None:
    parents = [str(parent) for parent in current.get("parents", []) if parent]
    return parents[0] if parents else None


def _review_bucket_for_parent(context: QueueContext, parent_id: str) -> str | None:
    for bucket, bucket_parent_id in _review_bucket_parent_ids(context, list(REVIEW_BUCKETS)).items():
        if bucket_parent_id == parent_id:
            return bucket
    return None


def _review_validate_folder_parent(context: QueueContext, current: dict[str, Any]) -> str:
    parent_id = _review_current_parent(current)
    if not parent_id or parent_id not in _review_allowed_parent_ids(context):
        raise ValueError("folder is not in a reviewable Drive bucket")
    bucket = _review_bucket_for_parent(context, parent_id)
    folder_name = str(current.get("name") or "")
    app_properties = dict(current.get("appProperties") or {})
    if not bucket or not _review_folder_matches_context(context, folder_name, app_properties, bucket):
        raise ValueError("folder does not belong to the selected review source")
    return parent_id


def _review_signature_for_current_folder(client: DriveClient, folder_id: str) -> tuple[str, dict[str, str | None], str]:
    current = client.get_file(folder_id, fields="id,name,parents,appProperties")
    frames = _frame_payload_from_folder(current)
    if not has_complete_frame_ids(frames):
        frames = _frame_payload_from_files(client.list_files(folder_id))
    frame_signature = _frame_signature_from_frames(frames) if has_complete_frame_ids(frames) else ""
    folder_name = str(current.get("name") or "")
    return folder_name, frames, frame_signature


def _log_label_route_error(
    error: object,
    *,
    folder_id: str = "",
    folder_name: str = "",
    label: str = "",
    source: str = "",
    site_key: str | None = None,
    queue_key: str = "",
) -> None:
    app.logger.exception(
        "label route failed folder_id=%s folder_name=%s label=%s source=%s site_key=%s queue=%s error=%s",
        folder_id,
        folder_name,
        label,
        source,
        site_key,
        queue_key,
        error,
    )


def _labeler_name() -> str:
    if not has_request_context():
        return "background"
    return str(session.get("labeler_name") or "local")


def _label_app_properties(label: str, context: QueueContext, labeler_name: str | None = None) -> dict[str, str]:
    properties = {
        "autolabel_final_label": label,
        "autolabel_labeled_at": datetime.now(timezone.utc).isoformat(),
        "autolabel_labeled_by": labeler_name or _labeler_name(),
        "autolabel_source": context.source,
        "autolabel_queue_key": context.queue_key,
    }
    if context.site_key:
        properties["autolabel_site_key"] = context.site_key
    return properties


LABEL_APP_PROPERTY_KEYS = (
    "autolabel_final_label",
    "autolabel_labeled_at",
    "autolabel_labeled_by",
    "autolabel_source",
    "autolabel_queue_key",
    "autolabel_site_key",
)


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


def _safe_table_slug(value: Any, fallback: str) -> str:
    slug = re.sub(r"\s+", "_", str(value or "").strip())
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug).strip("_.-")
    return slug or fallback


def _camera_id_candidates(raw_folder_name: str, metadata: dict[str, Any] | None = None) -> list[str]:
    metadata = metadata or {}
    candidates: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in candidates:
            candidates.append(text)

    for key in ("camera_id", "camera_name", "triplet_stem", "raw_camera_id"):
        add(metadata.get(key))
    add(raw_folder_name)

    for candidate in list(candidates):
        ipc = re.search(r"IPC[\s_-]*(\d+)", candidate, re.IGNORECASE)
        if ipc:
            add(f"IPC{int(ipc.group(1))}")
            add(f"CH-CH{int(ipc.group(1)):02d}")
        channel = _extract_reolink_channel_code(candidate)
        if channel:
            add(channel)
            number = _extract_reolink_channel_number(channel)
            if number is not None:
                add(f"IPC{number}")
    return candidates


def _supabase_table_label(table_row: dict[str, Any] | None, crop: dict[str, Any], idx: int) -> str:
    source_metadata = crop.get("source_metadata") if isinstance(crop.get("source_metadata"), dict) else {}
    for value in (
        source_metadata.get("original_name"),
        source_metadata.get("label"),
        (table_row or {}).get("host_facing_label"),
        (table_row or {}).get("table_number"),
        (table_row or {}).get("internal_name"),
        crop.get("table_label"),
        crop.get("table_id"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return f"table_{idx + 1}"


def _normalize_supabase_crops_as_camera(
    camera_source: dict[str, Any],
    crops: list[dict[str, Any]],
    table_rows: dict[str, dict[str, Any]],
) -> tuple[int, dict[str, Any], list[tuple[str, list, tuple[int, int, int, int], list]]] | None:
    from processor import bbox_from_polygon

    camera_number = _screenrecord_camera_number_from_metadata(
        "",
        {
            "camera_id": camera_source.get("config", {}).get("edge_camera_id")
            if isinstance(camera_source.get("config"), dict)
            else "",
            "camera_name": camera_source.get("name"),
        },
    )
    if camera_number is None:
        camera_number = 0

    table_polygons: list[tuple[str, list, tuple[int, int, int, int], list]] = []
    table_metadata_by_id: dict[str, dict[str, Any]] = {}
    seen_slugs: set[str] = set()
    image_width = 0
    image_height = 0
    for idx, crop in enumerate(crops):
        raw_polygon = crop.get("polygon")
        if not isinstance(raw_polygon, list) or len(raw_polygon) < 3:
            continue
        polygon = [
            (float(point[0]), float(point[1]))
            for point in raw_polygon
            if isinstance(point, (list, tuple)) and len(point) == 2
        ]
        if len(polygon) < 3:
            continue
        if len(polygon) == 4:
            polygon = _ordered_quadrilateral_points(polygon)

        table_row = table_rows.get(str(crop.get("table_id") or ""))
        label = _supabase_table_label(table_row, crop, idx)
        table_id = _safe_table_slug(label, f"table_{idx + 1}")
        if table_id in seen_slugs:
            table_id = f"{table_id}_{idx + 1}"
        seen_slugs.add(table_id)

        tight_bbox = bbox_from_polygon(polygon)
        table_polygons.append((table_id, polygon, tight_bbox, polygon))
        table_metadata_by_id[table_id] = {
            "label": label,
            "restaurant_id": crop.get("restaurant_id") or (table_row or {}).get("restaurant_id"),
            "table_id": crop.get("table_id"),
            "camera_source_id": crop.get("camera_source_id"),
            "table_camera_crops_id": crop.get("id"),
            "crop_version": crop.get("version"),
            "crop_source": crop.get("source"),
            "frame_width": crop.get("frame_width"),
            "frame_height": crop.get("frame_height"),
        }
        try:
            image_width = max(image_width, int(crop.get("frame_width") or 0))
            image_height = max(image_height, int(crop.get("frame_height") or 0))
        except (TypeError, ValueError):
            pass

    if not table_polygons:
        return None

    camera_payload = {
        "camera_number": camera_number,
        "camera_id": camera_source.get("id"),
        "camera_source_id": camera_source.get("id"),
        "camera_name": camera_source.get("name"),
        "image_width": image_width,
        "image_height": image_height,
        "source": "supabase_table_camera_crops",
        "_table_metadata_by_id": table_metadata_by_id,
    }
    return camera_number, camera_payload, table_polygons


def _supabase_table_rows_by_id(client: SupabaseCropClient, table_ids: list[str]) -> dict[str, dict[str, Any]]:
    ids = [table_id for table_id in table_ids if _is_valid_uuid(table_id)]
    if not ids:
        return {}
    try:
        rows = client.select(
            "tables",
            {
                "select": "id,restaurant_id,table_number,host_facing_label,internal_name,is_active",
                "id": f"in.({','.join(ids)})",
                "limit": str(max(1, len(ids))),
            },
        )
    except Exception:
        return {}
    return {str(row.get("id")): row for row in rows if row.get("id")}


def _supabase_active_crops_for_camera(client: SupabaseCropClient, camera_source_id: str) -> list[dict[str, Any]]:
    cached = _supabase_cached_tables(camera_source_id)
    if cached is not None:
        _set_supabase_crop_status(
            enabled=client.enabled,
            last_lookup_at=_utc_iso_now(),
            last_cache_hit=True,
            last_camera_source_id=camera_source_id,
            last_table_count=len(cached),
        )
        return cached

    try:
        rows = client.select(
            "table_camera_crops",
            {
                "select": "*",
                "camera_source_id": f"eq.{camera_source_id}",
                "is_active": "eq.true",
                "limit": "1000",
            },
        )
    except Exception as exc:
        stale = _supabase_cached_tables(camera_source_id, allow_stale=True)
        _set_supabase_crop_status(
            enabled=client.enabled,
            last_lookup_at=_utc_iso_now(),
            last_cache_hit=stale is not None,
            last_error=str(exc),
            last_camera_source_id=camera_source_id,
            last_table_count=len(stale or []),
        )
        return stale or []

    _store_supabase_cached_tables(camera_source_id, rows)
    _set_supabase_crop_status(
        enabled=client.enabled,
        last_lookup_at=_utc_iso_now(),
        last_cache_hit=False,
        last_error=None,
        last_camera_source_id=camera_source_id,
        last_table_count=len(rows),
    )
    return rows


def _camera_source_matches_site_key(camera_source: dict[str, Any], site_key: str | None) -> bool:
    expected = str(site_key or "").strip()
    if not expected:
        return True
    config = camera_source.get("config") if isinstance(camera_source.get("config"), dict) else {}
    values = {
        str(config.get("edge_node_id") or "").strip(),
        str(config.get("edge_camera_key") or "").strip(),
    }
    return expected in values or any(f"__{expected}__" in value for value in values if value)


def _resolve_supabase_camera_source(
    raw_folder_name: str,
    metadata: dict[str, Any] | None = None,
    *,
    site_key: str | None = None,
) -> dict[str, Any] | None:
    client = _get_supabase_crop_client()
    if not client.enabled:
        _set_supabase_crop_status(enabled=False)
        return None

    metadata = metadata or {}
    explicit_id = str(metadata.get("camera_source_id") or "").strip()
    if _is_valid_uuid(explicit_id):
        try:
            camera = _select_one_supabase(
                client,
                "camera_sources",
                {
                    "select": "id,name,restaurant_id,is_active,config",
                    "id": f"eq.{explicit_id}",
                },
            )
            if camera and camera.get("is_active") is not False and _camera_source_matches_site_key(camera, site_key):
                return camera
        except Exception as exc:
            _set_supabase_crop_status(enabled=True, last_error=str(exc), last_lookup_at=_utc_iso_now())

    site_id = str(metadata.get("site_id") or "").strip()
    node_id = str(metadata.get("node_id") or "").strip() or str(site_key or "").strip()
    for camera_id in _camera_id_candidates(raw_folder_name, metadata):
        params = {
            "select": "id,node_id,restaurant_id,site_id,camera_id,camera_name,camera_source_id,metadata",
            "camera_id": f"eq.{camera_id}",
            "limit": "50",
        }
        if site_id:
            params["site_id"] = f"eq.{site_id}"
        try:
            registry_rows = client.select("edge_camera_registry", params)
        except Exception as exc:
            _set_supabase_crop_status(enabled=True, last_error=str(exc), last_lookup_at=_utc_iso_now())
            registry_rows = []
        if node_id:
            registry_rows = [row for row in registry_rows if str(row.get("node_id") or "") == node_id] or registry_rows
        for row in registry_rows:
            camera_source_id = str(row.get("camera_source_id") or "").strip()
            if not _is_valid_uuid(camera_source_id):
                continue
            try:
                camera = _select_one_supabase(
                    client,
                    "camera_sources",
                    {
                        "select": "id,name,restaurant_id,is_active,config",
                        "id": f"eq.{camera_source_id}",
                    },
                )
            except Exception as exc:
                _set_supabase_crop_status(enabled=True, last_error=str(exc), last_lookup_at=_utc_iso_now())
                continue
            if camera and camera.get("is_active") is not False and _camera_source_matches_site_key(camera, site_key):
                return camera

    restaurant_id = str(metadata.get("restaurant_id") or "").strip()
    if _is_valid_uuid(restaurant_id):
        try:
            candidates = client.select(
                "camera_sources",
                {
                    "select": "id,name,restaurant_id,is_active,config",
                    "restaurant_id": f"eq.{restaurant_id}",
                    "limit": "1000",
                },
            )
        except Exception as exc:
            _set_supabase_crop_status(enabled=True, last_error=str(exc), last_lookup_at=_utc_iso_now())
            candidates = []
        camera_ids = set(_camera_id_candidates(raw_folder_name, metadata))
        for camera in candidates:
            if camera.get("is_active") is False:
                continue
            if not _camera_source_matches_site_key(camera, site_key):
                continue
            config = camera.get("config") if isinstance(camera.get("config"), dict) else {}
            values = {
                str(camera.get("name") or "").strip(),
                str(config.get("edge_camera_id") or "").strip(),
                str(config.get("edge_camera_key") or "").strip(),
            }
            if any(value in camera_ids for value in values if value):
                return camera

    return None


def _resolve_supabase_crop_tables(
    raw_folder_name: str,
    metadata: dict[str, Any] | None = None,
    *,
    site_key: str | None = None,
) -> tuple[int, dict[str, Any], list[tuple[str, list, tuple[int, int, int, int], list]]] | None:
    client = _get_supabase_crop_client()
    if not client.enabled:
        _set_supabase_crop_status(enabled=False)
        return None

    camera_source = _resolve_supabase_camera_source(raw_folder_name, metadata, site_key=site_key)
    if not camera_source or not camera_source.get("id"):
        return None
    camera_source_id = str(camera_source["id"])
    crops = _supabase_active_crops_for_camera(client, camera_source_id)
    if not crops:
        return None
    table_rows = _supabase_table_rows_by_id(
        client,
        [str(crop.get("table_id") or "") for crop in crops],
    )
    return _normalize_supabase_crops_as_camera(camera_source, crops, table_rows)


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
    supabase_match = _resolve_supabase_crop_tables(raw_folder_name, {}, site_key=site_key)
    if supabase_match is not None:
        return supabase_match

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


def _screenrecord_camera_number_from_metadata(raw_folder_name: str, metadata: dict[str, Any]) -> int | None:
    candidates = [
        str(metadata.get("camera_id") or ""),
        str(metadata.get("camera_name") or ""),
        str(metadata.get("triplet_stem") or ""),
        raw_folder_name,
    ]
    for candidate in candidates:
        channel_number = _extract_reolink_channel_number(candidate)
        if channel_number is not None:
            return channel_number
        ipc = re.search(r"IPC[\s_-]*(\d+)", candidate, re.IGNORECASE)
        if ipc:
            return int(ipc.group(1))
    return None


def _screenrecord_channel_code_from_metadata(raw_folder_name: str, metadata: dict[str, Any]) -> str | None:
    for candidate in (
        str(metadata.get("camera_id") or ""),
        str(metadata.get("camera_name") or ""),
        str(metadata.get("triplet_stem") or ""),
        raw_folder_name,
    ):
        channel_code = _extract_reolink_channel_code(candidate)
        if channel_code:
            return channel_code
    camera_number = _screenrecord_camera_number_from_metadata(raw_folder_name, metadata)
    return f"CH-CH{camera_number:02d}" if camera_number is not None else None


def _mapped_camera_tables_for_screenrecord_folder(
    raw_folder_name: str,
    metadata: dict[str, Any],
    *,
    site_key: str | None = None,
    client: DriveClient | None = None,
) -> tuple[int, dict[str, Any], list[tuple[str, list, tuple[int, int, int, int], list]]] | None:
    supabase_match = _resolve_supabase_crop_tables(raw_folder_name, metadata, site_key=site_key)
    if supabase_match is not None:
        return supabase_match

    channel_number = _screenrecord_camera_number_from_metadata(raw_folder_name, metadata)
    if channel_number is None:
        return None

    if _site_uses_manual_crop_configs(site_key):
        if client is None:
            raise RuntimeError("Drive client is required for manual Reolink crop configs.")
        channel_code = _screenrecord_channel_code_from_metadata(raw_folder_name, metadata)
        if not channel_code:
            return None
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
    for input_folder_id in _context_input_folder_ids(context):
        for item in client.list_folders(input_folder_id, fields="id,name,mimeType,parents,appProperties"):
            name = item.get("name")
            if name:
                names.add(str(name))
    for folder_name in LABEL_DESTINATIONS:
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


def _list_screenrecord_true_ten_folders(
    client: DriveClient,
    context: QueueContext,
) -> list[dict[str, Any]]:
    node_folder_id = context.folder_ids.get(SCREENRECORD_TRUE_TEN_NODE_KEY)
    if not node_folder_id:
        return []
    return sorted(
        client.list_folders(
            node_folder_id,
            fields="id,name,mimeType,parents,appProperties,modifiedTime",
        ),
        key=lambda item: str(item.get("name", "")).lower(),
    )


def _screenrecord_output_unlabeled_folder_id(client: DriveClient, context: QueueContext) -> str:
    existing = context.folder_ids.get(SCREENRECORD_THREE_FRAME_UNLABELED_KEY)
    if existing:
        return existing
    node_root_id = context.folder_ids.get("root")
    if not node_root_id:
        return context.input_folder_id
    three_frame_id = client.ensure_subfolder(node_root_id, SCREENRECORD_THREE_FRAME_FOLDER_NAME)
    unlabeled_id = client.ensure_subfolder(three_frame_id, "unlabeled")
    context.folder_ids[SCREENRECORD_THREE_FRAME_UNLABELED_KEY] = unlabeled_id
    return unlabeled_id


def _screenrecord_state_raw_folder(raw_folder: dict[str, Any]) -> dict[str, Any]:
    state = dict(raw_folder)
    state["id"] = f"screenrecord:{raw_folder.get('id') or raw_folder.get('name')}"
    return state


def _camera_table_metadata(camera: dict[str, Any], table_id: str) -> dict[str, Any]:
    metadata_by_id = camera.get("_table_metadata_by_id")
    if isinstance(metadata_by_id, dict):
        value = metadata_by_id.get(table_id)
        if isinstance(value, dict):
            return value
    return {}


def _source_capture_identity(raw_name: str, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    stem = str(metadata.get("triplet_stem") or "").strip()
    triplet_index = metadata.get("triplet_index")
    if stem and triplet_index is not None:
        try:
            return f"{stem}_t{int(triplet_index):04d}"
        except (TypeError, ValueError):
            return f"{stem}_t{triplet_index}"
    for key in ("raw_folder_name", "source_folder_name", "capture_id"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return str(raw_name or "").strip()


def _artifact_identity(raw_name: str, table_metadata: dict[str, Any], metadata: dict[str, Any] | None = None) -> str:
    source = _source_capture_identity(raw_name, metadata)
    table = (
        str(table_metadata.get("table_camera_crops_id") or "").strip()
        or str(table_metadata.get("table_id") or "").strip()
        or str((metadata or {}).get("table_id") or "").strip()
        or str(((metadata or {}).get("table") or {}).get("label") if isinstance((metadata or {}).get("table"), dict) else "").strip()
    )
    return f"{source}|{table}" if source and table else ""


def _metadata_identity(metadata: dict[str, Any]) -> str:
    table_metadata = {
        "table_camera_crops_id": metadata.get("table_camera_crops_id"),
        "table_id": metadata.get("supabase_table_id") or metadata.get("table_id"),
    }
    raw_name = str(metadata.get("raw_folder_name") or metadata.get("triplet_stem") or "").strip()
    return _artifact_identity(raw_name, table_metadata, metadata)


def _existing_generated_artifact_identities(client: DriveClient, context: QueueContext) -> set[str]:
    identities: set[str] = set()
    for input_folder_id in _context_input_folder_ids(context):
        for folder in client.list_folders(input_folder_id, fields="id,name,mimeType,parents,appProperties"):
            metadata_item = client.find_file_by_name(str(folder["id"]), "metadata.json")
            metadata = _load_json_file_from_drive(client, metadata_item)
            if not metadata:
                continue
            identity = _metadata_identity(metadata)
            if identity:
                identities.add(identity)
    return identities


def _record_generated_reolink_artifacts(
    *,
    generated_names: list[str],
    existing_names: set[str],
    existing_identities: set[str],
    raw_name: str,
    table_polygons: list[tuple[str, list, tuple[int, int, int, int], list]],
    camera: dict[str, Any],
    metadata: dict[str, Any] | None,
    label_source: LabelSource,
) -> int:
    generated_name_set = set(generated_names)
    recorded = 0
    for table_id, *_rest in table_polygons:
        unprefixed_name = _derived_reolink_folder_name(raw_name, table_id)
        prefixed_name = _apply_source_prefix(unprefixed_name, label_source)
        if unprefixed_name not in generated_name_set and prefixed_name not in generated_name_set:
            continue
        existing_names.add(unprefixed_name)
        existing_names.add(prefixed_name)
        identity = _artifact_identity(raw_name, _camera_table_metadata(camera, table_id), metadata or {})
        if identity:
            existing_identities.add(identity)
        recorded += 1
    return recorded


def _parse_drive_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _cleanup_processed_raw_folder(
    client: DriveClient,
    folder_id: str,
    *,
    retention_days: int = PROCESSED_RAW_RETENTION_DAYS,
) -> int:
    cutoff = datetime.now(timezone.utc).timestamp() - (retention_days * 24 * 60 * 60)
    cleaned = 0
    for item in client.list_files(
        folder_id,
        fields="id,name,mimeType,parents,modifiedTime,trashed",
    ):
        if item.get("trashed"):
            continue
        modified = _parse_drive_timestamp(item.get("modifiedTime"))
        if modified is None or modified.timestamp() > cutoff:
            continue
        client.trash_file(str(item["id"]))
        cleaned += 1
    return cleaned


def _move_reolink_raw_to_processed(
    client: DriveClient,
    context: QueueContext,
    raw_folder: dict[str, Any],
) -> None:
    processed_id = context.folder_ids.get(PROCESSED_RAW_FOLDER_NAME)
    if not processed_id:
        return
    parents = [str(parent) for parent in raw_folder.get("parents", []) if parent]
    if parents == [processed_id]:
        return
    client.move_file(
        str(raw_folder["id"]),
        new_parent_id=processed_id,
        remove_parent_id=context.seed_folder_id,
    )
    raw_folder["parents"] = [processed_id]


def _reolink_raw_drive_preprocess_status(raw_folder: dict[str, Any]) -> str:
    return str((raw_folder.get("appProperties") or {}).get("autolabel_preprocess_status") or "")


def _stamp_reolink_raw_preprocess_status(
    client: DriveClient,
    context: QueueContext,
    raw_folder: dict[str, Any],
    *,
    status: str,
    generated: int = 0,
    reason: str = "",
) -> None:
    metadata = dict(raw_folder.get("appProperties") or {})
    metadata.update(
        {
            "autolabel_preprocess_status": status,
            "autolabel_preprocess_site_key": context.site_key or "",
            "autolabel_preprocess_queue_key": context.queue_key,
            "autolabel_preprocess_generated": str(int(generated)),
            "autolabel_preprocess_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if reason:
        metadata["autolabel_preprocess_reason"] = reason[:1200]
    try:
        updated = client.update_file_metadata(
            str(raw_folder["id"]),
            {"appProperties": metadata},
            fields="id,name,mimeType,parents,appProperties",
        )
        raw_folder["appProperties"] = dict(updated.get("appProperties") or metadata)
    except DriveClientError:
        return


def _preprocess_state_dir() -> Path:
    configured = os.environ.get("PREPROCESS_STATE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    if os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("RAILWAY_PROJECT_ID"):
        return Path("/data/autolabeler")
    return Path(tempfile.gettempdir()) / "AutoLabeler" / "preprocess_state"


def _preprocess_state_path() -> Path:
    return _preprocess_state_dir() / PREPROCESS_STATE_FILE_NAME


def _label_history_path() -> Path:
    return _preprocess_state_dir() / LABEL_HISTORY_FILE_NAME


def _label_jobs_path() -> Path:
    return _preprocess_state_dir() / LABEL_JOBS_FILE_NAME


def _supabase_crop_cache_path() -> Path:
    return _preprocess_state_dir() / SUPABASE_CROP_CACHE_FILE_NAME


@contextmanager
def _state_file_lock(lock_name: str):
    lock_path = _preprocess_state_dir() / f"{lock_name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    locked = False
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            locked = True
        except (ImportError, OSError):
            locked = False
        yield
    finally:
        if locked:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
        handle.close()


def _state_tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_valid_uuid(value: Any) -> bool:
    try:
        import uuid

        uuid.UUID(str(value or ""))
        return True
    except (TypeError, ValueError):
        return False


def _supabase_rest_url_from_database_url(database_url: str) -> str:
    value = database_url.strip()
    if not value:
        return ""
    if value.startswith(("http://", "https://")):
        return value.rstrip("/")

    try:
        parsed = urlparse(value)
    except ValueError:
        return ""

    host = parsed.hostname or ""
    project_ref = ""
    if host.startswith("db.") and host.endswith(".supabase.co"):
        project_ref = host.split(".", 2)[1]
    elif host.endswith(".pooler.supabase.com") and parsed.username:
        username = unquote(parsed.username)
        if "." in username:
            project_ref = username.rsplit(".", 1)[-1]

    if not project_ref:
        return ""
    return f"https://{project_ref}.supabase.co"


def _supabase_rest_config() -> tuple[str, str, str]:
    configured_url = (
        os.environ.get("SUPABASE_URL", "").strip()
        or _supabase_rest_url_from_database_url(os.environ.get("DATABASE_URL", ""))
        or _supabase_rest_url_from_database_url(os.environ.get("DB_URL", ""))
    )
    url = configured_url.rstrip("/")
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.environ.get("SUPABASE_SECRET_KEY", "").strip()
        or os.environ.get("DATABASE_SERVICE_ROLE_KEY", "").strip()
        or os.environ.get("DB_SERVICE_ROLE_KEY", "").strip()
    )
    schema = os.environ.get("SUPABASE_DB_SCHEMA", "public").strip() or "public"
    return url, key, schema


def _supabase_crop_client_configured() -> bool:
    url, key, _schema = _supabase_rest_config()
    return bool(url and key)


class SupabaseCropClient:
    """Tiny PostgREST reader for the crop tables ScreenRecord already uses."""

    def __init__(
        self,
        *,
        url: str | None = None,
        key: str | None = None,
        schema: str | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        configured_url, configured_key, configured_schema = _supabase_rest_config()
        self.url = (url if url is not None else configured_url).strip().rstrip("/")
        self.key = (key if key is not None else configured_key).strip()
        self.schema = (schema if schema is not None else configured_schema).strip() or "public"
        self.timeout_seconds = timeout_seconds

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.key)

    def select(self, table: str, params: dict[str, str]) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        import httpx

        endpoint = f"{self.url}/rest/v1/{table}"
        headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Accept": "application/json",
            "Accept-Profile": self.schema,
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(endpoint, params=params, headers=headers)
            response.raise_for_status()
            payload = response.json()
        return payload if isinstance(payload, list) else []


def _get_supabase_crop_client() -> SupabaseCropClient:
    return SupabaseCropClient()


def _set_supabase_crop_status(**updates: Any) -> None:
    with _supabase_crop_cache_lock:
        _supabase_crop_status.update(updates)


def _supabase_crop_status_snapshot() -> dict[str, Any]:
    with _supabase_crop_cache_lock:
        return dict(_supabase_crop_status)


def _load_supabase_crop_cache_file() -> dict[str, Any]:
    path = _supabase_crop_cache_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_supabase_crop_cache_file(cache: dict[str, Any]) -> None:
    path = _supabase_crop_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _state_tmp_path(path)
    tmp_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _supabase_cached_tables(camera_source_id: str, *, allow_stale: bool = False) -> list[dict[str, Any]] | None:
    now = time.time()
    with _supabase_crop_cache_lock:
        entry = _supabase_crop_cache.get(camera_source_id)
        if entry:
            expires_at = float(entry.get("expires_at") or 0.0)
            if allow_stale or expires_at > now:
                tables = entry.get("tables")
                if isinstance(tables, list):
                    return [dict(item) for item in tables if isinstance(item, dict)]

    file_cache = _load_supabase_crop_cache_file()
    entry = file_cache.get(camera_source_id)
    if not isinstance(entry, dict):
        return None
    expires_at = float(entry.get("expires_at") or 0.0)
    if not allow_stale and expires_at <= now:
        return None
    tables = entry.get("tables")
    if not isinstance(tables, list):
        return None
    with _supabase_crop_cache_lock:
        _supabase_crop_cache[camera_source_id] = entry
    return [dict(item) for item in tables if isinstance(item, dict)]


def _store_supabase_cached_tables(camera_source_id: str, tables: list[dict[str, Any]]) -> None:
    entry = {
        "camera_source_id": camera_source_id,
        "cached_at": _utc_iso_now(),
        "expires_at": time.time() + SUPABASE_CROP_CACHE_TTL_SECONDS,
        "tables": tables,
    }
    with _supabase_crop_cache_lock:
        _supabase_crop_cache[camera_source_id] = entry
    with _state_file_lock("supabase_crop_cache"):
        cache = _load_supabase_crop_cache_file()
        cache[camera_source_id] = entry
        _save_supabase_crop_cache_file(cache)


def _select_one_supabase(client: SupabaseCropClient, table: str, params: dict[str, str]) -> dict[str, Any] | None:
    rows = client.select(table, {**params, "limit": params.get("limit", "1")})
    return rows[0] if rows else None


def _cache_warm_shared_lock_path() -> Path:
    return _preprocess_state_dir() / "cache_warm.lock"


def _ready_maintainer_shared_lock_path() -> Path:
    return _preprocess_state_dir() / "ready_maintainer.lock"


def _read_cache_warm_shared_lock(path: Path | None = None) -> dict[str, Any] | None:
    lock_path = path or _cache_warm_shared_lock_path()
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _acquire_cache_warm_shared_lock() -> dict[str, Any] | None:
    lock_path = _cache_warm_shared_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()

    existing = _read_cache_warm_shared_lock(lock_path)
    if existing is not None:
        started_at = float(existing.get("started_at_epoch") or 0)
        if started_at and (now - started_at) > CACHE_WARM_LOCK_STALE_SECONDS:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                return None

    token = {
        "pid": os.getpid(),
        "started_at_epoch": now,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(str(lock_path), flags)
    except FileExistsError:
        return None
    except OSError:
        return None

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(token, sort_keys=True))
    except OSError:
        try:
            lock_path.unlink()
        except OSError:
            pass
        return None
    return token


def _release_cache_warm_shared_lock(token: dict[str, Any] | None) -> None:
    if not token:
        return
    lock_path = _cache_warm_shared_lock_path()
    existing = _read_cache_warm_shared_lock(lock_path)
    if existing and existing.get("started_at_epoch") != token.get("started_at_epoch"):
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _read_ready_maintainer_shared_lock(path: Path | None = None) -> dict[str, Any] | None:
    lock_path = path or _ready_maintainer_shared_lock_path()
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _acquire_ready_maintainer_shared_lock() -> dict[str, Any] | None:
    lock_path = _ready_maintainer_shared_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()

    existing = _read_ready_maintainer_shared_lock(lock_path)
    if existing is not None:
        started_at = float(existing.get("started_at_epoch") or 0)
        if started_at and (now - started_at) > READY_MAINTAINER_LOCK_STALE_SECONDS:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                return None

    token = {
        "pid": os.getpid(),
        "started_at_epoch": now,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        return None
    except OSError:
        return None

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(token, sort_keys=True))
    except OSError:
        try:
            lock_path.unlink()
        except OSError:
            pass
        return None
    return token


def _release_ready_maintainer_shared_lock(token: dict[str, Any] | None) -> None:
    if not token:
        return
    lock_path = _ready_maintainer_shared_lock_path()
    existing = _read_ready_maintainer_shared_lock(lock_path)
    if existing and existing.get("started_at_epoch") != token.get("started_at_epoch"):
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _label_history_empty() -> dict[str, Any]:
    return {
        "schema_version": LABEL_HISTORY_SCHEMA_VERSION,
        "queues": {},
    }


def _label_jobs_empty() -> dict[str, Any]:
    return {
        "schema_version": LABEL_JOBS_SCHEMA_VERSION,
        "jobs": {},
    }


def _load_label_history_unlocked() -> dict[str, Any]:
    path = _label_history_path()
    if not path.exists():
        return _label_history_empty()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _label_history_empty()
    if not isinstance(data, dict):
        return _label_history_empty()
    queues = data.get("queues")
    if not isinstance(queues, dict):
        queues = {}
    return {
        "schema_version": LABEL_HISTORY_SCHEMA_VERSION,
        "queues": queues,
    }


def _save_label_history_unlocked(history: dict[str, Any]) -> None:
    path = _label_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": LABEL_HISTORY_SCHEMA_VERSION,
        "queues": history.get("queues") or {},
    }
    tmp_path = _state_tmp_path(path)
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _frame_signature_from_frames(frames: dict[str, str | None]) -> str:
    keys = _ordered_frame_keys(frames)
    return "|".join(str(frames.get(key) or "") for key in keys)


def _content_signature_from_frames(frames: dict[str, str | None]) -> str:
    parts: list[str] = []
    keys = _ordered_frame_keys(frames)
    if not keys:
        return ""
    for key in keys:
        file_id = frames.get(key)
        if not file_id:
            return ""
        thumb_path = _thumb_path_for_file(str(file_id))
        if not thumb_path.exists():
            return ""
        try:
            digest = hashlib.sha256(thumb_path.read_bytes()).hexdigest()
        except OSError:
            return ""
        parts.append(digest)
    return "|".join(parts)


def _label_history_keys(
    context: QueueContext,
    folder_id: str,
    folder_name: str,
    frame_signature: str,
    content_signature: str = "",
) -> list[str]:
    keys = []
    if folder_id:
        keys.append(f"id:{folder_id}")
    if folder_name:
        keys.append(f"name:{folder_name}")
    if frame_signature:
        keys.append(f"frames:{frame_signature}")
    if content_signature:
        keys.append(f"thumbs:{content_signature}")
    return keys


def _label_history_queue(history: dict[str, Any], queue_key: str) -> dict[str, Any]:
    queues = history.setdefault("queues", {})
    queue = queues.setdefault(queue_key, {})
    if not isinstance(queue, dict):
        queue = {}
        queues[queue_key] = queue
    labeled = queue.setdefault("labeled", {})
    if not isinstance(labeled, dict):
        queue["labeled"] = {}
    return queue


def _label_history_records_for_queue(queue_key: str) -> dict[str, Any]:
    with _label_history_lock:
        with _state_file_lock("label_history"):
            history = _load_label_history_unlocked()
            queue = (history.get("queues") or {}).get(queue_key) or {}
            labeled = queue.get("labeled") or {}
            return dict(labeled) if isinstance(labeled, dict) else {}


def _label_history_lookup_in_records(
    labeled_records: dict[str, Any],
    context: QueueContext,
    folder_id: str,
    folder_name: str,
    frame_signature: str,
    content_signature: str = "",
) -> dict[str, Any] | None:
    for key in _label_history_keys(context, folder_id, folder_name, frame_signature, content_signature):
        record = labeled_records.get(key)
        if isinstance(record, dict):
            return record
    return None


def _label_history_lookup(
    context: QueueContext,
    folder_id: str,
    folder_name: str,
    frame_signature: str,
    content_signature: str = "",
) -> dict[str, Any] | None:
    labeled_records = _label_history_records_for_queue(context.queue_key)
    return _label_history_lookup_in_records(
        labeled_records,
        context,
        folder_id,
        folder_name,
        frame_signature,
        content_signature,
    )


def _record_label_history(
    context: QueueContext,
    folder_id: str,
    folder_name: str,
    frame_signature: str,
    label: str,
    content_signature: str = "",
) -> None:
    record = {
        "label": label,
        "folder_id": folder_id,
        "folder_name": folder_name,
        "frame_signature": frame_signature,
        "content_signature": content_signature,
        "queue_key": context.queue_key,
        "source": context.source,
        "site_key": context.site_key,
        "labeled_at": datetime.now(timezone.utc).isoformat(),
        "labeled_by": _labeler_name(),
    }
    with _label_history_lock:
        with _state_file_lock("label_history"):
            history = _load_label_history_unlocked()
            queue = _label_history_queue(history, context.queue_key)
            labeled = queue.setdefault("labeled", {})
            for key in _label_history_keys(context, folder_id, folder_name, frame_signature, content_signature):
                labeled[key] = record
            _save_label_history_unlocked(history)


def _remove_label_history(
    context: QueueContext,
    folder_id: str,
    folder_name: str,
    frame_signature: str,
    content_signature: str = "",
) -> None:
    with _label_history_lock:
        with _state_file_lock("label_history"):
            history = _load_label_history_unlocked()
            queue = (history.get("queues") or {}).get(context.queue_key) or {}
            labeled = queue.get("labeled")
            if not isinstance(labeled, dict):
                return
            for key in _label_history_keys(context, folder_id, folder_name, frame_signature, content_signature):
                labeled.pop(key, None)
            _save_label_history_unlocked(history)


def _load_label_jobs_unlocked() -> dict[str, Any]:
    path = _label_jobs_path()
    if not path.exists():
        return _label_jobs_empty()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _label_jobs_empty()
    if not isinstance(data, dict):
        return _label_jobs_empty()
    jobs = data.get("jobs")
    if not isinstance(jobs, dict):
        jobs = {}
    return {
        "schema_version": LABEL_JOBS_SCHEMA_VERSION,
        "jobs": jobs,
    }


def _save_label_jobs_unlocked(state: dict[str, Any]) -> None:
    path = _label_jobs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": LABEL_JOBS_SCHEMA_VERSION,
        "jobs": state.get("jobs") or {},
    }
    tmp_path = _state_tmp_path(path)
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _label_job_key(context: QueueContext, folder_id: str) -> str:
    return f"{context.queue_key}:{folder_id}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso(dt: datetime | None = None) -> str:
    return (dt or _utc_now()).isoformat()


def _parse_iso_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _label_job_due_at(label: str) -> datetime:
    return _utc_now() + timedelta(seconds=LABEL_JOB_UNDO_SECONDS)


def _label_job_is_due(job: dict[str, Any], now: datetime | None = None) -> bool:
    due_at = _parse_iso_datetime(job.get("not_before"))
    if due_at is None:
        return True
    return due_at <= (now or _utc_now())


def _enqueue_label_job(
    context: QueueContext,
    *,
    folder_id: str,
    parent_id: str,
    folder_name: str,
    frames: dict[str, str | None],
    frame_signature: str,
    content_signature: str,
    label: str,
) -> dict[str, Any]:
    now = _utc_iso()
    not_before = _utc_iso(_label_job_due_at(label))
    job_id = _label_job_key(context, folder_id)
    with _label_jobs_lock:
        with _state_file_lock("label_jobs"):
            state = _load_label_jobs_unlocked()
            jobs = state.setdefault("jobs", {})
            existing = jobs.get(job_id)
            existing_job = existing if isinstance(existing, dict) else {}
            if existing_job.get("status") == "succeeded":
                return dict(existing_job)
            job = {
                "id": job_id,
                "status": "pending",
                "attempts": 0,
                "folder_id": folder_id,
                "parent_id": parent_id,
                "folder_name": folder_name,
                "frames": frames,
                "frame_signature": frame_signature,
                "content_signature": content_signature,
                "label": label,
                "source": context.source,
                "site_key": context.site_key,
                "queue_key": context.queue_key,
                "labeler_name": _labeler_name(),
                "created_at": existing_job.get("created_at", now),
                "updated_at": now,
                "not_before": not_before,
                "undo_expires_at": not_before,
                "last_error": None,
            }
            jobs[job_id] = job
            _save_label_jobs_unlocked(state)
            return dict(job)


def _get_label_job(job_id: str) -> dict[str, Any] | None:
    with _label_jobs_lock:
        with _state_file_lock("label_jobs"):
            state = _load_label_jobs_unlocked()
            job = (state.get("jobs") or {}).get(job_id)
            return dict(job) if isinstance(job, dict) else None


def _clear_label_queue_caches(context: QueueContext, folder_id: str) -> None:
    _invalidate_listing_cache(context.queue_key)
    with _hydrated_folder_cache_lock:
        _hydrated_folder_cache.pop(_hydrated_cache_key(context.queue_key, folder_id), None)


def _label_destination_parent_ids(context: QueueContext) -> dict[str, str]:
    return {
        str(destination_id): destination_label
        for destination_label, destination_id in context.folder_ids.items()
        if destination_label in LABEL_DESTINATIONS and destination_id
    }


def _clear_drive_label_metadata(client: DriveClient, folder_id: str, current: dict[str, Any]) -> None:
    app_properties = dict(current.get("appProperties") or {})
    clear_properties = {
        key: None
        for key in LABEL_APP_PROPERTY_KEYS
        if key in app_properties
    }
    if clear_properties:
        client.update_file_metadata(
            folder_id,
            {"appProperties": clear_properties},
            fields="id,name,mimeType,parents,appProperties",
        )


def _mark_label_job_canceled(
    context: QueueContext,
    folder_id: str,
    *,
    note: str | None = None,
) -> None:
    job_id = _label_job_key(context, folder_id)
    with _label_jobs_lock:
        with _state_file_lock("label_jobs"):
            state = _load_label_jobs_unlocked()
            jobs = state.setdefault("jobs", {})
            job = jobs.get(job_id)
            if not isinstance(job, dict):
                job = {
                    "id": job_id,
                    "folder_id": folder_id,
                    "source": context.source,
                    "site_key": context.site_key,
                    "queue_key": context.queue_key,
                }
                jobs[job_id] = job
            job["status"] = "canceled"
            job["updated_at"] = _utc_iso()
            job["last_error"] = note
            _save_label_jobs_unlocked(state)


def _cancel_label_job(
    context: QueueContext,
    *,
    client: DriveClient,
    folder_id: str,
    parent_id: str,
    folder_name: str,
    frame_signature: str,
    content_signature: str,
) -> dict[str, Any]:
    job_id = _label_job_key(context, folder_id)
    canceled = False
    with _label_jobs_lock:
        with _state_file_lock("label_jobs"):
            state = _load_label_jobs_unlocked()
            job = (state.get("jobs") or {}).get(job_id)
            if isinstance(job, dict) and job.get("status") == "pending" and not _label_job_is_due(job):
                job["status"] = "canceled"
                job["updated_at"] = _utc_iso()
                job["last_error"] = None
                _save_label_jobs_unlocked(state)
                canceled = True
    if canceled:
        _remove_label_history(context, folder_id, folder_name, frame_signature, content_signature)
        _clear_label_queue_caches(context, folder_id)
        return {"canceled": True, "restored": False}

    input_parent_ids = _context_input_folder_ids(context)
    restore_parent_id = parent_id if parent_id in input_parent_ids else context.input_folder_id
    current = client.get_file(folder_id, fields="id,name,parents,appProperties")
    current_parents = [str(parent) for parent in current.get("parents", []) if parent]
    label_parent_ids = _label_destination_parent_ids(context)
    current_label_parent = next(
        (parent for parent in current_parents if parent in label_parent_ids),
        None,
    )

    if restore_parent_id in current_parents:
        restored = True
    elif current_label_parent:
        client.move_file(folder_id, new_parent_id=restore_parent_id, remove_parent_id=current_label_parent)
        restored = True
    else:
        return {"canceled": False, "restored": False}

    _clear_drive_label_metadata(client, folder_id, current)
    _mark_label_job_canceled(
        context,
        folder_id,
        note="Undo restored folder from Drive label destination." if current_label_parent else None,
    )
    _remove_label_history(context, folder_id, folder_name, frame_signature, content_signature)
    _clear_label_queue_caches(context, folder_id)
    return {"canceled": True, "restored": restored}


def _label_job_rate_limit_snapshot() -> dict[str, Any]:
    with _label_job_rate_limit_lock:
        cooldown_until = _label_job_rate_limit_cooldown_until
        cooldown_seconds = 0.0
        if cooldown_until is not None:
            cooldown_seconds = max(0.0, (cooldown_until - _utc_now()).total_seconds())
        return {
            "rate_limit_cooldown_until": _utc_iso(cooldown_until) if cooldown_until else None,
            "rate_limit_cooldown_seconds": cooldown_seconds,
            "last_rate_limit_error": _label_job_last_rate_limit_error,
            "last_move_attempt_at": _utc_iso(_label_job_last_attempt_at) if _label_job_last_attempt_at else None,
            "label_job_min_interval_seconds": LABEL_JOB_MIN_INTERVAL_SECONDS,
        }


def _label_job_rate_limit_delay_seconds() -> float:
    with _label_job_rate_limit_lock:
        if _label_job_rate_limit_cooldown_until is None:
            return 0.0
        return max(0.0, (_label_job_rate_limit_cooldown_until - _utc_now()).total_seconds())


def _mark_label_job_attempt() -> None:
    global _label_job_last_attempt_at
    with _label_job_rate_limit_lock:
        _label_job_last_attempt_at = _utc_now()


def _clear_label_job_rate_limit_cooldown() -> None:
    global _label_job_rate_limit_cooldown_until
    with _label_job_rate_limit_lock:
        if _label_job_rate_limit_cooldown_until and _label_job_rate_limit_cooldown_until <= _utc_now():
            _label_job_rate_limit_cooldown_until = None


def _looks_like_drive_rate_limit_error(error: object) -> bool:
    text = str(error or "").lower()
    return any(
        marker in text
        for marker in (
            "429",
            "too many requests",
            "ratelimitexceeded",
            "user rate limit exceeded",
            "userratelimitexceeded",
            "quota",
        )
    )


def _record_label_job_rate_limit(error: object) -> None:
    global _label_job_rate_limit_cooldown_until, _label_job_rate_limit_cooldown_seconds
    global _label_job_last_rate_limit_error
    message = str(error or "Drive rate limit")
    with _label_job_rate_limit_lock:
        now = _utc_now()
        cooldown = _label_job_rate_limit_cooldown_seconds
        _label_job_rate_limit_cooldown_until = now + timedelta(seconds=cooldown)
        _label_job_last_rate_limit_error = message[:1200]
        _label_job_rate_limit_cooldown_seconds = min(
            LABEL_JOB_RATE_LIMIT_MAX_COOLDOWN_SECONDS,
            max(LABEL_JOB_RATE_LIMIT_COOLDOWN_SECONDS, cooldown * 2),
        )


def _record_label_job_success() -> None:
    global _label_job_rate_limit_cooldown_until, _label_job_rate_limit_cooldown_seconds
    global _label_job_last_rate_limit_error
    with _label_job_rate_limit_lock:
        _label_job_rate_limit_cooldown_until = None
        _label_job_rate_limit_cooldown_seconds = LABEL_JOB_RATE_LIMIT_COOLDOWN_SECONDS
        _label_job_last_rate_limit_error = None


def _label_job_pace_delay_seconds() -> float:
    with _label_job_rate_limit_lock:
        last_attempt_at = _label_job_last_attempt_at
    if last_attempt_at is None or LABEL_JOB_MIN_INTERVAL_SECONDS <= 0:
        return 0.0
    elapsed = (_utc_now() - last_attempt_at).total_seconds()
    return max(0.0, LABEL_JOB_MIN_INTERVAL_SECONDS - elapsed)


def _label_job_destination_context(
    client: DriveClient,
    job: dict[str, Any],
) -> tuple[QueueContext, str, str, str]:
    source = str(job.get("source") or VIDEO_SOURCE)
    site_key = str(job.get("site_key") or "").strip() or None
    context = _resolve_queue_context(client, source, site_key)
    folder_id = str(job.get("folder_id") or "")
    label = str(job.get("label") or "").lower()
    destination_id = context.folder_ids.get(label, "")
    if not folder_id or label not in LABEL_DESTINATIONS or not destination_id:
        raise ValueError("label job is missing folder_id or label destination")
    return context, folder_id, label, destination_id


def _verify_succeeded_label_jobs(client: DriveClient) -> dict[str, Any]:
    with _label_jobs_lock:
        with _state_file_lock("label_jobs"):
            state = _load_label_jobs_unlocked()
            succeeded_jobs = [
                dict(job)
                for job in (state.get("jobs") or {}).values()
                if isinstance(job, dict) and job.get("status") == "succeeded"
            ]

    checked = 0
    mismatch_count = 0
    reopened_count = 0
    errors: list[dict[str, Any]] = []
    for job in succeeded_jobs:
        checked += 1
        job_id = str(job.get("id") or "")
        try:
            context, folder_id, label, destination_id = _label_job_destination_context(client, job)
            current = client.get_file(folder_id, fields="id,name,parents,appProperties")
            current_parents = [str(parent) for parent in current.get("parents", []) if parent]
            label_parent_ids = {
                destination_id
                for destination_label, destination_id in context.folder_ids.items()
                if destination_label in LABEL_DESTINATIONS
            }
            is_mismatch = (
                destination_id not in current_parents
                and (
                    context.input_folder_id in current_parents
                    or any(parent in label_parent_ids for parent in current_parents)
                )
            )
        except Exception as exc:
            errors.append({"id": job_id, "error": str(exc)})
            continue
        if not is_mismatch:
            continue
        mismatch_count += 1
        with _label_jobs_lock:
            with _state_file_lock("label_jobs"):
                state = _load_label_jobs_unlocked()
                live_job = (state.get("jobs") or {}).get(job_id)
                if isinstance(live_job, dict) and live_job.get("status") == "succeeded":
                    now = _utc_iso()
                    live_job["status"] = "pending"
                    live_job["updated_at"] = now
                    live_job["not_before"] = now
                    live_job["undo_expires_at"] = now
                    live_job["last_error"] = (
                        f"Verification reopened job: Drive folder is not in '{label}' destination."
                    )
                    _save_label_jobs_unlocked(state)
                    reopened_count += 1
    if reopened_count:
        _schedule_label_job_worker()
    return {
        "checked": checked,
        "verified_mismatch_count": mismatch_count,
        "reopened_count": reopened_count,
        "errors": errors[:LABEL_JOB_ERROR_LIMIT],
    }


def _label_jobs_status_payload(
    *,
    verify: bool = False,
    client: DriveClient | None = None,
) -> dict[str, Any]:
    verification = None
    if verify:
        verification = _verify_succeeded_label_jobs(client or DriveClient())

    with _label_jobs_lock:
        with _state_file_lock("label_jobs"):
            state = _load_label_jobs_unlocked()
            stale_reset_count = _reset_stale_label_jobs_unlocked(state)
            recoverable_failed_reset_count = _reset_recoverable_failed_label_jobs_unlocked(state)
            jobs = [job for job in (state.get("jobs") or {}).values() if isinstance(job, dict)]

    counts = {"pending": 0, "delayed": 0, "processing": 0, "succeeded": 0, "failed": 0, "canceled": 0}
    recent_errors: list[dict[str, Any]] = []
    last_success_at = None
    now = _utc_now()
    next_due_at = None
    for job in jobs:
        status = str(job.get("status") or "pending")
        if status not in counts:
            status = "pending"
        if status == "pending" and not _label_job_is_due(job, now):
            counts["delayed"] += 1
            due_value = job.get("not_before")
            if isinstance(due_value, str) and (next_due_at is None or due_value < next_due_at):
                next_due_at = due_value
        else:
            counts[status] += 1
        if status == "succeeded":
            updated_at = job.get("updated_at")
            if isinstance(updated_at, str) and (last_success_at is None or updated_at > last_success_at):
                last_success_at = updated_at
        if job.get("last_error") and len(recent_errors) < LABEL_JOB_ERROR_LIMIT:
            recent_errors.append(
                {
                    "id": job.get("id"),
                    "folder_name": job.get("folder_name"),
                    "attempts": job.get("attempts"),
                    "error": job.get("last_error"),
                }
            )

    path = _label_jobs_path()
    writable = False
    error = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        probe = path.parent / ".label_jobs_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        writable = True
    except OSError as exc:
        error = str(exc)

    return {
        "path": str(path),
        "writable": writable,
        "inflight": _label_job_worker_inflight,
        "counts": counts,
        "confirmed_moved": counts["succeeded"],
        "waiting_to_move": counts["pending"] + counts["delayed"] + counts["processing"],
        "active_move_attempts": counts["processing"],
        "last_success_at": last_success_at,
        "next_due_at": next_due_at,
        "undo_seconds": LABEL_JOB_UNDO_SECONDS,
        "stale_processing_seconds": LABEL_JOB_PROCESSING_STALE_SECONDS,
        "stale_reset_count": stale_reset_count,
        "recoverable_failed_reset_count": recoverable_failed_reset_count,
        "recent_errors": recent_errors,
        "verification": verification,
        **_label_job_rate_limit_snapshot(),
        "error": error,
    }


def _reset_stale_label_jobs_unlocked(state: dict[str, Any]) -> int:
    now = _utc_now()
    reset_count = 0
    for job in (state.get("jobs") or {}).values():
        if not isinstance(job, dict) or job.get("status") != "processing":
            continue
        updated_at = _parse_iso_datetime(job.get("updated_at"))
        if updated_at is None or (now - updated_at).total_seconds() >= LABEL_JOB_PROCESSING_STALE_SECONDS:
            job["status"] = "pending"
            job["updated_at"] = _utc_iso(now)
            job["last_error"] = "Recovered stale processing job after worker restart."
            reset_count += 1
    if reset_count:
        _save_label_jobs_unlocked(state)
    return reset_count


def _recoverable_label_job_error(message: object) -> bool:
    text = str(message or "")
    return "Working outside of request context" in text


def _reset_recoverable_failed_label_jobs_unlocked(
    state: dict[str, Any],
    *,
    limit: int | None = None,
) -> int:
    reset_count = 0
    for job in (state.get("jobs") or {}).values():
        if limit is not None and reset_count >= limit:
            break
        if not isinstance(job, dict) or job.get("status") != "failed":
            continue
        if not _recoverable_label_job_error(job.get("last_error")):
            continue
        job["status"] = "pending"
        job["attempts"] = 0
        job["updated_at"] = _utc_iso()
        job["last_error"] = LABEL_JOB_RECOVERED_ERROR
        reset_count += 1
    if reset_count:
        _save_label_jobs_unlocked(state)
    return reset_count


def _next_due_label_job_at_unlocked(state: dict[str, Any]) -> datetime | None:
    next_due = None
    for job in (state.get("jobs") or {}).values():
        if not isinstance(job, dict) or job.get("status") != "pending":
            continue
        due_at = _parse_iso_datetime(job.get("not_before")) or _utc_now()
        if next_due is None or due_at < next_due:
            next_due = due_at
    return next_due


def _claim_next_label_job_unlocked(state: dict[str, Any], *, force_due: bool = False) -> dict[str, Any] | None:
    now_dt = _utc_now()
    now = _utc_iso(now_dt)
    jobs = state.setdefault("jobs", {})
    _reset_stale_label_jobs_unlocked(state)
    _reset_recoverable_failed_label_jobs_unlocked(state, limit=1)
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        status = str(job.get("status") or "pending")
        attempts = int(job.get("attempts") or 0)
        is_retryable_failed = status == "failed" and attempts < LABEL_JOB_MAX_ATTEMPTS
        if (status == "pending" or is_retryable_failed) and (force_due or _label_job_is_due(job, now_dt)):
            job["status"] = "processing"
            job["attempts"] = attempts + 1
            job["updated_at"] = now
            _save_label_jobs_unlocked(state)
            return dict(job)
    return None


def _finish_label_job(job_id: str, *, status: str, error: str | None = None) -> None:
    now = _utc_iso()
    with _label_jobs_lock:
        with _state_file_lock("label_jobs"):
            state = _load_label_jobs_unlocked()
            job = (state.get("jobs") or {}).get(job_id)
            if not isinstance(job, dict):
                return
            attempts = int(job.get("attempts") or 0)
            job["status"] = status
            job["updated_at"] = now
            job["last_error"] = error
            if error and attempts >= LABEL_JOB_MAX_ATTEMPTS:
                job["status"] = "failed"
            elif error:
                job["status"] = "pending"
            _save_label_jobs_unlocked(state)


def _push_label_job_to_drive(client: DriveClient, job: dict[str, Any]) -> None:
    context, folder_id, label, destination_id = _label_job_destination_context(client, job)
    parent_id = str(job.get("parent_id") or "")
    if not folder_id or not parent_id:
        raise ValueError("label job is missing folder_id, parent_id, or label")
    if parent_id not in _context_input_folder_ids(context):
        raise ValueError("label job parent_id does not match the active queue")

    current = client.get_file(folder_id, fields="id,name,parents,appProperties")
    current_parents = [str(parent) for parent in current.get("parents", []) if parent]
    label_parent_ids = {
        destination_label: destination_id
        for destination_label, destination_id in context.folder_ids.items()
        if destination_label in LABEL_DESTINATIONS
    }
    current_label_parent = next(
        (parent for parent in current_parents if parent in label_parent_ids.values()),
        None,
    )

    if destination_id in current_parents:
        pass
    elif parent_id in current_parents:
        client.move_file(folder_id, new_parent_id=destination_id, remove_parent_id=parent_id)
    elif current_label_parent:
        client.move_file(folder_id, new_parent_id=destination_id, remove_parent_id=current_label_parent)
    else:
        raise RuntimeError("folder is no longer in the source or target Drive folder")

    label_metadata = dict(current.get("appProperties") or {})
    label_metadata.update(
        _label_app_properties(label, context, labeler_name=str(job.get("labeler_name") or "background"))
    )
    client.update_file_metadata(
        folder_id,
        {"appProperties": label_metadata},
        fields="id,name,mimeType,parents,appProperties",
    )
    with _hydrated_folder_cache_lock:
        _hydrated_folder_cache.pop(_hydrated_cache_key(context.queue_key, folder_id), None)
    _remove_folder_from_listing_cache(context.queue_key, folder_id)


def _drain_label_jobs_once(client: DriveClient | None = None, *, force_due: bool = False) -> int:
    active_client = client
    if not force_due:
        _clear_label_job_rate_limit_cooldown()
        if _label_job_rate_limit_delay_seconds() > 0:
            return 0
    with _label_jobs_lock:
        with _state_file_lock("label_jobs"):
            state = _load_label_jobs_unlocked()
            job = _claim_next_label_job_unlocked(state, force_due=force_due)
    if job is None:
        return 0
    job_id = str(job.get("id") or "")
    try:
        if active_client is None:
            active_client = DriveClient()
        _mark_label_job_attempt()
        _push_label_job_to_drive(active_client, job)
    except Exception as exc:
        if _looks_like_drive_rate_limit_error(exc):
            _record_label_job_rate_limit(exc)
        _finish_label_job(job_id, status="pending", error=str(exc))
        return 0
    _record_label_job_success()
    _finish_label_job(job_id, status="succeeded")
    return 1


def _next_label_job_delay_seconds() -> float | None:
    rate_limit_delay = _label_job_rate_limit_delay_seconds()
    if rate_limit_delay > 0:
        return rate_limit_delay
    pace_delay = _label_job_pace_delay_seconds()
    if pace_delay > 0:
        return pace_delay + (random.uniform(0.0, LABEL_JOB_JITTER_SECONDS) if LABEL_JOB_JITTER_SECONDS > 0 else 0.0)
    with _label_jobs_lock:
        with _state_file_lock("label_jobs"):
            state = _load_label_jobs_unlocked()
            _reset_stale_label_jobs_unlocked(state)
            next_due = _next_due_label_job_at_unlocked(state)
    if next_due is None:
        return None
    return max(0.0, (next_due - _utc_now()).total_seconds())


def _run_label_job_worker() -> None:
    global _label_job_worker_inflight, _label_job_worker_rerun_requested
    try:
        while True:
            processed = _drain_label_jobs_once()
            delay = _next_label_job_delay_seconds()
            if delay is None:
                return
            if delay > 0:
                time.sleep(min(delay, 5.0))
    finally:
        should_rerun = False
        with _label_job_worker_lock:
            if _label_job_worker_rerun_requested:
                _label_job_worker_rerun_requested = False
                should_rerun = True
            else:
                _label_job_worker_inflight = False
        if should_rerun:
            _label_job_executor.submit(_run_label_job_worker)


def _schedule_label_job_worker() -> bool:
    global _label_job_worker_inflight, _label_job_worker_rerun_requested
    with _label_job_worker_lock:
        if _label_job_worker_inflight:
            _label_job_worker_rerun_requested = True
            return False
        _label_job_worker_inflight = True
    _label_job_executor.submit(_run_label_job_worker)
    return True


if os.environ.get("LABEL_DRAIN_ON_STARTUP", "1").strip().lower() not in {"0", "false", "no", "off"}:
    _schedule_label_job_worker()


def _load_preprocess_state() -> dict[str, Any]:
    path = _preprocess_state_path()
    if not path.exists():
        return {
            "schema_version": PREPROCESS_STATE_SCHEMA_VERSION,
            "reolink_processed": {},
        }

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema_version": PREPROCESS_STATE_SCHEMA_VERSION,
            "reolink_processed": {},
        }

    if not isinstance(data, dict):
        data = {}
    processed = data.get("reolink_processed")
    if not isinstance(processed, dict):
        processed = {}
    return {
        "schema_version": PREPROCESS_STATE_SCHEMA_VERSION,
        "reolink_processed": processed,
    }


def _save_preprocess_state(state: dict[str, Any]) -> None:
    path = _preprocess_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PREPROCESS_STATE_SCHEMA_VERSION,
        "reolink_processed": state.get("reolink_processed") or {},
    }
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _reolink_state_key(context: QueueContext, raw_folder: dict[str, Any]) -> str:
    site_key = context.site_key or "unknown-site"
    raw_id = str(raw_folder.get("id") or "")
    raw_name = str(raw_folder.get("name") or "")
    return f"{site_key}:{raw_id or raw_name}"


def _reolink_state_name_key(context: QueueContext, raw_folder: dict[str, Any]) -> str:
    site_key = context.site_key or "unknown-site"
    raw_name = str(raw_folder.get("name") or "")
    return f"{site_key}:name:{raw_name}" if raw_name else _reolink_state_key(context, raw_folder)


def _reolink_raw_folder_processed(
    state: dict[str, Any],
    context: QueueContext,
    raw_folder: dict[str, Any],
) -> bool:
    processed = state.get("reolink_processed") or {}
    drive_status = _reolink_raw_drive_preprocess_status(raw_folder)
    return (
        _reolink_state_key(context, raw_folder) in processed
        or _reolink_state_name_key(context, raw_folder) in processed
        or drive_status == "complete"
    )


def _mark_reolink_raw_folder_processed(
    state: dict[str, Any],
    context: QueueContext,
    raw_folder: dict[str, Any],
    *,
    status: str,
    generated: int = 0,
    reason: str = "",
) -> None:
    processed = state.setdefault("reolink_processed", {})
    record = {
        "site_key": context.site_key,
        "raw_folder_id": str(raw_folder.get("id") or ""),
        "raw_folder_name": str(raw_folder.get("name") or ""),
        "status": status,
        "generated": int(generated),
        "reason": reason[:1200],
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    processed[_reolink_state_key(context, raw_folder)] = record
    processed[_reolink_state_name_key(context, raw_folder)] = record
    _save_preprocess_state(state)


def _missing_manual_crop_channels(
    client: DriveClient,
    context: QueueContext,
) -> list[str]:
    if not context.site_key or not _site_uses_manual_crop_configs(context.site_key):
        return []
    if _supabase_crop_client_configured():
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

    raw_folders = []
    if folder_ids.get("unassociated"):
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
        # frame_0.jpg is intentionally used as the reference image: every group
        # has a frame_0 regardless of N, and we only need one frame to extract
        # the camera's image dimensions for crop calibration.
        frame_item = source_files.get("frame_0.jpg")
        if not frame_item or not frame_item.get("id"):
            continue
        width = 0
        height = 0
        try:
            from PIL import Image

            with tempfile.TemporaryDirectory(prefix="reolink_reference_") as tmpdir:
                reference_path = Path(tmpdir) / "frame_0.jpg"
                client.download_file_to_path(str(frame_item["id"]), reference_path)
                with Image.open(reference_path) as image:
                    width = image.width
                    height = image.height
        except Exception:
            width = 0
            height = 0
        reference_payload = {
            "site_key": site_key,
            "site_label": site.display_name,
            "channel_code": normalized_channel,
            "raw_folder_id": str(raw_folder["id"]),
            "raw_folder_name": str(raw_folder.get("name", "")),
            "frame_file_id": str(frame_item["id"]),
            "preview_url": f"/api/preview/{frame_item['id']}",
            "source": "unassociated",
            "width": width,
            "height": height,
        }
        # Treat the folder as fully present if it has at least the legacy
        # 3 frames OR a complete contiguous frame_0..N-1 set. We only need
        # one valid reference frame, so accept any folder whose first frame
        # exists; richer N detection happens during materialization.
        n_present = _detect_n_from_file_names(source_files)
        if n_present >= LEGACY_FRAMES_PER_GROUP and all(
            source_files.get(f"frame_{idx}.jpg") for idx in range(n_present)
        ):
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


def _download_drive_files_parallel(
    client: DriveClient,
    file_items: list[dict[str, Any]],
    output_paths: list[Path],
) -> list[Path]:
    """Download independent Drive files concurrently with thread-local clients."""
    if len(file_items) != len(output_paths):
        raise ValueError("file_items and output_paths must have the same length")
    if not file_items:
        return []

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    if len(file_items) == 1:
        client.download_file_to_path(str(file_items[0]["id"]), output_paths[0])
        return output_paths

    def _parallel_drive_client() -> DriveClient:
        return DriveClient() if type(client) is DriveClient else client

    def _download_one(item: dict[str, Any], output_path: Path) -> Path:
        _parallel_drive_client().download_file_to_path(str(item["id"]), output_path)
        return output_path

    max_workers = min(REOLINK_FRAME_DOWNLOAD_WORKERS, len(file_items))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_download_one, item, output_path)
            for item, output_path in zip(file_items, output_paths)
        ]
        for future in as_completed(futures):
            future.result()
    return output_paths


def _table_drive_client(client: DriveClient) -> DriveClient:
    return DriveClient() if type(client) is DriveClient else client


def _detect_people_for_frames(frame_paths: list[Path], yolo_model: Any) -> list[list[dict[str, Any]]]:
    from person_detector import assign_track_ids, detect_people_batch, detect_people_in_frame

    if callable(yolo_model):
        frame_detections: list[list[dict[str, Any]]] = []
        for batch_start in range(0, len(frame_paths), REOLINK_YOLO_BATCH_FRAMES):
            batch = frame_paths[batch_start:batch_start + REOLINK_YOLO_BATCH_FRAMES]
            frame_detections.extend(detect_people_batch(batch, yolo_model))
    else:
        frame_detections = [detect_people_in_frame(frame_path, yolo_model) for frame_path in frame_paths]
    assign_track_ids(frame_detections)
    return frame_detections


def _detect_people_for_frame_groups(
    frame_groups: list[list[Path]],
    yolo_model: Any,
) -> list[list[list[dict[str, Any]]]]:
    from person_detector import assign_track_ids, detect_people_batch, detect_people_in_frame

    if not frame_groups:
        return []
    if callable(yolo_model):
        flat_paths = [frame_path for frame_paths in frame_groups for frame_path in frame_paths]
        flat_detections: list[list[dict[str, Any]]] = []
        for batch_start in range(0, len(flat_paths), REOLINK_YOLO_BATCH_FRAMES):
            batch = flat_paths[batch_start:batch_start + REOLINK_YOLO_BATCH_FRAMES]
            flat_detections.extend(detect_people_batch(batch, yolo_model))
        grouped: list[list[list[dict[str, Any]]]] = []
        offset = 0
        for frame_paths in frame_groups:
            detections = flat_detections[offset:offset + len(frame_paths)]
            assign_track_ids(detections)
            grouped.append(detections)
            offset += len(frame_paths)
        return grouped

    grouped = []
    for frame_paths in frame_groups:
        detections = [detect_people_in_frame(frame_path, yolo_model) for frame_path in frame_paths]
        assign_track_ids(detections)
        grouped.append(detections)
    return grouped


@dataclass
class _ScreenRecordTrueTenCandidate:
    raw_folder: dict[str, Any]
    state_folder: dict[str, Any]
    source_files: dict[str, dict[str, Any]]
    metadata: dict[str, Any]
    camera: dict[str, Any]
    missing_table_polygons: list[tuple[str, list, tuple[int, int, int, int], list]]


def _materialize_screenrecord_true_ten_artifacts(
    client: DriveClient,
    context: QueueContext,
    raw_folder: dict[str, Any],
    missing_table_polygons: list[tuple[str, list, tuple[int, int, int, int], list]],
    metadata: dict[str, Any],
) -> list[str]:
    source_files = {
        item["name"]: item
        for item in client.list_files(
            raw_folder["id"],
            fields="id,name,mimeType,parents,appProperties",
        )
    }
    mapped = _mapped_camera_tables_for_screenrecord_folder(
        str(raw_folder.get("name") or ""),
        metadata,
        site_key=context.site_key,
        client=client,
    )
    if mapped is None:
        return []

    _channel_number, camera, all_table_polygons = mapped
    table_polygons_by_id = {
        table_id: (table_id, tight_poly, tight_bbox, zone_poly)
        for table_id, tight_poly, tight_bbox, zone_poly in all_table_polygons
    }
    selected_polygons = [
        table_polygons_by_id[table_id]
        for table_id, *_rest in missing_table_polygons
        if table_id in table_polygons_by_id
    ]
    if not selected_polygons:
        return []

    candidate = _ScreenRecordTrueTenCandidate(
        raw_folder=raw_folder,
        state_folder=_screenrecord_state_raw_folder(raw_folder),
        source_files=source_files,
        metadata=metadata,
        camera=camera,
        missing_table_polygons=selected_polygons,
    )
    results = _materialize_screenrecord_true_ten_batch(client, context, [candidate])
    return results.get(str(raw_folder.get("id") or ""), [])


def _materialize_screenrecord_true_ten_batch(
    client: DriveClient,
    context: QueueContext,
    candidates: list[_ScreenRecordTrueTenCandidate],
) -> dict[str, list[str]]:
    """Build final labeler artifacts from ScreenRecord 10frametrue folders.

    Each source folder has ten full frames. The output remains the existing
    compact 3-frame table crop folder plus perception_v2.json built from all ten
    full frames. This batches downloads and YOLO across folders while preserving
    per-folder artifact semantics.
    """
    from PIL import Image
    from person_detector import build_perception_for_table
    from processor import perspective_crop_polygon, save_jpeg, _scale_table_polygons

    if not candidates:
        return {}

    label_source = _resolve_label_source(context.source, context.site_key)
    output_parent_id = _screenrecord_output_unlabeled_folder_id(client, context)
    selected_source_indices = [0, 5, 9]
    timings: dict[str, float] = {}
    total_started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="screenrecord_10frame_batch_") as tmpdir:
        tmp = Path(tmpdir)
        work: list[dict[str, Any]] = []
        all_frame_items: list[dict[str, Any]] = []
        all_frame_paths: list[Path] = []
        for candidate_index, candidate in enumerate(candidates):
            frame_items = [candidate.source_files.get(f"frame_{idx}.jpg") for idx in range(10)]
            if any(item is None for item in frame_items):
                continue
            frame_paths = [
                tmp / f"candidate_{candidate_index}" / f"frame_{idx}.jpg"
                for idx in range(len(frame_items))
            ]
            all_frame_items.extend([item for item in frame_items if item is not None])
            all_frame_paths.extend(frame_paths)
            work.append({"candidate": candidate, "frame_paths": frame_paths})

        if not work:
            return {}

        download_started = time.perf_counter()
        _download_drive_files_parallel(client, all_frame_items, all_frame_paths)
        timings["download_ms"] = (time.perf_counter() - download_started) * 1000

        for item in work:
            frame_paths = item["frame_paths"]
            candidate = item["candidate"]
            with Image.open(frame_paths[0]) as image:
                frame_h, frame_w = image.height, image.width
            img_shape = (frame_h, frame_w)
            ref_w = int(
                candidate.camera.get("image_width")
                or candidate.camera.get("frame_width")
                or candidate.camera.get("width")
                or frame_w
            )
            ref_h = int(
                candidate.camera.get("image_height")
                or candidate.camera.get("frame_height")
                or candidate.camera.get("height")
                or frame_h
            )
            scaled_polygons = candidate.missing_table_polygons
            if ref_w != frame_w or ref_h != frame_h:
                scaled_polygons = _scale_table_polygons(
                    candidate.missing_table_polygons,
                    ref_w,
                    ref_h,
                    frame_w,
                    frame_h,
                )
            item["img_shape"] = img_shape
            item["scaled_polygons"] = scaled_polygons

        yolo_started = time.perf_counter()
        yolo_model = _get_yolo_model()
        grouped_detections = _detect_people_for_frame_groups(
            [item["frame_paths"] for item in work],
            yolo_model,
        )
        timings["yolo_ms"] = (time.perf_counter() - yolo_started) * 1000
        for item, detections in zip(work, grouped_detections):
            item["frame_detections"] = detections

        def _materialize_one_table(item: dict[str, Any], table_entry: tuple[str, list, tuple[int, int, int, int], list]) -> tuple[str, str]:
            candidate: _ScreenRecordTrueTenCandidate = item["candidate"]
            frame_paths: list[Path] = item["frame_paths"]
            frame_detections = item["frame_detections"]
            img_shape = item["img_shape"]
            table_id, tight_poly, _tight_bbox, zone_poly = table_entry
            table_metadata = _camera_table_metadata(candidate.camera, table_id)
            metadata = candidate.metadata
            raw_folder = candidate.raw_folder
            raw_name = str(raw_folder.get("name") or metadata.get("triplet_stem") or "triplet")
            derived_name = _apply_source_prefix(
                _derived_reolink_folder_name(raw_name, table_id),
                label_source,
            )
            table_client = _table_drive_client(client)
            dest_folder_id = table_client.ensure_subfolder(output_parent_id, derived_name)
            uploaded_frame_ids: dict[str, str | None] = {}

            for output_idx, source_idx in enumerate(selected_source_indices):
                cropped = perspective_crop_polygon(frame_paths[source_idx], zone_poly)
                crop_path = tmp / "crops" / f"{derived_name}_f{output_idx}.jpg"
                save_jpeg(cropped, crop_path)
                uploaded = table_client.upload_or_update_file(
                    crop_path,
                    dest_folder_id,
                    file_name=f"frame_{output_idx}.jpg",
                )
                uploaded_frame_ids[f"frame_{output_idx}"] = str(uploaded["id"])

            source_captured_at = list(metadata.get("captured_at_utc") or metadata.get("source_captured_at_utc") or [])
            selected_captured_at = [
                source_captured_at[idx]
                for idx in selected_source_indices
                if idx < len(source_captured_at)
            ]
            artifact_metadata = {
                **metadata,
                "frame_count": 3,
                "source_frame_count": 10,
                "selected_source_frame_indices": selected_source_indices,
                "selected_captured_at_utc": selected_captured_at,
                "captured_at_utc": selected_captured_at,
                "table_id": table_id,
                "table": {
                    "label": table_metadata.get("label") or table_id,
                },
                "raw_folder_id": raw_folder.get("id"),
                "raw_folder_name": raw_name,
                "restaurant_id": table_metadata.get("restaurant_id") or metadata.get("restaurant_id"),
                "supabase_table_id": table_metadata.get("table_id"),
                "table_camera_crops_id": table_metadata.get("table_camera_crops_id"),
                "camera_source_id": table_metadata.get("camera_source_id") or metadata.get("camera_source_id"),
                "crop_version": table_metadata.get("crop_version"),
                "crop_source": table_metadata.get("crop_source") or candidate.camera.get("source") or "fallback_json",
                "artifact_identity": _artifact_identity(raw_name, table_metadata, metadata),
                "perception_file": PERCEPTION_V2_FILE_NAME,
            }
            table_client.upsert_bytes(
                dest_folder_id,
                "metadata.json",
                json.dumps(artifact_metadata, indent=2).encode("utf-8"),
                mime_type="application/json",
            )

            perception = build_perception_for_table(
                frame_detections,
                tight_poly,
                img_shape,
                n_frames=10,
            )
            perception_path = tmp / "perception" / f"{derived_name}_{PERCEPTION_V2_FILE_NAME}"
            perception_path.parent.mkdir(parents=True, exist_ok=True)
            perception_path.write_text(json.dumps(perception, indent=2), encoding="utf-8")
            table_client.upload_or_update_file(
                perception_path,
                dest_folder_id,
                file_name=PERCEPTION_V2_FILE_NAME,
                mime_type="application/json",
            )

            table_client.update_file_metadata(
                dest_folder_id,
                {"appProperties": build_folder_app_properties(uploaded_frame_ids)},
            )
            return str(raw_folder.get("id") or ""), derived_name

        materialize_started = time.perf_counter()
        tasks = [
            (item, table_entry)
            for item in work
            for table_entry in item["scaled_polygons"]
        ]
        generated_by_folder: dict[str, list[str]] = {
            str(item["candidate"].raw_folder.get("id") or ""): []
            for item in work
        }
        max_workers = min(REOLINK_TABLE_MATERIALIZE_WORKERS, len(tasks))
        if max_workers <= 1:
            for item, table_entry in tasks:
                try:
                    folder_id, generated_name = _materialize_one_table(item, table_entry)
                    generated_by_folder.setdefault(folder_id, []).append(generated_name)
                except Exception as exc:
                    raw_folder = item["candidate"].raw_folder
                    print(f"[screenrecord-true-ten] failed to materialize {raw_folder.get('name')}: {exc}")
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_materialize_one_table, item, table_entry): item
                    for item, table_entry in tasks
                }
                for future in as_completed(futures):
                    try:
                        folder_id, generated_name = future.result()
                    except Exception as exc:
                        raw_folder = futures[future]["candidate"].raw_folder
                        print(f"[screenrecord-true-ten] failed to materialize {raw_folder.get('name')}: {exc}")
                        continue
                    generated_by_folder.setdefault(folder_id, []).append(generated_name)
        timings["materialize_ms"] = (time.perf_counter() - materialize_started) * 1000

        total_ms = (time.perf_counter() - total_started) * 1000
        generated_count = sum(len(names) for names in generated_by_folder.values())
        folders_per_min = (len(work) / (total_ms / 60000.0)) if total_ms > 0 else 0.0
        _log_timing(
            "screenrecord_true_ten_batch",
            total_ms=f"{total_ms:.1f}",
            download_ms=f"{timings.get('download_ms', 0.0):.1f}",
            yolo_ms=f"{timings.get('yolo_ms', 0.0):.1f}",
            materialize_ms=f"{timings.get('materialize_ms', 0.0):.1f}",
            folders=len(work),
            frames=len(all_frame_paths),
            tables=len(tasks),
            generated=generated_count,
            yolo_batch_frames=REOLINK_YOLO_BATCH_FRAMES,
            folders_per_min=f"{folders_per_min:.2f}",
            queue=context.queue_key,
        )
        return generated_by_folder


def _materialize_reolink_table_crops(
    client: DriveClient,
    context: QueueContext,
    raw_folder: dict[str, Any],
    missing_table_polygons: list[tuple[str, list, tuple[int, int, int, int], list]],
    *,
    local_frame_paths: list[Path] | None = None,
    local_metadata_path: Path | None = None,
) -> list[str]:
    """Crop a raw triplet into per-table folders under unlabeled/.

    By default the source frames live on Drive under raw_folder["id"]. When
    `local_frame_paths` is provided (zip-sourced triplets), Drive list/download
    is skipped and the local files are read directly. `local_metadata_path`
    optionally points at a metadata.json on disk that should be copied to each
    destination folder.
    """
    from PIL import Image
    from person_detector import build_perception_for_table
    from processor import (
        perspective_crop_polygon,
        sample_frame_indices,
        save_jpeg,
        _scale_table_polygons,
    )

    using_local = local_frame_paths is not None

    source_files: dict[str, dict[str, Any]] = {}
    if not using_local:
        source_files = {
            item["name"]: item
            for item in client.list_files(
                raw_folder["id"],
                fields="id,name,mimeType,parents,appProperties",
            )
        }
        n_frames = _detect_n_from_file_names(source_files)
        if n_frames < LEGACY_FRAMES_PER_GROUP:
            return []
        frame_items = [source_files.get(f"frame_{idx}.jpg") for idx in range(n_frames)]
        if any(item is None for item in frame_items):
            return []
    else:
        n_frames = len(local_frame_paths)
        if n_frames < LEGACY_FRAMES_PER_GROUP or not all(p.exists() for p in local_frame_paths):
            return []
        frame_items = [None] * n_frames  # unused when using_local

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
        if using_local:
            frame_paths = list(local_frame_paths)
        else:
            frame_paths = [tmp / f"frame_{idx}.jpg" for idx in range(len(frame_items))]
            _download_drive_files_parallel(client, frame_items, frame_paths)

        with Image.open(frame_paths[0]) as image:
            frame_h, frame_w = image.height, image.width
        img_shape = (frame_h, frame_w)

        ref_w = int(camera.get("image_width") or camera.get("frame_width") or camera.get("width") or frame_w)
        ref_h = int(camera.get("image_height") or camera.get("frame_height") or camera.get("height") or frame_h)
        scaled_polygons = selected_polygons
        if ref_w != frame_w or ref_h != frame_h:
            scaled_polygons = _scale_table_polygons(selected_polygons, ref_w, ref_h, frame_w, frame_h)

        sample_indices = sample_frame_indices(n_frames)
        perception_file_name = PERCEPTION_V2_FILE_NAME if n_frames == 10 else None
        frame_detections = None
        if perception_file_name is not None:
            yolo_model = _get_yolo_model()
            frame_detections = _detect_people_for_frames(frame_paths, yolo_model)

        label_source = _resolve_label_source(context.source, context.site_key)
        source_metadata = None
        if using_local:
            if local_metadata_path is not None and local_metadata_path.exists():
                try:
                    loaded = json.loads(local_metadata_path.read_text(encoding="utf-8"))
                    source_metadata = loaded if isinstance(loaded, dict) else {}
                except (OSError, json.JSONDecodeError):
                    source_metadata = {}
            else:
                source_metadata = {}
        else:
            source_metadata = _load_json_file_from_drive(client, source_files.get("metadata.json")) or {}

        def _materialize_one_table(
            table_entry: tuple[str, list, tuple[int, int, int, int], list],
        ) -> str:
            table_id, tight_poly, _tight_bbox, zone_poly = table_entry
            table_metadata = _camera_table_metadata(camera, table_id)
            derived_name = _apply_source_prefix(
                _derived_reolink_folder_name(raw_folder["name"], table_id),
                label_source,
            )
            table_client = _table_drive_client(client)
            dest_folder_id = table_client.ensure_subfolder(context.input_folder_id, derived_name)
            uploaded_frame_ids: dict[str, str | None] = {
                f"frame_{i}": None for i in sample_indices
            }

            for frame_idx in sample_indices:
                frame_path = frame_paths[frame_idx]
                cropped = perspective_crop_polygon(frame_path, zone_poly)
                crop_path = tmp / "crops" / f"{derived_name}_f{frame_idx}.jpg"
                save_jpeg(cropped, crop_path)
                uploaded = table_client.upload_or_update_file(
                    crop_path,
                    dest_folder_id,
                    file_name=f"frame_{frame_idx}.jpg",
                )
                uploaded_frame_ids[f"frame_{frame_idx}"] = str(uploaded["id"])

            table_client.update_file_metadata(
                dest_folder_id,
                {"appProperties": build_folder_app_properties(uploaded_frame_ids)},
            )

            if frame_detections is not None and perception_file_name is not None:
                perception = build_perception_for_table(
                    frame_detections, tight_poly, img_shape, n_frames=n_frames
                )
                perception_path = tmp / "perception" / f"{derived_name}_{perception_file_name}"
                perception_path.parent.mkdir(parents=True, exist_ok=True)
                perception_path.write_text(json.dumps(perception, indent=2), encoding="utf-8")
                table_client.upload_or_update_file(
                    perception_path,
                    dest_folder_id,
                    file_name=perception_file_name,
                    mime_type="application/json",
                )

            metadata = dict(source_metadata or {})
            metadata.update(
                {
                    "table_id": table_id,
                    "table": {"label": table_metadata.get("label") or table_id},
                    "restaurant_id": table_metadata.get("restaurant_id") or metadata.get("restaurant_id"),
                    "supabase_table_id": table_metadata.get("table_id"),
                    "table_camera_crops_id": table_metadata.get("table_camera_crops_id"),
                    "camera_source_id": table_metadata.get("camera_source_id") or metadata.get("camera_source_id"),
                    "crop_version": table_metadata.get("crop_version"),
                    "crop_source": table_metadata.get("crop_source") or camera.get("source") or "fallback_json",
                    "artifact_identity": _artifact_identity(str(raw_folder["name"]), table_metadata, metadata),
                    "perception_file": perception_file_name,
                }
            )
            table_client.upsert_bytes(
                dest_folder_id,
                "metadata.json",
                json.dumps(metadata, indent=2).encode("utf-8"),
                mime_type="application/json",
            )
            return derived_name

        max_workers = min(REOLINK_TABLE_MATERIALIZE_WORKERS, len(scaled_polygons))
        if max_workers <= 1:
            generated_names = [_materialize_one_table(entry) for entry in scaled_polygons]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                generated_names = list(executor.map(_materialize_one_table, scaled_polygons))

        return generated_names


@dataclass
class _UnassociatedZipBatch:
    """One zip from <site>/unassociated_zips/, downloaded and extracted locally.

    `triplets` is a list of synthetic raw_folder dicts (id/name only) that
    `_prepare_reolink_unlabeled_queue` can consume the same way it consumes
    Drive-listed raw folders. Each entry carries `_local_frame_paths` and
    `_local_metadata_path` so `_materialize_reolink_table_crops` can read
    files from local disk.
    """

    zip_file_id: str
    zip_name: str
    triplets: list[dict[str, Any]]
    work_dir: Path
    _tmp_handle: tempfile.TemporaryDirectory[str]
    _success_ids: set[str]
    _failure_ids: set[str]

    def mark_success(self, triplet_id: str) -> None:
        self._success_ids.add(triplet_id)

    def mark_failure(self, triplet_id: str) -> None:
        self._failure_ids.add(triplet_id)

    def all_terminal(self) -> bool:
        terminal = self._success_ids | self._failure_ids
        return all(t["id"] in terminal for t in self.triplets)

    def cleanup(self) -> None:
        try:
            self._tmp_handle.cleanup()
        except Exception:
            pass


def _iterate_unassociated_zip_batches(
    client: DriveClient,
    context: QueueContext,
    *,
    max_batches: int | None = None,
) -> Iterator[_UnassociatedZipBatch]:
    """Yield extracted zip batches from <site>/unassociated_zips/, oldest first.

    The caller should mark each triplet's success/failure on the yielded handle
    and then either let the iterator finalize the zip on the next iteration
    (deletes the zip if every triplet reached a terminal state and at least one
    succeeded), or call `cleanup()` to release the local temp dir without
    deleting the zip from Drive.
    """
    zips_folder_id = context.folder_ids.get(UNASSOCIATED_ZIPS_FOLDER_NAME)
    if not zips_folder_id:
        return

    zip_files = client.list_files(
        zips_folder_id,
        fields="id,name,mimeType,modifiedTime,size",
    )
    zip_files = [
        f
        for f in zip_files
        if str(f.get("name", "")).endswith(".zip")
        and str(f.get("name", "")) != UNASSOCIATED_ZIPS_MANIFEST_FILE
    ]
    zip_files.sort(key=lambda f: str(f.get("modifiedTime") or ""))

    if max_batches is not None:
        zip_files = zip_files[:max_batches]

    for zip_file in zip_files:
        zip_id = str(zip_file["id"])
        zip_name = str(zip_file.get("name") or zip_id)

        tmp_handle = tempfile.TemporaryDirectory(prefix=f"reolink_zip_{zip_id}_")
        work_dir = Path(tmp_handle.name)
        zip_local_path = work_dir / zip_name
        try:
            client.download_file_to_path(zip_id, zip_local_path)
        except Exception as exc:
            print(f"[zip-batch] failed to download zip {zip_name} ({zip_id}): {exc}")
            tmp_handle.cleanup()
            continue

        extract_dir = work_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_local_path, "r") as zf:
                zf.extractall(extract_dir)
        except Exception as exc:
            print(f"[zip-batch] failed to extract zip {zip_name} ({zip_id}): {exc}")
            tmp_handle.cleanup()
            continue

        manifest_path = extract_dir / UNASSOCIATED_ZIPS_INNER_MANIFEST
        if not manifest_path.exists():
            print(f"[zip-batch] zip {zip_name} missing {UNASSOCIATED_ZIPS_INNER_MANIFEST}; skipping")
            tmp_handle.cleanup()
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[zip-batch] zip {zip_name} has unreadable manifest: {exc}")
            tmp_handle.cleanup()
            continue

        triplets: list[dict[str, Any]] = []
        for entry in manifest.get("triplets") or []:
            triplet_id = str(entry.get("triplet_id") or "")
            triplet_name = str(entry.get("triplet_name") or "")
            sanitized_path = str(entry.get("sanitized_path") or "")
            if not triplet_id or not triplet_name or not sanitized_path:
                continue
            triplet_dir = extract_dir / sanitized_path
            # Detect N by scanning what the producer actually wrote — the
            # manifest schema may not specify it on older zips. Fall back to
            # the legacy 3-frame layout if no frame files match the pattern.
            present_indices = sorted({
                int(m.group(1))
                for child in triplet_dir.iterdir() if child.is_file()
                for m in [_FRAME_FILENAME_RE.match(child.name)]
                if m
            }) if triplet_dir.is_dir() else []
            n_frames_in_zip = len(present_indices) or LEGACY_FRAMES_PER_GROUP
            frame_paths = [triplet_dir / f"frame_{idx}.jpg" for idx in range(n_frames_in_zip)]
            if not all(p.exists() for p in frame_paths):
                continue
            metadata_path = triplet_dir / "metadata.json"
            triplets.append(
                {
                    "id": triplet_id,
                    "name": triplet_name,
                    "parents": [],
                    "appProperties": {},
                    "_local_frame_paths": frame_paths,
                    "_local_metadata_path": metadata_path if metadata_path.exists() else None,
                    "_zip_sourced": True,
                }
            )

        if not triplets:
            print(f"[zip-batch] zip {zip_name} contained no usable triplets; skipping")
            tmp_handle.cleanup()
            continue

        batch = _UnassociatedZipBatch(
            zip_file_id=zip_id,
            zip_name=zip_name,
            triplets=triplets,
            work_dir=extract_dir,
            _tmp_handle=tmp_handle,
            _success_ids=set(),
            _failure_ids=set(),
        )
        try:
            yield batch
        finally:
            _finalize_zip_batch(client, batch)


def _finalize_zip_batch(client: DriveClient, batch: _UnassociatedZipBatch) -> None:
    """Delete the zip from Drive iff every triplet reached a terminal state and
    at least one succeeded. Otherwise leave the zip on Drive for retry."""
    try:
        if batch.all_terminal() and batch._success_ids and not batch._failure_ids:
            try:
                client.delete_file(batch.zip_file_id)
                print(
                    f"[zip-batch] deleted processed zip {batch.zip_name} "
                    f"({batch.zip_file_id}) — {len(batch._success_ids)} triplets cropped"
                )
            except Exception as exc:
                print(
                    f"[zip-batch] failed to delete processed zip {batch.zip_name} "
                    f"({batch.zip_file_id}): {exc}"
                )
        elif batch._failure_ids:
            print(
                f"[zip-batch] left zip {batch.zip_name} on Drive "
                f"({len(batch._success_ids)} succeeded, {len(batch._failure_ids)} failed/skipped) "
                f"— will retry next prewarm"
            )
    finally:
        batch.cleanup()


def _prepare_reolink_unlabeled_queue(
    client: DriveClient,
    context: QueueContext,
    target_unlabeled_count: int,
    current_visible_count: int | None = None,
    deadline_monotonic: float | None = None,
) -> int:
    if context.source != REOLINK_SOURCE:
        return 0

    label_source = _resolve_label_source(context.source, context.site_key)
    with _reolink_generation_lock:
        _assert_manual_crop_setup_ready(client, context)
        unlabeled_folders: list[dict[str, Any]] = []
        for input_folder_id in _context_input_folder_ids(context):
            unlabeled_folders.extend(
                client.list_folders(
                    input_folder_id,
                    fields="id,name,mimeType,parents,appProperties",
                )
            )
        existing_names = _existing_generated_folder_names(client, context)
        existing_identities = _existing_generated_artifact_identities(client, context)
        unlabeled_count = len(unlabeled_folders)
        visible_count = unlabeled_count if current_visible_count is None else current_visible_count
        generated_any = False
        generated_count = 0

        if context.folder_ids.get(PROCESSED_RAW_FOLDER_NAME):
            _cleanup_processed_raw_folder(
                client,
                context.folder_ids[PROCESSED_RAW_FOLDER_NAME],
            )
        if visible_count >= target_unlabeled_count:
            return 0

        preprocess_state = _load_preprocess_state()

        # ScreenRecord-native path: use ready 3-frame artifacts first, then
        # materialize missing artifacts from root-level 10frametrue folders.
        true_ten_scan_started = time.perf_counter()
        true_ten_folders = _list_screenrecord_true_ten_folders(client, context)
        true_ten_scanned = 0
        true_ten_candidates = 0
        true_ten_index = 0
        while true_ten_index < len(true_ten_folders) and visible_count < target_unlabeled_count:
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                break

            batch_candidates: list[_ScreenRecordTrueTenCandidate] = []
            while (
                true_ten_index < len(true_ten_folders)
                and len(batch_candidates) < REOLINK_TRUE_TEN_BATCH_SIZE
                and visible_count < target_unlabeled_count
            ):
                if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                    break
                true_ten_folder = true_ten_folders[true_ten_index]
                true_ten_index += 1
                true_ten_scanned += 1

                state_folder = _screenrecord_state_raw_folder(true_ten_folder)
                if _reolink_raw_folder_processed(preprocess_state, context, state_folder):
                    continue

                source_files = {
                    item["name"]: item
                    for item in client.list_files(
                        true_ten_folder["id"],
                        fields="id,name,mimeType,parents,appProperties",
                    )
                }
                metadata = _load_json_file_from_drive(client, source_files.get("metadata.json"))
                if not metadata:
                    continue

                mapped = _mapped_camera_tables_for_screenrecord_folder(
                    str(true_ten_folder.get("name", "")),
                    metadata,
                    site_key=context.site_key,
                    client=client,
                )
                if mapped is None:
                    continue

                _channel_number, _camera, table_polygons = mapped
                missing_table_polygons = [
                    entry
                    for entry in table_polygons
                    if (
                        _derived_reolink_folder_name(str(true_ten_folder["name"]), entry[0])
                        not in existing_names
                        and _apply_source_prefix(
                            _derived_reolink_folder_name(str(true_ten_folder["name"]), entry[0]),
                            label_source,
                        )
                        not in existing_names
                        and _artifact_identity(
                            str(true_ten_folder.get("name", "")),
                            _camera_table_metadata(_camera, entry[0]),
                            metadata,
                        )
                        not in existing_identities
                    )
                ]
                if not missing_table_polygons:
                    _mark_reolink_raw_folder_processed(
                        preprocess_state,
                        context,
                        state_folder,
                        status="complete",
                        generated=0,
                    )
                    continue

                batch_candidates.append(
                    _ScreenRecordTrueTenCandidate(
                        raw_folder=true_ten_folder,
                        state_folder=state_folder,
                        source_files=source_files,
                        metadata=metadata,
                        camera=_camera,
                        missing_table_polygons=missing_table_polygons,
                    )
                )
                true_ten_candidates += 1

            if not batch_candidates:
                continue

            generated_by_folder = _materialize_screenrecord_true_ten_batch(
                client,
                context,
                batch_candidates,
            )
            for candidate in batch_candidates:
                true_ten_folder = candidate.raw_folder
                generated_names = generated_by_folder.get(str(true_ten_folder.get("id") or ""), [])
                raw_generated_count = 0
                recorded_generated = _record_generated_reolink_artifacts(
                    generated_names=generated_names,
                    existing_names=existing_names,
                    existing_identities=existing_identities,
                    raw_name=str(true_ten_folder.get("name", "")),
                    table_polygons=candidate.missing_table_polygons,
                    camera=candidate.camera,
                    metadata=candidate.metadata,
                    label_source=label_source,
                )
                for _name in generated_names:
                    unlabeled_count += 1
                    visible_count += 1
                    generated_any = True
                    generated_count += 1
                    raw_generated_count += 1
                raw_generated_count = max(raw_generated_count, recorded_generated)

                if raw_generated_count >= len(candidate.missing_table_polygons):
                    _mark_reolink_raw_folder_processed(
                        preprocess_state,
                        context,
                        candidate.state_folder,
                        status="complete",
                        generated=raw_generated_count,
                    )

        true_ten_scan_ms = (time.perf_counter() - true_ten_scan_started) * 1000
        if true_ten_scanned or true_ten_candidates:
            _log_timing(
                "screenrecord_true_ten_scan",
                total_ms=f"{true_ten_scan_ms:.1f}",
                scanned=true_ten_scanned,
                candidates=true_ten_candidates,
                generated=generated_count,
                batch_size=REOLINK_TRUE_TEN_BATCH_SIZE,
                target=target_unlabeled_count,
                visible=visible_count,
                deadline_hit=int(deadline_monotonic is not None and time.monotonic() >= deadline_monotonic),
                queue=context.queue_key,
            )

        if visible_count >= target_unlabeled_count:
            if generated_any:
                _invalidate_listing_cache(context.queue_key)
            return generated_count

        # Phase 3: drain zipped unprocessed batches first (older queue from
        # before the per-triplet upload pattern; lives at <site>/unassociated_zips/).
        # Each zip is a batch of raw triplets compacted by
        # scripts/compact_unassociated_to_zips.py. We download, extract,
        # process each triplet from local files, then delete the zip when
        # every triplet inside it has reached a terminal state.
        for zip_batch in _iterate_unassociated_zip_batches(client, context):
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                break
            if visible_count >= target_unlabeled_count:
                break
            for triplet in zip_batch.triplets:
                if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                    break
                if visible_count >= target_unlabeled_count:
                    break
                triplet_id = triplet["id"]
                if _reolink_raw_folder_processed(preprocess_state, context, triplet):
                    zip_batch.mark_success(triplet_id)
                    continue

                mapped = _mapped_camera_tables_for_reolink_folder(
                    str(triplet.get("name", "")),
                    site_key=context.site_key,
                    client=client,
                )
                if mapped is None:
                    zip_batch.mark_failure(triplet_id)
                    continue

                _channel_number, _camera, table_polygons = mapped
                missing_table_polygons = [
                    entry
                    for entry in table_polygons
                    if (
                        _derived_reolink_folder_name(str(triplet["name"]), entry[0])
                        not in existing_names
                        and _apply_source_prefix(
                            _derived_reolink_folder_name(str(triplet["name"]), entry[0]),
                            label_source,
                        )
                        not in existing_names
                        and _artifact_identity(
                            str(triplet.get("name", "")),
                            _camera_table_metadata(_camera, entry[0]),
                            {},
                        )
                        not in existing_identities
                    )
                ]
                if not missing_table_polygons:
                    _mark_reolink_raw_folder_processed(
                        preprocess_state,
                        context,
                        triplet,
                        status="complete",
                        generated=0,
                    )
                    zip_batch.mark_success(triplet_id)
                    continue

                try:
                    generated_names = _materialize_reolink_table_crops(
                        client,
                        context,
                        triplet,
                        missing_table_polygons,
                        local_frame_paths=triplet["_local_frame_paths"],
                        local_metadata_path=triplet.get("_local_metadata_path"),
                    )
                except Exception as exc:
                    print(
                        f"[zip-batch] crop failed for {triplet.get('name')} ({triplet_id}): {exc}"
                    )
                    zip_batch.mark_failure(triplet_id)
                    continue

                raw_generated_count = 0
                recorded_generated = _record_generated_reolink_artifacts(
                    generated_names=generated_names,
                    existing_names=existing_names,
                    existing_identities=existing_identities,
                    raw_name=str(triplet.get("name", "")),
                    table_polygons=missing_table_polygons,
                    camera=_camera,
                    metadata={},
                    label_source=label_source,
                )
                for _name in generated_names:
                    unlabeled_count += 1
                    visible_count += 1
                    generated_any = True
                    generated_count += 1
                    raw_generated_count += 1
                raw_generated_count = max(raw_generated_count, recorded_generated)

                if raw_generated_count > 0:
                    _mark_reolink_raw_folder_processed(
                        preprocess_state,
                        context,
                        triplet,
                        status="complete",
                        generated=raw_generated_count,
                    )
                    zip_batch.mark_success(triplet_id)
                else:
                    zip_batch.mark_failure(triplet_id)

        if visible_count >= target_unlabeled_count:
            if generated_any:
                _invalidate_listing_cache(context.queue_key)
            return generated_count

        raw_folders = _list_reolink_raw_folders(client, context)

        for raw_folder in raw_folders:
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                break
            if visible_count >= target_unlabeled_count:
                break

            if _reolink_raw_folder_processed(preprocess_state, context, raw_folder):
                if _reolink_raw_drive_preprocess_status(raw_folder) != "complete":
                    _stamp_reolink_raw_preprocess_status(
                        client,
                        context,
                        raw_folder,
                        status="complete",
                    )
                _move_reolink_raw_to_processed(client, context, raw_folder)
                continue

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
                if (
                    _derived_reolink_folder_name(str(raw_folder["name"]), entry[0])
                    not in existing_names
                    and _apply_source_prefix(
                        _derived_reolink_folder_name(str(raw_folder["name"]), entry[0]),
                        label_source,
                    )
                    not in existing_names
                    and _artifact_identity(
                        str(raw_folder.get("name", "")),
                        _camera_table_metadata(_camera, entry[0]),
                        {},
                    )
                    not in existing_identities
                )
            ]
            if not missing_table_polygons:
                _mark_reolink_raw_folder_processed(
                    preprocess_state,
                    context,
                    raw_folder,
                    status="complete",
                    generated=0,
                )
                _stamp_reolink_raw_preprocess_status(
                    client,
                    context,
                    raw_folder,
                    status="complete",
                    generated=0,
                )
                _move_reolink_raw_to_processed(client, context, raw_folder)
                continue

            _stamp_reolink_raw_preprocess_status(
                client,
                context,
                raw_folder,
                status="in_progress",
            )
            generated_names = _materialize_reolink_table_crops(
                client,
                context,
                raw_folder,
                missing_table_polygons,
            )
            raw_generated_count = 0
            recorded_generated = _record_generated_reolink_artifacts(
                generated_names=generated_names,
                existing_names=existing_names,
                existing_identities=existing_identities,
                raw_name=str(raw_folder.get("name", "")),
                table_polygons=missing_table_polygons,
                camera=_camera,
                metadata={},
                label_source=label_source,
            )
            for _name in generated_names:
                unlabeled_count += 1
                visible_count += 1
                generated_any = True
                generated_count += 1
                raw_generated_count += 1
            raw_generated_count = max(raw_generated_count, recorded_generated)

            if raw_generated_count > 0:
                _mark_reolink_raw_folder_processed(
                    preprocess_state,
                    context,
                    raw_folder,
                    status="complete",
                    generated=raw_generated_count,
                )
                _stamp_reolink_raw_preprocess_status(
                    client,
                    context,
                    raw_folder,
                    status="complete",
                    generated=raw_generated_count,
                )
                _move_reolink_raw_to_processed(client, context, raw_folder)
            if raw_generated_count == 0:
                continue

        if generated_any:
            _invalidate_listing_cache(context.queue_key)
        return generated_count


def drain_reolink_preprocessing(
    client: DriveClient | None = None,
    site_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Materialize all missing Reolink per-table folders and then stop."""
    drive = client or DriveClient()
    requested_site_keys = site_keys or [site.site_key for site in REOLINK_SITES]
    deadline_monotonic = (
        time.monotonic() + REOLINK_PREPROCESS_MAX_SECONDS
        if REOLINK_PREPROCESS_MAX_SECONDS > 0
        else None
    )
    summary: dict[str, Any] = {
        "sites": {},
        "generated": 0,
        "errors": {},
        "max_seconds": REOLINK_PREPROCESS_MAX_SECONDS,
    }
    for site_key in requested_site_keys:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            summary["stopped_reason"] = "max_seconds"
            break
        try:
            context = _resolve_queue_context(drive, REOLINK_SOURCE, site_key)
            generated = _prepare_reolink_unlabeled_queue(
                drive,
                context,
                target_unlabeled_count=1_000_000_000,
                deadline_monotonic=deadline_monotonic,
            )
            summary["sites"][site_key] = {"generated": generated}
            summary["generated"] += generated
        except Exception as exc:
            summary["sites"][site_key] = {"generated": 0, "error": str(exc)}
            summary["errors"][site_key] = str(exc)
    return summary


def _run_reolink_preprocess_background(
    source: str,
    site_key: str | None,
    target_unlabeled_count: int,
    queue_key: str,
) -> None:
    try:
        drive = DriveClient()
        context = _resolve_queue_context(drive, source, site_key)
        generated = _prepare_reolink_unlabeled_queue(
            drive,
            context,
            target_unlabeled_count=target_unlabeled_count,
            current_visible_count=None,
        )
        if generated:
            _invalidate_listing_cache(context.queue_key)
    except Exception as exc:
        print(f"[auto-preprocess] reolink run failed: {exc}")
    finally:
        with _reolink_preprocess_lock:
            _reolink_preprocess_inflight.discard(queue_key)


def _maybe_trigger_reolink_preprocess(
    context: QueueContext,
    total_unlabeled: int,
    target_unlabeled_count: int,
) -> None:
    if context.source != REOLINK_SOURCE or not context.site_key:
        return
    if total_unlabeled >= target_unlabeled_count:
        return

    with _reolink_preprocess_lock:
        if context.queue_key in _reolink_preprocess_inflight:
            return
        _reolink_preprocess_inflight.add(context.queue_key)

    _reolink_preprocess_executor.submit(
        _run_reolink_preprocess_background,
        context.source,
        context.site_key,
        target_unlabeled_count,
        context.queue_key,
    )


def _context_input_folder_ids(context: QueueContext) -> tuple[str, ...]:
    folder_ids = [context.input_folder_id]
    if context.source == REOLINK_SOURCE:
        screenrecord_unlabeled = context.folder_ids.get(SCREENRECORD_THREE_FRAME_UNLABELED_KEY)
        if screenrecord_unlabeled:
            folder_ids.append(screenrecord_unlabeled)
    unique: list[str] = []
    for folder_id in folder_ids:
        if folder_id and folder_id not in unique:
            unique.append(folder_id)
    return tuple(unique)


def _fetch_source_listing(client: DriveClient, context: QueueContext) -> list[dict[str, str]]:
    folders: list[dict[str, str]] = []
    for input_folder_id in _context_input_folder_ids(context):
        folders.extend(
            client.list_folders(
                input_folder_id,
                fields="id,name,mimeType,parents,appProperties,modifiedTime",
            )
        )
    return sorted(folders, key=lambda item: str(item.get("name", "")).lower())


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

    if cached_listing is not None and force_refresh:
        _schedule_listing_refresh(context)
        return cached_listing

    if cached_listing is not None and not force_refresh:
        _schedule_listing_refresh(context)
        return cached_listing

    if force_refresh and context.source == REOLINK_SOURCE and has_request_context():
        _schedule_listing_refresh(context)
        return []

    listing = _fetch_source_listing(client, context)
    _set_listing_cache(context.queue_key, listing)
    return list(listing)


def _filter_label_history_hidden_subfolders(
    subfolders: list[dict[str, str]],
    context: QueueContext,
    labeled_records: dict[str, Any],
) -> tuple[list[dict[str, str]], int]:
    if not labeled_records:
        return subfolders, 0

    visible: list[dict[str, str]] = []
    hidden = 0
    for folder in subfolders:
        folder_id = str(folder.get("id") or "")
        folder_name = str(folder.get("name") or "")
        if _label_history_lookup_in_records(labeled_records, context, folder_id, folder_name, ""):
            hidden += 1
            _remove_folder_from_listing_cache(context.queue_key, folder_id)
            with _hydrated_folder_cache_lock:
                _hydrated_folder_cache.pop(_hydrated_cache_key(context.queue_key, folder_id), None)
            continue
        visible.append(folder)
    return visible, hidden


def _frame_payload_from_files(files: list[dict]) -> dict[str, str | None]:
    file_map = {f["name"]: f["id"] for f in files}
    present_indices = sorted({
        int(m.group(1))
        for name in file_map
        for m in [_FRAME_FILENAME_RE.match(str(name))]
        if m
    })
    if not present_indices:
        return {f"frame_{i}": file_map.get(f"frame_{i}.jpg") for i in range(LEGACY_FRAMES_PER_GROUP)}
    if present_indices == list(range(present_indices[-1] + 1)):
        indices = list(range(present_indices[-1] + 1))
    elif tuple(present_indices) == SPARSE_SAMPLE_FRAME_INDICES:
        indices = present_indices
    else:
        indices = list(range(present_indices[-1] + 1))
    return {f"frame_{i}": file_map.get(f"frame_{i}.jpg") for i in indices}


def _frame_payload_from_folder(folder: dict[str, object]) -> dict[str, str | None]:
    return extract_frame_ids_from_item(folder)


def _file_by_name(files: list[dict[str, Any]]) -> dict[str, Any]:
    return {str(item.get("name") or ""): item for item in files}


def _load_json_file_from_drive(client: DriveClient, file_item: dict[str, Any] | None) -> dict[str, Any] | None:
    if not file_item or not file_item.get("id"):
        return None
    try:
        raw = client.download_file_content(str(file_item["id"]))
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _perception_payload_is_from_ten_frames(
    perception: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    if not isinstance(perception, dict):
        return False
    try:
        if int(perception.get("n_frames") or 0) == 10:
            return True
    except (TypeError, ValueError):
        pass
    # Only use metadata as supporting evidence when the perception explicitly
    # declares schema v2; otherwise legacy 3-frame sidecars are too ambiguous.
    try:
        schema_version = int(perception.get("schema_version") or 0)
        source_frame_count = int((metadata or {}).get("source_frame_count") or 0)
    except (TypeError, ValueError):
        return False
    return schema_version >= 2 and source_frame_count == 10


def _normalize_perception_sidecar(
    client: DriveClient,
    folder: dict[str, Any],
    files_by_name: dict[str, dict[str, Any]],
) -> None:
    metadata = _load_json_file_from_drive(client, files_by_name.get("metadata.json"))
    canonical = files_by_name.get(PERCEPTION_V2_FILE_NAME)
    if canonical:
        perception = _load_json_file_from_drive(client, canonical)
        if _perception_payload_is_from_ten_frames(perception, metadata):
            folder["_perception_file_id"] = str(canonical["id"])
            folder["_perception_file_name"] = PERCEPTION_V2_FILE_NAME
        return

    for legacy_name in LEGACY_PERCEPTION_FILE_NAMES:
        legacy = files_by_name.get(legacy_name)
        if not legacy:
            continue
        perception = _load_json_file_from_drive(client, legacy)
        if not _perception_payload_is_from_ten_frames(perception, metadata):
            continue
        data = client.download_file_content(str(legacy["id"]))
        uploaded = client.upsert_bytes(
            str(folder["id"]),
            PERCEPTION_V2_FILE_NAME,
            data,
            mime_type="application/json",
        )
        folder["_perception_file_id"] = str(uploaded["id"])
        folder["_perception_file_name"] = PERCEPTION_V2_FILE_NAME
        return


def _build_folder_payload(
    folder: dict[str, str],
    context: QueueContext,
    frames: dict[str, str | None],
) -> dict:
    frame_signature = _frame_signature_from_frames(frames)
    content_signature = _content_signature_from_frames(frames)
    preview_urls = {
        key: f"/api/preview/{file_id}"
        for key, file_id in frames.items()
        if file_id
    }
    thumb_urls = {
        key: f"/api/thumb/{file_id}"
        for key, file_id in frames.items()
        if file_id
    }
    parents = [str(parent) for parent in folder.get("parents", []) if parent]
    parent_id = parents[0] if parents else context.input_folder_id
    source_label = _folder_source_label(folder, context, parent_id)
    return {
        "folder_id": folder["id"],
        "folder_name": folder["name"],
        "parent_id": parent_id,
        "source": context.source,
        "source_label": source_label,
        "site_key": context.site_key,
        "queue_key": context.queue_key,
        "frames": frames,
        "frame_signature": frame_signature,
        "content_signature": content_signature,
        "preview_urls": preview_urls,
        "thumb_urls": thumb_urls,
        "cache_ready": _thumbs_cache_ready(frames),
        "perception_file_id": folder.get("_perception_file_id"),
        "perception_file_name": folder.get("_perception_file_name"),
        "metadata_file_id": folder.get("_metadata_file_id"),
    }


def _folder_source_label(folder: dict[str, Any], context: QueueContext, parent_id: str) -> str:
    if context.source == REOLINK_SOURCE:
        screenrecord_unlabeled = context.folder_ids.get(SCREENRECORD_THREE_FRAME_UNLABELED_KEY)
        if screenrecord_unlabeled and parent_id == screenrecord_unlabeled:
            return SCREENRECORD_TRUE_TEN_FOLDER_NAME
        if folder.get("_perception_file_name") == PERCEPTION_V2_FILE_NAME:
            return SCREENRECORD_TRUE_TEN_FOLDER_NAME
        return context.input_folder_name
    return context.source


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


def _thumb_path_for_file(file_id: str) -> Path:
    return _ensure_cache_dir() / f"{file_id}.thumb.jpg"


def _frames_cache_ready(frames: dict[str, str | None]) -> bool:
    keys = _ordered_frame_keys(frames)
    if not keys:
        return False
    for key in keys:
        file_id = frames.get(key)
        if not file_id or not _cache_path_for_file(file_id).exists():
            return False
    return True


def _thumbs_cache_ready(frames: dict[str, str | None]) -> bool:
    keys = _ordered_frame_keys(frames)
    if not keys:
        return False
    for key in keys:
        file_id = frames.get(key)
        if not file_id or not _thumb_path_for_file(file_id).exists():
            return False
    return True


def _folder_cache_ready(folder: dict) -> bool:
    return _thumbs_cache_ready(folder.get("frames", {}))


def _hydrate_folder(client: DriveClient, context: QueueContext, folder: dict[str, str]) -> dict | None:
    files: list[dict[str, Any]] | None = None
    if context.source == REOLINK_SOURCE:
        files = client.list_files(folder["id"])
        files_by_name = _file_by_name(files)
        if "metadata.json" in files_by_name:
            folder["_metadata_file_id"] = str(files_by_name["metadata.json"]["id"])
        _normalize_perception_sidecar(client, folder, files_by_name)

    frames = _frame_payload_from_folder(folder)
    if has_complete_frame_ids(frames):
        if files is None:
            files = client.list_files(folder["id"])
        if not _frame_ids_belong_to_files(frames, files):
            frames = _frame_payload_from_files(files)
            if not has_complete_frame_ids(frames):
                return None
            if context.persist_frame_metadata:
                _persist_folder_frame_metadata(client, folder, frames)
        return _build_folder_payload(folder, context, frames)

    if files is None:
        files = client.list_files(folder["id"])
    frames = _frame_payload_from_files(files)
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


def _ensure_thumb_for_file(file_id: str, client: DriveClient | None = None) -> tuple[Path, bool, float, float]:
    thumb_path = _thumb_path_for_file(file_id)
    cache_path = _cache_path_for_file(file_id)
    cache_hit = thumb_path.exists()
    download_ms = 0.0
    encode_ms = 0.0

    if cache_hit:
        return thumb_path, True, download_ms, encode_ms

    if not cache_path.exists():
        active_client = client or DriveClient()
        download_started = time.perf_counter()
        active_client.download_file_to_path(file_id, cache_path)
        download_ms = (time.perf_counter() - download_started) * 1000

    from PIL import Image

    encode_started = time.perf_counter()
    with Image.open(cache_path) as img:
        rgb = img.convert("RGB")
        rgb.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 4), Image.Resampling.LANCZOS)
        rgb.save(thumb_path, "JPEG", quality=THUMB_QUALITY, optimize=True)
    encode_ms = (time.perf_counter() - encode_started) * 1000
    return thumb_path, False, download_ms, encode_ms


def _warm_thumb(file_id: str) -> None:
    try:
        thumb_path, _, _, _ = _ensure_thumb_for_file(file_id, DriveClient())
        try:
            os.utime(thumb_path, None)
        except OSError:
            pass
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
        for key in _ordered_frame_keys(frames):
            file_id = frames.get(key)
            if not file_id:
                continue
            if _thumb_path_for_file(file_id).exists():
                continue
            with _preview_prewarm_lock:
                if file_id in _preview_prewarm_inflight:
                    continue
                _preview_prewarm_inflight.add(file_id)
            _preview_prewarm_executor.submit(_warm_thumb, file_id)
            scheduled += 1
    return scheduled


def _cleanup_hidden_folder(context: QueueContext, folder_id: str) -> None:
    try:
        client = DriveClient()
        current = client.get_file(folder_id, fields="id,name,parents,appProperties")
        parents = [str(parent) for parent in current.get("parents", []) if parent]
        if context.input_folder_id not in parents:
            return
        metadata = dict(current.get("appProperties") or {})
        metadata.setdefault("autolabel_duplicate_cleanup", datetime.now(timezone.utc).isoformat())
        client.update_file_metadata(
            folder_id,
            {"appProperties": metadata},
            fields="id,name,mimeType,parents,appProperties",
        )
        client.move_file(
            folder_id,
            new_parent_id=context.folder_ids["discarded"],
            remove_parent_id=context.input_folder_id,
        )
        _remove_folder_from_listing_cache(context.queue_key, folder_id)
    except Exception:
        return
    finally:
        with _duplicate_cleanup_lock:
            _duplicate_cleanup_inflight.discard((context.queue_key, folder_id))


def _schedule_hidden_folder_cleanup(context: QueueContext, folder_id: str) -> bool:
    key = (context.queue_key, folder_id)
    with _duplicate_cleanup_lock:
        if key in _duplicate_cleanup_inflight:
            return False
        _duplicate_cleanup_inflight.add(key)
    _duplicate_cleanup_executor.submit(_cleanup_hidden_folder, context, folder_id)
    return True


def _cache_warm_state_snapshot() -> dict[str, Any]:
    with _cache_warm_lock:
        state = dict(_cache_warm_state)
        state["errors"] = list(_cache_warm_state.get("errors", []))
        return state


def _set_cache_warm_state(**updates: Any) -> None:
    with _cache_warm_lock:
        _cache_warm_state.update(updates)


def _append_cache_warm_error(message: str) -> None:
    with _cache_warm_lock:
        errors = list(_cache_warm_state.get("errors", []))
        if len(errors) < CACHE_WARM_ERROR_LIMIT:
            errors.append(message)
        _cache_warm_state["errors"] = errors
        _cache_warm_state["last_error"] = message


def _increment_cache_warm_state(**increments: int) -> None:
    with _cache_warm_lock:
        for key, amount in increments.items():
            _cache_warm_state[key] = int(_cache_warm_state.get(key) or 0) + amount


def _cache_warm_stop_requested() -> bool:
    with _cache_warm_lock:
        return bool(_cache_warm_state.get("stop_requested"))


def _request_cache_warm_stop() -> dict[str, Any]:
    with _cache_warm_lock:
        if _cache_warm_state.get("inflight"):
            _cache_warm_state["stop_requested"] = True
        state = dict(_cache_warm_state)
        state["errors"] = list(_cache_warm_state.get("errors", []))
        return state


def _cache_warm_contexts(
    client: DriveClient,
    source: str | None = None,
    site_key: str | None = None,
) -> list[QueueContext]:
    if source:
        return [_resolve_queue_context(client, source, site_key)]

    contexts: list[QueueContext] = []
    for label_source in LABEL_SOURCES:
        try:
            contexts.append(
                _resolve_queue_context(
                    client,
                    label_source.source,
                    label_source.site_key,
                )
            )
        except Exception as exc:
            _append_cache_warm_error(f"{label_source.queue_key}: {exc}")
    return contexts


def _warm_cache_file_once(client: DriveClient, file_id: str) -> None:
    cache_path = _cache_path_for_file(file_id)
    thumb_path = _thumb_path_for_file(file_id)
    full_existed = cache_path.exists()
    thumb_existed = thumb_path.exists()

    _ensure_thumb_for_file(file_id, client)

    increments = {"frames_seen": 1}
    if full_existed:
        increments["skipped_full_res"] = 1
    elif cache_path.exists():
        increments["full_res_cached"] = 1

    if thumb_existed:
        increments["skipped_thumbs"] = 1
    elif thumb_path.exists():
        increments["thumbs_cached"] = 1

    _increment_cache_warm_state(**increments)


def _warm_cache_for_context(client: DriveClient, context: QueueContext, limit: int | None = None) -> None:
    _set_cache_warm_state(current_queue=context.queue_key)
    subfolders = _list_source_subfolders(client, context)
    if limit is not None:
        subfolders = subfolders[:limit]
    _increment_cache_warm_state(folders_scanned=len(subfolders))

    for batch_start in range(0, len(subfolders), CACHE_WARM_BATCH_SIZE):
        if _cache_warm_stop_requested():
            break
        folder_batch = subfolders[batch_start:batch_start + CACHE_WARM_BATCH_SIZE]
        # Hydrate folders in this batch in parallel using the shared prewarm pool.
        futures = []
        for folder in folder_batch:
            if _cache_warm_stop_requested():
                break
            fut = _preview_prewarm_executor.submit(_warm_cache_folder_parallel, context, folder)
            futures.append(fut)
        for fut in futures:
            if _cache_warm_stop_requested():
                break
            try:
                fut.result()
            except Exception:
                pass
        if CACHE_WARM_BATCH_PAUSE_SECONDS > 0:
            time.sleep(CACHE_WARM_BATCH_PAUSE_SECONDS)

    _increment_cache_warm_state(queues_completed=1)


def _warm_cache_folder_parallel(context: QueueContext, folder: dict[str, str]) -> None:
    """Hydrate a folder, warm its thumbs, and keep the app-side payload hot."""
    try:
        payload = _hydrate_folder_with_fresh_client(context, folder)
        if payload is None:
            _set_cached_hydrated_folder(context.queue_key, folder["id"], None)
            return
        _increment_cache_warm_state(folders_hydrated=1)
        frames = payload.get("frames", {})
        history_record = _label_history_lookup(
            context,
            str(payload.get("folder_id") or ""),
            str(payload.get("folder_name") or ""),
            str(payload.get("frame_signature") or ""),
            str(payload.get("content_signature") or ""),
        )
        if history_record:
            _set_cached_hydrated_folder(context.queue_key, folder["id"], None)
            _remove_folder_from_listing_cache(context.queue_key, str(payload.get("folder_id") or ""))
            _schedule_hidden_folder_cleanup(context, str(payload.get("folder_id") or ""))
            return

        file_futures = []
        for key in _ordered_frame_keys(frames):
            if _cache_warm_stop_requested():
                break
            file_id = frames.get(key)
            if not file_id:
                continue
            fut = _preview_prewarm_executor.submit(_warm_cache_file_parallel, str(file_id))
            file_futures.append(fut)
        for fut in file_futures:
            try:
                fut.result()
            except Exception:
                pass
        payload["content_signature"] = _content_signature_from_frames(frames)
        payload["cache_ready"] = _folder_cache_ready(payload)
        history_record = _label_history_lookup(
            context,
            str(payload.get("folder_id") or ""),
            str(payload.get("folder_name") or ""),
            str(payload.get("frame_signature") or ""),
            str(payload.get("content_signature") or ""),
        )
        if history_record:
            _set_cached_hydrated_folder(context.queue_key, folder["id"], None)
            _remove_folder_from_listing_cache(context.queue_key, str(payload.get("folder_id") or ""))
            _schedule_hidden_folder_cleanup(context, str(payload.get("folder_id") or ""))
            return
        _set_cached_hydrated_folder(context.queue_key, folder["id"], payload)
        if payload["cache_ready"]:
            _increment_cache_warm_state(folders_hot_cached=1)
    except Exception as exc:
        folder_name = str(folder.get("name") or folder.get("id") or "unknown")
        _append_cache_warm_error(f"{context.queue_key}/{folder_name}: {exc}")


def _warm_cache_file_parallel(file_id: str) -> None:
    """Download and thumbnail a single file. Increments cache warm state counters."""
    if _cache_warm_stop_requested():
        return
    cache_path = _cache_path_for_file(file_id)
    thumb_path = _thumb_path_for_file(file_id)
    full_existed = cache_path.exists()
    thumb_existed = thumb_path.exists()
    _ensure_thumb_for_file(file_id)
    increments: dict[str, int] = {"frames_seen": 1}
    if full_existed:
        increments["skipped_full_res"] = 1
    elif cache_path.exists():
        increments["full_res_cached"] = 1
    if thumb_existed:
        increments["skipped_thumbs"] = 1
    elif thumb_path.exists():
        increments["thumbs_cached"] = 1
    _increment_cache_warm_state(**increments)


def _run_cache_warm_background(
    source: str | None,
    site_key: str | None,
    limit: int | None,
    shared_lock: dict[str, Any] | None = None,
    release_shared_lock: bool = False,
) -> None:
    acquired_here = release_shared_lock
    if shared_lock is None:
        shared_lock = _acquire_cache_warm_shared_lock()
        acquired_here = shared_lock is not None
        if shared_lock is None:
            _set_cache_warm_state(
                inflight=False,
                current_queue=None,
                completed_at=datetime.now(timezone.utc).isoformat(),
                last_error="Another cache warm worker is already running.",
                shared_lock=_read_cache_warm_shared_lock(),
                shared_lock_path=str(_cache_warm_shared_lock_path()),
            )
            return

    started_at = datetime.now(timezone.utc).isoformat()
    requested = {
        "source": source or "all",
        "site_key": site_key,
        "limit": limit,
    }
    with _cache_warm_lock:
        _cache_warm_state.update(
            {
                "inflight": True,
                "started_at": started_at,
                "completed_at": None,
                "requested": requested,
                "current_queue": None,
                "queues_total": 0,
                "queues_completed": 0,
                "folders_scanned": 0,
                "folders_hydrated": 0,
                "folders_hot_cached": 0,
                "frames_seen": 0,
                "full_res_cached": 0,
                "thumbs_cached": 0,
                "skipped_full_res": 0,
                "skipped_thumbs": 0,
                "errors": [],
                "last_error": None,
                "stop_requested": False,
                "batch_size": CACHE_WARM_BATCH_SIZE,
                "shared_lock": shared_lock,
                "shared_lock_path": str(_cache_warm_shared_lock_path()),
            }
        )

    try:
        client = DriveClient()
        contexts = _cache_warm_contexts(client, source, site_key)
        _set_cache_warm_state(queues_total=len(contexts))
        for context in contexts:
            if _cache_warm_stop_requested():
                break
            _warm_cache_for_context(client, context, limit)
    except Exception as exc:
        _append_cache_warm_error(str(exc))
    finally:
        if acquired_here:
            _release_cache_warm_shared_lock(shared_lock)
        _set_cache_warm_state(
            inflight=False,
            current_queue=None,
            completed_at=datetime.now(timezone.utc).isoformat(),
            shared_lock=None,
        )


def _start_cache_warm(source: str | None, site_key: str | None, limit: int | None) -> tuple[bool, dict[str, Any]]:
    with _cache_warm_lock:
        if _cache_warm_state.get("inflight"):
            state = dict(_cache_warm_state)
            state["errors"] = list(_cache_warm_state.get("errors", []))
            return False, state
        _cache_warm_state["inflight"] = True
        _cache_warm_state["stop_requested"] = False

    shared_lock = _acquire_cache_warm_shared_lock()
    if shared_lock is None:
        _set_cache_warm_state(
            inflight=False,
            current_queue=None,
            completed_at=datetime.now(timezone.utc).isoformat(),
            last_error="Another cache warm worker is already running.",
            shared_lock=_read_cache_warm_shared_lock(),
            shared_lock_path=str(_cache_warm_shared_lock_path()),
        )
        return False, _cache_warm_state_snapshot()

    _set_cache_warm_state(
        shared_lock=shared_lock,
        shared_lock_path=str(_cache_warm_shared_lock_path()),
    )
    _cache_warm_executor.submit(_run_cache_warm_background, source, site_key, limit, shared_lock, True)
    return True, _cache_warm_state_snapshot()


def _set_ready_maintainer_state(**updates: Any) -> None:
    with _ready_maintainer_lock:
        _ready_maintainer_state.update(updates)


def _ready_maintainer_state_snapshot() -> dict[str, Any]:
    with _ready_maintainer_lock:
        state = dict(_ready_maintainer_state)
    with _cache_warm_lock:
        state["cache_warming"] = bool(_cache_warm_state.get("inflight"))
    return state


def _run_ready_maintainer_once() -> None:
    shared_lock = _acquire_ready_maintainer_shared_lock()
    if shared_lock is None:
        return

    client = DriveClient()
    generated_total = 0
    last_error = None
    try:
        for label_source in LABEL_SOURCES:
            _set_ready_maintainer_state(current_queue=label_source.queue_key)
            try:
                context = _resolve_queue_context(
                    client,
                    label_source.source,
                    label_source.site_key,
                )
                subfolders = _list_source_subfolders(client, context)
                labeled_records = _label_history_records_for_queue(context.queue_key)
                subfolders, _history_hidden = _filter_label_history_hidden_subfolders(
                    subfolders,
                    context,
                    labeled_records,
                )
                _ready_folders, ready_stats = _collect_ready_folders(
                    subfolders,
                    context,
                    limit=1,
                    labeled_records=labeled_records,
                )
                visible_count = max(
                    0,
                    len(subfolders)
                    - int(ready_stats["hidden_labeled"])
                    - int(ready_stats["duplicate_signatures"]),
                )
                if context.source == VIDEO_SOURCE:
                    if AUTOLABEL_VIDEO_AUTO_PREPROCESS:
                        _maybe_trigger_video_preprocess(context, visible_count)
                elif context.source == REOLINK_SOURCE:
                    generated = _prepare_reolink_unlabeled_queue(
                        client,
                        context,
                        target_unlabeled_count=REOLINK_PREWARM_TARGET,
                        current_visible_count=visible_count,
                    )
                    generated_total += generated
                    if generated:
                        _invalidate_listing_cache(context.queue_key)
            except CropSetupRequiredError as exc:
                last_error = str(exc)
            except Exception as exc:
                last_error = f"{label_source.queue_key}: {exc}"

        _start_cache_warm(None, None, REOLINK_PREWARM_TARGET)
        _set_ready_maintainer_state(
            current_queue=None,
            last_run_at=time.time(),
            generated=generated_total,
            last_error=last_error,
        )
    finally:
        _release_ready_maintainer_shared_lock(shared_lock)


def _ready_maintainer_startup_enabled() -> bool:
    if app.testing or "pytest" in sys.modules:
        return False
    argv0 = Path(sys.argv[0]).name
    if argv0 == "main.py" and "--label" not in sys.argv:
        return False
    return os.environ.get("LABEL_READY_MAINTAINER_ON_STARTUP", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _auto_start_ready_maintainer() -> bool:
    if not _ready_maintainer_startup_enabled():
        return False
    return _ensure_ready_maintainer_started()


def _run_ready_maintainer_loop() -> None:
    while True:
        _set_ready_maintainer_state(inflight=True, started=True)
        try:
            _run_ready_maintainer_once()
        except Exception as exc:
            _set_ready_maintainer_state(last_error=str(exc), current_queue=None)
            print(f"[ready-maintainer] run failed: {exc}")
        finally:
            _set_ready_maintainer_state(inflight=False)
        time.sleep(_READY_MAINTAINER_INTERVAL_SECONDS)


def _ensure_ready_maintainer_started() -> bool:
    global _ready_maintainer_started
    if app.testing:
        return False
    with _ready_maintainer_lock:
        if _ready_maintainer_started:
            return False
        _ready_maintainer_started = True
        _ready_maintainer_state["started"] = True
    _ready_maintainer_executor.submit(_run_ready_maintainer_loop)
    return True


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
        "unlabeled": sum(len(client.list_folders(folder_id)) for folder_id in _context_input_folder_ids(context)),
    }
    for name in LABEL_DESTINATIONS:
        stats[name] = len(client.list_folders(context.folder_ids[name]))
    return stats


def _collect_ready_folders(
    subfolders: list[dict[str, str]],
    context: QueueContext,
    limit: int,
    labeled_records: dict[str, Any],
) -> tuple[list[dict], dict[str, int | float]]:
    request_started = time.perf_counter()
    ready: list[dict] = []
    fallback: list[dict] = []
    seen_signatures: set[str] = set()
    nonready = 0
    hidden_labeled = 0
    duplicate_signatures = 0
    hydrated_valid = 0
    scanned = 0
    hydrate_ms = 0.0
    hydrate_requested = 0
    hydrate_cache_hits = 0
    hydrate_cache_misses = 0
    hydrate_worker_max = 0
    budget_exhausted = False
    first_unready_idx = len(subfolders)

    target_scan = min(len(subfolders), max(limit * READY_SCAN_MULTIPLIER, limit))
    target_scan = min(target_scan, READY_SCAN_MAX)

    while scanned < target_scan and len(ready) < limit:
        if scanned > 0 and (time.perf_counter() - request_started) * 1000 >= QUEUE_HYDRATE_BUDGET_MS:
            budget_exhausted = True
            break
        remaining_scan = target_scan - scanned
        batch_span = min(
            remaining_scan,
            QUEUE_HYDRATE_BATCH_SIZE,
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
            signature = str(payload.get("frame_signature") or "")
            if signature and signature in seen_signatures:
                duplicate_signatures += 1
                continue
            history_record = _label_history_lookup_in_records(
                labeled_records,
                context,
                str(payload.get("folder_id") or ""),
                str(payload.get("folder_name") or ""),
                signature,
                str(payload.get("content_signature") or ""),
            )
            if history_record:
                hidden_labeled += 1
                _remove_folder_from_listing_cache(context.queue_key, str(payload.get("folder_id") or ""))
                with _hydrated_folder_cache_lock:
                    _hydrated_folder_cache.pop(
                        _hydrated_cache_key(context.queue_key, str(payload.get("folder_id") or "")),
                        None,
                    )
                _schedule_hidden_folder_cleanup(context, str(payload.get("folder_id") or ""))
                continue
            if signature:
                seen_signatures.add(signature)
            hydrated_valid += 1
            payload["cache_ready"] = _folder_cache_ready(payload)
            if payload["cache_ready"]:
                ready.append(payload)
            else:
                nonready += 1
                if len(fallback) < limit:
                    fallback.append(payload)
                first_unready_idx = min(first_unready_idx, absolute_idx)
                _schedule_preview_prewarm([payload])
            if len(ready) >= limit:
                break

        scanned += len(folder_batch)
        if not ready and len(fallback) >= limit:
            break
        if (time.perf_counter() - request_started) * 1000 >= QUEUE_HYDRATE_BUDGET_MS:
            budget_exhausted = True
            break

    prewarm_scan_start = first_unready_idx if first_unready_idx < len(subfolders) else scanned
    folder_prewarm_scheduled = _schedule_folder_hydration_prewarm(
        subfolders,
        prewarm_scan_start,
        context,
    )

    returned = ready if ready else fallback
    return returned, {
        "scanned": scanned,
        "hydrate_ms": hydrate_ms,
        "hydrate_requested": hydrate_requested,
        "hydrate_cache_hits": hydrate_cache_hits,
        "hydrate_cache_misses": hydrate_cache_misses,
        "hydrate_worker_max": hydrate_worker_max,
        "folder_prewarm_scheduled": folder_prewarm_scheduled,
        "prewarm_scan_start": prewarm_scan_start,
        "hydrated_valid": hydrated_valid,
        "hidden_labeled": hidden_labeled,
        "duplicate_signatures": duplicate_signatures,
        "nonready": nonready,
        "returned_uncached": 0 if ready else len(fallback),
        "budget_exhausted": int(budget_exhausted),
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


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True})


@app.route("/login", methods=["GET", "POST"])
def login():
    if not AUTH_REQUIRED:
        return redirect(url_for("index"))

    error = None
    next_url = request.args.get("next") or request.form.get("next") or url_for("index")
    if not next_url.startswith("/"):
        next_url = url_for("index")

    if request.method == "POST":
        configured_password = LABELER_PASSWORD
        supplied_password = request.form.get("password", "")
        if configured_password and hmac.compare_digest(configured_password, supplied_password):
            labeler_name = request.form.get("labeler_name", "").strip() or "labeler"
            session.clear()
            session["authenticated"] = True
            session["labeler_name"] = labeler_name[:80]
            _csrf_token()
            return redirect(next_url)
        error = "Invalid password."

    if not LABELER_PASSWORD:
        error = "LABELER_PASSWORD is not configured."
    return render_template("login.html", error=error, next_url=next_url)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login") if AUTH_REQUIRED else url_for("index"))


@app.route("/")
def index():
    _schedule_label_job_worker()
    return render_template("label.html")


@app.route("/cleanup/legacy")
def legacy_cleanup():
    return render_template(
        "review.html",
        review_mode="legacy",
        page_title="Legacy Cleanup",
    )


@app.route("/cleanup/crops")
def crop_cleanup():
    return render_template("crop_cleanup.html")


@app.route("/review/labeled")
def labeled_review():
    return render_template(
        "review.html",
        review_mode="labeled",
        page_title="Labeled Review",
    )


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
                "source": REOLINK_SOURCE if reolink_sites else VIDEO_SOURCE,
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
        subfolders = _list_source_subfolders(
            client,
            context,
            force_refresh=request.args.get("refresh", "0") == "1",
        )
        if context.source == REOLINK_SOURCE:
            _maybe_trigger_reolink_preprocess(
                context,
                len(subfolders),
                INTERACTIVE_REOLINK_PREWARM_TARGET,
            )
        result = [
            {
                "folder_id": f["id"],
                "folder_name": f["name"],
                "parent_id": next(iter(f.get("parents", []) or []), context.input_folder_id),
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
    _schedule_label_job_worker()
    try:
        client = get_client()
        source, site_key = _request_source_args()
        context = _resolve_queue_context(client, source, site_key)

        limit = int(request.args.get("limit", str(QUEUE_BATCH_DEFAULT)) or str(QUEUE_BATCH_DEFAULT))
        limit = max(1, min(limit, QUEUE_BATCH_MAX))
        include_stats = request.args.get("include_stats", "0") == "1"
        force_refresh = request.args.get("refresh", "0") == "1"
        list_started = time.perf_counter()
        subfolders = _list_source_subfolders(client, context, force_refresh=force_refresh)
        list_ms = (time.perf_counter() - list_started) * 1000
        labeled_records = _label_history_records_for_queue(context.queue_key)
        subfolders, history_hidden = _filter_label_history_hidden_subfolders(
            subfolders,
            context,
            labeled_records,
        )
        total_unlabeled = len(subfolders)

        ready_folders, ready_stats = _collect_ready_folders(
            subfolders,
            context,
            limit,
            labeled_records,
        )
        visible_unlabeled_estimate = max(
            0,
            total_unlabeled
            - int(ready_stats["hidden_labeled"])
            - int(ready_stats["duplicate_signatures"]),
        )
        visible_count_for_refill = (
            visible_unlabeled_estimate
            if len(ready_folders) >= limit
            else len(ready_folders) + int(ready_stats["nonready"])
        )

        if context.source == VIDEO_SOURCE:
            _maybe_trigger_video_preprocess(context, visible_count_for_refill)
        elif context.source == REOLINK_SOURCE:
            _maybe_trigger_reolink_preprocess(
                context,
                visible_count_for_refill,
                INTERACTIVE_REOLINK_PREWARM_TARGET,
            )

        preview_prewarm_scheduled = _schedule_preview_prewarm(ready_folders)
        ready_buffer_count = sum(1 for folder in ready_folders if folder.get("cache_ready"))
        returned_count = len(ready_folders)
        warming_count = int(ready_stats["nonready"])

        response: dict[str, object] = {
            "folders": ready_folders,
            "next_cursor": 0,
            "source_context": context.to_payload(),
            "total_unlabeled": total_unlabeled,
            "visible_unlabeled_estimate": visible_unlabeled_estimate,
            "ready_target": REOLINK_PREWARM_TARGET,
            "has_more": total_unlabeled > returned_count,
            "ready_buffer_count": ready_buffer_count,
            "warming_count": warming_count,
            "retry_ms": QUEUE_RETRY_MS if total_unlabeled > 0 and returned_count < limit else 0,
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
            returned_uncached=ready_stats["returned_uncached"],
            warming=warming_count,
            scanned=ready_stats["scanned"],
            hydrated_valid=ready_stats["hydrated_valid"],
            hidden_labeled=ready_stats["hidden_labeled"],
            history_hidden=history_hidden,
            duplicate_signatures=ready_stats["duplicate_signatures"],
            visible_unlabeled=visible_unlabeled_estimate,
            cache_hits=ready_stats["hydrate_cache_hits"],
            cache_misses=ready_stats["hydrate_cache_misses"],
            workers=ready_stats["hydrate_worker_max"],
            budget_exhausted=ready_stats["budget_exhausted"],
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


@app.route("/api/review/folders")
def api_review_folders():
    try:
        client = get_client()
        source, site_key = _request_source_args()
        context = _resolve_queue_context(client, source, site_key)
        mode = str(request.args.get("mode") or "labeled").strip().lower()
        default_buckets = REVIEW_LEGACY_DEFAULT_BUCKETS if mode == "legacy" else REVIEW_LABELED_DEFAULT_BUCKETS
        buckets = _parse_csv_arg("bucket", default_buckets)
        limit = max(1, min(120, int(request.args.get("limit", "30") or "30")))
        cursor = max(0, int(request.args.get("cursor", "0") or "0"))
        filters = {
            "q": str(request.args.get("q") or "").strip(),
            "channel": str(request.args.get("channel") or "").strip(),
            "table": str(request.args.get("table") or "").strip(),
            "frame_count": str(request.args.get("frame_count") or "").strip(),
            "folder_source_type": str(request.args.get("folder_source_type") or "").strip(),
            "crop_source_kind": str(request.args.get("crop_source_kind") or "").strip(),
        }
        page, next_cursor, candidate_total = _review_list_folders(
            client,
            context,
            buckets,
            filters,
            limit=limit,
            cursor=cursor,
        )
        return jsonify(
            {
                "folders": page,
                "next_cursor": next_cursor,
                "total": candidate_total,
                "total_is_candidate_count": True,
                "source_context": context.to_payload(),
                "buckets": buckets,
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (DriveClientError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cleanup/crops/inventory")
def api_cleanup_crops_inventory():
    try:
        client = get_client()
        source, site_key = _request_source_args()
        context = _resolve_queue_context(client, source, site_key)
        buckets = _parse_csv_arg("bucket", CROP_CLEANUP_DEFAULT_BUCKETS)
        filters = {
            "q": str(request.args.get("q") or "").strip(),
            "channel": str(request.args.get("channel") or "").strip(),
            "table": str(request.args.get("table") or "").strip(),
            "frame_count": str(request.args.get("frame_count") or "").strip(),
            "folder_source_type": str(request.args.get("folder_source_type") or "").strip(),
            "crop_source_kind": str(request.args.get("crop_source_kind") or "").strip(),
        }
        supabase_cards = [
            card
            for card in _cleanup_supabase_crop_cards(client, context)
            if _cleanup_card_matches_filters(card, filters)
        ]
        fallback_groups = _cleanup_fallback_groups(client, context, buckets, filters)
        return jsonify(
            {
                "source_context": context.to_payload(),
                "supabase_crops": supabase_cards,
                "fallback_groups": fallback_groups,
                "buckets": buckets,
                "counts": {
                    "supabase_crops": len(supabase_cards),
                    "fallback_groups": len(fallback_groups),
                    "fallback_folders": sum(int(group.get("folder_count") or 0) for group in fallback_groups),
                },
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (DriveClientError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cleanup/crops/trash", methods=["POST"])
def api_cleanup_crops_trash():
    try:
        data = _request_json_payload()
        source, site_key = _payload_source_args(data)
        folder_ids = [str(item).strip() for item in (data.get("folder_ids") or []) if str(item).strip()]
        confirm = str(data.get("confirm") or "").strip()
        if confirm != "TRASH":
            return jsonify({"error": "confirm must be TRASH"}), 400
        if not folder_ids:
            return jsonify({"error": "folder_ids required"}), 400
        if len(folder_ids) > 500:
            return jsonify({"error": "at most 500 folders can be trashed at once"}), 400

        client = get_client()
        context = _resolve_queue_context(client, source, site_key)
        validated: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for folder_id in folder_ids:
            current = client.get_file(folder_id, fields="id,name,parents,appProperties")
            try:
                parent_id = _review_validate_folder_parent(context, current)
                bucket = _review_bucket_for_parent(context, parent_id) or "current"
                if not _cleanup_folder_matches_context(
                    context,
                    str(current.get("name") or ""),
                    dict(current.get("appProperties") or {}),
                    bucket,
                ):
                    raise ValueError("folder does not belong to the selected cleanup source")
                payload = _review_payload_for_folder(client, context, current, bucket, parent_id)
                if payload is None or str(payload.get("crop_source_kind") or "") not in CROP_CLEANUP_FALLBACK_KINDS:
                    raise ValueError("folder is not a fallback/manual JSON crop artifact")
                validated.append({"folder_id": folder_id, "parent_id": parent_id, "payload": payload})
            except Exception as exc:
                rejected.append({"folder_id": folder_id, "error": str(exc)})

        if rejected:
            return jsonify({"error": "one or more folders are not safe to trash", "rejected": rejected}), 400

        results: list[dict[str, Any]] = []
        for item in validated:
            folder_id = item["folder_id"]
            payload = item["payload"]
            folder_name, frames, frame_signature = _review_signature_for_current_folder(client, folder_id)
            content_signature = _content_signature_from_frames(frames) if has_complete_frame_ids(frames) else ""
            client.trash_file(folder_id)
            _remove_label_history(context, folder_id, folder_name, frame_signature, content_signature)
            _clear_label_queue_caches(context, folder_id)
            results.append(
                {
                    "folder_id": folder_id,
                    "folder_name": payload.get("folder_name") or folder_name,
                    "trashed": True,
                    "crop_source_kind": payload.get("crop_source_kind"),
                }
            )
        return jsonify(
            {
                "ok": True,
                "trashed": len(results),
                "results": results,
                "source_context": context.to_payload(),
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (DriveClientError, RuntimeError, OSError, TypeError) as e:
        return jsonify({"error": str(e), "code": "cleanup_crop_trash_failed"}), 500


@app.route("/api/review/relabel", methods=["POST"])
def api_review_relabel():
    try:
        data = _request_json_payload()
        source, site_key = _payload_source_args(data)
        target_label = str(data.get("target_label") or data.get("label") or "").strip().lower()
        folder_ids = [str(item).strip() for item in (data.get("folder_ids") or []) if str(item).strip()]
        if target_label not in LABEL_DESTINATIONS:
            return jsonify({"error": f"target_label must be one of {', '.join(LABEL_DESTINATIONS)}"}), 400
        if not folder_ids:
            return jsonify({"error": "folder_ids required"}), 400
        if len(folder_ids) > 200:
            return jsonify({"error": "at most 200 folders can be relabeled at once"}), 400

        client = get_client()
        context = _resolve_queue_context(client, source, site_key)
        destination_id = context.folder_ids[target_label]
        results: list[dict[str, Any]] = []
        for folder_id in folder_ids:
            current = client.get_file(folder_id, fields="id,name,parents,appProperties")
            current_parent = _review_validate_folder_parent(context, current)
            if current_parent != destination_id:
                client.move_file(folder_id, new_parent_id=destination_id, remove_parent_id=current_parent)
            label_metadata = dict(current.get("appProperties") or {})
            label_metadata.update(_label_app_properties(target_label, context))
            client.update_file_metadata(
                folder_id,
                {"appProperties": label_metadata},
                fields="id,name,mimeType,parents,appProperties",
            )
            folder_name, frames, frame_signature = _review_signature_for_current_folder(client, folder_id)
            content_signature = _content_signature_from_frames(frames) if has_complete_frame_ids(frames) else ""
            _record_label_history(
                context,
                folder_id,
                folder_name,
                frame_signature,
                target_label,
                content_signature,
            )
            _clear_label_queue_caches(context, folder_id)
            results.append({"folder_id": folder_id, "label": target_label, "moved": current_parent != destination_id})
        return jsonify({"ok": True, "updated": len(results), "results": results, "source_context": context.to_payload()})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (DriveClientError, RuntimeError, OSError, TypeError) as e:
        return jsonify({"error": str(e), "code": "review_relabel_failed"}), 500


@app.route("/api/review/trash", methods=["POST"])
def api_review_trash():
    try:
        data = _request_json_payload()
        source, site_key = _payload_source_args(data)
        folder_ids = [str(item).strip() for item in (data.get("folder_ids") or []) if str(item).strip()]
        confirm = str(data.get("confirm") or "").strip()
        if confirm != "TRASH":
            return jsonify({"error": "confirm must be TRASH"}), 400
        if not folder_ids:
            return jsonify({"error": "folder_ids required"}), 400
        if len(folder_ids) > 200:
            return jsonify({"error": "at most 200 folders can be trashed at once"}), 400

        client = get_client()
        context = _resolve_queue_context(client, source, site_key)
        results: list[dict[str, Any]] = []
        for folder_id in folder_ids:
            current = client.get_file(folder_id, fields="id,name,parents,appProperties")
            _review_validate_folder_parent(context, current)
            folder_name, frames, frame_signature = _review_signature_for_current_folder(client, folder_id)
            content_signature = _content_signature_from_frames(frames) if has_complete_frame_ids(frames) else ""
            client.trash_file(folder_id)
            _remove_label_history(context, folder_id, folder_name, frame_signature, content_signature)
            _clear_label_queue_caches(context, folder_id)
            results.append({"folder_id": folder_id, "trashed": True})
        return jsonify({"ok": True, "trashed": len(results), "results": results, "source_context": context.to_payload()})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (DriveClientError, RuntimeError, OSError, TypeError) as e:
        return jsonify({"error": str(e), "code": "review_trash_failed"}), 500


@app.route("/api/review/compare")
def api_review_compare():
    try:
        client = get_client()
        source, site_key = _request_source_args()
        context = _resolve_queue_context(client, source, site_key)
        folder_id = str(request.args.get("folder_id") or "").strip()
        if not folder_id:
            return jsonify({"error": "folder_id required"}), 400

        current = client.get_file(folder_id, fields="id,name,parents,appProperties,modifiedTime")
        _review_validate_folder_parent(context, current)
        current_parent = _review_current_parent(current) or ""
        current_payload = _review_payload_for_folder(client, context, current, "current", current_parent)
        if current_payload is None:
            return jsonify({"matches": [], "folder": None})
        channel = str(current_payload.get("channel_hint") or "").lower()
        table = str(current_payload.get("table_hint") or "").lower()
        name_bits = [
            bit.lower()
            for bit in re.split(r"[^A-Za-z0-9]+", str(current_payload.get("folder_name") or ""))
            if len(bit) >= 3
        ]

        candidates, _next_cursor, _candidate_total = _review_list_folders(
            client,
            context,
            list(REVIEW_LEGACY_DEFAULT_BUCKETS),
            {
                "q": "",
                "channel": "",
                "table": "",
                "frame_count": "",
                "folder_source_type": "",
                "crop_source_kind": "",
            },
            limit=120,
            cursor=0,
        )
        scored: list[tuple[int, dict[str, Any]]] = []
        current_kind = str(current_payload.get("crop_source_kind") or "")
        for candidate in candidates:
            if candidate.get("folder_id") == folder_id:
                continue
            score = 0
            candidate_kind = str(candidate.get("crop_source_kind") or "")
            if current_kind == "supabase" and candidate_kind in {"fallback_json", "drive_crop_config"}:
                score += 12
            elif current_kind in {"fallback_json", "drive_crop_config"} and candidate_kind == "supabase":
                score += 12
            candidate_name = str(candidate.get("folder_name") or "").lower()
            if channel and channel == str(candidate.get("channel_hint") or "").lower():
                score += 5
            if table and table == str(candidate.get("table_hint") or "").lower():
                score += 4
            score += min(3, sum(1 for bit in name_bits if bit in candidate_name))
            if score > 0:
                scored.append((score, candidate))
        scored.sort(key=lambda item: (item[0], str(item[1].get("modified_time") or "")), reverse=True)
        return jsonify({"folder": current_payload, "matches": [item for _score, item in scored[:12]]})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (DriveClientError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cache/status")
def api_cache_status():
    cache_dir = _ensure_cache_dir()
    include_counts = request.args.get("scan", "0") == "1"
    writable = False
    error = None
    try:
        probe = cache_dir / ".cache_status_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        writable = True
    except OSError as exc:
        error = str(exc)

    full_res_count = 0
    thumb_count = 0
    total_bytes = 0
    if include_counts:
        try:
            for path in cache_dir.glob("*.jpg"):
                try:
                    total_bytes += path.stat().st_size
                except OSError:
                    pass
                if path.name.endswith(".thumb.jpg"):
                    thumb_count += 1
                else:
                    full_res_count += 1
        except OSError as exc:
            error = error or str(exc)

    temp_root = Path(tempfile.gettempdir()).resolve()
    with _hydrated_folder_cache_lock:
        hydrated_cache_entries = len(_hydrated_folder_cache)
        hydrated_hot_entries = sum(
            1
            for _cached_at, payload in _hydrated_folder_cache.values()
            if isinstance(payload, dict) and payload.get("cache_ready")
        )
    try:
        resolved_cache_dir = cache_dir.resolve()
    except OSError:
        resolved_cache_dir = cache_dir
    railway_env = _RAILWAY_ENV
    configured_cache = os.environ.get("LABEL_CACHE_DIR", "").strip()
    uses_temp_cache = str(resolved_cache_dir).startswith(str(temp_root))
    expected_volume_path = "/data/label_cache"
    preprocess_dir = _preprocess_state_dir()
    production_warning = None
    if railway_env and not str(resolved_cache_dir).startswith("/data/"):
        production_warning = "Railway cache is not under /data; cached images may not survive redeploys."
    elif uses_temp_cache:
        production_warning = "Cache is under the system temp directory; use LABEL_CACHE_DIR for persistence."

    return jsonify(
        {
            "cache_dir": str(cache_dir),
            "writable": writable,
            "configured_cache_dir": configured_cache or None,
            "expected_volume_cache_dir": expected_volume_path,
            "preprocess_state_dir": str(preprocess_dir),
            "label_history_path": str(_label_history_path()),
            "label_jobs_path": str(_label_jobs_path()),
            "label_jobs": _label_jobs_status_payload(),
            "railway_environment": railway_env,
            "uses_temp_cache": uses_temp_cache,
            "cache_max_mb": CACHE_MAX_MB,
            "cache_ttl_hours": CACHE_TTL_HOURS,
            "ready_target": REOLINK_PREWARM_TARGET,
            "label_ready_target_configured": LABEL_READY_TARGET_CONFIGURED,
            "hydrated_cache_entries": hydrated_cache_entries,
            "hydrated_hot_entries": hydrated_hot_entries,
            "scan_included": include_counts,
            "full_res_count": full_res_count if include_counts else None,
            "thumb_count": thumb_count if include_counts else None,
            "size_mb": round(total_bytes / (1024 * 1024), 2) if include_counts else None,
            "error": error,
            "production_warning": production_warning,
        }
    )


@app.route("/api/cache/warm", methods=["POST"])
def api_cache_warm_start():
    data = request.get_json(silent=True) or {}
    source = (data.get("source") or request.args.get("source") or "").strip().lower() or None
    site_key = (data.get("site_key") or data.get("site") or request.args.get("site") or "").strip() or None
    raw_limit = data.get("limit", request.args.get("limit"))
    limit = None
    if raw_limit not in (None, ""):
        try:
            limit = max(1, int(raw_limit))
        except (TypeError, ValueError):
            return jsonify({"error": "limit must be a positive integer"}), 400
    elif LABEL_READY_TARGET_CONFIGURED:
        limit = REOLINK_PREWARM_TARGET

    try:
        if source:
            _resolve_label_source(source, site_key)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    started, state = _start_cache_warm(source, site_key, limit)
    return jsonify({"started": started, "state": state}), 202 if started else 409


@app.route("/api/cache/warm/status")
def api_cache_warm_status():
    return jsonify(_cache_warm_state_snapshot())


@app.route("/api/cache/warm/cancel", methods=["POST"])
def api_cache_warm_cancel():
    return jsonify({"stop_requested": True, "state": _request_cache_warm_stop()})


@app.route("/api/label/jobs/status")
def api_label_jobs_status():
    _schedule_label_job_worker()
    verify = str(request.args.get("verify") or "").strip().lower() in {"1", "true", "yes", "on"}
    return jsonify(_label_jobs_status_payload(verify=verify, client=get_client() if verify else None))


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


@app.route("/api/thumb/<file_id>")
def api_thumb(file_id: str):
    """Serve a downscaled JPEG (512px wide by default) for fast buffer warming.

    Reuses the full-res cache at {file_id}.jpg and writes a parallel
    {file_id}.thumb.jpg on first hit. Both share CACHE_DIR's LRU/TTL cleanup.
    """
    request_started = time.perf_counter()
    _cleanup_cache_if_needed()

    try:
        thumb_path, cache_hit, download_ms, encode_ms = _ensure_thumb_for_file(file_id, get_client())
    except DriveClientError as e:
        abort(404, description=str(e))
    except Exception as e:
        abort(500, description=f"Thumbnail generation failed: {e}")

    try:
        os.utime(thumb_path, None)
    except OSError:
        pass

    try:
        size_bytes = thumb_path.stat().st_size
    except OSError:
        size_bytes = 0

    total_ms = (time.perf_counter() - request_started) * 1000
    _log_timing(
        "api_thumb",
        total_ms=f"{total_ms:.1f}",
        download_ms=f"{download_ms:.1f}",
        encode_ms=f"{encode_ms:.1f}",
        cache="hit" if cache_hit else "miss",
        size_kb=f"{size_bytes / 1024:.1f}",
        file_id=file_id,
    )
    return send_file(thumb_path, mimetype="image/jpeg", conditional=True, max_age=3600)


@app.route("/api/label", methods=["POST"])
def api_label():
    """Record a label intent durably, then move the folder on Drive in the background."""
    request_started = time.perf_counter()
    folder_id = ""
    folder_name = ""
    label = ""
    source = ""
    site_key = None
    queue_key = ""
    try:
        data = _request_json_payload()
        folder_id = str(data.get("folder_id", "")).strip()
        parent_id = str(data.get("parent_id", "")).strip()
        label = str(data.get("label", "")).strip().lower()
        source, site_key = _payload_source_args(data)

        if not folder_id or not parent_id:
            return jsonify({"error": "folder_id and parent_id required"}), 400
        if label not in LABEL_DESTINATIONS:
            return jsonify({"error": f"label must be one of {', '.join(LABEL_DESTINATIONS)}"}), 400

        context = _resolve_queue_context(get_client(), source, site_key)
        queue_key = context.queue_key
        if parent_id not in _context_input_folder_ids(context):
            return jsonify({"error": "parent_id does not match the active queue"}), 400

        raw_frames = data.get("frames") if isinstance(data.get("frames"), dict) else {}
        frames = _frames_from_client_payload(raw_frames)
        frame_signature = str(data.get("frame_signature") or "").strip()
        if not frame_signature:
            frame_signature = _frame_signature_from_frames(frames)
        if not has_complete_frame_ids(frames):
            frames = {key: None for key in _ordered_frame_keys(frames)}
        folder_name = str(data.get("folder_name") or "").strip()
        content_signature = str(data.get("content_signature") or "").strip()
        if not content_signature and has_complete_frame_ids(frames):
            content_signature = _content_signature_from_frames(frames)

        existing_history = _label_history_lookup(context, folder_id, folder_name, frame_signature, content_signature)
        existing_job = _get_label_job(_label_job_key(context, folder_id))
        can_replace_pending = (
            isinstance(existing_job, dict)
            and existing_job.get("status") == "pending"
            and not _label_job_is_due(existing_job)
        )
        if existing_history and not can_replace_pending:
            _remove_folder_from_listing_cache(context.queue_key, folder_id)
            return jsonify({"error": "already_labeled", "code": "already_labeled"}), 409

        _record_label_history(
            context,
            folder_id,
            folder_name,
            frame_signature,
            label,
            content_signature,
        )
        job = _enqueue_label_job(
            context,
            folder_id=folder_id,
            parent_id=parent_id,
            folder_name=folder_name,
            frames=frames,
            frame_signature=frame_signature,
            content_signature=content_signature,
            label=label,
        )
        with _hydrated_folder_cache_lock:
            _hydrated_folder_cache.pop(_hydrated_cache_key(context.queue_key, folder_id), None)
        _remove_folder_from_listing_cache(context.queue_key, folder_id)
        worker_started = _schedule_label_job_worker()
        total_ms = (time.perf_counter() - request_started) * 1000
        _log_timing(
            "api_label",
            total_ms=f"{total_ms:.1f}",
            label=label,
            folder_id=folder_id,
            queue=context.queue_key,
            job=str(job.get("id") or ""),
            worker_started=worker_started,
        )
        return jsonify(
            {
                "ok": True,
                "queued": True,
                "job_id": job.get("id"),
                "not_before": job.get("not_before"),
                "undo_expires_at": job.get("undo_expires_at"),
                "queued_label": label,
                "drive_move_status": "queued",
                "source_context": context.to_payload(),
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (DriveClientError, RuntimeError, OSError, TypeError) as e:
        _log_label_route_error(
            e,
            folder_id=folder_id,
            folder_name=folder_name,
            label=label,
            source=source,
            site_key=site_key,
            queue_key=queue_key,
        )
        return jsonify({"error": str(e), "code": "label_queue_failed"}), 500
    except Exception as e:
        _log_label_route_error(
            e,
            folder_id=folder_id,
            folder_name=folder_name,
            label=label,
            source=source,
            site_key=site_key,
            queue_key=queue_key,
        )
        return jsonify({"error": "Internal server error", "code": "internal_error"}), 500


@app.route("/api/label/cancel", methods=["POST"])
def api_label_cancel():
    folder_id = ""
    folder_name = ""
    source = ""
    site_key = None
    queue_key = ""
    try:
        data = _request_json_payload()
        folder_id = str(data.get("folder_id", "")).strip()
        parent_id = str(data.get("parent_id", "")).strip()
        source, site_key = _payload_source_args(data)
        if not folder_id:
            return jsonify({"error": "folder_id required"}), 400

        client = get_client()
        context = _resolve_queue_context(client, source, site_key)
        queue_key = context.queue_key
        raw_frames = data.get("frames") if isinstance(data.get("frames"), dict) else {}
        frames = _frames_from_client_payload(raw_frames)
        frame_signature = str(data.get("frame_signature") or "").strip()
        if not frame_signature:
            frame_signature = _frame_signature_from_frames(frames)
        folder_name = str(data.get("folder_name") or "").strip()
        content_signature = str(data.get("content_signature") or "").strip()
        result = _cancel_label_job(
            context,
            client=client,
            folder_id=folder_id,
            parent_id=parent_id,
            folder_name=folder_name,
            frame_signature=frame_signature,
            content_signature=content_signature,
        )
        if not result["canceled"]:
            return jsonify({"error": "label job is no longer undoable", "code": "not_undoable"}), 409
        response = {"ok": True, "canceled": True, "source_context": context.to_payload()}
        if result["restored"]:
            response["restored"] = True
        return jsonify(response)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (DriveClientError, RuntimeError, OSError, TypeError) as e:
        _log_label_route_error(
            e,
            folder_id=folder_id,
            folder_name=folder_name,
            source=source,
            site_key=site_key,
            queue_key=queue_key,
        )
        return jsonify({"error": str(e), "code": "label_cancel_failed"}), 500
    except Exception as e:
        _log_label_route_error(
            e,
            folder_id=folder_id,
            folder_name=folder_name,
            source=source,
            site_key=site_key,
            queue_key=queue_key,
        )
        return jsonify({"error": "Internal server error", "code": "internal_error"}), 500


@app.route("/api/stats")
def api_stats():
    """Return counts of folders in each category."""
    request_started = time.perf_counter()
    try:
        client = get_client()
        source, site_key = _request_source_args()
        context = _resolve_queue_context(client, source, site_key)
        stats = _compute_stats(client, context)
        if context.source == VIDEO_SOURCE:
            _maybe_trigger_video_preprocess(context, stats.get("unlabeled", 0))
        elif context.source == REOLINK_SOURCE:
            _maybe_trigger_reolink_preprocess(
                context,
                stats.get("unlabeled", 0),
                INTERACTIVE_REOLINK_PREWARM_TARGET,
            )
        total_ms = (time.perf_counter() - request_started) * 1000
        _log_timing("api_stats", total_ms=f"{total_ms:.1f}", queue=context.queue_key, **stats)
        return jsonify({**stats, "source_context": context.to_payload()})
    except CropSetupRequiredError as e:
        return jsonify(e.to_payload()), 409
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (DriveClientError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500


def _run_video_preprocess_background(max_videos: int) -> None:
    """Worker body for the background video preprocess executor."""
    global _video_preprocess_state

    try:
        from processor import run_processor

        tables_json_path = Path(__file__).parent / "approved_table_rectangles.json"
        summary = run_processor(
            project_root_id=_root_id(),
            tables_json_path=tables_json_path,
            client=DriveClient(),
            max_videos=max_videos,
        )
        with _video_preprocess_lock:
            _video_preprocess_state.update(
                last_run_at=time.time(),
                last_run_videos=summary.videos_with_new_work,
                last_run_triplets=summary.triplets_uploaded,
                last_error=None,
            )
    except Exception as exc:
        with _video_preprocess_lock:
            _video_preprocess_state.update(
                last_run_at=time.time(),
                last_error=str(exc),
            )
        print(f"[auto-preprocess] video run failed: {exc}")
    finally:
        # Invalidate the video listing cache so the new folders show up.
        _invalidate_listing_cache(VIDEO_SOURCE)
        with _video_preprocess_lock:
            _video_preprocess_state["inflight"] = False


def _maybe_trigger_video_preprocess(context: QueueContext, unlabeled_count: int) -> None:
    """Kick off a background video preprocess run when the queue is drained."""
    if AUTOLABEL_VIDEO_LOW_WATERMARK <= 0:
        return
    if context.source != VIDEO_SOURCE:
        return
    if unlabeled_count >= AUTOLABEL_VIDEO_LOW_WATERMARK:
        return

    with _video_preprocess_lock:
        if _video_preprocess_state["inflight"]:
            return
        _video_preprocess_state["inflight"] = True

    _video_preprocess_executor.submit(
        _run_video_preprocess_background, AUTOLABEL_VIDEO_BATCH_SIZE
    )


@app.route("/api/preprocess/status")
def api_preprocess_status():
    """Expose whether a background preprocess run is in flight (useful for UI badges)."""
    if request.args.get("start", "0") == "1":
        _ensure_ready_maintainer_started()
    with _video_preprocess_lock:
        video_state = dict(_video_preprocess_state)
    with _reolink_preprocess_lock:
        reolink_inflight = sorted(_reolink_preprocess_inflight)
    maintainer_state = _ready_maintainer_state_snapshot()
    return jsonify(
        {
        "video": {
            "inflight": bool(video_state["inflight"]),
            "last_run_at": video_state["last_run_at"],
            "last_run_videos": int(video_state["last_run_videos"] or 0),
            "last_run_triplets": int(video_state.get("last_run_triplets") or 0),
            "last_error": video_state["last_error"],
            "low_watermark": AUTOLABEL_VIDEO_LOW_WATERMARK,
            "ready_target": AUTOLABEL_VIDEO_LOW_WATERMARK,
            "batch_size": AUTOLABEL_VIDEO_BATCH_SIZE,
            "auto_preprocess": AUTOLABEL_VIDEO_AUTO_PREPROCESS,
        },
            "reolink": {
                "prewarm_target": REOLINK_PREWARM_TARGET,
                "ready_target": REOLINK_PREWARM_TARGET,
                "sites": [site.site_key for site in REOLINK_SITES],
                "inflight": bool(reolink_inflight),
                "inflight_queues": reolink_inflight,
                "true_ten_batch_size": REOLINK_TRUE_TEN_BATCH_SIZE,
                "yolo_batch_frames": REOLINK_YOLO_BATCH_FRAMES,
                "preprocess_max_seconds": REOLINK_PREPROCESS_MAX_SECONDS,
                "supabase_crops": _supabase_crop_status_snapshot(),
            },
            "ready_target": REOLINK_PREWARM_TARGET,
            "label_ready_target_configured": LABEL_READY_TARGET_CONFIGURED,
            "throughput_target_images": LABEL_THROUGHPUT_TARGET_IMAGES,
            "throughput_target_folders": LABEL_THROUGHPUT_TARGET_FOLDERS,
            "maintainer": {
                "inflight": bool(maintainer_state["inflight"]),
                "started": bool(maintainer_state["started"]),
                "current_queue": maintainer_state["current_queue"],
                "last_run_at": maintainer_state["last_run_at"],
                "generated": int(maintainer_state.get("generated") or 0),
                "cache_warming": bool(maintainer_state["cache_warming"]),
                "last_error": maintainer_state["last_error"],
            },
        }
    )


def run_label_ui(port: int = 8080) -> None:
    host = os.environ.get("LABEL_UI_HOST", "127.0.0.1").strip() or "127.0.0.1"
    display_host = "localhost" if host in {"127.0.0.1", "0.0.0.0"} else host
    print(f"Starting label UI at http://{display_host}:{port}")
    _cleanup_cache_if_needed(force=True)
    _ensure_ready_maintainer_started()
    print(f"Preview cache: {CACHE_DIR}")
    print(
        "Timing logs: "
        f"{'on' if TIMING_LOGS_ENABLED else 'off'}"
        f" (min {TIMING_LOG_MIN_MS:.0f} ms)"
    )
    # Request-scoped Drive clients and locked shared caches make threaded
    # serving practical here, which helps keep the warm queue filled.
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
