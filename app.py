#!/usr/bin/env python3
"""
Video Table Cropper - Flask Web App

Drag-and-drop interface to crop videos based on JSON bounding boxes.
Supports both axis-aligned and rotated bounding boxes.
"""

import io
import json
import math
import os
import re
import shutil
import subprocess
import time
import uuid
import zipfile
from pathlib import Path

from flask import Response, Flask, abort, g, jsonify, redirect, render_template, request, send_file, stream_with_context, url_for

from camera_table_metadata import (
    detect_camera_from_filename as detect_camera_from_metadata,
    get_legacy_camera_config as get_camera_config_from_metadata,
    iter_camera_configs,
)
from db import database_enabled, db_healthcheck
from drive_client import DriveClient, DriveClientError
from env_loader import load_local_env
from dataset_schema import HUMAN_LABELS, PREVIEW_FILE_BY_KIND
from drive_roots import resolve_video_pipeline_roots
from drive_queue_store import DriveQueueStore
from sample_builder import load_sample_payload
from segment_cropper import SegmentCropError, crop_segment_from_path
from segment_parser import SegmentParserError, parse_segment_json
from video_review_service import (
    archive_sample,
    describe_sample_exports,
    ensure_cached_sample,
    ensure_review_roots,
    export_sample,
    index_review_samples,
    recycle_sample,
    undo_export_manifests,
)
from export_worker import DriveExportWorker
from video_review_store import VideoReviewStore
from video_review_store_pg import VideoReviewStorePG
from worker_state_store_pg import WorkerStateStorePG
from worker_runtime import load_worker_runtime_state, resolve_processor_cache_dir, worker_runtime_status_path

app = Flask(__name__)
load_local_env()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

# Configuration
BASE_DIR = Path(__file__).parent
APP_ENV = str(os.environ.get("APP_ENV", "development")).strip().lower()
LEGACY_ROUTES_ENABLED = _env_bool("ENABLE_LEGACY_ROUTES", default=APP_ENV != "production")
DEFAULT_DRIVE_CACHE_DIR = os.environ.get(
    "DRIVE_CACHE_DIR",
    "/tmp/drive_cache" if APP_ENV == "production" else str(BASE_DIR / "drive_cache"),
)
VIDEO_REVIEW_BATCH_LIMIT_DEFAULT = int(
    os.environ.get(
        "VIDEO_REVIEW_BATCH_LIMIT_DEFAULT",
        "60" if APP_ENV == "production" else os.environ.get("DRIVE_BATCH_LIMIT_DEFAULT", "200"),
    )
)
X_ROBOTS_TAG = os.environ.get("X_ROBOTS_TAG", "noindex, nofollow" if APP_ENV == "production" else "").strip()
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
OUTPUT_FOLDER = Path(__file__).parent / "output"
FRAMES_FOLDER = Path(__file__).parent / "frames"
DRIVE_CACHE_FOLDER = Path(DEFAULT_DRIVE_CACHE_DIR)
DRIVE_IMAGE_CACHE_FOLDER = DRIVE_CACHE_FOLDER / "images"
DRIVE_PREVIEW_CACHE_FOLDER = DRIVE_CACHE_FOLDER / "previews"
DRIVE_VIDEO_REVIEW_CACHE_FOLDER = DRIVE_CACHE_FOLDER / "video_review"
PROCESSOR_CACHE_DIR = resolve_processor_cache_dir(base_dir=BASE_DIR)
WORKER_RUNTIME_STATUS_PATH = worker_runtime_status_path(PROCESSOR_CACHE_DIR)
DRIVE_QUEUE_DB = BASE_DIR / "drive_queue.db"
DRIVE_VIDEO_REVIEW_DB = BASE_DIR / "video_review_queue.db"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_JSON_EXTENSIONS = {".json"}
MODE_LABELS = {
    "dirty_clean": {"dirty", "clean"},
    "dirty_clean_occupied": {"dirty", "clean", "occupied"},
}
DRIVE_BATCH_LIMIT_DEFAULT = int(os.environ.get("DRIVE_BATCH_LIMIT_DEFAULT", "200"))
DRIVE_BATCH_LIMIT_MAX = int(os.environ.get("DRIVE_BATCH_LIMIT_MAX", "500"))
DRIVE_PREVIEW_CACHE_TTL_SECONDS = int(os.environ.get("DRIVE_PREVIEW_CACHE_TTL_SECONDS", "86400"))
DRIVE_OUTPUT_BINARY_ROOT_ID = os.environ.get("DRIVE_OUTPUT_ROOT_BINARY_FOLDER_ID")
DRIVE_OUTPUT_MULTICLASS_ROOT_ID = os.environ.get("DRIVE_OUTPUT_ROOT_MULTICLASS_FOLDER_ID")
DRIVE_SOURCE_ROOT_FOLDER_ID = os.environ.get("DRIVE_SOURCE_ROOT_FOLDER_ID")
LEGACY_DRIVE_EXPORT_ROOT_ID = os.environ.get("DRIVE_OUTPUT_ROOT_MULTICLASS_FOLDER_ID")
DRIVE_PROJECT_ROOT_FOLDER_ID = os.environ.get("DRIVE_PROJECT_ROOT_FOLDER_ID")
DRIVE_REVIEW_QUEUE_ROOT_ID = os.environ.get("DRIVE_REVIEW_QUEUE_ROOT_ID")
DRIVE_OUTPUT_TEMPORAL_STATE_ROOT_ID = os.environ.get("DRIVE_OUTPUT_TEMPORAL_STATE_ROOT_ID")
DRIVE_OUTPUT_DIRTY_CLEAN_SURFACE_ROOT_ID = os.environ.get("DRIVE_OUTPUT_DIRTY_CLEAN_SURFACE_ROOT_ID")
DRIVE_OUTPUT_OCCUPANCY_MLP_ROOT_ID = os.environ.get("DRIVE_OUTPUT_OCCUPANCY_MLP_ROOT_ID")
DRIVE_OUTPUT_SAM_AUDIT_ROOT_ID = os.environ.get("DRIVE_OUTPUT_SAM_AUDIT_ROOT_ID")

# Ensure folders exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)
FRAMES_FOLDER.mkdir(exist_ok=True)
DRIVE_IMAGE_CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
DRIVE_PREVIEW_CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
DRIVE_VIDEO_REVIEW_CACHE_FOLDER.mkdir(parents=True, exist_ok=True)

queue_store = DriveQueueStore(DRIVE_QUEUE_DB)
video_review_store = VideoReviewStorePG() if database_enabled() else VideoReviewStore(DRIVE_VIDEO_REVIEW_DB)
worker_state_store = WorkerStateStorePG() if database_enabled() else None


def _make_export_drive_client() -> DriveClient:
    """Create a fresh DriveClient for the background export worker (not request-scoped)."""
    return DriveClient()


_export_worker = DriveExportWorker(
    store=video_review_store,
    cache_root=DRIVE_VIDEO_REVIEW_CACHE_FOLDER,
    get_client_fn=_make_export_drive_client,
    get_output_roots_fn=lambda: get_video_review_output_roots(),
)
_export_worker.start()

def detect_camera_from_filename(filename: str) -> str:
    """Extract camera ID from filename (e.g., 'IPC3' from '3_1_Mimosas_IPC3_...')."""
    return detect_camera_from_metadata(filename)


def get_camera_config(camera_id: str) -> dict:
    """Get stored JSON config for a camera."""
    return get_camera_config_from_metadata(camera_id)


def allowed_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def allowed_json(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_JSON_EXTENSIONS


def get_drive_client() -> DriveClient:
    """Return a request-scoped Drive client to avoid cross-request transport reuse."""
    client = getattr(g, "drive_client", None)
    if client is None:
        client = DriveClient()
        g.drive_client = client
    return client


def get_output_root_for_mode(mode: str) -> str | None:
    if mode == "dirty_clean":
        return DRIVE_OUTPUT_BINARY_ROOT_ID
    if mode == "dirty_clean_occupied":
        return DRIVE_OUTPUT_MULTICLASS_ROOT_ID
    return None


def normalize_batch_limit(raw_value: int | str | None, default: int | None = None) -> int:
    fallback = DRIVE_BATCH_LIMIT_DEFAULT if default is None else default
    try:
        if raw_value is None:
            return fallback
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return fallback
    return max(1, min(parsed, DRIVE_BATCH_LIMIT_MAX))


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def require_legacy_routes() -> None:
    if not LEGACY_ROUTES_ENABLED:
        abort(404)


@app.after_request
def apply_response_headers(response):
    if X_ROBOTS_TAG:
        response.headers["X-Robots-Tag"] = X_ROBOTS_TAG
    return response


def get_video_pipeline_roots(client: DriveClient | None = None) -> dict[str, str | None]:
    client = client or get_drive_client()
    return resolve_video_pipeline_roots(
        client,
        project_root_id=DRIVE_PROJECT_ROOT_FOLDER_ID,
        source_root_id=os.environ.get("DRIVE_VIDEO_SOURCE_ROOT_ID"),
        review_root_id=DRIVE_REVIEW_QUEUE_ROOT_ID,
        temporal_root_id=DRIVE_OUTPUT_TEMPORAL_STATE_ROOT_ID,
        surface_root_id=DRIVE_OUTPUT_DIRTY_CLEAN_SURFACE_ROOT_ID,
        occupancy_root_id=DRIVE_OUTPUT_OCCUPANCY_MLP_ROOT_ID,
        audit_root_id=DRIVE_OUTPUT_SAM_AUDIT_ROOT_ID,
    )


def get_default_video_review_root() -> str:
    if DRIVE_REVIEW_QUEUE_ROOT_ID:
        return DRIVE_REVIEW_QUEUE_ROOT_ID
    if not DRIVE_PROJECT_ROOT_FOLDER_ID:
        return ""
    try:
        roots = get_video_pipeline_roots()
    except DriveClientError:
        return DRIVE_PROJECT_ROOT_FOLDER_ID
    return str(roots.get("review") or DRIVE_PROJECT_ROOT_FOLDER_ID)


def cleanup_preview_cache(session_id: str) -> None:
    """Best-effort cleanup of stale preview files for a session."""
    import time

    session_preview_folder = DRIVE_PREVIEW_CACHE_FOLDER / session_id
    if not session_preview_folder.exists():
        return

    now = time.time()
    ttl_seconds = DRIVE_PREVIEW_CACHE_TTL_SECONDS
    for preview in session_preview_folder.glob("*.jpg"):
        try:
            if now - preview.stat().st_mtime > ttl_seconds:
                preview.unlink()
        except FileNotFoundError:
            continue


def ensure_drive_item_preview(item: dict) -> Path:
    """
    Ensure cached cropped preview exists for queue item and return its path.

    Caches source image downloads and per-item crop previews locally.
    """
    session_id = item["session_id"]
    item_id = item["id"]
    source_file_id = item["source_file_id"]

    source_cache_path = DRIVE_IMAGE_CACHE_FOLDER / f"{source_file_id}.bin"
    preview_path = DRIVE_PREVIEW_CACHE_FOLDER / session_id / f"{item_id}.jpg"

    if preview_path.exists():
        return preview_path

    client = get_drive_client()
    if not source_cache_path.exists():
        client.download_file_to_path(source_file_id, source_cache_path)

    segment = item.get("segment")
    if not segment:
        segment = json.loads(item["segment_json"])

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    crop_segment_from_path(source_cache_path, segment, preview_path)
    return preview_path


def ensure_source_image_processed_if_complete(session_id: str, item: dict) -> None:
    """Move source image from active folder to _processed when all queue items are resolved."""
    source_file_id = item["source_file_id"]
    if not queue_store.is_source_file_complete(session_id, source_file_id):
        return
    if item.get("source_file_in_processed"):
        return

    client = get_drive_client()
    try:
        processed_root_id = client.ensure_subfolder(item["source_root_folder_id"], "_processed")
        processed_group_id = client.ensure_subfolder(
            processed_root_id,
            folder_bucket_name(item["source_folder_name"], item["source_folder_id"]),
        )
        client.move_file(
            source_file_id,
            new_parent_id=processed_group_id,
            remove_parent_id=item["source_folder_id"],
        )
        queue_store.mark_source_file_processed(session_id, source_file_id, True)
    except DriveClientError:
        # Keep label/skip action successful even if source archival move fails.
        return


def maybe_restore_source_image_to_active(session_id: str, item: dict) -> None:
    """Restore source image from _processed to active source folder when undo re-opens pending work."""
    if not item.get("source_file_in_processed"):
        return

    client = get_drive_client()
    try:
        client.move_file(
            item["source_file_id"],
            new_parent_id=item["source_folder_id"],
        )
        queue_store.mark_source_file_processed(session_id, item["source_file_id"], False)
    except DriveClientError:
        return


def crop_video(input_path: Path, output_path: Path, x1: int, y1: int, x2: int, y2: int) -> bool:
    """Crop a video using ffmpeg (axis-aligned bbox)."""
    width = x2 - x1
    height = y2 - y1
    crop_filter = f"crop={width}:{height}:{x1}:{y1}"

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vf", crop_filter,
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "aac",
        "-y",
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def crop_rotated_video(input_path: Path, output_path: Path, rotated_bbox: dict, frame_width: int, frame_height: int) -> bool:
    """
    Crop a rotated rectangular region from video.

    Strategy:
    1. Crop the axis-aligned bounding box containing all corners
    2. Pad to center the rotation point
    3. Rotate to straighten the region
    4. Crop the final rectangle from center
    """
    center = rotated_bbox.get("center", [0, 0])
    size = rotated_bbox.get("size", [100, 100])
    angle = rotated_bbox.get("angle", 0)
    corners = rotated_bbox.get("corners", [])

    if not corners or len(corners) < 4:
        return False

    cx, cy = center
    final_w, final_h = int(round(size[0])), int(round(size[1]))

    # Make dimensions even for codec compatibility
    final_w = final_w + (final_w % 2)
    final_h = final_h + (final_h % 2)

    # Get axis-aligned bounding box of corners
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]

    margin = 10
    min_x = max(0, int(min(xs)) - margin)
    min_y = max(0, int(min(ys)) - margin)
    max_x = min(frame_width, int(max(xs)) + margin + 1)
    max_y = min(frame_height, int(max(ys)) + margin + 1)

    # First crop dimensions
    crop1_w = max_x - min_x
    crop1_h = max_y - min_y
    crop1_x = min_x
    crop1_y = min_y

    # Bbox center in cropped coordinates
    new_cx = cx - min_x
    new_cy = cy - min_y

    # Calculate padded size (large enough to center bbox and fit final crop after rotation)
    S = 2 * int(max(new_cx, new_cy, crop1_w - new_cx, crop1_h - new_cy)) + max(final_w, final_h) + 100
    S = S + (S % 2)  # Make even

    # Padding to center bbox at (S/2, S/2)
    pad_left = int(S / 2 - new_cx)
    pad_top = int(S / 2 - new_cy)
    pad_right = S - crop1_w - pad_left
    pad_bottom = S - crop1_h - pad_top

    # Ensure non-negative padding
    if pad_left < 0:
        pad_right -= pad_left
        pad_left = 0
    if pad_top < 0:
        pad_bottom -= pad_top
        pad_top = 0
    if pad_right < 0:
        pad_left -= pad_right
        pad_right = 0
    if pad_bottom < 0:
        pad_top -= pad_bottom
        pad_bottom = 0

    # Recalculate S after adjustments
    S = crop1_w + pad_left + pad_right
    S_h = crop1_h + pad_top + pad_bottom
    if S != S_h:
        # Make square by adding to the smaller dimension
        if S > S_h:
            diff = S - S_h
            pad_bottom += diff
        else:
            diff = S_h - S
            pad_right += diff
            S = S_h

    # Rotation angle (negative to un-rotate)
    angle_rad = -angle * math.pi / 180

    # Final crop position (centered)
    final_crop_x = int((S - final_w) / 2)
    final_crop_y = int((S - final_h) / 2)

    # Build ffmpeg filter chain
    filter_str = (
        f"crop={crop1_w}:{crop1_h}:{crop1_x}:{crop1_y},"
        f"pad={S}:{S}:{pad_left}:{pad_top}:black,"
        f"rotate={angle_rad}:ow={S}:oh={S}:c=black,"
        f"crop={final_w}:{final_h}:{final_crop_x}:{final_crop_y}"
    )

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vf", filter_str,
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "aac",
        "-y",
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_video_info(video_path: Path) -> dict:
    """Get video information using ffprobe."""
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration,r_frame_rate,width,height",
        "-of", "json",
        str(video_path)
    ]

    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {}

        data = json.load(io.StringIO(result.stdout))
        if not data.get("streams"):
            return {}

        stream = data["streams"][0]

        # Parse frame rate
        fps_str = stream.get("r_frame_rate", "30/1")
        fps_parts = fps_str.split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1])

        return {
            "duration": float(stream.get("duration", 0)),
            "fps": fps,
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "resolution": f"{stream.get('width')}x{stream.get('height')}"
        }

    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        return {}


def format_timestamp(seconds: float) -> str:
    """Convert seconds to timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}h{minutes:02d}m{secs:02d}s"
    else:
        return f"{minutes:02d}m{secs:02d}s"


def extract_frames_from_video(video_path: Path, output_dir: Path, interval: int = 30, quality: int = 2) -> bool:
    """Extract frames from video at regular intervals."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video info
    video_info = get_video_info(video_path)
    if not video_info:
        return False

    # Extract frames using ffmpeg
    temp_pattern = str(output_dir / "temp_%04d.jpg")
    fps_value = f"1/{interval}"

    extract_cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps_value}",
        "-q:v", str(quality),
        "-y",
        temp_pattern
    ]

    try:
        result = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return False

    except subprocess.TimeoutExpired:
        return False

    # Rename frames to include timestamps
    temp_frames = sorted(output_dir.glob("temp_*.jpg"))
    frames_info = []

    for idx, temp_frame in enumerate(temp_frames):
        timestamp_sec = idx * interval
        timestamp_str = format_timestamp(timestamp_sec)

        new_name = f"frame_{idx:04d}_{timestamp_str}.jpg"
        new_path = output_dir / new_name

        temp_frame.rename(new_path)

        frames_info.append({
            "frame_number": idx,
            "filename": new_name,
            "timestamp_seconds": timestamp_sec,
            "timestamp_formatted": format_timestamp(timestamp_sec),
            "file_size_bytes": new_path.stat().st_size
        })

    # Generate metadata
    from datetime import datetime
    metadata = {
        "video_info": {
            "source_video": str(video_path.name),
            "video_name": video_path.stem,
            "duration_seconds": video_info.get("duration", 0),
            "fps": video_info.get("fps", 0),
            "resolution": video_info.get("resolution", "unknown"),
            "extraction_date": datetime.now().isoformat()
        },
        "extraction_params": {
            "interval_seconds": interval,
            "format": "jpg",
            "quality": quality,
            "total_frames_extracted": len(frames_info)
        },
        "frames": frames_info
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return True


@app.route("/")
def drive_home():
    """Default landing page for Drive-first workflow."""
    if not LEGACY_ROUTES_ENABLED:
        return redirect(url_for("video_review_page"))
    return render_template(
        "drive_home.html",
        default_source_root=DRIVE_SOURCE_ROOT_FOLDER_ID or "",
        show_extended_nav=LEGACY_ROUTES_ENABLED,
    )


@app.route("/drive/binary")
def drive_binary_page():
    """Drive labeling page for dirty/clean model."""
    require_legacy_routes()
    return render_template(
        "drive_labeling.html",
        mode="dirty_clean",
        labels=["dirty", "clean"],
        mode_title="Dirty vs Clean",
        default_source_root=DRIVE_SOURCE_ROOT_FOLDER_ID or "",
        default_batch_limit=DRIVE_BATCH_LIMIT_DEFAULT,
        show_extended_nav=LEGACY_ROUTES_ENABLED,
    )


@app.route("/drive/multiclass")
def drive_multiclass_page():
    """Drive labeling page for dirty/clean/occupied model."""
    require_legacy_routes()
    return render_template(
        "drive_labeling.html",
        mode="dirty_clean_occupied",
        labels=["dirty", "clean", "occupied"],
        mode_title="Dirty vs Clean vs Occupied",
        default_source_root=DRIVE_SOURCE_ROOT_FOLDER_ID or "",
        default_batch_limit=DRIVE_BATCH_LIMIT_DEFAULT,
        show_extended_nav=LEGACY_ROUTES_ENABLED,
    )


@app.route("/video-review")
def video_review_page():
    default_review_root = get_default_video_review_root()
    worker_status_payload = current_worker_status_payload()
    return render_template(
        "video_review.html",
        default_review_root=default_review_root,
        default_batch_limit=VIDEO_REVIEW_BATCH_LIMIT_DEFAULT,
        initial_worker_status=worker_status_payload["status"],
        worker_status_path=worker_status_payload["status_path"],
        show_extended_nav=LEGACY_ROUTES_ENABLED,
    )


@app.route("/legacy")
def index():
    require_legacy_routes()
    return render_template("index.html")


@app.route("/legacy/frames")
@app.route("/frames")
def frames_page():
    require_legacy_routes()
    return render_template("frames.html")


@app.route("/healthz", methods=["GET"])
def healthz():
    db_status = db_healthcheck()
    drive_configured = bool(
        os.environ.get("DRIVE_SERVICE_ACCOUNT_JSON")
        or os.environ.get("DRIVE_SERVICE_ACCOUNT_JSON_B64")
        or os.environ.get("DRIVE_SERVICE_ACCOUNT_JSON_PATH")
    )
    review_root_configured = bool(DRIVE_REVIEW_QUEUE_ROOT_ID or DRIVE_PROJECT_ROOT_FOLDER_ID)
    healthy = db_status["healthy"] and drive_configured and review_root_configured
    status_code = 200 if healthy else 503
    return (
        jsonify(
            {
                "success": healthy,
                "app_env": APP_ENV,
                "database": db_status,
                "drive_configured": drive_configured,
                "review_root_configured": review_root_configured,
                "legacy_routes_enabled": LEGACY_ROUTES_ENABLED,
            }
        ),
        status_code,
    )


def parse_source_folder_ids(raw_value) -> list[str]:
    """Normalize optional source folder ID input into a list."""
    if not raw_value:
        return []
    if isinstance(raw_value, list):
        values = raw_value
    elif isinstance(raw_value, str):
        values = raw_value.replace(",", "\n").splitlines()
    else:
        return []
    normalized = [str(v).strip() for v in values if str(v).strip()]
    # Preserve order while deduplicating.
    return list(dict.fromkeys(normalized))


def slugify_folder_name(value: str) -> str:
    """Convert Drive folder names into safe, consistent bucket names."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return slug or "folder"


def folder_bucket_name(folder_name: str, folder_id: str) -> str:
    """Stable normalized folder key used in output/recycle/processed trees."""
    return f"{slugify_folder_name(folder_name)}__{folder_id[:8]}"


def ensure_drive_mode_destination_folder(
    client: DriveClient,
    mode: str,
    label: str,
    source_folder_id: str,
    source_folder_name: str,
    segment_id: str,
) -> str:
    """Create destination path in Drive output tree and return folder ID."""
    output_root_id = get_output_root_for_mode(mode)
    if not output_root_id:
        raise DriveClientError(
            f"Missing output root for mode '{mode}'. "
            "Set DRIVE_OUTPUT_ROOT_BINARY_FOLDER_ID / DRIVE_OUTPUT_ROOT_MULTICLASS_FOLDER_ID."
        )

    label_folder_id = client.ensure_subfolder(output_root_id, label)
    source_folder_bucket_id = client.ensure_subfolder(
        label_folder_id,
        folder_bucket_name(source_folder_name, source_folder_id),
    )
    segment_bucket_id = client.ensure_subfolder(source_folder_bucket_id, f"segment_{segment_id}")
    return segment_bucket_id


def ensure_recycle_folder(
    client: DriveClient,
    source_root_folder_id: str,
    session_id: str,
    source_folder_id: str,
    source_folder_name: str,
) -> str:
    recycle_root_id = client.ensure_subfolder(source_root_folder_id, "_recycle")
    session_recycle_id = client.ensure_subfolder(recycle_root_id, session_id)
    return client.ensure_subfolder(
        session_recycle_id,
        folder_bucket_name(source_folder_name, source_folder_id),
    )


def get_video_review_output_roots() -> dict[str, str | None]:
    roots = get_video_pipeline_roots()
    return {
        "temporal": roots.get("temporal"),
        "surface": roots.get("surface"),
        "occupancy": roots.get("occupancy"),
        "audit": roots.get("audit"),
    }


def ensure_video_review_cached_sample(item: dict) -> Path:
    client = get_drive_client()
    return ensure_cached_sample(client, item, DRIVE_VIDEO_REVIEW_CACHE_FOLDER)


def video_review_preview_path(item: dict, kind: str) -> Path:
    if kind not in PREVIEW_FILE_BY_KIND:
        raise ValueError(f"Unsupported preview kind '{kind}'")
    sample_dir = ensure_video_review_cached_sample(item)
    preview_path = sample_dir / PREVIEW_FILE_BY_KIND[kind]
    if not preview_path.exists():
        raise FileNotFoundError(preview_path)
    return preview_path


def current_worker_status_payload() -> dict:
    if worker_state_store is not None:
        try:
            return {
                "status": worker_state_store.get_status(),
                "status_path": "postgres://worker_status/video-dataset-worker",
            }
        except Exception as exc:  # pragma: no cover - depends on live database state
            app.logger.warning("Falling back to empty worker status payload: %s", exc)
            fallback = load_worker_runtime_state(WORKER_RUNTIME_STATUS_PATH)
            fallback["last_error"] = str(exc)
            fallback["message"] = "Worker status is unavailable until database migrations are applied."
            return {
                "status": fallback,
                "status_path": "postgres://worker_status/video-dataset-worker",
            }
    return {
        "status": load_worker_runtime_state(WORKER_RUNTIME_STATUS_PATH),
        "status_path": str(WORKER_RUNTIME_STATUS_PATH),
    }


@app.route("/api/video-review/worker-status", methods=["GET"])
def video_review_worker_status():
    return jsonify({"success": True, **current_worker_status_payload()})


@app.route("/api/video-review/worker-status/stream", methods=["GET"])
def video_review_worker_status_stream():
    def generate():
        last_event_seq = None
        keepalive_ticks = 0
        while True:
            payload = current_worker_status_payload()
            event_seq = payload["status"].get("event_seq")
            if event_seq != last_event_seq:
                yield f"data: {json.dumps(payload)}\n\n"
                last_event_seq = event_seq
                keepalive_ticks = 0
            else:
                keepalive_ticks += 1
                if keepalive_ticks >= 5:
                    yield ": keepalive\n\n"
                    keepalive_ticks = 0
            time.sleep(2)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/drive/session/start", methods=["POST"])
def drive_start_session():
    """Initialize a Drive labeling session by indexing source folders into queue items."""
    payload = request.get_json(silent=True) or {}
    mode = payload.get("mode")
    if mode not in MODE_LABELS:
        return jsonify({"error": "mode must be one of: dirty_clean, dirty_clean_occupied"}), 400

    source_parent_folder_id = payload.get("source_parent_folder_id") or DRIVE_SOURCE_ROOT_FOLDER_ID
    source_folder_ids = parse_source_folder_ids(payload.get("source_folder_ids"))
    batch_limit = normalize_batch_limit(payload.get("batch_limit"))

    if not source_parent_folder_id and not source_folder_ids:
        return jsonify(
            {
                "error": (
                    "Provide source_parent_folder_id and/or source_folder_ids. "
                    "You can also set DRIVE_SOURCE_ROOT_FOLDER_ID."
                )
            }
        ), 400

    try:
        client = get_drive_client()
    except DriveClientError as exc:
        return jsonify({"error": str(exc)}), 500

    source_folders: dict[str, dict] = {}
    indexing_errors: list[str] = []

    if source_parent_folder_id:
        try:
            parent_folders = client.list_folders_recursive(source_parent_folder_id, include_root=False)
            for folder in parent_folders:
                folder["source_root_folder_id"] = source_parent_folder_id
                source_folders[folder["id"]] = folder
        except DriveClientError as exc:
            return jsonify({"error": str(exc)}), 500

    for folder_id in source_folder_ids:
        try:
            folder_meta = client.get_file(folder_id)
        except DriveClientError as exc:
            indexing_errors.append(f"Failed to read folder {folder_id}: {exc}")
            continue

        if folder_meta.get("mimeType") != "application/vnd.google-apps.folder":
            indexing_errors.append(f"{folder_id} is not a Drive folder")
            continue

        parents = folder_meta.get("parents", [])
        if source_parent_folder_id and source_parent_folder_id in parents:
            folder_meta["source_root_folder_id"] = source_parent_folder_id
        else:
            folder_meta["source_root_folder_id"] = parents[0] if parents else (source_parent_folder_id or folder_id)
        source_folders[folder_meta["id"]] = folder_meta

    if not source_folders:
        return jsonify({"error": "No source folders found", "details": indexing_errors}), 400

    session_id = str(uuid.uuid4())[:12]
    queue_store.create_session(
        session_id=session_id,
        mode=mode,
        source_parent_folder_id=source_parent_folder_id,
        source_folder_ids=list(source_folders.keys()),
        batch_limit=batch_limit,
    )

    queue_items: list[dict] = []
    folder_errors: list[str] = []
    folder_stats: list[dict] = []

    for folder in source_folders.values():
        folder_id = folder["id"]
        folder_name = folder.get("name", folder_id)
        source_root_folder_id = folder.get("source_root_folder_id") or folder_id

        segment_file = client.find_file_by_name(folder_id, "segment.json")
        segments = None
        segment_source = "segment.json"
        camera_id = detect_camera_from_filename(folder_name)

        if segment_file:
            try:
                segment_bytes = client.download_file_content(segment_file["id"])
                segments = parse_segment_json(segment_bytes)
            except (DriveClientError, SegmentParserError) as exc:
                folder_errors.append(f"{folder_name}: segment.json parse failed ({exc})")

        # Fallback: infer segmentation from camera ID in folder name (e.g. IPC6)
        if segments is None and camera_id:
            camera_config = get_camera_config(camera_id)
            if camera_config:
                try:
                    segments = parse_segment_json({"tables": camera_config.get("tables", [])})
                    segment_source = f"camera_config:{camera_id}"
                except SegmentParserError as exc:
                    folder_errors.append(f"{folder_name}: camera config parse failed ({exc})")

        if segments is None:
            missing_msg = (
                "missing segment.json and no matching camera config in folder name"
                if not segment_file
                else "unable to parse segment config from JSON or camera fallback"
            )
            folder_errors.append(f"{folder_name}: {missing_msg}")
            continue

        try:
            image_files = client.list_image_files(folder_id)
        except DriveClientError as exc:
            folder_errors.append(f"{folder_name}: failed to list images ({exc})")
            continue

        for image_file in image_files:
            for segment in segments:
                queue_items.append(
                    {
                        "session_id": session_id,
                        "source_root_folder_id": source_root_folder_id,
                        "source_folder_id": folder_id,
                        "source_folder_name": folder_name,
                        "source_file_id": image_file["id"],
                        "source_file_name": image_file.get("name", image_file["id"]),
                        "source_file_mime_type": image_file.get("mimeType"),
                        "segment_id": segment["segment_id"],
                        "segment": segment,
                    }
                )

        folder_stats.append(
            {
                "folder_id": folder_id,
                "folder_name": folder_name,
                "images_found": len(image_files),
                "segments_found": len(segments),
                "queue_items_generated": len(image_files) * len(segments),
                "camera_id": camera_id,
                "segment_source": segment_source,
            }
        )

    inserted_count = queue_store.add_queue_items(queue_items)
    stats = queue_store.get_stats(session_id)

    return jsonify(
        {
            "success": True,
            "session_id": session_id,
            "mode": mode,
            "source_folder_count": len(source_folders),
            "queue_items_created": inserted_count,
            "batch_limit": batch_limit,
            "folder_stats": folder_stats,
            "errors": indexing_errors + folder_errors,
            "stats": stats,
        }
    )


@app.route("/api/drive/session/<session_id>/batch", methods=["GET"])
def drive_get_batch(session_id: str):
    """Return next page of pending queue items and ensure preview images are available."""
    session = queue_store.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    limit = normalize_batch_limit(request.args.get("limit") or session.get("batch_limit"))
    try:
        cursor = int(request.args.get("cursor", 0))
    except ValueError:
        cursor = 0

    cleanup_preview_cache(session_id)
    items = queue_store.get_pending_batch(session_id, limit, cursor)

    batch_items: list[dict] = []
    preview_errors: list[str] = []
    for item in items:
        try:
            ensure_drive_item_preview(item)
        except (DriveClientError, SegmentCropError, OSError, ValueError) as exc:
            preview_errors.append(f"Item {item['id']}: {exc}")
            continue

        batch_items.append(
            {
                "id": item["id"],
                "source_file_id": item["source_file_id"],
                "source_file_name": item["source_file_name"],
                "source_folder_id": item["source_folder_id"],
                "source_folder_name": item["source_folder_name"],
                "segment_id": item["segment_id"],
                "image_url": f"/api/drive/session/{session_id}/image/{item['id']}",
            }
        )

    next_cursor = batch_items[-1]["id"] if batch_items else cursor
    has_more = queue_store.has_pending_after(session_id, next_cursor)

    return jsonify(
        {
            "success": True,
            "session_id": session_id,
            "mode": session["mode"],
            "items": batch_items,
            "count": len(batch_items),
            "limit": limit,
            "cursor": cursor,
            "next_cursor": next_cursor,
            "has_more": has_more,
            "errors": preview_errors,
            "stats": queue_store.get_stats(session_id),
        }
    )


@app.route("/api/drive/session/<session_id>/image/<int:item_id>", methods=["GET"])
def drive_get_item_image(session_id: str, item_id: int):
    """Serve cropped segment preview for queue item."""
    item = queue_store.get_item(session_id, item_id)
    if not item:
        return jsonify({"error": "Queue item not found"}), 404

    try:
        preview_path = ensure_drive_item_preview(item)
    except (DriveClientError, SegmentCropError, OSError, ValueError) as exc:
        return jsonify({"error": f"Could not generate preview: {exc}"}), 500

    return send_file(preview_path, mimetype="image/jpeg")


@app.route("/api/drive/session/<session_id>/label", methods=["POST"])
def drive_label_items(session_id: str):
    """Apply a label to selected pending items and move outputs into Drive label folders."""
    session = queue_store.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    payload = request.get_json(silent=True) or {}
    label = str(payload.get("label", "")).strip().lower()
    allowed_labels = MODE_LABELS[session["mode"]]
    if label not in allowed_labels:
        return jsonify({"error": f"Invalid label '{label}' for mode {session['mode']}"}), 400

    try:
        item_ids = [int(item_id) for item_id in payload.get("item_ids", [])]
    except (TypeError, ValueError):
        return jsonify({"error": "item_ids must be an array of integers"}), 400

    if not item_ids:
        return jsonify({"error": "No item_ids provided"}), 400

    try:
        client = get_drive_client()
    except DriveClientError as exc:
        return jsonify({"error": str(exc)}), 500

    items = queue_store.get_items(session_id, item_ids, status="pending")
    if not items:
        return jsonify({"error": "No pending items found for supplied item_ids"}), 400

    labeled_ids: list[int] = []
    action_errors: list[str] = []
    for item in items:
        try:
            preview_path = ensure_drive_item_preview(item)
            destination_folder_id = ensure_drive_mode_destination_folder(
                client=client,
                mode=session["mode"],
                label=label,
                source_folder_id=item["source_folder_id"],
                source_folder_name=item["source_folder_name"],
                segment_id=item["segment_id"],
            )
            destination_name = (
                f"{Path(item['source_file_name']).stem}_segment_{item['segment_id']}.jpg"
            )
            uploaded = client.upload_file(
                preview_path,
                destination_folder_id,
                file_name=destination_name,
                mime_type="image/jpeg",
            )
            queue_store.update_item_after_label(
                session_id=session_id,
                item_id=item["id"],
                label=label,
                output_file_id=uploaded["id"],
            )
            queue_store.log_action(
                session_id=session_id,
                queue_item_id=item["id"],
                action_type="label",
                prev_status=item["status"],
                new_status="labeled",
                prev_label=item.get("label"),
                new_label=label,
                moved_file_id=uploaded["id"],
            )
            labeled_ids.append(item["id"])
            ensure_source_image_processed_if_complete(session_id, item)
        except (DriveClientError, SegmentCropError, OSError, ValueError) as exc:
            action_errors.append(f"Item {item['id']}: {exc}")

    return jsonify(
        {
            "success": True,
            "labeled_item_ids": labeled_ids,
            "requested_count": len(item_ids),
            "processed_count": len(labeled_ids),
            "errors": action_errors,
            "stats": queue_store.get_stats(session_id),
        }
    )


@app.route("/api/drive/session/<session_id>/skip", methods=["POST"])
def drive_skip_items(session_id: str):
    """Skip pending items by moving crops to recycle folder and marking them skipped."""
    session = queue_store.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    payload = request.get_json(silent=True) or {}
    try:
        item_ids = [int(item_id) for item_id in payload.get("item_ids", [])]
    except (TypeError, ValueError):
        return jsonify({"error": "item_ids must be an array of integers"}), 400

    if not item_ids:
        return jsonify({"error": "No item_ids provided"}), 400

    try:
        client = get_drive_client()
    except DriveClientError as exc:
        return jsonify({"error": str(exc)}), 500

    items = queue_store.get_items(session_id, item_ids, status="pending")
    if not items:
        return jsonify({"error": "No pending items found for supplied item_ids"}), 400

    skipped_ids: list[int] = []
    action_errors: list[str] = []
    for item in items:
        try:
            preview_path = ensure_drive_item_preview(item)
            recycle_folder_id = ensure_recycle_folder(
                client=client,
                source_root_folder_id=item["source_root_folder_id"],
                session_id=session_id,
                source_folder_id=item["source_folder_id"],
                source_folder_name=item["source_folder_name"],
            )
            recycle_name = (
                f"{Path(item['source_file_name']).stem}_segment_{item['segment_id']}_skipped.jpg"
            )
            recycled = client.upload_file(
                preview_path,
                recycle_folder_id,
                file_name=recycle_name,
                mime_type="image/jpeg",
            )
            queue_store.update_item_after_skip(
                session_id=session_id,
                item_id=item["id"],
                recycle_file_id=recycled["id"],
            )
            queue_store.log_action(
                session_id=session_id,
                queue_item_id=item["id"],
                action_type="skip",
                prev_status=item["status"],
                new_status="skipped",
                prev_label=item.get("label"),
                new_label=None,
                moved_file_id=recycled["id"],
            )
            skipped_ids.append(item["id"])
            ensure_source_image_processed_if_complete(session_id, item)
        except (DriveClientError, SegmentCropError, OSError, ValueError) as exc:
            action_errors.append(f"Item {item['id']}: {exc}")

    return jsonify(
        {
            "success": True,
            "skipped_item_ids": skipped_ids,
            "requested_count": len(item_ids),
            "processed_count": len(skipped_ids),
            "errors": action_errors,
            "stats": queue_store.get_stats(session_id),
        }
    )


@app.route("/api/drive/session/<session_id>/undo", methods=["POST"])
def drive_undo_last_action(session_id: str):
    """Undo the most recent non-undone label/skip action in a session."""
    session = queue_store.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    action = queue_store.get_last_action(session_id)
    if not action:
        return jsonify({"error": "No action available to undo"}), 400

    item = queue_store.get_item(session_id, int(action["queue_item_id"]))
    if not item:
        queue_store.mark_action_undone(int(action["id"]))
        return jsonify({"error": "Original queue item not found for undo"}), 400

    try:
        client = get_drive_client()
        if action.get("moved_file_id"):
            # Reverting an action removes the uploaded artifact for that action.
            try:
                client.delete_file(action["moved_file_id"])
            except DriveClientError:
                # File may already be removed manually; keep undo idempotent.
                pass

        queue_store.restore_item(
            session_id=session_id,
            item_id=item["id"],
            status=action.get("prev_status") or "pending",
            label=action.get("prev_label"),
        )
        queue_store.mark_action_undone(int(action["id"]))

        if (action.get("prev_status") or "pending") == "pending":
            maybe_restore_source_image_to_active(session_id, item)
    except DriveClientError as exc:
        return jsonify({"error": f"Undo failed: {exc}"}), 500

    return jsonify(
        {
            "success": True,
            "undone_action_id": action["id"],
            "item_id": item["id"],
            "stats": queue_store.get_stats(session_id),
        }
    )


@app.route("/api/drive/session/<session_id>/stats", methods=["GET"])
def drive_session_stats(session_id: str):
    session = queue_store.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    return jsonify(
        {
            "success": True,
            "session_id": session_id,
            "mode": session["mode"],
            "stats": queue_store.get_stats(session_id),
        }
    )


@app.route("/api/video-review/session/start", methods=["POST"])
def video_review_start_session():
    payload = request.get_json(silent=True) or {}
    batch_limit = normalize_batch_limit(payload.get("batch_limit"), default=VIDEO_REVIEW_BATCH_LIMIT_DEFAULT)

    try:
        client = get_drive_client()
    except DriveClientError as exc:
        return jsonify({"error": str(exc)}), 500

    supplied_folder_id = payload.get("review_root_folder_id") or DRIVE_REVIEW_QUEUE_ROOT_ID or DRIVE_PROJECT_ROOT_FOLDER_ID
    if not supplied_folder_id:
        return jsonify({"error": "review_root_folder_id or DRIVE_PROJECT_ROOT_FOLDER_ID is required"}), 400

    try:
        supplied_meta = client.get_file(supplied_folder_id)
        supplied_name = str(supplied_meta.get("name", "")).strip().lower()
    except DriveClientError as exc:
        return jsonify({"error": f"Could not read supplied folder: {exc}"}), 500

    try:
        if supplied_name == "review_queue":
            review_root_folder_id = supplied_folder_id
        else:
            review_root_folder_id = client.ensure_subfolder(supplied_folder_id, "review_queue")

        session_id = str(uuid.uuid4())[:12]
        roots = ensure_review_roots(client, review_root_folder_id)
        video_review_store.create_session(
            session_id=session_id,
            review_root_folder_id=review_root_folder_id,
            pending_root_folder_id=roots["pending"],
            batch_limit=batch_limit,
        )
        inserted = 0
        errors: list[str] = []
        if not database_enabled():
            inserted, errors, _ = index_review_samples(client, video_review_store, session_id, review_root_folder_id)
        else:
            inserted = int(video_review_store.get_stats(session_id)["status_counts"]["pending"])
        return jsonify(
            {
                "success": True,
                "session_id": session_id,
                "review_root_folder_id": review_root_folder_id,
                "pending_root_folder_id": roots["pending"],
                "queue_items_created": inserted,
                "batch_limit": batch_limit,
                "errors": errors,
                "stats": video_review_store.get_stats(session_id),
            }
        )
    except DriveClientError as exc:
        return jsonify({"error": f"Drive setup failed: {exc}"}), 500
    except Exception as exc:
        app.logger.exception("Video review session start failed")
        return jsonify({"error": f"Video review session start failed: {exc}"}), 500


@app.route("/api/video-review/session/<session_id>/batch", methods=["GET"])
def video_review_get_batch(session_id: str):
    session = video_review_store.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    index_errors: list[str] = []
    if not database_enabled():
        try:
            client = get_drive_client()
            _, index_errors, _ = index_review_samples(
                client,
                video_review_store,
                session_id,
                session["review_root_folder_id"],
            )
        except DriveClientError as exc:
            return jsonify({"error": f"Could not refresh review queue: {exc}"}), 500

    limit = normalize_batch_limit(
        request.args.get("limit"),
        default=int(session.get("batch_limit") or VIDEO_REVIEW_BATCH_LIMIT_DEFAULT),
    )
    try:
        cursor = int(request.args.get("cursor", 0))
    except ValueError:
        cursor = 0

    items = video_review_store.get_pending_batch(session_id, limit, cursor)
    batch_items = []
    errors: list[str] = []
    for item in items:
        try:
            ensure_video_review_cached_sample(item)
        except (DriveClientError, OSError, ValueError) as exc:
            errors.append(f"Item {item['id']}: {exc}")
            continue

        sample = item["sample"]
        batch_items.append(
            {
                "id": item["id"],
                "sample_id": sample["sample_id"],
                "camera_id": sample["source_video"]["camera_id"],
                "video_name": sample["source_video"]["video_name"],
                "table_track_id": sample["table"]["table_track_id"],
                "anchor_time_seconds": sample["timing"]["anchor_time_seconds"],
                "associated_people_count": len(sample.get("people", [])),
                "preview_urls": {
                    kind: f"/api/video-review/session/{session_id}/preview/{item['id']}/{kind}"
                    for kind in ("anchor", "t_minus_10", "t_minus_20")
                },
            }
        )

    next_cursor = batch_items[-1]["id"] if batch_items else cursor
    has_more = video_review_store.has_pending_after(session_id, next_cursor)
    return jsonify(
        {
            "success": True,
            "items": batch_items,
            "count": len(batch_items),
            "cursor": cursor,
            "next_cursor": next_cursor,
            "has_more": has_more,
            "errors": index_errors + errors,
            "stats": video_review_store.get_stats(session_id),
        }
    )


@app.route("/api/video-review/session/<session_id>/preview/<int:item_id>/<kind>", methods=["GET"])
def video_review_preview(session_id: str, item_id: int, kind: str):
    item = video_review_store.get_item(session_id, item_id)
    if not item:
        return jsonify({"error": "Queue item not found"}), 404

    try:
        preview_path = video_review_preview_path(item, kind)
    except (DriveClientError, OSError, ValueError) as exc:
        return jsonify({"error": f"Could not load preview: {exc}"}), 500

    return send_file(preview_path, mimetype="image/jpeg")


@app.route("/api/video-review/session/<session_id>/debug/<int:item_id>", methods=["GET"])
def video_review_debug_item(session_id: str, item_id: int):
    session = video_review_store.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    item = video_review_store.get_item(session_id, item_id)
    if not item:
        return jsonify({"error": "Queue item not found"}), 404

    try:
        sample_dir = ensure_video_review_cached_sample(item)
        sample_payload = load_sample_payload(sample_dir)
        debug_payload = describe_sample_exports(sample_dir, sample_payload, get_video_review_output_roots())
    except (DriveClientError, OSError, ValueError) as exc:
        return jsonify({"error": f"Could not build debug preview: {exc}"}), 500

    return jsonify(
        {
            "success": True,
            "item_id": item_id,
            "sample": debug_payload,
        }
    )


@app.route("/api/video-review/session/<session_id>/label", methods=["POST"])
def video_review_label_items(session_id: str):
    session = video_review_store.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    payload = request.get_json(silent=True) or {}
    label = str(payload.get("label", "")).strip().lower()
    if label not in HUMAN_LABELS:
        return jsonify({"error": f"Invalid label '{label}'"}), 400

    try:
        item_ids = [int(item_id) for item_id in payload.get("item_ids", [])]
    except (TypeError, ValueError):
        return jsonify({"error": "item_ids must be an array of integers"}), 400

    if not item_ids:
        return jsonify({"error": "No item_ids provided"}), 400

    items = video_review_store.get_items(session_id, item_ids, status="pending")
    if not items:
        return jsonify({"error": "No pending items found for supplied item_ids"}), 400

    processed_ids: list[int] = []
    action_errors: list[str] = []
    for item in items:
        try:
            video_review_store.label_item_optimistic(
                session_id=session_id,
                item_id=item["id"],
                label=label,
            )
            video_review_store.log_action(
                session_id=session_id,
                queue_item_id=item["id"],
                action_type="label",
                prev_status=item["status"],
                new_status="labeled",
                prev_label=item.get("label"),
                new_label=label,
                exported_folder_ids=[],
                moved_folder_id=item["sample_folder_id"],
                archive_parent_folder_id=None,
            )
            processed_ids.append(item["id"])
        except Exception as exc:
            action_errors.append(f"Item {item['id']}: {exc}")

    return jsonify(
        {
            "success": True,
            "labeled_item_ids": processed_ids,
            "requested_count": len(item_ids),
            "processed_count": len(processed_ids),
            "errors": action_errors,
            "stats": video_review_store.get_stats(session_id),
        }
    )


@app.route("/api/video-review/session/<session_id>/skip", methods=["POST"])
def video_review_skip_items(session_id: str):
    session = video_review_store.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    payload = request.get_json(silent=True) or {}
    try:
        item_ids = [int(item_id) for item_id in payload.get("item_ids", [])]
    except (TypeError, ValueError):
        return jsonify({"error": "item_ids must be an array of integers"}), 400

    if not item_ids:
        return jsonify({"error": "No item_ids provided"}), 400

    try:
        client = get_drive_client()
    except DriveClientError as exc:
        return jsonify({"error": str(exc)}), 500

    items = video_review_store.get_items(session_id, item_ids, status="pending")
    if not items:
        return jsonify({"error": "No pending items found for supplied item_ids"}), 400

    roots = ensure_review_roots(client, session["review_root_folder_id"])
    skipped_ids: list[int] = []
    action_errors: list[str] = []
    for item in items:
        try:
            sample_dir = ensure_video_review_cached_sample(item)
            sample_payload = load_sample_payload(sample_dir)
            recycle_parent_folder_id = recycle_sample(
                client,
                roots,
                sample_payload,
                item["sample_folder_id"],
                item["source_parent_folder_id"],
                session_id,
            )
            video_review_store.update_item_after_skip(
                session_id=session_id,
                item_id=item["id"],
                archived_parent_folder_id=recycle_parent_folder_id,
            )
            video_review_store.log_action(
                session_id=session_id,
                queue_item_id=item["id"],
                action_type="skip",
                prev_status=item["status"],
                new_status="skipped",
                prev_label=item.get("label"),
                new_label=None,
                exported_folder_ids=[],
                moved_folder_id=item["sample_folder_id"],
                archive_parent_folder_id=recycle_parent_folder_id,
            )
            skipped_ids.append(item["id"])
        except (DriveClientError, OSError, ValueError) as exc:
            action_errors.append(f"Item {item['id']}: {exc}")

    return jsonify(
        {
            "success": True,
            "skipped_item_ids": skipped_ids,
            "requested_count": len(item_ids),
            "processed_count": len(skipped_ids),
            "errors": action_errors,
            "stats": video_review_store.get_stats(session_id),
        }
    )


@app.route("/api/video-review/session/<session_id>/trash", methods=["POST"])
def video_review_trash_items(session_id: str):
    session = video_review_store.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    payload = request.get_json(silent=True) or {}
    try:
        item_ids = [int(item_id) for item_id in payload.get("item_ids", [])]
    except (TypeError, ValueError):
        return jsonify({"error": "item_ids must be an array of integers"}), 400

    if not item_ids:
        return jsonify({"error": "No item_ids provided"}), 400

    try:
        client = get_drive_client()
    except DriveClientError as exc:
        return jsonify({"error": str(exc)}), 500

    items = video_review_store.get_items(session_id, item_ids, status="pending")
    if not items:
        return jsonify({"error": "No pending items found for supplied item_ids"}), 400

    trashed_ids: list[int] = []
    action_errors: list[str] = []
    for item in items:
        try:
            client.trash_file(item["sample_folder_id"])
            video_review_store.update_item_after_trash(session_id=session_id, item_id=item["id"])
            video_review_store.log_action(
                session_id=session_id,
                queue_item_id=item["id"],
                action_type="trash",
                prev_status=item["status"],
                new_status="trashed",
                prev_label=item.get("label"),
                new_label=None,
                exported_folder_ids=[],
                moved_folder_id=item["sample_folder_id"],
                archive_parent_folder_id=None,
            )
            trashed_ids.append(item["id"])
        except DriveClientError as exc:
            action_errors.append(f"Item {item['id']}: {exc}")

    return jsonify(
        {
            "success": True,
            "trashed_item_ids": trashed_ids,
            "requested_count": len(item_ids),
            "processed_count": len(trashed_ids),
            "errors": action_errors,
            "stats": video_review_store.get_stats(session_id),
        }
    )


@app.route("/api/video-review/session/<session_id>/undo", methods=["POST"])
def video_review_undo_last_action(session_id: str):
    session = video_review_store.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    action = video_review_store.get_last_action(session_id)
    if not action:
        return jsonify({"error": "No action available to undo"}), 400

    item = video_review_store.get_item(session_id, int(action["queue_item_id"]))
    if not item:
        video_review_store.mark_action_undone(int(action["id"]))
        return jsonify({"error": "Original queue item not found for undo"}), 400

    try:
        client = get_drive_client()
        for folder_id in action.get("exported_folder_ids", []):
            try:
                client.delete_file(folder_id)
            except DriveClientError:
                continue

        if action.get("new_label"):
            sample_dir = ensure_video_review_cached_sample(item)
            sample_payload = load_sample_payload(sample_dir)
            undo_export_manifests(client, sample_payload, action["new_label"], get_video_review_output_roots())

        if action.get("action_type") == "trash" and action.get("moved_folder_id"):
            client.untrash_file(action["moved_folder_id"])
        elif action.get("moved_folder_id") and action.get("archive_parent_folder_id"):
            client.move_file(
                action["moved_folder_id"],
                new_parent_id=item["source_parent_folder_id"],
                remove_parent_id=action["archive_parent_folder_id"],
            )

        video_review_store.restore_item(
            session_id=session_id,
            item_id=item["id"],
            status=action.get("prev_status") or "pending",
            label=action.get("prev_label"),
        )
        video_review_store.mark_action_undone(int(action["id"]))
    except DriveClientError as exc:
        return jsonify({"error": f"Undo failed: {exc}"}), 500

    return jsonify(
        {
            "success": True,
            "undone_action_id": action["id"],
            "item_id": item["id"],
            "stats": video_review_store.get_stats(session_id),
        }
    )


@app.route("/api/video-review/session/<session_id>/stats", methods=["GET"])
def video_review_session_stats(session_id: str):
    session = video_review_store.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    return jsonify(
        {
            "success": True,
            "session_id": session_id,
            "stats": video_review_store.get_stats(session_id),
        }
    )


@app.route("/upload", methods=["POST"])
def upload():
    """Handle file uploads."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = file.filename
    file_type = request.form.get("type", "unknown")

    # Generate unique filename to avoid collisions
    unique_id = str(uuid.uuid4())[:8]
    ext = Path(filename).suffix
    safe_name = f"{unique_id}_{Path(filename).stem}{ext}"
    save_path = UPLOAD_FOLDER / safe_name

    if file_type == "video" and allowed_video(filename):
        file.save(save_path)
        return jsonify({"success": True, "filename": safe_name, "original": filename})
    elif file_type == "json" and allowed_json(filename):
        file.save(save_path)
        # Parse and return table info
        try:
            with open(save_path) as f:
                data = json.load(f)
            tables = data.get("tables", [])
            return jsonify({
                "success": True,
                "filename": safe_name,
                "original": filename,
                "tables": len(tables),
                "video_name": data.get("video_name", "unknown")
            })
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON file"}), 400
    else:
        return jsonify({"error": f"Invalid file type: {filename}"}), 400


@app.route("/cameras", methods=["GET"])
def get_cameras():
    """Return list of available cameras."""
    cameras = []
    for camera_id, config in iter_camera_configs():
        cameras.append({
            "id": camera_id,
            "name": f"Camera {camera_id.replace('IPC', '')}",
            "tables": len(config.get("tables", []))
        })
    return jsonify({"cameras": sorted(cameras, key=lambda x: x["id"])})


@app.route("/process", methods=["POST"])
def process():
    """Process multiple videos with their assigned cameras."""
    data = request.json
    videos = data.get("videos", [])  # Array of {filename, camera}

    if not videos:
        return jsonify({"error": "No videos provided"}), 400

    # Validate all videos exist and have camera assignments
    for video in videos:
        if not video.get("filename") or not video.get("camera"):
            return jsonify({"error": "Each video needs filename and camera"}), 400

        video_path = UPLOAD_FOLDER / video["filename"]
        if not video_path.exists():
            return jsonify({"error": f"Video not found: {video['filename']}"}), 404

        camera_config = get_camera_config(video["camera"])
        if not camera_config:
            return jsonify({"error": f"Unknown camera: {video['camera']}"}), 400

    # Create unique output folder for this batch job
    job_id = str(uuid.uuid4())[:8]
    job_output = OUTPUT_FOLDER / job_id
    job_output.mkdir(exist_ok=True)

    results = []

    # Process each video with its camera config
    for video in videos:
        video_filename = video["filename"]
        camera_id = video["camera"]
        video_path = UPLOAD_FOLDER / video_filename
        camera_config = get_camera_config(camera_id)

        tables = camera_config.get("tables", [])
        frame_width = camera_config.get("frame_width", 1280)
        frame_height = camera_config.get("frame_height", 720)

        # Process each table in this camera's config
        for idx, table in enumerate(tables):
            if not table.get("saved", True) or table.get("skip_reason"):
                continue

            table_id = table.get("id", idx)
            # Include camera ID in output filename
            output_name = f"{camera_id}_{video_path.stem}_table_{table_id:02d}.mp4"
            output_path = job_output / output_name

            # Crop using rotated_bbox or bbox
            rotated_bbox = table.get("rotated_bbox")
            bbox = table.get("bbox", {})

            success = False
            bbox_info = {}

            if rotated_bbox and rotated_bbox.get("corners"):
                success = crop_rotated_video(
                    video_path, output_path, rotated_bbox, frame_width, frame_height
                )
                bbox_info = {
                    "center": rotated_bbox.get("center"),
                    "size": rotated_bbox.get("size"),
                    "angle": rotated_bbox.get("angle")
                }
            elif bbox:
                x1, y1, x2, y2 = bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)
                if x2 > x1 and y2 > y1:
                    success = crop_video(video_path, output_path, x1, y1, x2, y2)
                bbox_info = bbox

            if success:
                results.append({
                    "table_id": table_id,
                    "camera": camera_id,  # NEW: Track source camera
                    "filename": output_name,
                    "download_url": f"/download/{job_id}/{output_name}",
                    "bbox": bbox_info,
                    "video_path": output_path
                })

    # Auto-extract frames from all cropped videos
    labeling_job_id = str(uuid.uuid4())[:8]
    labeling_folder = FRAMES_FOLDER / labeling_job_id
    labeling_folder.mkdir(parents=True, exist_ok=True)

    frame_metadata = []
    for result in results:
        cropped_video_path = result["video_path"]
        video_frames_dir = labeling_folder / f"{cropped_video_path.stem}_frames"

        extract_success = extract_frames_from_video(
            cropped_video_path,
            video_frames_dir,
            interval=30,
            quality=2
        )

        if extract_success:
            for frame in sorted(video_frames_dir.glob("frame_*.jpg")):
                frame_metadata.append({
                    "filename": frame.name,
                    "table_video": cropped_video_path.stem,
                    "camera": result["camera"],  # NEW: Include camera source
                    "relative_path": f"{video_frames_dir.name}/{frame.name}"
                })

    # Save frame metadata
    metadata_file = labeling_folder / "frames_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(frame_metadata, f, indent=2)

    # Clean up video_path from results (not JSON serializable)
    for result in results:
        del result["video_path"]

    return jsonify({
        "success": True,
        "job_id": job_id,
        "videos": results,
        "count": len(results),
        "labeling_job_id": labeling_job_id,
        "frame_count": len(frame_metadata),
        "cameras_processed": list(set(v["camera"] for v in videos))  # NEW
    })


@app.route("/download/<job_id>/<filename>")
def download(job_id: str, filename: str):
    """Download a cropped video."""
    file_path = OUTPUT_FOLDER / job_id / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True)


@app.route("/download-zip/<job_id>")
def download_zip(job_id: str):
    """Download all cropped videos as a ZIP file."""
    job_folder = OUTPUT_FOLDER / job_id
    if not job_folder.exists():
        return jsonify({"error": "Job not found"}), 404

    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in job_folder.iterdir():
            if file_path.is_file():
                zf.write(file_path, file_path.name)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"cropped_videos_{job_id}.zip"
    )


@app.route("/cleanup", methods=["POST"])
def cleanup():
    """Clean up uploaded and output files."""
    try:
        # Clear uploads
        for f in UPLOAD_FOLDER.iterdir():
            if f.is_file():
                f.unlink()
        # Clear outputs
        for d in OUTPUT_FOLDER.iterdir():
            if d.is_dir():
                shutil.rmtree(d)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload-frame-video", methods=["POST"])
def upload_frame_video():
    """Handle video upload for frame extraction."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = file.filename

    if not allowed_video(filename):
        return jsonify({"error": f"Invalid file type: {filename}"}), 400

    # Generate unique filename
    unique_id = str(uuid.uuid4())[:8]
    ext = Path(filename).suffix
    safe_name = f"{unique_id}_{Path(filename).stem}{ext}"
    save_path = UPLOAD_FOLDER / safe_name

    file.save(save_path)

    # Get video info
    video_info = get_video_info(save_path)

    return jsonify({
        "success": True,
        "filename": safe_name,
        "original": filename,
        "duration": video_info.get("duration", 0),
        "resolution": video_info.get("resolution", "unknown"),
        "fps": video_info.get("fps", 0)
    })


@app.route("/process-frames", methods=["POST"])
def process_frames():
    """Process video and extract frames."""
    data = request.json
    video_filename = data.get("video")
    interval = int(data.get("interval", 30))
    quality = int(data.get("quality", 2))

    if not video_filename:
        return jsonify({"error": "Missing video file"}), 400

    video_path = UPLOAD_FOLDER / video_filename
    if not video_path.exists():
        return jsonify({"error": "Video file not found"}), 404

    # Create unique output folder for this job
    job_id = str(uuid.uuid4())[:8]
    job_output = FRAMES_FOLDER / job_id
    job_output.mkdir(parents=True, exist_ok=True)

    # Extract frames
    success = extract_frames_from_video(video_path, job_output, interval, quality)

    if not success:
        return jsonify({"error": "Frame extraction failed"}), 500

    # Count extracted frames
    frame_files = list(job_output.glob("frame_*.jpg"))

    # Read metadata
    metadata_path = job_output / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    return jsonify({
        "success": True,
        "job_id": job_id,
        "frame_count": len(frame_files),
        "metadata": metadata,
        "download_url": f"/download-frames/{job_id}"
    })


@app.route("/download-frames/<job_id>")
def download_frames(job_id: str):
    """Download all extracted frames as a ZIP file."""
    job_folder = FRAMES_FOLDER / job_id
    if not job_folder.exists():
        return jsonify({"error": "Job not found"}), 404

    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in job_folder.iterdir():
            if file_path.is_file():
                zf.write(file_path, file_path.name)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"frames_{job_id}.zip"
    )


@app.route("/legacy/label/<job_id>")
@app.route("/label/<job_id>")
def label_page(job_id: str):
    """Serve labeling interface for a frame extraction job."""
    return render_template("labeling.html", job_id=job_id)


@app.route("/get-frames/<job_id>")
def get_frames(job_id: str):
    """Get list of frames for labeling."""
    job_folder = FRAMES_FOLDER / job_id
    if not job_folder.exists():
        return jsonify({"error": "Job not found"}), 404

    # Read metadata
    metadata_path = job_folder / "frames_metadata.json"
    frames = []

    if metadata_path.exists():
        with open(metadata_path) as f:
            frames = json.load(f)
    else:
        # Fallback: scan directory
        for frame_file in job_folder.rglob("frame_*.jpg"):
            frames.append({
                "filename": frame_file.name,
                "table_video": "unknown",
                "relative_path": str(frame_file.relative_to(job_folder))
            })

    return jsonify({
        "success": True,
        "frames": frames,
        "frame_count": len(frames)
    })


@app.route("/frame-image/<job_id>/<path:filename>")
def serve_frame(job_id: str, filename: str):
    """Serve individual frame image."""
    # Handle nested paths (video_frames/frame_0001.jpg)
    frame_path = FRAMES_FOLDER / job_id / filename
    if not frame_path.exists():
        return jsonify({"error": "Frame not found"}), 404

    return send_file(frame_path, mimetype="image/jpeg")


@app.route("/download-labeled/<job_id>", methods=["POST"])
def download_labeled(job_id: str):
    """Download labeled frames organized by category."""
    data = request.json
    labels = data.get("labels", {})  # {relative_path: "clean"|"occupied"|"dirty"}

    job_folder = FRAMES_FOLDER / job_id
    if not job_folder.exists():
        return jsonify({"error": "Job not found"}), 404

    # Create ZIP with folders
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add frames to appropriate folders
        for relative_path, label in labels.items():
            # relative_path is like "table_00_frames/frame_0001_00m30s.jpg"
            frame_path = job_folder / relative_path

            if frame_path.exists():
                # Create a unique name: table_name + filename
                # e.g., "table_00_frame_0001_00m30s.jpg"
                parts = relative_path.split('/')
                if len(parts) >= 2:
                    folder_name = parts[0].replace('_frames', '')
                    file_name = parts[-1]
                    unique_name = f"{folder_name}_{file_name}"
                else:
                    unique_name = frame_path.name

                # Add to ZIP in category folder
                zf.write(frame_path, f"{label}/{unique_name}")

        # Add metadata
        metadata = {
            "labels": labels,
            "counts": {
                "clean": sum(1 for l in labels.values() if l == "clean"),
                "occupied": sum(1 for l in labels.values() if l == "occupied"),
                "dirty": sum(1 for l in labels.values() if l == "dirty")
            },
            "total": len(labels)
        }
        zf.writestr("labels.json", json.dumps(metadata, indent=2))

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"labeled_frames_{job_id}.zip"
    )


@app.route("/legacy/api/export-labeled-to-drive/<job_id>", methods=["POST"])
def export_labeled_to_drive(job_id: str):
    """
    Export final legacy labeled frames to Drive.

    Upload scope intentionally excludes intermediate artifacts.
    """
    data = request.get_json(silent=True) or {}
    labels = data.get("labels", {})
    if not isinstance(labels, dict) or not labels:
        return jsonify({"error": "labels payload is required"}), 400

    if not LEGACY_DRIVE_EXPORT_ROOT_ID:
        return jsonify(
            {
                "error": (
                    "DRIVE_OUTPUT_ROOT_MULTICLASS_FOLDER_ID is not configured. "
                    "Set it before using legacy Drive export."
                )
            }
        ), 500

    job_folder = FRAMES_FOLDER / job_id
    if not job_folder.exists():
        return jsonify({"error": "Job not found"}), 404

    try:
        client = get_drive_client()
    except DriveClientError as exc:
        return jsonify({"error": str(exc)}), 500

    try:
        exports_root_id = client.ensure_subfolder(LEGACY_DRIVE_EXPORT_ROOT_ID, "legacy_exports")
        job_export_folder_id = client.ensure_subfolder(exports_root_id, f"job_{job_id}")
    except DriveClientError as exc:
        return jsonify({"error": f"Failed to create export folders: {exc}"}), 500

    uploaded_count = 0
    upload_errors: list[str] = []
    counts = {"clean": 0, "occupied": 0, "dirty": 0}

    for relative_path, label in labels.items():
        if label not in counts:
            upload_errors.append(f"Unsupported label '{label}' for {relative_path}")
            continue

        frame_path = job_folder / relative_path
        if not frame_path.exists():
            upload_errors.append(f"Missing frame file: {relative_path}")
            continue

        parts = relative_path.split("/")
        if len(parts) >= 2:
            folder_name = parts[0].replace("_frames", "")
            file_name = parts[-1]
            upload_name = f"{folder_name}_{file_name}"
        else:
            upload_name = frame_path.name

        try:
            label_folder_id = client.ensure_subfolder(job_export_folder_id, label)
            client.upload_file(
                frame_path,
                label_folder_id,
                file_name=upload_name,
                mime_type="image/jpeg",
            )
            uploaded_count += 1
            counts[label] += 1
        except DriveClientError as exc:
            upload_errors.append(f"{relative_path}: {exc}")

    metadata = {
        "job_id": job_id,
        "labels": labels,
        "counts": counts,
        "uploaded_count": uploaded_count,
        "requested_count": len(labels),
        "errors": upload_errors,
    }

    try:
        client.upload_bytes(
            json.dumps(metadata, indent=2).encode("utf-8"),
            parent_id=job_export_folder_id,
            file_name=f"labels_{job_id}.json",
            mime_type="application/json",
        )
    except DriveClientError as exc:
        upload_errors.append(f"Failed to upload labels metadata JSON: {exc}")

    return jsonify(
        {
            "success": True,
            "job_id": job_id,
            "uploaded_count": uploaded_count,
            "requested_count": len(labels),
            "counts": counts,
            "errors": upload_errors,
        }
    )


if __name__ == "__main__":
    print("=" * 50)
    print("Video Table Cropper")
    print("=" * 50)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    port = int(os.environ.get("PORT", 8080))
    debug = env_flag("FLASK_DEBUG", default=False)
    use_reloader = env_flag("FLASK_USE_RELOADER", default=False)
    print(f"Output folder: {OUTPUT_FOLDER}")
    print()
    print(f"Open http://localhost:{port} in your browser")
    print("=" * 50)
    app.run(debug=debug, host="0.0.0.0", port=port, use_reloader=use_reloader)
