"""
--process mode: downloads videos from Drive raw_videos/, extracts frames,
draws table polygon overlays, uploads frame groups to temp_processing/, then
crops per-table and uploads to unlabeled/.

Internally a "group" is N frames (default 10 via FRAMES_PER_GROUP_VIDEO);
historically called a "triplet" when N was 3, and the name is preserved on
folder suffixes (`_t{idx:04d}`) and metadata keys for backwards compatibility.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from drive_client import DriveClient
from extract_frames import extract_frames
from person_detector import (
    assign_track_ids,
    build_perception_for_table,
    detect_people_batch,
    load_yolo_model,
)
from queue_metadata import build_folder_app_properties

VIDEO_FOLDER_PREFIX = "mimosas"
KNOWN_FOLDER_PREFIXES = ("mimosas", "matthews")
PREPROCESS_STATUS_PROPERTY = "autolabel_preprocess_status"
PREPROCESS_AT_PROPERTY = "autolabel_preprocessed_at"
PREPROCESS_ERROR_PROPERTY = "autolabel_preprocess_error"
PREPROCESS_TRIPLETS_PROPERTY = "autolabel_preprocess_triplets"
PREPROCESS_COMPLETE_STATUSES = {"complete", "skipped", "error"}
PROCESSED_RAW_FOLDER_NAME = "processed_raw"
PROCESSED_RAW_RETENTION_DAYS = max(
    1, int(os.environ.get("PROCESSED_RAW_RETENTION_DAYS", "14") or "14")
)

AUTOLABEL_UPLOAD_WORKERS = max(1, int(os.environ.get("AUTOLABEL_UPLOAD_WORKERS", "16") or "16"))
AUTOLABEL_TABLE_WORKERS = max(1, int(os.environ.get("AUTOLABEL_TABLE_WORKERS", "8") or "8"))


def _apply_video_prefix(name: str) -> str:
    """Prepend the video source's restaurant prefix to a folder name, idempotently."""
    if not name:
        return name
    for known in KNOWN_FOLDER_PREFIXES:
        if name.startswith(f"{known}-"):
            return name
    return f"{VIDEO_FOLDER_PREFIX}-{name}"

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
FRAME_INTERVAL_SECONDS = 3  # extract one frame every N seconds

# Group-size bounds. The "triplet" name is preserved everywhere it touches the
# wire/Drive protocol (folder suffix `_t{idx:04d}`, app-property
# `autolabel_preprocess_triplets`, edge config `frames_per_triplet`) for
# backwards compatibility with deployed clients and existing labeled data.
# Internally we treat a "group" as N frames; N is detected per-group at runtime.
MIN_FRAMES_PER_GROUP = 2
MAX_FRAMES_PER_GROUP = 16
# Default group size for the video-import path (where the processor regroups a
# flat list of frames extracted from a single video file). The Pi-zip path
# detects N from the manifest instead.
FRAMES_PER_GROUP_VIDEO = 10


def detect_n_frames(
    frame_paths: list[Path] | None = None,
    metadata: dict | None = None,
    *,
    file_count: int | None = None,
) -> int:
    """Infer the number of frames in a group.

    Priority: explicit metadata field → file_count argument → len(frame_paths).
    Validates the result against MIN/MAX_FRAMES_PER_GROUP.
    """
    n: int | None = None
    if metadata:
        meta_n = metadata.get("frames_per_triplet")
        if isinstance(meta_n, int):
            n = meta_n
    if n is None and file_count is not None:
        n = int(file_count)
    if n is None and frame_paths is not None:
        n = len(frame_paths)
    if n is None:
        raise ValueError("detect_n_frames: no source provided")
    if not (MIN_FRAMES_PER_GROUP <= n <= MAX_FRAMES_PER_GROUP):
        raise ValueError(
            f"n_frames={n} outside allowed range [{MIN_FRAMES_PER_GROUP},{MAX_FRAMES_PER_GROUP}]"
        )
    return n


def frame_keys(n_frames: int) -> tuple[str, ...]:
    """('frame_0', ..., 'frame_{n-1}')"""
    return tuple(f"frame_{i}" for i in range(n_frames))


def frame_filenames(n_frames: int) -> tuple[str, ...]:
    """('frame_0.jpg', ..., 'frame_{n-1}.jpg')"""
    return tuple(f"frame_{i}.jpg" for i in range(n_frames))


@dataclass
class VideoProcessResult:
    status: str
    uploaded_triplets: int = 0
    reason: str = ""


@dataclass
class ProcessorRunSummary:
    scanned: int = 0
    already_marked: int = 0
    completed: int = 0
    skipped: int = 0
    errored: int = 0
    videos_with_new_work: int = 0
    triplets_uploaded: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.scanned - self.already_marked - self.completed - self.skipped - self.errored)


# ── camera / table loading ─────────────────────────────────────────────────

def _load_tables_json(json_path: Path) -> list[dict[str, Any]]:
    """Load the approved_table_rectangles.json file."""
    if not json_path.exists():
        raise FileNotFoundError(f"Table JSON not found: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8")).get("cameras", [])


def _ipc_number(text: str) -> int | None:
    """Extract the IPC number from a string like 'IPC3', 'Mimosas IPC3', '3_something'."""
    m = re.search(r"IPC[\s_-]*(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"(\d+)", text.strip())
    if m:
        return int(m.group(1))
    return None


def detect_camera_from_filename(filename: str) -> int | None:
    """Return IPC number from a video filename, or None if undetectable."""
    name = Path(filename).stem
    n = _ipc_number(name)
    if n is not None:
        return n
    m = re.match(r"^\s*(\d+)(?:[_\-]|$)", name)
    if m:
        return int(m.group(1))
    return None


def find_camera_tables(cameras: list[dict], ipc_number: int) -> dict[str, Any] | None:
    """Find the camera entry whose name/number matches the given IPC number."""
    for cam in cameras:
        # check camera_number field
        if cam.get("camera_number") == ipc_number:
            return cam
        # check camera_id / camera_name / source_image for IPC mention
        for field in ("camera_id", "camera_name", "source_image", "camera"):
            n = _ipc_number(str(cam.get(field, "")))
            if n == ipc_number:
                return cam
    return None


def polygon_from_table(table: dict[str, Any]) -> list[tuple[float, float]] | None:
    """Return the tight_rect polygon as a list of (x, y) tuples, or None."""
    tight_rect = table.get("tight_rect")
    if tight_rect and tight_rect.get("polygon"):
        return [(float(p[0]), float(p[1])) for p in tight_rect["polygon"]]
    return None


def zone_polygon_from_table(table: dict[str, Any]) -> list[tuple[float, float]] | None:
    """Return the zone_rect polygon as (x, y) tuples, or None if absent."""
    zone_rect = table.get("zone_rect")
    if zone_rect and zone_rect.get("polygon"):
        return [(float(p[0]), float(p[1])) for p in zone_rect["polygon"]]
    return None


def perspective_crop_polygon(image_path: Path, polygon: list[tuple[float, float]]) -> Image.Image:
    """
    Warp-crop image to the exact bounds of a (possibly rotated) 4-point polygon.
    Sorts the 4 points into top-left, top-right, bottom-right, bottom-left order,
    then uses PIL QUAD transform to deskew to a clean rectangle.
    Falls back to axis-aligned crop if polygon doesn't have exactly 4 points.
    """
    img = Image.open(image_path).convert("RGB")
    if len(polygon) != 4:
        # Fallback: axis-aligned crop
        bbox = bbox_from_polygon(polygon)
        return img.crop(bbox)

    pts = sorted(polygon, key=lambda p: p[1])   # sort by y
    top2 = sorted(pts[:2], key=lambda p: p[0])  # left-to-right within top pair
    bot2 = sorted(pts[2:], key=lambda p: p[0])  # left-to-right within bottom pair
    tl, tr = top2
    bl, br = bot2

    out_w = int(round(max(math.hypot(tr[0]-tl[0], tr[1]-tl[1]),
                          math.hypot(br[0]-bl[0], br[1]-bl[1]))))
    out_h = int(round(max(math.hypot(bl[0]-tl[0], bl[1]-tl[1]),
                          math.hypot(br[0]-tr[0], br[1]-tr[1]))))
    out_w = max(out_w, 1)
    out_h = max(out_h, 1)

    # PIL QUAD data: source coords mapping to dest corners (0,0),(0,h),(w,h),(w,0)
    data = [tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1]]
    return img.transform((out_w, out_h), Image.QUAD, data, Image.BICUBIC)


def bbox_from_polygon(polygon: list[tuple[float, float]]) -> tuple[int, int, int, int]:
    """Return axis-aligned bounding box (x1, y1, x2, y2) for a polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (int(math.floor(min(xs))), int(math.floor(min(ys))),
            int(math.ceil(max(xs))), int(math.ceil(max(ys))))


# ── image helpers ──────────────────────────────────────────────────────────

def draw_overlays(image_path: Path, polygons: list[list[tuple[float, float]]]) -> Image.Image:
    """Open image, draw all polygons as semi-transparent overlays, return PIL image."""
    img = Image.open(image_path).convert("RGB")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = [
        (255, 59, 48, 100),   # red
        (52, 199, 89, 100),   # green
        (0, 122, 255, 100),   # blue
        (255, 149, 0, 100),   # orange
        (175, 82, 222, 100),  # purple
        (255, 204, 0, 100),   # yellow
    ]
    for i, polygon in enumerate(polygons):
        color = colors[i % len(colors)]
        draw.polygon(polygon, fill=color, outline=(255, 255, 255, 220))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def draw_person_detections(img: Image.Image, detections: list[dict]) -> Image.Image:
    """Draw person bounding boxes and track IDs onto a PIL image (for debug temp_processing frames)."""
    draw = ImageDraw.Draw(img)
    colors = [
        (255, 59, 48),   (52, 199, 89),  (0, 122, 255),
        (255, 149, 0),   (175, 82, 222), (255, 204, 0),
        (90, 200, 250),  (255, 45, 85),
    ]
    track_color_map: dict[str, tuple] = {}
    color_idx = 0
    for det in detections:
        tid = det.get("track_id", "?")
        if tid not in track_color_map:
            track_color_map[tid] = colors[color_idx % len(colors)]
            color_idx += 1
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        color = track_color_map[tid]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{tid} {det['score']:.2f}"
        draw.rectangle([x1, y1 - 16, x1 + len(label) * 7, y1], fill=color)
        draw.text((x1 + 2, y1 - 15), label, fill=(255, 255, 255))
    return img


def crop_to_bbox(image_path: Path, bbox: tuple[int, int, int, int]) -> Image.Image:
    """Crop image to a bounding box (x1, y1, x2, y2), clamped to image bounds."""
    img = Image.open(image_path).convert("RGB")
    x1 = max(0, bbox[0])
    y1 = max(0, bbox[1])
    x2 = min(img.width, bbox[2])
    y2 = min(img.height, bbox[3])
    return img.crop((x1, y1, x2, y2))


def save_jpeg(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=90)


# ── Drive folder helpers ───────────────────────────────────────────────────

def ensure_drive_folders(client: DriveClient, project_root_id: str) -> dict[str, str]:
    """Ensure all label workflow subfolders exist under the project root."""
    names = [
        "raw_videos",
        "temp_processing",
        "unlabeled",
        "clean",
        "dirty",
        "occupied",
        "label_later",
        "discarded",
        PROCESSED_RAW_FOLDER_NAME,
    ]
    return {name: client.ensure_subfolder(project_root_id, name) for name in names}


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


def _move_video_to_processed_raw(
    client: DriveClient,
    video_meta: dict[str, Any],
    folders: dict[str, str],
) -> None:
    processed_id = folders.get(PROCESSED_RAW_FOLDER_NAME)
    if not processed_id:
        return
    parents = [str(parent) for parent in video_meta.get("parents", []) if parent]
    if parents == [processed_id]:
        return
    remove_parent_id = parents[0] if parents else folders.get("raw_videos")
    client.move_file(
        str(video_meta["id"]),
        new_parent_id=processed_id,
        remove_parent_id=remove_parent_id,
    )
    video_meta["parents"] = [processed_id]


def _fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


# ── main process loop ──────────────────────────────────────────────────────

def _find_videos_recursive(client: DriveClient, folder_id: str) -> list[dict]:
    """Recursively find all video files in a folder and its subfolders. Skips images."""
    video_files = []
    for item in client.list_files(folder_id, fields="id,name,mimeType,parents,appProperties"):
        if item.get("mimeType") == "application/vnd.google-apps.folder":
            video_files.extend(_find_videos_recursive(client, item["id"]))
        elif Path(item["name"]).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS:
            video_files.append(item)
    return video_files


def _video_preprocess_status(video_meta: dict[str, Any]) -> str | None:
    app_properties = video_meta.get("appProperties") or {}
    status = app_properties.get(PREPROCESS_STATUS_PROPERTY)
    return str(status) if status else None


def _mark_video_preprocess_status(
    client: DriveClient,
    video_id: str,
    status: str,
    *,
    reason: str = "",
    uploaded_triplets: int = 0,
) -> None:
    properties = {
        PREPROCESS_STATUS_PROPERTY: status,
        PREPROCESS_AT_PROPERTY: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        PREPROCESS_TRIPLETS_PROPERTY: str(uploaded_triplets),
    }
    if reason:
        properties[PREPROCESS_ERROR_PROPERTY] = reason[:1200]
    client.update_file_metadata(
        video_id,
        {"appProperties": properties},
        fields="id,name,mimeType,parents,appProperties",
    )


def run_processor(
    project_root_id: str,
    tables_json_path: Path,
    client: DriveClient,
    max_videos: int | None = None,
) -> ProcessorRunSummary:
    cameras = _load_tables_json(tables_json_path)
    folders = ensure_drive_folders(client, project_root_id)
    _cleanup_processed_raw_folder(client, folders[PROCESSED_RAW_FOLDER_NAME])
    summary = ProcessorRunSummary()

    print("Scanning raw_videos/ (including subfolders)...")
    video_files = _find_videos_recursive(client, folders["raw_videos"])
    summary.scanned = len(video_files)

    if not video_files:
        print("No videos found in raw_videos/ (or its subfolders).")
        return summary

    max_new_videos = max_videos if max_videos is not None and max_videos > 0 else None

    if max_new_videos is None:
        print(f"Found {len(video_files)} video(s) to process.")
    else:
        print(f"Found {len(video_files)} candidate video(s); processing up to {max_new_videos} with new work.")

    yolo_model = load_yolo_model()

    with tempfile.TemporaryDirectory(prefix="autolabeler_") as tmpdir:
        tmp = Path(tmpdir)
        total = len(video_files)
        processed_videos = 0
        video_durations: list[float] = []
        for video_idx, video_meta in enumerate(video_files, 1):
            existing_status = _video_preprocess_status(video_meta)
            if existing_status in PREPROCESS_COMPLETE_STATUSES:
                print(f"\n[{video_idx}/{total}] {video_meta['name']} already marked {existing_status}, skipping.")
                summary.already_marked += 1
                continue

            t_start = time.time()
            print(f"\n[{video_idx}/{total}] ", end="")
            try:
                result = _process_video(video_meta, cameras, folders, client, tmp, yolo_model)
                _mark_video_preprocess_status(
                    client,
                    str(video_meta["id"]),
                    result.status,
                    reason=result.reason,
                    uploaded_triplets=result.uploaded_triplets,
                )
            except Exception as exc:
                result = VideoProcessResult(status="error", reason=str(exc))
                _mark_video_preprocess_status(
                    client,
                    str(video_meta["id"]),
                    "error",
                    reason=str(exc),
                )
                print(f"  Error: {exc}")

            if result.status == "complete":
                summary.completed += 1
                _move_video_to_processed_raw(client, video_meta, folders)
            elif result.status == "skipped":
                summary.skipped += 1
            elif result.status == "error":
                summary.errored += 1

            if result.uploaded_triplets > 0:
                processed_videos += 1
                summary.videos_with_new_work += 1
                summary.triplets_uploaded += result.uploaded_triplets
            elapsed = time.time() - t_start
            video_durations.append(elapsed)
            avg = sum(video_durations) / len(video_durations)
            remaining = (total - video_idx) * avg
            if max_new_videos is None:
                print(f"  Progress: {video_idx}/{total} videos scanned | "
                      f"avg {avg:.0f}s/video | "
                      f"ETA ~{_fmt_eta(remaining)}")
            else:
                print(f"  Progress: {video_idx}/{total} videos scanned | "
                      f"{processed_videos}/{max_new_videos} with new work | "
                      f"avg {avg:.0f}s/video | "
                      f"ETA ~{_fmt_eta(remaining)}")
                if processed_videos >= max_new_videos:
                    break

    print(
        "Done. "
        f"scanned={summary.scanned} already_marked={summary.already_marked} "
        f"completed={summary.completed} skipped={summary.skipped} errored={summary.errored} "
        f"triplets_uploaded={summary.triplets_uploaded}"
    )
    return summary


def _scale_bbox(bbox: tuple[int,int,int,int], sx: float, sy: float) -> tuple[int,int,int,int]:
    return (int(bbox[0]*sx), int(bbox[1]*sy), int(bbox[2]*sx), int(bbox[3]*sy))


def _scale_table_polygons(
    table_polygons: list,
    ref_w: int,
    ref_h: int,
    frame_w: int,
    frame_h: int,
) -> list:
    """Scale table polygon coords from JSON reference resolution to actual frame resolution."""
    sx = frame_w / ref_w
    sy = frame_h / ref_h
    scaled = []
    for table_id, tight_poly, tight_bbox, zone_poly in table_polygons:
        new_tight = [(x * sx, y * sy) for x, y in tight_poly]
        new_zone  = [(x * sx, y * sy) for x, y in zone_poly]
        scaled.append((table_id, new_tight, _scale_bbox(tight_bbox, sx, sy), new_zone))
    return scaled


def _process_video(
    video_meta: dict[str, Any],
    cameras: list[dict],
    folders: dict[str, str],
    client: DriveClient,
    tmp: Path,
    yolo_model=None,
) -> VideoProcessResult:
    video_name = video_meta["name"]
    video_id = video_meta["id"]
    video_stem = Path(video_name).stem

    print(f"\n── {video_name}")

    # Detect camera
    ipc_num = detect_camera_from_filename(video_name)
    if ipc_num is None:
        reason = f"could not detect camera from filename '{video_name}'"
        print(f"  Skipping: {reason}")
        return VideoProcessResult(status="skipped", reason=reason)

    camera = find_camera_tables(cameras, ipc_num)
    if camera is None:
        reason = f"no table config found for IPC{ipc_num}"
        print(f"  Skipping: {reason}")
        return VideoProcessResult(status="skipped", reason=reason)

    tables = camera.get("tables", [])
    if not tables:
        reason = f"camera IPC{ipc_num} has no tables defined"
        print(f"  Skipping: {reason}")
        return VideoProcessResult(status="skipped", reason=reason)

    # Build polygon list per table
    # tuple: (table_id, tight_polygon, tight_bbox, zone_polygon)
    table_polygons: list[tuple[str, list, tuple[int,int,int,int], list]] = []
    for i, table in enumerate(tables):
        tight_poly = polygon_from_table(table)
        if tight_poly is None:
            print(f"  Warning: table {i} has no polygon, skipping")
            continue
        table_id = str(table.get("label") or table.get("mask_id") or i).replace(" ", "_")
        tight_bbox = bbox_from_polygon(tight_poly)
        zone_poly = zone_polygon_from_table(table) or tight_poly  # fallback to tight if no zone_rect
        table_polygons.append((table_id, tight_poly, tight_bbox, zone_poly))

    if not table_polygons:
        reason = f"no usable table polygons for IPC{ipc_num}"
        print(f"  Skipping: {reason}")
        return VideoProcessResult(status="skipped", reason=reason)

    all_polygons = [p for _, p, _, _ in table_polygons]

    # Download video
    video_path = tmp / video_name
    print(f"  Downloading...")
    client.download_file_to_path(video_id, video_path)

    # Extract frames
    frames_dir = tmp / f"{video_stem}_frames"
    print(f"  Extracting frames every {FRAME_INTERVAL_SECONDS}s...")
    success, frames_info = extract_frames(video_path, frames_dir, interval=FRAME_INTERVAL_SECONDS)
    if not success or not frames_info:
        reason = "frame extraction failed"
        print(f"  {reason}.")
        return VideoProcessResult(status="error", reason=reason)

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        reason = "no frames extracted"
        print(f"  {reason}.")
        return VideoProcessResult(status="error", reason=reason)

    # Get actual frame resolution and scale table polygons if needed
    with Image.open(frame_paths[0]) as _img:
        frame_h, frame_w = _img.height, _img.width
    img_shape = (frame_h, frame_w)

    ref_w = int(camera.get("image_width") or camera.get("frame_width") or camera.get("width") or frame_w)
    ref_h = int(camera.get("image_height") or camera.get("frame_height") or camera.get("height") or frame_h)
    if ref_w != frame_w or ref_h != frame_h:
        print(f"  Scaling polygons from {ref_w}×{ref_h} → {frame_w}×{frame_h}")
        table_polygons = _scale_table_polygons(table_polygons, ref_w, ref_h, frame_w, frame_h)
        all_polygons = [p for _, p, _, _ in table_polygons]

    # Group frames into non-overlapping groups of FRAMES_PER_GROUP_VIDEO.
    # The variable is still called `triplets` and folder names still use the
    # `_t{idx:04d}` suffix for protocol/data backwards compatibility — see the
    # comment on FRAMES_PER_GROUP_VIDEO above.
    n_frames = FRAMES_PER_GROUP_VIDEO
    triplets = [
        frame_paths[i:i + n_frames]
        for i in range(0, len(frame_paths) - (n_frames - 1), n_frames)
    ]
    if not triplets:
        reason = f"not enough frames for a group (need >={n_frames}, got {len(frame_paths)})"
        print(f"  {reason}.")
        return VideoProcessResult(status="skipped", reason=reason)

    perception_note = " + perception" if yolo_model is not None else " (no perception — install ultralytics)"
    print(f"  {len(frame_paths)} frames → {len(triplets)} group(s) of {n_frames}, {len(table_polygons)} table(s){perception_note}")

    # Build set of generated folder names across all destinations for resume.
    # This intentionally includes legacy unprefixed names so online deploys do
    # not duplicate older Drive data under the newer restaurant prefixes.
    already_uploaded: set[str] = set()
    for dest in ("temp_processing", "unlabeled", "clean", "dirty", "occupied", "label_later", "discarded"):
        for f in client.list_folders(folders[dest]):
            already_uploaded.add(f["name"])

    upload_executor = ThreadPoolExecutor(max_workers=AUTOLABEL_UPLOAD_WORKERS)
    table_executor = ThreadPoolExecutor(max_workers=AUTOLABEL_TABLE_WORKERS)
    uploaded_triplets = 0
    try:
        for triplet_idx, triplet in enumerate(triplets):
            raw_triplet_name = f"{video_stem}_t{triplet_idx:04d}"
            triplet_name = _apply_video_prefix(raw_triplet_name)

            # Resume: skip if this triplet was already uploaded
            if triplet_name in already_uploaded or raw_triplet_name in already_uploaded:
                print(f"  Triplet {triplet_idx+1}/{len(triplets)} already uploaded, skipping.")
                continue

            # ── Person detection for this triplet (full frames, once per triplet) ──
            frame_detections = None
            if yolo_model is not None:
                frame_detections = detect_people_batch(list(triplet), yolo_model)
                assign_track_ids(frame_detections)

            # ── temp_processing: 3 frames with all overlays drawn (parallel upload) ──
            temp_subfolder_id = client.ensure_subfolder(folders["temp_processing"], triplet_name)
            overlay_futures = []
            for frame_idx, frame_path in enumerate(triplet):
                overlaid = draw_overlays(frame_path, all_polygons)
                if frame_detections is not None:
                    overlaid = draw_person_detections(overlaid, frame_detections[frame_idx])
                out = tmp / "overlay" / f"{triplet_name}_f{frame_idx}.jpg"
                save_jpeg(overlaid, out)
                overlay_futures.append(
                    upload_executor.submit(
                        client.upload_or_update_file,
                        out,
                        temp_subfolder_id,
                        file_name=f"frame_{frame_idx}.jpg",
                    )
                )
            for fut in overlay_futures:
                fut.result()

            # ── unlabeled: one subfolder per table — tables processed in parallel ──
            def _process_table(entry):
                table_id, tight_poly, _tight_bbox, zone_poly = entry
                raw_subfolder_name = f"{video_stem}_{table_id}_t{triplet_idx:04d}"
                subfolder_name = _apply_video_prefix(raw_subfolder_name)
                if subfolder_name in already_uploaded or raw_subfolder_name in already_uploaded:
                    print(f"  Table folder {raw_subfolder_name} already exists, skipping.")
                    return False

                unlabeled_subfolder_id = client.ensure_subfolder(
                    folders["unlabeled"], subfolder_name
                )
                uploaded_frame_ids: dict[str, str | None] = {
                    f"frame_{i}": None for i in range(n_frames)
                }

                crop_futures = {}
                for frame_idx, frame_path in enumerate(triplet):
                    cropped = perspective_crop_polygon(frame_path, zone_poly)
                    out = tmp / "crops" / f"{subfolder_name}_f{frame_idx}.jpg"
                    save_jpeg(cropped, out)
                    crop_futures[frame_idx] = upload_executor.submit(
                        client.upload_or_update_file,
                        out,
                        unlabeled_subfolder_id,
                        file_name=f"frame_{frame_idx}.jpg",
                    )
                for frame_idx, fut in crop_futures.items():
                    uploaded = fut.result()
                    uploaded_frame_ids[f"frame_{frame_idx}"] = str(uploaded["id"])

                client.update_file_metadata(
                    unlabeled_subfolder_id,
                    {"appProperties": build_folder_app_properties(uploaded_frame_ids)},
                )

                if frame_detections is not None:
                    perception = build_perception_for_table(
                        frame_detections, tight_poly, img_shape, n_frames=n_frames
                    )
                    perc_path = tmp / "perception" / f"{subfolder_name}_perception.json"
                    perc_path.parent.mkdir(parents=True, exist_ok=True)
                    perc_path.write_text(json.dumps(perception, indent=2), encoding="utf-8")
                    client.upload_or_update_file(
                        perc_path,
                        unlabeled_subfolder_id,
                        file_name="perception.json",
                    )
                return True

            table_futures = [
                table_executor.submit(_process_table, entry)
                for entry in table_polygons
            ]
            table_uploaded = False
            for fut in as_completed(table_futures):
                table_uploaded = bool(fut.result()) or table_uploaded

            pct = int((triplet_idx + 1) / len(triplets) * 100)
            if table_uploaded:
                print(f"  Triplet {triplet_idx+1}/{len(triplets)} ({pct}%) uploaded.")
                uploaded_triplets += 1
            else:
                print(f"  Triplet {triplet_idx+1}/{len(triplets)} ({pct}%) had no new table crops.")
    finally:
        upload_executor.shutdown(wait=True)
        table_executor.shutdown(wait=True)

    if uploaded_triplets == 0:
        print(f"  No new triplets to upload.")
        return VideoProcessResult(status="complete", uploaded_triplets=0)

    print(f"  Done: {video_name}")
    return VideoProcessResult(status="complete", uploaded_triplets=uploaded_triplets)
