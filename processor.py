"""
--process mode: downloads videos from Drive raw_videos/, extracts frames,
draws table polygon overlays, uploads triplets to temp_processing/, then
crops per-table and uploads to unlabeled/.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from drive_client import DriveClient
from extract_frames import extract_frames
from person_detector import (
    assign_track_ids,
    build_perception_for_table,
    detect_people_in_frame,
    load_yolo_model,
)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
FRAME_INTERVAL_SECONDS = 3  # extract one frame every N seconds


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
    """Ensure all 6 named subfolders exist under the project root. Returns {name: folder_id}."""
    names = ["raw_videos", "temp_processing", "unlabeled", "clean", "dirty", "occupied"]
    return {name: client.ensure_subfolder(project_root_id, name) for name in names}


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
    for item in client.list_files(folder_id):
        if item.get("mimeType") == "application/vnd.google-apps.folder":
            video_files.extend(_find_videos_recursive(client, item["id"]))
        elif Path(item["name"]).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS:
            video_files.append(item)
    return video_files


def run_processor(project_root_id: str, tables_json_path: Path, client: DriveClient) -> None:
    cameras = _load_tables_json(tables_json_path)
    folders = ensure_drive_folders(client, project_root_id)

    print("Scanning raw_videos/ (including subfolders)...")
    video_files = _find_videos_recursive(client, folders["raw_videos"])

    if not video_files:
        print("No videos found in raw_videos/ (or its subfolders).")
        return

    print(f"Found {len(video_files)} video(s) to process.")

    yolo_model = load_yolo_model()

    with tempfile.TemporaryDirectory(prefix="autolabeler_") as tmpdir:
        tmp = Path(tmpdir)
        total = len(video_files)
        video_durations: list[float] = []
        for video_idx, video_meta in enumerate(video_files, 1):
            t_start = time.time()
            print(f"\n[{video_idx}/{total}] ", end="")
            _process_video(video_meta, cameras, folders, client, tmp, yolo_model)
            elapsed = time.time() - t_start
            video_durations.append(elapsed)
            avg = sum(video_durations) / len(video_durations)
            remaining = (total - video_idx) * avg
            print(f"  Progress: {video_idx}/{total} videos done | "
                  f"avg {avg:.0f}s/video | "
                  f"ETA ~{_fmt_eta(remaining)}")

    print("Done.")


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
) -> None:
    video_name = video_meta["name"]
    video_id = video_meta["id"]
    video_stem = Path(video_name).stem

    print(f"\n── {video_name}")

    # Detect camera
    ipc_num = detect_camera_from_filename(video_name)
    if ipc_num is None:
        print(f"  Skipping: could not detect camera from filename '{video_name}'")
        return

    camera = find_camera_tables(cameras, ipc_num)
    if camera is None:
        print(f"  Skipping: no table config found for IPC{ipc_num}")
        return

    tables = camera.get("tables", [])
    if not tables:
        print(f"  Skipping: camera IPC{ipc_num} has no tables defined")
        return

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
        print(f"  Skipping: no usable table polygons for IPC{ipc_num}")
        return

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
        print(f"  Frame extraction failed.")
        return

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        print(f"  No frames extracted.")
        return

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

    # Group into non-overlapping triplets
    triplets = [frame_paths[i:i+3] for i in range(0, len(frame_paths) - 2, 3)]
    if not triplets:
        print(f"  Not enough frames for a triplet (need ≥3, got {len(frame_paths)}).")
        return

    perception_note = " + perception" if yolo_model is not None else " (no perception — install ultralytics)"
    print(f"  {len(frame_paths)} frames → {len(triplets)} triplet(s), {len(table_polygons)} table(s){perception_note}")

    # Build set of already-uploaded triplet names across all destination folders for resume
    already_uploaded: set[str] = set()
    for dest in ("temp_processing", "unlabeled", "clean", "dirty", "occupied"):
        for f in client.list_folders(folders[dest]):
            already_uploaded.add(f["name"])

    for triplet_idx, triplet in enumerate(triplets):
        triplet_name = f"{video_stem}_t{triplet_idx:04d}"

        # Resume: skip if this triplet was already uploaded
        if triplet_name in already_uploaded:
            print(f"  Triplet {triplet_idx+1}/{len(triplets)} already uploaded, skipping.")
            continue

        # ── Person detection for this triplet (full frames, once per triplet) ──
        frame_detections = None
        if yolo_model is not None:
            frame_detections = [detect_people_in_frame(fp, yolo_model) for fp in triplet]
            assign_track_ids(frame_detections)

        # ── temp_processing: 3 frames with all overlays drawn ──────────────
        temp_subfolder_id = client.ensure_subfolder(folders["temp_processing"], triplet_name)
        for frame_idx, frame_path in enumerate(triplet):
            overlaid = draw_overlays(frame_path, all_polygons)
            if frame_detections is not None:
                overlaid = draw_person_detections(overlaid, frame_detections[frame_idx])
            out = tmp / "overlay" / f"{triplet_name}_f{frame_idx}.jpg"
            save_jpeg(overlaid, out)
            client.upload_or_update_file(out, temp_subfolder_id, file_name=f"frame_{frame_idx}.jpg")

        # ── unlabeled: one subfolder per table, 3 cropped frames + perception ──
        for table_id, tight_poly, tight_bbox, zone_poly in table_polygons:
            subfolder_name = f"{video_stem}_{table_id}_t{triplet_idx:04d}"
            unlabeled_subfolder_id = client.ensure_subfolder(folders["unlabeled"], subfolder_name)

            for frame_idx, frame_path in enumerate(triplet):
                cropped = perspective_crop_polygon(frame_path, zone_poly)
                out = tmp / "crops" / f"{subfolder_name}_f{frame_idx}.jpg"
                save_jpeg(cropped, out)
                client.upload_or_update_file(out, unlabeled_subfolder_id, file_name=f"frame_{frame_idx}.jpg")

            # Write perception.json if person detection ran
            if frame_detections is not None:
                perception = build_perception_for_table(frame_detections, tight_poly, img_shape)
                perc_path = tmp / "perception" / f"{subfolder_name}_perception.json"
                perc_path.parent.mkdir(parents=True, exist_ok=True)
                perc_path.write_text(json.dumps(perception, indent=2), encoding="utf-8")
                client.upload_or_update_file(perc_path, unlabeled_subfolder_id, file_name="perception.json")

        pct = int((triplet_idx + 1) / len(triplets) * 100)
        print(f"  Triplet {triplet_idx+1}/{len(triplets)} ({pct}%) uploaded.")

    print(f"  Done: {video_name}")
