"""
demo_overlay_video.py

Takes a folder with a single video (input_demo/), detects the IPC camera number
from the filename, loads table polygons from approved_table_rectangles.json,
runs YOLOv8 person detection on every frame, draws both overlays, and
reconstructs an output video: input_demo/output_overlay.mp4

Usage:
    python demo_overlay_video.py
    python demo_overlay_video.py --input input_demo --output output_overlay.mp4
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from processor import (
    draw_overlays,
    draw_person_detections,
    find_camera_tables,
    polygon_from_table,
    _load_tables_json,
    _ipc_number,
)
from person_detector import (
    assign_track_ids,
    detect_people_in_frame,
    load_yolo_model,
)


TABLES_JSON = Path(__file__).parent / "approved_table_rectangles.json"


def _build_table_polygons(camera: dict) -> list[list[tuple[float, float]]]:
    polygons = []
    for table in camera.get("tables", []):
        poly = polygon_from_table(table)
        if poly:
            polygons.append(poly)
    return polygons


def _scale_polygons(polygons, ref_w, ref_h, frame_w, frame_h):
    if ref_w == frame_w and ref_h == frame_h:
        return polygons
    sx = frame_w / ref_w
    sy = frame_h / ref_h
    return [[(x * sx, y * sy) for x, y in poly] for poly in polygons]


def process_video(input_path: Path, output_path: Path) -> None:
    # ── Load table JSON ────────────────────────────────────────────────────
    cameras = _load_tables_json(TABLES_JSON)

    # ── Detect IPC number from filename ───────────────────────────────────
    ipc_num = _ipc_number(input_path.stem)
    if ipc_num is None:
        # try scanning the full path components
        for part in input_path.parts:
            ipc_num = _ipc_number(part)
            if ipc_num is not None:
                break
    if ipc_num is None:
        print(f"Could not detect IPC number from '{input_path.name}'.")
        print("Available cameras:", [c.get("camera_id") for c in cameras])
        sys.exit(1)

    camera = find_camera_tables(cameras, ipc_num)
    if camera is None:
        print(f"No camera entry found for IPC{ipc_num} in {TABLES_JSON.name}")
        sys.exit(1)

    print(f"  Camera: {camera.get('camera_id')}  ({len(camera.get('tables', []))} tables)")

    # ── Load YOLO ─────────────────────────────────────────────────────────
    yolo_model = load_yolo_model()
    if yolo_model is None:
        print("ERROR: ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    # ── Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {input_path}")
        sys.exit(1)

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Video: {frame_w}x{frame_h}  {fps:.1f}fps  ~{total} frames")

    # ── Scale table polygons to match actual frame size ────────────────────
    ref_w = int(camera.get("image_width")  or frame_w)
    ref_h = int(camera.get("image_height") or frame_h)
    raw_polygons = _build_table_polygons(camera)
    polygons     = _scale_polygons(raw_polygons, ref_w, ref_h, frame_w, frame_h)

    print(f"  Drawing {len(polygons)} table polygon(s)")

    # ── Set up writer ─────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))

    frame_idx = 0
    with tempfile.TemporaryDirectory(prefix="demo_overlay_") as tmpdir:
        tmp = Path(tmpdir)

        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            # Save frame to temp file so existing PIL-based helpers can load it
            frame_path = tmp / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), bgr)

            # ── Table polygon overlay ──────────────────────────────────────
            pil_img = draw_overlays(frame_path, polygons)

            # ── Person detection overlay ───────────────────────────────────
            detections = detect_people_in_frame(frame_path, yolo_model)
            # assign_track_ids works on a list-of-frames; here we pass single frame
            assign_track_ids([detections])
            pil_img = draw_person_detections(pil_img, detections)

            # ── Write frame ───────────────────────────────────────────────
            out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            writer.write(out_bgr)

            frame_idx += 1
            if frame_idx % 30 == 0:
                pct = int(frame_idx / max(total, 1) * 100)
                print(f"  Frame {frame_idx}/{total} ({pct}%)")

    cap.release()
    writer.release()
    print(f"\nDone. Output: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="input_demo",        help="Input folder containing the video")
    parser.add_argument("--output", default="output_overlay.mp4", help="Output video filename (inside input folder)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"ERROR: Input folder not found: {input_dir}")
        sys.exit(1)

    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    videos = [f for f in input_dir.iterdir() if f.suffix.lower() in video_extensions]
    if not videos:
        print(f"ERROR: No video files found in {input_dir}")
        sys.exit(1)
    if len(videos) > 1:
        print(f"Multiple videos found, using: {videos[0].name}")

    input_path  = videos[0]
    output_path = input_dir / args.output

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    process_video(input_path, output_path)


if __name__ == "__main__":
    main()
