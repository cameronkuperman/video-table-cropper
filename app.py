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
import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
OUTPUT_FOLDER = Path(__file__).parent / "output"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_JSON_EXTENSIONS = {".json"}

# Ensure folders exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)


def allowed_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def allowed_json(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_JSON_EXTENSIONS


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


@app.route("/")
def index():
    return render_template("index.html")


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


@app.route("/process", methods=["POST"])
def process():
    """Process video with JSON bounding boxes (supports both axis-aligned and rotated)."""
    data = request.json
    video_filename = data.get("video")
    json_filenames = data.get("jsons", [])

    if not video_filename or not json_filenames:
        return jsonify({"error": "Missing video or JSON files"}), 400

    video_path = UPLOAD_FOLDER / video_filename
    if not video_path.exists():
        return jsonify({"error": "Video file not found"}), 404

    # Create unique output folder for this job
    job_id = str(uuid.uuid4())[:8]
    job_output = OUTPUT_FOLDER / job_id
    job_output.mkdir(exist_ok=True)

    results = []

    for json_filename in json_filenames:
        json_path = UPLOAD_FOLDER / json_filename
        if not json_path.exists():
            continue

        try:
            with open(json_path) as f:
                json_data = json.load(f)
        except json.JSONDecodeError:
            continue

        tables = json_data.get("tables", [])
        video_name = json_data.get("video_name", video_path.stem)

        # Get frame dimensions from JSON (for rotated bbox support)
        frame_width = json_data.get("frame_width", 1920)
        frame_height = json_data.get("frame_height", 1080)

        for idx, table in enumerate(tables):
            # Skip if saved=false or skip_reason is set
            if not table.get("saved", True):
                continue
            if table.get("skip_reason"):
                continue

            table_id = table.get("id", idx)
            output_name = f"{video_name}_table_{table_id:02d}.mp4"
            output_path = job_output / output_name

            # Check for rotated_bbox (new format) vs bbox (old format)
            rotated_bbox = table.get("rotated_bbox")
            bbox = table.get("bbox", {})

            success = False

            if rotated_bbox and rotated_bbox.get("corners"):
                # New format: rotated bounding box
                success = crop_rotated_video(
                    video_path, output_path, rotated_bbox, frame_width, frame_height
                )
                bbox_info = {
                    "center": rotated_bbox.get("center"),
                    "size": rotated_bbox.get("size"),
                    "angle": rotated_bbox.get("angle")
                }
            elif bbox:
                # Old format: axis-aligned bounding box
                x1 = bbox.get("x1", 0)
                y1 = bbox.get("y1", 0)
                x2 = bbox.get("x2", 0)
                y2 = bbox.get("y2", 0)

                if x2 > x1 and y2 > y1:
                    success = crop_video(video_path, output_path, x1, y1, x2, y2)
                bbox_info = bbox
            else:
                continue

            if success:
                results.append({
                    "table_id": table_id,
                    "filename": output_name,
                    "download_url": f"/download/{job_id}/{output_name}",
                    "bbox": bbox_info
                })

    return jsonify({
        "success": True,
        "job_id": job_id,
        "videos": results,
        "count": len(results)
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


if __name__ == "__main__":
    print("=" * 50)
    print("Video Table Cropper")
    print("=" * 50)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print()
    print("Open http://localhost:8080 in your browser")
    print("=" * 50)
    app.run(debug=True, port=8080)
