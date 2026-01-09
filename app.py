#!/usr/bin/env python3
"""
Video Table Cropper - Flask Web App

Drag-and-drop interface to crop videos based on JSON bounding boxes.
"""

import io
import json
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
    """Crop a video using ffmpeg."""
    width = x2 - x1
    height = y2 - y1
    crop_filter = f"crop={width}:{height}:{x1}:{y1}"

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vf", crop_filter,
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "aac",  # Re-encode audio to AAC (compatible with MP4)
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
    """Process video with JSON bounding boxes."""
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

        for idx, table in enumerate(tables):
            # Use table id if available, otherwise use index
            table_id = table.get("id", idx)
            bbox = table.get("bbox", {})

            # Get bbox coordinates
            x1 = bbox.get("x1", 0)
            y1 = bbox.get("y1", 0)
            x2 = bbox.get("x2", 0)
            y2 = bbox.get("y2", 0)

            # Only skip if bbox is invalid (zero or negative dimensions)
            if x2 <= x1 or y2 <= y1:
                continue

            output_name = f"{video_name}_table_{table_id:02d}.mp4"
            output_path = job_output / output_name

            success = crop_video(video_path, output_path, x1, y1, x2, y2)

            if success:
                results.append({
                    "table_id": table_id,
                    "filename": output_name,
                    "download_url": f"/download/{job_id}/{output_name}",
                    "bbox": bbox
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
