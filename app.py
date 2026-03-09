"""
--label mode: Flask UI that reads unlabeled/ subfolders from Drive,
shows 3 images per folder, and moves the folder on Drive when labeled.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from drive_client import DriveClient, DriveClientError
from env_loader import load_local_env

load_local_env()

app = Flask(__name__)

CACHE_DIR = Path(__file__).parent / "label_cache"

# ── Drive client (lazy, one instance) ────────────────────────────────────

_drive_client: DriveClient | None = None


def get_client() -> DriveClient:
    global _drive_client
    if _drive_client is None:
        _drive_client = DriveClient()
    return _drive_client


# ── Folder ID helpers ─────────────────────────────────────────────────────

def _root_id() -> str:
    root = os.environ.get("DRIVE_PROJECT_ROOT_FOLDER_ID", "").strip()
    if not root:
        raise RuntimeError("DRIVE_PROJECT_ROOT_FOLDER_ID is not set in .env")
    return root


def _folder_ids() -> dict[str, str]:
    """Return {name: folder_id} for all 6 subfolders, creating if needed."""
    client = get_client()
    root = _root_id()
    names = ["raw_videos", "temp_processing", "unlabeled", "clean", "dirty", "occupied"]
    return {name: client.ensure_subfolder(root, name) for name in names}


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("label.html")


@app.route("/api/folders")
def api_folders():
    """Return list of unlabeled subfolders with their Drive file IDs."""
    try:
        client = get_client()
        folder_ids = _folder_ids()
        unlabeled_id = folder_ids["unlabeled"]

        subfolders = client.list_folders(unlabeled_id)
        result = []
        for folder in subfolders:
            files = client.list_files(folder["id"])
            file_map = {f["name"]: f["id"] for f in files}
            result.append({
                "folder_id": folder["id"],
                "folder_name": folder["name"],
                "parent_id": unlabeled_id,
                "frames": {
                    "frame_0": file_map.get("frame_0.jpg"),
                    "frame_1": file_map.get("frame_1.jpg"),
                    "frame_2": file_map.get("frame_2.jpg"),
                },
            })
        return jsonify({"folders": result})
    except (DriveClientError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/preview/<file_id>")
def api_preview(file_id: str):
    """Serve a Drive image file, caching it locally."""
    cache_path = CACHE_DIR / f"{file_id}.jpg"
    if not cache_path.exists():
        try:
            client = get_client()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            client.download_file_to_path(file_id, cache_path)
        except DriveClientError as e:
            abort(404, description=str(e))
    return send_file(cache_path, mimetype="image/jpeg")


@app.route("/api/label", methods=["POST"])
def api_label():
    """Move a folder from unlabeled/ to clean/, dirty/, or occupied/."""
    data = request.get_json(force=True)
    folder_id = data.get("folder_id", "").strip()
    parent_id = data.get("parent_id", "").strip()
    label = data.get("label", "").strip().lower()

    if not folder_id or not parent_id:
        return jsonify({"error": "folder_id and parent_id required"}), 400
    if label not in ("clean", "dirty", "occupied"):
        return jsonify({"error": "label must be clean, dirty, or occupied"}), 400

    try:
        client = get_client()
        folder_ids = _folder_ids()
        dest_id = folder_ids[label]
        client.move_file(folder_id, new_parent_id=dest_id, remove_parent_id=parent_id)
        return jsonify({"ok": True, "moved_to": label})
    except (DriveClientError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    """Return counts of folders in each category."""
    try:
        client = get_client()
        folder_ids = _folder_ids()
        stats = {}
        for name in ("unlabeled", "clean", "dirty", "occupied"):
            folders = client.list_folders(folder_ids[name])
            stats[name] = len(folders)
        return jsonify(stats)
    except (DriveClientError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500


def run_label_ui(port: int = 8080) -> None:
    print(f"Starting label UI at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
