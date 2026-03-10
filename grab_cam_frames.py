"""
Grab one representative frame per unique camera from raw_videos/ on Drive.
Output: local cam_data/ folder with IPC<N>.jpg per camera.

Usage:
    python grab_cam_frames.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from drive_client import DriveClient
from env_loader import load_local_env
from processor import detect_camera_from_filename, _find_videos_recursive

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
OUT_DIR = Path(__file__).parent / "cam_data"


def extract_single_frame(video_path: Path, out_path: Path) -> bool:
    """Extract the first frame of a video using ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-frames:v", "1", "-q:v", "2", "-y", str(out_path)],
        capture_output=True,
    )
    return result.returncode == 0 and out_path.exists()


def main() -> None:
    load_local_env()

    import os
    root_id = os.environ.get("DRIVE_PROJECT_ROOT_FOLDER_ID", "").strip()
    if not root_id:
        raise SystemExit("DRIVE_PROJECT_ROOT_FOLDER_ID not set in .env")

    client = DriveClient()
    raw_videos_id = client.ensure_subfolder(root_id, "raw_videos")

    print("Scanning raw_videos/ ...")
    all_videos = _find_videos_recursive(client, raw_videos_id)
    print(f"Found {len(all_videos)} video file(s).")

    OUT_DIR.mkdir(exist_ok=True)
    seen_cameras: set[int] = set()

    with tempfile.TemporaryDirectory(prefix="cam_grab_") as tmpdir:
        tmp = Path(tmpdir)
        for video in all_videos:
            ipc = detect_camera_from_filename(video["name"])
            if ipc is None:
                print(f"  Skip (no IPC detected): {video['name']}")
                continue
            if ipc in seen_cameras:
                continue  # already have a frame for this camera

            out_path = OUT_DIR / f"IPC{ipc}.jpg"
            print(f"  IPC{ipc} ← {video['name']} ... ", end="", flush=True)

            video_path = tmp / video["name"]
            client.download_file_to_path(video["id"], video_path)

            if extract_single_frame(video_path, out_path):
                print("ok")
                seen_cameras.add(ipc)
            else:
                print("FAILED (ffmpeg error)")

            video_path.unlink(missing_ok=True)

    print(f"\nDone. {len(seen_cameras)} camera(s) saved to {OUT_DIR}/")
    for ipc in sorted(seen_cameras):
        print(f"  {OUT_DIR / f'IPC{ipc}.jpg'}")


if __name__ == "__main__":
    main()
