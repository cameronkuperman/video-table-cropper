#!/usr/bin/env python3
"""
Frame Extraction Tool for Video Table Cropper

Standalone CLI tool to extract frames from videos at regular intervals.
Generates training data for ML models with organized output and metadata.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Constants
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_INTERVAL = 30  # seconds
DEFAULT_QUALITY = 2    # ffmpeg jpeg quality (lower is better, 2-5 recommended)
DEFAULT_FORMAT = "jpg"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "frames"


def check_ffmpeg_installed() -> bool:
    """Check if ffmpeg and ffprobe are installed."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def validate_video(path: Path) -> bool:
    """Check if video file exists and has valid extension."""
    if not path.exists():
        print(f"Error: Video file not found: {path}")
        return False

    if path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        print(f"Error: Unsupported video format: {path.suffix}")
        print(f"Supported formats: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}")
        return False

    return True


def get_video_info(video_path: Path) -> Optional[Dict]:
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
            print(f"Error probing video: {result.stderr}")
            return None

        data = json.loads(result.stdout)
        if not data.get("streams"):
            print("Error: No video stream found")
            return None

        stream = data["streams"][0]

        # Parse frame rate (comes as "30/1" or "30000/1001")
        fps_str = stream.get("r_frame_rate", "30/1")
        fps_parts = fps_str.split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1])

        # Get duration
        duration = float(stream.get("duration", 0))

        return {
            "duration": duration,
            "fps": fps,
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "resolution": f"{stream.get('width')}x{stream.get('height')}"
        }

    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError) as e:
        print(f"Error getting video info: {e}")
        return None


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS or HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}h{minutes:02d}m{secs:02d}s"
    else:
        return f"{minutes:02d}m{secs:02d}s"


def extract_frames(
    video_path: Path,
    output_dir: Path,
    interval: int = 30,
    quality: int = 2,
    format: str = "jpg",
    resume: bool = False,
    verbose: bool = False
) -> Tuple[bool, List[Dict]]:
    """
    Extract frames from video at regular intervals.

    Returns:
        (success: bool, frames_info: List[Dict])
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video info
    video_info = get_video_info(video_path)
    if not video_info:
        return False, []

    duration = video_info["duration"]
    expected_frames = int(duration // interval) + 1

    print(f"\nExtracting frames from: {video_path.name}")
    print(f"Video duration: {format_timestamp(duration)} ({duration:.1f}s)")
    print(f"Frame interval: {interval}s")
    print(f"Expected frames: {expected_frames}")
    print(f"Output directory: {output_dir}")
    print()

    # Check if we should resume
    existing_frames = []
    if resume:
        existing_frames = sorted(output_dir.glob(f"*.{format}"))
        if existing_frames:
            print(f"Found {len(existing_frames)} existing frames, will resume...")

    # Extract frames using ffmpeg fps filter
    temp_pattern = str(output_dir / f"temp_%04d.{format}")
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
        if verbose:
            print(f"Running: {' '.join(extract_cmd)}\n")
            result = subprocess.run(extract_cmd, text=True, timeout=600)
        else:
            result = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"Error extracting frames: {result.stderr if not verbose else ''}")
            return False, []

    except subprocess.TimeoutExpired:
        print("Error: Frame extraction timed out (>10 minutes)")
        return False, []

    # Rename frames to include timestamps
    temp_frames = sorted(output_dir.glob(f"temp_*.{format}"))
    frames_info = []

    for idx, temp_frame in enumerate(temp_frames):
        timestamp_sec = idx * interval
        timestamp_str = format_timestamp(timestamp_sec)

        new_name = f"frame_{idx:04d}_{timestamp_str}.{format}"
        new_path = output_dir / new_name

        temp_frame.rename(new_path)

        frames_info.append({
            "frame_number": idx,
            "filename": new_name,
            "timestamp_seconds": timestamp_sec,
            "timestamp_formatted": format_timestamp(timestamp_sec),
            "file_size_bytes": new_path.stat().st_size
        })

    print(f"✓ Successfully extracted {len(frames_info)} frames")

    return True, frames_info


def generate_metadata(
    video_path: Path,
    video_info: Dict,
    frames_info: List[Dict],
    interval: int,
    quality: int,
    format: str
) -> Dict:
    """Generate metadata JSON for the extracted frames."""
    return {
        "video_info": {
            "source_video": str(video_path.absolute()),
            "video_name": video_path.stem,
            "duration_seconds": video_info["duration"],
            "fps": video_info["fps"],
            "resolution": video_info["resolution"],
            "extraction_date": datetime.now().isoformat()
        },
        "extraction_params": {
            "interval_seconds": interval,
            "format": format,
            "quality": quality,
            "total_frames_extracted": len(frames_info)
        },
        "frames": frames_info
    }


def process_single_video(
    video_path: Path,
    output_dir: Optional[Path] = None,
    interval: int = DEFAULT_INTERVAL,
    quality: int = DEFAULT_QUALITY,
    format: str = DEFAULT_FORMAT,
    resume: bool = False,
    verbose: bool = False
) -> bool:
    """Process a single video file."""
    if not validate_video(video_path):
        return False

    # Set default output directory
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR / f"{video_path.stem}_frames"

    # Get video info for metadata
    video_info = get_video_info(video_path)
    if not video_info:
        return False

    # Extract frames
    success, frames_info = extract_frames(
        video_path, output_dir, interval, quality, format, resume, verbose
    )

    if not success:
        return False

    # Generate metadata
    metadata = generate_metadata(
        video_path, video_info, frames_info, interval, quality, format
    )

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_path}")
    print(f"\nTotal frames: {len(frames_info)}")

    # Calculate storage size
    total_size = sum(f["file_size_bytes"] for f in frames_info)
    total_mb = total_size / (1024 * 1024)
    print(f"Total size: {total_mb:.1f} MB")

    return True


def batch_process(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    interval: int = DEFAULT_INTERVAL,
    quality: int = DEFAULT_QUALITY,
    format: str = DEFAULT_FORMAT,
    resume: bool = False,
    verbose: bool = False
) -> None:
    """Process all videos in a directory."""
    if not input_dir.is_dir():
        print(f"Error: Not a directory: {input_dir}")
        return

    # Find all video files
    video_files = []
    for ext in ALLOWED_VIDEO_EXTENSIONS:
        video_files.extend(input_dir.glob(f"*{ext}"))

    if not video_files:
        print(f"No video files found in: {input_dir}")
        return

    print(f"Found {len(video_files)} video(s) to process\n")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    for idx, video_path in enumerate(sorted(video_files), 1):
        print(f"\n[{idx}/{len(video_files)}] Processing: {video_path.name}")
        print("-" * 60)

        # Set output directory for this video
        if output_dir:
            video_output = output_dir / f"{video_path.stem}_frames"
        else:
            video_output = DEFAULT_OUTPUT_DIR / f"{video_path.stem}_frames"

        success = process_single_video(
            video_path, video_output, interval, quality, format, resume, verbose
        )

        if success:
            success_count += 1
        else:
            fail_count += 1
            print(f"✗ Failed to process: {video_path.name}")

    print("\n" + "=" * 60)
    print(f"Batch processing complete!")
    print(f"Success: {success_count}/{len(video_files)}")
    if fail_count > 0:
        print(f"Failed: {fail_count}/{len(video_files)}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract frames from videos at regular intervals for ML training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract frames every 30 seconds (default)
  python3 extract_frames.py video.mp4

  # Extract frames every 60 seconds
  python3 extract_frames.py video.mp4 -i 60

  # Specify output directory
  python3 extract_frames.py video.mp4 -o training_data/

  # Batch process all videos in a directory
  python3 extract_frames.py --batch videos/ -o frames/

  # Resume interrupted extraction
  python3 extract_frames.py video.mp4 --resume
        """
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input video file or directory (with --batch)"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        help="Output directory (default: frames/<video_name>_frames/)"
    )

    parser.add_argument(
        "-i", "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Frame extraction interval in seconds (default: {DEFAULT_INTERVAL})"
    )

    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=DEFAULT_QUALITY,
        help=f"JPEG quality 1-31, lower is better (default: {DEFAULT_QUALITY})"
    )

    parser.add_argument(
        "-f", "--format",
        choices=["jpg", "png"],
        default=DEFAULT_FORMAT,
        help=f"Output image format (default: {DEFAULT_FORMAT})"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all videos in input directory"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing frames and resume extraction"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show ffmpeg output"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    # Check ffmpeg installation
    if not check_ffmpeg_installed():
        print("Error: ffmpeg and ffprobe must be installed")
        print("Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)")
        return 1

    # Batch processing mode
    if args.batch:
        batch_process(
            args.input,
            args.output_dir,
            args.interval,
            args.quality,
            args.format,
            args.resume,
            args.verbose
        )
        return 0

    # Single video mode
    success = process_single_video(
        args.input,
        args.output_dir,
        args.interval,
        args.quality,
        args.format,
        args.resume,
        args.verbose
    )

    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
