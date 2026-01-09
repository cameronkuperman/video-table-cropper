# Video Table Cropper

A simple web app to crop videos based on bounding box coordinates from JSON files.

**Use case:** You have a video and JSON files containing detected table regions (with x1, y1, x2, y2 coordinates). This tool creates separate cropped videos for each table region.

![Demo](https://img.shields.io/badge/status-ready-green)

## Installation

### 1. Install ffmpeg (required for video processing)

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html

### 2. Install Python dependencies

```bash
pip3 install flask
```

## Usage

### Start the server

```bash
cd video-table-cropper
python3 app.py
```

Open **http://localhost:8080** in your browser.

### How to use

1. **Drag & drop** your video file (MP4, MOV, AVI, MKV, WebM) onto the left zone
2. **Drag & drop** your JSON file(s) onto the right zone
3. Click **"Crop Videos"**
4. **Download** each cropped video

## JSON Format

Your JSON file should have this structure:

```json
{
  "video_name": "my_video",
  "tables": [
    {
      "id": 0,
      "confidence": 0.95,
      "bbox": {
        "x1": 100,
        "y1": 200,
        "x2": 500,
        "y2": 600
      },
      "saved": true
    },
    {
      "id": 1,
      "bbox": {
        "x1": 50,
        "y1": 50,
        "x2": 300,
        "y2": 250
      },
      "saved": true
    }
  ]
}
```

**Required fields per table:**
- `bbox.x1`, `bbox.y1` - Top-left corner (pixels)
- `bbox.x2`, `bbox.y2` - Bottom-right corner (pixels)

**Optional fields:**
- `id` - Table identifier (used in output filename)
- `saved` - Set to `false` to skip this table
- `skip_reason` - Set to `"too_small"` to skip

## Output

Cropped videos are named: `{video_name}_table_{id}.mp4`

Example: `my_video_table_00.mp4`, `my_video_table_01.mp4`, etc.

## Troubleshooting

**"Access to localhost was denied" (port 5000)**
- macOS uses port 5000 for AirPlay. This app uses port 8080 instead.

**"ffmpeg not found"**
- Make sure ffmpeg is installed: `brew install ffmpeg`

**Video processing fails**
- Check that your video file isn't corrupted
- Ensure bounding box coordinates are within the video dimensions
