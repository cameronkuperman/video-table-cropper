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

The app supports two JSON formats:

### Format 1: Axis-aligned bounding boxes

```json
{
  "video_name": "my_video",
  "tables": [
    {
      "id": 0,
      "bbox": {
        "x1": 100,
        "y1": 200,
        "x2": 500,
        "y2": 600
      },
      "saved": true
    }
  ]
}
```

### Format 2: Rotated bounding boxes

For tables detected at an angle:

```json
{
  "video_name": "my_video",
  "frame_width": 1280,
  "frame_height": 720,
  "tables": [
    {
      "id": 0,
      "rotated_bbox": {
        "center": [500, 300],
        "size": [200, 150],
        "angle": -45.0,
        "corners": [
          [400, 200],
          [600, 200],
          [600, 400],
          [400, 400]
        ]
      },
      "saved": true
    }
  ]
}
```

**Rotated bbox fields:**
- `center` - [x, y] center point of the rotated rectangle
- `size` - [width, height] of the rectangle (output dimensions)
- `angle` - Rotation angle in degrees
- `corners` - Four [x, y] corner points defining the rotated rectangle

**Optional fields:**
- `id` - Table identifier (used in output filename)
- `saved` - Set to `false` to skip this table
- `skip_reason` - Set to skip this table

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
