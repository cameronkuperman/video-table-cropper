# Video Table Cropper

A web app with two tools:
1. **Table Cropper** - Crop videos based on bounding box coordinates from JSON files
2. **Frame Extraction** - Extract frames from videos for ML training datasets

**Use cases:**
- Crop videos to isolate detected table regions
- Generate training data from restaurant CCTV footage
- Extract frames at regular intervals for classification tasks

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

---

## Tool 1: Integrated Crop & Label Workflow (RECOMMENDED)

The main workflow combines table cropping with automatic frame extraction and an interactive labeling interface.

### Complete Workflow

1. **Upload** video + JSON (table bounding boxes) on the main page
2. Click **"Crop Videos"** → System automatically:
   - Crops each table from the video
   - Extracts frames every 30 seconds from each cropped video
3. Click **"Label Frames"** → Opens interactive labeling page
4. **Drag frames** into categories:
   - **Clean** - Empty, clean table
   - **Occupied** - People sitting at table
   - **Dirty** - Table needs cleaning (dishes, mess)
5. Click **"Download Labeled Frames"** → Get ZIP with:
   ```
   labeled_frames.zip
   ├── clean/
   │   ├── frame_0001_00m00s.jpg
   │   └── ...
   ├── occupied/
   │   └── ...
   ├── dirty/
   │   └── ...
   └── labels.json
   ```

**Perfect for**: Building ML training datasets for restaurant table classification

---

## Tool 2: Table Cropper (Standalone)

If you only need cropped videos without labeling:

1. **Drag & drop** your video file onto the left zone
2. **Drag & drop** your JSON file(s) onto the right zone
3. Click **"Crop Videos"**
4. **Download** each cropped video

---

## Tool 3: Frame Extraction (Standalone CLI)

For batch processing or custom workflows, use the standalone CLI tool to extract frames from videos.

### Web Interface

1. Go to **http://localhost:8080/frames**
2. **Drag & drop** your video file
3. Set your **extraction settings**:
   - **Frame Interval**: How often to extract (default: 30 seconds)
   - **JPEG Quality**: 1-31, lower is better (default: 2 = high quality)
4. Click **"Extract Frames"**
5. **Download** all frames as a ZIP file

**Frame output:**
- Each frame is named: `frame_0000_00m00s.jpg` (frame number + timestamp)
- Includes `metadata.json` with video info and frame details
- Organized in folders by video name

### CLI Tool

For batch processing or automation, use the standalone CLI script:

```bash
# Extract frames every 30 seconds (default)
python3 extract_frames.py video.mp4

# Extract frames every 60 seconds
python3 extract_frames.py video.mp4 -i 60

# Specify output directory
python3 extract_frames.py video.mp4 -o training_data/

# Batch process all videos in a folder
python3 extract_frames.py --batch videos/ -o frames/

# Resume interrupted extraction
python3 extract_frames.py video.mp4 --resume

# Show help
python3 extract_frames.py --help
```

**CLI Options:**
- `-i, --interval` - Frame extraction interval in seconds (default: 30)
- `-q, --quality` - JPEG quality 1-31, lower is better (default: 2)
- `-f, --format` - Output format: jpg or png (default: jpg)
- `-o, --output-dir` - Output directory
- `--batch` - Process all videos in a directory
- `--resume` - Skip existing frames and resume
- `-v, --verbose` - Show ffmpeg output

**Output structure:**
```
frames/
└── video_name_frames/
    ├── frame_0000_00m00s.jpg
    ├── frame_0001_00m30s.jpg
    ├── frame_0002_01m00s.jpg
    └── metadata.json
```

**Why 30 seconds?**
- For restaurant CCTV, tables change state slowly (empty → occupied → food arrives → cleared)
- 30-second intervals capture state transitions without redundant frames
- ~120 frames per hour of video
- Adjust with `-i` based on your needs (faster scenes = shorter interval)

---

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
