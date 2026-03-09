# AutoLabeler

Label table images from Drive videos. Two commands, local-only, no database.

## Drive folder structure

One root folder on Google Drive. All subfolders are auto-created:

```
<your root folder>/
├── raw_videos/         ← drop source videos here
├── temp_processing/    ← 3-frame triplets with table overlays drawn on
├── unlabeled/          ← cropped table triplets waiting to be labeled
├── clean/              ← labeled output
├── dirty/              ← labeled output
└── occupied/           ← labeled output
```

## Setup

### 1. Python + ffmpeg

```bash
brew install python@3.12 ffmpeg   # macOS
# or: sudo apt install python3 ffmpeg
```

### 2. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Create `.env`

```bash
cp .env.example .env
# edit .env and fill in your values
```

You need:
- A Google Cloud service account with Drive API enabled
- The service account email shared on your root Drive folder
- The root folder ID (from URL: `drive.google.com/drive/folders/<ID>`)

## Usage

### Process videos

```bash
python main.py --process
```

Downloads videos from `raw_videos/`, extracts frames, draws polygon overlays,
uploads 3-frame triplets to `temp_processing/`, crops each table and uploads
to `unlabeled/`.

### Label images

```bash
python main.py --label
# open http://localhost:8080
```

Shows 3 images per unlabeled folder. Click **Occupied / Dirty / Clean**
(or press `1` / `2` / `3`). The folder moves to the matching Drive folder instantly.
Press `→` or `Space` to skip.

## Table geometry

Polygons are loaded from `approved_table_rectangles.json`. Camera is detected
from the video filename (e.g. `ipc3_...mp4` → IPC3 → matched by `camera_number: 3`).
