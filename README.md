# AutoLabeler

Label table images from Google Drive. Two commands, local-only, no database.

## Drive folder structure

### Video mode

One project root folder on Google Drive. The video pipeline uses these subfolders:

```
<your root folder>/
├── raw_videos/         ← drop source videos here
├── temp_processing/    ← 3-frame triplets with table overlays drawn on
├── unlabeled/          ← cropped table triplets waiting to be labeled
├── clean/              ← labeled output
├── dirty/              ← labeled output
└── occupied/           ← labeled output
```

`label_later/` is also auto-created for the label UI.

### Reolink mode

Reolink sites are separate sibling folders that are configured directly in `app.py`.
Each site should look like:

```
<site root>/
├── unassociated/       ← raw screenshot triplets grouped by channel/time
├── crop_configs/       ← permanent manual crop configs (Matthews only)
├── unlabeled/          ← generated per-table crops waiting to be labeled
├── clean/              ← labeled output
├── dirty/              ← labeled output
├── occupied/           ← labeled output
└── label_later/        ← labeled output
```

Each raw triplet folder in `unassociated/` should contain `frame_0.jpg`, `frame_1.jpg`,
and `frame_2.jpg`. The labeler maps channel names like `CH-CH03` to the matching camera
geometry (`IPC3`), runs the same YOLO/perception + table-crop flow used by the video
pipeline, and materializes per-table folders into `unlabeled/`.

`reolink-matthews-01` is now special-cased:
- it does **not** use the general `CH-CHNN -> IPCN` mapping
- it requires a saved `crop_configs/CH-CHNN.json` per channel before queue generation
- those saved 4-point polygons drive cropped `frame_0.jpg`, `frame_1.jpg`, `frame_2.jpg`,
  plus `perception.json`, exactly like the original video pipeline

`restaurant-pi-1` stays on the general IPC-mapping behavior.

If `metadata.json` is present in the raw triplet folder it is copied into each generated
per-table folder and preserved when that folder is later moved to a label destination.

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
(or press `1` / `2` / `3`). Press `4` for **Label Later**.

The UI now supports two source types:
- `Video`: reads from the project root `unlabeled/`
- `Reolink`: generates per-table crops from a selected site `unassociated/`, then labels
  the derived folders in that site `unlabeled/`

For Matthews setup, open:

```text
http://localhost:8080/crop-editor?site=reolink-matthews-01
```

Pick a channel, open its full-frame reference image, click four corners per table, and save.
If any Matthews channel in `unassociated/` is missing a saved config, the Reolink queue will
stop and link you to the crop editor first.

In both modes, labeling moves the entire triplet folder to the matching Drive folder instantly.
Press `→` or `Space` to skip.

## Table geometry

Polygons are loaded from `approved_table_rectangles.json`. Camera is detected
from the video filename (e.g. `ipc3_...mp4` → IPC3 → matched by `camera_number: 3`).
