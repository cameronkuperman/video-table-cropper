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
├── occupied/           ← labeled output
├── label_later/        ← labeled output
└── discarded/          ← rejected output
```

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
├── label_later/        ← labeled output
└── discarded/          ← rejected output
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

### Deploy online

This repo is set up for Railway with a Docker image that includes `ffmpeg`,
Gunicorn, and YOLO dependencies.

Create a Railway web service from the repo and set:

```text
DRIVE_PROJECT_ROOT_FOLDER_ID=...
DRIVE_SERVICE_ACCOUNT_JSON_B64=...
AUTH_REQUIRED=1
LABELER_PASSWORD=...
FLASK_SECRET_KEY=...
LABEL_CACHE_DIR=/data/label_cache
LABEL_CACHE_MAX_MB=20000
LABEL_CACHE_TTL_HOURS=336
PREPROCESS_STATE_DIR=/data/autolabeler
LABEL_JOB_UNDO_SECONDS=30
LABEL_JOB_PROCESSING_STALE_SECONDS=300
LABEL_JOB_MIN_INTERVAL_SECONDS=2.0
LABEL_JOB_RATE_LIMIT_COOLDOWN_SECONDS=120
LABEL_REOLINK_PREWARM_TARGET=5000
LABEL_PREWARM_FOLDER_COUNT=60
LABEL_READY_SCAN_MAX=180
```

Mount a Railway volume on the web service at `/data` when using
`LABEL_CACHE_DIR=/data/label_cache`. Label history, pending Drive label moves,
and Reolink preprocess state live under `PREPROCESS_STATE_DIR`; on Railway that
should be `/data/autolabeler`.

To fill the persistent image cache after deploys or new queue generation:

```bash
curl -X POST https://YOUR_DEPLOYMENT/api/cache/warm
curl https://YOUR_DEPLOYMENT/api/cache/warm/status
curl https://YOUR_DEPLOYMENT/api/cache/status
curl https://YOUR_DEPLOYMENT/api/cache/status?scan=1
curl https://YOUR_DEPLOYMENT/api/label/jobs/status
```

The warmer walks the current unlabeled queues, downloads each missing Drive
frame once, writes `{file_id}.jpg` plus `{file_id}.thumb.jpg` into
`LABEL_CACHE_DIR`, and skips files already present on the volume. If it needs
to be stopped, call `POST /api/cache/warm/cancel`.

Label clicks are recorded to the volume first, held briefly for undo/relabel,
then pushed to Drive by a background worker. Use `/api/label/jobs/status` to
confirm pending and delayed jobs are draining after a deploy or browser close.

The default web start command is:

```bash
gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 2 --threads 4 --timeout 180 app:app
```

Create a second Railway service or job from the same repo for finite preprocessing:

```bash
python main.py --preprocess-until-empty --sources all
```

For Reolink preprocessing persistence, mount a Railway volume at `/data` and set:

```text
PREPROCESS_STATE_DIR=/data/autolabeler
PROCESSED_RAW_RETENTION_DAYS=14
```

That command processes unmarked videos in `raw_videos/`, materializes missing
Reolink crops from every configured site, stamps raw videos as `complete`,
`skipped`, or `error`, prints a JSON summary, then exits. Real video batches need
enough ephemeral disk for source downloads and extracted frames; avoid tiny free
storage for large uploads.

Reolink raw triplets are recorded in that local state directory after successful
crop materialization, so later job runs skip them without extra Drive metadata
writes. Existing Drive destination folders are still checked as a fallback if
the local state file is missing. Successfully processed raw inputs are moved to
`processed_raw/` and trashed after `PROCESSED_RAW_RETENTION_DAYS`.

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
Press `→` or `Space` to discard the current triplet. Discarded triplets are moved
to `discarded/` and are treated as already handled by future preprocessing.

## Table geometry

Polygons are loaded from `approved_table_rectangles.json`. Camera is detected
from the video filename (e.g. `ipc3_...mp4` → IPC3 → matched by `camera_number: 3`).
