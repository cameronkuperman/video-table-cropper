# video-table-cropper

Flask review UI plus an offline Drive worker for grouped video review.

The current launch path is:
- `web`: Flask app on Railway
- `db`: Postgres on Railway
- `worker`: GPU worker on Runpod
- `storage`: Google Drive for raw inputs, review bundles, and exported artifacts

For this phase, `/video-review` is the production workflow. Legacy Drive/image labeling and legacy local video tools remain in the repo for local use, but they are hidden automatically when `APP_ENV=production` and `ENABLE_LEGACY_ROUTES=false`.

## What Changed

Production state is no longer expected to live in local SQLite/files only.

The app now supports:
- Postgres-backed shared review queue state via `DATABASE_URL`
- Postgres-backed worker heartbeat and processed-video markers
- SSE worker-status streaming at `/api/video-review/worker-status/stream`
- ephemeral preview caching under `/tmp/drive_cache` in production
- static per-camera table metadata loaded from `approved_table_rectangles.json` when present, otherwise `approved_tables.json`
- service-account credentials from:
  - `DRIVE_SERVICE_ACCOUNT_JSON`
  - `DRIVE_SERVICE_ACCOUNT_JSON_B64`
  - `DRIVE_SERVICE_ACCOUNT_JSON_PATH`

The worker now uses `approved_table_rectangles.json` for tables when it exists, otherwise it falls back to `approved_tables.json`. SAM is only used for people. It still uploads review bundles to Drive, but it also upserts queue rows into Postgres so the UI does not need to rescan Drive for every session.

## Local Dev

### Prereqs

- Python 3.12 recommended
- `ffmpeg`
- Drive service account with access to the Shared Drive/project folders

### Install `ffmpeg`

macOS:

```bash
brew install ffmpeg
```

Ubuntu/Debian:

```bash
sudo apt install ffmpeg
```

### Run the web app

```bash
bash run_local_app.sh
```

### Run the worker

```bash
bash run_worker.sh
```

The runner scripts reuse the existing virtualenv unless dependencies changed. Set `BOOTSTRAP_DEPS=1` to force a reinstall.

## Core Environment Variables

### Shared

```bash
export DRIVE_PROJECT_ROOT_FOLDER_ID=...
export DRIVE_VIDEO_SOURCE_ROOT_ID=...
export DRIVE_REVIEW_QUEUE_ROOT_ID=...
export DRIVE_OUTPUT_TEMPORAL_STATE_ROOT_ID=...
export DRIVE_OUTPUT_DIRTY_CLEAN_SURFACE_ROOT_ID=...
export DRIVE_OUTPUT_OCCUPANCY_MLP_ROOT_ID=...
export DRIVE_OUTPUT_SAM_AUDIT_ROOT_ID=...
```

Static table metadata defaults to [`approved_table_rectangles.json`](approved_table_rectangles.json) when present in the repo root, otherwise [`approved_tables.json`](approved_tables.json). Override it with:

```bash
export APPROVED_TABLES_JSON_PATH=/absolute/path/to/approved_tables.json
```

The downstream geometry handoff for `tight_rect` and `zone_rect` is documented in [`STATIC_TABLE_GEOMETRY_CONTRACT.md`](STATIC_TABLE_GEOMETRY_CONTRACT.md).

### Drive credentials

Use one of these, in this priority order:

```bash
export DRIVE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
```

or

```bash
export DRIVE_SERVICE_ACCOUNT_JSON_B64=...
```

or

```bash
export DRIVE_SERVICE_ACCOUNT_JSON_PATH=/absolute/path/to/service-account.json
```

### Web

```bash
export APP_ENV=production
export ENABLE_LEGACY_ROUTES=false
export VIDEO_REVIEW_BATCH_LIMIT_DEFAULT=60
export X_ROBOTS_TAG="noindex, nofollow"
export DATABASE_URL=postgresql://...
```

### Worker

```bash
export DATABASE_URL=postgresql://...
export PROCESSOR_CONTINUOUS=true
export PROCESSOR_POLL_SECONDS=30
export PROCESSOR_MAX_PENDING_SAMPLES=500
export PROCESSOR_RESUME_PENDING_SAMPLES=250
export PROCESSOR_TRASH_SOURCE_VIDEOS=true
export PROCESSOR_CLEANUP_FRAMES=true
export PROCESSOR_CLEANUP_REVIEW_CACHE=true
export PROCESSOR_CLEANUP_LOCAL_VIDEO_WHEN_SOURCE_TRASHED=true
export PROCESSOR_FRAME_INTERVAL=10
export SAM3_CHECKPOINT_PATH=...
export SAM3_CONFIG_NAME=...
```

## Postgres Migration

Apply [`migrations/001_video_review_pg.sql`](migrations/001_video_review_pg.sql) once to the production database before starting the app/worker with `DATABASE_URL`.

Example:

```bash
psql "$DATABASE_URL" -f migrations/001_video_review_pg.sql
```

Tables created:
- `video_review_sessions`
- `video_review_items`
- `video_review_actions`
- `worker_status`
- `processed_videos`

## Production Behavior

### Web

- `/video-review` auto-starts a review session if a review root is configured
- batch retrieval comes from Postgres when `DATABASE_URL` is set
- preview/sample files are fetched from Drive on demand and cached locally
- `/healthz` returns `200` only when:
  - the app can respond
  - database healthcheck passes when enabled
  - Drive credentials are configured
  - a review root is configured

### Worker

- scans Drive source folders
- skips processed videos using `processed_videos`
- loads static tables from `approved_tables.json`
- runs SAM only for `person`
- uploads review bundles to `review_queue/pending/...`
- upserts shared queue rows into `video_review_items`
- writes heartbeat/status into `worker_status`
- cleans transient local `frames/` and `review/` folders after processing
- keeps or removes the downloaded local source video based on whether the Drive source was trashed and `PROCESSOR_CLEANUP_LOCAL_VIDEO_WHEN_SOURCE_TRASHED`

## Containers

### Web image

[`Dockerfile.web`](Dockerfile.web)

- base: `python:3.12-slim`
- installs `ffmpeg`
- installs `requirements.web.txt`
- runs `gunicorn app:app`

### Worker image

[`Dockerfile.worker`](Dockerfile.worker)

- base: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`
- installs `ffmpeg`
- installs `requirements.worker.txt`
- runs the worker in continuous mode

## Railway + Runpod Deployment

### Railway

1. Create a Railway project.
2. Add PostgreSQL.
3. Deploy the web service with [`Dockerfile.web`](Dockerfile.web).
4. Set `/healthz` as the healthcheck path.
5. Set env vars:
   - `DATABASE_URL`
   - Drive credentials env
   - Drive root IDs
   - `APP_ENV=production`
   - `ENABLE_LEGACY_ROUTES=false`
   - `VIDEO_REVIEW_BATCH_LIMIT_DEFAULT=60`
   - `X_ROBOTS_TAG=noindex, nofollow`
6. Do not attach a volume.
7. Run the SQL migration once.

### Runpod

1. Create one always-on GPU pod.
2. Use [`Dockerfile.worker`](Dockerfile.worker).
3. Set the same `DATABASE_URL`, Drive credentials, and Drive root env vars.
4. Set the SAM checkpoint/config env vars.
5. Keep the pod warm.

Recommended starting GPU tier:
- RTX 4090 24 GB
- RTX A5000 24 GB
- RTX A6000 48 GB

## Verification

Basic checks:

```bash
python3 -m py_compile app.py video_dataset_worker.py db.py video_review_store_pg.py worker_state_store_pg.py
bash -n run_local_app.sh
bash -n run_worker.sh
python3 -m unittest discover -s tests
```

Deployment smoke tests:
- `GET /healthz`
- `GET /api/video-review/worker-status`
- worker heartbeat row appears in `worker_status`
- processing one video inserts `pending` rows into `video_review_items`
- `/video-review` loads cards without manual Drive reindex

## Current Limitations

- Drive is still the artifact store for this phase; GCS/object storage is a later optimization.
- There is no auth layer in this launch configuration.
- Legacy pages still exist in the repo, but production is intentionally centered on `/video-review`.
