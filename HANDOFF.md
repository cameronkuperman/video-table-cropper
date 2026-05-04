# Handoff: Drive item-cap relief + zip-aware preprocessing

## Problem

Shared "shire" Drive hit the **per-shared-drive item cap** (~500K items, separate from storage GB). Symptoms:

1. ~78,721 raw Reolink screenshot triplets sitting in `<site>/unassociated/` — each = 1 folder + 3 frames = 4 items, ~315K items total. Largest single contributor.
2. Cap blocked all new item creation, so the labeling pipeline couldn't generate new crops in `unlabeled/` and labels couldn't propagate.

User confirmed re-cropping isn't needed (crop configs are dialed in). Goal: archive raw triplets into per-batch zips so cold input costs ~1 item per batch, and have the prewarm pipeline read those zips on demand.

## Architecture (final state)

```
<site>/unassociated/        loose triplet folders (live Pi uploads land here)
<site>/unassociated_zips/   one .zip per batch of 500 triplets + .compactor_manifest.jsonl
<site>/processed_raw/       (legacy) raws moved here after Drive-sourced processing
<root>/temp_processing/     debug overlays — no read path; safe to wipe
<root>/unlabeled/           cropped triplets ready to label (per-folder; UI needs random access)
<root>/{clean,dirty,occupied,label_later,discarded}/   labeled output (single shared set, all sources merge here)
```

**Pipeline flow after this work**:

```
unassociated/<triplet>/  ─────────────────────────────────────────┐
                                                                  │
unassociated_zips/<batch>.zip  ─►  download+extract locally  ────►├─►  _materialize_reolink_table_crops
                                                                  │       (Drive-sourced or local-paths)
                                                                  │
                                                                  ▼
                                                        unlabeled/<derived>/{frame_*.jpg}
                                                                  │
                                                                  ▼   user labels in UI
                                                          {clean,dirty,occupied,label_later,discarded}/
```

## What's been built

### A. One-shot compactor: `scripts/compact_unassociated_to_zips.py` (NEW FILE)

Reads `<site>/unassociated/` → batches into 500-triplet zips → uploads to `<site>/unassociated_zips/` → 4-step verify (size + md5 + roundtrip SHA256 + zip-listing match) → permanent-deletes originals.

Modes (`--mode` is required):
- `dry-run` — read-only counts. ZERO Drive writes.
- `test-batch` — wipes `temp_processing/`, archives ONE 50-triplet batch end-to-end with full verification, stops.
- `confirm` — wipes `temp_processing/`, then full run across both Reolink sites.

Important flags:
- `--batch-size N` (default 500)
- `--min-age-minutes N` (default 60) — skips recently-modified triplets to avoid racing the live Pi uploader
- `--download-workers N` / `--delete-workers N` (default 4 each)
- `--limit-batches N` (testing aid)
- `--site key` (repeatable; defaults to all configured Reolink sites)

Resilience features:
- Each parallel thread gets its own `DriveClient` — `googleapiclient`'s SSL stack segfaults under heavy concurrent reuse from a single service object. The thread-local pattern fixes it. See `_thread_local_client()` in the script.
- Per-triplet failure tolerance: a single SSL/network glitch on one triplet doesn't crash the batch. Failed triplets are dropped from the zip and stay on Drive for retry. Batch only aborts if >10% fail.
- Manifest log at `<site>/unassociated_zips/.compactor_manifest.jsonl` — one JSONL line per completed batch with `zip_file_id`, `triplet_ids[]`, `skipped_triplet_ids[]`. Written BEFORE deletes, so a crash mid-run leaves the system idempotent on rerun.
- Drive API retry budget bumped: 10 attempts, base 2.0s exponential = ~17 min max wait per stuck call.
- Local disk: `tempfile.TemporaryDirectory` per batch; auto-cleanup. Peak ~2-6 GB during one batch, zero leftover after.

### B. Phase 3 — zip-aware prewarm in `app.py` (IN-PLACE EDITS)

Constants added near other folder-name constants:
- [app.py:147-149](app.py#L147-L149): `UNASSOCIATED_ZIPS_FOLDER_NAME`, `UNASSOCIATED_ZIPS_MANIFEST_FILE`, `UNASSOCIATED_ZIPS_INNER_MANIFEST`.

Imports:
- [app.py:19](app.py#L19): added `import zipfile`.
- [app.py:24](app.py#L24): added `Iterator` to `typing` import.

Folder lookup (`_reolink_site_folder_ids`):
- [app.py:707-714](app.py#L707-L714) (cached path) and [app.py:736-740](app.py#L736-L740) (cold path): if `<site>/unassociated_zips/` exists on Drive, its folder ID is added to `folder_ids` under the `UNASSOCIATED_ZIPS_FOLDER_NAME` key. NOT auto-created — it's only created by the compactor script. Absence = no zip-batch processing this run.

Cropping (`_materialize_reolink_table_crops`, [app.py:2335](app.py#L2335)):
- Now accepts optional `local_frame_paths: list[Path] | None` and `local_metadata_path: Path | None` kwargs.
- When `local_frame_paths` is provided (zip-sourced triplets), the function skips `client.list_files` + `client.download_file_to_path` and reads frames straight from disk. Everything downstream (YOLO, perspective crop, upload to `unlabeled/`, `perception.json` write) is unchanged.
- For `metadata.json` copy: Drive-sourced uses `_copy_optional_json_file` as before; zip-sourced uses `local_metadata_path.read_bytes()` + `client.upsert_bytes`.

Zip iterator + finalizer ([app.py:2491-2644](app.py#L2491-L2644)):
- `_UnassociatedZipBatch` dataclass: holds zip metadata, list of synthetic `raw_folder` dicts, work_dir, success/failure id sets, and a `_tmp_handle` (TemporaryDirectory) that auto-cleans when the batch finalizes.
- `_iterate_unassociated_zip_batches(client, context, *, max_batches=None)` generator: lists `.zip` files in `<site>/unassociated_zips/` (excluding `.compactor_manifest.jsonl`), oldest-first by `modifiedTime`. For each: download → extract → read inner `MANIFEST.json` → build per-triplet records that look like Drive `raw_folder` dicts but carry `_local_frame_paths` and `_local_metadata_path`. Yields one batch at a time; finalizer runs in `finally` (so cleanup happens even if caller raises).
- `_finalize_zip_batch(client, batch)`: deletes the zip from Drive ONLY if every triplet reached a terminal state AND at least one succeeded AND none failed. Partial-success zips stay on Drive for retry next prewarm. Always cleans up the local temp dir.

Prewarm integration (`_prepare_reolink_unlabeled_queue`, [app.py:2647-2766](app.py#L2647-L2766)):
- After `processed_raw/` cleanup and visibility check, BEFORE the existing `_list_reolink_raw_folders` loop, drain zip batches:
  - For each `zip_batch.triplets`, run the same shape of logic as the Drive-sourced loop (state-already-processed check → camera mapping → missing-table-polygons filter → call `_materialize_reolink_table_crops` with `local_frame_paths=...` → mark success or failure on the batch).
  - Skips the Drive-touching `_stamp_reolink_raw_preprocess_status` and `_move_reolink_raw_to_processed` calls (no Drive original to stamp/move).
  - Still calls `_mark_reolink_raw_folder_processed` (writes to local preprocess state file, keyed by triplet ID — fine for zip-sourced).
- Then falls through to the existing loose-`unassociated/` loop unchanged.

## What's left for the user / next agent

### Step 1 — validate the compactor on real data

```bash
python scripts/compact_unassociated_to_zips.py --mode test-batch
```

Expected outcome:
- Phase 0 says `temp_processing/ is already empty; skipping.` (it was wiped earlier in the prior crashed run).
- Lists ~57,830 eligible triplets in `restaurant-pi-1` (give or take new arrivals).
- Downloads 50 → zips → uploads → 4-step verifies → appends manifest log entry → permanent-deletes 50 originals → stops.
- One zip should appear in `<root>/restaurant-pi-1/unassociated_zips/`.

If that succeeds, manually open one zip in Drive UI and confirm the structure: `<sanitized_name>__<folder_id>/frame_0.jpg`, `frame_1.jpg`, `frame_2.jpg`, plus `MANIFEST.json` at the root.

### Step 2 — full run

```bash
python scripts/compact_unassociated_to_zips.py --mode confirm 2>&1 | tee compactor.log
```

Estimated 8-13 hours overnight. Frees ~315K items.

### Step 3 — restart Flask app and let prewarm drain zips

When the labeling UI is opened for a Reolink site, `_prepare_reolink_unlabeled_queue` will iterate `unassociated_zips/` and start producing crops in `unlabeled/`. Each fully-processed zip gets deleted from Drive automatically.

### Step 4 — diagnose label propagation (independent)

```bash
curl 'http://localhost:5000/api/label_jobs/status?verify=1' | jq
```

Calls [_verify_succeeded_label_jobs](app.py#L1709). Reports counts by status, reopens jobs marked `succeeded` whose folders aren't actually in destination on Drive. Tells us if labels not propagating is cap-related (resolves automatically after Step 2) or a separate worker bug.

## Known gotchas

1. **`googleapiclient` is not thread-safe** under heavy reuse of a single service object — caused a segfault in the first compactor run. Fix is per-thread `DriveClient` instances. Don't undo this without keeping the workaround.

2. **Drive trash on shared drives still counts toward item cap** until trash auto-empties (~30 days). The compactor uses `client.delete_file` (permanent) on purpose — `trash_file` would not free the cap.

3. **The cleanup function `_cleanup_processed_raw_folder` ([app.py:1101](app.py#L1101)) uses `trash_file`, not `delete_file`** — so trashed `processed_raw/` items linger for ~30 days before auto-empty actually frees their cap slots. Phase 3 reduces reliance on `processed_raw/` (zip-sourced triplets skip the move-to-processed step entirely) but for legacy Drive-sourced flows this slow-leak is still present. Worth addressing in a follow-up: switch to `delete_file` since user confirmed no re-cropping.

4. **Pre-existing test failure**: `tests/test_label_sources.py::test_reolink_preprocess_records_existing_drive_folders_in_local_state` fails on clean `main` (verified). Not introduced by Phase 3.

5. **Slow YOLO/torch import in tests**: tests using the `client`/`fake_drive` pytest fixtures load heavy ML deps and can hang for several minutes on first invocation. Targeted prewarm tests that don't use those fixtures run in <1s.

6. **`temp_processing/` was wiped** during the first compactor run (1,096 items, ~115 GB). User confirmed this is correct — those were debug overlay frames with no read path. If a future user wants the overlays back, gate the upload at [processor.py:612-613](processor.py#L612-L613) behind an env flag rather than re-enabling unconditionally.

7. **Race guard**: compactor skips triplets modified within `--min-age-minutes` (default 60). Prevents fighting the live Pi uploader. Don't lower this below ~10 minutes without checking the upload cadence.

## Files touched

- `scripts/compact_unassociated_to_zips.py` — NEW
- `app.py` — edits in `_reolink_site_folder_ids`, `_materialize_reolink_table_crops`, `_prepare_reolink_unlabeled_queue`; new `_UnassociatedZipBatch`, `_iterate_unassociated_zip_batches`, `_finalize_zip_batch`; new constants; added `zipfile` and `Iterator` imports.

## Files NOT touched (intentional)

- `processor.py` — video pipeline analog. Same zip pattern would apply if needed; out of scope for this work.
- `drive_client.py` — no changes needed; existing `upload_file`, `download_file_to_path`, `delete_file`, `ensure_subfolder`, `find_file_by_name` all sufficient.
- The labeling UI / `unlabeled/` flow — kept per-folder. Random per-frame access from the UI doesn't play well with zips; only worth tackling if discard rate is high enough to make `unlabeled/` zipping worthwhile (deferred Phase 4).

## Reference: original plan

Full design doc with rejected alternatives, verification strategy, and decision log is at `~/.claude/plans/wait-one-more-thing-partitioned-crystal.md`.

## Single-line summary

Run `python scripts/compact_unassociated_to_zips.py --mode test-batch`, then `--mode confirm` overnight, then restart the Flask app. Cap relief is automatic; backlog labeling is automatic via Phase 3 prewarm.
