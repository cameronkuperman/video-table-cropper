#!/usr/bin/env python3
"""Pre-build contact sheets (and per-frame thumbs) for all unlabeled folders.

Converts the cold Drive-download path into warm local-disk reads so the labeler
never waits on Google Drive. Building each sheet also caches the 3 per-frame
thumbnails (used by the click-to-zoom fallback). Idempotent: folders whose sheet
already exists on disk are skipped.

IMPORTANT: run this ON the web service (same process env) so it writes to the
volume the labeler reads from — CACHE_DIR resolves to /data/label_cache on
Railway. Confirm the volume has room and LABEL_CACHE_MAX_MB is large enough that
serving won't LRU-evict the backfilled sheets.

Usage:
    python scripts/backfill_contact_sheets.py --source reolink --workers 48
    python scripts/backfill_contact_sheets.py --source reolink --limit 500   # smoke test
    python scripts/backfill_contact_sheets.py --dry-run                      # count only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

os.environ.setdefault("LABEL_READY_MAINTAINER_ON_STARTUP", "0")
os.environ.setdefault("LABEL_DRAIN_ON_STARTUP", "0")
os.environ.setdefault("LABEL_TIMING_LOGS", "0")
# Ensure sheets are produced regardless of the serving toggle.
os.environ.setdefault("LABEL_CONTACT_SHEETS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import app as label_app  # noqa: E402
from drive_client import DriveClient  # noqa: E402
from drive_client import DriveClientError  # noqa: E402


def _resolve_contexts(client: DriveClient, source: str, site: str | None):
    if source == label_app.REOLINK_SOURCE and site in (None, "all"):
        return [
            label_app._resolve_queue_context(client, source, s.site_key)
            for s in label_app.REOLINK_SITES
        ]
    return [label_app._resolve_queue_context(client, source, site)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="reolink", help="reolink or video")
    parser.add_argument("--site", default="all", help="reolink site key, or 'all'")
    parser.add_argument("--workers", type=int, default=48, help="thread pool size")
    parser.add_argument("--limit", type=int, default=0, help="cap folders processed (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="list/count only, build nothing")
    args = parser.parse_args()

    client = DriveClient()
    contexts = _resolve_contexts(client, args.source, args.site)
    cache_dir = label_app._ensure_cache_dir()
    print(f"CACHE_DIR = {cache_dir}")
    print(f"LABEL_CACHE_MAX_MB = {label_app.CACHE_MAX_MB}")
    print(f"Resolving folders from {len(contexts)} context(s)...")

    # (context, folder) work items, skipping folders whose sheet already exists.
    work: list[tuple] = []
    listed = 0
    for context in contexts:
        for folder in label_app._list_source_subfolders(client, context):
            listed += 1
            if args.limit and len(work) >= args.limit:
                break
            sheet_path = label_app._contact_sheet_path_for_folder(folder["id"])
            if sheet_path.exists():
                continue
            work.append((context, folder))
        if args.limit and len(work) >= args.limit:
            break

    print(f"Listed {listed} folders; {len(work)} need contact sheets "
          f"(others already cached).")
    if args.dry_run or not work:
        est_mb = len(work) * 0.035  # ~35 KB/sheet observed
        print(f"Dry run: would build {len(work)} sheets (~{est_mb:.0f} MB).")
        return 0

    counters = {"built": 0, "skipped": 0, "errors": 0}
    lock = Lock()
    started = time.perf_counter()

    def task(item):
        context, folder = item
        frames_payload = label_app._hydrate_folder(label_app.warm_client(), context, folder)
        if not frames_payload:
            return "error"
        file_ids = [fid for fid in (frames_payload.get("frames") or {}).values() if fid]
        if not file_ids:
            return "error"
        _, cache_hit, _, _ = label_app._ensure_contact_sheet_for_folder(
            folder["id"], file_ids, label_app.warm_client()
        )
        return "skipped" if cache_hit else "built"

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(task, item) for item in work]
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                result = fut.result()
            except (DriveClientError, Exception):  # noqa: BLE001
                result = "error"
            with lock:
                counters[result] = counters.get(result, 0) + 1
            if done % 100 == 0:
                elapsed = time.perf_counter() - started
                rate = done / elapsed * 60 if elapsed else 0
                print(f"  {done}/{len(work)}  built={counters['built']} "
                      f"errors={counters['errors']}  {rate:.0f}/min", flush=True)

    elapsed = time.perf_counter() - started
    print(f"\nDone in {elapsed / 60:.1f} min: built={counters['built']} "
          f"skipped={counters['skipped']} errors={counters['errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
