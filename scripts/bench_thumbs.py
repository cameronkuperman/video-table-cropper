#!/usr/bin/env python3
"""Benchmark thumbnail generation throughput and effective parallelism.

Runs a COLD pass (clears the on-disk cache for the target frames, then
generates thumbnails through a ThreadPoolExecutor) followed by a WARM pass
(everything already cached). Reports per-stage medians, effective parallelism,
and triplets/min so each optimization step can be validated against numbers.

Usage:
    python scripts/bench_thumbs.py --source reolink --folders 100 --workers 32
    python scripts/bench_thumbs.py --source video --folders 50 --workers 16

Effective parallelism = (sum of per-task wall times) / (total wall time).
A value near `--workers` means the pool is genuinely concurrent; a value near
1 means the work is serializing (e.g. on the GIL).
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("LABEL_READY_MAINTAINER_ON_STARTUP", "0")
os.environ.setdefault("LABEL_DRAIN_ON_STARTUP", "0")
# Keep the harness output clean; we do our own timing aggregation.
os.environ.setdefault("LABEL_TIMING_LOGS", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import app as label_app  # noqa: E402
from drive_client import DriveClient  # noqa: E402


def _resolve_contexts(client: DriveClient, source: str, site: str | None):
    """Return one or more queue contexts to pull unlabeled folders from."""
    if source == label_app.REOLINK_SOURCE and site in (None, "all"):
        return [
            label_app._resolve_queue_context(client, source, s.site_key)
            for s in label_app.REOLINK_SITES
        ]
    return [label_app._resolve_queue_context(client, source, site)]


def _collect_frame_ids(client: DriveClient, contexts, folder_limit: int):
    """List + hydrate folders across contexts until we have `folder_limit`."""
    triplets: list[list[str]] = []
    for context in contexts:
        if len(triplets) >= folder_limit:
            break
        subfolders = label_app._list_source_subfolders(client, context)
        for folder in subfolders:
            if len(triplets) >= folder_limit:
                break
            payload = label_app._hydrate_folder(client, context, folder)
            if not payload:
                continue
            frames = payload.get("frames") or {}
            file_ids = [fid for fid in frames.values() if fid]
            if file_ids:
                triplets.append(file_ids)
    return triplets


def _clear_cache(file_ids: list[str], clear_fullres: bool) -> None:
    for file_id in file_ids:
        thumb = label_app._thumb_path_for_file(file_id)
        try:
            thumb.unlink()
        except OSError:
            pass
        if clear_fullres:
            full = label_app._cache_path_for_file(file_id)
            try:
                full.unlink()
            except OSError:
                pass


def _run_pass(file_ids: list[str], workers: int):
    """Generate thumbs for all file_ids concurrently; collect per-task stats."""

    def task(file_id: str):
        started = time.perf_counter()
        # warm_client() is thread-local: one reused Drive client per worker.
        _, cache_hit, download_ms, encode_ms = label_app._ensure_thumb_for_file(
            file_id, label_app.warm_client()
        )
        wall_ms = (time.perf_counter() - started) * 1000
        return {
            "wall_ms": wall_ms,
            "download_ms": download_ms,
            "encode_ms": encode_ms,
            "cache_hit": cache_hit,
        }

    results = []
    total_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(task, fid) for fid in file_ids]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                results.append({"error": str(exc)})
    total_wall_ms = (time.perf_counter() - total_started) * 1000
    return results, total_wall_ms


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _report(label: str, results: list[dict], total_wall_ms: float, triplet_count: int) -> None:
    ok = [r for r in results if "error" not in r]
    errors = len(results) - len(ok)
    walls = [r["wall_ms"] for r in ok]
    downloads = [r["download_ms"] for r in ok if r["download_ms"] > 0]
    encodes = [r["encode_ms"] for r in ok if r["encode_ms"] > 0]
    sum_wall = sum(walls)
    eff_par = (sum_wall / total_wall_ms) if total_wall_ms else 0.0
    triplets_per_min = (triplet_count / (total_wall_ms / 1000) * 60) if total_wall_ms else 0.0

    print(f"\n=== {label} PASS ===")
    print(f"  frames:               {len(results)} ({errors} errors)")
    print(f"  total wall:           {total_wall_ms / 1000:.1f}s")
    print(f"  per-frame wall  p50:  {_median(walls):.0f}ms")
    if walls:
        print(f"  per-frame wall  p90:  {sorted(walls)[int(len(walls) * 0.9)] if len(walls) > 1 else walls[0]:.0f}ms")
    print(f"  download        p50:  {_median(downloads):.0f}ms (n={len(downloads)})")
    print(f"  encode          p50:  {_median(encodes):.0f}ms (n={len(encodes)})")
    print(f"  effective parallelism: {eff_par:.1f}x")
    print(f"  THROUGHPUT:            {triplets_per_min:.0f} triplets/min  "
          f"({triplets_per_min * 60:.0f}/hr)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="reolink", help="reolink or video")
    parser.add_argument("--site", default="all", help="reolink site key, or 'all'")
    parser.add_argument("--folders", type=int, default=100, help="number of triplets to bench")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("LABEL_PREVIEW_PREWARM_WORKERS", "32") or "32"),
        help="thread pool size",
    )
    parser.add_argument(
        "--keep-fullres",
        action="store_true",
        help="on the cold pass, keep the full-res cache (measures encode only, not download)",
    )
    args = parser.parse_args()

    client = DriveClient()
    contexts = _resolve_contexts(client, args.source, args.site)
    print(f"Resolving up to {args.folders} folders from {len(contexts)} context(s)...")
    triplets = _collect_frame_ids(client, contexts, args.folders)
    if not triplets:
        print("No folders/frames found — check source/site and Drive credentials.")
        return 1
    file_ids = [fid for triplet in triplets for fid in triplet]
    print(f"Collected {len(triplets)} triplets ({len(file_ids)} frames). "
          f"Workers={args.workers}. clear_fullres={not args.keep_fullres}")

    _clear_cache(file_ids, clear_fullres=not args.keep_fullres)
    cold_results, cold_wall = _run_pass(file_ids, args.workers)
    _report("COLD", cold_results, cold_wall, len(triplets))

    warm_results, warm_wall = _run_pass(file_ids, args.workers)
    _report("WARM", warm_results, warm_wall, len(triplets))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
