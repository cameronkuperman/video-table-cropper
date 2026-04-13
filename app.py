"""
--label mode: Flask UI that reads unlabeled/ subfolders from Drive,
shows 3 images per folder, and moves the folder on Drive when labeled.
"""

from __future__ import annotations

import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from flask import Flask, abort, g, jsonify, render_template, request, send_file

from drive_client import DriveClient, DriveClientError
from env_loader import load_local_env
from queue_metadata import (
    build_folder_app_properties,
    extract_frame_ids_from_item,
    has_complete_frame_ids,
)

load_local_env()

app = Flask(__name__)

QUEUE_BATCH_DEFAULT = max(36, int(os.environ.get("LABEL_QUEUE_BATCH_DEFAULT", "72") or "72"))
QUEUE_BATCH_MAX = max(QUEUE_BATCH_DEFAULT, int(os.environ.get("LABEL_QUEUE_BATCH_MAX", "300") or "300"))
CACHE_CLEANUP_INTERVAL_SECONDS = 300
UNLABELED_LIST_CACHE_SECONDS = max(
    15, int(os.environ.get("LABEL_UNLABELED_CACHE_SECONDS", "300") or "300")
)
HYDRATE_MAX_WORKERS = max(2, int(os.environ.get("LABEL_QUEUE_HYDRATE_WORKERS", "12") or "12"))
PREVIEW_PREWARM_MAX_WORKERS = max(
    2, int(os.environ.get("LABEL_PREVIEW_PREWARM_WORKERS", "24") or "24")
)
FOLDER_PREWARM_MAX_WORKERS = max(
    2, int(os.environ.get("LABEL_FOLDER_PREWARM_WORKERS", "12") or "12")
)
PREWARM_FOLDER_COUNT = max(12, int(os.environ.get("LABEL_PREWARM_FOLDER_COUNT", "180") or "180"))
HYDRATED_FOLDER_CACHE_TTL_SECONDS = max(60, int(os.environ.get("LABEL_HYDRATED_CACHE_TTL_SECONDS", "900") or "900"))
READY_SCAN_MULTIPLIER = max(2, int(os.environ.get("LABEL_READY_SCAN_MULTIPLIER", "12") or "12"))
READY_SCAN_MAX = max(100, int(os.environ.get("LABEL_READY_SCAN_MAX", "720") or "720"))
QUEUE_RETRY_MS = max(100, int(os.environ.get("LABEL_QUEUE_RETRY_MS", "250") or "250"))
TIMING_LOGS_ENABLED = os.environ.get("LABEL_TIMING_LOGS", "1").strip().lower() not in {
    "",
    "0",
    "false",
    "no",
    "off",
}
TIMING_LOG_MIN_MS = max(0.0, float(os.environ.get("LABEL_TIMING_LOG_MIN_MS", "0") or "0"))


def _default_cache_dir() -> Path:
    configured = os.environ.get("LABEL_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()

    local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
    if local_appdata:
        return Path(local_appdata) / "AutoLabeler" / "label_cache"

    return Path(tempfile.gettempdir()) / "AutoLabeler" / "label_cache"


CACHE_DIR = _default_cache_dir()
CACHE_TTL_HOURS = max(1, int(os.environ.get("LABEL_CACHE_TTL_HOURS", "72") or "72"))
CACHE_MAX_MB = max(64, int(os.environ.get("LABEL_CACHE_MAX_MB", "1024") or "1024"))

# Drive client + cached folder IDs
_folder_ids_cache: dict[str, str] | None = None
_folder_ids_lock = Lock()
_cache_cleanup_lock = Lock()
_last_cache_cleanup_monotonic = 0.0
_unlabeled_listing_cache: list[dict[str, str]] | None = None
_unlabeled_listing_cached_at = 0.0
_unlabeled_listing_lock = Lock()
_unlabeled_listing_refresh_executor = ThreadPoolExecutor(max_workers=1)
_unlabeled_listing_refresh_inflight = False
_preview_prewarm_executor = ThreadPoolExecutor(max_workers=PREVIEW_PREWARM_MAX_WORKERS)
_preview_prewarm_inflight: set[str] = set()
_preview_prewarm_lock = Lock()
_folder_prewarm_executor = ThreadPoolExecutor(max_workers=FOLDER_PREWARM_MAX_WORKERS)
_folder_prewarm_inflight: set[str] = set()
_folder_prewarm_lock = Lock()
_hydrated_folder_cache: dict[str, tuple[float, dict | None]] = {}
_hydrated_folder_cache_lock = Lock()


def _log_timing(event: str, **fields: object) -> None:
    if not TIMING_LOGS_ENABLED:
        return

    elapsed_ms = fields.get("total_ms")
    if isinstance(elapsed_ms, (int, float)) and elapsed_ms < TIMING_LOG_MIN_MS:
        return

    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[timing] {event} {details}".rstrip())


def get_client() -> DriveClient:
    client = getattr(g, "_drive_client", None)
    if client is None:
        client = DriveClient()
        g._drive_client = client
    return client


def _ensure_cache_dir() -> Path:
    global CACHE_DIR

    candidates = [CACHE_DIR]
    temp_cache = Path(tempfile.gettempdir()) / "AutoLabeler" / "label_cache"
    repo_cache = Path(__file__).parent / "label_cache"

    for candidate in (temp_cache, repo_cache):
        if candidate not in candidates:
            candidates.append(candidate)

    last_error: OSError | None = None
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            CACHE_DIR = candidate
            return candidate
        except OSError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    return CACHE_DIR


def _root_id() -> str:
    root = os.environ.get("DRIVE_PROJECT_ROOT_FOLDER_ID", "").strip()
    if not root:
        raise RuntimeError("DRIVE_PROJECT_ROOT_FOLDER_ID is not set in .env")
    return root


def _folder_ids() -> dict[str, str]:
    """Return {name: folder_id} for all required subfolders. Cached after first call."""
    global _folder_ids_cache
    if _folder_ids_cache is None:
        with _folder_ids_lock:
            if _folder_ids_cache is None:
                client = get_client()
                root = _root_id()
                names = [
                    "raw_videos",
                    "temp_processing",
                    "unlabeled",
                    "clean",
                    "dirty",
                    "occupied",
                    "label_later",
                ]
                _folder_ids_cache = {name: client.ensure_subfolder(root, name) for name in names}
    return _folder_ids_cache


def _invalidate_unlabeled_listing_cache() -> None:
    global _unlabeled_listing_cache, _unlabeled_listing_cached_at
    with _unlabeled_listing_lock:
        _unlabeled_listing_cache = None
        _unlabeled_listing_cached_at = 0.0


def _set_unlabeled_listing_cache(listing: list[dict[str, str]]) -> None:
    global _unlabeled_listing_cache, _unlabeled_listing_cached_at
    with _unlabeled_listing_lock:
        _unlabeled_listing_cache = listing
        _unlabeled_listing_cached_at = time.monotonic()


def _fetch_unlabeled_listing(client: DriveClient, unlabeled_id: str) -> list[dict[str, str]]:
    return sorted(
        client.list_folders(
            unlabeled_id,
            fields="id,name,mimeType,parents,appProperties",
        ),
        key=lambda item: str(item.get("name", "")).lower(),
    )


def _refresh_unlabeled_listing_in_background(unlabeled_id: str) -> None:
    global _unlabeled_listing_refresh_inflight
    try:
        listing = _fetch_unlabeled_listing(DriveClient(), unlabeled_id)
        _set_unlabeled_listing_cache(listing)
    except Exception:
        return
    finally:
        with _unlabeled_listing_lock:
            _unlabeled_listing_refresh_inflight = False


def _schedule_unlabeled_listing_refresh(unlabeled_id: str) -> bool:
    global _unlabeled_listing_refresh_inflight
    with _unlabeled_listing_lock:
        if _unlabeled_listing_refresh_inflight:
            return False
        _unlabeled_listing_refresh_inflight = True
    _unlabeled_listing_refresh_executor.submit(_refresh_unlabeled_listing_in_background, unlabeled_id)
    return True


def _list_unlabeled_subfolders(
    client: DriveClient,
    unlabeled_id: str,
    force_refresh: bool = False,
) -> list[dict[str, str]]:
    now = time.monotonic()
    with _unlabeled_listing_lock:
        cached_listing = list(_unlabeled_listing_cache) if _unlabeled_listing_cache is not None else None
        cached_at = _unlabeled_listing_cached_at

    cache_is_fresh = cached_listing is not None and (now - cached_at) < UNLABELED_LIST_CACHE_SECONDS
    if cache_is_fresh and not force_refresh:
        return cached_listing

    if cached_listing is not None and not force_refresh:
        _schedule_unlabeled_listing_refresh(unlabeled_id)
        return cached_listing

    listing = _fetch_unlabeled_listing(client, unlabeled_id)
    _set_unlabeled_listing_cache(listing)
    return list(listing)


def _frame_payload_from_files(files: list[dict]) -> dict[str, str | None]:
    file_map = {f["name"]: f["id"] for f in files}
    return {
        "frame_0": file_map.get("frame_0.jpg"),
        "frame_1": file_map.get("frame_1.jpg"),
        "frame_2": file_map.get("frame_2.jpg"),
    }


def _frame_payload_from_folder(folder: dict[str, object]) -> dict[str, str | None]:
    return extract_frame_ids_from_item(folder)


def _build_folder_payload(
    folder: dict[str, str],
    parent_id: str,
    frames: dict[str, str | None],
) -> dict:
    preview_urls = {
        key: f"/api/preview/{file_id}"
        for key, file_id in frames.items()
        if file_id
    }
    return {
        "folder_id": folder["id"],
        "folder_name": folder["name"],
        "parent_id": parent_id,
        "frames": frames,
        "preview_urls": preview_urls,
        "cache_ready": _frames_cache_ready(frames),
    }


def _persist_folder_frame_metadata(
    client: DriveClient,
    folder: dict[str, str],
    frames: dict[str, str | None],
) -> None:
    if not has_complete_frame_ids(frames):
        return
    try:
        metadata = dict(folder.get("appProperties") or {})
        metadata.update(build_folder_app_properties(frames))
        client.update_file_metadata(
            folder["id"],
            {"appProperties": metadata},
            fields="id,appProperties",
        )
        folder["appProperties"] = metadata
    except DriveClientError:
        return


def _cache_path_for_file(file_id: str) -> Path:
    return _ensure_cache_dir() / f"{file_id}.jpg"


def _frames_cache_ready(frames: dict[str, str | None]) -> bool:
    for key in ("frame_0", "frame_1", "frame_2"):
        file_id = frames.get(key)
        if not file_id or not _cache_path_for_file(file_id).exists():
            return False
    return True


def _folder_cache_ready(folder: dict) -> bool:
    return _frames_cache_ready(folder.get("frames", {}))


def _hydrate_folder(client: DriveClient, folder: dict[str, str], parent_id: str) -> dict | None:
    frames = _frame_payload_from_folder(folder)
    if has_complete_frame_ids(frames):
        return _build_folder_payload(folder, parent_id, frames)

    frames = _frame_payload_from_files(client.list_files(folder["id"]))
    if not has_complete_frame_ids(frames):
        return None

    _persist_folder_frame_metadata(client, folder, frames)
    return _build_folder_payload(folder, parent_id, frames)


def _get_cached_hydrated_folder(folder_id: str) -> dict | None | object:
    now = time.monotonic()
    with _hydrated_folder_cache_lock:
        cached = _hydrated_folder_cache.get(folder_id)
        if not cached:
            return _MISSING
        cached_at, payload = cached
        if (now - cached_at) > HYDRATED_FOLDER_CACHE_TTL_SECONDS:
            _hydrated_folder_cache.pop(folder_id, None)
            return _MISSING
        return payload


def _set_cached_hydrated_folder(folder_id: str, payload: dict | None) -> None:
    with _hydrated_folder_cache_lock:
        _hydrated_folder_cache[folder_id] = (time.monotonic(), payload)


_MISSING = object()


def _warm_file(file_id: str) -> None:
    try:
        cache_path = _cache_path_for_file(file_id)
        if cache_path.exists():
            try:
                os.utime(cache_path, None)
            except OSError:
                pass
            return

        DriveClient().download_file_to_path(file_id, cache_path)
    except Exception:
        return
    finally:
        with _preview_prewarm_lock:
            _preview_prewarm_inflight.discard(file_id)


def _warm_folder_payload(folder: dict[str, str], parent_id: str) -> None:
    try:
        payload = _hydrate_folder_with_fresh_client(folder, parent_id)
        _set_cached_hydrated_folder(folder["id"], payload)
        if payload is not None and not _folder_cache_ready(payload):
            _schedule_preview_prewarm([payload])
    except Exception:
        return
    finally:
        with _folder_prewarm_lock:
            _folder_prewarm_inflight.discard(folder["id"])


def _schedule_preview_prewarm(hydrated_folders: list[dict]) -> int:
    if not hydrated_folders:
        return 0

    warm_targets = hydrated_folders[:PREWARM_FOLDER_COUNT]
    scheduled = 0
    for folder in warm_targets:
        frames = folder.get("frames", {})
        for key in ("frame_0", "frame_1", "frame_2"):
            file_id = frames.get(key)
            if not file_id:
                continue
            if _cache_path_for_file(file_id).exists():
                continue
            with _preview_prewarm_lock:
                if file_id in _preview_prewarm_inflight:
                    continue
                _preview_prewarm_inflight.add(file_id)
            _preview_prewarm_executor.submit(_warm_file, file_id)
            scheduled += 1
    return scheduled


def _hydrate_folder_with_fresh_client(folder: dict[str, str], parent_id: str) -> dict | None:
    # googleapiclient service objects are safer to keep thread-local.
    client = DriveClient()
    return _hydrate_folder(client, folder, parent_id)


def _hydrate_folders_parallel(
    folders: list[dict[str, str]],
    parent_id: str,
) -> tuple[list[dict | None], dict[str, int]]:
    if not folders:
        return [], {
            "requested": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "workers": 0,
        }

    results: list[dict | None | object] = [_MISSING] * len(folders)
    uncached: list[tuple[int, dict[str, str]]] = []

    for idx, folder in enumerate(folders):
        cached = _get_cached_hydrated_folder(folder["id"])
        if cached is _MISSING:
            uncached.append((idx, folder))
        else:
            if cached is not None:
                cached["cache_ready"] = _folder_cache_ready(cached)
            results[idx] = cached

    hydrate_stats = {
        "requested": len(folders),
        "cache_hits": len(folders) - len(uncached),
        "cache_misses": len(uncached),
        "workers": 0,
    }

    if uncached:
        if len(uncached) == 1:
            idx, folder = uncached[0]
            payload = _hydrate_folder(get_client(), folder, parent_id)
            if payload is not None:
                payload["cache_ready"] = _folder_cache_ready(payload)
            _set_cached_hydrated_folder(folder["id"], payload)
            results[idx] = payload
        else:
            max_workers = min(HYDRATE_MAX_WORKERS, len(uncached))
            hydrate_stats["workers"] = max_workers
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                payloads = list(
                    executor.map(
                        lambda item: _hydrate_folder_with_fresh_client(item[1], parent_id),
                        uncached,
                    )
                )
            for (idx, folder), payload in zip(uncached, payloads):
                if payload is not None:
                    payload["cache_ready"] = _folder_cache_ready(payload)
                _set_cached_hydrated_folder(folder["id"], payload)
                results[idx] = payload

    return [None if payload is _MISSING else payload for payload in results], hydrate_stats


def _schedule_folder_hydration_prewarm(
    subfolders: list[dict[str, str]],
    start_idx: int,
    parent_id: str,
) -> int:
    scheduled = 0
    end_idx = min(len(subfolders), start_idx + PREWARM_FOLDER_COUNT)
    for folder in subfolders[start_idx:end_idx]:
        if _get_cached_hydrated_folder(folder["id"]) is not _MISSING:
            continue
        with _folder_prewarm_lock:
            if folder["id"] in _folder_prewarm_inflight:
                continue
            _folder_prewarm_inflight.add(folder["id"])
        _folder_prewarm_executor.submit(_warm_folder_payload, folder, parent_id)
        scheduled += 1
    return scheduled


def _compute_stats(client: DriveClient, folder_ids: dict[str, str]) -> dict[str, int]:
    stats: dict[str, int] = {}
    for name in ("unlabeled", "clean", "dirty", "occupied", "label_later"):
        stats[name] = len(client.list_folders(folder_ids[name]))
    return stats


def _collect_ready_folders(
    subfolders: list[dict[str, str]],
    parent_id: str,
    limit: int,
) -> tuple[list[dict], dict[str, int | float]]:
    ready: list[dict] = []
    nonready = 0
    hydrated_valid = 0
    scanned = 0
    hydrate_ms = 0.0
    hydrate_requested = 0
    hydrate_cache_hits = 0
    hydrate_cache_misses = 0
    hydrate_worker_max = 0
    first_unready_idx = len(subfolders)

    target_scan = min(len(subfolders), max(limit * READY_SCAN_MULTIPLIER, PREWARM_FOLDER_COUNT))
    target_scan = min(target_scan, READY_SCAN_MAX)

    while scanned < target_scan and len(ready) < limit:
        remaining_scan = target_scan - scanned
        batch_span = min(
            remaining_scan,
            max(HYDRATE_MAX_WORKERS, (limit - len(ready)) * 2),
        )
        folder_batch = subfolders[scanned:scanned + batch_span]
        if not folder_batch:
            break

        hydrate_started = time.perf_counter()
        payloads, hydrate_stats = _hydrate_folders_parallel(folder_batch, parent_id)
        hydrate_ms += (time.perf_counter() - hydrate_started) * 1000
        hydrate_requested += hydrate_stats["requested"]
        hydrate_cache_hits += hydrate_stats["cache_hits"]
        hydrate_cache_misses += hydrate_stats["cache_misses"]
        hydrate_worker_max = max(hydrate_worker_max, hydrate_stats["workers"])

        for offset, payload in enumerate(payloads):
            absolute_idx = scanned + offset
            if payload is None:
                continue
            hydrated_valid += 1
            payload["cache_ready"] = _folder_cache_ready(payload)
            if payload["cache_ready"]:
                ready.append(payload)
            else:
                nonready += 1
                first_unready_idx = min(first_unready_idx, absolute_idx)
                _schedule_preview_prewarm([payload])
            if len(ready) >= limit:
                break

        scanned += len(folder_batch)

    prewarm_scan_start = first_unready_idx if first_unready_idx < len(subfolders) else scanned
    folder_prewarm_scheduled = _schedule_folder_hydration_prewarm(subfolders, prewarm_scan_start, parent_id)

    return ready, {
        "scanned": scanned,
        "hydrate_ms": hydrate_ms,
        "hydrate_requested": hydrate_requested,
        "hydrate_cache_hits": hydrate_cache_hits,
        "hydrate_cache_misses": hydrate_cache_misses,
        "hydrate_worker_max": hydrate_worker_max,
        "folder_prewarm_scheduled": folder_prewarm_scheduled,
        "prewarm_scan_start": prewarm_scan_start,
        "hydrated_valid": hydrated_valid,
        "nonready": nonready,
    }


def _cleanup_cache_if_needed(force: bool = False) -> None:
    global _last_cache_cleanup_monotonic

    now_monotonic = time.monotonic()
    if not force and (now_monotonic - _last_cache_cleanup_monotonic) < CACHE_CLEANUP_INTERVAL_SECONDS:
        return

    with _cache_cleanup_lock:
        now_monotonic = time.monotonic()
        if not force and (now_monotonic - _last_cache_cleanup_monotonic) < CACHE_CLEANUP_INTERVAL_SECONDS:
            return

        cache_dir = _ensure_cache_dir()
        now_epoch = time.time()
        ttl_seconds = CACHE_TTL_HOURS * 3600
        max_bytes = CACHE_MAX_MB * 1024 * 1024

        files = [path for path in cache_dir.iterdir() if path.is_file()]
        for path in files:
            try:
                age_seconds = now_epoch - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age_seconds > ttl_seconds:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

        file_stats: list[tuple[float, int, Path]] = []
        total_bytes = 0
        for path in cache_dir.iterdir():
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            total_bytes += stat.st_size
            file_stats.append((stat.st_mtime, stat.st_size, path))

        if total_bytes > max_bytes:
            for _, size, path in sorted(file_stats, key=lambda item: item[0]):
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                total_bytes -= size
                if total_bytes <= max_bytes:
                    break

        _last_cache_cleanup_monotonic = now_monotonic


@app.route("/")
def index():
    return render_template("label.html")


@app.route("/api/folders")
def api_folders():
    """Return list of unlabeled subfolders (names + IDs only, no file listing)."""
    try:
        client = get_client()
        folder_ids = _folder_ids()
        unlabeled_id = folder_ids["unlabeled"]
        subfolders = _list_unlabeled_subfolders(
            client,
            unlabeled_id,
            force_refresh=request.args.get("refresh", "0") == "1",
        )
        result = [
            {"folder_id": f["id"], "folder_name": f["name"], "parent_id": unlabeled_id}
            for f in subfolders
        ]
        return jsonify({"folders": result})
    except (DriveClientError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/queue")
def api_queue():
    """Return a ready-to-render batch of unlabeled folders whose previews are locally cached."""
    request_started = time.perf_counter()
    try:
        client = get_client()
        folder_ids = _folder_ids()
        unlabeled_id = folder_ids["unlabeled"]

        limit = int(request.args.get("limit", str(QUEUE_BATCH_DEFAULT)) or str(QUEUE_BATCH_DEFAULT))
        limit = max(1, min(limit, QUEUE_BATCH_MAX))
        include_stats = request.args.get("include_stats", "0") == "1"
        force_refresh = request.args.get("refresh", "0") == "1"

        list_started = time.perf_counter()
        subfolders = _list_unlabeled_subfolders(client, unlabeled_id, force_refresh=force_refresh)
        list_ms = (time.perf_counter() - list_started) * 1000
        total_unlabeled = len(subfolders)

        ready_folders, ready_stats = _collect_ready_folders(subfolders, unlabeled_id, limit)

        preview_prewarm_scheduled = _schedule_preview_prewarm(ready_folders)
        ready_buffer_count = len(ready_folders)
        warming_count = int(ready_stats["nonready"])

        response: dict[str, object] = {
            "folders": ready_folders,
            "next_cursor": 0,
            "total_unlabeled": total_unlabeled,
            "has_more": total_unlabeled > ready_buffer_count,
            "ready_buffer_count": ready_buffer_count,
            "warming_count": warming_count,
            "retry_ms": QUEUE_RETRY_MS if total_unlabeled > 0 and ready_buffer_count < limit else 0,
        }
        if include_stats:
            stats_started = time.perf_counter()
            response["stats"] = _compute_stats(client, folder_ids)
            stats_ms = (time.perf_counter() - stats_started) * 1000
        else:
            stats_ms = 0.0

        total_ms = (time.perf_counter() - request_started) * 1000
        _log_timing(
            "api_queue",
            total_ms=f"{total_ms:.1f}",
            list_ms=f"{list_ms:.1f}",
            hydrate_ms=f"{ready_stats['hydrate_ms']:.1f}",
            stats_ms=f"{stats_ms:.1f}",
            limit=limit,
            returned=len(ready_folders),
            ready_buffer=ready_buffer_count,
            warming=warming_count,
            scanned=ready_stats["scanned"],
            hydrated_valid=ready_stats["hydrated_valid"],
            cache_hits=ready_stats["hydrate_cache_hits"],
            cache_misses=ready_stats["hydrate_cache_misses"],
            workers=ready_stats["hydrate_worker_max"],
            prewarm_folders=ready_stats["folder_prewarm_scheduled"],
            prewarm_files=preview_prewarm_scheduled,
            total_unlabeled=total_unlabeled,
            include_stats=int(include_stats),
            refresh=int(force_refresh),
        )
        return jsonify(response)
    except (DriveClientError, RuntimeError, ValueError) as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/folder/<folder_id>/frames")
def api_folder_frames(folder_id: str):
    """Return frame file IDs for a single subfolder."""
    try:
        client = get_client()
        folder = client.get_file(folder_id, fields="id,name,parents,appProperties")
        frames = _frame_payload_from_folder(folder)
        if not has_complete_frame_ids(frames):
            frames = _frame_payload_from_files(client.list_files(folder_id))
        return jsonify(frames)
    except DriveClientError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/preview/<file_id>")
def api_preview(file_id: str):
    """Serve a Drive image file, caching it locally."""
    request_started = time.perf_counter()
    _cleanup_cache_if_needed()

    cache_path = _cache_path_for_file(file_id)
    cache_hit = cache_path.exists()
    download_ms = 0.0
    if not cache_path.exists():
        try:
            client = get_client()
            download_started = time.perf_counter()
            client.download_file_to_path(file_id, cache_path)
            download_ms = (time.perf_counter() - download_started) * 1000
        except DriveClientError as e:
            abort(404, description=str(e))

    try:
        os.utime(cache_path, None)
    except OSError:
        pass

    try:
        size_bytes = cache_path.stat().st_size
    except OSError:
        size_bytes = 0

    total_ms = (time.perf_counter() - request_started) * 1000
    _log_timing(
        "api_preview",
        total_ms=f"{total_ms:.1f}",
        download_ms=f"{download_ms:.1f}",
        cache="hit" if cache_hit else "miss",
        size_kb=f"{size_bytes / 1024:.1f}",
        file_id=file_id,
    )
    return send_file(cache_path, mimetype="image/jpeg", conditional=True, max_age=3600)


@app.route("/api/label", methods=["POST"])
def api_label():
    """Move a folder from unlabeled/ to a label destination."""
    request_started = time.perf_counter()
    data = request.get_json(force=True)
    folder_id = data.get("folder_id", "").strip()
    parent_id = data.get("parent_id", "").strip()
    label = data.get("label", "").strip().lower()

    if not folder_id or not parent_id:
        return jsonify({"error": "folder_id and parent_id required"}), 400
    if label not in ("clean", "dirty", "occupied", "label_later"):
        return jsonify({"error": "label must be clean, dirty, occupied, or label_later"}), 400

    try:
        client = get_client()
        folder_ids = _folder_ids()
        dest_id = folder_ids[label]
        move_started = time.perf_counter()
        client.move_file(folder_id, new_parent_id=dest_id, remove_parent_id=parent_id)
        move_ms = (time.perf_counter() - move_started) * 1000
        with _hydrated_folder_cache_lock:
            _hydrated_folder_cache.pop(folder_id, None)
        with _unlabeled_listing_lock:
            global _unlabeled_listing_cache
            if _unlabeled_listing_cache is not None:
                _unlabeled_listing_cache = [
                    item for item in _unlabeled_listing_cache if item.get("id") != folder_id
                ]
        total_ms = (time.perf_counter() - request_started) * 1000
        _log_timing(
            "api_label",
            total_ms=f"{total_ms:.1f}",
            move_ms=f"{move_ms:.1f}",
            label=label,
            folder_id=folder_id,
        )
        return jsonify({"ok": True, "moved_to": label})
    except (DriveClientError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    """Return counts of folders in each category."""
    request_started = time.perf_counter()
    try:
        client = get_client()
        folder_ids = _folder_ids()
        stats = _compute_stats(client, folder_ids)
        total_ms = (time.perf_counter() - request_started) * 1000
        _log_timing("api_stats", total_ms=f"{total_ms:.1f}", **stats)
        return jsonify(stats)
    except (DriveClientError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500


def run_label_ui(port: int = 8080) -> None:
    print(f"Starting label UI at http://localhost:{port}")
    _cleanup_cache_if_needed(force=True)
    print(f"Preview cache: {CACHE_DIR}")
    print(
        "Timing logs: "
        f"{'on' if TIMING_LOGS_ENABLED else 'off'}"
        f" (min {TIMING_LOG_MIN_MS:.0f} ms)"
    )
    # Request-scoped Drive clients and locked shared caches make threaded
    # serving practical here, which helps keep the warm queue filled.
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)
