#!/usr/bin/env python3
"""Archive Reolink raw triplet folders from <site>/unassociated/ into zip
batches under <site>/unassociated_zips/, then permanently delete the originals.

Frees shared-drive item-cap headroom: each triplet (1 folder + 3 frames = 4
items) becomes ~1 zip file. Re-cropping is not supported; the zip + local
extract is the canonical recovery path.

Modes:
  --mode dry-run     Read-only. Counts triplets per site and prints what would
                     happen. No Drive writes (no uploads, no deletes, no
                     temp_processing wipe).
  --mode test-batch  Wipes temp_processing/, archives ONE batch (capped at 50
                     triplets) end-to-end with full verification, then stops.
  --mode confirm     Wipes temp_processing/, then archives every batch across
                     both Reolink sites until done.

Usage:
  DRIVE_PROJECT_ROOT_FOLDER_ID=...
  DRIVE_SERVICE_ACCOUNT_JSON_PATH=...
  python scripts/compact_unassociated_to_zips.py --mode dry-run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import socket
import ssl
import sys
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httplib2

# Disable the Flask label-drain worker that auto-schedules at app import.
os.environ.setdefault("LABEL_DRAIN_ON_STARTUP", "0")
# Bump Drive API retries for the long-running compactor: SSL hiccups under
# parallel load are common on slow connections. 10 attempts with base 2.0s
# backoff = ~17 minutes max wait per call (2,4,8,16,32,64,128,256,512,1024s).
os.environ.setdefault("DRIVE_API_RETRY_ATTEMPTS", "10")
os.environ.setdefault("DRIVE_API_RETRY_BASE_SECONDS", "2.0")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drive_client import DriveClient, DriveClientError, FOLDER_MIME  # noqa: E402
from googleapiclient.errors import HttpError  # noqa: E402
from googleapiclient.http import MediaFileUpload  # noqa: E402
import app as label_app  # noqa: E402


TEST_BATCH_CAP = 50
DEFAULT_BATCH_SIZE = 500
DEFAULT_MIN_AGE_MINUTES = 60
DEFAULT_DOWNLOAD_WORKERS = 4  # conservative: googleapiclient SSL stack is fragile under concurrent thread reuse
DEFAULT_DELETE_WORKERS = 4
DEFAULT_BATCH_FAILURE_TOLERANCE = 0.10  # abort batch if >10% of triplets fail

# Resumable upload settings: large zips on slow connections die with SSL timeouts
# during a single non-resumable POST. Chunked resumable uploads survive
# transient drops by resuming from the last acked byte instead of restarting.
RESUMABLE_CHUNK_BYTES = max(1, int(os.environ.get("COMPACTOR_UPLOAD_CHUNK_MB", "8") or "8")) * 1024 * 1024
RESUMABLE_MAX_ATTEMPTS_PER_CHUNK = max(1, int(os.environ.get("COMPACTOR_UPLOAD_CHUNK_ATTEMPTS", "10") or "10"))
RESUMABLE_MAX_SESSIONS = max(1, int(os.environ.get("COMPACTOR_UPLOAD_SESSIONS", "3") or "3"))
ZIPS_FOLDER_NAME = "unassociated_zips"
MANIFEST_LOG_NAME = ".compactor_manifest.jsonl"
MANIFEST_FILE_NAME = "MANIFEST.json"
SCHEMA_VERSION = 1

EXPECTED_FRAME_NAMES = ("frame_0.jpg", "frame_1.jpg", "frame_2.jpg")


@dataclass
class SiteContext:
    site_key: str
    display_name: str
    site_root_id: str
    unassociated_id: str
    zips_folder_id: str | None  # None in dry-run (never created)


@dataclass
class TripletFolder:
    folder_id: str
    name: str
    sanitized_name: str  # safe-on-disk, name__id pattern
    modified_time: dt.datetime | None
    frame_files: list[dict[str, Any]]  # populated lazily before zipping


@dataclass
class BatchResult:
    site_key: str
    batch_id: str
    triplet_count: int
    bytes_uploaded: int
    items_freed: int


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_name(name: str, file_id: str) -> str:
    base = SAFE_NAME_RE.sub("_", name).strip("._-") or "triplet"
    return f"{base}__{file_id}"


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse_drive_timestamp(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _log(message: str) -> None:
    timestamp = _utc_now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Phase 0: temp_processing wipe
# ---------------------------------------------------------------------------


def wipe_temp_processing(client: DriveClient, *, dry_run: bool) -> int:
    """Permanently delete every direct child of top-level temp_processing/.

    Returns count of items deleted (or that would be deleted in dry-run).
    """
    root_id = label_app._root_id()
    folder = client.find_file_by_name(root_id, "temp_processing", mime_type=FOLDER_MIME)
    if not folder or not folder.get("id"):
        _log("Phase 0: temp_processing/ not present; skipping.")
        return 0

    children = client.list_files(str(folder["id"]), fields="id,name,mimeType")
    if not children:
        _log("Phase 0: temp_processing/ is already empty; skipping.")
        return 0

    if dry_run:
        _log(f"Phase 0 [dry-run]: would permanently delete {len(children)} top-level items under temp_processing/.")
        return len(children)

    _log(f"Phase 0: permanently deleting {len(children)} top-level items under temp_processing/...")
    deleted = 0
    for item in children:
        item_id = str(item["id"])
        try:
            client.delete_file(item_id)
            deleted += 1
            if deleted % 25 == 0:
                _log(f"  ...deleted {deleted}/{len(children)}")
        except DriveClientError as exc:
            _log(f"  WARNING: failed to delete {item.get('name')!r} ({item_id}): {exc}")
    _log(f"Phase 0: done. Deleted {deleted}/{len(children)} top-level items.")
    return deleted


# ---------------------------------------------------------------------------
# Site context resolution
# ---------------------------------------------------------------------------


def resolve_site_context(client: DriveClient, site_key: str, *, dry_run: bool) -> SiteContext:
    site = label_app._resolve_site_config(site_key)
    site_root_id = label_app._discover_reolink_root_id(client, site)

    unassociated = client.find_file_by_name(site_root_id, "unassociated", mime_type=FOLDER_MIME)
    if not unassociated or not unassociated.get("id"):
        raise RuntimeError(
            f"Site {site_key!r} is missing the required 'unassociated' folder under site root {site_root_id}."
        )

    zips_folder_id: str | None = None
    if dry_run:
        existing_zips = client.find_file_by_name(site_root_id, ZIPS_FOLDER_NAME, mime_type=FOLDER_MIME)
        zips_folder_id = str(existing_zips["id"]) if existing_zips and existing_zips.get("id") else None
    else:
        zips_folder_id = client.ensure_subfolder(site_root_id, ZIPS_FOLDER_NAME)

    return SiteContext(
        site_key=site_key,
        display_name=site.display_name,
        site_root_id=site_root_id,
        unassociated_id=str(unassociated["id"]),
        zips_folder_id=zips_folder_id,
    )


# ---------------------------------------------------------------------------
# Manifest log (resumability)
# ---------------------------------------------------------------------------


def load_archived_triplet_ids(client: DriveClient, ctx: SiteContext) -> set[str]:
    """Read .compactor_manifest.jsonl from <site>/unassociated_zips/ and return
    the set of triplet folder IDs already covered by an existing zip.
    """
    if not ctx.zips_folder_id:
        return set()

    log_file = client.find_file_by_name(ctx.zips_folder_id, MANIFEST_LOG_NAME)
    if not log_file or not log_file.get("id"):
        return set()

    raw = client.download_file_content(str(log_file["id"]))
    text = raw.decode("utf-8", errors="replace")
    archived: set[str] = set()
    zip_ids_seen: dict[str, bool] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        zip_id = str(entry.get("zip_file_id") or "")
        triplet_ids = entry.get("triplet_ids") or []
        if not zip_id or not isinstance(triplet_ids, list):
            continue
        if zip_id not in zip_ids_seen:
            try:
                client.get_file(zip_id, fields="id,trashed")
                zip_ids_seen[zip_id] = True
            except Exception:
                zip_ids_seen[zip_id] = False
        if zip_ids_seen.get(zip_id):
            archived.update(str(tid) for tid in triplet_ids)

    return archived


def append_manifest_entry(
    client: DriveClient,
    ctx: SiteContext,
    entry: dict[str, Any],
) -> None:
    """Append a single JSONL entry to the on-Drive manifest log. Reads the
    existing file, appends a line locally, and replaces the file via update.
    """
    if not ctx.zips_folder_id:
        raise RuntimeError("zips_folder_id is required to append manifest entry")

    existing = client.find_file_by_name(ctx.zips_folder_id, MANIFEST_LOG_NAME)
    new_line = json.dumps(entry, sort_keys=True) + "\n"

    if existing and existing.get("id"):
        existing_id = str(existing["id"])
        try:
            current_bytes = client.download_file_content(existing_id)
        except Exception:
            current_bytes = b""
        merged = current_bytes + new_line.encode("utf-8")
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp.write(merged)
            tmp_path = Path(tmp.name)
        try:
            client.update_file(existing_id, tmp_path, mime_type="application/x-ndjson")
        finally:
            tmp_path.unlink(missing_ok=True)
        return

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp.write(new_line.encode("utf-8"))
        tmp_path = Path(tmp.name)
    try:
        client.upload_file(
            tmp_path,
            ctx.zips_folder_id,
            file_name=MANIFEST_LOG_NAME,
            mime_type="application/x-ndjson",
        )
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Triplet listing
# ---------------------------------------------------------------------------


def list_eligible_triplets(
    client: DriveClient,
    ctx: SiteContext,
    *,
    min_age_minutes: int,
    archived_ids: set[str],
) -> list[TripletFolder]:
    cutoff = _utc_now() - dt.timedelta(minutes=max(0, min_age_minutes))
    raw = client.list_folders(
        ctx.unassociated_id,
        fields="id,name,modifiedTime,parents",
    )
    eligible: list[TripletFolder] = []
    for item in raw:
        folder_id = str(item.get("id") or "")
        if not folder_id or folder_id in archived_ids:
            continue
        modified = _parse_drive_timestamp(item.get("modifiedTime"))
        if modified is not None and modified > cutoff:
            continue
        name = str(item.get("name") or "triplet")
        eligible.append(
            TripletFolder(
                folder_id=folder_id,
                name=name,
                sanitized_name=_sanitize_name(name, folder_id),
                modified_time=modified,
                frame_files=[],
            )
        )
    eligible.sort(key=lambda t: (t.modified_time or _utc_now(), t.folder_id))
    return eligible


# ---------------------------------------------------------------------------
# Per-batch zip + verify + delete
# ---------------------------------------------------------------------------


_thread_local = threading.local()


def _thread_local_client() -> DriveClient:
    """Return a DriveClient unique to this thread. googleapiclient's SSL
    stack is not safe under heavy concurrent reuse from multiple threads —
    each thread gets its own client instance so OpenSSL state is isolated.
    """
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = DriveClient()
        _thread_local.client = client
    return client


def _hydrate_batch_files(
    client: DriveClient,
    batch: list[TripletFolder],
    *,
    workers: int,
) -> tuple[list[TripletFolder], list[tuple[TripletFolder, Exception]]]:
    """Populate triplet.frame_files for each triplet in batch.

    Returns (successful, failures). A triplet is "successful" if list_files
    succeeded AND it has all 3 expected frames. Failures are kept separately
    so the caller can decide whether to abort the batch or skip them.
    """
    def hydrate_one(triplet: TripletFolder) -> tuple[TripletFolder, Exception | None]:
        try:
            worker_client = client if workers <= 1 else _thread_local_client()
            files = worker_client.list_files(triplet.folder_id, fields="id,name,mimeType,size,md5Checksum")
            frames = [
                f
                for f in files
                if str(f.get("mimeType", "")) != FOLDER_MIME
                and str(f.get("name", "")) in EXPECTED_FRAME_NAMES
            ]
            triplet.frame_files = frames
            if len(frames) != len(EXPECTED_FRAME_NAMES):
                return triplet, RuntimeError(
                    f"expected {len(EXPECTED_FRAME_NAMES)} frames, got {len(frames)}"
                )
            return triplet, None
        except Exception as exc:
            return triplet, exc

    successful: list[TripletFolder] = []
    failures: list[tuple[TripletFolder, Exception]] = []

    if workers <= 1 or len(batch) <= 1:
        for triplet in batch:
            t, exc = hydrate_one(triplet)
            if exc is None:
                successful.append(t)
            else:
                failures.append((t, exc))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(hydrate_one, triplet) for triplet in batch]
            for future in concurrent.futures.as_completed(futures):
                t, exc = future.result()
                if exc is None:
                    successful.append(t)
                else:
                    failures.append((t, exc))

    for triplet, exc in failures:
        _log(f"  WARNING: hydrate failed for {triplet.name!r} ({triplet.folder_id}): {exc}")

    return successful, failures


def _build_batch_id(site_key: str, batch_index: int) -> str:
    timestamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    return f"unassociated_{site_key}_{timestamp}_b{batch_index:04d}"


def _download_batch_to_local(
    client: DriveClient,
    batch: list[TripletFolder],
    work_dir: Path,
    *,
    workers: int,
) -> tuple[dict[str, Any], list[TripletFolder], list[tuple[TripletFolder, Exception]]]:
    """Download every frame for every triplet in the batch into work_dir.

    Returns (manifest, archived_triplets, failures). Triplets that don't end
    up with all 3 frames on disk are dropped from the manifest and reported
    as failures so the caller can leave their originals on Drive for retry.
    """
    download_jobs: list[tuple[TripletFolder, dict[str, Any], Path]] = []
    for triplet in batch:
        if not triplet.frame_files:
            continue
        target_dir = work_dir / triplet.sanitized_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for frame in triplet.frame_files:
            target_path = target_dir / str(frame["name"])
            download_jobs.append((triplet, frame, target_path))

    def fetch(job: tuple[TripletFolder, dict[str, Any], Path]) -> tuple[TripletFolder, dict[str, Any], Path, int | None, Exception | None]:
        triplet, frame, target_path = job
        try:
            worker_client = client if workers <= 1 else _thread_local_client()
            worker_client.download_file_to_path(str(frame["id"]), target_path)
            size = target_path.stat().st_size
            return triplet, frame, target_path, size, None
        except Exception as exc:
            return triplet, frame, target_path, None, exc

    by_triplet: dict[str, dict[str, Any]] = {}
    triplet_errors: dict[str, Exception] = {}
    for triplet in batch:
        by_triplet[triplet.folder_id] = {
            "triplet_id": triplet.folder_id,
            "triplet_name": triplet.name,
            "sanitized_path": triplet.sanitized_name,
            "frames": [],
            "frame_bytes_total": 0,
            "modified_time": triplet.modified_time.isoformat() if triplet.modified_time else None,
        }

    overall_bytes = 0

    def record(triplet: TripletFolder, frame: dict[str, Any], target_path: Path, size: int | None, exc: Exception | None) -> int:
        nonlocal overall_bytes
        if exc is not None:
            triplet_errors[triplet.folder_id] = exc
            try:
                target_path.unlink(missing_ok=True)
            except Exception:
                pass
            return 0
        entry = by_triplet[triplet.folder_id]
        entry["frames"].append(
            {
                "frame_id": str(frame["id"]),
                "frame_name": str(frame["name"]),
                "bytes": int(size or 0),
                "drive_md5": frame.get("md5Checksum"),
            }
        )
        entry["frame_bytes_total"] += int(size or 0)
        overall_bytes += int(size or 0)
        return int(size or 0)

    if workers <= 1:
        for job in download_jobs:
            triplet, frame, target_path, size, exc = fetch(job)
            record(triplet, frame, target_path, size, exc)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_job = {executor.submit(fetch, job): job for job in download_jobs}
            for future in concurrent.futures.as_completed(future_to_job):
                triplet, frame, target_path, size, exc = future.result()
                record(triplet, frame, target_path, size, exc)

    archived_triplets: list[TripletFolder] = []
    failures: list[tuple[TripletFolder, Exception]] = []
    triplets_meta: list[dict[str, Any]] = []

    expected = len(EXPECTED_FRAME_NAMES)
    for triplet in batch:
        entry = by_triplet[triplet.folder_id]
        entry["frames"].sort(key=lambda f: f["frame_name"])
        entry["frame_count"] = len(entry["frames"])
        if len(entry["frames"]) == expected and triplet.folder_id not in triplet_errors:
            triplets_meta.append(entry)
            archived_triplets.append(triplet)
        else:
            err = triplet_errors.get(
                triplet.folder_id,
                RuntimeError(f"triplet ended with {len(entry['frames'])}/{expected} frames"),
            )
            failures.append((triplet, err))
            triplet_dir = work_dir / triplet.sanitized_name
            try:
                shutil.rmtree(triplet_dir, ignore_errors=True)
            except Exception:
                pass

    for triplet, exc in failures:
        _log(f"  WARNING: download failed for {triplet.name!r} ({triplet.folder_id}): {exc}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "triplet_count": len(triplets_meta),
        "triplets": triplets_meta,
        "frame_bytes_total": overall_bytes,
    }
    return manifest, archived_triplets, failures


def _zip_local_dir(work_dir: Path, zip_path: Path) -> tuple[int, str]:
    """Zip everything under work_dir into zip_path. Returns (bytes, sha256_hex)."""
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(work_dir.rglob("*")):
            if path.is_dir():
                continue
            arcname = path.relative_to(work_dir).as_posix()
            zf.write(path, arcname)

    hasher = hashlib.sha256()
    with zip_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return zip_path.stat().st_size, hasher.hexdigest()


def _verify_uploaded_zip(
    client: DriveClient,
    uploaded_id: str,
    *,
    expected_size: int,
    local_md5: str,
    local_sha256: str,
    expected_arcnames: set[str],
    expected_sizes: dict[str, int],
    work_root: Path,
) -> None:
    meta = client.get_file(uploaded_id, fields="id,name,size,md5Checksum,mimeType")
    drive_size = int(meta.get("size") or 0)
    if drive_size != expected_size:
        raise RuntimeError(
            f"Uploaded zip size mismatch: drive={drive_size}, local={expected_size} (id={uploaded_id})."
        )

    drive_md5 = str(meta.get("md5Checksum") or "")
    if drive_md5 and drive_md5.lower() != local_md5.lower():
        raise RuntimeError(
            f"Uploaded zip md5 mismatch: drive={drive_md5}, local={local_md5} (id={uploaded_id})."
        )

    roundtrip_path = work_root / "verify_download.zip"
    client.download_file_to_path(uploaded_id, roundtrip_path)
    hasher = hashlib.sha256()
    with roundtrip_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    roundtrip_sha = hasher.hexdigest()
    if roundtrip_sha != local_sha256:
        raise RuntimeError(
            f"Round-trip SHA256 mismatch: drive={roundtrip_sha}, local={local_sha256} (id={uploaded_id})."
        )

    with zipfile.ZipFile(roundtrip_path, mode="r") as zf:
        names_in_zip = {info.filename for info in zf.infolist()}
        sizes_in_zip = {info.filename: info.file_size for info in zf.infolist()}

    missing = expected_arcnames - names_in_zip
    if missing:
        raise RuntimeError(
            f"Round-trip zip is missing {len(missing)} expected entries (e.g. {sorted(missing)[:3]})."
        )
    for arcname, expected_bytes in expected_sizes.items():
        actual = sizes_in_zip.get(arcname)
        if actual != expected_bytes:
            raise RuntimeError(
                f"Round-trip zip entry {arcname!r} byte count differs: drive={actual}, expected={expected_bytes}."
            )

    roundtrip_path.unlink(missing_ok=True)


def _upload_zip_resumable(
    client: DriveClient,
    local_path: Path,
    parent_id: str,
    *,
    file_name: str,
    chunk_bytes: int = RESUMABLE_CHUNK_BYTES,
    progress_log_every: int = 5,
    local_md5: str | None = None,
) -> dict[str, Any]:
    """Upload a large zip via resumable chunked upload.

    Survives transient SSL drops and broken pipes mid-upload by retrying the
    same chunk (Google's resumable protocol acks bytes; we resume from the last
    ack point, not from byte 0).
    """
    if not local_path.exists():
        raise DriveClientError(f"Local file does not exist: {local_path}")

    total_bytes = local_path.stat().st_size

    class _RestartUploadSession(Exception):
        pass

    def completed_existing_upload() -> dict[str, Any] | None:
        existing = client.find_file_by_name(parent_id, file_name)
        if not existing or not existing.get("id"):
            return None
        meta = client.get_file(str(existing["id"]), fields="id,name,size,md5Checksum,mimeType,parents")
        if int(meta.get("size") or 0) != total_bytes:
            return None
        drive_md5 = str(meta.get("md5Checksum") or "")
        if local_md5 and drive_md5 and drive_md5.lower() != local_md5.lower():
            return None
        _log(f"  ...found completed prior upload for {file_name}; reusing Drive file {meta['id']}")
        return meta

    def build_request():
        metadata = {"name": file_name, "parents": [parent_id]}
        media = MediaFileUpload(
            str(local_path),
            mimetype="application/zip",
            resumable=True,
            chunksize=chunk_bytes,
        )
        return client.service.files().create(
            body=metadata,
            media_body=media,
            fields="id,name,size,md5Checksum,mimeType,parents",
            supportsAllDrives=True,
        )

    def is_retryable_upload_exception(exc: Exception) -> bool:
        if isinstance(exc, HttpError):
            return DriveClient._is_retryable_exception(exc)
        return isinstance(
            exc,
            (
                httplib2.HttpLib2Error,
                ssl.SSLError,
                socket.timeout,
                socket.gaierror,
                ConnectionError,
                BrokenPipeError,
                OSError,
            ),
        )

    prior = completed_existing_upload()
    if prior:
        return prior

    last_error: Exception | None = None
    for session_attempt in range(1, RESUMABLE_MAX_SESSIONS + 1):
        request = build_request()
        last_logged_pct = -1
        chunk_index = 0
        response = None

        try:
            while response is None:
                chunk_index += 1
                status = None
                for attempt in range(1, RESUMABLE_MAX_ATTEMPTS_PER_CHUNK + 1):
                    try:
                        # Use a fresh transport for each chunk attempt. Reusing
                        # httplib2's socket after BrokenPipe/SSL resets can make
                        # every retry fail immediately on the same dead connection.
                        http = client.fresh_authorized_http()
                        status, response = request.next_chunk(http=http)
                        break
                    except (
                        HttpError,
                        httplib2.HttpLib2Error,
                        ssl.SSLError,
                        socket.timeout,
                        socket.gaierror,
                        ConnectionError,
                        BrokenPipeError,
                        OSError,
                    ) as exc:
                        last_error = exc
                        if not is_retryable_upload_exception(exc):
                            raise DriveClientError(
                                f"Drive resumable upload hit non-retryable error for {file_name} "
                                f"chunk #{chunk_index}: {exc}"
                            ) from exc
                        if attempt >= RESUMABLE_MAX_ATTEMPTS_PER_CHUNK:
                            raise _RestartUploadSession(exc) from exc
                        backoff = min(120, 2 ** attempt)
                        _log(
                            f"  WARNING: chunk #{chunk_index} attempt {attempt} for {file_name} hit "
                            f"{type(exc).__name__}: {exc}; retrying in {backoff}s"
                        )
                        time.sleep(backoff)

                if status is not None and progress_log_every > 0:
                    uploaded = int(status.resumable_progress)
                    pct = int(100 * uploaded / max(1, total_bytes))
                    if pct >= last_logged_pct + progress_log_every:
                        _log(
                            f"  ...uploaded {uploaded / 1024 / 1024:.1f}/{total_bytes / 1024 / 1024:.1f} MB "
                            f"({pct}%)"
                        )
                        last_logged_pct = pct
        except _RestartUploadSession as exc:
            prior = completed_existing_upload()
            if prior:
                return prior
            if session_attempt >= RESUMABLE_MAX_SESSIONS:
                cause = exc.__cause__ or last_error or exc
                raise DriveClientError(
                    f"Drive resumable upload failed for {file_name} after "
                    f"{RESUMABLE_MAX_SESSIONS} upload sessions: {cause}"
                ) from cause
            _log(
                f"  WARNING: upload session {session_attempt}/{RESUMABLE_MAX_SESSIONS} for "
                f"{file_name} stalled; starting a fresh resumable session"
            )
            time.sleep(min(120, 10 * session_attempt))
            continue

        if not isinstance(response, dict) or not response.get("id"):
            raise DriveClientError(f"Drive resumable upload returned no id for {file_name}")
        return response

    raise DriveClientError(f"Drive resumable upload failed for {file_name}: {last_error}")


def _delete_originals(client: DriveClient, batch: list[TripletFolder], *, workers: int) -> int:
    counter_lock = threading.Lock()
    deleted = 0

    def delete_one(triplet: TripletFolder) -> bool:
        try:
            worker_client = client if workers <= 1 else _thread_local_client()
            worker_client.delete_file(triplet.folder_id)
            return True
        except DriveClientError as exc:
            if DriveClient.is_not_found_error(exc):
                _log(
                    f"  WARNING: original triplet {triplet.name!r} ({triplet.folder_id}) "
                    "was already gone; counting it as deleted."
                )
                return True
            _log(f"  WARNING: failed to delete original triplet {triplet.name!r} ({triplet.folder_id}): {exc}")
            return False

    if workers <= 1 or len(batch) <= 1:
        for triplet in batch:
            if delete_one(triplet):
                deleted += 1
        return deleted

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for ok in executor.map(delete_one, batch):
            if ok:
                with counter_lock:
                    deleted += 1
    return deleted


def process_batch(
    client: DriveClient,
    ctx: SiteContext,
    batch: list[TripletFolder],
    *,
    batch_index: int,
    total_batches: int,
    dry_run: bool,
    download_workers: int,
    delete_workers: int,
    failure_tolerance: float = DEFAULT_BATCH_FAILURE_TOLERANCE,
) -> BatchResult:
    batch_id = _build_batch_id(ctx.site_key, batch_index)

    if dry_run:
        _log(
            f"[{ctx.site_key}] batch {batch_index + 1}/{total_batches} [dry-run]: "
            f"would archive {len(batch)} triplets as {batch_id}.zip"
        )
        return BatchResult(
            site_key=ctx.site_key,
            batch_id=batch_id,
            triplet_count=len(batch),
            bytes_uploaded=0,
            items_freed=0,
        )

    if ctx.zips_folder_id is None:
        raise RuntimeError("zips_folder_id is required outside dry-run")

    hydrated, hydrate_failures = _hydrate_batch_files(client, batch, workers=download_workers)
    initial = len(batch)
    if hydrate_failures and len(hydrate_failures) > max(1, int(initial * failure_tolerance)):
        raise RuntimeError(
            f"Hydrate failures exceeded tolerance: {len(hydrate_failures)}/{initial} "
            f"(>{failure_tolerance:.0%}). Retry the script; the failed triplets stay on Drive."
        )
    if not hydrated:
        _log(f"[{ctx.site_key}] batch {batch_index + 1}/{total_batches}: no hydratable triplets, skipping.")
        return BatchResult(
            site_key=ctx.site_key,
            batch_id=batch_id,
            triplet_count=0,
            bytes_uploaded=0,
            items_freed=0,
        )

    with tempfile.TemporaryDirectory(prefix=f"{batch_id}_") as tmpdir:
        work_root = Path(tmpdir)
        contents_dir = work_root / "contents"
        contents_dir.mkdir(parents=True, exist_ok=True)

        _log(f"[{ctx.site_key}] batch {batch_index + 1}/{total_batches}: downloading {len(hydrated)} triplets ({download_workers} workers)...")
        manifest, archived_triplets, download_failures = _download_batch_to_local(
            client, hydrated, contents_dir, workers=download_workers
        )
        total_failures = len(hydrate_failures) + len(download_failures)
        if total_failures and total_failures > max(1, int(initial * failure_tolerance)):
            raise RuntimeError(
                f"Combined hydrate+download failures exceeded tolerance: "
                f"{total_failures}/{initial} (>{failure_tolerance:.0%}). "
                f"Retry the script; the failed triplets stay on Drive."
            )
        if not archived_triplets:
            _log(f"[{ctx.site_key}] batch {batch_index + 1}/{total_batches}: no triplets fully downloaded, skipping.")
            return BatchResult(
                site_key=ctx.site_key,
                batch_id=batch_id,
                triplet_count=0,
                bytes_uploaded=0,
                items_freed=0,
            )
        manifest["site_key"] = ctx.site_key
        manifest["batch_id"] = batch_id
        manifest["created_at"] = _utc_now().isoformat()

        manifest_path = contents_dir / MANIFEST_FILE_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        zip_path = work_root / f"{batch_id}.zip"
        _log(f"[{ctx.site_key}] batch {batch_index + 1}/{total_batches}: zipping...")
        zip_size, local_sha = _zip_local_dir(contents_dir, zip_path)

        md5 = hashlib.md5()
        with zip_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                md5.update(chunk)
        local_md5 = md5.hexdigest()

        _log(
            f"[{ctx.site_key}] batch {batch_index + 1}/{total_batches}: uploading "
            f"{zip_size / 1024 / 1024:.1f} MB to unassociated_zips/ (resumable, "
            f"{RESUMABLE_CHUNK_BYTES // 1024 // 1024} MB chunks)..."
        )
        uploaded = _upload_zip_resumable(
            client,
            zip_path,
            ctx.zips_folder_id,
            file_name=zip_path.name,
            local_md5=local_md5,
        )
        uploaded_id = str(uploaded["id"])

        expected_arcnames: set[str] = {MANIFEST_FILE_NAME}
        expected_sizes: dict[str, int] = {}
        for triplet_meta in manifest["triplets"]:
            for frame_meta in triplet_meta["frames"]:
                arc = f"{triplet_meta['sanitized_path']}/{frame_meta['frame_name']}"
                expected_arcnames.add(arc)
                expected_sizes[arc] = int(frame_meta["bytes"])

        _log(f"[{ctx.site_key}] batch {batch_index + 1}/{total_batches}: verifying upload (size + md5 + roundtrip + manifest)...")
        _verify_uploaded_zip(
            client,
            uploaded_id,
            expected_size=zip_size,
            local_md5=local_md5,
            local_sha256=local_sha,
            expected_arcnames=expected_arcnames,
            expected_sizes=expected_sizes,
            work_root=work_root,
        )

        log_entry = {
            "schema_version": SCHEMA_VERSION,
            "site_key": ctx.site_key,
            "batch_id": batch_id,
            "zip_file_id": uploaded_id,
            "zip_bytes": zip_size,
            "zip_sha256": local_sha,
            "zip_md5": local_md5,
            "triplet_count": len(archived_triplets),
            "triplet_ids": [t.folder_id for t in archived_triplets],
            "skipped_triplet_ids": [t.folder_id for t, _ in (hydrate_failures + download_failures)],
            "created_at": manifest["created_at"],
        }
        _log(f"[{ctx.site_key}] batch {batch_index + 1}/{total_batches}: appending manifest log entry...")
        append_manifest_entry(client, ctx, log_entry)

        _log(
            f"[{ctx.site_key}] batch {batch_index + 1}/{total_batches}: deleting "
            f"{len(archived_triplets)} archived triplet folders ({delete_workers} workers)..."
        )
        deleted = _delete_originals(client, archived_triplets, workers=delete_workers)
        items_freed = sum(1 + len(t.frame_files) for t in archived_triplets[:deleted])

    skipped_total = len(hydrate_failures) + len(download_failures)
    _log(
        f"[{ctx.site_key}] batch {batch_index + 1}/{total_batches}: done. "
        f"zipped {len(archived_triplets)} triplets, skipped {skipped_total} (left on Drive for retry), "
        f"deleted {deleted} originals, freed ~{items_freed} items."
    )

    return BatchResult(
        site_key=ctx.site_key,
        batch_id=batch_id,
        triplet_count=len(archived_triplets),
        bytes_uploaded=zip_size,
        items_freed=items_freed,
    )


# ---------------------------------------------------------------------------
# Site-level driver
# ---------------------------------------------------------------------------


def chunked(items: list[TripletFolder], size: int) -> Iterable[list[TripletFolder]]:
    for offset in range(0, len(items), size):
        yield items[offset : offset + size]


def run_for_site(
    client: DriveClient,
    site_key: str,
    *,
    mode: str,
    batch_size: int,
    limit_batches: int | None,
    min_age_minutes: int,
    download_workers: int,
    delete_workers: int,
) -> list[BatchResult]:
    dry_run = mode == "dry-run"
    test_batch = mode == "test-batch"
    effective_batch_size = min(batch_size, TEST_BATCH_CAP) if test_batch else batch_size

    _log(f"[{site_key}] resolving site root...")
    ctx = resolve_site_context(client, site_key, dry_run=dry_run)
    _log(f"[{site_key}] site root id={ctx.site_root_id}, unassociated id={ctx.unassociated_id}")

    archived_ids: set[str] = set()
    if ctx.zips_folder_id:
        _log(f"[{site_key}] reading manifest log to skip already-archived triplets...")
        archived_ids = load_archived_triplet_ids(client, ctx)
        _log(f"[{site_key}] {len(archived_ids)} triplet ids already archived in prior runs.")

    triplets = list_eligible_triplets(
        client,
        ctx,
        min_age_minutes=min_age_minutes,
        archived_ids=archived_ids,
    )
    _log(
        f"[{site_key}] found {len(triplets)} eligible triplets "
        f"(after age>{min_age_minutes}min and archive-skip filters)."
    )
    if not triplets:
        return []

    batches = list(chunked(triplets, effective_batch_size))
    if test_batch:
        batches = batches[:1]
    elif limit_batches is not None:
        batches = batches[: max(0, limit_batches)]

    results: list[BatchResult] = []
    for index, batch in enumerate(batches):
        result = process_batch(
            client,
            ctx,
            batch,
            batch_index=index,
            total_batches=len(batches),
            dry_run=dry_run,
            download_workers=download_workers,
            delete_workers=delete_workers,
        )
        results.append(result)
        if test_batch and not dry_run:
            _log(f"[{site_key}] test-batch complete; stopping. Rerun with --mode confirm for full run.")
            break
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=("dry-run", "test-batch", "confirm"),
        required=True,
        help="dry-run = read-only counts; test-batch = one batch end-to-end; confirm = full run.",
    )
    parser.add_argument(
        "--site",
        action="append",
        default=None,
        help=(
            "Restrict to a specific Reolink site_key (e.g. restaurant-pi-1). "
            "Can be passed multiple times. Defaults to all configured Reolink sites."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument(
        "--min-age-minutes",
        type=int,
        default=DEFAULT_MIN_AGE_MINUTES,
        help="Skip triplets modified within this many minutes (race guard). Default 60.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=DEFAULT_DOWNLOAD_WORKERS,
        help=f"Concurrent download/list_files workers per batch. Default {DEFAULT_DOWNLOAD_WORKERS}.",
    )
    parser.add_argument(
        "--delete-workers",
        type=int,
        default=DEFAULT_DELETE_WORKERS,
        help=f"Concurrent delete workers per batch. Default {DEFAULT_DELETE_WORKERS}.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.site:
        target_sites = args.site
    else:
        target_sites = [site.site_key for site in label_app.REOLINK_SITES]

    if not target_sites:
        _log("No Reolink sites configured; nothing to do.")
        return 0

    _log(
        f"mode={args.mode}, sites={target_sites}, batch_size={args.batch_size}, "
        f"min_age_minutes={args.min_age_minutes}, download_workers={args.download_workers}, "
        f"delete_workers={args.delete_workers}"
    )
    client = DriveClient()

    if args.mode != "dry-run":
        wipe_temp_processing(client, dry_run=False)
    else:
        wipe_temp_processing(client, dry_run=True)

    all_results: list[BatchResult] = []
    for site_key in target_sites:
        try:
            site_results = run_for_site(
                client,
                site_key,
                mode=args.mode,
                batch_size=args.batch_size,
                limit_batches=args.limit_batches,
                min_age_minutes=args.min_age_minutes,
                download_workers=args.download_workers,
                delete_workers=args.delete_workers,
            )
        except Exception as exc:
            _log(f"[{site_key}] FAILED: {exc}")
            raise
        all_results.extend(site_results)

    _log("=== summary ===")
    if not all_results:
        _log("No batches produced.")
        return 0
    by_site: dict[str, list[BatchResult]] = {}
    for r in all_results:
        by_site.setdefault(r.site_key, []).append(r)
    for site_key, results in by_site.items():
        triplets = sum(r.triplet_count for r in results)
        items_freed = sum(r.items_freed for r in results)
        bytes_uploaded = sum(r.bytes_uploaded for r in results)
        _log(
            f"[{site_key}] batches={len(results)}, triplets_archived={triplets}, "
            f"items_freed~{items_freed}, bytes_uploaded={bytes_uploaded / 1024 / 1024:.1f} MB"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
