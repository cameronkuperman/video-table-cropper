#!/usr/bin/env python3
"""Offline GPU worker that converts Drive videos into review-ready grouped samples."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from camera_table_metadata import (
    approved_tables_path,
    build_static_tables_for_frame,
    detect_camera_from_filename,
    get_normalized_camera_config,
)
from db import database_enabled
from env_loader import load_local_env
from dataset_schema import (
    GENERATOR_VERSION,
    PERCEPTION_NPZ_NAME,
    PREVIEW_ANCHOR_NAME,
    PREVIEW_PREV_1_NAME,
    PREVIEW_PREV_2_NAME,
    SAMPLE_JSON_NAME,
    SAMPLE_SCHEMA_VERSION,
    TIGHT_ANCHOR_NAME,
    VECTOR_CONTRACT_VERSION,
    VECTOR_DIM,
    build_sample_id,
    slugify,
)
from drive_client import DriveClient, DriveClientError
from drive_roots import resolve_video_pipeline_roots
from extract_frames import extract_frames
from sample_builder import (
    bbox_with_padding,
    crop_image,
    encode_mask_rle,
    preview_padding_for_bbox,
    write_review_bundle,
)
from segment_cropper import crop_with_polygonal_rect
from sam3_adapter import Sam3Adapter, Sam3AdapterError
from tracking import assign_track_ids
from video_review_store_pg import VideoReviewStorePG
from worker_state_store_pg import WorkerStateStorePG
from worker_runtime import default_worker_runtime_state, worker_runtime_status_path, write_worker_runtime_state

load_local_env()

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def is_video_file(file_meta: dict[str, Any]) -> bool:
    name = str(file_meta.get("name", "")).lower()
    return Path(name).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def is_ignored_source_folder(folder_meta: dict[str, Any]) -> bool:
    name = str(folder_meta.get("name", "")).strip()
    return name.startswith("_")


def parse_csv_values(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def normalize_source_folder_selector(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def expanded_mask_intersection(person_mask: np.ndarray, expanded_mask: np.ndarray) -> np.ndarray:
    return np.logical_and(person_mask.astype(bool), expanded_mask.astype(bool))


def bbox_intersects_mask(bbox_xyxy: tuple[float, float, float, float], mask: np.ndarray) -> bool:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox_xyxy]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(mask.shape[1], x2)
    y2 = min(mask.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return False
    return bool(mask[y1:y2, x1:x2].any())


def _serialize_rect_payload(rect: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(rect, dict):
        return None

    payload = {
        "center_x": round(float(rect["center_x"]), 1),
        "center_y": round(float(rect["center_y"]), 1),
        "width_px": round(float(rect["width_px"]), 1),
        "height_px": round(float(rect["height_px"]), 1),
        "angle_deg": round(float(rect["angle_deg"]), 1),
    }
    polygon = rect.get("polygon")
    if isinstance(polygon, list):
        payload["polygon"] = [[int(round(float(x))), int(round(float(y)))] for x, y in polygon]
    else:
        payload["polygon"] = []
    return payload


def _table_bbox_xywh(table: dict[str, Any]) -> list[int]:
    bbox_xywh = table.get("bbox_xywh")
    if isinstance(bbox_xywh, list) and len(bbox_xywh) >= 4:
        return [int(round(float(value))) for value in bbox_xywh[:4]]

    x1, y1, x2, y2 = table["bbox_xyxy"]
    return [int(x1), int(y1), int(max(x2 - x1, 0)), int(max(y2 - y1, 0))]


def summarize_source_folders(client: DriveClient, source_root_id: str) -> list[dict[str, Any]]:
    folders = client.list_folders_recursive(source_root_id, include_root=True)
    summaries: list[dict[str, Any]] = []
    folders_sorted = sorted(folders, key=lambda folder: str(folder.get("name", folder.get("id", ""))))
    for folder in folders_sorted:
        if is_ignored_source_folder(folder):
            continue
        folder_id = str(folder["id"])
        folder_name = str(folder.get("name", folder_id)).strip()
        files = sorted(client.list_files(folder_id), key=lambda file_meta: str(file_meta.get("name", "")))
        video_files = [file_meta for file_meta in files if file_meta.get("mimeType") != "application/vnd.google-apps.folder" and is_video_file(file_meta)]
        supported_videos = []
        unsupported_videos = []
        for file_meta in video_files:
            video_name = str(file_meta.get("name", ""))
            camera_id = detect_camera_from_filename(video_name)
            if camera_id and get_normalized_camera_config(camera_id):
                supported_videos.append(video_name)
            else:
                unsupported_videos.append(video_name)
        summaries.append(
            {
                "id": folder_id,
                "name": folder_name,
                "video_count": len(video_files),
                "sample_video_names": [str(file_meta.get("name", "")) for file_meta in video_files[:5]],
                "supported_video_count": len(supported_videos),
                "sample_supported_video_names": supported_videos[:5],
                "unsupported_video_count": len(unsupported_videos),
                "sample_unsupported_video_names": unsupported_videos[:5],
            }
        )
    return summaries


class VideoDatasetWorker:
    def __init__(
        self,
        client: DriveClient,
        adapter: Sam3Adapter,
        source_root_id: str,
        review_root_id: str,
        cache_dir: Path,
        frame_interval: int = 10,
        expansion_strength: float = 1.0,
        force_reprocess: bool = False,
        max_videos: int | None = None,
        include_source_folder_names: list[str] | None = None,
        max_samples_per_run: int | None = None,
        max_pending_samples: int | None = None,
        resume_pending_samples: int | None = None,
        poll_seconds: int = 60,
        continuous: bool = False,
        trash_source_videos: bool = False,
        cleanup_frames: bool = True,
        cleanup_review_cache: bool = True,
        cleanup_local_video_when_source_trashed: bool = True,
        review_store: VideoReviewStorePG | None = None,
        worker_state_store: WorkerStateStorePG | None = None,
    ) -> None:
        self.client = client
        self.adapter = adapter
        self.source_root_id = source_root_id
        self.review_root_id = review_root_id
        self.cache_dir = cache_dir
        self.frame_interval = frame_interval
        self.expansion_strength = expansion_strength
        self.force_reprocess = force_reprocess
        self.max_videos = max_videos
        self.include_source_folder_names = {name.strip() for name in (include_source_folder_names or []) if name.strip()}
        self.include_source_folder_selectors = {
            normalize_source_folder_selector(name) for name in self.include_source_folder_names
        }
        self.max_samples_per_run = max_samples_per_run
        self.max_pending_samples = max_pending_samples
        self.resume_pending_samples = (
            min(resume_pending_samples, max_pending_samples)
            if resume_pending_samples and max_pending_samples
            else resume_pending_samples
        )
        self.poll_seconds = max(5, poll_seconds)
        self.continuous = continuous
        self.trash_source_videos = bool(trash_source_videos)
        self.cleanup_frames = bool(cleanup_frames)
        self.cleanup_review_cache = bool(cleanup_review_cache)
        self.cleanup_local_video_when_source_trashed = bool(cleanup_local_video_when_source_trashed)
        self.review_store = review_store
        self.worker_state_store = worker_state_store
        self._static_table_cache: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
        self.pending_root_id = self.client.ensure_subfolder(review_root_id, "pending")
        self.worker_state_dir = self.cache_dir / "_worker_state"
        self.worker_state_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = worker_runtime_status_path(self.cache_dir)
        self.status_seq = 0
        self.videos_seen_count = 0
        self.videos_selected_count = 0
        self.videos_processed_count = 0
        self.videos_skipped_count = 0
        self.samples_created_total = 0
        self.last_pending_sample_count = 0
        self.last_source_folder_summaries: list[dict[str, Any]] = []
        self.last_error: str | None = None
        self.stop_reason: str | None = None
        self._cleanup_startup_cache()
        self.last_pending_sample_count = self._count_pending_samples()
        self._emit_status("idle", "Worker initialized and ready to scan Drive.", worker_running=True)

    def _cache_dir_for_video(self, video_name: str) -> Path:
        return self.cache_dir / slugify(Path(video_name).stem)

    def _current_video_payload(self, video_meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": video_meta.get("id"),
            "name": video_meta.get("name"),
            "camera_id": detect_camera_from_filename(str(video_meta.get("name", ""))) or "unknown_camera",
            "source_folder_id": video_meta.get("source_folder_id"),
            "source_folder_name": video_meta.get("source_folder_name"),
        }

    def _static_tables_for_shape(self, camera_id: str, image_shape: tuple[int, int]) -> list[dict[str, Any]]:
        cache_key = (camera_id, int(image_shape[0]), int(image_shape[1]))
        if cache_key not in self._static_table_cache:
            self._static_table_cache[cache_key] = build_static_tables_for_frame(
                camera_id,
                image_shape,
                expansion_strength=self.expansion_strength,
            )

        cached_tables = self._static_table_cache[cache_key]
        return [dict(table) for table in cached_tables]

    def _folder_matches_filter(self, folder_name: str, folder_id: str) -> bool:
        if not self.include_source_folder_selectors:
            return True
        normalized_name = normalize_source_folder_selector(folder_name)
        normalized_id = str(folder_id).strip().lower()
        return normalized_name in self.include_source_folder_selectors or normalized_id in self.include_source_folder_selectors

    def _camera_id_for_video(self, video_meta: dict[str, Any]) -> str | None:
        return detect_camera_from_filename(str(video_meta.get("name", "")))

    def _video_supported_for_processing(self, video_meta: dict[str, Any]) -> tuple[bool, str]:
        video_name = str(video_meta.get("name", ""))
        camera_id = self._camera_id_for_video(video_meta)
        if not camera_id:
            return False, f"{video_name}: could not infer camera id from filename"
        if get_normalized_camera_config(camera_id) is None:
            return False, f"{video_name}: no approved table metadata for {camera_id}"
        return True, camera_id

    def _emit_status(
        self,
        state: str,
        message: str,
        *,
        current_video: dict[str, Any] | None = None,
        worker_running: bool | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.status_seq += 1
        payload = default_worker_runtime_state()
        payload.update(
            {
                "event_seq": self.status_seq,
                "state": state,
                "message": message,
                "worker_running": state not in {"completed", "error"} if worker_running is None else bool(worker_running),
                "updated_at_epoch": int(time.time()),
                "current_video": current_video,
                "last_error": self.last_error,
                "stop_reason": self.stop_reason,
                "counters": {
                    "videos_seen": self.videos_seen_count,
                    "videos_selected": self.videos_selected_count,
                    "videos_processed": self.videos_processed_count,
                    "videos_skipped": self.videos_skipped_count,
                    "sample_count_created": self.samples_created_total,
                    "pending_sample_count": self.last_pending_sample_count,
                },
                "config": {
                    "frame_interval": self.frame_interval,
                    "continuous": self.continuous,
                    "poll_seconds": self.poll_seconds,
                    "max_videos": self.max_videos,
                    "max_samples_per_run": self.max_samples_per_run,
                    "max_pending_samples": self.max_pending_samples,
                    "resume_pending_samples": self.resume_pending_samples,
                    "trash_source_videos": self.trash_source_videos,
                    "cleanup_frames": self.cleanup_frames,
                    "cleanup_review_cache": self.cleanup_review_cache,
                    "cleanup_local_video_when_source_trashed": self.cleanup_local_video_when_source_trashed,
                },
            }
        )
        if extra:
            payload.update(extra)
        if self.worker_state_store is not None:
            self.worker_state_store.write_status(payload)
            return
        write_worker_runtime_state(self.status_path, payload)

    def _remove_dir_if_present(self, path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)

    def _remove_file_if_present(self, path: Path) -> None:
        if path.exists():
            path.unlink()

    def _maybe_remove_empty_cache_dir(self, video_cache_dir: Path) -> None:
        if not video_cache_dir.exists():
            return
        if any(video_cache_dir.iterdir()):
            return
        video_cache_dir.rmdir()

    def _cleanup_video_artifacts(self, video_cache_dir: Path) -> None:
        if self.cleanup_frames:
            self._remove_dir_if_present(video_cache_dir / "frames")
        if self.cleanup_review_cache:
            self._remove_dir_if_present(video_cache_dir / "review")
        self._maybe_remove_empty_cache_dir(video_cache_dir)

    def _cleanup_local_video_cache_if_allowed(self, video_meta: dict[str, Any], *, source_video_trashed: bool) -> None:
        if not source_video_trashed or not self.cleanup_local_video_when_source_trashed:
            return
        video_cache_dir = self._cache_dir_for_video(str(video_meta["name"]))
        local_video_path = video_cache_dir / str(video_meta["name"])
        self._remove_file_if_present(local_video_path)
        self._maybe_remove_empty_cache_dir(video_cache_dir)

    def _cleanup_startup_cache(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for child in self.cache_dir.iterdir():
            if not child.is_dir() or child.name == self.worker_state_dir.name:
                continue
            if self.cleanup_frames:
                self._remove_dir_if_present(child / "frames")
            if self.cleanup_review_cache:
                self._remove_dir_if_present(child / "review")
            self._maybe_remove_empty_cache_dir(child)

        if not self.cleanup_local_video_when_source_trashed or self.worker_state_store is not None:
            return

        for marker_path in self.worker_state_dir.glob("*.json"):
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not marker.get("source_video_trashed"):
                continue
            video_name = str(marker.get("video_name") or "").strip()
            if not video_name:
                continue
            video_cache_dir = self._cache_dir_for_video(video_name)
            self._remove_file_if_present(video_cache_dir / video_name)
            self._maybe_remove_empty_cache_dir(video_cache_dir)

    def list_source_videos(self) -> list[dict[str, Any]]:
        folder_summaries = summarize_source_folders(self.client, self.source_root_id)
        self.last_source_folder_summaries = [
            {
                **summary,
                "matched_filter": self._folder_matches_filter(summary["name"], summary["id"]),
            }
            for summary in folder_summaries
        ]
        videos: list[dict[str, Any]] = []
        for summary in self.last_source_folder_summaries:
            folder_name = summary["name"]
            folder_id = summary["id"]
            if not self._folder_matches_filter(folder_name, folder_id):
                continue
            files = sorted(self.client.list_files(folder_id), key=lambda file_meta: str(file_meta.get("name", "")))
            for file_meta in files:
                if file_meta.get("mimeType") == "application/vnd.google-apps.folder":
                    continue
                if is_video_file(file_meta):
                    enriched = dict(file_meta)
                    enriched["source_folder_id"] = folder_id
                    enriched["source_folder_name"] = folder_name
                    videos.append(enriched)
        return videos

    def _sample_folder_exists(self, table_parent_id: str, sample_id: str) -> bool:
        existing = self.client.find_file_by_name(table_parent_id, sample_id, mime_type="application/vnd.google-apps.folder")
        if not existing:
            return False
        sample_json = self.client.find_file_by_name(existing["id"], SAMPLE_JSON_NAME)
        return sample_json is not None

    def _video_marker_path(self, video_id: str) -> Path:
        return self.worker_state_dir / f"{video_id}.json"

    def _is_video_marked_processed(self, video_id: str) -> bool:
        if self.worker_state_store is not None:
            return self.worker_state_store.is_video_processed(video_id)
        return self._video_marker_path(video_id).exists()

    def _mark_video_processed(
        self,
        video_meta: dict[str, Any],
        created_samples: int,
        *,
        source_video_trashed: bool = False,
    ) -> None:
        if self.worker_state_store is not None:
            self.worker_state_store.mark_video_processed(
                video_meta,
                created_samples,
                source_video_trashed=source_video_trashed,
            )
            return
        payload = {
            "video_id": video_meta["id"],
            "video_name": video_meta["name"],
            "source_folder_id": video_meta.get("source_folder_id"),
            "source_folder_name": video_meta.get("source_folder_name"),
            "created_samples": created_samples,
            "source_video_trashed": source_video_trashed,
            "cache_dir_name": self._cache_dir_for_video(str(video_meta["name"])).name,
            "processed_at_epoch": int(time.time()),
        }
        self._video_marker_path(video_meta["id"]).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _maybe_trash_source_video(self, video_meta: dict[str, Any], created_samples: int) -> bool:
        if not self.trash_source_videos or created_samples <= 0:
            return False
        self.client.trash_file(video_meta["id"])
        print(f"[worker] trashed source video after processing: {video_meta['name']}")
        return True

    def _count_pending_samples(self) -> int:
        if self.review_store is not None:
            total = self.review_store.count_pending_items()
            self.last_pending_sample_count = total
            return total
        total = 0
        for folder in self.client.list_folders_recursive(self.pending_root_id, include_root=False):
            if self.client.find_file_by_name(folder["id"], SAMPLE_JSON_NAME):
                total += 1
        self.last_pending_sample_count = total
        return total

    def _resume_threshold(self) -> int:
        if self.resume_pending_samples is not None and self.resume_pending_samples >= 0:
            return self.resume_pending_samples
        if self.max_pending_samples is None:
            return 0
        return max(0, self.max_pending_samples // 2)

    def _maybe_wait_for_pending_capacity(self) -> bool:
        if self.max_pending_samples is None or self.max_pending_samples <= 0:
            return True

        pending_samples = self._count_pending_samples()
        if pending_samples < self.max_pending_samples:
            return True

        if not self.continuous:
            self.stop_reason = (
                f"Pending review queue reached {pending_samples} samples "
                f"(limit {self.max_pending_samples})."
            )
            self._emit_status(
                "waiting_for_review_capacity",
                self.stop_reason,
                worker_running=False,
            )
            return False

        resume_threshold = self._resume_threshold()
        while pending_samples > resume_threshold:
            self._emit_status(
                "waiting_for_review_capacity",
                (
                    f"Pending review queue is full at {pending_samples} samples. "
                    f"Waiting until it drops to {resume_threshold}."
                ),
                worker_running=True,
            )
            print(
                json.dumps(
                    {
                        "status": "waiting_for_review_capacity",
                        "pending_samples": pending_samples,
                        "resume_when_at_or_below": resume_threshold,
                        "poll_seconds": self.poll_seconds,
                    }
                )
            )
            time.sleep(self.poll_seconds)
            pending_samples = self._count_pending_samples()

        self._emit_status(
            "ready_to_process",
            f"Pending review queue dropped to {pending_samples}; resuming processing.",
            worker_running=True,
        )
        return True

    def process_all(self) -> dict[str, Any]:
        processed: list[str] = []
        skipped: list[str] = []
        errors: list[str] = []
        self.stop_reason = None
        self.last_error = None
        self._emit_status("listing_source_videos", "Scanning Drive source folders for videos.", worker_running=True)
        all_videos = self.list_source_videos()
        self.videos_seen_count = len(all_videos)
        supported_videos: list[dict[str, Any]] = []
        unsupported_messages: list[str] = []
        for video_meta in all_videos:
            supported, detail = self._video_supported_for_processing(video_meta)
            if supported:
                supported_videos.append(video_meta)
            else:
                unsupported_messages.append(detail)

        if unsupported_messages:
            self.videos_skipped_count += len(unsupported_messages)
            preview = "; ".join(unsupported_messages[:5])
            if len(unsupported_messages) > 5:
                preview += f"; ... (+{len(unsupported_messages) - 5} more)"
            print(f"[worker] skipping unsupported videos: {preview}")

        if not all_videos and self.include_source_folder_names:
            matched_folders = [
                summary for summary in self.last_source_folder_summaries if summary.get("matched_filter")
            ]
            if matched_folders:
                matched_names = ", ".join(
                    f"{summary['name']} ({summary['video_count']} videos)" for summary in matched_folders[:10]
                )
                self.stop_reason = (
                    "Matched source folder filters but found no video files to process: "
                    f"{matched_names}."
                )
            else:
                visible_folders = [
                    summary for summary in self.last_source_folder_summaries if summary.get("video_count", 0) > 0
                ]
                visible_names = ", ".join(summary["name"] for summary in visible_folders[:15]) or "none"
                self.stop_reason = (
                    "No source folders matched the requested filter(s): "
                    f"{', '.join(sorted(self.include_source_folder_names))}. "
                    f"Folders with videos under the current Drive source root: {visible_names}."
                )
            print(f"[worker] {self.stop_reason}")
        if all_videos and not supported_videos:
            self.stop_reason = (
                "Discovered source videos, but none have approved table metadata for processing. "
                f"First examples: {', '.join(unsupported_messages[:5])}"
            )
            print(f"[worker] {self.stop_reason}")

        videos = supported_videos
        if self.max_videos is not None and self.max_videos > 0:
            videos = videos[: self.max_videos]
        self.videos_selected_count = len(videos)
        self._emit_status(
            "ready_to_process",
            f"Selected {len(videos)} supported videos from {len(all_videos)} found in Drive.",
            worker_running=True,
        )
        for video_meta in videos:
            if not self._maybe_wait_for_pending_capacity():
                break
            if self._is_video_marked_processed(video_meta["id"]) and not self.force_reprocess:
                skipped.append(f"{video_meta['name']}: already processed")
                self.videos_skipped_count += 1
                print(f"[worker] skipping already processed video: {video_meta['name']}")
                continue
            try:
                current_video = self._current_video_payload(video_meta)
                self._emit_status(
                    "processing_video",
                    f"Processing {video_meta['name']}.",
                    current_video=current_video,
                    worker_running=True,
                )
                print(f"[worker] processing video: {video_meta['name']}")
                created = self.process_video(video_meta)
                self.videos_processed_count += 1
                processed.append(f"{video_meta['name']}: {created} samples")
                print(f"[worker] completed video: {video_meta['name']} ({created} samples)")
                if not self.stop_reason:
                    source_video_trashed = False
                    try:
                        source_video_trashed = self._maybe_trash_source_video(video_meta, created)
                    except DriveClientError as exc:
                        errors.append(f"{video_meta['name']}: source video trash failed: {exc}")
                        print(f"[worker] warning: could not trash source video {video_meta['name']}: {exc}")
                    self._mark_video_processed(
                        video_meta,
                        created,
                        source_video_trashed=source_video_trashed,
                    )
                    self._cleanup_local_video_cache_if_allowed(
                        video_meta,
                        source_video_trashed=source_video_trashed,
                    )
                self._emit_status(
                    "ready_to_process",
                    f"Completed {video_meta['name']} with {created} samples.",
                    worker_running=True,
                )
            except Exception as exc:  # pragma: no cover - runtime path
                errors.append(f"{video_meta['name']}: {exc}")
                self.last_error = str(exc)
                self._emit_status(
                    "error",
                    f"Error processing {video_meta['name']}: {exc}",
                    current_video=self._current_video_payload(video_meta),
                    worker_running=True,
                )
                print(f"[worker] error processing video {video_meta['name']}: {exc}")
                traceback.print_exc()
                continue
            if self.stop_reason:
                print(f"[worker] stopping early: {self.stop_reason}")
                break
        result = {
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
            "video_count_seen": len(all_videos),
            "video_count_selected": len(videos),
            "source_folder_filter": sorted(self.include_source_folder_names),
            "max_videos": self.max_videos,
            "sample_count_created": self.samples_created_total,
            "pending_sample_count": self._count_pending_samples(),
            "stop_reason": self.stop_reason,
            "continuous": self.continuous,
        }
        final_message = self.stop_reason or (
            f"Worker run completed. Processed {self.videos_processed_count} videos and created {self.samples_created_total} samples."
        )
        self._emit_status("completed", final_message, worker_running=False, extra={"result": result})
        return result

    def process_video(self, video_meta: dict[str, Any]) -> int:
        video_id = video_meta["id"]
        video_name = video_meta["name"]
        camera_id = detect_camera_from_filename(video_name) or "unknown_camera"
        video_cache_dir = self._cache_dir_for_video(video_name)
        frames_dir = video_cache_dir / "frames"
        review_dir = video_cache_dir / "review"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        if review_dir.exists():
            shutil.rmtree(review_dir)
        review_dir.mkdir(parents=True, exist_ok=True)

        local_video_path = video_cache_dir / video_name
        if not local_video_path.exists() or self.force_reprocess:
            video_cache_dir.mkdir(parents=True, exist_ok=True)
            self._emit_status(
                "downloading_video",
                f"Downloading {video_name} from Drive.",
                current_video=self._current_video_payload(video_meta),
                worker_running=True,
            )
            self.client.download_file_to_path(video_id, local_video_path)

        self._emit_status(
            "extracting_frames",
            f"Extracting frames for {video_name}.",
            current_video=self._current_video_payload(video_meta),
            worker_running=True,
        )
        success, frames_info = extract_frames(
            local_video_path,
            frames_dir,
            interval=self.frame_interval,
            quality=2,
            format="jpg",
            resume=not self.force_reprocess,
            verbose=False,
        )
        if not success:
            raise RuntimeError("Frame extraction failed")

        self._emit_status(
            "detecting_frames",
            f"Running detections on {len(frames_info)} frames for {video_name}.",
            current_video=self._current_video_payload(video_meta),
            worker_running=True,
        )
        frame_records = self._process_frames(frames_dir, frames_info, camera_id=camera_id)
        if not frame_records:
            print(f"[worker] no frame records produced for {video_name}")
            self._cleanup_video_artifacts(video_cache_dir)
            return 0
        if len(frame_records) < 4 or max(int(frame["timestamp_seconds"]) for frame in frame_records) < 30:
            print(
                f"[worker] video too short for a 30-second training bundle: "
                f"{video_name} ({len(frame_records)} extracted frames)"
            )
            self._cleanup_video_artifacts(video_cache_dir)
            return 0

        table_frames = [frame["tables"] for frame in frame_records]
        person_frames = [frame["people"] for frame in frame_records]
        frame_shape = frame_records[0]["image_shape"]
        assign_track_ids(person_frames, kind="person", frame_shape=frame_shape)

        created = 0
        camera_folder_id = self.client.ensure_subfolder(self.pending_root_id, slugify(camera_id))
        video_folder_id = self.client.ensure_subfolder(camera_folder_id, slugify(Path(video_name).stem))
        self._emit_status(
            "uploading_samples",
            f"Building and uploading review bundles for {video_name}.",
            current_video=self._current_video_payload(video_meta),
            worker_running=True,
        )

        for anchor_idx in range(3, len(frame_records)):
            anchor_time = int(frame_records[anchor_idx]["timestamp_seconds"])
            if anchor_time % 30 != 0:
                continue

            triplet = frame_records[anchor_idx - 2 : anchor_idx + 1]
            if [frame["timestamp_seconds"] for frame in triplet] != [anchor_time - 20, anchor_time - 10, anchor_time]:
                continue

            anchor_tables = {table["track_id"]: table for table in triplet[-1]["tables"]}
            previous_tables = [{table["track_id"]: table for table in frame["tables"]} for frame in triplet[:-1]]

            for table_track_id, anchor_table in anchor_tables.items():
                if any(table_track_id not in frame_tables for frame_tables in previous_tables):
                    continue

                sample_id = build_sample_id(camera_id, video_name, table_track_id, anchor_time)
                table_parent_id = self.client.ensure_subfolder(video_folder_id, slugify(table_track_id))
                if not self.force_reprocess and self._sample_folder_exists(table_parent_id, sample_id):
                    continue

                sample_folder = review_dir / sample_id
                sample_folder.mkdir(parents=True, exist_ok=True)
                sample_payload, perception_payload, review_images = self._build_sample_bundle(
                    video_meta=video_meta,
                    camera_id=camera_id,
                    triplet=triplet,
                    table_track_id=table_track_id,
                )
                write_review_bundle(sample_folder, sample_payload, review_images, perception_payload)
                upload_refs = self._upload_review_bundle(table_parent_id, sample_id, sample_folder)
                if self.review_store is not None:
                    self.review_store.upsert_queue_item(
                        {
                            "sample_id": sample_payload["sample_id"],
                            "sample_folder_id": upload_refs["sample_folder_id"],
                            "sample_folder_name": sample_id,
                            "review_root_folder_id": self.review_root_id,
                            "source_parent_folder_id": table_parent_id,
                            "source_video_drive_file_id": video_meta["id"],
                            "source_video_name": video_meta["name"],
                            "camera_id": camera_id,
                            "table_track_id": str(table_track_id),
                            "anchor_time_seconds": int(sample_payload["timing"]["anchor_time_seconds"]),
                            "preview_anchor_file_id": upload_refs.get("preview_anchor_file_id"),
                            "preview_t_minus_10_file_id": upload_refs.get("preview_t_minus_10_file_id"),
                            "preview_t_minus_20_file_id": upload_refs.get("preview_t_minus_20_file_id"),
                            "tight_anchor_file_id": upload_refs.get("tight_anchor_file_id"),
                            "perception_file_id": upload_refs.get("perception_file_id"),
                            "sample": sample_payload,
                        }
                    )
                created += 1
                self.samples_created_total += 1
                if self.max_samples_per_run is not None and self.max_samples_per_run > 0:
                    if self.samples_created_total >= self.max_samples_per_run:
                        self.stop_reason = (
                            f"Created {self.samples_created_total} samples "
                            f"(limit {self.max_samples_per_run})."
                        )
                        return created

        if created == 0:
            print(f"[worker] video produced 0 review samples: {video_name}")
        self._cleanup_video_artifacts(video_cache_dir)
        return created

    def _process_frames(
        self,
        frames_dir: Path,
        frames_info: list[dict[str, Any]],
        *,
        camera_id: str,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for frame_info in frames_info:
            image_path = frames_dir / frame_info["filename"]
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                try:
                    tables = self._static_tables_for_shape(camera_id, (image.height, image.width))
                except KeyError as exc:
                    raise RuntimeError(
                        f"No approved table metadata found for camera {camera_id}. "
                        f"Add it to {approved_tables_path()} before processing."
                    ) from exc

                detections = self.adapter.detect_objects(image, prompts=("person",))
                raw_people = detections.get("person", [])

                people: list[dict[str, Any]] = []
                for detection in raw_people:
                    ys, xs = np.nonzero(detection.mask)
                    if xs.size == 0:
                        continue
                    people.append(
                        {
                            "label": detection.label,
                            "score": detection.score,
                            "vector": detection.vector.astype(np.float32),
                            "vector_source": detection.vector_source,
                            "bbox_xyxy": detection.bbox_xyxy,
                            "centroid_xy": (float(xs.mean()), float(ys.mean())),
                            "mask": detection.mask.astype(bool),
                            "track_id": None,
                        }
                    )

                records.append(
                    {
                        "image_path": image_path,
                        "timestamp_seconds": int(frame_info["timestamp_seconds"]),
                        "image_shape": (image.height, image.width),
                        "tables": tables,
                        "people": people,
                    }
                )
        return records

    def _build_sample_bundle(
        self,
        video_meta: dict[str, Any],
        camera_id: str,
        triplet: list[dict[str, Any]],
        table_track_id: str,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Image.Image]]:
        table_rows = []
        review_images: dict[str, Image.Image] = {}
        people_rows: list[dict[str, Any]] = []
        person_vecs_full: list[np.ndarray] = []
        person_vecs_intersection: list[np.ndarray] = []
        person_track_ids: list[str] = []
        person_frame_indices: list[int] = []
        person_weights: list[float] = []
        table_vecs: list[np.ndarray] = []
        frame_times: list[int] = []
        frame_people_by_track: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
        anchor_hull_points: list[list[float]] = []
        anchor_table_meta: dict[str, Any] | None = None
        bbox_fallback_used = False

        for frame_idx, frame in enumerate(triplet):
            with Image.open(frame["image_path"]) as image:
                image = image.convert("RGB")
                table = next(item for item in frame["tables"] if item["track_id"] == table_track_id)
                tight_mask = table["tight_mask"].astype(bool)
                expanded_mask = table["expanded_mask"].astype(bool)
                zone_rect = table.get("zone_rect")
                if zone_rect:
                    review_crop = crop_with_polygonal_rect(image, zone_rect)
                else:
                    bbox_fallback_used = True
                    preview_bbox = bbox_with_padding(
                        table["expanded_bbox_xyxy"],
                        frame["image_shape"],
                        preview_padding_for_bbox(table["expanded_bbox_xyxy"]),
                    )
                    review_crop = crop_image(image, preview_bbox)

                if frame_idx == 0:
                    review_images["t_minus_20"] = review_crop
                elif frame_idx == 1:
                    review_images["t_minus_10"] = review_crop
                else:
                    review_images["anchor"] = review_crop
                    tight_rect = table.get("tight_rect")
                    if tight_rect:
                        review_images["tight_anchor"] = crop_with_polygonal_rect(image, tight_rect)
                    else:
                        bbox_fallback_used = True
                        tight_bbox = bbox_with_padding(table["bbox_xyxy"], frame["image_shape"], 0)
                        review_images["tight_anchor"] = crop_image(image, tight_bbox)
                    anchor_hull_points = table["hull_points"]
                    anchor_table_meta = table

                table_vecs.append(table["vector"].astype(np.float32))
                frame_times.append(int(frame["timestamp_seconds"]))
                table_rows.append(
                    {
                        "frame_index": frame_idx,
                        "timestamp_seconds": int(frame["timestamp_seconds"]),
                        "tight_bbox": [int(value) for value in table["bbox_xyxy"]],
                        "expanded_bbox": [int(value) for value in table["expanded_bbox_xyxy"]],
                        "tight_area": int(tight_mask.sum()),
                        "expanded_area": int(expanded_mask.sum()),
                        "tight_mask_rle": encode_mask_rle(tight_mask),
                        "expanded_mask_rle": encode_mask_rle(expanded_mask),
                        "compactness": float(table["compactness"]),
                    }
                )

                table_centroid = table["centroid_xy"]
                for person in frame["people"]:
                    intersection_mask = expanded_mask_intersection(person["mask"], expanded_mask)
                    if not intersection_mask.any() and not bbox_intersects_mask(person["bbox_xyxy"], expanded_mask):
                        continue
                    overlap_px = int(intersection_mask.sum())
                    person_area = max(int(person["mask"].sum()), 1)
                    expanded_area = max(int(expanded_mask.sum()), 1)
                    overlap_frac_person = overlap_px / float(person_area)
                    overlap_frac_expanded = overlap_px / float(expanded_area)
                    distance_norm = float(
                        math.hypot(person["centroid_xy"][0] - table_centroid[0], person["centroid_xy"][1] - table_centroid[1])
                        / max(math.hypot(frame["image_shape"][1], frame["image_shape"][0]), 1.0)
                    )
                    if overlap_px > 0:
                        intersection_vector = self.adapter.derive_region_vector(image, intersection_mask)
                    else:
                        intersection_vector = np.zeros((VECTOR_DIM,), dtype=np.float32)
                    weight = float(0.6 * overlap_frac_person + 0.4 * overlap_frac_expanded)
                    track_id = str(person["track_id"])
                    displacement = None
                    if frame_idx > 0 and (frame_idx - 1) in frame_people_by_track.get(track_id, {}):
                        prev_centroid = frame_people_by_track[track_id][frame_idx - 1]
                        displacement = float(
                            math.hypot(
                                person["centroid_xy"][0] - prev_centroid[0],
                                person["centroid_xy"][1] - prev_centroid[1],
                            )
                            / max(math.hypot(frame["image_shape"][1], frame["image_shape"][0]), 1.0)
                        )
                    frame_people_by_track[track_id][frame_idx] = person["centroid_xy"]
                    people_rows.append(
                        {
                            "frame_index": frame_idx,
                            "track_id": track_id,
                            "overlap_px": overlap_px,
                            "overlap_frac_of_person": overlap_frac_person,
                            "overlap_frac_of_expanded": overlap_frac_expanded,
                            "distance_norm": distance_norm,
                            "displacement_from_prev": displacement,
                        }
                    )
                    person_vecs_full.append(person["vector"].astype(np.float32))
                    person_vecs_intersection.append(intersection_vector.astype(np.float32))
                    person_track_ids.append(track_id)
                    person_frame_indices.append(frame_idx)
                    person_weights.append(weight)

        anchor_time = frame_times[-1]
        sample_id = build_sample_id(camera_id, video_meta["name"], table_track_id, anchor_time)
        perception_payload = {
            "table_vecs_tight": np.stack(table_vecs).astype(np.float32),
            "person_vecs_full": np.stack(person_vecs_full).astype(np.float32)
            if person_vecs_full
            else np.zeros((0, VECTOR_DIM), dtype=np.float32),
            "person_vecs_intersection": np.stack(person_vecs_intersection).astype(np.float32)
            if person_vecs_intersection
            else np.zeros((0, VECTOR_DIM), dtype=np.float32),
            "person_frame_index": np.array(person_frame_indices, dtype=np.int32),
            "person_track_id": np.array(person_track_ids, dtype="U64"),
            "person_weight_for_mlp": np.array(person_weights, dtype=np.float32),
        }

        sample_payload = {
            "schema_version": SAMPLE_SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "sample_id": sample_id,
            "source_video": {
                "drive_file_id": video_meta["id"],
                "video_name": video_meta["name"],
                "camera_id": camera_id,
                "image_width": int(triplet[-1]["image_shape"][1]),
                "image_height": int(triplet[-1]["image_shape"][0]),
                "source_folder_id": video_meta.get("source_folder_id"),
                "source_folder_name": video_meta.get("source_folder_name"),
            },
            "timing": {
                "anchor_time_seconds": anchor_time,
                "frame_times_seconds": frame_times,
            },
            "table": {
                "table_track_id": table_track_id,
                "mask_id": anchor_table_meta.get("mask_id") if anchor_table_meta else None,
                "label": str(anchor_table_meta.get("table_label") or table_track_id) if anchor_table_meta else table_track_id,
                "bbox": _table_bbox_xywh(anchor_table_meta) if anchor_table_meta else [0, 0, 0, 0],
                "tight_rect": _serialize_rect_payload(anchor_table_meta.get("tight_rect")) if anchor_table_meta else None,
                "zone_rect": _serialize_rect_payload(anchor_table_meta.get("zone_rect")) if anchor_table_meta else None,
                "tight_hull_anchor_polygon": anchor_hull_points,
                "compactness_anchor": float(table_rows[-1]["compactness"]),
            },
            "frames": table_rows,
            "people": people_rows,
            "label": {"human_label": None, "occupancy_binary_label": None},
            "vector_spec": {
                "contract": VECTOR_CONTRACT_VERSION,
                "dim": VECTOR_DIM,
                "sam3_checkpoint_id": os.environ.get("SAM3_CHECKPOINT_PATH", ""),
                "sam3_model_id": os.environ.get("SAM3_CONFIG_NAME", ""),
            },
            "quality_flags": {"bbox_fallback_used": bbox_fallback_used, "dropped_people_count": 0, "worker_notes": []},
        }
        return sample_payload, perception_payload, review_images

    def _upload_review_bundle(self, table_parent_id: str, sample_id: str, sample_folder: Path) -> dict[str, str]:
        sample_folder_id = self.client.ensure_subfolder(table_parent_id, sample_id)
        preview_anchor = self.client.upload_or_update_file(
            sample_folder / PREVIEW_ANCHOR_NAME,
            sample_folder_id,
            PREVIEW_ANCHOR_NAME,
            "image/jpeg",
        )
        preview_prev_1 = self.client.upload_or_update_file(
            sample_folder / PREVIEW_PREV_1_NAME,
            sample_folder_id,
            PREVIEW_PREV_1_NAME,
            "image/jpeg",
        )
        preview_prev_2 = self.client.upload_or_update_file(
            sample_folder / PREVIEW_PREV_2_NAME,
            sample_folder_id,
            PREVIEW_PREV_2_NAME,
            "image/jpeg",
        )
        tight_anchor = self.client.upload_or_update_file(
            sample_folder / TIGHT_ANCHOR_NAME,
            sample_folder_id,
            TIGHT_ANCHOR_NAME,
            "image/jpeg",
        )
        sample_json_file = self.client.upsert_bytes(
            sample_folder_id,
            SAMPLE_JSON_NAME,
            (sample_folder / SAMPLE_JSON_NAME).read_bytes(),
            "application/json",
        )
        perception_file = self.client.upsert_bytes(
            sample_folder_id,
            PERCEPTION_NPZ_NAME,
            (sample_folder / PERCEPTION_NPZ_NAME).read_bytes(),
            "application/octet-stream",
        )
        return {
            "sample_folder_id": sample_folder_id,
            "preview_anchor_file_id": preview_anchor["id"],
            "preview_t_minus_10_file_id": preview_prev_1["id"],
            "preview_t_minus_20_file_id": preview_prev_2["id"],
            "tight_anchor_file_id": tight_anchor["id"],
            "sample_json_file_id": sample_json_file["id"],
            "perception_file_id": perception_file["id"],
        }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process Drive videos into grouped review samples.")
    parser.add_argument("--project-root", default=os.environ.get("DRIVE_PROJECT_ROOT_FOLDER_ID"))
    parser.add_argument("--source-root", default=os.environ.get("DRIVE_VIDEO_SOURCE_ROOT_ID"))
    parser.add_argument("--review-root", default=os.environ.get("DRIVE_REVIEW_QUEUE_ROOT_ID"))
    parser.add_argument("--cache-dir", default=os.environ.get("PROCESSOR_LOCAL_CACHE_DIR", "worker_cache"))
    parser.add_argument("--frame-interval", type=int, default=10)
    parser.add_argument("--expansion-strength", type=float, default=1.0)
    parser.add_argument("--force-reprocess", action="store_true")
    parser.add_argument("--max-videos", type=int, default=int(os.environ.get("PROCESSOR_MAX_VIDEOS_PER_RUN", "0") or "0"))
    parser.add_argument("--max-samples", type=int, default=int(os.environ.get("PROCESSOR_MAX_SAMPLES_PER_RUN", "0") or "0"))
    parser.add_argument(
        "--max-pending-samples",
        type=int,
        default=int(os.environ.get("PROCESSOR_MAX_PENDING_SAMPLES", "0") or "0"),
    )
    parser.add_argument(
        "--resume-pending-samples",
        type=int,
        default=int(os.environ.get("PROCESSOR_RESUME_PENDING_SAMPLES", "0") or "0"),
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=int(os.environ.get("PROCESSOR_POLL_SECONDS", "60") or "60"),
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        default=parse_bool(os.environ.get("PROCESSOR_CONTINUOUS"), default=False),
    )
    parser.add_argument(
        "--trash-source-videos",
        action="store_true",
        default=parse_bool(os.environ.get("PROCESSOR_TRASH_SOURCE_VIDEOS"), default=False),
    )
    parser.add_argument(
        "--keep-frames",
        action="store_false",
        dest="cleanup_frames",
        default=parse_bool(os.environ.get("PROCESSOR_CLEANUP_FRAMES"), default=True),
    )
    parser.add_argument(
        "--keep-review-cache",
        action="store_false",
        dest="cleanup_review_cache",
        default=parse_bool(os.environ.get("PROCESSOR_CLEANUP_REVIEW_CACHE"), default=True),
    )
    parser.add_argument(
        "--keep-local-video-after-trash",
        action="store_false",
        dest="cleanup_local_video_when_source_trashed",
        default=parse_bool(os.environ.get("PROCESSOR_CLEANUP_LOCAL_VIDEO_WHEN_SOURCE_TRASHED"), default=True),
    )
    parser.add_argument("--source-folder-names", default=os.environ.get("PROCESSOR_SOURCE_FOLDER_NAMES", ""))
    parser.add_argument(
        "--list-source-folders",
        action="store_true",
        help="List available Drive source folders and exit without loading SAM or processing videos.",
    )
    parser.add_argument("--checkpoint-path", default=os.environ.get("SAM3_CHECKPOINT_PATH"))
    parser.add_argument("--config-name", default=os.environ.get("SAM3_CONFIG_NAME"))
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if not args.project_root and (not args.source_root or not args.review_root):
        parser.error("--project-root or both --source-root and --review-root are required")

    try:
        print("[worker] initializing Drive client")
        client = DriveClient()
        print("[worker] resolving Drive pipeline roots")
        roots = resolve_video_pipeline_roots(
            client,
            project_root_id=args.project_root,
            source_root_id=args.source_root,
            review_root_id=args.review_root,
            temporal_root_id=os.environ.get("DRIVE_OUTPUT_TEMPORAL_STATE_ROOT_ID"),
            surface_root_id=os.environ.get("DRIVE_OUTPUT_DIRTY_CLEAN_SURFACE_ROOT_ID"),
            occupancy_root_id=os.environ.get("DRIVE_OUTPUT_OCCUPANCY_MLP_ROOT_ID"),
            audit_root_id=os.environ.get("DRIVE_OUTPUT_SAM_AUDIT_ROOT_ID"),
        )
        source_root_id = roots.get("source")
        review_root_id = roots.get("review")
        if not source_root_id or not review_root_id:
            print(
                json.dumps(
                    {
                        "success": False,
                        "error": "Could not resolve source/review roots. Set DRIVE_PROJECT_ROOT_FOLDER_ID or explicit DRIVE_VIDEO_SOURCE_ROOT_ID and DRIVE_REVIEW_QUEUE_ROOT_ID.",
                    },
                    indent=2,
                )
            )
            return 1

        if args.list_source_folders:
            summaries = summarize_source_folders(client, source_root_id)
            print(
                json.dumps(
                    {
                        "success": True,
                        "source_root_id": source_root_id,
                        "folder_count": len(summaries),
                        "folders": summaries,
                    },
                    indent=2,
                )
            )
            return 0

        print(
            "[worker] loading SAM3 adapter "
            f"(config={args.config_name or 'default'}, checkpoint={args.checkpoint_path or 'default'})"
        )
        print("[worker] first startup on a local CPU machine can take a while here")
        adapter = Sam3Adapter(checkpoint_path=args.checkpoint_path, config_name=args.config_name)
        print(f"[worker] SAM3 ready (backend={adapter.backend}, device={adapter.device})")
    except (DriveClientError, Sam3AdapterError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, indent=2))
        return 1

    review_store = VideoReviewStorePG() if database_enabled() else None
    worker_state_store = WorkerStateStorePG() if database_enabled() else None

    print(
        "[worker] starting scan "
        f"(continuous={bool(args.continuous)}, max_videos={args.max_videos or 'all'}, "
        f"source_folder_names={parse_csv_values(args.source_folder_names) or ['all']})"
    )
    worker = VideoDatasetWorker(
        client=client,
        adapter=adapter,
        source_root_id=source_root_id,
        review_root_id=review_root_id,
        cache_dir=Path(args.cache_dir),
        frame_interval=args.frame_interval,
        expansion_strength=args.expansion_strength,
        force_reprocess=args.force_reprocess,
        max_videos=args.max_videos if args.max_videos and args.max_videos > 0 else None,
        include_source_folder_names=parse_csv_values(args.source_folder_names),
        max_samples_per_run=args.max_samples if args.max_samples and args.max_samples > 0 else None,
        max_pending_samples=args.max_pending_samples if args.max_pending_samples and args.max_pending_samples > 0 else None,
        resume_pending_samples=args.resume_pending_samples if args.resume_pending_samples and args.resume_pending_samples > 0 else None,
        poll_seconds=args.poll_seconds,
        continuous=bool(args.continuous),
        trash_source_videos=bool(args.trash_source_videos),
        cleanup_frames=bool(args.cleanup_frames),
        cleanup_review_cache=bool(args.cleanup_review_cache),
        cleanup_local_video_when_source_trashed=bool(args.cleanup_local_video_when_source_trashed),
        review_store=review_store,
        worker_state_store=worker_state_store,
    )
    result = worker.process_all()
    print(json.dumps({"success": True, **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
