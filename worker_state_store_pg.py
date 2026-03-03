"""Postgres-backed worker heartbeat and processed-video state."""

from __future__ import annotations

import json
import time
from typing import Any

from db import db_connection
from worker_runtime import default_worker_runtime_state


def _epoch(value: Any) -> int | None:
    if value is None:
        return None
    if hasattr(value, "timestamp"):
        return int(value.timestamp())
    return int(value)


def _json_load(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


class WorkerStateStorePG:
    """Persistence layer for worker runtime status and processed-video markers."""

    def __init__(self, worker_id: str = "video-dataset-worker") -> None:
        self.worker_id = worker_id

    def write_status(self, payload: dict[str, Any]) -> None:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO worker_status(
                        worker_id,
                        event_seq,
                        state,
                        message,
                        worker_running,
                        current_video,
                        last_error,
                        stop_reason,
                        counters,
                        config,
                        heartbeat_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb, %s::jsonb, NOW()
                    )
                    ON CONFLICT (worker_id)
                    DO UPDATE SET
                        event_seq = EXCLUDED.event_seq,
                        state = EXCLUDED.state,
                        message = EXCLUDED.message,
                        worker_running = EXCLUDED.worker_running,
                        current_video = EXCLUDED.current_video,
                        last_error = EXCLUDED.last_error,
                        stop_reason = EXCLUDED.stop_reason,
                        counters = EXCLUDED.counters,
                        config = EXCLUDED.config,
                        heartbeat_at = NOW(),
                        updated_at = NOW()
                    """,
                    (
                        self.worker_id,
                        int(payload.get("event_seq", 0)),
                        payload.get("state", "unknown"),
                        payload.get("message", ""),
                        bool(payload.get("worker_running", False)),
                        json.dumps(payload.get("current_video")),
                        payload.get("last_error"),
                        payload.get("stop_reason"),
                        json.dumps(payload.get("counters") or {}),
                        json.dumps(payload.get("config") or {}),
                    ),
                )
            conn.commit()

    def get_status(self) -> dict[str, Any]:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM worker_status
                    WHERE worker_id = %s
                    """,
                    (self.worker_id,),
                )
                row = cur.fetchone()
            conn.commit()

        status = default_worker_runtime_state()
        if not row:
            return status

        status.update(
            {
                "event_seq": int(row["event_seq"]),
                "state": row["state"],
                "message": row["message"],
                "worker_running": bool(row["worker_running"]),
                "current_video": _json_load(row.get("current_video"), None),
                "last_error": row.get("last_error"),
                "stop_reason": row.get("stop_reason"),
                "counters": _json_load(row.get("counters"), {}),
                "config": _json_load(row.get("config"), {}),
                "updated_at_epoch": _epoch(row.get("heartbeat_at") or row.get("updated_at")),
                "heartbeat_at_epoch": _epoch(row.get("heartbeat_at")),
            }
        )
        heartbeat_at_epoch = status.get("heartbeat_at_epoch")
        status["stale"] = (
            heartbeat_at_epoch is not None and (int(time.time()) - int(heartbeat_at_epoch)) > 60
        )
        return status

    def is_video_processed(self, video_id: str) -> bool:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT video_id
                    FROM processed_videos
                    WHERE video_id = %s
                    """,
                    (video_id,),
                )
                row = cur.fetchone()
            conn.commit()
        return row is not None

    def mark_video_processed(
        self,
        video_meta: dict[str, Any],
        created_samples: int,
        *,
        source_video_trashed: bool = False,
    ) -> None:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO processed_videos(
                        video_id,
                        video_name,
                        source_folder_id,
                        source_folder_name,
                        created_samples,
                        source_video_trashed
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (video_id)
                    DO UPDATE SET
                        video_name = EXCLUDED.video_name,
                        source_folder_id = EXCLUDED.source_folder_id,
                        source_folder_name = EXCLUDED.source_folder_name,
                        created_samples = EXCLUDED.created_samples,
                        source_video_trashed = EXCLUDED.source_video_trashed,
                        processed_at = NOW()
                    """,
                    (
                        video_meta["id"],
                        video_meta["name"],
                        video_meta.get("source_folder_id"),
                        video_meta.get("source_folder_name"),
                        int(created_samples),
                        bool(source_video_trashed),
                    ),
                )
            conn.commit()
