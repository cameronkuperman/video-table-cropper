"""Postgres-backed queue state for grouped video review samples."""

from __future__ import annotations

import json
from typing import Any

from db import db_connection

LEASE_MINUTES = 20


def _json_load(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


class VideoReviewStorePG:
    """Persistence layer for production video review sessions and globally shared queue items."""

    def __init__(self, worker_id: str = "video-dataset-worker") -> None:
        self.worker_id = worker_id

    @staticmethod
    def _row_to_dict(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        result.setdefault("session_id", result.get("claimed_by_session_id") or "shared")
        if "sample_json" in result:
            result["sample"] = _json_load(result.get("sample_json"), {})
        if "exported_folder_ids" in result:
            result["exported_folder_ids"] = _json_load(result.get("exported_folder_ids"), [])
        return result

    @staticmethod
    def _lease_clause(session_id: str) -> tuple[str, list[Any]]:
        return (
            """
            (
                claimed_by_session_id IS NULL
                OR claimed_by_session_id = %s
                OR claim_expires_at IS NULL
                OR claim_expires_at < NOW()
            )
            """,
            [session_id],
        )

    def create_session(
        self,
        session_id: str,
        review_root_folder_id: str,
        pending_root_folder_id: str,
        batch_limit: int,
    ) -> None:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO video_review_sessions(
                        id,
                        review_root_folder_id,
                        pending_root_folder_id,
                        batch_limit
                    )
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id)
                    DO UPDATE SET
                        review_root_folder_id = EXCLUDED.review_root_folder_id,
                        pending_root_folder_id = EXCLUDED.pending_root_folder_id,
                        batch_limit = EXCLUDED.batch_limit,
                        last_seen_at = NOW()
                    """,
                    (session_id, review_root_folder_id, pending_root_folder_id, batch_limit),
                )
            conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM video_review_sessions
                    WHERE id = %s
                    """,
                    (session_id,),
                )
                row = cur.fetchone()
            conn.commit()
        return dict(row) if row else None

    def touch_session(self, session_id: str) -> None:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE video_review_sessions
                    SET last_seen_at = NOW()
                    WHERE id = %s
                    """,
                    (session_id,),
                )
            conn.commit()

    def upsert_queue_item(self, item: dict[str, Any]) -> int:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO video_review_items(
                        sample_id,
                        sample_folder_id,
                        sample_folder_name,
                        review_root_folder_id,
                        source_parent_folder_id,
                        source_video_drive_file_id,
                        source_video_name,
                        camera_id,
                        table_track_id,
                        anchor_time_seconds,
                        preview_anchor_file_id,
                        preview_t_minus_10_file_id,
                        preview_t_minus_20_file_id,
                        tight_anchor_file_id,
                        perception_file_id,
                        sample_json
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                    )
                    ON CONFLICT (sample_id)
                    DO UPDATE SET
                        sample_folder_id = EXCLUDED.sample_folder_id,
                        sample_folder_name = EXCLUDED.sample_folder_name,
                        review_root_folder_id = EXCLUDED.review_root_folder_id,
                        source_parent_folder_id = EXCLUDED.source_parent_folder_id,
                        source_video_drive_file_id = EXCLUDED.source_video_drive_file_id,
                        source_video_name = EXCLUDED.source_video_name,
                        camera_id = EXCLUDED.camera_id,
                        table_track_id = EXCLUDED.table_track_id,
                        anchor_time_seconds = EXCLUDED.anchor_time_seconds,
                        preview_anchor_file_id = EXCLUDED.preview_anchor_file_id,
                        preview_t_minus_10_file_id = EXCLUDED.preview_t_minus_10_file_id,
                        preview_t_minus_20_file_id = EXCLUDED.preview_t_minus_20_file_id,
                        tight_anchor_file_id = EXCLUDED.tight_anchor_file_id,
                        perception_file_id = EXCLUDED.perception_file_id,
                        sample_json = EXCLUDED.sample_json,
                        updated_at = NOW()
                    RETURNING id
                    """,
                    (
                        item["sample_id"],
                        item["sample_folder_id"],
                        item["sample_folder_name"],
                        item["review_root_folder_id"],
                        item["source_parent_folder_id"],
                        item["source_video_drive_file_id"],
                        item["source_video_name"],
                        item["camera_id"],
                        item["table_track_id"],
                        item["anchor_time_seconds"],
                        item.get("preview_anchor_file_id"),
                        item.get("preview_t_minus_10_file_id"),
                        item.get("preview_t_minus_20_file_id"),
                        item.get("tight_anchor_file_id"),
                        item.get("perception_file_id"),
                        json.dumps(item["sample"]),
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        return int(row["id"]) if row else 0

    def count_pending_items(self) -> int:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS count FROM video_review_items WHERE status = 'pending'")
                row = cur.fetchone()
            conn.commit()
        return int(row["count"]) if row else 0

    def has_any_pending_items(self) -> bool:
        return self.count_pending_items() > 0

    def get_pending_batch(self, session_id: str, limit: int, cursor: int = 0) -> list[dict[str, Any]]:
        self.touch_session(session_id)
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    WITH candidates AS (
                        SELECT id
                        FROM video_review_items
                        WHERE status = 'pending'
                          AND id > %s
                          AND (
                              claimed_by_session_id IS NULL
                              OR claimed_by_session_id = %s
                              OR claim_expires_at IS NULL
                              OR claim_expires_at < NOW()
                          )
                        ORDER BY id ASC
                        LIMIT %s
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE video_review_items AS items
                    SET
                        claimed_by_session_id = %s,
                        claim_expires_at = NOW() + INTERVAL '{LEASE_MINUTES} minutes',
                        updated_at = NOW()
                    FROM candidates
                    WHERE items.id = candidates.id
                    RETURNING items.*
                    """,
                    (cursor, session_id, limit, session_id),
                )
                rows = cur.fetchall()
            conn.commit()
        decoded = [self._row_to_dict(row) for row in rows if row is not None]
        return sorted((row for row in decoded if row is not None), key=lambda row: int(row["id"]))

    def get_item(self, session_id: str, item_id: int) -> dict[str, Any] | None:
        self.touch_session(session_id)
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM video_review_items
                    WHERE id = %s
                    """,
                    (item_id,),
                )
                row = cur.fetchone()
            conn.commit()
        return self._row_to_dict(row)

    def get_items(self, session_id: str, item_ids: list[int], status: str | None = None) -> list[dict[str, Any]]:
        if not item_ids:
            return []

        self.touch_session(session_id)
        placeholders = ",".join(["%s"] * len(item_ids))
        params: list[Any] = [*item_ids]
        clauses = [f"id IN ({placeholders})"]
        if status:
            clauses.append("status = %s")
            params.append(status)
        if status == "pending":
            clauses.append(
                """
                (
                    claimed_by_session_id = %s
                    OR claimed_by_session_id IS NULL
                    OR claim_expires_at IS NULL
                    OR claim_expires_at < NOW()
                )
                """
            )
            params.append(session_id)

        query = f"""
            SELECT *
            FROM video_review_items
            WHERE {' AND '.join(clauses)}
            ORDER BY id ASC
        """

        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
            conn.commit()
        return [self._row_to_dict(row) for row in rows if row is not None]

    def _assert_pending_update(self, cur: Any, item_id: int, session_id: str, assignments_sql: str, values: list[Any]) -> None:
        cur.execute(
            f"""
            UPDATE video_review_items
            SET
                {assignments_sql},
                claimed_by_session_id = NULL,
                claim_expires_at = NULL,
                updated_at = NOW()
            WHERE id = %s
              AND status = 'pending'
              AND (
                  claimed_by_session_id = %s
                  OR claimed_by_session_id IS NULL
                  OR claim_expires_at IS NULL
                  OR claim_expires_at < NOW()
              )
            """,
            [*values, item_id, session_id],
        )
        if cur.rowcount != 1:
            raise RuntimeError(f"Queue item {item_id} is no longer claimable for this session")

    def update_item_after_label(
        self,
        session_id: str,
        item_id: int,
        label: str,
        exported_folder_ids: list[str],
        archived_parent_folder_id: str,
    ) -> None:
        with db_connection() as conn:
            with conn.cursor() as cur:
                self._assert_pending_update(
                    cur,
                    item_id,
                    session_id,
                    "status = 'labeled', label = %s, exported_folder_ids = %s::jsonb, archived_parent_folder_id = %s",
                    [label, json.dumps(exported_folder_ids), archived_parent_folder_id],
                )
            conn.commit()

    def update_item_after_skip(
        self,
        session_id: str,
        item_id: int,
        archived_parent_folder_id: str,
    ) -> None:
        with db_connection() as conn:
            with conn.cursor() as cur:
                self._assert_pending_update(
                    cur,
                    item_id,
                    session_id,
                    "status = 'skipped', label = NULL, exported_folder_ids = '[]'::jsonb, archived_parent_folder_id = %s",
                    [archived_parent_folder_id],
                )
            conn.commit()

    def update_item_after_trash(self, session_id: str, item_id: int) -> None:
        with db_connection() as conn:
            with conn.cursor() as cur:
                self._assert_pending_update(
                    cur,
                    item_id,
                    session_id,
                    "status = 'trashed', label = NULL, exported_folder_ids = '[]'::jsonb, archived_parent_folder_id = NULL",
                    [],
                )
            conn.commit()

    def restore_item(self, session_id: str, item_id: int, status: str, label: str | None) -> None:
        self.touch_session(session_id)
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE video_review_items
                    SET
                        status = %s,
                        label = %s,
                        exported_folder_ids = '[]'::jsonb,
                        archived_parent_folder_id = NULL,
                        claimed_by_session_id = NULL,
                        claim_expires_at = NULL,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (status, label, item_id),
                )
            conn.commit()

    def log_action(
        self,
        session_id: str,
        queue_item_id: int,
        action_type: str,
        prev_status: str,
        new_status: str,
        prev_label: str | None,
        new_label: str | None,
        exported_folder_ids: list[str] | None,
        moved_folder_id: str | None,
        archive_parent_folder_id: str | None,
    ) -> None:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO video_review_actions(
                        session_id,
                        queue_item_id,
                        action_type,
                        prev_status,
                        new_status,
                        prev_label,
                        new_label,
                        exported_folder_ids,
                        moved_folder_id,
                        archive_parent_folder_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                    """,
                    (
                        session_id,
                        queue_item_id,
                        action_type,
                        prev_status,
                        new_status,
                        prev_label,
                        new_label,
                        json.dumps(exported_folder_ids or []),
                        moved_folder_id,
                        archive_parent_folder_id,
                    ),
                )
                cur.execute(
                    """
                    DELETE FROM video_review_actions
                    WHERE session_id = %s
                      AND id NOT IN (
                          SELECT id
                          FROM video_review_actions
                          WHERE session_id = %s
                          ORDER BY id DESC
                          LIMIT 200
                      )
                    """,
                    (session_id, session_id),
                )
            conn.commit()

    def get_last_action(self, session_id: str) -> dict[str, Any] | None:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM video_review_actions
                    WHERE session_id = %s
                      AND undone = FALSE
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (session_id,),
                )
                row = cur.fetchone()
            conn.commit()
        if not row:
            return None
        result = dict(row)
        result["exported_folder_ids"] = _json_load(result.get("exported_folder_ids"), [])
        return result

    def mark_action_undone(self, action_id: int) -> None:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE video_review_actions
                    SET undone = TRUE
                    WHERE id = %s
                    """,
                    (action_id,),
                )
            conn.commit()

    def get_stats(self, session_id: str) -> dict[str, Any]:
        self.touch_session(session_id)
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM video_review_items
                    GROUP BY status
                    """
                )
                status_rows = cur.fetchall()
                cur.execute(
                    """
                    SELECT label, COUNT(*) AS count
                    FROM video_review_items
                    WHERE status = 'labeled'
                    GROUP BY label
                    """
                )
                label_rows = cur.fetchall()
                cur.execute("SELECT COUNT(*) AS count FROM video_review_items")
                total_row = cur.fetchone()
                cur.execute(
                    """
                    SELECT id
                    FROM video_review_items
                    WHERE status = 'pending'
                    ORDER BY id ASC
                    LIMIT 1
                    """
                )
                pending_row = cur.fetchone()
            conn.commit()

        status_counts = {row["status"]: row["count"] for row in status_rows}
        label_counts = {(row["label"] or "unlabeled"): row["count"] for row in label_rows}
        return {
            "total": int(total_row["count"]) if total_row else 0,
            "status_counts": {
                "pending": int(status_counts.get("pending", 0)),
                "labeled": int(status_counts.get("labeled", 0)),
                "skipped": int(status_counts.get("skipped", 0)),
                "trashed": int(status_counts.get("trashed", 0)),
            },
            "label_counts": {key: int(value) for key, value in label_counts.items()},
            "next_pending_item_id": int(pending_row["id"]) if pending_row else None,
        }

    def has_pending_after(self, session_id: str, cursor: int) -> bool:
        self.touch_session(session_id)
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM video_review_items
                    WHERE status = 'pending'
                      AND id > %s
                      AND (
                          claimed_by_session_id IS NULL
                          OR claimed_by_session_id = %s
                          OR claim_expires_at IS NULL
                          OR claim_expires_at < NOW()
                      )
                    LIMIT 1
                    """,
                    (cursor, session_id),
                )
                row = cur.fetchone()
            conn.commit()
        return row is not None

    def label_item_optimistic(
        self,
        session_id: str,
        item_id: int,
        label: str,
    ) -> None:
        with db_connection() as conn:
            with conn.cursor() as cur:
                self._assert_pending_update(
                    cur,
                    item_id,
                    session_id,
                    "status = 'labeled', label = %s, export_status = 'pending'",
                    [label],
                )
            conn.commit()

    def claim_pending_exports(self, limit: int = 10) -> list[dict[str, Any]]:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH candidates AS (
                        SELECT id
                        FROM video_review_items
                        WHERE export_status IN ('pending', 'failed')
                          AND export_attempts < 3
                        ORDER BY id ASC
                        LIMIT %s
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE video_review_items AS items
                    SET
                        export_status = 'in_progress',
                        export_attempts = items.export_attempts + 1,
                        updated_at = NOW()
                    FROM candidates
                    WHERE items.id = candidates.id
                    RETURNING items.*
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
            conn.commit()
        return [self._row_to_dict(row) for row in rows if row is not None]

    def complete_export(
        self,
        item_id: int,
        exported_folder_ids: list[str],
        archived_parent_folder_id: str,
    ) -> None:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE video_review_items
                    SET export_status = 'completed',
                        export_error = NULL,
                        exported_folder_ids = %s::jsonb,
                        archived_parent_folder_id = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (json.dumps(exported_folder_ids), archived_parent_folder_id, item_id),
                )
            conn.commit()

    def fail_export(self, item_id: int, error: str) -> None:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE video_review_items
                    SET export_status = 'failed',
                        export_error = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (error, item_id),
                )
            conn.commit()
