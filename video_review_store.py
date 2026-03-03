"""SQLite-backed queue state for grouped video review samples."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class VideoReviewStore:
    """Persistence layer for video review sessions, queue items, and undo history."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    review_root_folder_id TEXT NOT NULL,
                    pending_root_folder_id TEXT NOT NULL,
                    batch_limit INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS queue_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    review_root_folder_id TEXT NOT NULL,
                    source_parent_folder_id TEXT NOT NULL,
                    sample_folder_id TEXT NOT NULL,
                    sample_folder_name TEXT NOT NULL,
                    sample_json TEXT NOT NULL,
                    preview_anchor_file_id TEXT,
                    preview_t_minus_10_file_id TEXT,
                    preview_t_minus_20_file_id TEXT,
                    tight_anchor_file_id TEXT,
                    perception_file_id TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    label TEXT,
                    exported_folder_ids_json TEXT,
                    archived_parent_folder_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(id),
                    UNIQUE(session_id, sample_folder_id)
                );

                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    queue_item_id INTEGER NOT NULL,
                    action_type TEXT NOT NULL,
                    prev_status TEXT,
                    new_status TEXT,
                    prev_label TEXT,
                    new_label TEXT,
                    exported_folder_ids_json TEXT,
                    moved_folder_id TEXT,
                    archive_parent_folder_id TEXT,
                    undone INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(id),
                    FOREIGN KEY(queue_item_id) REFERENCES queue_items(id)
                );

                CREATE INDEX IF NOT EXISTS idx_video_queue_session_status_id
                    ON queue_items(session_id, status, id);
                CREATE INDEX IF NOT EXISTS idx_video_actions_session_undone
                    ON actions(session_id, undone, id);
                """
            )
            # Add export tracking columns (idempotent ALTER TABLEs for existing DBs)
            for stmt in (
                "ALTER TABLE queue_items ADD COLUMN export_status TEXT NOT NULL DEFAULT 'none'",
                "ALTER TABLE queue_items ADD COLUMN export_error TEXT",
                "ALTER TABLE queue_items ADD COLUMN export_attempts INTEGER NOT NULL DEFAULT 0",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # column already exists

    @staticmethod
    def _decode_json_field(value: str | None) -> Any:
        if not value:
            return None
        return json.loads(value)

    @classmethod
    def _row_to_dict(cls, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        if result.get("sample_json"):
            result["sample"] = json.loads(result["sample_json"])
        if "exported_folder_ids_json" in result:
            result["exported_folder_ids"] = cls._decode_json_field(result.get("exported_folder_ids_json")) or []
        return result

    def create_session(
        self,
        session_id: str,
        review_root_folder_id: str,
        pending_root_folder_id: str,
        batch_limit: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(id, review_root_folder_id, pending_root_folder_id, batch_limit)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, review_root_folder_id, pending_root_folder_id, batch_limit),
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def add_queue_items(self, items: list[dict[str, Any]]) -> int:
        if not items:
            return 0
        inserted = 0
        with self._connect() as conn:
            for item in items:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO queue_items(
                        session_id,
                        review_root_folder_id,
                        source_parent_folder_id,
                        sample_folder_id,
                        sample_folder_name,
                        sample_json,
                        preview_anchor_file_id,
                        preview_t_minus_10_file_id,
                        preview_t_minus_20_file_id,
                        tight_anchor_file_id,
                        perception_file_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["session_id"],
                        item["review_root_folder_id"],
                        item["source_parent_folder_id"],
                        item["sample_folder_id"],
                        item["sample_folder_name"],
                        json.dumps(item["sample"]),
                        item.get("preview_anchor_file_id"),
                        item.get("preview_t_minus_10_file_id"),
                        item.get("preview_t_minus_20_file_id"),
                        item.get("tight_anchor_file_id"),
                        item.get("perception_file_id"),
                    ),
                )
                inserted += cursor.rowcount
        return inserted

    def get_pending_batch(self, session_id: str, limit: int, cursor: int = 0) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM queue_items
                WHERE session_id = ?
                  AND status = 'pending'
                  AND id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (session_id, cursor, limit),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows if row is not None]

    def get_item(self, session_id: str, item_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM queue_items WHERE session_id = ? AND id = ?",
                (session_id, item_id),
            ).fetchone()
        return self._row_to_dict(row)

    def get_items(self, session_id: str, item_ids: list[int], status: str | None = None) -> list[dict[str, Any]]:
        if not item_ids:
            return []

        placeholders = ",".join("?" for _ in item_ids)
        params: list[Any] = [session_id, *item_ids]
        status_clause = ""
        if status:
            status_clause = " AND status = ?"
            params.append(status)

        query = f"""
            SELECT *
            FROM queue_items
            WHERE session_id = ?
              AND id IN ({placeholders})
              {status_clause}
            ORDER BY id ASC
        """

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_dict(row) for row in rows if row is not None]

    def update_item_after_label(
        self,
        session_id: str,
        item_id: int,
        label: str,
        exported_folder_ids: list[str],
        archived_parent_folder_id: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE queue_items
                SET status = 'labeled',
                    label = ?,
                    exported_folder_ids_json = ?,
                    archived_parent_folder_id = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ? AND id = ?
                """,
                (
                    label,
                    json.dumps(exported_folder_ids),
                    archived_parent_folder_id,
                    session_id,
                    item_id,
                ),
            )

    def update_item_after_skip(
        self,
        session_id: str,
        item_id: int,
        archived_parent_folder_id: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE queue_items
                SET status = 'skipped',
                    label = NULL,
                    exported_folder_ids_json = NULL,
                    archived_parent_folder_id = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ? AND id = ?
                """,
                (archived_parent_folder_id, session_id, item_id),
            )

    def update_item_after_trash(self, session_id: str, item_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE queue_items
                SET status = 'trashed',
                    label = NULL,
                    exported_folder_ids_json = NULL,
                    archived_parent_folder_id = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ? AND id = ?
                """,
                (session_id, item_id),
            )

    def restore_item(self, session_id: str, item_id: int, status: str, label: str | None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE queue_items
                SET status = ?,
                    label = ?,
                    exported_folder_ids_json = NULL,
                    archived_parent_folder_id = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ? AND id = ?
                """,
                (status, label, session_id, item_id),
            )

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
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO actions(
                    session_id,
                    queue_item_id,
                    action_type,
                    prev_status,
                    new_status,
                    prev_label,
                    new_label,
                    exported_folder_ids_json,
                    moved_folder_id,
                    archive_parent_folder_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            conn.execute(
                """
                DELETE FROM actions
                WHERE session_id = ?
                  AND id NOT IN (
                      SELECT id FROM actions
                      WHERE session_id = ?
                      ORDER BY id DESC
                      LIMIT 200
                  )
                """,
                (session_id, session_id),
            )

    def get_last_action(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM actions
                WHERE session_id = ?
                  AND undone = 0
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["exported_folder_ids"] = self._decode_json_field(result.get("exported_folder_ids_json")) or []
        return result

    def mark_action_undone(self, action_id: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE actions SET undone = 1 WHERE id = ?", (action_id,))

    def get_stats(self, session_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            status_rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM queue_items
                WHERE session_id = ?
                GROUP BY status
                """,
                (session_id,),
            ).fetchall()
            label_rows = conn.execute(
                """
                SELECT label, COUNT(*) AS count
                FROM queue_items
                WHERE session_id = ?
                  AND status = 'labeled'
                GROUP BY label
                """,
                (session_id,),
            ).fetchall()
            total_row = conn.execute(
                "SELECT COUNT(*) AS count FROM queue_items WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            pending_row = conn.execute(
                """
                SELECT id
                FROM queue_items
                WHERE session_id = ?
                  AND status = 'pending'
                ORDER BY id ASC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

        status_counts = {row["status"]: row["count"] for row in status_rows}
        label_counts = {(row["label"] or "unlabeled"): row["count"] for row in label_rows}
        return {
            "total": total_row["count"] if total_row else 0,
            "status_counts": {
                "pending": status_counts.get("pending", 0),
                "labeled": status_counts.get("labeled", 0),
                "skipped": status_counts.get("skipped", 0),
                "trashed": status_counts.get("trashed", 0),
            },
            "label_counts": label_counts,
            "next_pending_item_id": pending_row["id"] if pending_row else None,
        }

    def has_pending_after(self, session_id: str, cursor: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id
                FROM queue_items
                WHERE session_id = ?
                  AND status = 'pending'
                  AND id > ?
                LIMIT 1
                """,
                (session_id, cursor),
            ).fetchone()
        return row is not None

    def label_item_optimistic(
        self,
        session_id: str,
        item_id: int,
        label: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE queue_items
                SET status = 'labeled',
                    label = ?,
                    export_status = 'pending',
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ? AND id = ?
                """,
                (label, session_id, item_id),
            )

    def claim_pending_exports(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM queue_items
                WHERE export_status IN ('pending', 'failed')
                  AND export_attempts < 3
                ORDER BY id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            ids = [row["id"] for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"""
                    UPDATE queue_items
                    SET export_status = 'in_progress',
                        export_attempts = export_attempts + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id IN ({placeholders})
                    """,
                    ids,
                )
        return [self._row_to_dict(row) for row in rows if row is not None]

    def complete_export(
        self,
        item_id: int,
        exported_folder_ids: list[str],
        archived_parent_folder_id: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE queue_items
                SET export_status = 'completed',
                    export_error = NULL,
                    exported_folder_ids_json = ?,
                    archived_parent_folder_id = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (json.dumps(exported_folder_ids), archived_parent_folder_id, item_id),
            )

    def fail_export(self, item_id: int, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE queue_items
                SET export_status = 'failed',
                    export_error = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (error, item_id),
            )
