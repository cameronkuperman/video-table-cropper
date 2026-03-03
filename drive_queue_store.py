"""SQLite-backed session and queue state for Drive labeling."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class DriveQueueStore:
    """Persistence layer for Drive labeling sessions, queue items, and undo actions."""

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
                    mode TEXT NOT NULL,
                    source_parent_folder_id TEXT,
                    source_folder_ids_json TEXT NOT NULL,
                    batch_limit INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS queue_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    source_root_folder_id TEXT NOT NULL,
                    source_folder_id TEXT NOT NULL,
                    source_folder_name TEXT NOT NULL,
                    source_file_id TEXT NOT NULL,
                    source_file_name TEXT NOT NULL,
                    source_file_mime_type TEXT,
                    segment_id TEXT NOT NULL,
                    segment_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    label TEXT,
                    output_file_id TEXT,
                    recycle_file_id TEXT,
                    source_file_in_processed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(id),
                    UNIQUE(session_id, source_file_id, segment_id)
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
                    moved_file_id TEXT,
                    undone INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(id),
                    FOREIGN KEY(queue_item_id) REFERENCES queue_items(id)
                );

                CREATE INDEX IF NOT EXISTS idx_queue_items_session_status_id
                    ON queue_items(session_id, status, id);
                CREATE INDEX IF NOT EXISTS idx_queue_items_session_source
                    ON queue_items(session_id, source_file_id);
                CREATE INDEX IF NOT EXISTS idx_actions_session_undone
                    ON actions(session_id, undone, id);
                """
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        if "segment_json" in result and result["segment_json"]:
            result["segment"] = json.loads(result["segment_json"])
        return result

    def create_session(
        self,
        session_id: str,
        mode: str,
        source_parent_folder_id: str | None,
        source_folder_ids: list[str],
        batch_limit: int,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(id, mode, source_parent_folder_id, source_folder_ids_json, batch_limit)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    mode,
                    source_parent_folder_id,
                    json.dumps(source_folder_ids),
                    batch_limit,
                ),
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()

        if not row:
            return None

        session = dict(row)
        session["source_folder_ids"] = json.loads(session["source_folder_ids_json"])
        return session

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
                        source_root_folder_id,
                        source_folder_id,
                        source_folder_name,
                        source_file_id,
                        source_file_name,
                        source_file_mime_type,
                        segment_id,
                        segment_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["session_id"],
                        item["source_root_folder_id"],
                        item["source_folder_id"],
                        item["source_folder_name"],
                        item["source_file_id"],
                        item["source_file_name"],
                        item.get("source_file_mime_type"),
                        item["segment_id"],
                        json.dumps(item["segment"]),
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
        output_file_id: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE queue_items
                SET status = 'labeled',
                    label = ?,
                    output_file_id = ?,
                    recycle_file_id = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ? AND id = ?
                """,
                (label, output_file_id, session_id, item_id),
            )

    def update_item_after_skip(
        self,
        session_id: str,
        item_id: int,
        recycle_file_id: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE queue_items
                SET status = 'skipped',
                    label = NULL,
                    recycle_file_id = ?,
                    output_file_id = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ? AND id = ?
                """,
                (recycle_file_id, session_id, item_id),
            )

    def restore_item(
        self,
        session_id: str,
        item_id: int,
        status: str,
        label: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE queue_items
                SET status = ?,
                    label = ?,
                    output_file_id = NULL,
                    recycle_file_id = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ? AND id = ?
                """,
                (status, label, session_id, item_id),
            )

    def mark_source_file_processed(self, session_id: str, source_file_id: str, processed: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE queue_items
                SET source_file_in_processed = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ? AND source_file_id = ?
                """,
                (1 if processed else 0, session_id, source_file_id),
            )

    def is_source_file_complete(self, session_id: str, source_file_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS pending_count
                FROM queue_items
                WHERE session_id = ?
                  AND source_file_id = ?
                  AND status = 'pending'
                """,
                (session_id, source_file_id),
            ).fetchone()
        return (row["pending_count"] if row else 0) == 0

    def log_action(
        self,
        session_id: str,
        queue_item_id: int,
        action_type: str,
        prev_status: str,
        new_status: str,
        prev_label: str | None,
        new_label: str | None,
        moved_file_id: str | None,
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
                    moved_file_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    queue_item_id,
                    action_type,
                    prev_status,
                    new_status,
                    prev_label,
                    new_label,
                    moved_file_id,
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
        return dict(row) if row else None

    def mark_action_undone(self, action_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE actions SET undone = 1 WHERE id = ?",
                (action_id,),
            )

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

            pending_after_cursor_row = conn.execute(
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
        label_counts = {
            (row["label"] or "unlabeled"): row["count"]
            for row in label_rows
        }

        return {
            "total": total_row["count"] if total_row else 0,
            "status_counts": {
                "pending": status_counts.get("pending", 0),
                "labeled": status_counts.get("labeled", 0),
                "skipped": status_counts.get("skipped", 0),
            },
            "label_counts": label_counts,
            "next_pending_item_id": pending_after_cursor_row["id"] if pending_after_cursor_row else None,
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
