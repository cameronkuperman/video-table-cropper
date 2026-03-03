"""Background daemon thread for asynchronous Drive export after labeling."""

from __future__ import annotations

import logging
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from drive_client import DriveClient, DriveClientError
from sample_builder import load_sample_payload
from video_review_service import (
    archive_sample,
    ensure_cached_sample,
    ensure_review_roots,
    export_sample,
)

logger = logging.getLogger(__name__)


class DriveExportWorker:
    """Polls for items with export_status='pending' and exports them to Drive.

    Designed to run as a single daemon thread inside the Flask process.
    """

    def __init__(
        self,
        store: Any,
        cache_root: Path,
        get_client_fn: Callable[[], DriveClient],
        get_output_roots_fn: Callable[[], dict[str, str | None]],
        interval: float = 2.0,
        batch_size: int = 5,
    ) -> None:
        self.store = store
        self.cache_root = cache_root
        self.get_client_fn = get_client_fn
        self.get_output_roots_fn = get_output_roots_fn
        self.interval = interval
        self.batch_size = batch_size
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="drive-export-worker")
        self._thread.start()
        logger.info("DriveExportWorker started (interval=%.1fs, batch_size=%d)", self.interval, self.batch_size)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                items = self.store.claim_pending_exports(limit=self.batch_size)
                if not items:
                    self._stop_event.wait(timeout=self.interval)
                    continue
                for item in items:
                    if self._stop_event.is_set():
                        break
                    self._process_item(item)
            except Exception:
                logger.exception("DriveExportWorker loop error")
                self._stop_event.wait(timeout=self.interval * 3)

    def _process_item(self, item: dict[str, Any]) -> None:
        item_id = item["id"]
        try:
            # Re-check that the item is still labeled (undo may have reverted it)
            fresh = self.store.get_item(
                item.get("session_id") or item.get("claimed_by_session_id") or "shared",
                item_id,
            )
            if not fresh or fresh.get("status") != "labeled":
                logger.info("Export skipped for item %s (status changed to %s)", item_id, fresh.get("status") if fresh else "missing")
                # Reset export_status since the label was undone
                self.store.fail_export(item_id, "item no longer labeled")
                return

            client = self.get_client_fn()
            output_roots = self.get_output_roots_fn()
            sample_dir = ensure_cached_sample(client, fresh, self.cache_root)
            sample_payload = load_sample_payload(sample_dir)
            label = fresh["label"]

            exported = export_sample(client, sample_dir, sample_payload, label, output_roots)

            session = self.store.get_session(
                fresh.get("session_id") or fresh.get("claimed_by_session_id") or "shared",
            )
            review_root_folder_id = session["review_root_folder_id"] if session else None
            archived_parent_folder_id = ""
            if review_root_folder_id:
                roots = ensure_review_roots(client, review_root_folder_id)
                archived_parent_folder_id = archive_sample(
                    client,
                    roots,
                    sample_payload,
                    fresh["sample_folder_id"],
                    fresh.get("source_parent_folder_id", ""),
                    label,
                )

            exported_folder_ids = [folder_id for _, folder_id in exported]
            self.store.complete_export(item_id, exported_folder_ids, archived_parent_folder_id)

            # Update the action record so undo knows which Drive folders to clean up
            session_id = fresh.get("session_id") or fresh.get("claimed_by_session_id") or "shared"
            self._update_action_export_ids(session_id, item_id, exported_folder_ids, archived_parent_folder_id)

            logger.info("Export completed for item %s (%d targets)", item_id, len(exported))
        except (DriveClientError, OSError, ValueError) as exc:
            logger.warning("Export failed for item %s: %s", item_id, exc)
            self.store.fail_export(item_id, str(exc))
        except Exception:
            logger.exception("Unexpected export error for item %s", item_id)
            self.store.fail_export(item_id, traceback.format_exc()[:500])

    def _update_action_export_ids(
        self,
        session_id: str,
        item_id: int,
        exported_folder_ids: list[str],
        archived_parent_folder_id: str,
    ) -> None:
        """Best-effort update of the action record with Drive artifact IDs for undo."""
        try:
            import json
            from db import database_enabled, db_connection

            if database_enabled():
                with db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE video_review_actions
                            SET exported_folder_ids = %s::jsonb,
                                archive_parent_folder_id = %s
                            WHERE queue_item_id = %s
                              AND session_id = %s
                              AND undone = FALSE
                              AND action_type = 'label'
                            """,
                            (json.dumps(exported_folder_ids), archived_parent_folder_id, item_id, session_id),
                        )
                    conn.commit()
            else:
                import sqlite3
                db_path = self.store.db_path if hasattr(self.store, "db_path") else None
                if not db_path:
                    return
                conn = sqlite3.connect(db_path)
                try:
                    conn.execute(
                        """
                        UPDATE actions
                        SET exported_folder_ids_json = ?,
                            archive_parent_folder_id = ?
                        WHERE queue_item_id = ?
                          AND session_id = ?
                          AND undone = 0
                          AND action_type = 'label'
                        """,
                        (json.dumps(exported_folder_ids), archived_parent_folder_id, item_id, session_id),
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception:
            logger.debug("Could not update action export IDs for item %s", item_id, exc_info=True)
