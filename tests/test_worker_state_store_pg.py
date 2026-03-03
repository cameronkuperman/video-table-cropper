import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

from worker_state_store_pg import WorkerStateStorePG


class FakeCursor:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self.row = row

    def execute(self, query: str, params: tuple[Any, ...] | None = None) -> None:
        self.query = query
        self.params = params

    def fetchone(self) -> dict[str, Any] | None:
        return self.row

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class FakeConnection:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self.row = row

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.row)

    def commit(self) -> None:
        return None


class WorkerStateStorePGTests(unittest.TestCase):
    def test_get_status_marks_stale_heartbeat(self) -> None:
        stale_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=120)
        row = {
            "event_seq": 4,
            "state": "processing_video",
            "message": "Working",
            "worker_running": True,
            "current_video": {"name": "video-1.mp4"},
            "last_error": None,
            "stop_reason": None,
            "counters": {"sample_count_created": 12},
            "config": {"poll_seconds": 30},
            "heartbeat_at": stale_heartbeat,
            "updated_at": stale_heartbeat,
        }

        @contextmanager
        def fake_db_connection():
            yield FakeConnection(row)

        with patch("worker_state_store_pg.db_connection", fake_db_connection):
            status = WorkerStateStorePG().get_status()

        self.assertEqual(status["event_seq"], 4)
        self.assertEqual(status["state"], "processing_video")
        self.assertTrue(status["stale"])
        self.assertEqual(status["current_video"]["name"], "video-1.mp4")

    def test_get_status_returns_default_payload_without_row(self) -> None:
        @contextmanager
        def fake_db_connection():
            yield FakeConnection(None)

        with patch("worker_state_store_pg.db_connection", fake_db_connection):
            status = WorkerStateStorePG().get_status()

        self.assertEqual(status["state"], "unknown")
        self.assertFalse(status["worker_running"])
