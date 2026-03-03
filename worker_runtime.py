"""Shared runtime-status helpers for the offline worker and Flask UI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

WORKER_STATE_DIR_NAME = "_worker_state"
WORKER_RUNTIME_FILE_NAME = "runtime_status.json"
WORKER_RUNTIME_SCHEMA_VERSION = 1


def resolve_processor_cache_dir(raw_value: str | Path | None = None, *, base_dir: Path | None = None) -> Path:
    raw_path = Path(raw_value or os.environ.get("PROCESSOR_LOCAL_CACHE_DIR", "worker_cache"))
    if not raw_path.is_absolute() and base_dir is not None:
        raw_path = base_dir / raw_path
    return raw_path


def worker_state_dir(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / WORKER_STATE_DIR_NAME


def worker_runtime_status_path(cache_dir: str | Path) -> Path:
    return worker_state_dir(cache_dir) / WORKER_RUNTIME_FILE_NAME


def default_worker_runtime_state() -> dict[str, Any]:
    return {
        "schema_version": WORKER_RUNTIME_SCHEMA_VERSION,
        "event_seq": 0,
        "state": "unknown",
        "message": "Worker has not reported status yet.",
        "worker_running": False,
        "updated_at_epoch": None,
        "current_video": None,
        "last_error": None,
        "stop_reason": None,
        "counters": {
            "videos_seen": 0,
            "videos_selected": 0,
            "videos_processed": 0,
            "videos_skipped": 0,
            "sample_count_created": 0,
            "pending_sample_count": 0,
        },
        "config": {},
    }


def load_worker_runtime_state(status_path: str | Path) -> dict[str, Any]:
    path = Path(status_path)
    state = default_worker_runtime_state()
    if not path.exists():
        return state

    try:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        state["state"] = "unavailable"
        state["message"] = "Worker status exists but could not be read."
        state["last_error"] = str(exc)
        return state

    if not isinstance(raw_payload, dict):
        state["state"] = "unavailable"
        state["message"] = "Worker status payload is not a JSON object."
        return state

    counters = dict(state["counters"])
    counters.update(raw_payload.get("counters") or {})
    config = dict(state["config"])
    config.update(raw_payload.get("config") or {})
    state.update(raw_payload)
    state["counters"] = counters
    state["config"] = config
    return state


def write_worker_runtime_state(status_path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(status_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)
    return path
