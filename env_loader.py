"""Minimal local .env loader used by the app and offline worker."""

from __future__ import annotations

import os
from pathlib import Path


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_local_env(env_path: Path | None = None) -> Path:
    """Load KEY=VALUE pairs from the repo-local .env file if present.

    Existing process environment values win over .env values.
    """
    path = env_path or (Path(__file__).resolve().parent / ".env")
    if not path.exists():
        return path

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())
        if not key:
            continue
        os.environ.setdefault(key, value)
    return path
