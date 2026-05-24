"""Storage backend factory for AutoLabeler.

The app still defaults to Google Drive. Azure Blob is opt-in via
``STORAGE_BACKEND=azure`` so existing Railway deployments keep their current
behavior unless their environment is explicitly changed.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from drive_client import DriveClient


def storage_backend() -> str:
    backend = (os.environ.get("STORAGE_BACKEND") or "drive").strip().lower()
    if backend in {"azure_blob", "blob"}:
        return "azure"
    if backend in {"drive", "azure"}:
        return backend
    raise RuntimeError(f"Unsupported STORAGE_BACKEND={backend!r}; expected 'drive' or 'azure'.")


def storage_root_id() -> str:
    if storage_backend() == "azure":
        root = (
            os.environ.get("AZURE_PROJECT_ROOT_PREFIX")
            or os.environ.get("DRIVE_PROJECT_ROOT_FOLDER_ID")
            or "project-root"
        )
        return root.strip().strip("/") or "project-root"

    root = os.environ.get("DRIVE_PROJECT_ROOT_FOLDER_ID", "").strip()
    if not root:
        raise RuntimeError("DRIVE_PROJECT_ROOT_FOLDER_ID is not set in .env")
    return root


def create_storage_client(
    drive_client_cls: Callable[[], Any] = DriveClient,
) -> Any:
    if storage_backend() == "azure":
        from blob_client import AzureBlobClient

        return AzureBlobClient()
    return drive_client_cls()
