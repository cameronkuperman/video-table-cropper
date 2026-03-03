"""Google Drive helper client for Drive-first labeling workflows."""

from __future__ import annotations

import base64
import io
import json
import os
import socket
import ssl
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"
RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
T = TypeVar("T")


class DriveClientError(RuntimeError):
    """Raised when Drive API calls fail or credentials are invalid."""


class DriveClient:
    """Thin wrapper around Google Drive v3 APIs used by this app."""

    def __init__(self, credentials_path: str | None = None) -> None:
        credentials = self._load_credentials(credentials_path)
        self.service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        self.retry_attempts = max(1, int(os.environ.get("DRIVE_API_RETRY_ATTEMPTS", "4") or "4"))
        self.retry_base_seconds = max(0.25, float(os.environ.get("DRIVE_API_RETRY_BASE_SECONDS", "1.0") or "1.0"))

    @staticmethod
    def _load_credentials(credentials_path: str | None = None):
        raw_json = os.environ.get("DRIVE_SERVICE_ACCOUNT_JSON")
        if raw_json:
            try:
                return service_account.Credentials.from_service_account_info(
                    json.loads(raw_json),
                    scopes=DRIVE_SCOPES,
                )
            except (json.JSONDecodeError, TypeError) as exc:
                raise DriveClientError(f"Invalid DRIVE_SERVICE_ACCOUNT_JSON: {exc}") from exc

        raw_b64 = os.environ.get("DRIVE_SERVICE_ACCOUNT_JSON_B64")
        if raw_b64:
            try:
                decoded = base64.b64decode(raw_b64).decode("utf-8")
                return service_account.Credentials.from_service_account_info(
                    json.loads(decoded),
                    scopes=DRIVE_SCOPES,
                )
            except (ValueError, json.JSONDecodeError, TypeError) as exc:
                raise DriveClientError(f"Invalid DRIVE_SERVICE_ACCOUNT_JSON_B64: {exc}") from exc

        creds_path = credentials_path or os.environ.get("DRIVE_SERVICE_ACCOUNT_JSON_PATH")
        if not creds_path:
            raise DriveClientError(
                "Drive credentials are not configured. Set DRIVE_SERVICE_ACCOUNT_JSON, "
                "DRIVE_SERVICE_ACCOUNT_JSON_B64, or DRIVE_SERVICE_ACCOUNT_JSON_PATH."
            )

        creds_file = Path(creds_path)
        if not creds_file.exists():
            raise DriveClientError(f"Service account file not found: {creds_file}")

        return service_account.Credentials.from_service_account_file(
            str(creds_file),
            scopes=DRIVE_SCOPES,
        )

    @staticmethod
    def _escape_query_value(value: str) -> str:
        return value.replace("'", "\\'")

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        if isinstance(exc, HttpError):
            status = getattr(getattr(exc, "resp", None), "status", None)
            return bool(status in RETRYABLE_HTTP_STATUSES)
        return isinstance(exc, (ssl.SSLError, socket.timeout, socket.gaierror, ConnectionError))

    def _execute_with_retry(self, operation: Callable[[], T], context: str) -> T:
        last_error: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                if not self._is_retryable_exception(exc) or attempt >= self.retry_attempts:
                    break
                time.sleep(self.retry_base_seconds * attempt)

        if isinstance(last_error, HttpError):
            raise DriveClientError(f"{context}: {last_error}") from last_error
        if last_error is not None:
            raise DriveClientError(f"{context}: {last_error}") from last_error
        raise DriveClientError(context)

    def _list_files(self, query: str, fields: str = "id,name,mimeType,parents") -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        page_token = None
        while True:
            response = self._execute_with_retry(
                lambda: (
                    self.service.files()
                    .list(
                        q=query,
                        fields=f"nextPageToken, files({fields})",
                        pageToken=page_token,
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    )
                    .execute()
                )
                ,
                "Drive list error",
            )
            results.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return results

    def get_file(self, file_id: str, fields: str = "id,name,mimeType,parents") -> dict[str, Any]:
        return self._execute_with_retry(
            lambda: (
                self.service.files()
                .get(
                    fileId=file_id,
                    fields=fields,
                    supportsAllDrives=True,
                )
                .execute()
            ),
            f"Drive get file error for {file_id}",
        )

    def list_folders(self, parent_id: str) -> list[dict[str, Any]]:
        escaped = self._escape_query_value(parent_id)
        query = f"'{escaped}' in parents and mimeType='{FOLDER_MIME}' and trashed=false"
        return self._list_files(query)

    def list_folders_recursive(self, parent_id: str, include_root: bool = True) -> list[dict[str, Any]]:
        """Recursively list all folders under a parent folder."""
        discovered: dict[str, dict[str, Any]] = {}
        queue: list[str] = [parent_id]

        while queue:
            current_parent = queue.pop(0)
            children = self.list_folders(current_parent)
            for child in children:
                child_id = child["id"]
                if child_id in discovered:
                    continue
                discovered[child_id] = child
                queue.append(child_id)

        if include_root:
            root_meta = self.get_file(parent_id)
            if root_meta.get("mimeType") == FOLDER_MIME:
                discovered[parent_id] = root_meta

        return list(discovered.values())

    def list_image_files(self, folder_id: str) -> list[dict[str, Any]]:
        escaped = self._escape_query_value(folder_id)
        query = f"'{escaped}' in parents and mimeType!='{FOLDER_MIME}' and trashed=false"
        files = self._list_files(query)
        return [f for f in files if str(f.get("mimeType", "")).startswith("image/")]

    def list_files(self, folder_id: str) -> list[dict[str, Any]]:
        escaped = self._escape_query_value(folder_id)
        query = f"'{escaped}' in parents and trashed=false"
        return self._list_files(query)

    def find_file_by_name(
        self, folder_id: str, file_name: str, mime_type: str | None = None
    ) -> dict[str, Any] | None:
        escaped_folder = self._escape_query_value(folder_id)
        escaped_name = self._escape_query_value(file_name)
        query = f"'{escaped_folder}' in parents and name='{escaped_name}' and trashed=false"
        if mime_type:
            escaped_mime = self._escape_query_value(mime_type)
            query += f" and mimeType='{escaped_mime}'"

        files = self._list_files(query)
        return files[0] if files else None

    def download_file_content(self, file_id: str) -> bytes:
        def _download_once() -> bytes:
            request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return buffer.getvalue()

        return self._execute_with_retry(
            _download_once,
            f"Drive download error for {file_id}",
        )

    def download_file_to_path(self, file_id: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.download_file_content(file_id)
        output_path.write_bytes(content)
        return output_path

    def ensure_subfolder(self, parent_id: str, folder_name: str) -> str:
        existing = self.find_file_by_name(parent_id, folder_name, mime_type=FOLDER_MIME)
        if existing:
            return existing["id"]

        metadata = {
            "name": folder_name,
            "mimeType": FOLDER_MIME,
            "parents": [parent_id],
        }
        created = self._execute_with_retry(
            lambda: (
                self.service.files()
                .create(
                    body=metadata,
                    fields="id,name",
                    supportsAllDrives=True,
                )
                .execute()
            ),
            "Drive create folder error",
        )
        if isinstance(created, dict) and created.get("id"):
            return str(created["id"])

        # Shared Drive folder creation can occasionally succeed while returning
        # an unexpectedly sparse payload. Re-read by name before failing.
        for attempt in range(3):
            fallback = self.find_file_by_name(parent_id, folder_name, mime_type=FOLDER_MIME)
            if fallback and fallback.get("id"):
                return str(fallback["id"])
            time.sleep(0.5 * (attempt + 1))

        raise DriveClientError(
            f"Drive created folder '{folder_name}' under '{parent_id}' but did not return an id. "
            f"Response payload: {created!r}"
        )

    def upload_file(
        self,
        local_path: Path,
        parent_id: str,
        file_name: str | None = None,
        mime_type: str | None = None,
    ) -> dict[str, Any]:
        if not local_path.exists():
            raise DriveClientError(f"Local file does not exist: {local_path}")

        metadata = {
            "name": file_name or local_path.name,
            "parents": [parent_id],
        }
        media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=False)

        created = self._execute_with_retry(
            lambda: (
                self.service.files()
                .create(
                    body=metadata,
                    media_body=media,
                    fields="id,name,mimeType,parents",
                    supportsAllDrives=True,
                )
                .execute()
            ),
            f"Drive upload error for {local_path.name}",
        )
        return created

    def update_file(
        self,
        file_id: str,
        local_path: Path,
        file_name: str | None = None,
        mime_type: str | None = None,
    ) -> dict[str, Any]:
        if not local_path.exists():
            raise DriveClientError(f"Local file does not exist: {local_path}")

        metadata: dict[str, Any] = {}
        if file_name:
            metadata["name"] = file_name
        media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=False)

        updated = self._execute_with_retry(
            lambda: (
                self.service.files()
                .update(
                    fileId=file_id,
                    body=metadata or None,
                    media_body=media,
                    fields="id,name,mimeType,parents",
                    supportsAllDrives=True,
                )
                .execute()
            ),
            f"Drive update error for {local_path.name}",
        )
        return updated

    def upload_bytes(
        self,
        data: bytes,
        parent_id: str,
        file_name: str,
        mime_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        metadata = {
            "name": file_name,
            "parents": [parent_id],
        }
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=False)

        created = self._execute_with_retry(
            lambda: (
                self.service.files()
                .create(
                    body=metadata,
                    media_body=media,
                    fields="id,name,mimeType,parents",
                    supportsAllDrives=True,
                )
                .execute()
            ),
            f"Drive upload bytes error for {file_name}",
        )
        return created

    def update_bytes(
        self,
        file_id: str,
        data: bytes,
        file_name: str | None = None,
        mime_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if file_name:
            metadata["name"] = file_name
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=False)

        updated = self._execute_with_retry(
            lambda: (
                self.service.files()
                .update(
                    fileId=file_id,
                    body=metadata or None,
                    media_body=media,
                    fields="id,name,mimeType,parents",
                    supportsAllDrives=True,
                )
                .execute()
            ),
            f"Drive update bytes error for {file_name or file_id}",
        )
        return updated

    def upsert_bytes(
        self,
        parent_id: str,
        file_name: str,
        data: bytes,
        mime_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        existing = self.find_file_by_name(parent_id, file_name)
        if existing:
            return self.update_bytes(existing["id"], data, file_name=file_name, mime_type=mime_type)
        return self.upload_bytes(data, parent_id=parent_id, file_name=file_name, mime_type=mime_type)

    def upload_or_update_file(
        self,
        local_path: Path,
        parent_id: str,
        file_name: str | None = None,
        mime_type: str | None = None,
    ) -> dict[str, Any]:
        target_name = file_name or local_path.name
        existing = self.find_file_by_name(parent_id, target_name)
        if existing:
            return self.update_file(existing["id"], local_path, file_name=target_name, mime_type=mime_type)
        return self.upload_file(local_path, parent_id=parent_id, file_name=target_name, mime_type=mime_type)

    def move_file(self, file_id: str, new_parent_id: str, remove_parent_id: str | None = None) -> dict[str, Any]:
        if not remove_parent_id:
            current = self.get_file(file_id, fields="id,parents")
            parent_ids = current.get("parents", [])
            remove_parent_id = ",".join(parent_ids)

        return self._execute_with_retry(
            lambda: (
                self.service.files()
                .update(
                    fileId=file_id,
                    addParents=new_parent_id,
                    removeParents=remove_parent_id,
                    fields="id,name,parents",
                    supportsAllDrives=True,
                )
                .execute()
            ),
            f"Drive move error for {file_id}",
        )

    def set_file_trashed(self, file_id: str, trashed: bool) -> dict[str, Any]:
        return self._execute_with_retry(
            lambda: (
                self.service.files()
                .update(
                    fileId=file_id,
                    body={"trashed": trashed},
                    fields="id,name,trashed,parents",
                    supportsAllDrives=True,
                )
                .execute()
            ),
            f"Drive trash toggle error for {file_id}",
        )

    def trash_file(self, file_id: str) -> dict[str, Any]:
        return self.set_file_trashed(file_id, True)

    def untrash_file(self, file_id: str) -> dict[str, Any]:
        return self.set_file_trashed(file_id, False)

    def delete_file(self, file_id: str) -> None:
        self._execute_with_retry(
            lambda: self.service.files().delete(fileId=file_id, supportsAllDrives=True).execute(),
            f"Drive delete error for {file_id}",
        )
