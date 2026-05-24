"""Azure Blob Storage client with the DriveClient-shaped API used by AutoLabeler.

Azure Blob has no real folders. This adapter represents folders as prefixes and
stores folder-level Drive-style metadata in a ``.folder_meta.json`` sidecar.
Blob names are used as stable file IDs.
"""

from __future__ import annotations

import mimetypes
import os
import posixpath
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from drive_client import DriveClientError, FOLDER_MIME
from storage_client import storage_root_id

FOLDER_META_FILE = ".folder_meta.json"
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError, OSError)


class AzureBlobClientError(DriveClientError):
    """Raised when Azure Blob operations fail."""


def _clean_prefix(value: str) -> str:
    return str(value or "").strip().strip("/")


def _join(*parts: str) -> str:
    cleaned = [_clean_prefix(part) for part in parts if _clean_prefix(part)]
    return posixpath.join(*cleaned) if cleaned else ""


def _parent_of(blob_name: str) -> str:
    parent = posixpath.dirname(_clean_prefix(blob_name))
    return parent if parent != "." else ""


def _name_of(item_id: str) -> str:
    return posixpath.basename(_clean_prefix(item_id))


def _folder_meta_name(folder_id: str) -> str:
    return _join(folder_id, FOLDER_META_FILE)


def _guess_mime_type(path_or_name: str, fallback: str = "application/octet-stream") -> str:
    guessed, _encoding = mimetypes.guess_type(path_or_name)
    return guessed or fallback


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class AzureBlobClient:
    def __init__(self, connection_string: str | None = None, container_name: str | None = None) -> None:
        try:
            from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
            from azure.storage.blob import BlobServiceClient, ContentSettings
        except ImportError as exc:
            raise AzureBlobClientError(
                "Azure Blob dependencies are missing. Install azure-storage-blob."
            ) from exc

        self._resource_exists_error = ResourceExistsError
        self._resource_not_found_error = ResourceNotFoundError
        self._content_settings_cls = ContentSettings
        self.retry_attempts = max(1, int(os.environ.get("AZURE_BLOB_RETRY_ATTEMPTS", "4") or "4"))
        self.retry_base_seconds = max(0.1, float(os.environ.get("AZURE_BLOB_RETRY_BASE_SECONDS", "0.5") or "0.5"))

        conn = connection_string or os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip()
        if not conn:
            raise AzureBlobClientError("AZURE_STORAGE_CONNECTION_STRING is not configured.")
        container = (container_name or os.environ.get("AZURE_BLOB_CONTAINER", "")).strip()
        if not container:
            raise AzureBlobClientError("AZURE_BLOB_CONTAINER is not configured.")

        api_version = os.environ.get("AZURE_BLOB_API_VERSION", "2023-11-03").strip() or "2023-11-03"
        service = BlobServiceClient.from_connection_string(conn, api_version=api_version)
        self.container_name = container
        self.container = service.get_container_client(container)
        try:
            self.container.create_container()
        except ResourceExistsError:
            pass

        self.root_prefix = storage_root_id()
        self.ensure_subfolder("", self.root_prefix)

    @staticmethod
    def is_not_found_error(exc: Exception) -> bool:
        cause = exc.__cause__ if isinstance(exc, DriveClientError) else exc
        return cause.__class__.__name__ == "ResourceNotFoundError"

    def _execute_with_retry(self, operation, context: str):
        last_error: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                retryable = isinstance(exc, RETRYABLE_EXCEPTIONS) or status_code in {408, 425, 429, 500, 502, 503, 504}
                if not retryable or attempt >= self.retry_attempts:
                    break
                time.sleep(self.retry_base_seconds * (2 ** (attempt - 1)))
        if last_error is not None:
            raise AzureBlobClientError(f"{context}: {last_error}") from last_error
        raise AzureBlobClientError(context)

    def _blob(self, blob_name: str):
        return self.container.get_blob_client(_clean_prefix(blob_name))

    def _exists(self, blob_name: str) -> bool:
        return bool(self._execute_with_retry(lambda: self._blob(blob_name).exists(), f"Azure exists error for {blob_name}"))

    def _read_json_blob(self, blob_name: str) -> dict[str, Any] | None:
        if not self._exists(blob_name):
            return None
        import json

        raw = self.download_file_content(blob_name)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_json_blob(self, blob_name: str, payload: dict[str, Any]) -> None:
        import json

        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self._upload_blob(blob_name, data, "application/json")

    def _default_folder_meta(self, folder_id: str) -> dict[str, Any]:
        folder_id = _clean_prefix(folder_id)
        parent = _parent_of(folder_id)
        return {
            "id": folder_id,
            "name": _name_of(folder_id),
            "mimeType": FOLDER_MIME,
            "parents": [parent] if parent else [],
            "appProperties": {},
            "modifiedTime": _iso_now(),
            "trashed": False,
        }

    def _get_folder_meta(self, folder_id: str) -> dict[str, Any] | None:
        folder_id = _clean_prefix(folder_id)
        meta = self._read_json_blob(_folder_meta_name(folder_id))
        if meta:
            meta.setdefault("id", folder_id)
            meta.setdefault("name", _name_of(folder_id))
            meta.setdefault("mimeType", FOLDER_MIME)
            meta.setdefault("parents", [_parent_of(folder_id)] if _parent_of(folder_id) else [])
            meta.setdefault("appProperties", {})
            meta.setdefault("modifiedTime", _iso_now())
            meta.setdefault("trashed", False)
            return meta
        return None

    def _put_folder_meta(self, folder_id: str, meta: dict[str, Any]) -> None:
        folder_id = _clean_prefix(folder_id)
        payload = dict(meta)
        payload["id"] = folder_id
        payload["name"] = payload.get("name") or _name_of(folder_id)
        payload["mimeType"] = FOLDER_MIME
        payload["parents"] = list(payload.get("parents") or ([_parent_of(folder_id)] if _parent_of(folder_id) else []))
        payload["modifiedTime"] = _iso_now()
        self._write_json_blob(_folder_meta_name(folder_id), payload)

    def _upload_blob(self, blob_name: str, data: bytes, mime_type: str) -> dict[str, Any]:
        blob_name = _clean_prefix(blob_name)
        content_settings = self._content_settings_cls(content_type=mime_type)
        self._execute_with_retry(
            lambda: self._blob(blob_name).upload_blob(
                data,
                overwrite=True,
                content_settings=content_settings,
            ),
            f"Azure upload error for {blob_name}",
        )
        props = self._execute_with_retry(
            lambda: self._blob(blob_name).get_blob_properties(),
            f"Azure get properties error for {blob_name}",
        )
        return {
            "id": blob_name,
            "name": _name_of(blob_name),
            "mimeType": getattr(getattr(props, "content_settings", None), "content_type", None) or mime_type,
            "parents": [_parent_of(blob_name)] if _parent_of(blob_name) else [],
            "modifiedTime": getattr(props, "last_modified", None).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            if getattr(props, "last_modified", None)
            else _iso_now(),
        }

    def get_file(self, file_id: str, fields: str = "id,name,mimeType,parents") -> dict[str, Any]:
        item_id = _clean_prefix(file_id)
        folder_meta = self._get_folder_meta(item_id)
        if folder_meta:
            return dict(folder_meta)
        if not self._exists(item_id):
            raise AzureBlobClientError(f"Azure file not found: {item_id}")
        props = self._execute_with_retry(
            lambda: self._blob(item_id).get_blob_properties(),
            f"Azure get file error for {item_id}",
        )
        return {
            "id": item_id,
            "name": _name_of(item_id),
            "mimeType": getattr(getattr(props, "content_settings", None), "content_type", None)
            or _guess_mime_type(item_id),
            "parents": [_parent_of(item_id)] if _parent_of(item_id) else [],
            "appProperties": {},
            "modifiedTime": getattr(props, "last_modified", None).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            if getattr(props, "last_modified", None)
            else _iso_now(),
            "trashed": False,
        }

    def ensure_subfolder(self, parent_id: str, folder_name: str) -> str:
        folder_id = _join(parent_id, folder_name)
        existing = self._get_folder_meta(folder_id)
        if existing and not existing.get("trashed"):
            return folder_id
        meta = self._default_folder_meta(folder_id)
        self._put_folder_meta(folder_id, meta)
        return folder_id

    def list_folders(self, parent_id: str, fields: str = "id,name,mimeType,parents") -> list[dict[str, Any]]:
        parent = _clean_prefix(parent_id)
        prefix = f"{parent}/" if parent else ""
        folders: dict[str, dict[str, Any]] = {}
        for blob in self.container.list_blobs(name_starts_with=prefix):
            name = str(blob.name)
            rest = name[len(prefix):] if prefix else name
            if not rest or "/" not in rest:
                continue
            child_name = rest.split("/", 1)[0]
            child_id = _join(parent, child_name)
            if child_id in folders:
                continue
            meta = self._get_folder_meta(child_id) or self._default_folder_meta(child_id)
            if not meta.get("trashed"):
                folders[child_id] = meta
        return list(folders.values())

    def list_folders_recursive(self, parent_id: str, include_root: bool = True) -> list[dict[str, Any]]:
        discovered: dict[str, dict[str, Any]] = {}
        queue = [_clean_prefix(parent_id)]
        while queue:
            current = queue.pop(0)
            for child in self.list_folders(current):
                child_id = str(child["id"])
                if child_id in discovered:
                    continue
                discovered[child_id] = child
                queue.append(child_id)
        if include_root:
            root_meta = self._get_folder_meta(parent_id)
            if root_meta:
                discovered[str(root_meta["id"])] = root_meta
        return list(discovered.values())

    def list_files(self, folder_id: str, fields: str = "id,name,mimeType,parents") -> list[dict[str, Any]]:
        folder = _clean_prefix(folder_id)
        prefix = f"{folder}/" if folder else ""
        files: list[dict[str, Any]] = list(self.list_folders(folder))
        for blob in self.container.list_blobs(name_starts_with=prefix):
            name = str(blob.name)
            rest = name[len(prefix):] if prefix else name
            if not rest or "/" in rest or rest == FOLDER_META_FILE:
                continue
            files.append(
                {
                    "id": name,
                    "name": rest,
                    "mimeType": getattr(getattr(blob, "content_settings", None), "content_type", None)
                    or _guess_mime_type(rest),
                    "parents": [folder] if folder else [],
                    "modifiedTime": getattr(blob, "last_modified", None).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
                    if getattr(blob, "last_modified", None)
                    else _iso_now(),
                    "trashed": False,
                }
            )
        return files

    def list_image_files(self, folder_id: str) -> list[dict[str, Any]]:
        return [item for item in self.list_files(folder_id) if str(item.get("mimeType", "")).startswith("image/")]

    def find_file_by_name(
        self, folder_id: str, file_name: str, mime_type: str | None = None
    ) -> dict[str, Any] | None:
        matches = self.find_files_by_name(folder_id, file_name, mime_type=mime_type)
        return matches[0] if matches else None

    def find_files_by_name(
        self, folder_id: str, file_name: str, mime_type: str | None = None
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        if mime_type in (None, FOLDER_MIME):
            item = self._get_folder_meta(_join(folder_id, file_name))
            if item and not item.get("trashed"):
                matches.append(item)
            if mime_type == FOLDER_MIME:
                return matches
        candidates = [item for item in self.list_files(folder_id) if item.get("name") == file_name]
        if mime_type:
            candidates = [item for item in candidates if item.get("mimeType") == mime_type]
        folder_ids = {str(item.get("id")) for item in matches}
        matches.extend(item for item in candidates if str(item.get("id")) not in folder_ids)
        return matches

    def download_file_content(self, file_id: str) -> bytes:
        blob_name = _clean_prefix(file_id)
        return self._execute_with_retry(
            lambda: self._blob(blob_name).download_blob().readall(),
            f"Azure download error for {blob_name}",
        )

    def download_file_to_path(self, file_id: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(self.download_file_content(file_id))
        return output_path

    def upload_file(
        self,
        local_path: Path,
        parent_id: str,
        file_name: str | None = None,
        mime_type: str | None = None,
    ) -> dict[str, Any]:
        if not local_path.exists():
            raise AzureBlobClientError(f"Local file does not exist: {local_path}")
        target_name = file_name or local_path.name
        return self._upload_blob(
            _join(parent_id, target_name),
            local_path.read_bytes(),
            mime_type or _guess_mime_type(target_name),
        )

    def update_file(
        self,
        file_id: str,
        local_path: Path,
        file_name: str | None = None,
        mime_type: str | None = None,
    ) -> dict[str, Any]:
        target_id = _join(_parent_of(file_id), file_name) if file_name else _clean_prefix(file_id)
        return self.upload_file(local_path, _parent_of(target_id), _name_of(target_id), mime_type=mime_type)

    def upload_bytes(
        self,
        data: bytes,
        parent_id: str,
        file_name: str,
        mime_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        return self._upload_blob(_join(parent_id, file_name), data, mime_type)

    def update_bytes(
        self,
        file_id: str,
        data: bytes,
        file_name: str | None = None,
        mime_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        target_id = _join(_parent_of(file_id), file_name) if file_name else _clean_prefix(file_id)
        return self._upload_blob(target_id, data, mime_type)

    def upsert_bytes(
        self,
        parent_id: str,
        file_name: str,
        data: bytes,
        mime_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        return self.upload_bytes(data, parent_id, file_name, mime_type=mime_type)

    def upload_or_update_file(
        self,
        local_path: Path,
        parent_id: str,
        file_name: str | None = None,
        mime_type: str | None = None,
    ) -> dict[str, Any]:
        return self.upload_file(local_path, parent_id, file_name=file_name, mime_type=mime_type)

    def update_file_metadata(
        self,
        file_id: str,
        metadata: dict[str, Any],
        fields: str = "id,name,mimeType,parents,appProperties",
    ) -> dict[str, Any]:
        folder_id = _clean_prefix(file_id)
        meta = self._get_folder_meta(folder_id) or self._default_folder_meta(folder_id)
        if "appProperties" in metadata:
            app_properties = dict(meta.get("appProperties") or {})
            for key, value in (metadata.get("appProperties") or {}).items():
                if value is None:
                    app_properties.pop(key, None)
                else:
                    app_properties[key] = str(value)
            meta["appProperties"] = app_properties
        for key in ("name", "parents", "trashed"):
            if key in metadata:
                meta[key] = metadata[key]
        self._put_folder_meta(folder_id, meta)
        return self._get_folder_meta(folder_id) or meta

    def move_file(self, file_id: str, new_parent_id: str, remove_parent_id: str | None = None) -> dict[str, Any]:
        source_id = _clean_prefix(file_id)
        dest_id = _join(new_parent_id, _name_of(source_id))
        if source_id == dest_id:
            return self.get_file(source_id)

        folder_meta = self._get_folder_meta(source_id)
        if folder_meta:
            self.ensure_subfolder(new_parent_id, _name_of(source_id))
            for blob in list(self.container.list_blobs(name_starts_with=f"{source_id}/")):
                old_name = str(blob.name)
                new_name = f"{dest_id}/{old_name[len(source_id) + 1:]}"
                data = self.download_file_content(old_name)
                content_type = getattr(getattr(blob, "content_settings", None), "content_type", None) or _guess_mime_type(new_name)
                self._upload_blob(new_name, data, content_type)
                self.delete_file(old_name)
            moved_meta = dict(folder_meta)
            moved_meta["parents"] = [new_parent_id]
            moved_meta["name"] = _name_of(dest_id)
            app_properties = dict(moved_meta.get("appProperties") or {})
            for key, value in list(app_properties.items()):
                raw = str(value)
                if raw.startswith(f"{source_id}/"):
                    app_properties[key] = f"{dest_id}/{raw[len(source_id) + 1:]}"
            moved_meta["appProperties"] = app_properties
            self._put_folder_meta(dest_id, moved_meta)
            self.delete_file(_folder_meta_name(source_id))
            return self._get_folder_meta(dest_id) or moved_meta

        data = self.download_file_content(source_id)
        props = self.get_file(source_id)
        uploaded = self._upload_blob(dest_id, data, str(props.get("mimeType") or _guess_mime_type(dest_id)))
        self.delete_file(source_id)
        return uploaded

    def set_file_trashed(self, file_id: str, trashed: bool) -> dict[str, Any]:
        item_id = _clean_prefix(file_id)
        meta = self._get_folder_meta(item_id)
        if meta:
            meta["trashed"] = bool(trashed)
            self._put_folder_meta(item_id, meta)
            return self._get_folder_meta(item_id) or meta
        if trashed:
            self.delete_file(item_id)
        return {"id": item_id, "name": _name_of(item_id), "trashed": bool(trashed), "parents": [_parent_of(item_id)]}

    def trash_file(self, file_id: str) -> dict[str, Any]:
        return self.set_file_trashed(file_id, True)

    def untrash_file(self, file_id: str) -> dict[str, Any]:
        return self.set_file_trashed(file_id, False)

    def delete_file(self, file_id: str) -> None:
        item_id = _clean_prefix(file_id)
        if not item_id:
            return
        if self._exists(item_id):
            self._execute_with_retry(lambda: self._blob(item_id).delete_blob(), f"Azure delete error for {item_id}")
            return
        prefix = f"{item_id}/"
        for blob in list(self.container.list_blobs(name_starts_with=prefix)):
            self._execute_with_retry(
                lambda blob_name=str(blob.name): self._blob(blob_name).delete_blob(),
                f"Azure delete error for {blob.name}",
            )
