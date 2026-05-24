import pytest

import storage_client


def test_storage_backend_defaults_to_drive(monkeypatch):
    monkeypatch.delenv("STORAGE_BACKEND", raising=False)
    assert storage_client.storage_backend() == "drive"


def test_azure_root_uses_azure_prefix(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "azure")
    monkeypatch.setenv("AZURE_PROJECT_ROOT_PREFIX", "project-root")
    monkeypatch.delenv("DRIVE_PROJECT_ROOT_FOLDER_ID", raising=False)
    assert storage_client.storage_root_id() == "project-root"


def test_unsupported_storage_backend_fails_fast(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "sftp")
    with pytest.raises(RuntimeError, match="Unsupported STORAGE_BACKEND"):
        storage_client.storage_backend()
