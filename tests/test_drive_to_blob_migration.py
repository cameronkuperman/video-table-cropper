import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import migrate_drive_to_blob as migration
from drive_client import FOLDER_MIME


class FakeDrive:
    def __init__(self):
        self.items = {}
        self.children = {}
        self.content = {}
        self.downloads = 0
        self.listed_folder_ids = []

    def add_folder(self, item_id, name, parent_id=None, app_properties=None):
        item = {
            "id": item_id,
            "name": name,
            "mimeType": FOLDER_MIME,
            "parents": [parent_id] if parent_id else [],
            "appProperties": app_properties or {},
        }
        self.items[item_id] = item
        self.children.setdefault(item_id, [])
        if parent_id:
            self.children.setdefault(parent_id, []).append(item_id)
        return item_id

    def add_file(self, item_id, name, parent_id, data, mime_type="image/jpeg"):
        item = {
            "id": item_id,
            "name": name,
            "mimeType": mime_type,
            "parents": [parent_id],
            "size": str(len(data)),
            "md5Checksum": hashlib.md5(data).hexdigest(),
        }
        self.items[item_id] = item
        self.content[item_id] = data
        self.children.setdefault(parent_id, []).append(item_id)
        return item_id

    def get_file(self, file_id, fields="id,name,mimeType,parents"):
        return dict(self.items[file_id])

    def list_files(self, folder_id, fields="id,name,mimeType,parents"):
        self.listed_folder_ids.append(folder_id)
        return [dict(self.items[item_id]) for item_id in self.children.get(folder_id, [])]

    def find_file_by_name(self, folder_id, file_name, mime_type=None):
        for item_id in self.children.get(folder_id, []):
            item = self.items[item_id]
            if item["name"] == file_name and (mime_type is None or item["mimeType"] == mime_type):
                return dict(item)
        return None

    def download_file_content(self, file_id):
        self.downloads += 1
        return self.content[file_id]


class FakeBlob:
    def __init__(self):
        self.folders = set()
        self.files = {}
        self.metadata = {}

    def ensure_subfolder(self, parent_id, folder_name):
        folder_id = migration._join(parent_id, folder_name)
        self.folders.add(folder_id)
        return folder_id

    def upload_bytes(self, data, parent_id, file_name, mime_type="application/octet-stream"):
        blob_id = migration._join(parent_id, file_name)
        self.files[blob_id] = bytes(data)
        return {"id": blob_id, "name": file_name, "mimeType": mime_type, "parents": [parent_id]}

    def update_file_metadata(self, file_id, metadata, fields="id,name,mimeType,parents,appProperties"):
        current = self.metadata.setdefault(file_id, {"appProperties": {}})
        current["appProperties"].update(metadata.get("appProperties") or {})
        return {"id": file_id, **current}

    def blob_size(self, blob_name):
        data = self.files.get(blob_name)
        return len(data) if data is not None else None


def _sample_drive():
    drive = FakeDrive()
    drive.add_folder("root", "AutoLabeler")
    drive.add_folder("unlabeled", "unlabeled", "root")
    drive.add_folder(
        "sample",
        "mimosas_table_1_t0001",
        "unlabeled",
        app_properties={
            "autolabel_queue_schema": "1",
            "autolabel_frame_0_id": "frame0",
            "autolabel_frame_1_id": "frame1",
        },
    )
    drive.add_file("frame0", "frame_0.jpg", "sample", b"frame-zero")
    drive.add_file("frame1", "frame_1.jpg", "sample", b"frame-one")
    drive.add_file("meta", "metadata.json", "sample", b'{"ok": true}', mime_type="application/json")
    return drive


def test_manifest_counts_and_paths():
    drive = _sample_drive()
    manifest = migration.build_manifest(drive, "root", "project-root")

    assert len(manifest.folders) == 3
    assert len(manifest.files) == 3
    assert manifest.known_bytes == len(b"frame-zero") + len(b"frame-one") + len(b'{"ok": true}')
    assert {item.dest_path for item in manifest.files} == {
        "project-root/unlabeled/mimosas_table_1_t0001/frame_0.jpg",
        "project-root/unlabeled/mimosas_table_1_t0001/frame_1.jpg",
        "project-root/unlabeled/mimosas_table_1_t0001/metadata.json",
    }


def test_migrate_translates_folder_app_properties(tmp_path):
    drive = _sample_drive()
    blob = FakeBlob()
    manifest = migration.build_manifest(drive, "root", "project-root")
    progress = migration.MigrationProgress(tmp_path / "progress.json", "root", "project-root")

    migration.migrate_items(drive, blob, manifest.items, progress, verify=True, skip_existing=True)

    folder_id = "project-root/unlabeled/mimosas_table_1_t0001"
    assert blob.files[f"{folder_id}/frame_0.jpg"] == b"frame-zero"
    assert blob.metadata[folder_id]["appProperties"]["autolabel_frame_0_id"] == f"{folder_id}/frame_0.jpg"
    assert blob.metadata[folder_id]["appProperties"]["autolabel_frame_1_id"] == f"{folder_id}/frame_1.jpg"
    assert blob.metadata[folder_id]["appProperties"]["autolabel_queue_schema"] == "1"


def test_existing_blob_is_skipped_and_progress_maps_file(tmp_path):
    drive = _sample_drive()
    blob = FakeBlob()
    existing_path = "project-root/unlabeled/mimosas_table_1_t0001/frame_0.jpg"
    blob.files[existing_path] = b"frame-zero"
    manifest = migration.build_manifest(drive, "root", "project-root")
    selected = [
        item
        for item in manifest.items
        if item.source_id in {"root", "unlabeled", "sample", "frame0"}
    ]
    progress = migration.MigrationProgress(tmp_path / "progress.json", "root", "project-root")

    migration.migrate_items(drive, blob, selected, progress, verify=True, skip_existing=True)

    assert drive.downloads == 0
    assert progress.file_map["frame0"] == existing_path


def test_deep_source_prefix_includes_ancestor_folders():
    drive = _sample_drive()
    manifest = migration.build_manifest(drive, "root", "project-root")

    selected = migration.select_items(
        manifest,
        source_path_prefixes=["unlabeled/mimosas_table_1_t0001"],
        limit_folders=None,
        limit_files=1,
    )

    assert {item.rel_path for item in selected if item.is_folder} == {
        "",
        "unlabeled",
        "unlabeled/mimosas_table_1_t0001",
    }


def test_scoped_manifest_scans_only_requested_prefix():
    drive = _sample_drive()
    drive.add_folder("clean", "clean", "root")
    drive.add_file("clean-frame", "frame_0.jpg", "clean", b"clean-frame")

    manifest = migration.build_scoped_manifest(
        drive,
        "root",
        "project-root",
        ["unlabeled"],
        progress_every=0,
    )

    assert {item.rel_path for item in manifest.folders} == {
        "",
        "unlabeled",
        "unlabeled/mimosas_table_1_t0001",
    }
    assert "project-root/clean/frame_0.jpg" not in {item.dest_path for item in manifest.files}


def test_fast_migration_does_not_prescan_unrelated_folders(tmp_path):
    drive = _sample_drive()
    drive.add_folder("clean", "clean", "root")
    drive.add_file("clean-frame", "frame_0.jpg", "clean", b"clean-frame")
    blob = FakeBlob()
    progress = migration.MigrationProgress(tmp_path / "progress.json", "root", "project-root")

    migration.migrate_prefix_without_scan(
        drive,
        blob,
        "root",
        "project-root",
        "unlabeled",
        progress,
        limit_folders=1,
        limit_files=None,
        verify=True,
        skip_existing=True,
    )

    folder_id = "project-root/unlabeled/mimosas_table_1_t0001"
    assert "clean" not in drive.listed_folder_ids
    assert blob.files[f"{folder_id}/frame_0.jpg"] == b"frame-zero"
    assert blob.metadata[folder_id]["appProperties"]["autolabel_frame_0_id"] == f"{folder_id}/frame_0.jpg"


def test_autolabeler_skeleton_matches_expected_blob_paths():
    blob = FakeBlob()

    created = migration.ensure_autolabeler_skeleton(blob, "project-root")

    expected = {
        "project-root/unlabeled",
        "project-root/clean",
        "project-root/dirty",
        "project-root/occupied",
        "project-root/label_later",
        "project-root/discarded",
        "project-root/raw_videos",
        "project-root/temp_processing",
        "project-root/processed_raw",
        "project-root/restaurant-pi-1",
        "project-root/restaurant-pi-1/unassociated",
        "project-root/restaurant-pi-1/unassociated_zips",
        "project-root/restaurant-pi-1/unlabeled",
        "project-root/restaurant-pi-1/processed_raw",
        "project-root/restaurant-pi-1/3frame/unlabeled",
        "project-root/reolink-matthews-01/crop_configs",
        "project-root/10frametrue/restaurant-pi-1",
        "project-root/10frametrue/reolink-matthews-01",
    }
    assert expected.issubset(blob.folders)
    assert expected.issubset(set(created))


def test_duplicate_destination_paths_are_reported():
    drive = FakeDrive()
    drive.add_folder("root", "AutoLabeler")
    drive.add_file("one", "frame_0.jpg", "root", b"one")
    drive.add_file("two", "frame_0.jpg", "root", b"two")

    manifest = migration.build_manifest(drive, "root", "project-root")

    assert manifest.duplicate_paths == {"project-root/frame_0.jpg": ["one", "two"]}
