from __future__ import annotations

import json
import io
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

import app as label_app
import processor
from queue_metadata import extract_frame_ids_from_item, has_complete_frame_ids


@pytest.fixture(autouse=True)
def isolate_supabase_crop_env(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DB_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
    monkeypatch.delenv("DATABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("DB_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_DB_SCHEMA", raising=False)
    label_app._supabase_crop_cache.clear()
    label_app._set_supabase_crop_status(
        enabled=False,
        last_error=None,
        last_lookup_at=None,
        last_cache_hit=False,
        last_camera_source_id=None,
        last_table_count=0,
    )


def test_supabase_config_accepts_database_url_project_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "https://example-ref.supabase.co")
    monkeypatch.setenv("DATABASE_SERVICE_ROLE_KEY", "service-key")

    assert label_app._supabase_rest_config() == (
        "https://example-ref.supabase.co",
        "service-key",
        "public",
    )


def test_supabase_config_derives_project_url_from_postgres_database_url(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://postgres:password@db.example-ref.supabase.co:5432/postgres",
    )
    monkeypatch.setenv("DB_SERVICE_ROLE_KEY", "service-key")

    assert label_app._supabase_rest_config() == (
        "https://example-ref.supabase.co",
        "service-key",
        "public",
    )


def test_supabase_url_takes_precedence_over_database_url(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://primary-ref.supabase.co")
    monkeypatch.setenv("DATABASE_URL", "https://secondary-ref.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")

    assert label_app._supabase_rest_config() == (
        "https://primary-ref.supabase.co",
        "service-key",
        "public",
    )


class FakeDriveClient:
    def __init__(self) -> None:
        self.items: dict[str, dict] = {}
        self.children: dict[str, list[str]] = defaultdict(list)
        self.created_subfolders: list[tuple[str, str, str]] = []
        self.moves: list[tuple[str, str, str | None]] = []
        self.trashed: list[str] = []
        self._seed()

    def _seed(self) -> None:
        self._add_folder("project-root", "project-root")

        for folder_id, folder_name in (
            ("video-raw", "raw_videos"),
            ("video-temp", "temp_processing"),
            ("video-unlabeled", "unlabeled"),
            ("video-clean", "clean"),
            ("video-dirty", "dirty"),
            ("video-occupied", "occupied"),
            ("video-later", "label_later"),
            ("site-matthews", "reolink-matthews-01"),
            ("site-restaurant", "restaurant-pi-1"),
        ):
            self._add_folder(folder_id, folder_name, "project-root")

        self._add_folder("video-triplet", "ipc3_table-4_t0001", "video-unlabeled")
        self._add_triplet_files("video-triplet", "video")

        self._add_folder("m-unassociated", "unassociated", "site-matthews")
        self._add_folder("m-unlabeled", "unlabeled", "site-matthews")
        self._add_folder("m-crop-configs", "crop_configs", "site-matthews")
        self._add_file(
            "m-ch03-config",
            "CH-CH03.json",
            "m-crop-configs",
            mime_type="application/json",
            content=json.dumps(
                {
                    "version": 1,
                    "site_key": "reolink-matthews-01",
                    "channel_code": "CH-CH03",
                    "reference": {
                        "raw_folder_id": "m-ready",
                        "raw_folder_name": "Reolink-CH-CH03_t0002",
                        "frame_file_id": "mready-frame0",
                        "width": 1920,
                        "height": 1080,
                    },
                    "crops": [
                        {
                            "name": "table_top_1",
                            "polygon": [[0, 0], [50, 0], [50, 50], [0, 50]],
                        },
                        {
                            "name": "table_top_2",
                            "polygon": [[60, 0], [110, 0], [110, 50], [60, 50]],
                        },
                    ],
                }
            ).encode("utf-8"),
        )
        self._add_folder("m-ready", "Reolink-CH-CH03_t0002", "m-unassociated")
        self._add_triplet_files("m-ready", "mready", include_metadata=True)
        self._add_folder("m-missing", "Reolink-CH-CH03_t0003", "m-unassociated")
        self._add_file("m-missing-f0", "frame_0.jpg", "m-missing")
        self._add_file("m-missing-f1", "frame_1.jpg", "m-missing")

        self._add_folder("r-unassociated", "unassociated", "site-restaurant")
        self._add_folder("r-unlabeled", "unlabeled", "site-restaurant")
        self._add_folder("r-ready", "Reolink-CH-CH04_t0004", "r-unassociated")
        self._add_triplet_files("r-ready", "rready", include_metadata=False)

    def _add_folder(
        self,
        item_id: str,
        name: str,
        parent_id: str | None = None,
        app_properties: dict | None = None,
    ) -> None:
        item = {
            "id": item_id,
            "name": name,
            "mimeType": label_app.FOLDER_MIME,
            "parents": [parent_id] if parent_id else [],
            "appProperties": app_properties or {},
            "modifiedTime": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "trashed": False,
        }
        self.items[item_id] = item
        if parent_id:
            self.children[parent_id].append(item_id)

    def _add_file(
        self,
        item_id: str,
        name: str,
        parent_id: str,
        mime_type: str = "image/jpeg",
        content: bytes | None = None,
    ) -> None:
        item = {
            "id": item_id,
            "name": name,
            "mimeType": mime_type,
            "parents": [parent_id],
            "appProperties": {},
            "content": content if content is not None else b"{}",
            "modifiedTime": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "trashed": False,
        }
        self.items[item_id] = item
        self.children[parent_id].append(item_id)

    def _add_triplet_files(
        self,
        parent_id: str,
        prefix: str,
        include_metadata: bool = False,
    ) -> None:
        self._add_file(f"{prefix}-frame0", "frame_0.jpg", parent_id)
        self._add_file(f"{prefix}-frame1", "frame_1.jpg", parent_id)
        self._add_file(f"{prefix}-frame2", "frame_2.jpg", parent_id)
        if include_metadata:
            self._add_file(
                f"{prefix}-metadata",
                "metadata.json",
                parent_id,
                mime_type="application/json",
            )

    def _copy(self, item_id: str) -> dict:
        item = self.items[item_id]
        copied = dict(item)
        copied["parents"] = list(item.get("parents", []))
        if "appProperties" in item:
            copied["appProperties"] = dict(item.get("appProperties", {}))
        return copied

    def find_file_by_name(self, folder_id: str, file_name: str, mime_type: str | None = None) -> dict | None:
        for child_id in self.children.get(folder_id, []):
            child = self.items[child_id]
            if child["name"] != file_name:
                continue
            if mime_type and child.get("mimeType") != mime_type:
                continue
            return self._copy(child_id)
        return None

    def ensure_subfolder(self, parent_id: str, folder_name: str) -> str:
        existing = self.find_file_by_name(parent_id, folder_name, mime_type=label_app.FOLDER_MIME)
        if existing:
            return existing["id"]

        folder_id = f"{parent_id}:{folder_name}"
        self._add_folder(folder_id, folder_name, parent_id)
        self.created_subfolders.append((parent_id, folder_name, folder_id))
        return folder_id

    def list_folders(self, parent_id: str, fields: str = "") -> list[dict]:
        return [
            self._copy(child_id)
            for child_id in self.children.get(parent_id, [])
            if self.items[child_id].get("mimeType") == label_app.FOLDER_MIME
            and not self.items[child_id].get("trashed")
        ]

    def list_files(self, folder_id: str, fields: str = "") -> list[dict]:
        return [
            self._copy(child_id)
            for child_id in self.children.get(folder_id, [])
            if not self.items[child_id].get("trashed")
        ]

    def find_files_by_name(self, folder_id: str, file_name: str, mime_type: str | None = None) -> list[dict]:
        return [
            self._copy(child_id)
            for child_id in self.children.get(folder_id, [])
            if self.items[child_id]["name"] == file_name
            and not self.items[child_id].get("trashed")
            and (mime_type is None or self.items[child_id].get("mimeType") == mime_type)
        ]

    def get_file(self, file_id: str, fields: str = "") -> dict:
        return self._copy(file_id)

    def update_file_metadata(self, file_id: str, metadata: dict, fields: str = "") -> dict:
        app_properties = self.items[file_id].setdefault("appProperties", {})
        for key, value in (metadata.get("appProperties", {}) or {}).items():
            if value is None:
                app_properties.pop(key, None)
            else:
                app_properties[key] = value
        return self._copy(file_id)

    def move_file(self, file_id: str, new_parent_id: str, remove_parent_id: str | None = None) -> dict:
        current_parent = self.items[file_id].get("parents", [None])[0]
        if current_parent in self.children:
            self.children[current_parent] = [
                child_id for child_id in self.children[current_parent] if child_id != file_id
            ]
        self.items[file_id]["parents"] = [new_parent_id]
        self.items[file_id]["modifiedTime"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self.children[new_parent_id].append(file_id)
        self.moves.append((file_id, new_parent_id, remove_parent_id))
        return self._copy(file_id)

    def trash_file(self, file_id: str) -> dict:
        self.items[file_id]["trashed"] = True
        self.trashed.append(file_id)
        return self._copy(file_id)

    def _trash_duplicate_named_files(self, parent_id: str, file_name: str, keep_file_id: str) -> None:
        for item in self.find_files_by_name(parent_id, file_name):
            item_id = str(item.get("id") or "")
            if item_id and item_id != keep_file_id:
                self.trash_file(item_id)

    def download_file_to_path(self, file_id: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(self.items[file_id].get("content", b"fake-image"))
        return output_path

    def download_file_content(self, file_id: str) -> bytes:
        return self.items[file_id].get("content", b"{}")

    def upsert_bytes(
        self,
        parent_id: str,
        file_name: str,
        data: bytes,
        mime_type: str = "application/octet-stream",
    ) -> dict:
        existing = self.find_file_by_name(parent_id, file_name)
        if existing:
            self.items[existing["id"]]["mimeType"] = mime_type
            self.items[existing["id"]]["content"] = data
            self._trash_duplicate_named_files(parent_id, file_name, existing["id"])
            return self._copy(existing["id"])

        item_id = f"{parent_id}:{file_name}"
        self._add_file(item_id, file_name, parent_id, mime_type=mime_type, content=data)
        self._trash_duplicate_named_files(parent_id, file_name, item_id)
        return self._copy(item_id)

    def upload_or_update_file(
        self,
        local_path: Path,
        parent_id: str,
        file_name: str | None = None,
        mime_type: str | None = None,
    ) -> dict:
        target_name = file_name or local_path.name
        return self.upsert_bytes(
            parent_id,
            target_name,
            local_path.read_bytes(),
            mime_type=mime_type or "image/jpeg",
        )


def _fake_prepare_reolink_unlabeled_queue(fake: FakeDriveClient, context, target_unlabeled_count: int) -> int:
    if context.source != label_app.REOLINK_SOURCE or not context.seed_folder_id:
        return 0

    label_app._assert_manual_crop_setup_ready(fake, context)
    existing_names = label_app._existing_generated_folder_names(fake, context)
    raw_folders = fake.list_folders(context.seed_folder_id)
    generated_count = 0
    for raw_folder in raw_folders:
        mapped = label_app._mapped_camera_tables_for_reolink_folder(
            raw_folder["name"],
            site_key=context.site_key,
            client=fake,
        )
        if mapped is None:
            continue

        if raw_folder["name"] == "Reolink-CH-CH03_t0003":
            continue

        _channel_number, _camera, table_polygons = mapped
        max_tables = 2 if raw_folder["name"] == "Reolink-CH-CH03_t0002" else 1
        for table_id, *_rest in table_polygons[:max_tables]:
            label_source = label_app._resolve_label_source(context.source, context.site_key)
            legacy_name = label_app._derived_reolink_folder_name(raw_folder["name"], table_id)
            derived_name = label_app._apply_source_prefix(legacy_name, label_source)
            if derived_name in existing_names or legacy_name in existing_names:
                continue
            folder_id = fake.ensure_subfolder(context.input_folder_id, derived_name)
            fake._add_triplet_files(folder_id, derived_name.replace("/", "_"))
            fake.update_file_metadata(
                folder_id,
                {"appProperties": label_app.build_folder_app_properties({
                    "frame_0": f"{derived_name.replace('/', '_')}-frame0",
                    "frame_1": f"{derived_name.replace('/', '_')}-frame1",
                    "frame_2": f"{derived_name.replace('/', '_')}-frame2",
                })},
            )
            existing_names.add(derived_name)
            generated_count += 1
    return generated_count


@pytest.fixture()
def fake_drive(monkeypatch, tmp_path):
    fake = FakeDriveClient()

    monkeypatch.setenv("DRIVE_PROJECT_ROOT_FOLDER_ID", "project-root")
    monkeypatch.setenv("PREPROCESS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(label_app, "DriveClient", lambda: fake)
    monkeypatch.setattr(label_app, "get_client", lambda: fake)
    monkeypatch.setattr(
        label_app,
        "_thumbs_cache_ready",
        lambda frames: all(frames.get(key) for key in ("frame_0", "frame_1", "frame_2")),
    )
    monkeypatch.setattr(label_app, "_schedule_preview_prewarm", lambda hydrated_folders: 0)
    monkeypatch.setattr(
        label_app,
        "_schedule_folder_hydration_prewarm",
        lambda subfolders, start_idx, context: 0,
    )
    monkeypatch.setattr(
        label_app,
        "_prepare_reolink_unlabeled_queue",
        lambda client, context, target_unlabeled_count, current_visible_count=None: _fake_prepare_reolink_unlabeled_queue(
            fake,
            context,
            target_unlabeled_count,
        ),
    )
    original_list_source_subfolders = label_app._list_source_subfolders

    def fake_list_source_subfolders(client, context, force_refresh=False):
        if context.source == label_app.REOLINK_SOURCE:
            existing = original_list_source_subfolders(client, context, force_refresh=True)
            if not existing:
                _fake_prepare_reolink_unlabeled_queue(fake, context, target_unlabeled_count=10)
                label_app._invalidate_listing_cache(context.queue_key)
        return original_list_source_subfolders(client, context, force_refresh=force_refresh)

    monkeypatch.setattr(label_app, "_list_source_subfolders", fake_list_source_subfolders)
    monkeypatch.setattr(label_app, "_maybe_trigger_video_preprocess", lambda context, unlabeled_count: None)
    monkeypatch.setattr(label_app, "_schedule_label_job_worker", lambda: False)

    label_app._source_folder_ids_cache.clear()
    label_app._listing_cache.clear()
    label_app._listing_refresh_inflight.clear()
    label_app._hydrated_folder_cache.clear()
    label_app._preview_prewarm_inflight.clear()
    label_app._folder_prewarm_inflight.clear()
    label_app._reolink_preprocess_inflight.clear()
    label_app._camera_config_cache = None
    label_app._crop_config_cache.clear()
    label_app._label_job_worker_rerun_requested = False
    label_app._label_job_last_attempt_at = None
    label_app._label_job_rate_limit_cooldown_until = None
    label_app._label_job_rate_limit_cooldown_seconds = label_app.LABEL_JOB_RATE_LIMIT_COOLDOWN_SECONDS
    label_app._label_job_last_rate_limit_error = None

    return fake


def _drain_label_jobs(fake_drive: FakeDriveClient) -> int:
    return label_app._drain_label_jobs_once(fake_drive, force_due=True)


def _label_payload(folder: dict, label: str) -> dict:
    return {
        "folder_id": folder["folder_id"],
        "parent_id": folder["parent_id"],
        "label": label,
        "source": folder["source"],
        "site_key": folder["site_key"],
        "folder_name": folder.get("folder_name"),
        "frames": folder.get("frames"),
        "frame_signature": folder.get("frame_signature"),
        "content_signature": folder.get("content_signature"),
    }


def _jpeg_bytes(width: int = 80, height: int = 60, color: tuple[int, int, int] = (20, 40, 60)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def client(fake_drive):
    return label_app.app.test_client()


def test_api_sources_exposes_video_and_reolink_sites(client):
    response = client.get("/api/sources")

    assert response.status_code == 200
    payload = response.get_json()
    assert [source["source"] for source in payload["sources"]] == ["video", "reolink"]
    assert [site["site_key"] for site in payload["reolink_sites"]] == [
        "restaurant-pi-1",
        "reolink-matthews-01",
    ]
    sites_by_key = {site["site_key"]: site for site in payload["reolink_sites"]}
    matthews_site = sites_by_key["reolink-matthews-01"]
    assert matthews_site["manual_crop_configs"] is True
    assert matthews_site["crop_editor_url"] == "/crop-editor?site=reolink-matthews-01"
    assert matthews_site["label"] == "Matthews"
    assert sites_by_key["restaurant-pi-1"]["label"] == "Mimosas (Photos)"
    assert payload["default_source"]["source"] == "reolink"
    assert payload["default_source"]["site_key"] == "restaurant-pi-1"


def test_reolink_hydration_normalizes_legacy_ten_frame_perception(fake_drive):
    fake_drive._add_folder("r-3frame", "3frame", "site-restaurant")
    fake_drive._add_folder("r-3frame-unlabeled", "unlabeled", "r-3frame")
    fake_drive._add_folder("sr-artifact", "front-camera_table_top_1_t0015", "r-3frame-unlabeled")
    fake_drive._add_triplet_files("sr-artifact", "sr-artifact", include_metadata=False)
    fake_drive._add_file(
        "sr-artifact-metadata",
        "metadata.json",
        "sr-artifact",
        mime_type="application/json",
        content=json.dumps({"source_frame_count": 10}).encode("utf-8"),
    )
    fake_drive._add_file(
        "sr-artifact-perception",
        "perception.json",
        "sr-artifact",
        mime_type="application/json",
        content=json.dumps({"schema_version": 2, "n_frames": 10}).encode("utf-8"),
    )
    label_app._source_folder_ids_cache.clear()

    context = label_app._resolve_queue_context(fake_drive, label_app.REOLINK_SOURCE, "restaurant-pi-1")
    folder = fake_drive.get_file("sr-artifact", fields="id,name,mimeType,parents,appProperties")
    payload = label_app._hydrate_folder(fake_drive, context, folder)

    assert payload is not None
    assert payload["parent_id"] == "r-3frame-unlabeled"
    assert payload["source_label"] == label_app.SCREENRECORD_TRUE_TEN_FOLDER_NAME
    assert payload["perception_file_name"] == label_app.PERCEPTION_V2_FILE_NAME
    assert fake_drive.find_file_by_name("sr-artifact", label_app.PERCEPTION_V2_FILE_NAME) is not None


def test_reolink_site_loads_without_unassociated_when_screenrecord_artifacts_exist(monkeypatch, tmp_path):
    fake = FakeDriveClient()
    monkeypatch.setenv("DRIVE_PROJECT_ROOT_FOLDER_ID", "project-root")
    monkeypatch.setenv("PREPROCESS_STATE_DIR", str(tmp_path))
    fake.children["site-restaurant"] = [
        child_id for child_id in fake.children["site-restaurant"] if child_id != "r-unassociated"
    ]
    fake.items.pop("r-unassociated")
    fake._add_folder("r-3frame", "3frame", "site-restaurant")
    fake._add_folder("r-3frame-unlabeled", "unlabeled", "r-3frame")
    fake._add_folder("sr-artifact-ready", "mimosas-IPC4_table_top_1_t0020", "r-3frame-unlabeled")
    fake._add_triplet_files("sr-artifact-ready", "sr-ready", include_metadata=True)
    label_app._source_folder_ids_cache.clear()
    label_app._listing_cache.clear()
    label_app._hydrated_folder_cache.clear()

    context = label_app._resolve_queue_context(fake, label_app.REOLINK_SOURCE, "restaurant-pi-1")
    listing = label_app._list_source_subfolders(fake, context, force_refresh=True)
    payload = label_app._hydrate_folder(fake, context, listing[0])

    assert context.seed_folder_id is None
    assert context.folder_ids[label_app.SCREENRECORD_THREE_FRAME_UNLABELED_KEY] == "r-3frame-unlabeled"
    assert [folder["name"] for folder in listing] == ["mimosas-IPC4_table_top_1_t0020"]
    assert payload["source_label"] == label_app.SCREENRECORD_TRUE_TEN_FOLDER_NAME


def test_reolink_hydration_ignores_ambiguous_perception(fake_drive):
    fake_drive._add_folder("artifact-no-ten", "front-camera_table_top_1_t0016", "r-unlabeled")
    fake_drive._add_triplet_files("artifact-no-ten", "artifact-no-ten", include_metadata=False)
    fake_drive._add_file(
        "artifact-no-ten-perception",
        "perception.json",
        "artifact-no-ten",
        mime_type="application/json",
        content=json.dumps({"schema_version": 1, "n_frames": 3}).encode("utf-8"),
    )

    context = label_app._resolve_queue_context(fake_drive, label_app.REOLINK_SOURCE, "restaurant-pi-1")
    folder = fake_drive.get_file("artifact-no-ten", fields="id,name,mimeType,parents,appProperties")
    payload = label_app._hydrate_folder(fake_drive, context, folder)

    assert payload is not None
    assert payload["perception_file_id"] is None
    assert fake_drive.find_file_by_name("artifact-no-ten", label_app.PERCEPTION_V2_FILE_NAME) is None


def test_screenrecord_true_ten_generation_creates_three_crops_and_perception(monkeypatch, tmp_path):
    import person_detector

    fake = FakeDriveClient()
    monkeypatch.setenv("DRIVE_PROJECT_ROOT_FOLDER_ID", "project-root")
    monkeypatch.setenv("PREPROCESS_STATE_DIR", str(tmp_path))
    label_app._source_folder_ids_cache.clear()
    label_app._listing_cache.clear()
    label_app._hydrated_folder_cache.clear()

    fake._add_folder("sr-10-root", "10frametrue", "project-root")
    fake._add_folder("sr-10-node", "restaurant-pi-1", "sr-10-root")
    fake._add_folder("sr-raw", "IPC4_t0015", "sr-10-node")
    for idx in range(10):
        fake._add_file(
            f"sr-raw-frame-{idx}",
            f"frame_{idx}.jpg",
            "sr-raw",
            content=_jpeg_bytes(color=(idx * 10, 20, 40)),
        )
    fake._add_file(
        "sr-raw-metadata",
        "metadata.json",
        "sr-raw",
        mime_type="application/json",
        content=json.dumps(
            {
                "site_id": "site-1",
                "node_id": "restaurant-pi-1",
                "camera_id": "IPC4",
                "camera_name": "IPC4",
                "triplet_index": 15,
                "triplet_stem": "IPC4",
                "captured_at_utc": [f"2026-03-12T00:00:{idx * 3:02d}Z" for idx in range(10)],
            }
        ).encode("utf-8"),
    )
    fake._add_folder("r-3frame", "3frame", "site-restaurant")
    fake._add_folder("r-3frame-unlabeled", "unlabeled", "r-3frame")

    monkeypatch.setattr(label_app, "_get_yolo_model", lambda: object())
    monkeypatch.setattr(person_detector, "detect_people_in_frame", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        label_app,
        "_mapped_camera_tables_for_screenrecord_folder",
        lambda *_args, **_kwargs: (
            4,
            {
                "camera_number": 4,
                "image_width": 80,
                "image_height": 60,
                "_table_metadata_by_id": {
                    "table_top_1": {
                        "label": "Table 23",
                        "restaurant_id": "restaurant-uuid",
                        "table_id": "table-uuid",
                        "camera_source_id": "camera-source-uuid",
                        "table_camera_crops_id": "crop-uuid",
                        "crop_source": "supabase_table_camera_crops",
                    }
                },
            },
            [("table_top_1", [(0, 0), (40, 0), (40, 40), (0, 40)], (0, 0, 40, 40), [(0, 0), (50, 0), (50, 50), (0, 50)])],
        ),
    )

    context = label_app._resolve_queue_context(fake, label_app.REOLINK_SOURCE, "restaurant-pi-1")
    generated = label_app._prepare_reolink_unlabeled_queue(
        fake,
        context,
        target_unlabeled_count=1,
    )

    assert generated == 1
    artifact = fake.find_file_by_name("r-3frame-unlabeled", "mimosas-IPC4_table_top_1_t0015", mime_type=label_app.FOLDER_MIME)
    assert artifact is not None
    files = {item["name"]: item for item in fake.list_files(artifact["id"])}
    assert {"frame_0.jpg", "frame_1.jpg", "frame_2.jpg", "metadata.json", label_app.PERCEPTION_V2_FILE_NAME} <= set(files)
    metadata = json.loads(fake.download_file_content(files["metadata.json"]["id"]).decode("utf-8"))
    assert metadata["restaurant_id"] == "restaurant-uuid"
    assert metadata["supabase_table_id"] == "table-uuid"
    assert metadata["camera_source_id"] == "camera-source-uuid"
    assert metadata["table_camera_crops_id"] == "crop-uuid"
    assert metadata["table"]["label"] == "Table 23"
    perception = json.loads(fake.download_file_content(files[label_app.PERCEPTION_V2_FILE_NAME]["id"]).decode("utf-8"))
    assert perception["schema_version"] == 2
    assert perception["n_frames"] == 10


def test_screenrecord_generation_dedupes_existing_metadata_identity(monkeypatch, tmp_path):
    fake = FakeDriveClient()
    monkeypatch.setenv("DRIVE_PROJECT_ROOT_FOLDER_ID", "project-root")
    monkeypatch.setenv("PREPROCESS_STATE_DIR", str(tmp_path))
    label_app._source_folder_ids_cache.clear()
    label_app._listing_cache.clear()
    label_app._hydrated_folder_cache.clear()
    fake.children["site-restaurant"] = [
        child_id for child_id in fake.children["site-restaurant"] if child_id != "r-unassociated"
    ]
    fake.items.pop("r-unassociated")

    fake._add_folder("sr-10-root", "10frametrue", "project-root")
    fake._add_folder("sr-10-node", "restaurant-pi-1", "sr-10-root")
    fake._add_folder("sr-raw-dupe", "IPC4_t0030", "sr-10-node")
    for idx in range(10):
        fake._add_file(
            f"sr-dupe-frame-{idx}",
            f"frame_{idx}.jpg",
            "sr-raw-dupe",
            content=_jpeg_bytes(color=(idx * 10, 50, 70)),
        )
    fake._add_file(
        "sr-dupe-metadata",
        "metadata.json",
        "sr-raw-dupe",
        mime_type="application/json",
        content=json.dumps({"camera_id": "IPC4", "triplet_stem": "IPC4", "triplet_index": 30}).encode("utf-8"),
    )
    fake._add_folder("r-3frame", "3frame", "site-restaurant")
    fake._add_folder("r-3frame-unlabeled", "unlabeled", "r-3frame")
    fake._add_folder("uuid-looking-artifact-folder", "d1cf6d88-4212-4ef2-9f7f-2d4cda0b2d2d", "r-3frame-unlabeled")
    fake._add_triplet_files("uuid-looking-artifact-folder", "dupe", include_metadata=False)
    fake._add_file(
        "uuid-looking-artifact-metadata",
        "metadata.json",
        "uuid-looking-artifact-folder",
        mime_type="application/json",
        content=json.dumps(
            {
                "triplet_stem": "IPC4",
                "triplet_index": 30,
                "table_camera_crops_id": "crop-uuid",
                "supabase_table_id": "table-uuid",
            }
        ).encode("utf-8"),
    )
    monkeypatch.setattr(
        label_app,
        "_mapped_camera_tables_for_screenrecord_folder",
        lambda *_args, **_kwargs: (
            4,
            {
                "camera_number": 4,
                "image_width": 80,
                "image_height": 60,
                "_table_metadata_by_id": {
                    "table_top_1": {
                        "table_id": "table-uuid",
                        "table_camera_crops_id": "crop-uuid",
                    }
                },
            },
            [("table_top_1", [(0, 0), (40, 0), (40, 40), (0, 40)], (0, 0, 40, 40), [(0, 0), (50, 0), (50, 50), (0, 50)])],
        ),
    )

    def fail_materialize(*_args, **_kwargs):
        raise AssertionError("matching metadata identity should prevent a duplicate artifact")

    monkeypatch.setattr(label_app, "_materialize_screenrecord_true_ten_artifacts", fail_materialize)

    context = label_app._resolve_queue_context(fake, label_app.REOLINK_SOURCE, "restaurant-pi-1")
    generated = label_app._prepare_reolink_unlabeled_queue(
        fake,
        context,
        target_unlabeled_count=2,
    )

    assert generated == 0


def test_screenrecord_artifacts_count_toward_reolink_generation_target(monkeypatch, tmp_path):
    fake = FakeDriveClient()
    monkeypatch.setenv("DRIVE_PROJECT_ROOT_FOLDER_ID", "project-root")
    monkeypatch.setenv("PREPROCESS_STATE_DIR", str(tmp_path))
    label_app._source_folder_ids_cache.clear()
    label_app._listing_cache.clear()
    label_app._hydrated_folder_cache.clear()

    fake._add_folder("sr-10-root", "10frametrue", "project-root")
    fake._add_folder("sr-10-node", "restaurant-pi-1", "sr-10-root")
    fake._add_folder("sr-raw-ready", "IPC4_t0015", "sr-10-node")
    fake._add_folder("r-3frame", "3frame", "site-restaurant")
    fake._add_folder("r-3frame-unlabeled", "unlabeled", "r-3frame")
    fake._add_folder("sr-ready-artifact", "mimosas-IPC4_table_top_1_t0015", "r-3frame-unlabeled")
    fake._add_triplet_files("sr-ready-artifact", "sr-ready-artifact", include_metadata=True)

    def fail_materialize(*_args, **_kwargs):
        raise AssertionError("ready ScreenRecord artifacts should satisfy the target")

    monkeypatch.setattr(label_app, "_materialize_screenrecord_true_ten_artifacts", fail_materialize)

    context = label_app._resolve_queue_context(fake, label_app.REOLINK_SOURCE, "restaurant-pi-1")
    generated = label_app._prepare_reolink_unlabeled_queue(
        fake,
        context,
        target_unlabeled_count=1,
    )

    assert generated == 0


def test_screenrecord_true_ten_generation_creates_missing_three_frame_branch(monkeypatch, tmp_path):
    import person_detector

    fake = FakeDriveClient()
    monkeypatch.setenv("DRIVE_PROJECT_ROOT_FOLDER_ID", "project-root")
    monkeypatch.setenv("PREPROCESS_STATE_DIR", str(tmp_path))
    label_app._source_folder_ids_cache.clear()
    label_app._listing_cache.clear()
    label_app._hydrated_folder_cache.clear()

    fake._add_folder("sr-10-root", "10frametrue", "project-root")
    fake._add_folder("sr-10-node", "restaurant-pi-1", "sr-10-root")
    fake._add_folder("sr-raw-no-branch", "IPC4_t0016", "sr-10-node")
    for idx in range(10):
        fake._add_file(
            f"sr-raw-no-branch-frame-{idx}",
            f"frame_{idx}.jpg",
            "sr-raw-no-branch",
            content=_jpeg_bytes(color=(idx * 10, 30, 50)),
        )
    fake._add_file(
        "sr-raw-no-branch-metadata",
        "metadata.json",
        "sr-raw-no-branch",
        mime_type="application/json",
        content=json.dumps({"camera_id": "IPC4", "camera_name": "IPC4", "triplet_index": 16}).encode("utf-8"),
    )

    monkeypatch.setattr(label_app, "_get_yolo_model", lambda: object())
    monkeypatch.setattr(person_detector, "detect_people_in_frame", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        label_app,
        "_mapped_camera_tables_for_screenrecord_folder",
        lambda *_args, **_kwargs: (
            4,
            {"camera_number": 4, "image_width": 80, "image_height": 60},
            [("table_top_1", [(0, 0), (40, 0), (40, 40), (0, 40)], (0, 0, 40, 40), [(0, 0), (50, 0), (50, 50), (0, 50)])],
        ),
    )

    context = label_app._resolve_queue_context(fake, label_app.REOLINK_SOURCE, "restaurant-pi-1")
    generated = label_app._prepare_reolink_unlabeled_queue(
        fake,
        context,
        target_unlabeled_count=1,
    )

    assert generated == 1
    three_frame = fake.find_file_by_name("site-restaurant", "3frame", mime_type=label_app.FOLDER_MIME)
    assert three_frame is not None
    unlabeled = fake.find_file_by_name(three_frame["id"], "unlabeled", mime_type=label_app.FOLDER_MIME)
    assert unlabeled is not None
    artifact = fake.find_file_by_name(unlabeled["id"], "mimosas-IPC4_table_top_1_t0016", mime_type=label_app.FOLDER_MIME)
    assert artifact is not None


def test_queue_is_source_aware_and_reolink_filters_incomplete_triplets(client):
    video_response = client.get("/api/queue?source=video&limit=10")
    video_payload = video_response.get_json()

    assert video_response.status_code == 200
    assert video_payload["source_context"]["queue_key"] == "video"
    assert [folder["folder_name"] for folder in video_payload["folders"]] == ["ipc3_table-4_t0001"]

    matthews_response = client.get("/api/queue?source=reolink&site=reolink-matthews-01&limit=10")
    matthews_payload = matthews_response.get_json()

    assert matthews_response.status_code == 200
    assert matthews_payload["source_context"]["queue_key"] == "reolink:reolink-matthews-01"
    assert [folder["folder_name"] for folder in matthews_payload["folders"]] == [
        "matthews-Reolink-CH-CH03_table_top_1_t0002",
        "matthews-Reolink-CH-CH03_table_top_2_t0002",
    ]
    assert matthews_payload["total_unlabeled"] == 2

    restaurant_response = client.get("/api/queue?source=reolink&site=restaurant-pi-1&limit=10")
    restaurant_payload = restaurant_response.get_json()

    assert restaurant_response.status_code == 200
    assert restaurant_payload["source_context"]["queue_key"] == "reolink:restaurant-pi-1"
    assert [folder["folder_name"] for folder in restaurant_payload["folders"]] == [
        "mimosas-Reolink-CH-CH04_table_top_1_t0004"
    ]


def test_queue_metadata_extracts_contiguous_and_sparse_frame_sets():
    legacy = {
        "appProperties": {
            "autolabel_frame_0_id": "f0",
            "autolabel_frame_1_id": "f1",
            "autolabel_frame_2_id": "f2",
        }
    }
    sparse = {
        "appProperties": {
            "autolabel_frame_0_id": "f0",
            "autolabel_frame_5_id": "f5",
            "autolabel_frame_9_id": "f9",
        }
    }
    damaged = {
        "appProperties": {
            "autolabel_frame_0_id": "f0",
            "autolabel_frame_2_id": "f2",
        }
    }
    full = {
        "appProperties": {
            f"autolabel_frame_{idx}_id": f"f{idx}"
            for idx in range(10)
        }
    }

    assert list(extract_frame_ids_from_item(legacy)) == ["frame_0", "frame_1", "frame_2"]
    assert extract_frame_ids_from_item(sparse) == {
        "frame_0": "f0",
        "frame_5": "f5",
        "frame_9": "f9",
    }
    assert has_complete_frame_ids(extract_frame_ids_from_item(sparse)) is True
    assert extract_frame_ids_from_item(damaged) == {
        "frame_0": "f0",
        "frame_1": None,
        "frame_2": "f2",
    }
    assert has_complete_frame_ids(extract_frame_ids_from_item(damaged)) is False
    assert list(extract_frame_ids_from_item(full)) == [f"frame_{idx}" for idx in range(10)]


def test_sparse_sample_file_listing_hydrates_present_frame_keys():
    files = [
        {"name": "frame_0.jpg", "id": "f0"},
        {"name": "frame_5.jpg", "id": "f5"},
        {"name": "frame_9.jpg", "id": "f9"},
    ]

    assert label_app._frame_payload_from_files(files) == {
        "frame_0": "f0",
        "frame_5": "f5",
        "frame_9": "f9",
    }


def test_sparse_non_sample_file_listing_hydrates_missing_slots():
    files = [
        {"name": "frame_0.jpg", "id": "f0"},
        {"name": "frame_2.jpg", "id": "f2"},
    ]

    frames = label_app._frame_payload_from_files(files)

    assert frames == {
        "frame_0": "f0",
        "frame_1": None,
        "frame_2": "f2",
    }
    assert has_complete_frame_ids(frames) is False


def test_queue_returns_uncached_fallback_when_no_frames_are_cached(client, fake_drive, monkeypatch):
    monkeypatch.setattr(label_app, "_thumbs_cache_ready", lambda frames: False)
    label_app._hydrated_folder_cache.clear()

    response = client.get("/api/queue?source=video&limit=10")
    payload = response.get_json()

    assert response.status_code == 200
    assert [folder["folder_name"] for folder in payload["folders"]] == ["ipc3_table-4_t0001"]
    assert payload["folders"][0]["cache_ready"] is False
    assert payload["ready_buffer_count"] == 0
    assert payload["warming_count"] >= 1


def test_queue_prefers_cached_folders_when_available(client):
    response = client.get("/api/queue?source=video&limit=10")
    payload = response.get_json()

    assert response.status_code == 200
    assert [folder["folder_name"] for folder in payload["folders"]] == ["ipc3_table-4_t0001"]
    assert payload["folders"][0]["cache_ready"] is True
    assert payload["ready_buffer_count"] == 1
    assert payload["folders"][0]["frame_signature"] == "video-frame0|video-frame1|video-frame2"
    assert payload["folders"][0]["thumb_urls"]["frame_0"].startswith("/api/thumb/")
    assert payload["folders"][0]["preview_urls"]["frame_0"].startswith("/api/preview/")


def test_queue_ignores_stale_frame_metadata_from_another_folder(client, fake_drive):
    fake_drive._add_folder("video-triplet-stale-meta", "ipc3_table-4_t0002", "video-unlabeled")
    fake_drive._add_triplet_files("video-triplet-stale-meta", "fresh-video")
    fake_drive.update_file_metadata(
        "video-triplet-stale-meta",
        {
            "appProperties": label_app.build_folder_app_properties(
                {
                    "frame_0": "video-frame0",
                    "frame_1": "video-frame1",
                    "frame_2": "video-frame2",
                }
            )
        },
    )
    label_app._listing_cache.clear()
    label_app._hydrated_folder_cache.clear()

    response = client.get("/api/queue?source=video&limit=10&refresh=1")
    payload = response.get_json()

    assert response.status_code == 200
    stale_folder = next(
        folder
        for folder in payload["folders"]
        if folder["folder_id"] == "video-triplet-stale-meta"
    )
    assert stale_folder["frames"] == {
        "frame_0": "fresh-video-frame0",
        "frame_1": "fresh-video-frame1",
        "frame_2": "fresh-video-frame2",
    }
    assert stale_folder["frame_signature"] == "fresh-video-frame0|fresh-video-frame1|fresh-video-frame2"
    assert fake_drive.items["video-triplet-stale-meta"]["appProperties"]["autolabel_frame_0_id"] == "fresh-video-frame0"


def test_drive_upsert_trashes_duplicate_same_name_siblings(fake_drive, tmp_path):
    folder_id = fake_drive.ensure_subfolder("video-unlabeled", "dupe-output")
    fake_drive._add_file("dupe-frame-old-a", "frame_0.jpg", folder_id, content=b"old-a")
    fake_drive._add_file("dupe-frame-old-b", "frame_0.jpg", folder_id, content=b"old-b")
    replacement = tmp_path / "frame_0.jpg"
    replacement.write_bytes(b"replacement")

    uploaded = fake_drive.upload_or_update_file(replacement, folder_id, file_name="frame_0.jpg")
    remaining = fake_drive.find_files_by_name(folder_id, "frame_0.jpg")

    assert uploaded["id"] == "dupe-frame-old-a"
    assert [item["id"] for item in remaining] == ["dupe-frame-old-a"]
    assert fake_drive.items["dupe-frame-old-a"]["content"] == b"replacement"
    assert fake_drive.items["dupe-frame-old-b"]["trashed"] is True


def test_queue_filters_history_hidden_folders_with_one_history_read(client, fake_drive, monkeypatch):
    context = label_app._resolve_queue_context(fake_drive, label_app.VIDEO_SOURCE, None)
    fake_drive._add_folder("video-triplet-2", "ipc3_table-4_t0002", "video-unlabeled")
    fake_drive._add_triplet_files("video-triplet-2", "video2")
    label_app._invalidate_listing_cache(context.queue_key)

    for folder in fake_drive.list_folders("video-unlabeled"):
        label_app._record_label_history(
            context,
            str(folder["id"]),
            str(folder["name"]),
            "",
            "discarded",
        )

    load_count = 0
    original_load = label_app._load_label_history_unlocked

    def counted_load_label_history():
        nonlocal load_count
        load_count += 1
        return original_load()

    monkeypatch.setattr(label_app, "_load_label_history_unlocked", counted_load_label_history)

    response = client.get("/api/queue?source=video&limit=10")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["folders"] == []
    assert payload["total_unlabeled"] == 0
    assert load_count == 1


def test_queue_dedupes_duplicate_frame_signatures(client, fake_drive):
    fake_drive._add_folder("video-triplet-copy", "ipc3_table-4_t0001_copy", "video-unlabeled")
    fake_drive.update_file_metadata(
        "video-triplet-copy",
        {
            "appProperties": label_app.build_folder_app_properties(
                {
                    "frame_0": "video-frame0",
                    "frame_1": "video-frame1",
                    "frame_2": "video-frame2",
                }
            )
        },
    )
    label_app._listing_cache.clear()
    label_app._hydrated_folder_cache.clear()

    response = client.get("/api/queue?source=video&limit=10&refresh=1")
    payload = response.get_json()

    assert response.status_code == 200
    assert [folder["frame_signature"] for folder in payload["folders"]] == [
        "video-frame0|video-frame1|video-frame2"
    ]


def test_thumb_cache_ready_uses_persistent_thumb_files(tmp_path, monkeypatch):
    monkeypatch.setenv("LABEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(label_app, "CACHE_DIR", tmp_path)
    frames = {
        "frame_0": "ready-0",
        "frame_1": "ready-1",
        "frame_2": "ready-2",
    }
    for file_id in frames.values():
        (tmp_path / f"{file_id}.thumb.jpg").write_bytes(b"thumb")

    assert label_app._thumbs_cache_ready(frames) is True


def test_thumbnail_prewarm_writes_thumb_cache_without_browser_request(fake_drive, tmp_path, monkeypatch):
    image_bytes = io.BytesIO()
    Image.new("RGB", (800, 450), color=(20, 40, 60)).save(image_bytes, format="JPEG")
    fake_drive.items["video-frame0"]["content"] = image_bytes.getvalue()

    monkeypatch.setenv("LABEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(label_app, "CACHE_DIR", tmp_path)

    thumb_path, cache_hit, download_ms, encode_ms = label_app._ensure_thumb_for_file(
        "video-frame0",
        fake_drive,
    )

    assert thumb_path == tmp_path / "video-frame0.thumb.jpg"
    assert thumb_path.exists()
    assert (tmp_path / "video-frame0.jpg").exists()
    assert cache_hit is False
    assert download_ms >= 0
    assert encode_ms >= 0


def test_cache_warmer_skips_existing_volume_files(fake_drive, tmp_path, monkeypatch):
    image_bytes = io.BytesIO()
    Image.new("RGB", (800, 450), color=(20, 40, 60)).save(image_bytes, format="JPEG")
    for file_id in ("video-frame0", "video-frame1", "video-frame2"):
        fake_drive.items[file_id]["content"] = image_bytes.getvalue()

    monkeypatch.setenv("LABEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(label_app, "CACHE_DIR", tmp_path)
    (tmp_path / "video-frame0.jpg").write_bytes(image_bytes.getvalue())
    (tmp_path / "video-frame0.thumb.jpg").write_bytes(image_bytes.getvalue())

    label_app._run_cache_warm_background(label_app.VIDEO_SOURCE, None, limit=1)
    state = label_app._cache_warm_state_snapshot()

    assert state["folders_scanned"] == 1
    assert state["folders_hydrated"] == 1
    assert state["frames_seen"] == 3
    assert state["full_res_cached"] == 2
    assert state["thumbs_cached"] == 2
    assert state["skipped_full_res"] == 1
    assert state["skipped_thumbs"] == 1
    assert state["folders_hot_cached"] == 1
    assert state["errors"] == []
    assert not label_app._cache_warm_shared_lock_path().exists()
    cached = label_app._get_cached_hydrated_folder(label_app.VIDEO_SOURCE, "video-triplet")
    assert cached is not label_app._MISSING
    assert cached["cache_ready"] is True


def test_label_ready_target_env_overrides_legacy_targets(monkeypatch):
    monkeypatch.setenv("LABEL_READY_TARGET", "359")
    monkeypatch.setenv("LABEL_REOLINK_PREWARM_TARGET", "5000")
    monkeypatch.setenv("AUTOLABEL_VIDEO_LOW_WATERMARK", "1000")
    monkeypatch.setenv("LABEL_PREWARM_FOLDER_COUNT", "60")

    assert label_app._ready_target_or_legacy_env("LABEL_REOLINK_PREWARM_TARGET", 1000) == 359
    assert label_app._ready_target_or_legacy_env("AUTOLABEL_VIDEO_LOW_WATERMARK", 1000) == 359
    assert label_app._ready_target_or_legacy_env("LABEL_PREWARM_FOLDER_COUNT", 400) == 359


def test_cache_warm_shared_lock_blocks_second_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("PREPROCESS_STATE_DIR", str(tmp_path))
    label_app._set_cache_warm_state(inflight=False, last_error=None)

    token = label_app._acquire_cache_warm_shared_lock()
    assert token is not None
    try:
        started, state = label_app._start_cache_warm(label_app.VIDEO_SOURCE, None, 1)
    finally:
        label_app._release_cache_warm_shared_lock(token)

    assert started is False
    assert state["inflight"] is False
    assert state["last_error"] == "Another cache warm worker is already running."
    assert state["shared_lock"]["started_at_epoch"] == token["started_at_epoch"]
    label_app._set_cache_warm_state(inflight=False, last_error=None, shared_lock=None, shared_lock_path=None)


def test_ready_maintainer_shared_lock_blocks_duplicate_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("PREPROCESS_STATE_DIR", str(tmp_path))

    token = label_app._acquire_ready_maintainer_shared_lock()
    assert token is not None

    def fail_drive_client():
        raise AssertionError("second worker should skip while ready maintainer lock is held")

    monkeypatch.setattr(label_app, "DriveClient", fail_drive_client)
    try:
        label_app._run_ready_maintainer_once()
    finally:
        label_app._release_ready_maintainer_shared_lock(token)


def test_ready_maintainer_auto_start_is_disabled_under_pytest(monkeypatch):
    calls: list[bool] = []
    monkeypatch.delenv("LABEL_READY_MAINTAINER_ON_STARTUP", raising=False)
    monkeypatch.setattr(label_app, "_ensure_ready_maintainer_started", lambda: calls.append(True) or True)

    assert label_app._auto_start_ready_maintainer() is False
    assert calls == []


def test_preprocess_status_starts_maintainer_only_when_requested(client, monkeypatch):
    calls: list[bool] = []
    monkeypatch.setattr(label_app, "_ensure_ready_maintainer_started", lambda: calls.append(True) or True)

    response = client.get("/api/preprocess/status")
    assert response.status_code == 200
    assert calls == []

    started_response = client.get("/api/preprocess/status?start=1")
    assert started_response.status_code == 200
    assert calls == [True]


def test_default_throughput_target_covers_4000_images():
    assert label_app.LABEL_THROUGHPUT_TARGET_IMAGES >= 4000
    assert label_app.LABEL_THROUGHPUT_TARGET_FOLDERS >= 1334
    assert label_app.REOLINK_PREWARM_TARGET >= label_app.LABEL_THROUGHPUT_TARGET_FOLDERS
    assert label_app.PREWARM_FOLDER_COUNT <= label_app.LABEL_THROUGHPUT_TARGET_FOLDERS
    assert label_app.READY_SCAN_MAX <= label_app.INTERACTIVE_READY_SCAN_CAP
    assert label_app.QUEUE_BATCH_MAX >= 600


def test_force_refresh_uses_stale_listing_cache_and_refreshes_in_background(monkeypatch):
    context = label_app.QueueContext(
        source=label_app.VIDEO_SOURCE,
        site_key=None,
        queue_key=label_app.VIDEO_SOURCE,
        display_name="Video",
        input_folder_name="unlabeled",
        input_folder_id="video-unlabeled",
        seed_folder_name=None,
        seed_folder_id=None,
        folder_ids={},
        persist_frame_metadata=False,
    )
    stale_listing = [{"id": "cached-folder", "name": "cached"}]
    label_app._set_listing_cache(context.queue_key, stale_listing)
    monkeypatch.setattr(label_app, "_schedule_listing_refresh", lambda ctx: True)

    def fail_fetch(_client, _context):
        raise AssertionError("force refresh should not block on Drive when cache exists")

    monkeypatch.setattr(label_app, "_fetch_source_listing", fail_fetch)

    try:
        assert label_app._list_source_subfolders(object(), context, force_refresh=True) == stale_listing
    finally:
        label_app._invalidate_listing_cache(context.queue_key)


def test_force_refresh_without_listing_cache_warms_in_background(monkeypatch):
    context = label_app.QueueContext(
        source=label_app.REOLINK_SOURCE,
        site_key="restaurant-pi-1",
        queue_key="reolink:restaurant-pi-1",
        display_name="Video",
        input_folder_name="unlabeled",
        input_folder_id="video-unlabeled",
        seed_folder_name=None,
        seed_folder_id=None,
        folder_ids={},
        persist_frame_metadata=False,
    )
    scheduled: list[str] = []
    label_app._invalidate_listing_cache(context.queue_key)
    monkeypatch.setattr(label_app, "_schedule_listing_refresh", lambda ctx: scheduled.append(ctx.queue_key) or True)

    def fail_fetch(_client, _context):
        raise AssertionError("cold force refresh should warm in background")

    monkeypatch.setattr(label_app, "_fetch_source_listing", fail_fetch)

    with label_app.app.test_request_context("/api/queue?source=reolink&site=restaurant-pi-1&refresh=1"):
        assert label_app._list_source_subfolders(object(), context, force_refresh=True) == []
        assert scheduled == [context.queue_key]


def test_interactive_preview_prewarm_is_bounded(tmp_path, monkeypatch):
    submitted: list[str] = []

    class FakeExecutor:
        def submit(self, _fn, file_id):
            submitted.append(file_id)

    monkeypatch.setenv("LABEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(label_app, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(label_app, "_preview_prewarm_executor", FakeExecutor())
    label_app._preview_prewarm_inflight.clear()
    folders = [
        {
            "frames": {
                "frame_0": f"folder-{idx}-0",
                "frame_1": f"folder-{idx}-1",
                "frame_2": f"folder-{idx}-2",
            }
        }
        for idx in range(label_app.PREWARM_FOLDER_COUNT + 25)
    ]

    scheduled = label_app._schedule_preview_prewarm(folders)

    assert scheduled == label_app.PREWARM_FOLDER_COUNT * 3
    assert len(submitted) == scheduled


def test_cache_status_reports_cache_dir_and_writable(client, tmp_path, monkeypatch):
    monkeypatch.setenv("LABEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(label_app, "CACHE_DIR", tmp_path)
    (tmp_path / "full.jpg").write_bytes(b"full")
    (tmp_path / "full.thumb.jpg").write_bytes(b"thumb")

    light_response = client.get("/api/cache/status")
    light_payload = light_response.get_json()

    assert light_response.status_code == 200
    assert light_payload["scan_included"] is False
    assert light_payload["full_res_count"] is None
    assert light_payload["thumb_count"] is None

    response = client.get("/api/cache/status?scan=1")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["cache_dir"] == str(tmp_path)
    assert payload["writable"] is True
    assert payload["scan_included"] is True
    assert payload["full_res_count"] == 1
    assert payload["thumb_count"] == 1
    assert payload["configured_cache_dir"] == str(tmp_path)
    assert payload["expected_volume_cache_dir"] == "/data/label_cache"
    assert payload["cache_max_mb"] == label_app.CACHE_MAX_MB
    assert payload["cache_ttl_hours"] == label_app.CACHE_TTL_HOURS
    assert payload["size_mb"] >= 0


def test_cache_warm_status_and_cancel_are_lightweight(client):
    status_response = client.get("/api/cache/warm/status")
    cancel_response = client.post("/api/cache/warm/cancel")

    assert status_response.status_code == 200
    assert "inflight" in status_response.get_json()
    assert cancel_response.status_code == 200
    assert cancel_response.get_json()["stop_requested"] is True


def test_reolink_label_moves_folder_within_same_site_tree(client, fake_drive):
    queue_response = client.get("/api/queue?source=reolink&site=reolink-matthews-01&limit=10")
    folder = queue_response.get_json()["folders"][0]

    label_response = client.post(
        "/api/label",
        json=_label_payload(folder, "clean"),
    )

    assert label_response.status_code == 200
    assert label_response.get_json()["queued"] is True
    _drain_label_jobs(fake_drive)
    shared_clean_id = fake_drive.find_file_by_name(
        "project-root",
        "clean",
        mime_type=label_app.FOLDER_MIME,
    )["id"]
    assert fake_drive.items[folder["folder_id"]]["parents"] == [shared_clean_id]
    # The per-site clean/ tree should no longer be the target.
    assert fake_drive.find_file_by_name(
        "site-matthews",
        "clean",
        mime_type=label_app.FOLDER_MIME,
    ) is None

    stats_response = client.get("/api/stats?source=reolink&site=reolink-matthews-01")
    stats_payload = stats_response.get_json()
    assert stats_payload["clean"] == 1
    assert stats_payload["unlabeled"] == 1


def test_reolink_label_accepts_screenrecord_three_frame_parent(client, fake_drive):
    fake_drive._add_folder("r-3frame", "3frame", "site-restaurant")
    fake_drive._add_folder("r-3frame-unlabeled", "unlabeled", "r-3frame")
    fake_drive._add_folder("sr-label-artifact", "mimosas-Swann-CH05_crop_t0001", "r-3frame-unlabeled")
    fake_drive._add_triplet_files("sr-label-artifact", "sr-label", include_metadata=True)
    label_app._source_folder_ids_cache.clear()
    label_app._listing_cache.clear()
    label_app._hydrated_folder_cache.clear()

    context = label_app._resolve_queue_context(fake_drive, label_app.REOLINK_SOURCE, "restaurant-pi-1")
    folder_item = fake_drive.get_file("sr-label-artifact", fields="id,name,mimeType,parents,appProperties")
    folder = label_app._hydrate_folder(fake_drive, context, folder_item)

    response = client.post("/api/label", json=_label_payload(folder, "clean"))

    assert response.status_code == 200
    assert response.get_json()["queued"] is True
    _drain_label_jobs(fake_drive)
    shared_clean_id = fake_drive.find_file_by_name(
        "project-root",
        "clean",
        mime_type=label_app.FOLDER_MIME,
    )["id"]
    assert fake_drive.items["sr-label-artifact"]["parents"] == [shared_clean_id]


def test_reolink_queue_allows_triplets_without_metadata_json(client):
    response = client.get("/api/queue?source=reolink&site=restaurant-pi-1&limit=10")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["folders"][0]["folder_name"] == "mimosas-Reolink-CH-CH04_table_top_1_t0004"


def test_apply_source_prefix_is_idempotent_and_per_restaurant():
    mimosas_video = label_app._resolve_label_source("video", None)
    mimosas_photos = label_app._resolve_label_source("reolink", "restaurant-pi-1")
    matthews = label_app._resolve_label_source("reolink", "reolink-matthews-01")

    assert label_app._apply_source_prefix("ipc3_table-4_t0001", mimosas_video) == "mimosas-ipc3_table-4_t0001"
    assert label_app._apply_source_prefix("Reolink-CH-CH04_table_top_1_t0004", mimosas_photos) == "mimosas-Reolink-CH-CH04_table_top_1_t0004"
    assert label_app._apply_source_prefix("Reolink-CH-CH03_table_top_1_t0002", matthews) == "matthews-Reolink-CH-CH03_table_top_1_t0002"

    already = label_app._apply_source_prefix("mimosas-ipc3_table-4_t0001", mimosas_video)
    assert already == "mimosas-ipc3_table-4_t0001"
    still = label_app._apply_source_prefix(already, mimosas_video)
    assert still == "mimosas-ipc3_table-4_t0001"

    cross = label_app._apply_source_prefix("matthews-Reolink-CH-CH03_t0002", mimosas_video)
    assert cross == "matthews-Reolink-CH-CH03_t0002"


def test_label_discarded_route(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]

    discarded_response = client.post(
        "/api/label",
        json=_label_payload(folder, "discarded"),
    )

    assert discarded_response.status_code == 200
    _drain_label_jobs(fake_drive)
    discarded_dest = fake_drive.find_file_by_name(
        "project-root", "discarded", mime_type=label_app.FOLDER_MIME
    )
    assert discarded_dest is not None
    assert fake_drive.items[folder["folder_id"]]["parents"] == [discarded_dest["id"]]


def test_mimosas_photos_label_routes_to_shared_clean(client, fake_drive):
    queue_response = client.get("/api/queue?source=reolink&site=restaurant-pi-1&limit=10")
    folder = queue_response.get_json()["folders"][0]

    label_response = client.post(
        "/api/label",
        json=_label_payload(folder, "clean"),
    )

    assert label_response.status_code == 200
    _drain_label_jobs(fake_drive)
    shared_clean = fake_drive.find_file_by_name(
        "project-root", "clean", mime_type=label_app.FOLDER_MIME
    )
    assert shared_clean is not None
    assert fake_drive.items[folder["folder_id"]]["parents"] == [shared_clean["id"]]
    # restaurant-pi-1 should not have a per-site clean/ tree anymore.
    assert fake_drive.find_file_by_name(
        "site-restaurant", "clean", mime_type=label_app.FOLDER_MIME
    ) is None


def test_label_route_rejects_unknown_label(client):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]

    bogus_response = client.post(
        "/api/label",
        json={
            "folder_id": folder["folder_id"],
            "parent_id": folder["parent_id"],
            "label": "bogus",
            "source": folder["source"],
            "site_key": folder["site_key"],
        },
    )

    assert bogus_response.status_code == 400
    assert "discarded" in bogus_response.get_json()["error"]


def test_label_stamps_metadata_and_rejects_stale_second_move(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]
    payload = _label_payload(folder, "dirty")

    first_response = client.post("/api/label", json=payload)
    replace_response = client.post("/api/label", json=payload)

    assert first_response.status_code == 200
    assert replace_response.status_code == 200
    _drain_label_jobs(fake_drive)
    second_response = client.post("/api/label", json=payload)
    folder_item = fake_drive.items[folder["folder_id"]]
    assert folder_item["appProperties"]["autolabel_final_label"] == "dirty"
    assert folder_item["appProperties"]["autolabel_source"] == "video"
    assert folder_item["appProperties"]["autolabel_queue_key"] == "video"
    assert folder_item["appProperties"]["autolabel_labeled_by"] == "local"
    assert second_response.status_code == 409
    assert second_response.get_json()["code"] == "already_labeled"


def test_label_route_records_durable_job_before_drive_move(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]

    response = client.post("/api/label", json=_label_payload(folder, "clean"))

    assert response.status_code == 200
    assert response.get_json()["queued"] is True
    assert response.get_json()["drive_move_status"] == "queued"
    assert fake_drive.items[folder["folder_id"]]["parents"] == ["video-unlabeled"]
    history = json.loads(label_app._label_history_path().read_text(encoding="utf-8"))
    assert "frames:video-frame0|video-frame1|video-frame2" in history["queues"]["video"]["labeled"]
    jobs = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    job = jobs["jobs"]["video:video-triplet"]
    assert job["status"] == "pending"
    assert job["label"] == "clean"


def test_label_route_preserves_sparse_sample_frame_keys(client, fake_drive):
    folder_id = fake_drive.ensure_subfolder("video-unlabeled", "ipc3_table-4_t0009")
    frames = {
        "frame_0": "sample-frame0",
        "frame_5": "sample-frame5",
        "frame_9": "sample-frame9",
    }
    for key, file_id in frames.items():
        fake_drive._add_file(file_id, f"{key}.jpg", folder_id)
    fake_drive.update_file_metadata(
        folder_id,
        {"appProperties": label_app.build_folder_app_properties(frames)},
    )

    payload = {
        "folder_id": folder_id,
        "parent_id": "video-unlabeled",
        "label": "clean",
        "source": "video",
        "site_key": None,
        "folder_name": "ipc3_table-4_t0009",
        "frames": frames,
    }
    response = client.post("/api/label", json=payload)

    assert response.status_code == 200
    jobs = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    job = jobs["jobs"][f"video:{folder_id}"]
    assert job["frames"] == frames
    assert job["frame_signature"] == "sample-frame0|sample-frame5|sample-frame9"


def test_label_route_returns_json_when_state_write_fails(client, monkeypatch):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]

    def fail_record_history(*_args, **_kwargs):
        raise OSError("state volume unavailable")

    monkeypatch.setattr(label_app, "_record_label_history", fail_record_history)

    response = client.post("/api/label", json=_label_payload(folder, "clean"))
    payload = response.get_json()

    assert response.status_code == 500
    assert response.content_type.startswith("application/json")
    assert payload["code"] == "label_queue_failed"
    assert "state volume unavailable" in payload["error"]


def test_state_tmp_paths_are_unique(tmp_path):
    target = tmp_path / "label_jobs.json"

    assert label_app._state_tmp_path(target) != label_app._state_tmp_path(target)


def test_label_job_status_is_lightweight(client, fake_drive):
    response = client.get("/api/label/jobs/status")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["writable"] is True
    assert payload["counts"]["pending"] == 0
    assert str(label_app._label_jobs_path()) == payload["path"]
    assert "delayed" in payload["counts"]
    assert payload["undo_seconds"] == label_app.LABEL_JOB_UNDO_SECONDS
    assert "stale_reset_count" in payload
    assert "recoverable_failed_reset_count" in payload
    assert payload["confirmed_moved"] == 0
    assert payload["waiting_to_move"] == 0
    assert "rate_limit_cooldown_seconds" in payload


def test_discard_job_waits_until_undo_deadline(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]

    response = client.post("/api/label", json=_label_payload(folder, "discarded"))
    assert response.status_code == 200

    assert label_app._drain_label_jobs_once(fake_drive) == 0
    assert fake_drive.items[folder["folder_id"]]["parents"] == ["video-unlabeled"]
    jobs = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    job = jobs["jobs"]["video:video-triplet"]
    assert job["status"] == "pending"
    assert job["not_before"] == response.get_json()["not_before"]

    assert _drain_label_jobs(fake_drive) == 1
    discarded_dest = fake_drive.find_file_by_name(
        "project-root", "discarded", mime_type=label_app.FOLDER_MIME
    )
    assert fake_drive.items[folder["folder_id"]]["parents"] == [discarded_dest["id"]]


def test_cancel_pending_label_job_removes_history_and_restores_queue(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]
    payload = _label_payload(folder, "discarded")

    assert client.post("/api/label", json=payload).status_code == 200
    cancel_response = client.post("/api/label/cancel", json=payload)

    assert cancel_response.status_code == 200
    jobs = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    assert jobs["jobs"]["video:video-triplet"]["status"] == "canceled"
    history = json.loads(label_app._label_history_path().read_text(encoding="utf-8"))
    assert history["queues"]["video"]["labeled"] == {}
    repeat_response = client.get("/api/queue?source=video&limit=10&refresh=1")
    assert repeat_response.get_json()["folders"][0]["folder_id"] == folder["folder_id"]


def test_cancel_after_drive_move_restores_folder_to_unlabeled(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]
    payload = _label_payload(folder, "clean")

    assert client.post("/api/label", json=payload).status_code == 200
    assert _drain_label_jobs(fake_drive) == 1
    assert fake_drive.items[folder["folder_id"]]["parents"] == ["video-clean"]

    cancel_response = client.post("/api/label/cancel", json=payload)
    assert cancel_response.status_code == 200
    assert cancel_response.get_json()["restored"] is True
    assert fake_drive.items[folder["folder_id"]]["parents"] == ["video-unlabeled"]
    for key in label_app.LABEL_APP_PROPERTY_KEYS:
        assert key not in fake_drive.items[folder["folder_id"]]["appProperties"]
    jobs = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    assert jobs["jobs"]["video:video-triplet"]["status"] == "canceled"
    assert "restored folder" in jobs["jobs"]["video:video-triplet"]["last_error"]
    history = json.loads(label_app._label_history_path().read_text(encoding="utf-8"))
    assert history["queues"]["video"]["labeled"] == {}
    repeat_response = client.get("/api/queue?source=video&limit=10&refresh=1")
    assert repeat_response.get_json()["folders"][0]["folder_id"] == folder["folder_id"]


def test_cancel_succeeds_when_moved_folder_is_already_back_in_input(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]
    payload = _label_payload(folder, "clean")

    assert client.post("/api/label", json=payload).status_code == 200
    assert _drain_label_jobs(fake_drive) == 1
    fake_drive.move_file(folder["folder_id"], "video-unlabeled", remove_parent_id="video-clean")

    cancel_response = client.post("/api/label/cancel", json=payload)
    assert cancel_response.status_code == 200
    assert cancel_response.get_json()["restored"] is True
    assert fake_drive.items[folder["folder_id"]]["parents"] == ["video-unlabeled"]
    for key in label_app.LABEL_APP_PROPERTY_KEYS:
        assert key not in fake_drive.items[folder["folder_id"]]["appProperties"]
    jobs = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    assert jobs["jobs"]["video:video-triplet"]["status"] == "canceled"


def test_cancel_from_different_label_destination_restores_to_unlabeled(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]
    payload = _label_payload(folder, "clean")

    assert client.post("/api/label", json=payload).status_code == 200
    assert _drain_label_jobs(fake_drive) == 1
    fake_drive.move_file(folder["folder_id"], "video-dirty", remove_parent_id="video-clean")

    cancel_response = client.post("/api/label/cancel", json=payload)
    assert cancel_response.status_code == 200
    assert cancel_response.get_json()["restored"] is True
    assert fake_drive.items[folder["folder_id"]]["parents"] == ["video-unlabeled"]
    jobs = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    assert jobs["jobs"]["video:video-triplet"]["status"] == "canceled"


def test_relabel_pending_job_replaces_label_without_duplicate_move(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]
    dirty_payload = _label_payload(folder, "dirty")
    clean_payload = _label_payload(folder, "clean")

    assert client.post("/api/label", json=dirty_payload).status_code == 200
    relabel_response = client.post("/api/label", json=clean_payload)

    assert relabel_response.status_code == 200
    jobs = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    assert jobs["jobs"]["video:video-triplet"]["label"] == "clean"
    assert _drain_label_jobs(fake_drive) == 1
    assert len(fake_drive.moves) == 1
    folder_item = fake_drive.items[folder["folder_id"]]
    assert folder_item["appProperties"]["autolabel_final_label"] == "clean"


def test_stale_processing_job_recovers_and_pushes_once(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]
    assert client.post("/api/label", json=_label_payload(folder, "clean")).status_code == 200

    jobs = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    job = jobs["jobs"]["video:video-triplet"]
    job["status"] = "processing"
    job["updated_at"] = (
        datetime.now(timezone.utc)
        - timedelta(seconds=label_app.LABEL_JOB_PROCESSING_STALE_SECONDS + 5)
    ).isoformat()
    job["not_before"] = (
        datetime.now(timezone.utc)
        - timedelta(seconds=1)
    ).isoformat()
    label_app._label_jobs_path().write_text(json.dumps(jobs), encoding="utf-8")

    assert label_app._drain_label_jobs_once(fake_drive) == 1
    jobs_after = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    assert jobs_after["jobs"]["video:video-triplet"]["status"] == "succeeded"
    assert len(fake_drive.moves) == 1


def test_already_moved_job_succeeds_without_duplicate_move(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]
    assert client.post("/api/label", json=_label_payload(folder, "clean")).status_code == 200

    clean_dest = fake_drive.find_file_by_name(
        "project-root", "clean", mime_type=label_app.FOLDER_MIME
    )
    fake_drive.update_file_metadata(
        folder["folder_id"],
        {"appProperties": label_app._label_app_properties("clean", label_app._resolve_queue_context(fake_drive, "video", None))},
    )
    fake_drive.move_file(folder["folder_id"], clean_dest["id"], remove_parent_id=folder["parent_id"])
    fake_drive.moves.clear()

    assert _drain_label_jobs(fake_drive) == 1
    assert fake_drive.moves == []


def test_metadata_only_label_still_requires_drive_move(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]
    assert client.post("/api/label", json=_label_payload(folder, "clean")).status_code == 200

    fake_drive.update_file_metadata(
        folder["folder_id"],
        {"appProperties": label_app._label_app_properties("clean", label_app._resolve_queue_context(fake_drive, "video", None))},
    )

    assert _drain_label_jobs(fake_drive) == 1
    assert fake_drive.items[folder["folder_id"]]["parents"] == ["video-clean"]
    assert fake_drive.moves == [(folder["folder_id"], "video-clean", folder["parent_id"])]


def test_label_retry_moves_after_prior_move_failure_with_metadata_present(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]
    assert client.post("/api/label", json=_label_payload(folder, "clean")).status_code == 200
    fake_drive.update_file_metadata(
        folder["folder_id"],
        {"appProperties": label_app._label_app_properties("clean", label_app._resolve_queue_context(fake_drive, "video", None))},
    )

    original_move_file = fake_drive.move_file
    failures_remaining = {"count": 1}

    def fail_once_move_file(*args, **kwargs):
        if failures_remaining["count"]:
            failures_remaining["count"] -= 1
            raise RuntimeError("temporary Drive move failure")
        return original_move_file(*args, **kwargs)

    fake_drive.move_file = fail_once_move_file

    assert _drain_label_jobs(fake_drive) == 0
    jobs = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    assert jobs["jobs"]["video:video-triplet"]["status"] == "pending"
    assert fake_drive.items[folder["folder_id"]]["parents"] == ["video-unlabeled"]

    assert _drain_label_jobs(fake_drive) == 1
    assert fake_drive.items[folder["folder_id"]]["parents"] == ["video-clean"]


def test_wrong_label_destination_is_corrected(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]
    assert client.post("/api/label", json=_label_payload(folder, "clean")).status_code == 200
    fake_drive.move_file(folder["folder_id"], "video-dirty", remove_parent_id=folder["parent_id"])
    fake_drive.moves.clear()

    assert _drain_label_jobs(fake_drive) == 1
    assert fake_drive.items[folder["folder_id"]]["parents"] == ["video-clean"]
    assert fake_drive.moves == [(folder["folder_id"], "video-clean", "video-dirty")]
    assert fake_drive.items[folder["folder_id"]]["appProperties"]["autolabel_final_label"] == "clean"


def test_verify_reopens_succeeded_job_not_in_drive_destination(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]
    assert client.post("/api/label", json=_label_payload(folder, "clean")).status_code == 200

    jobs = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    job = jobs["jobs"]["video:video-triplet"]
    job["status"] = "succeeded"
    job["updated_at"] = datetime.now(timezone.utc).isoformat()
    label_app._label_jobs_path().write_text(json.dumps(jobs), encoding="utf-8")
    fake_drive.update_file_metadata(
        folder["folder_id"],
        {"appProperties": label_app._label_app_properties("clean", label_app._resolve_queue_context(fake_drive, "video", None))},
    )

    response = client.get("/api/label/jobs/status?verify=1")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["verification"]["verified_mismatch_count"] == 1
    assert payload["verification"]["reopened_count"] == 1
    assert payload["counts"]["pending"] == 1
    assert payload["counts"]["succeeded"] == 0


def test_label_worker_reruns_when_schedule_requested_while_inflight(monkeypatch):
    submitted = []

    class FakeExecutor:
        def submit(self, fn):
            submitted.append(fn)
            return None

    monkeypatch.setattr(label_app, "_label_job_executor", FakeExecutor())
    monkeypatch.setattr(label_app, "_drain_label_jobs_once", lambda: 0)
    monkeypatch.setattr(label_app, "_next_label_job_delay_seconds", lambda: None)

    label_app._label_job_worker_inflight = True
    label_app._label_job_worker_rerun_requested = False
    try:
        assert label_app._schedule_label_job_worker() is False
        assert label_app._label_job_worker_rerun_requested is True

        label_app._run_label_job_worker()

        assert submitted == [label_app._run_label_job_worker]
        assert label_app._label_job_worker_inflight is True
        assert label_app._label_job_worker_rerun_requested is False
    finally:
        label_app._label_job_worker_inflight = False
        label_app._label_job_worker_rerun_requested = False


def test_rate_limit_cooldown_grows_until_success(monkeypatch):
    monkeypatch.setattr(label_app, "LABEL_JOB_RATE_LIMIT_COOLDOWN_SECONDS", 1.0)
    monkeypatch.setattr(label_app, "LABEL_JOB_RATE_LIMIT_MAX_COOLDOWN_SECONDS", 8.0)
    label_app._label_job_rate_limit_cooldown_until = None
    label_app._label_job_rate_limit_cooldown_seconds = 1.0
    label_app._label_job_last_rate_limit_error = None

    label_app._record_label_job_rate_limit(RuntimeError("429 too many requests"))
    assert label_app._label_job_rate_limit_cooldown_seconds == 2.0

    label_app._label_job_rate_limit_cooldown_until = datetime.now(timezone.utc) - timedelta(seconds=1)
    label_app._clear_label_job_rate_limit_cooldown()
    assert label_app._label_job_rate_limit_cooldown_seconds == 2.0

    label_app._record_label_job_rate_limit(RuntimeError("quota exceeded"))
    assert label_app._label_job_rate_limit_cooldown_seconds == 4.0

    label_app._record_label_job_success()
    assert label_app._label_job_rate_limit_cooldown_until is None
    assert label_app._label_job_rate_limit_cooldown_seconds == 1.0
    assert label_app._label_job_last_rate_limit_error is None


def test_due_label_jobs_drain_one_at_a_time(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]
    assert client.post("/api/label", json=_label_payload(folder, "clean")).status_code == 200
    fake_drive._add_folder("video-triplet-2", "ipc3_table-5_t0002", "video-unlabeled")
    fake_drive._add_triplet_files("video-triplet-2", "video2")

    jobs = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    second_job = dict(jobs["jobs"]["video:video-triplet"])
    second_job["id"] = "video:video-triplet-2"
    second_job["folder_id"] = "video-triplet-2"
    second_job["folder_name"] = "ipc3_table-5_t0002"
    jobs["jobs"]["video:video-triplet-2"] = second_job
    label_app._label_jobs_path().write_text(json.dumps(jobs), encoding="utf-8")

    assert _drain_label_jobs(fake_drive) == 1
    jobs_after_first = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    assert jobs_after_first["jobs"]["video:video-triplet"]["status"] == "succeeded"
    assert jobs_after_first["jobs"]["video:video-triplet-2"]["status"] == "pending"

    assert _drain_label_jobs(fake_drive) == 1
    jobs_after_second = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    assert jobs_after_second["jobs"]["video:video-triplet-2"]["status"] == "succeeded"


def test_background_label_push_does_not_need_request_context(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]

    assert client.post("/api/label", json=_label_payload(folder, "clean")).status_code == 200

    with label_app.app.app_context():
        assert _drain_label_jobs(fake_drive) == 1
    folder_item = fake_drive.items[folder["folder_id"]]
    assert folder_item["appProperties"]["autolabel_final_label"] == "clean"
    assert folder_item["appProperties"]["autolabel_labeled_by"] == "local"


def test_recoverable_failed_jobs_reset_for_retry(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]
    assert client.post("/api/label", json=_label_payload(folder, "clean")).status_code == 200

    jobs = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    job = jobs["jobs"]["video:video-triplet"]
    job["status"] = "failed"
    job["attempts"] = label_app.LABEL_JOB_MAX_ATTEMPTS
    job["last_error"] = "Working outside of request context."
    job["not_before"] = (
        datetime.now(timezone.utc)
        - timedelta(seconds=1)
    ).isoformat()
    label_app._label_jobs_path().write_text(json.dumps(jobs), encoding="utf-8")

    assert label_app._drain_label_jobs_once(fake_drive) == 1
    jobs_after = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    assert jobs_after["jobs"]["video:video-triplet"]["status"] == "succeeded"


def test_recoverable_failed_jobs_are_retried_one_per_worker_run(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]
    assert client.post("/api/label", json=_label_payload(folder, "clean")).status_code == 200

    fake_drive._add_folder("video-triplet-2", "ipc3_table-5_t0002", "video-unlabeled")
    fake_drive._add_triplet_files("video-triplet-2", "video2")
    jobs = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    first_job = jobs["jobs"]["video:video-triplet"]
    second_job = dict(first_job)
    second_job["id"] = "video:video-triplet-2"
    second_job["folder_id"] = "video-triplet-2"
    second_job["folder_name"] = "ipc3_table-5_t0002"
    jobs["jobs"]["video:video-triplet-2"] = second_job
    for job in jobs["jobs"].values():
        job["status"] = "failed"
        job["attempts"] = label_app.LABEL_JOB_MAX_ATTEMPTS
        job["last_error"] = "Working outside of request context."
        job["not_before"] = (
            datetime.now(timezone.utc)
            - timedelta(seconds=1)
        ).isoformat()
    label_app._label_jobs_path().write_text(json.dumps(jobs), encoding="utf-8")

    assert label_app._drain_label_jobs_once(fake_drive) == 1
    jobs_after = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    assert jobs_after["jobs"]["video:video-triplet"]["status"] == "succeeded"
    assert jobs_after["jobs"]["video:video-triplet-2"]["status"] == "failed"


def test_missing_drive_folder_failure_stays_failed(client, fake_drive):
    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]
    assert client.post("/api/label", json=_label_payload(folder, "clean")).status_code == 200

    jobs = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    job = jobs["jobs"]["video:video-triplet"]
    job["status"] = "failed"
    job["attempts"] = label_app.LABEL_JOB_MAX_ATTEMPTS
    job["last_error"] = "folder is no longer in the source or target Drive folder"
    job["not_before"] = (
        datetime.now(timezone.utc)
        - timedelta(seconds=1)
    ).isoformat()
    label_app._label_jobs_path().write_text(json.dumps(jobs), encoding="utf-8")

    assert label_app._drain_label_jobs_once(fake_drive) == 0
    jobs_after = json.loads(label_app._label_jobs_path().read_text(encoding="utf-8"))
    job_after = jobs_after["jobs"]["video:video-triplet"]
    assert job_after["status"] == "failed"
    assert job_after["attempts"] == label_app.LABEL_JOB_MAX_ATTEMPTS


def test_labeled_content_signature_never_returns_to_queue(client, fake_drive, monkeypatch):
    monkeypatch.setattr(label_app, "_content_signature_from_frames", lambda frames: "same-thumb-content")
    queue_response = client.get("/api/queue?source=video&limit=10&refresh=1")
    folder = queue_response.get_json()["folders"][0]
    payload = _label_payload(folder, "clean")

    label_response = client.post("/api/label", json=payload)
    assert label_response.status_code == 200

    fake_drive._add_folder("video-triplet-visual-rerun", "different_name_same_pixels", "video-unlabeled")
    fake_drive.update_file_metadata(
        "video-triplet-visual-rerun",
        {
            "appProperties": label_app.build_folder_app_properties(
                {
                    "frame_0": "new-frame0",
                    "frame_1": "new-frame1",
                    "frame_2": "new-frame2",
                }
            )
        },
    )

    repeat_response = client.get("/api/queue?source=video&limit=10&refresh=1")
    assert repeat_response.get_json()["folders"] == []


def test_labeled_frame_signature_never_returns_to_queue_across_sessions(client, fake_drive, tmp_path, monkeypatch):
    monkeypatch.setenv("PREPROCESS_STATE_DIR", str(tmp_path))
    queue_response = client.get("/api/queue?source=video&limit=10&refresh=1")
    folder = queue_response.get_json()["folders"][0]

    label_response = client.post(
        "/api/label",
        json=_label_payload(folder, "clean"),
    )
    assert label_response.status_code == 200

    fake_drive._add_folder("video-triplet-rerun", "ipc3_table-4_t0001_rerun", "video-unlabeled")
    fake_drive.update_file_metadata(
        "video-triplet-rerun",
        {
            "appProperties": label_app.build_folder_app_properties(
                {
                    "frame_0": "video-frame0",
                    "frame_1": "video-frame1",
                    "frame_2": "video-frame2",
                }
            )
        },
    )
    label_app._listing_cache.clear()
    label_app._hydrated_folder_cache.clear()

    repeat_response = client.get("/api/queue?source=video&limit=10&refresh=1")
    repeat_payload = repeat_response.get_json()

    assert repeat_response.status_code == 200
    assert repeat_payload["folders"] == []
    history = json.loads((tmp_path / label_app.LABEL_HISTORY_FILE_NAME).read_text())
    assert "frames:video-frame0|video-frame1|video-frame2" in history["queues"]["video"]["labeled"]


def test_cache_dir_reuses_existing_repo_cache(monkeypatch, tmp_path):
    fake_repo = tmp_path / "repo"
    repo_cache = fake_repo / "label_cache"
    temp_root = tmp_path / "tmp"
    temp_cache = temp_root / "AutoLabeler" / "label_cache"
    repo_cache.mkdir(parents=True)
    (repo_cache / "already-warm.jpg").write_bytes(b"cached")

    monkeypatch.delenv("LABEL_CACHE_DIR", raising=False)
    monkeypatch.setattr(label_app, "__file__", str(fake_repo / "app.py"))
    monkeypatch.setattr(label_app.tempfile, "gettempdir", lambda: str(temp_root))
    monkeypatch.setattr(label_app, "CACHE_DIR", temp_cache)

    assert label_app._ensure_cache_dir() == repo_cache
    assert label_app.CACHE_DIR == repo_cache
    assert not temp_cache.exists()


def test_auth_requires_login_and_csrf_for_mutations(client, monkeypatch):
    monkeypatch.setattr(label_app, "AUTH_REQUIRED", True)
    monkeypatch.setattr(label_app, "LABELER_PASSWORD", "pw")

    assert client.get("/healthz").status_code == 200
    assert client.get("/api/sources").status_code == 401

    login_response = client.post("/login", data={"password": "pw", "labeler_name": "sam"})
    assert login_response.status_code == 302
    assert client.get("/api/sources").status_code == 200

    queue_response = client.get("/api/queue?source=video&limit=10")
    folder = queue_response.get_json()["folders"][0]
    payload = {
        "folder_id": folder["folder_id"],
        "parent_id": folder["parent_id"],
        "label": "clean",
        "source": folder["source"],
        "site_key": folder["site_key"],
    }

    assert client.post("/api/label", json=payload).status_code == 403
    with client.session_transaction() as session:
        csrf_token = session["_csrf_token"]
    ok_response = client.post(
        "/api/label",
        json=payload,
        headers={"X-CSRF-Token": csrf_token},
    )
    assert ok_response.status_code == 200


def test_matthews_crop_status_and_existing_config_are_exposed(client):
    status_response = client.get("/api/reolink/crop-configs/status?site=reolink-matthews-01")

    assert status_response.status_code == 200
    status_payload = status_response.get_json()
    assert status_payload["site_key"] == "reolink-matthews-01"
    assert status_payload["channels"] == [
        {
            "channel_code": "CH-CH03",
            "has_config": True,
            "crop_count": 2,
            "reference_available": True,
            "setup_url": "/crop-editor?site=reolink-matthews-01&channel=CH-CH03",
        }
    ]

    config_response = client.get("/api/reolink/crop-config?site=reolink-matthews-01&channel=CH-CH03")

    assert config_response.status_code == 200
    config_payload = config_response.get_json()
    assert config_payload["has_config"] is True
    assert config_payload["config"]["channel_code"] == "CH-CH03"
    assert [crop["name"] for crop in config_payload["config"]["crops"]] == [
        "table_top_1",
        "table_top_2",
    ]
    assert config_payload["reference"]["frame_file_id"] == "mready-frame0"


def test_matthews_crop_config_can_be_saved(client, fake_drive):
    response = client.post(
        "/api/reolink/crop-config",
        json={
            "site_key": "reolink-matthews-01",
            "channel_code": "CH-CH03",
            "reference": {
                "raw_folder_id": "m-ready",
                "raw_folder_name": "Reolink-CH-CH03_t0002",
                "frame_file_id": "mready-frame0",
                "width": 1920,
                "height": 1080,
            },
            "crops": [
                {
                    "name": "table_1",
                    "polygon": [[1, 2], [11, 2], [11, 12], [1, 12]],
                }
            ],
        },
    )

    assert response.status_code == 200
    saved = json.loads(fake_drive.download_file_content("m-ch03-config").decode("utf-8"))
    assert [crop["name"] for crop in saved["crops"]] == ["table_1"]
    assert saved["reference"]["width"] == 1920


def test_matthews_queue_blocks_when_channel_is_missing_crop_config(client, fake_drive):
    fake_drive._add_folder("m-ch05", "Reolink-CH-CH05_t0005", "m-unassociated")
    fake_drive._add_triplet_files("m-ch05", "mch05", include_metadata=True)

    response = client.get("/api/queue?source=reolink&site=reolink-matthews-01&limit=10")

    assert response.status_code == 409
    payload = response.get_json()
    assert payload["setup_required"] is True
    assert payload["missing_channels"] == ["CH-CH05"]
    assert payload["setup_url"] == "/crop-editor?site=reolink-matthews-01&channel=CH-CH05"


def test_channel_name_maps_generally_to_matching_ipc_number():
    assert label_app._extract_reolink_channel_number("CH-CH03_t4377") == 3
    assert label_app._extract_reolink_channel_number("Reolink-CH-CH11_t0002") == 11
    assert label_app._derived_reolink_folder_name("CH-CH03_t4377", "table_top_1") == "CH-CH03_table_top_1_t4377"


def test_ordered_quadrilateral_points_normalizes_crossed_click_order():
    points = [(10.0, 10.0), (100.0, 100.0), (100.0, 10.0), (10.0, 100.0)]

    ordered = label_app._ordered_quadrilateral_points(points)

    assert ordered == [
        (10.0, 10.0),
        (100.0, 10.0),
        (100.0, 100.0),
        (10.0, 100.0),
    ]


def test_video_processor_marks_completed_and_moves_raw_video(fake_drive, monkeypatch):
    fake_drive._add_file("video-1", "IPC3_sample.mp4", "video-raw", mime_type="video/mp4")
    calls = []

    def fake_process_video(video_meta, cameras, folders, client, tmp, yolo_model=None):
        calls.append(video_meta["id"])
        return processor.VideoProcessResult(status="complete", uploaded_triplets=2)

    monkeypatch.setattr(processor, "load_yolo_model", lambda: None)
    monkeypatch.setattr(processor, "_process_video", fake_process_video)

    first = processor.run_processor(
        "project-root",
        Path(__file__).resolve().parents[1] / "approved_table_rectangles.json",
        fake_drive,
    )
    second = processor.run_processor(
        "project-root",
        Path(__file__).resolve().parents[1] / "approved_table_rectangles.json",
        fake_drive,
    )

    assert calls == ["video-1"]
    assert first.completed == 1
    assert first.triplets_uploaded == 2
    assert second.scanned == 0
    assert fake_drive.items["video-1"]["appProperties"][processor.PREPROCESS_STATUS_PROPERTY] == "complete"
    assert fake_drive.items["video-1"]["appProperties"][processor.PREPROCESS_TRIPLETS_PROPERTY] == "2"
    assert fake_drive.items["video-1"]["parents"] == ["project-root:processed_raw"]


def test_processor_sampling_and_perception_artifact_policy():
    assert processor.sample_frame_indices(3) == (0, 1, 2)
    assert processor.sample_frame_indices(10) == (0, 5, 9)
    assert processor.perception_filename_for_n_frames(3) is None
    assert processor.perception_filename_for_n_frames(10) == "perception_v2.json"
    assert processor.perception_filename_for_n_frames(4) is None


def test_video_processor_does_not_move_failed_video(fake_drive, monkeypatch):
    fake_drive._add_file("video-err", "IPC3_error.mp4", "video-raw", mime_type="video/mp4")

    def fail_process(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(processor, "load_yolo_model", lambda: None)
    monkeypatch.setattr(processor, "_process_video", fail_process)

    summary = processor.run_processor(
        "project-root",
        Path(__file__).resolve().parents[1] / "approved_table_rectangles.json",
        fake_drive,
    )

    assert summary.errored == 1
    assert fake_drive.items["video-err"]["parents"] == ["video-raw"]


def test_video_processed_raw_cleanup_trashes_only_old_items(fake_drive):
    processed_id = fake_drive.ensure_subfolder("project-root", processor.PROCESSED_RAW_FOLDER_NAME)
    fake_drive._add_file("old-video", "old.mp4", processed_id, mime_type="video/mp4")
    fake_drive._add_file("new-video", "new.mp4", processed_id, mime_type="video/mp4")
    fake_drive.items["old-video"]["modifiedTime"] = (
        datetime.now(timezone.utc) - timedelta(days=15)
    ).isoformat().replace("+00:00", "Z")
    fake_drive.items["new-video"]["modifiedTime"] = (
        datetime.now(timezone.utc) - timedelta(days=2)
    ).isoformat().replace("+00:00", "Z")

    cleaned = processor._cleanup_processed_raw_folder(
        fake_drive,
        processed_id,
        retention_days=14,
    )

    assert cleaned == 1
    assert fake_drive.items["old-video"]["trashed"] is True
    assert fake_drive.items["new-video"]["trashed"] is False


def test_reolink_drain_treats_legacy_unprefixed_folders_as_existing(fake_drive):
    legacy_name = "Reolink-CH-CH04_table_top_1_t0004"
    legacy_id = fake_drive.ensure_subfolder("r-unlabeled", legacy_name)
    fake_drive._add_triplet_files(legacy_id, "legacy-rready")

    summary = label_app.drain_reolink_preprocessing(fake_drive, ["restaurant-pi-1"])

    assert summary["generated"] == 0
    assert fake_drive.find_file_by_name(
        "r-unlabeled",
        "mimosas-Reolink-CH-CH04_table_top_1_t0004",
        mime_type=label_app.FOLDER_MIME,
    ) is None


def test_reolink_preprocess_records_raw_folder_in_local_state_and_skips_rerun(monkeypatch, tmp_path):
    fake = FakeDriveClient()
    monkeypatch.setenv("DRIVE_PROJECT_ROOT_FOLDER_ID", "project-root")
    monkeypatch.setenv("PREPROCESS_STATE_DIR", str(tmp_path))
    label_app._source_folder_ids_cache.clear()
    label_app._crop_config_cache.clear()

    calls = []

    def fake_materialize(client, context, raw_folder, missing_table_polygons):
        calls.append(raw_folder["id"])
        label_source = label_app._resolve_label_source(context.source, context.site_key)
        return [
            label_app._apply_source_prefix(
                label_app._derived_reolink_folder_name(raw_folder["name"], table_id),
                label_source,
            )
            for table_id, *_rest in missing_table_polygons
        ]

    monkeypatch.setattr(label_app, "_materialize_reolink_table_crops", fake_materialize)

    context = label_app._resolve_queue_context(fake, label_app.REOLINK_SOURCE, "restaurant-pi-1")
    first_count = label_app._prepare_reolink_unlabeled_queue(
        fake,
        context,
        target_unlabeled_count=1_000_000,
    )
    second_count = label_app._prepare_reolink_unlabeled_queue(
        fake,
        context,
        target_unlabeled_count=1_000_000,
    )

    assert first_count > 0
    assert second_count == 0
    assert calls == ["r-ready"]
    state = json.loads((tmp_path / label_app.PREPROCESS_STATE_FILE_NAME).read_text())
    record = state["reolink_processed"]["restaurant-pi-1:r-ready"]
    assert record["status"] == "complete"
    assert record["generated"] == first_count
    assert fake.items["r-ready"]["appProperties"]["autolabel_preprocess_status"] == "complete"
    assert fake.items["r-ready"]["appProperties"]["autolabel_preprocess_site_key"] == "restaurant-pi-1"
    assert fake.items["r-ready"]["parents"] == ["site-restaurant:processed_raw"]


def test_reolink_preprocess_skips_raw_folder_already_in_local_state(monkeypatch, tmp_path):
    fake = FakeDriveClient()
    monkeypatch.setenv("DRIVE_PROJECT_ROOT_FOLDER_ID", "project-root")
    monkeypatch.setenv("PREPROCESS_STATE_DIR", str(tmp_path))
    state = {
        "schema_version": label_app.PREPROCESS_STATE_SCHEMA_VERSION,
        "reolink_processed": {
            "restaurant-pi-1:r-ready": {
                "site_key": "restaurant-pi-1",
                "raw_folder_id": "r-ready",
                "raw_folder_name": "Reolink-CH-CH04_t0004",
                "status": "complete",
                "generated": 1,
                "reason": "",
                "processed_at": "2026-04-24T00:00:00Z",
            }
        },
    }
    (tmp_path / label_app.PREPROCESS_STATE_FILE_NAME).write_text(json.dumps(state), encoding="utf-8")
    label_app._source_folder_ids_cache.clear()

    def fail_materialize(*_args, **_kwargs):
        raise AssertionError("state-recorded raw folder should not be materialized again")

    monkeypatch.setattr(label_app, "_materialize_reolink_table_crops", fail_materialize)

    context = label_app._resolve_queue_context(fake, label_app.REOLINK_SOURCE, "restaurant-pi-1")
    generated = label_app._prepare_reolink_unlabeled_queue(
        fake,
        context,
        target_unlabeled_count=1_000_000,
    )

    assert generated == 0
    assert fake.items["r-ready"]["appProperties"]["autolabel_preprocess_status"] == "complete"


def test_reolink_preprocess_records_existing_drive_folders_in_local_state(monkeypatch, tmp_path):
    fake = FakeDriveClient()
    monkeypatch.setenv("DRIVE_PROJECT_ROOT_FOLDER_ID", "project-root")
    monkeypatch.setenv("PREPROCESS_STATE_DIR", str(tmp_path))
    label_app._source_folder_ids_cache.clear()

    mapped = label_app._mapped_camera_tables_for_reolink_folder(
        "Reolink-CH-CH04_t0004",
        site_key="restaurant-pi-1",
        client=fake,
    )
    assert mapped is not None
    for table_id, *_rest in mapped[2]:
        legacy_name = f"Reolink-CH-CH04_{table_id}_t0004"
        legacy_id = fake.ensure_subfolder("r-unlabeled", legacy_name)
        fake._add_triplet_files(legacy_id, f"legacy-rready-{table_id}")

    def fail_materialize(*_args, **_kwargs):
        raise AssertionError("existing Drive folders should prevent materialization")

    monkeypatch.setattr(label_app, "_materialize_reolink_table_crops", fail_materialize)

    context = label_app._resolve_queue_context(fake, label_app.REOLINK_SOURCE, "restaurant-pi-1")
    generated = label_app._prepare_reolink_unlabeled_queue(
        fake,
        context,
        target_unlabeled_count=1_000_000,
    )

    assert generated == 0
    state = json.loads((tmp_path / label_app.PREPROCESS_STATE_FILE_NAME).read_text())
    assert state["reolink_processed"]["restaurant-pi-1:r-ready"]["status"] == "complete"
    assert fake.items["r-ready"]["parents"] == ["site-restaurant:processed_raw"]


def test_reolink_preprocess_failed_materialization_does_not_mark_state(monkeypatch, tmp_path):
    fake = FakeDriveClient()
    monkeypatch.setenv("DRIVE_PROJECT_ROOT_FOLDER_ID", "project-root")
    monkeypatch.setenv("PREPROCESS_STATE_DIR", str(tmp_path))
    label_app._source_folder_ids_cache.clear()

    def fail_materialize(*_args, **_kwargs):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(label_app, "_materialize_reolink_table_crops", fail_materialize)

    context = label_app._resolve_queue_context(fake, label_app.REOLINK_SOURCE, "restaurant-pi-1")
    with pytest.raises(RuntimeError, match="upload failed"):
        label_app._prepare_reolink_unlabeled_queue(
            fake,
            context,
            target_unlabeled_count=1_000_000,
        )

    assert not (tmp_path / label_app.PREPROCESS_STATE_FILE_NAME).exists()
    assert fake.items["r-ready"]["appProperties"]["autolabel_preprocess_status"] == "in_progress"
    assert fake.items["r-ready"]["parents"] == ["r-unassociated"]


def test_reolink_processed_raw_cleanup_trashes_only_old_items(fake_drive):
    processed_id = fake_drive.ensure_subfolder("site-restaurant", label_app.PROCESSED_RAW_FOLDER_NAME)
    fake_drive._add_folder("old-raw", "old_raw", processed_id)
    fake_drive._add_folder("new-raw", "new_raw", processed_id)
    fake_drive.items["old-raw"]["modifiedTime"] = (
        datetime.now(timezone.utc) - timedelta(days=15)
    ).isoformat().replace("+00:00", "Z")
    fake_drive.items["new-raw"]["modifiedTime"] = (
        datetime.now(timezone.utc) - timedelta(days=2)
    ).isoformat().replace("+00:00", "Z")

    cleaned = label_app._cleanup_processed_raw_folder(
        fake_drive,
        processed_id,
        retention_days=14,
    )

    assert cleaned == 1
    assert fake_drive.items["old-raw"]["trashed"] is True
    assert fake_drive.items["new-raw"]["trashed"] is False
