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
        ]

    def list_files(self, folder_id: str, fields: str = "") -> list[dict]:
        return [self._copy(child_id) for child_id in self.children.get(folder_id, [])]

    def get_file(self, file_id: str, fields: str = "") -> dict:
        return self._copy(file_id)

    def update_file_metadata(self, file_id: str, metadata: dict, fields: str = "") -> dict:
        self.items[file_id].setdefault("appProperties", {}).update(metadata.get("appProperties", {}))
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
            return self._copy(existing["id"])

        item_id = f"{parent_id}:{file_name}"
        self._add_file(item_id, file_name, parent_id, mime_type=mime_type, content=data)
        return self._copy(item_id)


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
def fake_drive(monkeypatch):
    fake = FakeDriveClient()

    monkeypatch.setenv("DRIVE_PROJECT_ROOT_FOLDER_ID", "project-root")
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
        lambda client, context, target_unlabeled_count: _fake_prepare_reolink_unlabeled_queue(
            fake,
            context,
            target_unlabeled_count,
        ),
    )
    monkeypatch.setattr(label_app, "_maybe_trigger_video_preprocess", lambda context, unlabeled_count: None)

    label_app._source_folder_ids_cache.clear()
    label_app._listing_cache.clear()
    label_app._listing_refresh_inflight.clear()
    label_app._hydrated_folder_cache.clear()
    label_app._preview_prewarm_inflight.clear()
    label_app._folder_prewarm_inflight.clear()
    label_app._camera_config_cache = None
    label_app._crop_config_cache.clear()

    return fake


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
    assert state["errors"] == []


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
        json={
            "folder_id": folder["folder_id"],
            "parent_id": folder["parent_id"],
            "label": "clean",
            "source": folder["source"],
            "site_key": folder["site_key"],
        },
    )

    assert label_response.status_code == 200
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
        json={
            "folder_id": folder["folder_id"],
            "parent_id": folder["parent_id"],
            "label": "discarded",
            "source": folder["source"],
            "site_key": folder["site_key"],
        },
    )

    assert discarded_response.status_code == 200
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
        json={
            "folder_id": folder["folder_id"],
            "parent_id": folder["parent_id"],
            "label": "clean",
            "source": folder["source"],
            "site_key": folder["site_key"],
        },
    )

    assert label_response.status_code == 200
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
    payload = {
        "folder_id": folder["folder_id"],
        "parent_id": folder["parent_id"],
        "label": "dirty",
        "source": folder["source"],
        "site_key": folder["site_key"],
    }

    first_response = client.post("/api/label", json=payload)
    second_response = client.post("/api/label", json=payload)

    assert first_response.status_code == 200
    folder_item = fake_drive.items[folder["folder_id"]]
    assert folder_item["appProperties"]["autolabel_final_label"] == "dirty"
    assert folder_item["appProperties"]["autolabel_source"] == "video"
    assert folder_item["appProperties"]["autolabel_queue_key"] == "video"
    assert folder_item["appProperties"]["autolabel_labeled_by"] == "local"
    assert second_response.status_code == 409
    assert second_response.get_json()["code"] == "already_labeled"


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
    assert fake.items["r-ready"]["appProperties"] == {}
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


def test_reolink_preprocess_records_existing_drive_folders_in_local_state(monkeypatch, tmp_path):
    fake = FakeDriveClient()
    monkeypatch.setenv("DRIVE_PROJECT_ROOT_FOLDER_ID", "project-root")
    monkeypatch.setenv("PREPROCESS_STATE_DIR", str(tmp_path))
    label_app._source_folder_ids_cache.clear()

    legacy_name = "Reolink-CH-CH04_table_top_1_t0004"
    legacy_id = fake.ensure_subfolder("r-unlabeled", legacy_name)
    fake._add_triplet_files(legacy_id, "legacy-rready")

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
