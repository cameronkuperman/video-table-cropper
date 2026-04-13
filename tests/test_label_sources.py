from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

import app as label_app


class FakeDriveClient:
    def __init__(self) -> None:
        self.items: dict[str, dict] = {}
        self.children: dict[str, list[str]] = defaultdict(list)
        self.created_subfolders: list[tuple[str, str, str]] = []
        self.moves: list[tuple[str, str, str | None]] = []
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
            "content": content if content is not None else b"{}",
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
        self.children[new_parent_id].append(file_id)
        self.moves.append((file_id, new_parent_id, remove_parent_id))
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


def _fake_prepare_reolink_unlabeled_queue(fake: FakeDriveClient, context, target_unlabeled_count: int) -> None:
    if context.source != label_app.REOLINK_SOURCE or not context.seed_folder_id:
        return

    label_app._assert_manual_crop_setup_ready(fake, context)
    existing_names = label_app._existing_generated_folder_names(fake, context)
    raw_folders = fake.list_folders(context.seed_folder_id)
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
            derived_name = label_app._derived_reolink_folder_name(raw_folder["name"], table_id)
            if derived_name in existing_names:
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


@pytest.fixture()
def fake_drive(monkeypatch):
    fake = FakeDriveClient()

    monkeypatch.setenv("DRIVE_PROJECT_ROOT_FOLDER_ID", "project-root")
    monkeypatch.setattr(label_app, "DriveClient", lambda: fake)
    monkeypatch.setattr(label_app, "get_client", lambda: fake)
    monkeypatch.setattr(
        label_app,
        "_frames_cache_ready",
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
        "reolink-matthews-01",
        "restaurant-pi-1",
    ]
    matthews_site = payload["reolink_sites"][0]
    assert matthews_site["manual_crop_configs"] is True
    assert matthews_site["crop_editor_url"] == "/crop-editor?site=reolink-matthews-01"


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
        "Reolink-CH-CH03_table_top_1_t0002",
        "Reolink-CH-CH03_table_top_2_t0002",
    ]
    assert matthews_payload["total_unlabeled"] == 2

    restaurant_response = client.get("/api/queue?source=reolink&site=restaurant-pi-1&limit=10")
    restaurant_payload = restaurant_response.get_json()

    assert restaurant_response.status_code == 200
    assert restaurant_payload["source_context"]["queue_key"] == "reolink:restaurant-pi-1"
    assert [folder["folder_name"] for folder in restaurant_payload["folders"]] == [
        "Reolink-CH-CH04_table_top_1_t0004"
    ]


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
    matthews_clean_id = fake_drive.find_file_by_name(
        "site-matthews",
        "clean",
        mime_type=label_app.FOLDER_MIME,
    )["id"]
    assert fake_drive.items[folder["folder_id"]]["parents"] == [matthews_clean_id]
    assert fake_drive.items[folder["folder_id"]]["parents"] != ["video-clean"]

    stats_response = client.get("/api/stats?source=reolink&site=reolink-matthews-01")
    stats_payload = stats_response.get_json()
    assert stats_payload["clean"] == 1
    assert stats_payload["unlabeled"] == 1


def test_reolink_queue_allows_triplets_without_metadata_json(client):
    response = client.get("/api/queue?source=reolink&site=restaurant-pi-1&limit=10")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["folders"][0]["folder_name"] == "Reolink-CH-CH04_table_top_1_t0004"


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
