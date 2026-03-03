import json
import tempfile
import unittest
from pathlib import Path

from dataset_schema import PERCEPTION_NPZ_NAME, PREVIEW_FILE_BY_KIND, SAMPLE_JSON_NAME
from video_review_service import ensure_cached_sample


class FakeDriveClient:
    def __init__(self) -> None:
        self.downloaded_ids: list[str] = []

    def download_file_to_path(self, file_id: str, output_path: Path) -> Path:
        self.downloaded_ids.append(file_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"payload-for-{file_id}".encode("utf-8"))
        return output_path


class EnsureCachedSampleTests(unittest.TestCase):
    def test_uses_shared_cache_namespace_without_session_id(self) -> None:
        client = FakeDriveClient()
        item = {
            "id": 7,
            "sample": {"sample_id": "sample-7", "label": {"human_label": None}},
            "preview_anchor_file_id": "anchor-id",
            "preview_t_minus_10_file_id": "prev-1-id",
            "preview_t_minus_20_file_id": "prev-2-id",
            "tight_anchor_file_id": "tight-id",
            "perception_file_id": "perception-id",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            sample_dir = ensure_cached_sample(client, item, Path(temp_dir))

            self.assertEqual(sample_dir, Path(temp_dir) / "shared" / "7")
            self.assertTrue((sample_dir / SAMPLE_JSON_NAME).exists())
            self.assertEqual(
                json.loads((sample_dir / SAMPLE_JSON_NAME).read_text(encoding="utf-8"))["sample_id"],
                "sample-7",
            )
            for file_name in PREVIEW_FILE_BY_KIND.values():
                self.assertTrue((sample_dir / file_name).exists(), file_name)
            self.assertTrue((sample_dir / PERCEPTION_NPZ_NAME).exists())
            self.assertEqual(
                client.downloaded_ids,
                ["anchor-id", "prev-1-id", "prev-2-id", "tight-id", "perception-id"],
            )
