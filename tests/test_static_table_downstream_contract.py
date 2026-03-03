import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from camera_table_metadata import build_static_tables_for_frame
from dataset_schema import OCCUPANCY_MLP_NPZ_NAME, VECTOR_DIM
from sample_builder import (
    prepare_audit_export,
    prepare_occupancy_export,
    prepare_surface_export,
    prepare_temporal_export,
    write_review_bundle,
)
from video_dataset_worker import VideoDatasetWorker


class _FakeAdapter:
    def derive_region_vector(self, image, mask):
        return np.zeros((VECTOR_DIM,), dtype=np.float32)


class StaticTableDownstreamContractTests(unittest.TestCase):
    def test_real_approved_tables_json_produces_compatible_exports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            frame_paths = []
            for index in range(3):
                frame_path = temp_root / f"frame_{index}.jpg"
                Image.new("RGB", (1280, 720), color=(24 + index, 24 + index, 24 + index)).save(frame_path)
                frame_paths.append(frame_path)

            tables = build_static_tables_for_frame("IPC3", (720, 1280))
            self.assertTrue(tables, "expected static tables for IPC3 from approved_tables.json")
            table_track_id = tables[0]["track_id"]

            triplet = []
            for timestamp_seconds, frame_path in zip((10, 20, 30), frame_paths):
                frame_tables = build_static_tables_for_frame("IPC3", (720, 1280))
                triplet.append(
                    {
                        "image_path": frame_path,
                        "timestamp_seconds": timestamp_seconds,
                        "image_shape": (720, 1280),
                        "tables": frame_tables,
                        "people": [],
                    }
                )

            worker = VideoDatasetWorker.__new__(VideoDatasetWorker)
            worker.adapter = _FakeAdapter()
            sample_payload, perception_payload, review_images = worker._build_sample_bundle(
                video_meta={
                    "id": "video-1",
                    "name": "3_1_Mimosas_IPC3_test.mp4",
                    "source_folder_id": "folder-1",
                    "source_folder_name": "2026-03-03",
                },
                camera_id="IPC3",
                triplet=triplet,
                table_track_id=table_track_id,
            )

            self.assertEqual(sample_payload["table"]["table_track_id"], table_track_id)
            self.assertEqual(len(sample_payload["frames"]), 3)
            self.assertIn("compactness_anchor", sample_payload["table"])
            self.assertIn("mask_id", sample_payload["table"])
            self.assertIn("label", sample_payload["table"])
            self.assertIn("bbox", sample_payload["table"])
            self.assertIn("tight_rect", sample_payload["table"])
            self.assertIn("zone_rect", sample_payload["table"])
            self.assertIn("tight_hull_anchor_polygon", sample_payload["table"])
            self.assertEqual(sample_payload["source_video"]["camera_id"], "IPC3")
            self.assertEqual(sample_payload["source_video"]["image_width"], 1280)
            self.assertEqual(sample_payload["source_video"]["image_height"], 720)
            self.assertEqual(len(sample_payload["table"]["tight_rect"]["polygon"]), 4)
            self.assertEqual(len(sample_payload["table"]["zone_rect"]["polygon"]), 4)
            self.assertFalse(sample_payload["quality_flags"]["bbox_fallback_used"])
            self.assertEqual(tuple(perception_payload["table_vecs_tight"].shape), (3, VECTOR_DIM))
            # Rotated crops may introduce black fill at edges; verify crops
            # are non-empty and contain meaningful image content (mean > 0).
            self.assertGreater(review_images["anchor"].size[0], 0)
            self.assertGreater(review_images["anchor"].size[1], 0)
            self.assertGreater(float(np.array(review_images["anchor"]).mean()), 0)
            self.assertGreater(review_images["tight_anchor"].size[0], 0)
            self.assertGreater(review_images["tight_anchor"].size[1], 0)
            self.assertGreater(float(np.array(review_images["tight_anchor"]).mean()), 0)

            for frame_meta in sample_payload["frames"]:
                self.assertIn("tight_bbox", frame_meta)
                self.assertIn("expanded_bbox", frame_meta)
                self.assertIn("tight_area", frame_meta)
                self.assertIn("expanded_area", frame_meta)
                self.assertGreater(frame_meta["tight_area"], 0)
                self.assertGreaterEqual(frame_meta["expanded_area"], frame_meta["tight_area"])

            sample_dir = temp_root / "sample_bundle"
            write_review_bundle(sample_dir, sample_payload, review_images, perception_payload)

            temporal_payload, temporal_files = prepare_temporal_export(sample_dir, "clean")
            surface_payload, surface_files = prepare_surface_export(sample_dir, "dirty")
            occupancy_payload, occupancy_files = prepare_occupancy_export(sample_dir, "occupied")
            audit_payload, audit_files = prepare_audit_export(sample_dir, "clean")

            self.assertEqual(temporal_payload["table"]["table_track_id"], table_track_id)
            self.assertEqual(surface_payload["table"]["table_track_id"], table_track_id)
            self.assertEqual(occupancy_payload["table"]["table_track_id"], table_track_id)
            self.assertEqual(audit_payload["table"]["table_track_id"], table_track_id)
            self.assertIn("frame_0.jpg", temporal_files)
            self.assertIn("surface.jpg", surface_files)
            self.assertIn(OCCUPANCY_MLP_NPZ_NAME, occupancy_files)
            self.assertIn("perception.npz", audit_files)

            with np.load(io.BytesIO(occupancy_files[OCCUPANCY_MLP_NPZ_NAME]), allow_pickle=False) as data:
                self.assertIn("x", data.files)
                self.assertIn("y", data.files)
                self.assertGreater(data["x"].size, 0)
