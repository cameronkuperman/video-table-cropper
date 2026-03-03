import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from camera_table_metadata import (
    approved_tables_path,
    build_static_tables_for_frame,
    detect_camera_from_filename,
    get_camera_config,
    get_legacy_camera_config,
    get_normalized_camera_config,
    load_camera_configs,
    load_normalized_camera_configs,
)


class CameraTableMetadataTests(unittest.TestCase):
    def test_approved_tables_path_prefers_rectangle_export_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rectangle_path = Path(temp_dir) / "approved_table_rectangles.json"
            fallback_path = Path(temp_dir) / "approved_tables.json"
            rectangle_path.write_text("{}", encoding="utf-8")
            fallback_path.write_text("{}", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                with patch("camera_table_metadata.DEFAULT_APPROVED_TABLES_PATHS", (rectangle_path, fallback_path)):
                    self.assertEqual(approved_tables_path(), rectangle_path)

    def test_detect_camera_from_filename_supports_ipc_and_numeric_prefix(self) -> None:
        self.assertEqual(detect_camera_from_filename("3_1_Mimosas_IPC3_2025.mp4"), "IPC3")
        self.assertEqual(detect_camera_from_filename("4_clip.mp4"), "IPC4")
        self.assertIsNone(detect_camera_from_filename("no_camera_here.mp4"))

    def test_load_and_build_static_tables_from_json(self) -> None:
        payload = {
            "IPC3": {
                "frame_width": 200,
                "frame_height": 100,
                "tables": [
                    {
                        "id": 7,
                        "bbox": {"x1": 20, "y1": 10, "x2": 80, "y2": 50},
                        "vector": [0.0, 1.0, 2.0, 3.0],
                    }
                ],
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            approved_path = Path(temp_dir) / "approved_tables.json"
            approved_path.write_text(json.dumps(payload), encoding="utf-8")

            with patch.dict(os.environ, {"APPROVED_TABLES_JSON_PATH": str(approved_path)}, clear=False):
                raw = load_camera_configs(force_reload=True)
                normalized = load_normalized_camera_configs(force_reload=True)
                self.assertIn("IPC3", raw)
                self.assertIn("IPC3", normalized)
                self.assertEqual(get_camera_config("IPC3")["frame_width"], 200)
                self.assertEqual(get_normalized_camera_config("3")["tables"][0]["track_id"], "table_7")

                tables = build_static_tables_for_frame("IPC3", (100, 200))

        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["track_id"], "table_7")
        self.assertEqual(tuple(tables[0]["vector"].shape), (256,))
        self.assertEqual(tables[0]["vector_source"], "approved_tables_json:vector")
        self.assertEqual(tables[0]["bbox_xyxy"], (20, 10, 81, 51))
        self.assertGreaterEqual(tables[0]["expanded_bbox_xyxy"][0], 0)

    def test_geometry_fallback_vector_is_used_when_metadata_vector_is_missing(self) -> None:
        payload = {
            "IPC9": {
                "frame_width": 100,
                "frame_height": 100,
                "tables": [
                    {
                        "id": 1,
                        "rotated_bbox": {
                            "corners": [[10, 10], [40, 10], [40, 40], [10, 40]],
                        },
                    }
                ],
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            approved_path = Path(temp_dir) / "approved_tables.json"
            approved_path.write_text(json.dumps(payload), encoding="utf-8")

            with patch.dict(os.environ, {"APPROVED_TABLES_JSON_PATH": str(approved_path)}, clear=False):
                load_camera_configs(force_reload=True)
                load_normalized_camera_configs(force_reload=True)
                tables = build_static_tables_for_frame("9", (100, 100))

        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["vector_source"], "geometry_fallback")
        self.assertEqual(tuple(tables[0]["vector"].shape), (256,))

    def test_real_export_shape_with_camera_list_and_dense_hulls(self) -> None:
        payload = {
            "exported_at": "2026-03-03T00:00:00Z",
            "cameras": [
                {
                    "camera_id": "mock-1",
                    "camera_name": "Mimosas IPC3",
                    "camera_number": 3,
                    "image_width": 200,
                    "image_height": 100,
                    "tables": [
                        {
                            "mask_id": 12,
                            "compactness": 0.77,
                            "hull_dense_points": [[20, 10], [80, 10], [80, 40], [20, 40]],
                            "expanded_hull_dense_points": [[10, 5], [90, 5], [90, 50], [10, 50]],
                        }
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            approved_path = Path(temp_dir) / "approved_tables.json"
            approved_path.write_text(json.dumps(payload), encoding="utf-8")

            with patch.dict(os.environ, {"APPROVED_TABLES_JSON_PATH": str(approved_path)}, clear=False):
                normalized = get_normalized_camera_config("IPC3")
                legacy = get_legacy_camera_config("IPC3")
                tables = build_static_tables_for_frame("IPC3", (100, 200))

        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["tables"][0]["track_id"], "table_12")
        self.assertEqual(normalized["tables"][0]["compactness"], 0.77)
        self.assertIsNotNone(normalized["tables"][0]["expanded_points"])
        self.assertEqual(legacy["tables"][0]["bbox"], {"x1": 20, "y1": 10, "x2": 81, "y2": 41})
        self.assertEqual(tables[0]["track_id"], "table_12")
        self.assertEqual(tables[0]["compactness"], 0.77)
        self.assertEqual(tables[0]["bbox_xyxy"], (20, 10, 81, 41))
        self.assertEqual(tables[0]["expanded_bbox_xyxy"], (10, 5, 91, 51))

    def test_tight_rect_and_zone_rect_take_priority_over_hulls(self) -> None:
        payload = {
            "cameras": [
                {
                    "camera_name": "Mimosas IPC3",
                    "image_width": 200,
                    "image_height": 100,
                    "tables": [
                        {
                            "mask_id": 9,
                            "tight_rect": {
                                "center_x": 50,
                                "center_y": 30,
                                "width_px": 40,
                                "height_px": 20,
                                "angle_deg": 0,
                                "polygon": [[30, 20], [70, 20], [70, 40], [30, 40]],
                            },
                            "zone_rect": {
                                "center_x": 50,
                                "center_y": 30,
                                "width_px": 80,
                                "height_px": 40,
                                "angle_deg": 0,
                                "polygon": [[10, 10], [90, 10], [90, 50], [10, 50]],
                            },
                            "hull_dense_points": [[1, 1], [2, 1], [2, 2], [1, 2]],
                            "expanded_hull_dense_points": [[0, 0], [3, 0], [3, 3], [0, 3]],
                        }
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            approved_path = Path(temp_dir) / "approved_tables.json"
            approved_path.write_text(json.dumps(payload), encoding="utf-8")

            with patch.dict(os.environ, {"APPROVED_TABLES_JSON_PATH": str(approved_path)}, clear=False):
                normalized = get_normalized_camera_config("IPC3")
                tables = build_static_tables_for_frame("IPC3", (100, 200))

        self.assertEqual(normalized["tables"][0]["points"], [[30.0, 20.0], [70.0, 20.0], [70.0, 40.0], [30.0, 40.0]])
        self.assertEqual(
            normalized["tables"][0]["expanded_points"],
            [[10.0, 10.0], [90.0, 10.0], [90.0, 50.0], [10.0, 50.0]],
        )
        self.assertEqual(
            normalized["tables"][0]["tight_rect"]["polygon"],
            [[30.0, 20.0], [70.0, 20.0], [70.0, 40.0], [30.0, 40.0]],
        )
        self.assertEqual(
            normalized["tables"][0]["zone_rect"]["polygon"],
            [[10.0, 10.0], [90.0, 10.0], [90.0, 50.0], [10.0, 50.0]],
        )
        self.assertEqual(tables[0]["bbox_xyxy"], (30, 20, 71, 41))
        self.assertEqual(tables[0]["expanded_bbox_xyxy"], (10, 10, 91, 51))

    def test_rect_params_without_polygon_use_backend_corner_order(self) -> None:
        payload = {
            "cameras": [
                {
                    "camera_name": "Mimosas IPC3",
                    "image_width": 200,
                    "image_height": 100,
                    "tables": [
                        {
                            "mask_id": 5,
                            "tight_rect": {
                                "center_x": 50,
                                "center_y": 30,
                                "width_px": 40,
                                "height_px": 20,
                                "angle_deg": 30,
                            },
                            "zone_rect": {
                                "center_x": 50,
                                "center_y": 30,
                                "width_px": 80,
                                "height_px": 40,
                                "angle_deg": 30,
                            },
                        }
                    ],
                }
            ]
        }

        def rect_points(cx: float, cy: float, width: float, height: float, angle_deg: float) -> list[list[float]]:
            theta = math.radians(angle_deg)
            ux = math.cos(theta)
            uy = math.sin(theta)
            vx = -math.sin(theta)
            vy = math.cos(theta)
            half_long = width / 2.0
            half_short = height / 2.0
            corners_local = [
                (-half_long, -half_short),
                (half_long, -half_short),
                (half_long, half_short),
                (-half_long, half_short),
            ]
            return [
                [cx + (long_off * ux) + (short_off * vx), cy + (long_off * uy) + (short_off * vy)]
                for long_off, short_off in corners_local
            ]

        with tempfile.TemporaryDirectory() as temp_dir:
            approved_path = Path(temp_dir) / "approved_tables.json"
            approved_path.write_text(json.dumps(payload), encoding="utf-8")

            with patch.dict(os.environ, {"APPROVED_TABLES_JSON_PATH": str(approved_path)}, clear=False):
                normalized = get_normalized_camera_config("IPC3")

        self.assertIsNotNone(normalized)
        expected_tight = rect_points(50.0, 30.0, 40.0, 20.0, 30.0)
        expected_zone = rect_points(50.0, 30.0, 80.0, 40.0, 30.0)

        for actual, expected in zip(normalized["tables"][0]["points"], expected_tight):
            self.assertAlmostEqual(actual[0], expected[0], places=6)
            self.assertAlmostEqual(actual[1], expected[1], places=6)

        for actual, expected in zip(normalized["tables"][0]["expanded_points"], expected_zone):
            self.assertAlmostEqual(actual[0], expected[0], places=6)
            self.assertAlmostEqual(actual[1], expected[1], places=6)
