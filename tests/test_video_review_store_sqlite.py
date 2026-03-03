import tempfile
import unittest
from pathlib import Path

from video_review_store import VideoReviewStore


def build_sample(sample_id: str, camera_id: str, video_name: str, table_track_id: str, anchor_time: int) -> dict:
    return {
        "sample_id": sample_id,
        "source_video": {"camera_id": camera_id, "video_name": video_name},
        "table": {"table_track_id": table_track_id},
        "timing": {"anchor_time_seconds": anchor_time},
        "label": {"human_label": None, "occupancy_binary_label": None},
    }


class VideoReviewStoreSqliteTests(unittest.TestCase):
    def test_crud_stats_and_undo_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = VideoReviewStore(Path(temp_dir) / "review.db")
            store.create_session("session-1", "review-root", "pending-root", 60)
            inserted = store.add_queue_items(
                [
                    {
                        "session_id": "session-1",
                        "review_root_folder_id": "review-root",
                        "source_parent_folder_id": "parent-a",
                        "sample_folder_id": "sample-folder-a",
                        "sample_folder_name": "sample-a",
                        "sample": build_sample("sample-a", "IPC3", "video-a.mp4", "table-1", 30),
                        "preview_anchor_file_id": "anchor-a",
                        "preview_t_minus_10_file_id": "prev1-a",
                        "preview_t_minus_20_file_id": "prev2-a",
                        "tight_anchor_file_id": "tight-a",
                        "perception_file_id": "perception-a",
                    },
                    {
                        "session_id": "session-1",
                        "review_root_folder_id": "review-root",
                        "source_parent_folder_id": "parent-b",
                        "sample_folder_id": "sample-folder-b",
                        "sample_folder_name": "sample-b",
                        "sample": build_sample("sample-b", "IPC4", "video-b.mp4", "table-2", 60),
                        "preview_anchor_file_id": "anchor-b",
                        "preview_t_minus_10_file_id": "prev1-b",
                        "preview_t_minus_20_file_id": "prev2-b",
                        "tight_anchor_file_id": "tight-b",
                        "perception_file_id": "perception-b",
                    },
                ]
            )

            self.assertEqual(inserted, 2)
            batch = store.get_pending_batch("session-1", 10, 0)
            self.assertEqual(len(batch), 2)
            self.assertEqual(batch[0]["sample"]["sample_id"], "sample-a")

            first_item = batch[0]
            store.update_item_after_label(
                "session-1",
                first_item["id"],
                "clean",
                ["export-folder-a"],
                "processed-parent",
            )
            store.log_action(
                session_id="session-1",
                queue_item_id=first_item["id"],
                action_type="label",
                prev_status=first_item["status"],
                new_status="labeled",
                prev_label=first_item.get("label"),
                new_label="clean",
                exported_folder_ids=["export-folder-a"],
                moved_folder_id="sample-folder-a",
                archive_parent_folder_id="processed-parent",
            )

            stats = store.get_stats("session-1")
            self.assertEqual(stats["status_counts"]["pending"], 1)
            self.assertEqual(stats["status_counts"]["labeled"], 1)
            self.assertEqual(stats["label_counts"]["clean"], 1)

            action = store.get_last_action("session-1")
            self.assertIsNotNone(action)
            store.restore_item("session-1", first_item["id"], action["prev_status"], action["prev_label"])
            store.mark_action_undone(action["id"])

            restored_stats = store.get_stats("session-1")
            self.assertEqual(restored_stats["status_counts"]["pending"], 2)
            self.assertEqual(restored_stats["status_counts"]["labeled"], 0)
            self.assertTrue(store.has_pending_after("session-1", 0))
