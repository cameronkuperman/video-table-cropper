import unittest

from video_review_store_pg import VideoReviewStorePG


class VideoReviewStorePGRowTests(unittest.TestCase):
    def test_row_to_dict_decodes_json_and_sets_shared_session_fallback(self) -> None:
        row = {
            "id": 11,
            "sample_json": '{"sample_id":"sample-11"}',
            "exported_folder_ids": '["folder-a"]',
            "claimed_by_session_id": None,
        }

        decoded = VideoReviewStorePG._row_to_dict(row)

        self.assertEqual(decoded["sample"]["sample_id"], "sample-11")
        self.assertEqual(decoded["exported_folder_ids"], ["folder-a"])
        self.assertEqual(decoded["session_id"], "shared")
