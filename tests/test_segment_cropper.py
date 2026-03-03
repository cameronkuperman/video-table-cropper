import unittest

import numpy as np
from PIL import Image

from segment_cropper import crop_with_polygonal_rect


class SegmentCropperTests(unittest.TestCase):
    def test_polygonal_rect_crop_masks_pixels_outside_polygon(self) -> None:
        image = Image.new("RGB", (11, 11), color=(120, 80, 40))
        rect = {
            "polygon": [[5, 1], [9, 5], [5, 9], [1, 5]],
            "center_x": 5,
            "center_y": 5,
            "width_px": 8,
            "height_px": 8,
            "angle_deg": 45,
        }

        cropped = crop_with_polygonal_rect(image, rect)
        pixels = np.array(cropped)

        self.assertEqual(cropped.size, (9, 9))
        self.assertEqual(tuple(pixels[0, 0]), (0, 0, 0))
        self.assertEqual(tuple(pixels[4, 4]), (120, 80, 40))
