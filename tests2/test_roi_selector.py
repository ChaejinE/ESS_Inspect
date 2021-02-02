import unittest
import numpy as np
from src.main.python.utils.roi_selector import RoiSelector


class TestRoiSelector(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_operate_fixed(self):
        image = np.ones([10, 10]) * 255.
        image = np.pad(image, [[10, 10], [10, 10]])

        roi_selector = RoiSelector(image, resize=(30, 30), is_mouse=False)
        expected_coord = [(10, 10, 20, 20)]
        roi_selector.operate(coordinates=expected_coord)

        self.assertEqual(expected_coord[0], roi_selector.get_roi_coordinates[0])
        roi_images = roi_selector.get_roi_images()

        for roi_image in roi_images:
            self.assertEqual((np.ones([10, 10]) * 255.).all(), roi_image.all())

    def test_operate_ismouse(self):
        image = np.ones([10, 10]) * 255.
        image = np.pad(image, [[10, 10], [10, 10]])

        roi_selector = RoiSelector(image, resize=(30, 30), is_mouse=True)
        roi_selector.operate()

        roi_images = roi_selector.get_roi_images()

        for idx, roi_image in enumerate(roi_images):
            y0, x0, y1, x1 = roi_selector.get_roi_coordinates[idx]
            self.assertEqual((y1-y0, x1-x0), roi_image.shape)

    def test_operate(self):
        import os
        import cv2
        target_dir = "path/to/image_dir"
        for file_name in os.listdir(target_dir):
            if not file_name.endswith(".jpg"):
                break
            img = cv2.imread(os.path.join(target_dir, file_name))
            roi = RoiSelector(img, (704, 256), is_save=False, is_mouse=True)
            roi.operate()
            """
            예시 
            roi.operate() -> is_mouse=True
            roi.operate([(132, 170, 250, 235), (533, 180, 671, 237)]) -> is_mouse=False
            print(roi.get_roi_coordinates)
            """