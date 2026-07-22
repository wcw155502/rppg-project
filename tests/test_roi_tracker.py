import unittest
import sys
import types

import numpy as np

# CI/精简环境可能未安装 OpenCV；该状态测试不调用任何 cv2 API，使用空模块即可。
try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules["cv2"] = types.ModuleType("cv2")

from legacy.roi_tracker import ROITracker


class FakeDetector:
    def __init__(self, detections):
        self.detections = iter(detections)

    def detect(self, _frame, max_num=1):
        boxes = next(self.detections)
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 5), None


class ROITrackerNoFaceTests(unittest.TestCase):
    def test_missing_face_clears_previous_boxes_immediately(self):
        detector = FakeDetector([
            [[10, 10, 90, 90, 0.99]],
            [],
        ])
        tracker = ROITracker(detector, detect_every_n_frames=1)
        tracker._init_cv_tracker = lambda _frame, _box: None
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        self.assertIsNotNone(tracker.update(frame))
        self.assertTrue(tracker.has_face())
        self.assertTrue(tracker.get_last_regions())

        self.assertIsNone(tracker.update(frame))
        self.assertFalse(tracker.has_face())
        self.assertEqual(tracker.get_last_regions(), {})
        self.assertIsNone(tracker.last_box)

    def test_model_roi_is_expanded_and_clipped_to_frame(self):
        tracker = ROITracker(FakeDetector([]), model_bbox_scale=1.2)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        regions = tracker._crop_subregions(frame, np.array([10, 20, 90, 80]))

        self.assertEqual(tracker.get_last_regions()["full"], (10, 20, 90, 80))
        self.assertEqual(tracker.get_last_regions()["model_full"], (2, 14, 98, 86))
        self.assertEqual(regions["model_full"].shape[:2], (72, 96))


if __name__ == "__main__":
    unittest.main()
