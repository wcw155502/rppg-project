import unittest

import numpy as np

from rppg.input.face_session import FaceSession, bbox_iou
from rppg.input.types import InputStatus


class FaceSessionTests(unittest.TestCase):
    CFG = {
        "lost_timeout_sec": 0.3,
        "warmup_sec": 1.0,
        "identity_iou_threshold": 0.3,
    }

    def test_new_face_warms_up_then_measures(self):
        session = FaceSession(self.CFG)
        box = np.array([10, 10, 50, 50])
        status, reset, reason = session.update_face(0, box, True)
        self.assertEqual(status, InputStatus.WARMING_UP)
        self.assertTrue(reset)
        self.assertEqual(reason, "new_face")
        status, reset, _ = session.update_face(1_100_000_000, box, True)
        self.assertEqual(status, InputStatus.MEASURING)
        self.assertFalse(reset)

    def test_identity_change_requires_reset(self):
        session = FaceSession(self.CFG)
        session.update_face(0, [0, 0, 40, 40], True)
        status, reset, reason = session.update_face(100_000_000, [100, 100, 140, 140], True)
        self.assertTrue(reset)
        self.assertEqual(reason, "face_changed")
        self.assertEqual(status, InputStatus.WARMING_UP)

    def test_lost_timeout_resets_once(self):
        session = FaceSession(self.CFG)
        session.update_face(0, [0, 0, 40, 40], True)
        status, reset, _ = session.update_missing(200_000_000)
        self.assertFalse(reset)
        status, reset, reason = session.update_missing(400_000_000)
        self.assertTrue(reset)
        self.assertEqual(reason, "face_lost")

    def test_bbox_iou(self):
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)


if __name__ == "__main__":
    unittest.main()
