import unittest
import sys
import types

import numpy as np

try:
    import cv2
except ImportError:
    sys.modules["cv2"] = types.ModuleType("cv2")
    cv2 = None


@unittest.skipIf(cv2 is None, "OpenCV unavailable")
class AlignmentTests(unittest.TestCase):
    def test_reference_landmarks_align_with_low_error(self):
        from input_pipeline.face_alignment import FaceAligner, REFERENCE_LANDMARKS_112

        aligner = FaceAligner(output_size=112)
        frame = np.zeros((112, 112, 3), dtype=np.uint8)
        aligned, detail = aligner.align(frame, REFERENCE_LANDMARKS_112)

        self.assertEqual(aligned.shape, frame.shape)
        self.assertLess(detail["mean_error_px"], 0.01)

    def test_roi_shapes_are_nonempty(self):
        from input_pipeline.face_alignment import FaceAligner

        aligner = FaceAligner(output_size=144)
        landmarks = aligner.reference.copy()
        rois, boxes, yaw = aligner.extract_rois(
            np.zeros((144, 144, 3), dtype=np.uint8), landmarks
        )
        self.assertTrue(all(roi.size > 0 for roi in rois.values()))
        self.assertIn("model_full", boxes)
        self.assertEqual(boxes["cheek_l"].shape, (4, 2))
        self.assertLess(abs(yaw), 0.01)

    def test_turned_face_shrinks_one_cheek(self):
        from input_pipeline.face_alignment import FaceAligner

        aligner = FaceAligner(output_size=144)
        landmarks = aligner.reference.copy()
        landmarks[2, 0] += 10
        _, boxes, yaw = aligner.extract_rois(
            np.zeros((144, 144, 3), dtype=np.uint8), landmarks
        )
        right_width = boxes["cheek_r"][:, 0].max() - boxes["cheek_r"][:, 0].min()
        left_width = boxes["cheek_l"][:, 0].max() - boxes["cheek_l"][:, 0].min()
        self.assertGreater(yaw, 0)
        self.assertLess(right_width, left_width)

    def test_efficientphys_roi_expands_without_changing_pos_rois(self):
        from input_pipeline.face_alignment import FaceAligner

        aligner = FaceAligner(output_size=144)
        face = np.zeros((144, 144, 3), dtype=np.uint8)
        _, normal_boxes, _ = aligner.extract_rois(face, aligner.reference, 1.0)
        _, expanded_boxes, _ = aligner.extract_rois(face, aligner.reference, 1.5)

        normal = normal_boxes["model_full"]
        expanded = expanded_boxes["model_full"]
        normal_area = (normal[2] - normal[0]) * (normal[3] - normal[1])
        expanded_area = (expanded[2] - expanded[0]) * (expanded[3] - expanded[1])
        self.assertGreater(expanded_area, normal_area)
        self.assertEqual(expanded, (0, 0, 144, 144))
        np.testing.assert_array_equal(
            expanded_boxes["cheek_l"], normal_boxes["cheek_l"]
        )


if __name__ == "__main__":
    unittest.main()
