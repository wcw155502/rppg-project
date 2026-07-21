import unittest

import numpy as np

from core.fusion import extract_skin_rgb_mean, pos_algorithm


class SkinOcclusionTests(unittest.TestCase):
    def test_skin_colored_roi_is_accepted(self):
        roi = np.full((20, 20, 3), [100, 130, 180], dtype=np.uint8)

        rgb, quality = extract_skin_rgb_mean(roi, min_skin_ratio=0.25)

        self.assertTrue(quality["visible"])
        self.assertAlmostEqual(quality["skin_ratio"], 1.0)
        np.testing.assert_allclose(rgb, [180, 130, 100])

    def test_dark_occluded_roi_is_rejected(self):
        roi = np.zeros((20, 20, 3), dtype=np.uint8)

        rgb, quality = extract_skin_rgb_mean(roi, min_skin_ratio=0.25)

        self.assertIsNone(rgb)
        self.assertFalse(quality["visible"])

    def test_pos_output_is_finite_and_has_input_length(self):
        fs = 30
        t = np.arange(fs * 3) / fs
        rgb = np.column_stack([
            180 + np.sin(2 * np.pi * 1.2 * t),
            130 + 2 * np.sin(2 * np.pi * 1.2 * t),
            100 + 0.5 * np.sin(2 * np.pi * 1.2 * t),
        ])

        bvp = pos_algorithm(rgb, fs)

        self.assertEqual(len(bvp), len(rgb))
        self.assertTrue(np.all(np.isfinite(bvp)))


if __name__ == "__main__":
    unittest.main()
