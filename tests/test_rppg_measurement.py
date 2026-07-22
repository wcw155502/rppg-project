import unittest

import numpy as np

from rppg.measurement.fusion import MultiSourceFusion
from rppg.measurement.rppg_measurement import MeasurementEvent, RPPGMeasurement
from rppg.measurement.smoother import MovingAverageSmoother


class RPPGMeasurementTests(unittest.TestCase):
    def build_measurement(self, update_interval_sec=1):
        return RPPGMeasurement(
            runner=None, fs=1, low_hz=0.1, high_hz=0.4, filter_order=1,
            bvp_window_sec=2, update_interval_sec=update_interval_sec, rois_used=[],
            use_pos=False, min_skin_ratio=0.25, min_valid_frame_ratio=0.7,
            max_quality_gap_sec=0.5,
            fusion=MultiSourceFusion(), smoother=MovingAverageSmoother(),
        )

    def test_observation_returns_one_lightweight_interface_result(self):
        measurement = self.build_measurement(update_interval_sec=10)

        result = measurement.consume(MeasurementEvent(1_000_000_000, {"model_full": np.zeros((2, 2, 3), dtype=np.uint8)}))

        self.assertEqual(result.state, "waiting")
        self.assertFalse(result.updated)
        self.assertIsNone(result.candidates)

    def test_short_quality_interrupt_holds_state(self):
        measurement = self.build_measurement(update_interval_sec=10)
        measurement.consume(MeasurementEvent(1_000_000_000))

        result = measurement.consume(MeasurementEvent.interrupt(1_300_000_000, "low_quality", "low_quality"))

        self.assertEqual(result.state, "held")
        self.assertEqual(result.reason, "low_quality")

    def test_long_quality_interrupt_resets(self):
        measurement = self.build_measurement(update_interval_sec=10)
        measurement.consume(MeasurementEvent(1_000_000_000))

        result = measurement.consume(MeasurementEvent.interrupt(1_600_000_000, "low_quality", "low_quality"))

        self.assertEqual(result.state, "reset")
        self.assertEqual(result.reason, "quality_gap")

    def test_explicit_reset_clears_immediately(self):
        measurement = self.build_measurement()
        result = measurement.consume(MeasurementEvent.reset(1_000_000_000, "face_changed"))

        self.assertEqual(result.state, "reset")
        self.assertEqual(result.reason, "face_changed")

    def test_pos_requires_the_full_analysis_window_before_evaluation(self):
        measurement = RPPGMeasurement(
            runner=None, fs=10, low_hz=0.7, high_hz=2.5, filter_order=1,
            bvp_window_sec=10, update_interval_sec=2, rois_used=["forehead"],
            use_pos=True, min_skin_ratio=0.25, min_valid_frame_ratio=0.7,
            max_quality_gap_sec=0.5,
            fusion=MultiSourceFusion(), smoother=MovingAverageSmoother(),
        )
        sample = np.array([180.0, 130.0, 100.0])
        measurement.rgb_buffers["forehead"] = [sample] * 99
        measurement.roi_quality_buffers["forehead"] = [1.0] * 99
        self.assertIsNone(measurement._evaluate_pos_roi("forehead"))

        measurement.rgb_buffers["forehead"].append(sample)
        measurement.roi_quality_buffers["forehead"].append(1.0)
        measurement._evaluate_signal = lambda _signal, _source: {"source": "pos_forehead", "hr": 72.0, "sqi": 1.0}
        self.assertIsNotNone(measurement._evaluate_pos_roi("forehead"))


if __name__ == "__main__":
    unittest.main()
