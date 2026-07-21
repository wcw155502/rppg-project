import unittest

import numpy as np

from input_pipeline.timing import FrameResampler, TimingMonitor
from input_pipeline.types import FramePacket
from input_pipeline.config import build_input_config


def packet(timestamp_ns, frame_id=0, dropped=0, value=0):
    return FramePacket(
        frame=np.full((2, 2, 3), value, dtype=np.uint8),
        timestamp_ns=timestamp_ns,
        frame_id=frame_id,
        dropped_frames=dropped,
    )


class TimingTests(unittest.TestCase):
    def test_resampler_produces_fixed_timestamps(self):
        resampler = FrameResampler(target_fps=10, max_gap_ms=250)
        first, reset, _ = resampler.push(packet(0, value=0))
        second, reset2, _ = resampler.push(packet(220_000_000, 1, value=220))

        self.assertFalse(reset)
        self.assertFalse(reset2)
        self.assertEqual([p.timestamp_ns for p in first + second], [0, 100_000_000, 200_000_000])
        self.assertEqual(int(second[0].frame.mean()), 100)

    def test_gap_requires_reset_and_is_not_interpolated(self):
        resampler = FrameResampler(target_fps=10, max_gap_ms=150)
        resampler.push(packet(0))

        output, reset, reason = resampler.push(packet(300_000_000, 1))

        self.assertTrue(reset)
        self.assertEqual(reason, "timestamp_gap")
        self.assertEqual([p.timestamp_ns for p in output], [300_000_000])

    def test_short_queue_drop_is_resampled_without_reset(self):
        resampler = FrameResampler(target_fps=30, max_gap_ms=150)
        resampler.push(packet(0))

        output, reset, reason = resampler.push(
            packet(66_666_667, frame_id=2, dropped=1, value=100)
        )

        self.assertFalse(reset)
        self.assertIsNone(reason)
        self.assertEqual(
            [p.timestamp_ns for p in output],
            [33_333_333, 66_666_666],
        )

    def test_realtime_mode_only_emits_latest_catchup_frame(self):
        resampler = FrameResampler(
            target_fps=10, max_gap_ms=500, max_outputs_per_push=1
        )
        resampler.push(packet(0))

        output, reset, _ = resampler.push(packet(350_000_000, frame_id=3, value=200))

        self.assertFalse(reset)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].timestamp_ns, 300_000_000)

    def test_timing_monitor_reports_expected_fps(self):
        monitor = TimingMonitor()
        stats = None
        for index in range(11):
            stats = monitor.update(index * 100_000_000)
        self.assertAlmostEqual(stats.fps, 10.0)
        self.assertAlmostEqual(stats.interval_std_ms, 0.0)

    def test_target_fps_must_match_signal_processing(self):
        with self.assertRaises(ValueError):
            build_input_config({
                "input_pipeline": {"target_fps": 25},
                "signal_processing": {"fs": 30},
            })


if __name__ == "__main__":
    unittest.main()
