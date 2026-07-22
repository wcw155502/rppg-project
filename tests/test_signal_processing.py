import unittest

import numpy as np

from rppg.measurement.signal_processing import (
    bandpass_filter,
    signal_quality_index,
    signal_quality_index_legacy,
)


class SignalQualityIndexTests(unittest.TestCase):
    FS = 30
    LOW_HZ = 0.7
    HIGH_HZ = 2.5

    def test_random_noise_does_not_receive_high_sqi(self):
        noise = np.random.default_rng(0).normal(size=self.FS * 10)

        sqi = signal_quality_index(noise, self.FS, self.LOW_HZ, self.HIGH_HZ)

        self.assertLess(sqi, 0.3)

    def test_clean_heart_rate_signal_receives_high_sqi(self):
        t = np.arange(self.FS * 10) / self.FS
        signal = np.sin(2 * np.pi * 1.2 * t) + 0.05 * np.random.default_rng(1).normal(size=t.size)

        sqi = signal_quality_index(signal, self.FS, self.LOW_HZ, self.HIGH_HZ)

        self.assertGreater(sqi, 0.9)

    def test_legacy_behavior_is_retained_for_comparison(self):
        noise = np.random.default_rng(0).normal(size=self.FS * 10)
        filtered_noise = bandpass_filter(noise, self.FS, self.LOW_HZ, self.HIGH_HZ)

        legacy_sqi = signal_quality_index_legacy(
            filtered_noise, self.FS, self.LOW_HZ, self.HIGH_HZ
        )

        self.assertGreater(legacy_sqi, 0.9)


if __name__ == "__main__":
    unittest.main()
