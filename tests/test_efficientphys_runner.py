import unittest

import numpy as np

from rppg.measurement.efficientphys import EfficientPhysRunner


class EfficientPhysRunnerWindowTests(unittest.TestCase):
    def build_runner(self):
        runner = EfficientPhysRunner.__new__(EfficientPhysRunner)
        runner.chunk_length = 5
        runner.stride = 2
        runner.frame_buffer = []
        runner._has_emitted_initial_bvp = False
        calls = iter([np.arange(4), np.arange(10, 14)])
        runner._infer_clip = lambda _clip: next(calls)
        return runner

    def test_first_prediction_keeps_complete_context_then_only_appends_tail(self):
        runner = self.build_runner()

        first = [runner.push_frame(index) for index in range(5)][-1]
        second = [runner.push_frame(index) for index in range(5, 7)][-1]

        np.testing.assert_array_equal(first, [0, 1, 2, 3])
        np.testing.assert_array_equal(second, [12, 13])

    def test_reset_restores_initial_context_behavior(self):
        runner = self.build_runner()
        [runner.push_frame(index) for index in range(5)]
        runner.reset()
        runner._infer_clip = lambda _clip: np.arange(20, 24)

        result = [runner.push_frame(index) for index in range(5)][-1]

        np.testing.assert_array_equal(result, [20, 21, 22, 23])


if __name__ == "__main__":
    unittest.main()
