"""
时序平滑模块，避免展示层心率数值跳变。
提供两种实现，通过配置切换，方便对比效果。
"""
from collections import deque


class KalmanHRSmoother:
    def __init__(self, process_var: float = 0.5, meas_var: float = 4.0):
        self.x = None
        self.p = 1.0
        self.q = process_var
        self.r = meas_var

    def update(self, measurement: float) -> float:
        if measurement is None:
            return self.x
        if self.x is None:
            self.x = measurement
            return self.x
        p_pred = self.p + self.q
        k = p_pred / (p_pred + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * p_pred
        return self.x

    def reset(self):
        self.x = None
        self.p = 1.0


class MovingAverageSmoother:
    def __init__(self, window_size: int = 5):
        self.buffer = deque(maxlen=window_size)

    def update(self, measurement: float) -> float:
        if measurement is None:
            return self.buffer[-1] if self.buffer else None
        self.buffer.append(measurement)
        return sum(self.buffer) / len(self.buffer)

    def reset(self):
        self.buffer.clear()


def build_smoother(cfg: dict):
    smoother_type = cfg.get("type", "kalman")
    if smoother_type == "kalman":
        return KalmanHRSmoother(cfg.get("process_var", 0.5), cfg.get("meas_var", 4.0))
    elif smoother_type == "moving_average":
        return MovingAverageSmoother(cfg.get("window_size", 5))
    else:
        raise ValueError(f"未知的smoother类型: {smoother_type}")
