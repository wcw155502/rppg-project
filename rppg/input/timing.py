from collections import deque

import numpy as np

from .types import FramePacket, TimingStats


class TimingMonitor:
    def __init__(self, window_size=300):
        self.timestamps = deque(maxlen=window_size)

    def update(self, timestamp_ns):
        self.timestamps.append(int(timestamp_ns))
        if len(self.timestamps) < 2:
            return TimingStats(sample_count=len(self.timestamps))
        intervals_ms = np.diff(np.asarray(self.timestamps, dtype=np.float64)) / 1e6
        duration_s = (self.timestamps[-1] - self.timestamps[0]) / 1e9
        fps = (len(self.timestamps) - 1) / duration_s if duration_s > 0 else 0.0
        return TimingStats(
            fps=float(fps),
            interval_mean_ms=float(intervals_ms.mean()),
            interval_std_ms=float(intervals_ms.std()),
            interval_p95_ms=float(np.percentile(intervals_ms, 95)),
            sample_count=len(self.timestamps),
        )

    def reset(self):
        self.timestamps.clear()


class FrameResampler:
    """将带时间戳的视频帧线性重采样到固定帧率。

    时间断点和采集队列丢帧不会被跨越插值，而是显式要求下游重置。
    """

    def __init__(self, target_fps=30.0, max_gap_ms=150.0, max_outputs_per_push=None):
        self.period_ns = int(round(1e9 / target_fps))
        self.max_gap_ns = int(max_gap_ms * 1e6)
        self.max_outputs_per_push = max_outputs_per_push
        self.previous = None
        self.next_timestamp_ns = None

    def push(self, packet):
        if self.previous is None:
            self.previous = packet
            self.next_timestamp_ns = packet.timestamp_ns + self.period_ns
            return [packet], False, None

        gap_ns = packet.timestamp_ns - self.previous.timestamp_ns
        if gap_ns <= 0:
            return [], False, None
        # 队列为了保持低延迟可能丢弃少量旧帧。只要真实时间间隔仍在允许范围内，
        # 就按时间戳重采样补齐，不应仅因 dropped_frames > 0 反复清空生理信号。
        # 只有真实时间断点过长时才禁止跨段插值并要求下游重置。
        if gap_ns > self.max_gap_ns:
            reason = "timestamp_gap"
            self.previous = packet
            self.next_timestamp_ns = packet.timestamp_ns + self.period_ns
            return [packet], True, reason

        output = []
        if self.max_outputs_per_push == 1 and self.next_timestamp_ns <= packet.timestamp_ns:
            # 实时视频不能在落后时把所有中间图像逐张补跑检测/对齐，否则会形成
            # “越补越慢”的正反馈。仅处理最新的目标时刻，跳过过期的中间图像。
            steps = (packet.timestamp_ns - self.next_timestamp_ns) // self.period_ns
            target_ns = self.next_timestamp_ns + steps * self.period_ns
            # 对整张HD图做浮点像素混合会消耗约几十毫秒，并可能制造非真实肤色。
            # 实时模式采用最近的真实采集帧，仅将其时间轴量化到固定采样时刻。
            frame = packet.frame
            output.append(FramePacket(
                frame=frame,
                timestamp_ns=target_ns,
                frame_id=packet.frame_id,
                source=f"{packet.source}:resampled_latest",
            ))
            self.next_timestamp_ns = target_ns + self.period_ns
            self.previous = packet
            return output, False, None

        while self.next_timestamp_ns <= packet.timestamp_ns:
            alpha = (self.next_timestamp_ns - self.previous.timestamp_ns) / gap_ns
            alpha = float(np.clip(alpha, 0.0, 1.0))
            if alpha <= 0:
                frame = self.previous.frame.copy()
            elif alpha >= 1:
                frame = packet.frame.copy()
            else:
                frame = np.clip(
                    self.previous.frame.astype(np.float32) * (1.0 - alpha)
                    + packet.frame.astype(np.float32) * alpha,
                    0,
                    255,
                ).astype(np.uint8)
            output.append(FramePacket(
                frame=frame,
                timestamp_ns=self.next_timestamp_ns,
                frame_id=packet.frame_id,
                source=f"{packet.source}:resampled",
            ))
            self.next_timestamp_ns += self.period_ns
        self.previous = packet
        return output, False, None

    def reset(self):
        self.previous = None
        self.next_timestamp_ns = None
