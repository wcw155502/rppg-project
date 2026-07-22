"""从可信 ROI 序列估计 rPPG 与心率的深模块。

人脸检测、对齐、摄像头和界面均位于模块外。调用者只通过
``consume(MeasurementEvent)`` 发送有效观测或中断事件。
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

import numpy as np

from rppg.measurement.fusion import MultiSourceFusion, extract_skin_rgb_mean, pos_algorithm
from rppg.measurement.signal_processing import (
    bandpass_filter,
    detrend_signal,
    estimate_hr_fft,
    reconstruct_bvp,
    signal_quality_index,
)


@dataclass(frozen=True)
class MeasurementEvent:
    """RPPGMeasurement 的唯一输入 interface。"""

    timestamp_ns: int
    rois: Mapping[str, np.ndarray] = field(default_factory=dict)
    usable: bool = True
    reset_reason: Optional[str] = None
    input_status: str = "measuring"

    @classmethod
    def reset(cls, timestamp_ns: int, reason: str, input_status: str = "reset"):
        return cls(timestamp_ns, usable=False, reset_reason=reason, input_status=input_status)

    @classmethod
    def interrupt(cls, timestamp_ns: int, reason: str, input_status: str):
        return cls(timestamp_ns, usable=False, reset_reason=reason, input_status=input_status)


@dataclass(frozen=True)
class RPPGMeasurementResult:
    """调用者可安全消费的轻量测量快照。"""

    state: str
    updated: bool
    heart_rate: Optional[float]
    raw_heart_rate: Optional[float]
    reason: Optional[str] = None
    candidates: Optional[List[dict]] = None

    def as_dict(self):
        detail = None
        if self.candidates is not None:
            detail = {"candidates": self.candidates, "final_hr": self.raw_heart_rate}
        return {
            "hr": self.heart_rate,
            "raw_hr": self.raw_heart_rate,
            "detail": detail,
            "measurement_state": self.state,
            "updated": self.updated,
            "reason": self.reason,
        }


class RPPGMeasurement:
    """维护 rPPG 窗口、候选源、融合和平滑的深模块。"""

    def __init__(
        self,
        *,
        runner,
        fs: int,
        low_hz: float,
        high_hz: float,
        filter_order: int,
        bvp_window_sec: float,
        update_interval_sec: float,
        rois_used,
        use_pos: bool,
        min_skin_ratio: float,
        min_valid_frame_ratio: float,
        max_quality_gap_sec: float,
        fusion: MultiSourceFusion,
        smoother,
    ):
        self.runner = runner
        self.fs = fs
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.filter_order = filter_order
        self.rois_used = tuple(rois_used)
        self.use_pos = use_pos
        self.min_skin_ratio = min_skin_ratio
        self.min_valid_frame_ratio = min_valid_frame_ratio
        self.max_quality_gap_ns = int(max_quality_gap_sec * 1e9)
        self.fusion = fusion
        self.smoother = smoother
        self.update_interval_frames = max(1, int(update_interval_sec * fs))
        # 统一使用完整的滑动分析窗：首次必须积满该窗口，之后每次更新复用旧样本。
        self.analysis_window_frames = int(fs * bvp_window_sec)
        self.pos_window_frames = self.analysis_window_frames
        self._frames_since_update = 0
        self._last_usable_timestamp_ns = None
        self._last_result = RPPGMeasurementResult("waiting", False, None, None)
        self._init_buffers(bvp_window_sec)

    def _init_buffers(self, bvp_window_sec):
        self.rgb_buffers = {roi_name: [] for roi_name in self.rois_used}
        self.roi_quality_buffers = {roi_name: [] for roi_name in self.rois_used}
        self.diff_signal_buffer = deque(maxlen=int(self.fs * bvp_window_sec))

    def consume(self, event: MeasurementEvent) -> RPPGMeasurementResult:
        if event.reset_reason and event.reset_reason not in {"low_quality", "warming_up"}:
            self.reset()
            return self._remember(RPPGMeasurementResult("reset", False, None, None, event.reset_reason))
        if not event.usable:
            return self._handle_interrupt(event)

        self._last_usable_timestamp_ns = event.timestamp_ns
        self._ingest(event.rois)
        self._frames_since_update += 1
        if self._frames_since_update < self.update_interval_frames:
            return self._snapshot("waiting")
        self._frames_since_update = 0
        return self._evaluate()

    def _handle_interrupt(self, event):
        if self._last_usable_timestamp_ns is None:
            return self._remember(RPPGMeasurementResult("waiting", False, None, None, event.reset_reason))
        elapsed = event.timestamp_ns - self._last_usable_timestamp_ns
        if elapsed >= self.max_quality_gap_ns:
            self.reset()
            return self._remember(RPPGMeasurementResult("reset", False, None, None, "quality_gap"))
        return self._snapshot("held", event.reset_reason)

    def _ingest(self, rois):
        if self.runner is not None and "model_full" in rois:
            diff_out = self.runner.push_frame(rois["model_full"])
            if diff_out is not None:
                self.diff_signal_buffer.extend(diff_out)
        if not self.use_pos:
            return
        for roi_name in self.rois_used:
            rgb = None
            quality = {"skin_ratio": 0.0, "visible": False}
            if roi_name in rois:
                rgb, quality = extract_skin_rgb_mean(rois[roi_name], self.min_skin_ratio)
            sample = rgb if rgb is not None else np.full(3, np.nan, dtype=np.float64)
            self.rgb_buffers[roi_name].append(sample)
            self.roi_quality_buffers[roi_name].append(quality["skin_ratio"])
            if len(self.rgb_buffers[roi_name]) > self.pos_window_frames:
                self.rgb_buffers[roi_name] = self.rgb_buffers[roi_name][-self.pos_window_frames:]
                self.roi_quality_buffers[roi_name] = self.roi_quality_buffers[roi_name][-self.pos_window_frames:]

    def _evaluate(self):
        candidates = []
        if self.runner is not None and len(self.diff_signal_buffer) >= self.analysis_window_frames:
            candidate = self._evaluate_diff(np.asarray(self.diff_signal_buffer), "efficientphys_full")
            if candidate is not None:
                candidates.append(candidate)
        if self.use_pos:
            for roi_name in self.rois_used:
                candidate = self._evaluate_pos_roi(roi_name)
                if candidate is not None:
                    candidates.append(candidate)
        if not candidates:
            return self._snapshot("no_candidates")
        final_hr, _ = self.fusion.fuse(candidates)
        if final_hr is None:
            return self._remember(RPPGMeasurementResult("untrusted", False, None, None, candidates=candidates))
        smoothed_hr = self.smoother.update(final_hr)
        return self._remember(RPPGMeasurementResult("estimated", True, smoothed_hr, final_hr, candidates=candidates))

    def _evaluate_pos_roi(self, roi_name):
        if len(self.rgb_buffers[roi_name]) < self.analysis_window_frames:
            return None
        rgb_seq = np.asarray(self.rgb_buffers[roi_name])
        valid_rows = np.all(np.isfinite(rgb_seq), axis=1)
        valid_ratio = float(valid_rows.mean())
        if valid_ratio < self.min_valid_frame_ratio:
            return None
        rgb_seq = self._interpolate_missing_rgb(rgb_seq, valid_rows)
        candidate = self._evaluate_signal(pos_algorithm(rgb_seq, fs=self.fs), f"pos_{roi_name}")
        if candidate is not None:
            candidate["visibility"] = valid_ratio
            candidate["skin_ratio"] = float(np.mean(self.roi_quality_buffers[roi_name]))
            candidate["sqi"] *= valid_ratio
        return candidate

    def _evaluate_diff(self, diff_signal, source):
        return self._evaluate_signal(reconstruct_bvp(diff_signal), source)

    def _evaluate_signal(self, signal, source):
        try:
            detrended = detrend_signal(signal, self.fs)
            sqi = signal_quality_index(detrended, self.fs, self.low_hz, self.high_hz)
            filtered = bandpass_filter(detrended, self.fs, self.low_hz, self.high_hz, self.filter_order)
            return {"source": source, "hr": estimate_hr_fft(filtered, self.fs, self.low_hz, self.high_hz), "sqi": sqi}
        except Exception:
            return None

    @staticmethod
    def _interpolate_missing_rgb(rgb_seq, valid_rows):
        result = rgb_seq.copy()
        indices = np.arange(len(result))
        valid_indices = indices[valid_rows]
        for channel in range(result.shape[1]):
            result[:, channel] = np.interp(indices, valid_indices, result[valid_rows, channel])
        return result

    def _snapshot(self, state, reason=None):
        previous = self._last_result
        return RPPGMeasurementResult(state, False, previous.heart_rate, previous.raw_heart_rate, reason)

    def _remember(self, result):
        self._last_result = result
        return result

    def reset(self):
        if self.runner is not None:
            self.runner.reset()
        self.smoother.reset()
        self._frames_since_update = 0
        self._last_usable_timestamp_ns = None
        self._last_result = RPPGMeasurementResult("waiting", False, None, None)
        self._init_buffers(self.pos_window_frames / self.fs)
