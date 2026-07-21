"""
rPPG总调度类：串联 人脸检测 -> ROI跟踪 -> EfficientPhys推理 -> POS兜底 -> 融合 -> 平滑。

设计原则：
- 每个环节都是独立可替换的组件（构造函数注入），方便后续换检测器/换模型/换融合策略时
  不用改动主流程代码。
- 任何一环出现异常都不应该让整个进程崩溃，工业场景要求"降级"而不是"报错退出"。
- 数据缓冲(每帧都做) 和 HR结果更新(节流，几秒一次) 是两件独立的事，
  不能耦合在一起，否则会出现"每帧都重新做一次FFT估计"导致的数值抖动。
"""
import logging
from collections import deque

import numpy as np

from core.face_detector import SCRFDDetector
from core.roi_tracker import ROITracker
from core.rppg_infer import EfficientPhysRunner
from core.signal_processing import (
    bandpass_filter,
    detrend_signal,
    estimate_hr_fft,
    reconstruct_bvp,
    signal_quality_index,
)
from core.fusion import extract_skin_rgb_mean, pos_algorithm, MultiSourceFusion
from core.smoother import build_smoother
from input_pipeline.types import InputStatus

logger = logging.getLogger(__name__)


class RPPGPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        fd_cfg = cfg["face_detector"]
        model_cfg = cfg["rppg_model"]
        sig_cfg = cfg["signal_processing"]
        fus_cfg = cfg["fusion"]

        self.detector = SCRFDDetector(
            onnx_path=fd_cfg["onnx_path"],
            input_size=tuple(fd_cfg["input_size"]),
            det_thresh=fd_cfg["det_thresh"],
            nms_thresh=fd_cfg["nms_thresh"],
            provider=fd_cfg.get("provider", "auto"),
        )
        self.roi_tracker = ROITracker(
            detector=self.detector,
            detect_every_n_frames=fd_cfg["detect_every_n_frames"],
            smooth_alpha=fd_cfg["bbox_smooth_alpha"],
            model_bbox_scale=model_cfg.get("bbox_scale", 1.2),
        )
        self.rois_used = fus_cfg["rois"]  # 这些子区域(forehead/cheek_l/cheek_r)只给POS传统算法做交叉验证用
        # EfficientPhys 只吃整张人脸框(full)，不能喂forehead/cheek这类子区域裁剪——
        # 训练时用的就是整脸resize到72x72，子区域crop和训练分布不一致，会显著拖累精度。
        # 只需要一份模型实例即可，不用像POS那样按ROI各建一份。
        self.use_efficientphys = bool(model_cfg.get("enabled", True))
        self.efficientphys_runner = None
        if self.use_efficientphys:
            self.efficientphys_runner = EfficientPhysRunner(
                model_py_path="models/EfficientPhys.py",
                weight_path=model_cfg["weight_path"],
                img_size=model_cfg["img_size"],
                frame_depth=model_cfg["frame_depth"],
                chunk_length=model_cfg["chunk_length"],
                stride=model_cfg["stride"],
                device=model_cfg["device"],
            )
        else:
            logger.warning("EfficientPhys 已按配置禁用；当前使用 POS 路径以保证实时性。")

        self.fs = sig_cfg["fs"]
        self.low_hz = sig_cfg["bandpass_low_hz"]
        self.high_hz = sig_cfg["bandpass_high_hz"]
        self.filter_order = sig_cfg["filter_order"]
        self.sqi_threshold = sig_cfg["sqi_threshold"]

        # HR估计用的滑动窗口长度（秒），窗口越长FFT频率分辨率越好、越稳定，但越滞后
        self.bvp_window_sec = sig_cfg.get("bvp_window_sec", 10)
        # HR结果更新节流间隔（秒），不是每帧都重算，避免数值抖动
        self.update_interval_sec = sig_cfg.get("update_interval_sec", 2)
        self.update_interval_frames = max(1, int(self.update_interval_sec * self.fs))
        self._frames_since_update = 0
        self._cached_result = None

        self.fusion = MultiSourceFusion(
            sqi_threshold=self.sqi_threshold,
            min_valid_sources=fus_cfg["min_valid_rois"],
        )
        self.use_pos_fallback = fus_cfg["use_traditional_fallback"]
        self.rgb_buffers = {roi_name: [] for roi_name in self.rois_used}
        self.roi_quality_buffers = {roi_name: [] for roi_name in self.rois_used}
        self.min_skin_ratio = fus_cfg.get("min_skin_ratio", 0.25)
        self.min_valid_frame_ratio = fus_cfg.get("min_valid_frame_ratio", 0.7)
        self.pos_window_frames = int(self.fs * self.bvp_window_sec)

        # EfficientPhys的rolling buffer只需要一份，对应full区域
        diff_buffer_maxlen = int(self.fs * self.bvp_window_sec)
        self.diff_signal_buffer = deque(maxlen=diff_buffer_maxlen)

        self.smoother = build_smoother(cfg["smoother"])

    def process_frame(self, frame_bgr: np.ndarray):
        """
        旧版兼容入口：内部仍使用 core/roi_tracker.py。
        新版实时入口应调用 process_input()，本函数保留以兼容现有调用方。

        输入单帧原始图像。数据缓冲每帧都做，但HR结果按update_interval_sec节流更新，
        两次更新之间返回上一次的缓存结果，不会每帧都变。
        """
        rois = self.roi_tracker.update(frame_bgr)
        if rois is None:
            return {"face_detected": False, "hr": None, "raw_hr": None, "detail": None}

        # ---- 每帧都做：喂数据进各自的buffer，保持新鲜 ----

        # EfficientPhys 使用单独扩展后的整脸框，默认是检测框的 1.2 倍。
        if self.use_efficientphys and "model_full" in rois:
            diff_out = self._safe_call(self.efficientphys_runner.push_frame, rois["model_full"])
            if diff_out is not None:
                self.diff_signal_buffer.extend(diff_out)

        # POS 传统算法可以继续吃多个子区域做交叉验证，它不依赖训练时的输入分布
        if self.use_pos_fallback:
            for roi_name in self.rois_used:
                rgb = None
                quality = {"skin_ratio": 0.0, "visible": False}
                if roi_name in rois:
                    rgb, quality = extract_skin_rgb_mean(rois[roi_name], self.min_skin_ratio)
                # 无效帧保留 NaN 占位，避免删除样本后破坏真实采样率。
                sample = rgb if rgb is not None else np.full(3, np.nan, dtype=np.float64)
                self.rgb_buffers[roi_name].append(sample)
                self.roi_quality_buffers[roi_name].append(quality["skin_ratio"])
                if len(self.rgb_buffers[roi_name]) > self.pos_window_frames:
                    self.rgb_buffers[roi_name] = self.rgb_buffers[roi_name][-self.pos_window_frames:]
                    self.roi_quality_buffers[roi_name] = self.roi_quality_buffers[roi_name][-self.pos_window_frames:]

        # ---- 节流：没到更新周期就直接返回缓存结果，不重新估计 ----
        self._frames_since_update += 1
        if self._frames_since_update < self.update_interval_frames:
            if self._cached_result is None:
                return {"face_detected": True, "hr": None, "raw_hr": None, "detail": None}
            return self._cached_result
        self._frames_since_update = 0

        candidates = []

        # EfficientPhys 分支：用rolling buffer里累积的完整窗口
        if self.use_efficientphys and len(self.diff_signal_buffer) >= self.fs * 2:
            cand = self._eval_diff_signal(np.array(self.diff_signal_buffer), source="efficientphys_full")
            if cand is not None:
                candidates.append(cand)

        # POS 传统算法分支（兜底/交叉验证），逐个子区域分别估计
        if self.use_pos_fallback:
            for roi_name in self.rois_used:
                if len(self.rgb_buffers[roi_name]) >= self.fs * 3:
                    rgb_seq = np.array(self.rgb_buffers[roi_name])
                    valid_rows = np.all(np.isfinite(rgb_seq), axis=1)
                    valid_ratio = float(valid_rows.mean())
                    if valid_ratio < self.min_valid_frame_ratio:
                        logger.debug(
                            f"[pos_{roi_name}] 遮挡过多，有效帧比例={valid_ratio:.2f}，跳过该ROI"
                        )
                        continue
                    rgb_seq = self._interpolate_missing_rgb(rgb_seq, valid_rows)
                    pos_signal = pos_algorithm(rgb_seq, fs=self.fs)
                    cand = self._eval_filtered_signal(pos_signal, source=f"pos_{roi_name}")
                    if cand is not None:
                        # 遮挡越多，融合权重越低；完全可见时不改变原 SQI。
                        cand["visibility"] = valid_ratio
                        cand["skin_ratio"] = float(np.mean(self.roi_quality_buffers[roi_name]))
                        cand["sqi"] *= valid_ratio
                        candidates.append(cand)

        if not candidates:
            if self._cached_result is None:
                return {"face_detected": True, "hr": None, "raw_hr": None, "detail": None}
            return self._cached_result

        final_hr, detail = self.fusion.fuse(candidates)
        smoothed_hr = self.smoother.update(final_hr) if final_hr is not None else self.smoother.x

        self._cached_result = {
            "face_detected": True,
            "hr": smoothed_hr,
            "raw_hr": final_hr,
            "detail": detail,
        }
        return self._cached_result

    def process_input(self, input_item):
        """消费 input_pipeline 产生的可信、对齐、固定时间轴输入。"""
        if input_item.reset_required:
            self.reset()
        if input_item.status != InputStatus.MEASURING:
            return {
                "face_detected": input_item.status != InputStatus.NO_FACE,
                "input_status": input_item.status.value,
                "hr": None,
                "raw_hr": None,
                "detail": None,
            }
        return self._process_rois(input_item.rois, input_status=input_item.status.value)

    def _process_rois(self, rois, input_status="measuring"):
        """新输入链路专用：ROI 已经过对齐、质量门控和等时间轴重采样。"""
        if self.use_efficientphys and "model_full" in rois:
            diff_out = self._safe_call(self.efficientphys_runner.push_frame, rois["model_full"])
            if diff_out is not None:
                self.diff_signal_buffer.extend(diff_out)

        if self.use_pos_fallback:
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

        self._frames_since_update += 1
        if self._frames_since_update < self.update_interval_frames:
            if self._cached_result is None:
                return {"face_detected": True, "input_status": input_status, "hr": None, "raw_hr": None, "detail": None}
            return {**self._cached_result, "input_status": input_status}
        self._frames_since_update = 0

        candidates = []
        if self.use_efficientphys and len(self.diff_signal_buffer) >= self.fs * 2:
            cand = self._eval_diff_signal(np.array(self.diff_signal_buffer), source="efficientphys_full")
            if cand is not None:
                candidates.append(cand)
        if self.use_pos_fallback:
            for roi_name in self.rois_used:
                if len(self.rgb_buffers[roi_name]) < self.fs * 3:
                    continue
                rgb_seq = np.array(self.rgb_buffers[roi_name])
                valid_rows = np.all(np.isfinite(rgb_seq), axis=1)
                valid_ratio = float(valid_rows.mean())
                if valid_ratio < self.min_valid_frame_ratio:
                    continue
                rgb_seq = self._interpolate_missing_rgb(rgb_seq, valid_rows)
                cand = self._eval_filtered_signal(pos_algorithm(rgb_seq, fs=self.fs), source=f"pos_{roi_name}")
                if cand is not None:
                    cand["visibility"] = valid_ratio
                    cand["skin_ratio"] = float(np.mean(self.roi_quality_buffers[roi_name]))
                    cand["sqi"] *= valid_ratio
                    candidates.append(cand)
        if not candidates:
            if self._cached_result is None:
                return {"face_detected": True, "input_status": input_status, "hr": None, "raw_hr": None, "detail": None}
            return {**self._cached_result, "input_status": input_status}
        final_hr, detail = self.fusion.fuse(candidates)
        smoothed_hr = self.smoother.update(final_hr)
        self._cached_result = {
            "face_detected": True,
            "input_status": input_status,
            "hr": smoothed_hr,
            "raw_hr": final_hr,
            "detail": detail,
        }
        return self._cached_result

    def _eval_diff_signal(self, diff_signal, source):
        try:
            # SQI 必须在带通之前计算；先积分，再由统一评估函数去趋势和滤波。
            bvp = reconstruct_bvp(diff_signal)
            return self._eval_filtered_signal(bvp, source)
        except Exception as e:
            logger.debug(f"[{source}] 信号处理失败: {e}")
            return None

    def _eval_filtered_signal(self, signal, source):
        try:
            bvp_detrended = detrend_signal(signal, self.fs)
            # 先基于未带通信号评估质量，防止滤波让噪声的 SQI 虚高。
            sqi = signal_quality_index(bvp_detrended, self.fs, self.low_hz, self.high_hz)
            bvp_filtered = bandpass_filter(
                bvp_detrended, self.fs, self.low_hz, self.high_hz, self.filter_order
            )
            hr = estimate_hr_fft(bvp_filtered, self.fs, self.low_hz, self.high_hz)
            return {"source": source, "hr": hr, "sqi": sqi}
        except Exception as e:
            logger.debug(f"[{source}] HR估计失败: {e}")
            return None

    @staticmethod
    def _interpolate_missing_rgb(rgb_seq, valid_rows):
        """按时间插值少量被遮挡帧，保持 POS 序列的固定采样间隔。"""
        result = rgb_seq.copy()
        indices = np.arange(len(result))
        valid_indices = indices[valid_rows]
        for channel in range(result.shape[1]):
            result[:, channel] = np.interp(
                indices, valid_indices, result[valid_rows, channel]
            )
        return result

    @staticmethod
    def _safe_call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"调用失败 {fn}: {e}")
            return None

    def reset(self):
        if self.efficientphys_runner is not None:
            self.efficientphys_runner.reset()
        self.rgb_buffers = {roi_name: [] for roi_name in self.rois_used}
        self.roi_quality_buffers = {roi_name: [] for roi_name in self.rois_used}
        self.diff_signal_buffer = deque(maxlen=int(self.fs * self.bvp_window_sec))
        self._frames_since_update = 0
        self._cached_result = None
        self.smoother.reset()
