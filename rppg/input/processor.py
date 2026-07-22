import numpy as np

from .config import build_input_config
from .face_alignment import FaceAligner
from .face_session import FaceSession
from .face_tracker import LandmarkFaceTracker
from .quality_gate import InputQualityGate
from .timing import FrameResampler, TimingMonitor
from .types import InputStatus, ProcessedInput


class TrustedInputPipeline:
    """把原始带时间戳视频帧转换成可信、对齐、等间隔的人脸输入。"""

    def __init__(self, app_cfg, detector=None):
        self.app_cfg = app_cfg
        self.cfg = build_input_config(app_cfg)
        fd_cfg = app_cfg["face_detector"]
        if detector is None:
            from rppg.input.face_detector import SCRFDDetector
            detector = SCRFDDetector(
                onnx_path=fd_cfg["onnx_path"],
                input_size=tuple(fd_cfg["input_size"]),
                det_thresh=fd_cfg["det_thresh"],
                nms_thresh=fd_cfg["nms_thresh"],
                provider=fd_cfg.get("provider", "auto"),
            )
        self.detector = detector
        face_cfg = self.cfg["face"]
        self.aligner = FaceAligner(face_cfg["aligned_size"])
        self.face_tracker = LandmarkFaceTracker(
            self.detector,
            detect_every_n_frames=face_cfg.get("detect_every_n_frames", 5),
            max_flow_error=face_cfg.get("max_flow_error", 20.0),
        )
        self.session = FaceSession(face_cfg)
        self.quality_gate = InputQualityGate(self.cfg["quality"])
        self.resampler = FrameResampler(
            self.cfg["target_fps"],
            self.cfg["max_time_gap_ms"],
            max_outputs_per_push=1,
        )
        self.timing = TimingMonitor(window_size=max(30, int(self.cfg["target_fps"] * 10)))

    def process(self, packet):
        packets, reset_required, reset_reason = self.resampler.push(packet)
        output = []
        if reset_required:
            self._reset_continuity()
        for fixed_packet in packets:
            item = self._process_resampled(fixed_packet)
            if reset_required:
                item.reset_required = True
                item.reset_reason = reset_reason
                item.status = InputStatus.TIME_GAP
                reset_required = False
            output.append(item)
        return output

    def _process_resampled(self, packet):
        timing = self.timing.update(packet.timestamp_ns)
        bbox, landmarks, _ = self.face_tracker.update(packet.frame)
        if bbox is None or landmarks is None:
            status, reset_required, reason = self.session.update_missing(packet.timestamp_ns)
            return ProcessedInput(
                timestamp_ns=packet.timestamp_ns,
                frame_id=packet.frame_id,
                status=status,
                frame=packet.frame,
                timing=timing,
                reset_required=reset_required,
                reset_reason=reason,
                track_id=self.session.track_id,
            )

        aligned, alignment = self.aligner.align(packet.frame, landmarks)
        if aligned is None:
            status, reset_required, reason = self.session.update_missing(packet.timestamp_ns)
            return ProcessedInput(
                timestamp_ns=packet.timestamp_ns,
                frame_id=packet.frame_id,
                status=status,
                frame=packet.frame,
                bbox=bbox,
                landmarks=landmarks,
                timing=timing,
                reset_required=reset_required,
                reset_reason=reason or "alignment_failed",
                track_id=self.session.track_id,
            )

        quality = self.quality_gate.evaluate(aligned)
        status, reset_required, reason = self.session.update_face(
            packet.timestamp_ns, bbox, quality.valid
        )
        rois, boxes, yaw = self.aligner.extract_rois(
            aligned,
            alignment["projected_landmarks"],
            1.0,
        )
        model_input, _ = self.aligner.align_efficientphys_context(
            packet.frame,
            landmarks,
            self.cfg["face"].get("efficientphys_context_scale", 1.0),
        )
        if model_input is None:
            model_input = aligned
        # 只替换 EfficientPhys 输入；POS 的 forehead/cheek ROI 均来自标准对齐画布。
        rois["model_full"] = model_input
        size = model_input.shape[0]
        boxes["model_full"] = (0, 0, size, size)
        max_abs_yaw = float(self.cfg["face"].get("max_abs_yaw", 0.45))
        if abs(yaw) > max_abs_yaw:
            quality.reasons = tuple(quality.reasons) + ("head_turned",)
            quality.valid = False
            status = InputStatus.LOW_QUALITY
        return ProcessedInput(
            timestamp_ns=packet.timestamp_ns,
            frame_id=packet.frame_id,
            status=status,
            frame=packet.frame,
            aligned_face=aligned,
            model_input_face=model_input,
            rois=rois,
            roi_boxes=boxes,
            bbox=bbox,
            landmarks=landmarks,
            quality=quality,
            timing=timing,
            reset_required=reset_required,
            reset_reason=reason,
            track_id=self.session.track_id,
        )

    def _reset_continuity(self):
        self.session.reset()
        self.quality_gate.reset()
        self.timing.reset()
        self.face_tracker.reset()

    def reset(self):
        self._reset_continuity()
        self.resampler.reset()
