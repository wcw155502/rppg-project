"""将旧/新输入 adapter 接入统一 RPPGMeasurement 的薄管线。"""
import time

from legacy.roi_tracker import ROITracker
from rppg.input.face_detector import SCRFDDetector
from rppg.input.types import InputStatus
from rppg.measurement.efficientphys import EfficientPhysRunner
from rppg.measurement.config import build_measurement_config
from rppg.measurement.fusion import MultiSourceFusion
from rppg.measurement.rppg_measurement import MeasurementEvent, RPPGMeasurement
from rppg.measurement.smoother import build_smoother


class RPPGPipeline:
    """兼容 adapter：新代码使用 process_input，旧代码可暂用 process_frame。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        fd_cfg = cfg["face_detector"]
        measurement_cfg = build_measurement_config(cfg)
        model_cfg = measurement_cfg["model"]
        sig_cfg = measurement_cfg["signal_processing"]
        fus_cfg = measurement_cfg["fusion"]
        self.detector = SCRFDDetector(
            onnx_path=fd_cfg["onnx_path"], input_size=tuple(fd_cfg["input_size"]),
            det_thresh=fd_cfg["det_thresh"], nms_thresh=fd_cfg["nms_thresh"],
            provider=fd_cfg.get("provider", "auto"),
        )
        # Deprecated compatibility adapter. New input uses input_pipeline/ instead.
        self.roi_tracker = ROITracker(
            detector=self.detector, detect_every_n_frames=fd_cfg["detect_every_n_frames"],
            smooth_alpha=fd_cfg["bbox_smooth_alpha"], model_bbox_scale=model_cfg.get("bbox_scale", 1.2),
        )
        self.use_efficientphys = bool(model_cfg.get("enabled", True))
        runner = None
        if self.use_efficientphys:
            runner = EfficientPhysRunner(
                model_py_path="models/EfficientPhys.py", weight_path=model_cfg["weight_path"],
                img_size=model_cfg["img_size"], frame_depth=model_cfg["frame_depth"],
                chunk_length=model_cfg["chunk_length"], stride=model_cfg["stride"], device=model_cfg["device"],
            )
        self.measurement = RPPGMeasurement(
            runner=runner, fs=sig_cfg["fs"], low_hz=sig_cfg["bandpass_low_hz"],
            high_hz=sig_cfg["bandpass_high_hz"], filter_order=sig_cfg["filter_order"],
            bvp_window_sec=sig_cfg.get("bvp_window_sec", 10),
            update_interval_sec=sig_cfg.get("update_interval_sec", 2), rois_used=fus_cfg["rois"],
            use_pos=fus_cfg["use_traditional_fallback"], min_skin_ratio=fus_cfg.get("min_skin_ratio", 0.25),
            min_valid_frame_ratio=fus_cfg.get("min_valid_frame_ratio", 0.7),
            max_quality_gap_sec=sig_cfg.get("max_quality_gap_sec", 0.5),
            fusion=MultiSourceFusion(sqi_threshold=sig_cfg["sqi_threshold"], min_valid_sources=fus_cfg["min_valid_rois"]),
            smoother=build_smoother(measurement_cfg["smoother"]),
        )

    def process_input(self, input_item):
        timestamp_ns = input_item.timestamp_ns
        if input_item.reset_required:
            self.measurement.consume(MeasurementEvent.reset(timestamp_ns, input_item.reset_reason or "input_reset"))
        if input_item.status == InputStatus.MEASURING:
            result = self.measurement.consume(MeasurementEvent(timestamp_ns, input_item.rois, input_status="measuring"))
        else:
            result = self.measurement.consume(MeasurementEvent.interrupt(timestamp_ns, input_item.status.value, input_item.status.value))
        return {"face_detected": input_item.status != InputStatus.NO_FACE, "input_status": input_item.status.value, **result.as_dict()}

    def process_frame(self, frame_bgr):
        """Deprecated compatibility adapter for core/roi_tracker.py callers."""
        timestamp_ns = time.monotonic_ns()
        rois = self.roi_tracker.update(frame_bgr)
        if rois is None:
            result = self.measurement.consume(MeasurementEvent.interrupt(timestamp_ns, "no_face", "no_face"))
            return {"face_detected": False, "input_status": "legacy_no_face", **result.as_dict()}
        result = self.measurement.consume(MeasurementEvent(timestamp_ns, rois, input_status="legacy"))
        return {"face_detected": True, "input_status": "legacy", **result.as_dict()}

    def reset(self):
        self.measurement.reset()
