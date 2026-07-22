"""
实时摄像头 rPPG demo 入口。

用法:
    python scripts/run_camera_demo.py --config configs/inference_config.yaml
"""
import argparse
import logging
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config, setup_logging
from rppg.input.camera_capture import AsyncCameraCapture
from rppg.input.config import build_input_config
from rppg.input.diagnostics import DiagnosticsRecorder
from rppg.input.processor import TrustedInputPipeline
from runtime.rppg_pipeline import RPPGPipeline

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/inference_config.yaml")
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument(
        "--legacy_input", action="store_true",
        help="使用旧版同步采集和ROITracker；默认启用新的可信输入链路",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["runtime"]["log_level"])

    pipeline = RPPGPipeline(cfg)

    if args.legacy_input:
        return run_legacy_demo(args.camera_id, cfg, pipeline)

    return run_trusted_demo(args.camera_id, cfg, pipeline)


def run_trusted_demo(camera_id, cfg, pipeline):
    input_cfg = build_input_config(cfg)
    trusted_input = TrustedInputPipeline(cfg, detector=pipeline.detector)
    recorder = DiagnosticsRecorder(
        input_cfg["diagnostics"], input_cfg["target_fps"], config_snapshot=cfg
    )
    last_hr_display = None
    last_item = None
    last_result = None

    status_text = {
        "no_face": "No Face",
        "warming_up": "Stabilizing",
        "low_quality": "Low Quality",
        "time_gap": "Timing Gap",
        "measuring": "Measuring",
    }
    status_color = {
        "no_face": (0, 0, 255),
        "warming_up": (0, 200, 255),
        "low_quality": (0, 100, 255),
        "time_gap": (0, 0, 255),
        "measuring": (0, 255, 0),
    }

    logger.info("启用可信输入链路：异步采集、时间重采样、五点对齐和质量门控。")
    logger.info(
        "EfficientPhys: %s",
        "enabled" if pipeline.use_efficientphys else "disabled (POS realtime mode)",
    )
    try:
        should_exit = False
        with AsyncCameraCapture(camera_id, input_cfg) as capture:
            while not should_exit:
                packet = capture.read(timeout=2.0)
                for item in trusted_input.process(packet):
                    last_item = item
                    recorder.record(item)
                    last_result = pipeline.process_input(item)
                    recorder.record_result(item, last_result)
                    if last_result.get("hr") is not None:
                        last_hr_display = last_result["hr"]
                    if item.status.value != "measuring":
                        last_hr_display = None

                if last_item is None:
                    continue
                display_frame = last_item.frame.copy()
                if last_item.bbox is not None:
                    x1, y1, x2, y2 = last_item.bbox.astype(int)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                if last_item.landmarks is not None:
                    for x, y in last_item.landmarks.astype(int):
                        cv2.circle(display_frame, (x, y), 2, (0, 255, 255), -1)

                status = last_item.status.value
                label = status_text.get(status, status)
                if status == "measuring" and last_hr_display is not None:
                    label = f"HR: {last_hr_display:.1f} bpm"
                cv2.putText(
                    display_frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, status_color.get(status, (255, 255, 255)), 2,
                )
                timing = last_item.timing
                if timing is not None:
                    cv2.putText(
                        display_frame,
                        f"Input FPS: {timing.fps:.2f}  jitter: {timing.interval_std_ms:.2f}ms",
                        (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    )
                cv2.putText(
                    display_frame,
                    "Model: EfficientPhys" if pipeline.use_efficientphys else "Model: POS realtime",
                    (20, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )
                if last_item.quality is not None and not last_item.quality.valid:
                    cv2.putText(
                        display_frame, "Quality: " + ",".join(last_item.quality.reasons),
                        (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1,
                    )
                if input_cfg["face"].get("show_aligned_face", False) and last_item.aligned_face is not None:
                    aligned_display = draw_aligned_rois(
                        last_item.aligned_face,
                        last_item.roi_boxes,
                        last_item.model_input_face,
                    )
                    display_frame = overlay_aligned_face(display_frame, aligned_display)
                cv2.imshow("rPPG Trusted Input Demo", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    should_exit = True
    finally:
        recorder.close()
        cv2.destroyAllWindows()


def run_legacy_demo(camera_id, cfg, pipeline):
    """修改前的同步采集/旧ROI路径，保留用于兼容和结果对照。"""

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 id={camera_id}")

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    configured_fps = cfg["signal_processing"]["fs"]
    if actual_fps and abs(actual_fps - configured_fps) > 2:
        logger.warning(
            f"摄像头实际帧率({actual_fps:.1f}fps)与配置fs({configured_fps}fps)不一致，"
            f"会导致心率估计系统性偏差，请修正configs/inference_config.yaml中的fs字段。"
        )

    logger.info("开始旧版实时采集，按 q 退出。")
    last_hr_display = None

    roi_colors = {
        "full": (255, 255, 0),      # 青色：整体人脸框
        "forehead": (0, 255, 0),     # 绿色：额头
        "cheek_l": (0, 165, 255),    # 橙色：左颊
        "cheek_r": (0, 165, 255),    # 橙色：右颊
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("读取摄像头帧失败，跳过。")
                continue

            result = pipeline.process_frame(frame)
            face_detected = result is not None and result.get("face_detected", False)
            if not face_detected:
                last_hr_display = None
            elif result["hr"] is not None:
                last_hr_display = result["hr"]

            # 每次有新的融合结果时，把各候选源的原始输出都打出来，方便排查是哪一路不准
            if face_detected and result.get("detail") and result["detail"].get("candidates"):
                cand_str = " | ".join(
                    f"{c['source']}: {c['hr']:.1f}bpm(sqi={c['sqi']:.2f})" for c in result["detail"]["candidates"]
                )
                logger.info(f"[候选源] {cand_str} => 融合后: {result['raw_hr']}")

            display_frame = frame.copy()

            if face_detected:
            # 只有当前帧存在人脸时才画 ROI，避免残留上一帧检测框。
                for roi_name, (x1, y1, x2, y2) in pipeline.roi_tracker.get_last_regions().items():
                    if roi_name == "model_full":
                        continue
                    color = roi_colors.get(roi_name, (200, 200, 200))
                    thickness = 1 if roi_name == "full" else 2
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                    if roi_name != "full":
                        cv2.putText(display_frame, roi_name, (x1, max(0, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                text = f"HR: {last_hr_display:.1f} bpm" if last_hr_display else "HR: --"
                text_color = (0, 255, 0)
            else:
                # OpenCV 内置字体不支持中文，因此画面使用等义的英文提示。
                text = "No Face"
                text_color = (0, 0, 255)
            cv2.putText(display_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)

            # 把各候选源的原始HR/SQI也叠加显示在画面上，方便对比是哪一路在拖后腿
            if face_detected and result.get("detail") and result["detail"].get("candidates"):
                for i, c in enumerate(result["detail"]["candidates"]):
                    line = f"{c['source']}: {c['hr']:.1f}bpm sqi={c['sqi']:.2f}"
                    cv2.putText(display_frame, line, (20, 70 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            cv2.imshow("rPPG Industrial Demo", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def draw_aligned_rois(aligned_face, roi_boxes, model_input_face=None):
    """在主画面嵌入 POS 标准画布与 EfficientPhys 扩展视野。"""
    display = aligned_face.copy()
    styles = {
        "forehead": ((0, 255, 0), 1, "POS forehead"),
        "cheek_l": ((0, 165, 255), 1, "POS cheek L"),
        "cheek_r": ((0, 165, 255), 1, "POS cheek R"),
    }
    for name, (color, thickness, label) in styles.items():
        if name not in roi_boxes:
            continue
        region = roi_boxes[name]
        if isinstance(region, np.ndarray):
            polygon = region.astype(np.int32)
            cv2.polylines(display, [polygon], True, color, thickness)
            x1, y1 = polygon[:, 0].min(), polygon[:, 1].min()
        else:
            x1, y1, x2, y2 = region
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        text_y = min(display.shape[0] - 3, max(10, int(y1) + 11))
        cv2.putText(
            display, label, (int(x1) + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1,
        )
    cv2.putText(display, "POS input", (4, display.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
    if model_input_face is None:
        return display

    model_display = model_input_face.copy()
    height, width = model_display.shape[:2]
    cv2.rectangle(model_display, (0, 0), (width - 1, height - 1), (255, 255, 0), 2)
    cv2.putText(
        model_display, "EfficientPhys context x1.3", (4, height - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.27, (255, 255, 0), 1,
    )
    return np.hstack([display, model_display])


def overlay_aligned_face(main_frame, aligned_display, max_size=288, margin=12):
    """把对齐人脸及算法ROI嵌入主画面右上角。"""
    output = main_frame.copy()
    frame_h, frame_w = output.shape[:2]
    max_height = min(max_size, frame_h - 2 * margin)
    max_width = min(max_size * 2, frame_w - 2 * margin)
    scale = min(max_height / aligned_display.shape[0], max_width / aligned_display.shape[1])
    if scale * min(aligned_display.shape[:2]) < 48:
        return output
    inset = cv2.resize(
        aligned_display,
        (round(aligned_display.shape[1] * scale), round(aligned_display.shape[0] * scale)),
        interpolation=cv2.INTER_NEAREST,
    )
    inset_h, inset_w = inset.shape[:2]
    x1 = frame_w - margin - inset_w
    y1 = margin
    x2, y2 = x1 + inset_w, y1 + inset_h
    output[y1:y2, x1:x2] = inset
    cv2.rectangle(output, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)
    cv2.putText(
        output, "POS / EfficientPhys inputs", (x1, min(frame_h - 4, y2 + 18)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )
    return output


if __name__ == "__main__":
    main()
