"""固定视频离线回放入口，用于确定性复现输入链路和心率结果。"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from input_pipeline.config import build_input_config
from input_pipeline.diagnostics import DiagnosticsRecorder
from input_pipeline.processor import TrustedInputPipeline
from input_pipeline.replay import VideoReplay
from pipeline.rppg_pipeline import RPPGPipeline
from utils.config_loader import load_config, setup_logging
from scripts.run_camera_demo import draw_aligned_rois, overlay_aligned_face


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--config", default="configs/inference_config.yaml")
    parser.add_argument("--no_display", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["runtime"]["log_level"])
    heart = RPPGPipeline(cfg)
    inputs = TrustedInputPipeline(cfg, detector=heart.detector)
    input_cfg = build_input_config(cfg)
    recorder = DiagnosticsRecorder(
        input_cfg["diagnostics"], input_cfg["target_fps"], config_snapshot=cfg
    )
    try:
        with VideoReplay(args.video, input_cfg["target_fps"]) as replay:
            for packet in replay:
                for item in inputs.process(packet):
                    recorder.record(item)
                    result = heart.process_input(item)
                    recorder.record_result(item, result)
                    if not args.no_display:
                        frame = item.frame.copy()
                        text = item.status.value
                        if result.get("hr") is not None:
                            text += f" HR={result['hr']:.1f}"
                        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if input_cfg["face"].get("show_aligned_face", False) and item.aligned_face is not None:
                            frame = overlay_aligned_face(
                                frame,
                                draw_aligned_rois(item.aligned_face, item.roi_boxes),
                            )
                        cv2.imshow("rPPG Video Replay", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            return
    finally:
        recorder.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
