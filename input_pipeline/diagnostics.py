import csv
import json
from datetime import datetime
from pathlib import Path

import cv2


class DiagnosticsRecorder:
    def __init__(self, cfg, target_fps, config_snapshot=None):
        self.enabled = bool(cfg.get("enabled", False))
        self.cfg = cfg
        self.target_fps = target_fps
        self.session_dir = None
        self.csv_file = None
        self.csv_writer = None
        self.source_writer = None
        self.aligned_writer = None
        self.results_file = None
        if not self.enabled:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session_dir = Path(cfg["output_dir"]) / stamp
        self.session_dir.mkdir(parents=True, exist_ok=False)
        with (self.session_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump({"target_fps": target_fps, "config": config_snapshot or {}}, f, ensure_ascii=False, indent=2)
        self.csv_file = (self.session_dir / "frames.csv").open("w", newline="", encoding="utf-8")
        fields = [
            "frame_id", "timestamp_ns", "status", "track_id", "reset_reason", "fps",
            "brightness", "overexposed_ratio", "underexposed_ratio", "sharpness",
            "motion", "skin_ratio", "quality_reasons", "bbox", "landmarks",
        ]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fields)
        self.csv_writer.writeheader()
        self.results_file = (self.session_dir / "results.jsonl").open("w", encoding="utf-8")

    def record(self, item):
        if not self.enabled:
            return
        quality = item.quality
        timing = item.timing
        self.csv_writer.writerow({
            "frame_id": item.frame_id,
            "timestamp_ns": item.timestamp_ns,
            "status": item.status.value,
            "track_id": item.track_id,
            "reset_reason": item.reset_reason or "",
            "fps": timing.fps if timing else 0.0,
            "brightness": quality.brightness if quality else 0.0,
            "overexposed_ratio": quality.overexposed_ratio if quality else 0.0,
            "underexposed_ratio": quality.underexposed_ratio if quality else 0.0,
            "sharpness": quality.sharpness if quality else 0.0,
            "motion": quality.motion if quality else 0.0,
            "skin_ratio": quality.skin_ratio if quality else 0.0,
            "quality_reasons": "|".join(quality.reasons) if quality else "",
            "bbox": item.bbox.tolist() if item.bbox is not None else "",
            "landmarks": item.landmarks.tolist() if item.landmarks is not None else "",
        })
        self.csv_file.flush()
        if self.cfg.get("save_source_video"):
            self.source_writer = self._write_video(self.source_writer, self.session_dir / "source.mp4", item.frame)
        if self.cfg.get("save_aligned_face_video") and item.aligned_face is not None:
            self.aligned_writer = self._write_video(
                self.aligned_writer, self.session_dir / "aligned_face.mp4", item.aligned_face
            )

    def record_result(self, item, result):
        """将 HR、各候选源和输入状态关联保存，便于定位哪一路发生跳变。"""
        if not self.enabled or self.results_file is None:
            return
        payload = {
            "frame_id": item.frame_id,
            "timestamp_ns": item.timestamp_ns,
            "input_status": item.status.value,
            "track_id": item.track_id,
            "hr": result.get("hr"),
            "raw_hr": result.get("raw_hr"),
            "detail": result.get("detail"),
        }
        self.results_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.results_file.flush()

    def _write_video(self, writer, path, frame):
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.target_fps, (w, h)
            )
        writer.write(frame)
        return writer

    def close(self):
        for writer in (self.source_writer, self.aligned_writer):
            if writer is not None:
                writer.release()
        if self.csv_file is not None:
            self.csv_file.close()
        if self.results_file is not None:
            self.results_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
