import logging
import queue
import threading
import time

import cv2

from .camera_control import configure_camera
from .types import FramePacket

logger = logging.getLogger(__name__)


class AsyncCameraCapture:
    """独立线程采集摄像头，避免模型推理阻塞采集时间轴。"""

    def __init__(self, camera_id, cfg):
        self.camera_id = camera_id
        self.cfg = cfg
        self.queue = queue.Queue(maxsize=int(cfg["queue_size"]))
        self.cap = None
        self.thread = None
        self.stop_event = threading.Event()
        self.frame_id = 0
        self.pending_drops = 0
        self.actual_config = {}
        self.last_error = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 id={self.camera_id}")
        self.actual_config = configure_camera(
            self.cap, self.cfg.get("camera", {}), self.cfg["target_fps"]
        )
        warmup_end = time.monotonic() + float(self.cfg["camera"].get("warmup_sec", 0.0))
        while time.monotonic() < warmup_end:
            self.cap.read()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self):
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            timestamp_ns = time.monotonic_ns()
            if not ok:
                self.last_error = "camera_read_failed"
                time.sleep(0.005)
                continue
            packet = FramePacket(
                frame=frame,
                timestamp_ns=timestamp_ns,
                frame_id=self.frame_id,
                dropped_frames=self.pending_drops,
            )
            self.pending_drops = 0
            self.frame_id += 1
            try:
                self.queue.put_nowait(packet)
            except queue.Full:
                # 丢弃最旧帧以控制端到端延迟，并在下一帧显式报告断点。
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
                self.pending_drops += 1
                try:
                    self.queue.put_nowait(packet)
                except queue.Full:
                    self.pending_drops += 1

    def read(self, timeout=1.0):
        return self.queue.get(timeout=timeout)

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        self.thread = None
        self.cap = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
