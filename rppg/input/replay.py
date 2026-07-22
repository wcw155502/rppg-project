import cv2

from .types import FramePacket


class VideoReplay:
    """以确定性时间戳回放固定视频，供离线复现输入链路问题。"""

    def __init__(self, path, fallback_fps=30.0):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {path}")
        reported_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = reported_fps if reported_fps > 0 else float(fallback_fps)
        self.period_ns = int(round(1e9 / self.fps))

    def __iter__(self):
        frame_id = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield FramePacket(
                frame=frame,
                timestamp_ns=frame_id * self.period_ns,
                frame_id=frame_id,
                source="video",
            )
            frame_id += 1

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
