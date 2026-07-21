import numpy as np

from .types import InputStatus


def bbox_iou(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class FaceSession:
    def __init__(self, cfg):
        self.lost_timeout_ns = int(float(cfg["lost_timeout_sec"]) * 1e9)
        self.warmup_ns = int(float(cfg["warmup_sec"]) * 1e9)
        self.identity_iou_threshold = float(cfg["identity_iou_threshold"])
        self.track_id = 0
        self.last_bbox = None
        self.last_seen_ns = None
        self.stable_since_ns = None

    def update_face(self, timestamp_ns, bbox, quality_valid):
        reset_required = False
        reset_reason = None
        if self.last_bbox is None:
            self.track_id += 1
            self.stable_since_ns = timestamp_ns
            reset_required = True
            reset_reason = "new_face"
        elif bbox_iou(self.last_bbox, bbox) < self.identity_iou_threshold:
            self.track_id += 1
            self.stable_since_ns = timestamp_ns
            reset_required = True
            reset_reason = "face_changed"
        self.last_bbox = np.asarray(bbox, dtype=np.float32).copy()
        self.last_seen_ns = timestamp_ns
        warmed_up = timestamp_ns - self.stable_since_ns >= self.warmup_ns
        if not quality_valid:
            status = InputStatus.LOW_QUALITY
        elif not warmed_up:
            status = InputStatus.WARMING_UP
        else:
            status = InputStatus.MEASURING
        return status, reset_required, reset_reason

    def update_missing(self, timestamp_ns):
        if self.last_seen_ns is None:
            return InputStatus.NO_FACE, False, None
        if timestamp_ns - self.last_seen_ns >= self.lost_timeout_ns:
            self.last_bbox = None
            self.last_seen_ns = None
            self.stable_since_ns = None
            return InputStatus.NO_FACE, True, "face_lost"
        return InputStatus.NO_FACE, False, None

    def reset(self):
        self.last_bbox = None
        self.last_seen_ns = None
        self.stable_since_ns = None
