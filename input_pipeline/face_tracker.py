import cv2
import numpy as np


class LandmarkFaceTracker:
    """定期 SCRFD 检测，中间用 LK 光流传播五点关键点和人脸框。"""

    def __init__(self, detector, detect_every_n_frames=5, max_flow_error=20.0):
        self.detector = detector
        self.detect_every = max(1, int(detect_every_n_frames))
        self.max_flow_error = float(max_flow_error)
        self.frame_count = 0
        self.previous_gray = None
        self.bbox = None
        self.landmarks = None

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        must_detect = (
            self.bbox is None
            or self.landmarks is None
            or self.previous_gray is None
            or self.frame_count % self.detect_every == 0
        )
        result = self._detect(frame) if must_detect else self._track(gray, frame)
        self.previous_gray = gray
        self.frame_count += 1
        return result

    def _detect(self, frame):
        dets, landmarks_all = self.detector.detect(frame, max_num=1)
        if dets.shape[0] == 0 or landmarks_all is None or len(landmarks_all) == 0:
            self.bbox = None
            self.landmarks = None
            return None, None, "detector_no_face"
        self.bbox = dets[0, :4].astype(np.float32)
        self.landmarks = np.asarray(landmarks_all[0], dtype=np.float32).reshape(5, 2)
        return self.bbox.copy(), self.landmarks.copy(), "detected"

    def _track(self, gray, frame):
        previous_points = self.landmarks.reshape(-1, 1, 2).astype(np.float32)
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            previous_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is None or status is None or int(status.sum()) < 5:
            return self._detect(frame)
        error_values = error.reshape(-1) if error is not None else np.zeros(5)
        if not np.all(np.isfinite(next_points)) or float(error_values.mean()) > self.max_flow_error:
            return self._detect(frame)

        next_points = next_points.reshape(5, 2)
        old_points = previous_points.reshape(5, 2)
        matrix, _ = cv2.estimateAffinePartial2D(old_points, next_points, method=cv2.LMEDS)
        if matrix is None:
            return self._detect(frame)
        corners = np.array([
            [self.bbox[0], self.bbox[1]], [self.bbox[2], self.bbox[1]],
            [self.bbox[2], self.bbox[3]], [self.bbox[0], self.bbox[3]],
        ], dtype=np.float32)
        moved = cv2.transform(corners[None, ...], matrix)[0]
        h, w = frame.shape[:2]
        bbox = np.array([
            np.clip(moved[:, 0].min(), 0, w), np.clip(moved[:, 1].min(), 0, h),
            np.clip(moved[:, 0].max(), 0, w), np.clip(moved[:, 1].max(), 0, h),
        ], dtype=np.float32)
        if bbox[2] - bbox[0] < 20 or bbox[3] - bbox[1] < 20:
            return self._detect(frame)
        self.bbox = bbox
        self.landmarks = next_points.astype(np.float32)
        return self.bbox.copy(), self.landmarks.copy(), "tracked"

    def reset(self):
        self.frame_count = 0
        self.previous_gray = None
        self.bbox = None
        self.landmarks = None
