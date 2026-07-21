"""
ROI提取与稳定模块。

工业场景下 rPPG 精度的隐形杀手是 ROI 抖动，不是模型本身。
本模块职责单一：给定检测结果或跟踪结果，输出稳定、平滑的多子区域裁剪图。
"""
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ROITracker:
    def __init__(self, detector, detect_every_n_frames=8, smooth_alpha=0.6,
                 model_bbox_scale=1.2):
        self.detector = detector
        self.detect_every = detect_every_n_frames
        self.alpha = smooth_alpha
        self.model_bbox_scale = max(1.0, float(model_bbox_scale))
        self.frame_idx = 0
        self.last_box = None
        self.cv_tracker = None
        self.tracking_failed_count = 0
        self.max_tracking_failures = 3
        self._warned_no_tracker = False
        self._no_tracker_available = False
        self.face_detected = False
        self.last_regions_boxes = {}  # 记录最近一次各子区域的像素坐标，供外部画框可视化使用

    def _clear_face_state(self):
        """确认无人脸时清除旧框，防止界面继续显示上一张脸的位置。"""
        self.face_detected = False
        self.last_box = None
        self.cv_tracker = None
        self.last_regions_boxes = {}

    def _init_cv_tracker(self, frame, box):
        """
        三级兜底：优先用 opencv-contrib 的 legacy KCF，
        其次尝试旧版本API位置，都没有则返回None，
        退化为"每次都用上一次检测框，靠更频繁的detect_every来补偿"的策略，
        不会因为环境没装contrib包而直接崩溃。
        """
        x1, y1, x2, y2 = box[:4]
        bbox_xywh = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        tracker = None
        try:
            tracker = cv2.legacy.TrackerKCF_create()
        except AttributeError:
            try:
                tracker = cv2.TrackerKCF_create()
            except AttributeError:
                if not self._warned_no_tracker:
                    logger.warning(
                        "[ROITracker] 当前opencv未安装tracker模块(需要opencv-contrib-python)，"
                        "已退化为纯检测模式：每帧都会重新跑人脸检测，CPU占用会更高。"
                        "建议执行: pip uninstall opencv-python -y && pip install opencv-contrib-python"
                    )
                    self._warned_no_tracker = True
                return None
        tracker.init(frame, bbox_xywh)
        return tracker

    def _smooth(self, new_box):
        if self.last_box is None:
            self.last_box = np.array(new_box, dtype=np.float32)
        else:
            self.last_box = self.alpha * self.last_box + (1 - self.alpha) * np.array(new_box, dtype=np.float32)
        return self.last_box

    def update(self, frame):
        """
        return: dict of ROI sub-images, or None（本帧无有效人脸）
        """
        need_detect = (self.frame_idx % self.detect_every == 0) or (self.last_box is None) \
            or (self.tracking_failed_count >= self.max_tracking_failures) \
            or self._no_tracker_available  # 没有可用跟踪器时，退化为每帧检测

        box = None
        if need_detect:
            dets, _ = self.detector.detect(frame, max_num=1)
            if dets.shape[0] > 0:
                box = dets[0][:4]
                self.cv_tracker = self._init_cv_tracker(frame, box)
                if self.cv_tracker is None:
                    self._no_tracker_available = True
                self.tracking_failed_count = 0
            else:
                self.tracking_failed_count += 1
                self._clear_face_state()
        else:
            if self.cv_tracker is not None:
                ok, bbox_xywh = self.cv_tracker.update(frame)
                if ok:
                    x, y, w, h = bbox_xywh
                    box = np.array([x, y, x + w, y + h])
                    self.tracking_failed_count = 0
                else:
                    self.tracking_failed_count += 1
                    self._clear_face_state()

        self.frame_idx += 1

        if box is None:
            self._clear_face_state()
            return None

        smoothed_box = self._smooth(box)
        regions = self._crop_subregions(frame, smoothed_box)
        if not regions:
            self._clear_face_state()
            return None
        self.face_detected = True
        return regions

    def _crop_subregions(self, frame, box):
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = [int(max(0, min(v, w_img if i % 2 == 0 else h_img)))
                           for i, v in enumerate(box)]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        w, h = x2 - x1, y2 - y1
        # EfficientPhys 单独使用扩展框；POS 子区域仍以原始人脸框定位。
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        model_w, model_h = w * self.model_bbox_scale, h * self.model_bbox_scale
        model_box = (
            max(0, int(cx - model_w / 2.0)),
            max(0, int(cy - model_h / 2.0)),
            min(w_img, int(cx + model_w / 2.0)),
            min(h_img, int(cy + model_h / 2.0)),
        )
        boxes = {
            "forehead": (x1 + int(0.2 * w), y1, x2 - int(0.2 * w), y1 + int(0.25 * h)),
            # 略向下移动双颊 ROI，减少眼镜框和镜片反光进入区域。
            "cheek_l": (x1 + int(0.05 * w), y1 + int(0.52 * h), x1 + int(0.38 * w), y1 + int(0.82 * h)),
            "cheek_r": (x2 - int(0.38 * w), y1 + int(0.52 * h), x2 - int(0.05 * w), y1 + int(0.82 * h)),
            "full": (x1, y1, x2, y2),
            "model_full": model_box,
        }
        regions = {name: frame[by1:by2, bx1:bx2] for name, (bx1, by1, bx2, by2) in boxes.items()}
        # 过滤掉裁剪结果为空的子区域（人脸贴近画面边缘时可能出现），坐标同步过滤保持一致
        valid_regions = {k: v for k, v in regions.items() if v.size > 0}
        self.last_regions_boxes = {k: boxes[k] for k in valid_regions}
        return valid_regions

    def get_last_regions(self):
        """返回最近一次各ROI子区域的像素坐标 {name: (x1,y1,x2,y2)}，供可视化画框使用。"""
        return self.last_regions_boxes

    def has_face(self):
        """返回当前帧是否检测或跟踪到有效人脸。"""
        return self.face_detected
