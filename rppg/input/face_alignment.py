import cv2
import numpy as np


# ArcFace 常用五点模板，基准尺寸 112x112。
REFERENCE_LANDMARKS_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


class FaceAligner:
    def __init__(self, output_size=144):
        self.output_size = int(output_size)
        self.reference = REFERENCE_LANDMARKS_112 * (self.output_size / 112.0)

    def align(self, frame, landmarks):
        return self._align_to_reference(frame, landmarks, self.reference)

    def align_efficientphys_context(self, frame, landmarks, context_scale=1.0):
        """生成仅供 EfficientPhys 使用的扩大视野对齐画布。

        POS 始终使用 ``align`` 的标准五点模板。这里将目标关键点向画布中心
        收缩，使同一张输出画布容纳 ``context_scale`` 倍的原始视野。
        """
        scale = max(1.0, float(context_scale))
        center = (self.output_size - 1) / 2.0
        expanded_reference = center + (self.reference - center) / scale
        return self._align_to_reference(frame, landmarks, expanded_reference)

    def _align_to_reference(self, frame, landmarks, reference):
        points = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(
            points, reference, method=cv2.LMEDS
        )
        if matrix is None:
            return None, None
        aligned = cv2.warpAffine(
            frame,
            matrix,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        projected = cv2.transform(points[None, ...], matrix)[0]
        error = float(np.mean(np.linalg.norm(projected - reference, axis=1)))
        return aligned, {
            "matrix": matrix,
            "mean_error_px": error,
            "projected_landmarks": projected,
        }

    def estimate_yaw(self, projected_landmarks):
        """用鼻尖相对双眼中点的水平偏移估计轻量转头量，约位于[-1,1]。"""
        points = np.asarray(projected_landmarks, dtype=np.float32).reshape(5, 2)
        eye_mid_x = float((points[0, 0] + points[1, 0]) / 2.0)
        half_eye_distance = max(1.0, float(abs(points[1, 0] - points[0, 0]) / 2.0))
        return float(np.clip((points[2, 0] - eye_mid_x) / half_eye_distance, -1.0, 1.0))

    def extract_rois(self, aligned_face, projected_landmarks=None, model_bbox_scale=1.0):
        size = aligned_face.shape[0]
        boxes = {
            # 避开眼睛和嘴部；额头受刘海影响时由质量门控拒绝。
            "forehead": (int(.25 * size), int(.12 * size), int(.75 * size), int(.34 * size)),
            "full": (int(.08 * size), int(.08 * size), int(.92 * size), int(.94 * size)),
        }
        yaw = self.estimate_yaw(projected_landmarks) if projected_landmarks is not None else 0.0
        # 更紧的双颊四边形：上沿低于眼镜，下沿高于嘴角，内沿避开鼻翼。
        # 鼻尖偏向哪一侧，哪一侧在图像中更压缩；对应ROI随转头实时收窄。
        left_shrink = max(0.45, 1.0 - max(0.0, -yaw) * 1.2)
        right_shrink = max(0.45, 1.0 - max(0.0, yaw) * 1.2)
        left_outer, left_inner = .19 * size, .39 * size
        right_inner, right_outer = .61 * size, .81 * size
        left_outer = left_inner - (left_inner - left_outer) * left_shrink
        right_outer = right_inner + (right_outer - right_inner) * right_shrink
        boxes["cheek_l"] = np.array([
            [left_outer, .50 * size], [left_inner, .51 * size],
            [.37 * size, .72 * size], [left_outer + .02 * size, .74 * size],
        ], dtype=np.int32)
        boxes["cheek_r"] = np.array([
            [right_inner, .51 * size], [right_outer, .50 * size],
            [right_outer - .02 * size, .74 * size], [.63 * size, .72 * size],
        ], dtype=np.int32)
        # 仅扩大 EfficientPhys 输入区域；POS 的额头/双颊多边形保持独立。
        # 区域受对齐画布边界约束，避免越界和引入虚假 padding。
        scale = max(1.0, float(model_bbox_scale))
        half = min(size / 2.0, size * 0.42 * scale)
        center = size / 2.0
        boxes["model_full"] = (
            max(0, int(center - half)), max(0, int(center - half)),
            min(size, int(center + half)), min(size, int(center + half)),
        )
        rois = {}
        for name, region in boxes.items():
            if isinstance(region, np.ndarray):
                x, y, w, h = cv2.boundingRect(region)
                crop = aligned_face[y:y + h, x:x + w].copy()
                local_polygon = region - np.array([x, y], dtype=np.int32)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [local_polygon], 255)
                rois[name] = cv2.bitwise_and(crop, crop, mask=mask)
            else:
                x1, y1, x2, y2 = region
                rois[name] = aligned_face[y1:y2, x1:x2]
        return rois, boxes, yaw
