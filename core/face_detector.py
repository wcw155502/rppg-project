"""
SCRFD 人脸检测器封装 (onnxruntime)

scrfd_10g_bnkps_shape640x640.onnx 是标准的 SCRFD 多尺度检测头模型：
- 固定输入 640x640
- 3个特征层 stride = [8, 16, 32]
- 每层2个anchor
- bnkps 后缀表示带5点landmark分支，因此共9个输出:
    [score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32]
"""
import cv2
import numpy as np
import onnxruntime as ort
import logging


logger = logging.getLogger(__name__)


def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def nms(dets, thresh):
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


class SCRFDDetector:
    def __init__(self, onnx_path, input_size=(640, 640), det_thresh=0.5,
                 nms_thresh=0.4, provider="auto"):
        available = ort.get_available_providers()
        requested = provider
        if provider in (None, "auto"):
            provider = (
                "CUDAExecutionProvider"
                if "CUDAExecutionProvider" in available
                else "CPUExecutionProvider"
            )
        elif provider not in available:
            logger.warning(
                "请求的 ONNX Runtime provider=%s 不可用，可用provider=%s；回退到CPU。",
                provider, available,
            )
            provider = "CPUExecutionProvider"
        self.provider = provider
        self.session = ort.InferenceSession(onnx_path, providers=[provider])
        logger.info(
            "SCRFD ONNX provider=%s (requested=%s, available=%s)",
            self.provider, requested, available,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_size = tuple(input_size)
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh
        self.input_mean = 127.5
        self.input_std = 128.0
        self.center_cache = {}
        self._parse_output_layout(len(self.output_names))

    def _parse_output_layout(self, num_outputs):
        if num_outputs == 6:
            self.fmc = 3
            self.feat_stride_fpn = [8, 16, 32]
            self.num_anchors = 2
            self.use_kps = False
        elif num_outputs == 9:
            self.fmc = 3
            self.feat_stride_fpn = [8, 16, 32]
            self.num_anchors = 2
            self.use_kps = True
        else:
            raise ValueError(
                f"未识别的SCRFD输出格式，output数量={num_outputs}，"
                f"预期6(无kps)或9(带kps)，请核对onnx导出方式或联系模型来源确认输出定义。"
            )

    def _preprocess(self, img):
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / self.input_std, self.input_size,
            (self.input_mean, self.input_mean, self.input_mean), swapRB=True
        )
        return blob

    def detect(self, img, max_num=1):
        """
        img: 原始BGR图像 (H,W,3)，任意分辨率，内部会letterbox到input_size
        return: bboxes (N,5) [x1,y1,x2,y2,score], kpss (N,5,2) or None
        """
        letterbox_img, scale, pad = self._letterbox(img)
        blob = self._preprocess(letterbox_img)
        outs = self.session.run(self.output_names, {self.input_name: blob})

        scores_list, bboxes_list, kpss_list = [], [], []
        input_h, input_w = self.input_size[1], self.input_size[0]

        for idx, stride in enumerate(self.feat_stride_fpn):
            scores = outs[idx].reshape(-1)
            bbox_preds = outs[idx + self.fmc].reshape(-1, 4) * stride
            height, width = input_h // stride, input_w // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape(-1, 2)
                if self.num_anchors > 1:
                    anchor_centers = np.repeat(anchor_centers, self.num_anchors, axis=0)
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= self.det_thresh)[0]
            if len(pos_inds) == 0:
                continue
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])

            if self.use_kps:
                kps_preds = outs[idx + self.fmc * 2].reshape(-1, 10) * stride
                kpss = distance2kps(anchor_centers, kps_preds).reshape(-1, 5, 2)
                kpss_list.append(kpss[pos_inds])

        if len(scores_list) == 0:
            return np.zeros((0, 5)), None

        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list)
        # 还原letterbox带来的缩放/padding
        bboxes = (bboxes - np.array([pad[0], pad[1], pad[0], pad[1]])) / scale

        pre_det = np.hstack([bboxes, scores[:, None]]).astype(np.float32)
        order = scores.argsort()[::-1]
        pre_det = pre_det[order]
        keep = nms(pre_det, self.nms_thresh)
        det = pre_det[keep]

        kpss = None
        if self.use_kps and len(kpss_list) > 0:
            kpss_all = np.concatenate(kpss_list)[order]
            kpss_all = (kpss_all - np.array([pad[0], pad[1]])) / scale
            kpss = kpss_all[keep]

        if max_num > 0 and det.shape[0] > max_num:
            # 只保留面积最大(离摄像头最近)的max_num个人脸，工业场景通常只关心主体人脸
            areas = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            keep_idx = areas.argsort()[::-1][:max_num]
            det = det[keep_idx]
            kpss = kpss[keep_idx] if kpss is not None else None

        return det, kpss

    def _letterbox(self, img):
        h, w = img.shape[:2]
        target_w, target_h = self.input_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        pad_x, pad_y = (target_w - new_w) // 2, (target_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return canvas, scale, (pad_x, pad_y)
