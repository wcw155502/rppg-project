"""
传统色彩空间投影算法(POS) + 多ROI/多算法融合逻辑。

工业场景下不建议只依赖单一深度模型，POS计算量极小，
可以和 EfficientPhys 并行跑，作为交叉验证和低质量场景下的兜底。

"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def extract_rgb_mean(roi_bgr: np.ndarray) -> np.ndarray:
    """对ROI图像求RGB三通道均值，返回顺序为[R,G,B]。"""
    b, g, r = roi_bgr[..., 0].mean(), roi_bgr[..., 1].mean(), roi_bgr[..., 2].mean()
    return np.array([r, g, b], dtype=np.float64)


def extract_skin_rgb_mean(roi_bgr: np.ndarray, min_skin_ratio: float = 0.25):
    """只统计可信肤色像素，并返回遮挡/曝光质量信息。

    采用轻量 YCbCr 肤色规则，不依赖额外的人脸解析模型。刘海、眼镜框、
    深色阴影和严重过曝区域通常不会进入掩码，因此不会污染 POS 的 RGB 均值。
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None, {"skin_ratio": 0.0, "visible": False}

    pixels = roi_bgr.astype(np.float32)
    b, g, r = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    # YCbCr 阈值兼顾常见肤色，并排除过暗、过曝像素。
    skin_mask = (
        (y >= 35.0) & (y <= 235.0)
        & (cb >= 77.0) & (cb <= 127.0)
        & (cr >= 133.0) & (cr <= 173.0)
    )
    skin_ratio = float(skin_mask.mean())
    visible = skin_ratio >= min_skin_ratio
    detail = {"skin_ratio": skin_ratio, "visible": visible}
    if not visible:
        return None, detail

    return np.array(
        [r[skin_mask].mean(), g[skin_mask].mean(), b[skin_mask].mean()],
        dtype=np.float64,
    ), detail


def pos_algorithm(rgb_seq: np.ndarray, fs: int, window_sec: float = 1.6) -> np.ndarray:
    """
    POS算法核心实现。
    rgb_seq: (T, 3) 逐帧RGB均值序列
    return: (T,) 一维BVP信号
    """
    T = rgb_seq.shape[0]
    win_len = max(2, int(window_sec * fs))
    H = np.zeros(T, dtype=np.float64)
    overlap_weight = np.zeros(T, dtype=np.float64)
    window = np.hanning(win_len)
    if not window.any():
        window = np.ones(win_len)

    for t in range(T - win_len + 1):
        C = rgb_seq[t: t + win_len].T  # (3, win_len)
        mean_c = C.mean(axis=1, keepdims=True)
        mean_c[mean_c == 0] = 1e-6
        Cn = C / mean_c

        S1 = Cn[1] - Cn[2]
        S2 = Cn[1] + Cn[2] - 2 * Cn[0]
        alpha = S1.std() / (S2.std() + 1e-9)
        P = S1 + alpha * S2

        # 加窗重叠相加可降低每个短窗边缘的不连续，减少 HR 跳动。
        H[t: t + win_len] += (P - P.mean()) * window
        overlap_weight[t: t + win_len] += window

    valid = overlap_weight > 1e-9
    H[valid] /= overlap_weight[valid]
    return H


# 旧版 POS 实现（保留用于历史结果对照，不再由推理管线调用）。
def pos_algorithm_legacy(rgb_seq: np.ndarray, fs: int, window_sec: float = 1.6) -> np.ndarray:
    T = rgb_seq.shape[0]
    win_len = max(2, int(window_sec * fs))
    H = np.zeros(T)
    for t in range(T - win_len):
        C = rgb_seq[t: t + win_len].T
        mean_c = C.mean(axis=1, keepdims=True)
        mean_c[mean_c == 0] = 1e-6
        Cn = C / mean_c
        S1 = Cn[1] - Cn[2]
        S2 = Cn[1] + Cn[2] - 2 * Cn[0]
        alpha = S1.std() / (S2.std() + 1e-9)
        P = S1 + alpha * S2
        H[t: t + win_len] += P - P.mean()
    return H


class MultiSourceFusion:
    """
    汇总 EfficientPhys 输出 + POS输出 + 多ROI结果，
    用SQI加权，输出最终可信的HR估计（或判定本窗口不可信）。
    """

    def __init__(self, sqi_threshold: float = 0.3, min_valid_sources: int = 1):
        self.sqi_threshold = sqi_threshold
        self.min_valid_sources = min_valid_sources

    def fuse(self, candidates):
        """
        candidates: List[dict]，每个元素形如 {"source": str, "hr": float, "sqi": float}
        return: (final_hr: float|None, detail: dict)
        """
        valid = [c for c in candidates if c["sqi"] >= self.sqi_threshold and c["hr"] is not None]

        detail = {
            "num_candidates": len(candidates),
            "num_valid": len(valid),
            "candidates": candidates,
        }

        if len(valid) < self.min_valid_sources:
            logger.debug(f"[Fusion] 有效信号源不足(valid={len(valid)}, 需要>={self.min_valid_sources})，本窗口不输出。")
            return None, detail

        weights = np.array([c["sqi"] for c in valid])
        hrs = np.array([c["hr"] for c in valid])
        final_hr = float(np.sum(hrs * weights) / np.sum(weights))
        detail["final_hr"] = final_hr
        return final_hr, detail
