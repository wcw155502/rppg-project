"""
预处理模块，严格对齐 EfficientPhys 训练时的数据配置。

重要: EfficientPhys 与 DeepPhys/TS-CAN 不同，网络内部自己做帧间差分，
因此这里只需要做 Standardized，不需要额外产出 DiffNormalized 分支。
如果你实际使用的训练config是别的组合，请对应修改 DATA_TYPE 逻辑。
"""
import cv2
import numpy as np


def resize_frames(frame_seq, size=(72, 72)):
    return [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frame_seq]


def standardize_clip(clip_bgr):
    """
    clip_bgr: List[np.ndarray] (H,W,3) BGR uint8
    return: np.ndarray (T,C,H,W) float32
    """
    arr = np.stack(clip_bgr).astype(np.float32)
    arr = arr[..., ::-1]  # BGR -> RGB，需与训练时颜色通道顺序一致
    mean = arr.mean(axis=(0, 1, 2), keepdims=True)
    std = arr.std(axis=(0, 1, 2), keepdims=True)
    std[std < 1e-6] = 1e-6
    arr = (arr - mean) / std
    arr = np.transpose(arr, (0, 3, 1, 2))  # (T,H,W,C) -> (T,C,H,W)
    return arr.astype(np.float32)


def preprocess_clip_for_efficientphys(clip_bgr, img_size=72):
    resized = resize_frames(clip_bgr, size=(img_size, img_size))
    return standardize_clip(resized)
