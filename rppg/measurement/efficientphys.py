"""
EfficientPhys 推理封装（PyTorch权重版本）。

已知 EfficientPhys（来自 rPPG-Toolbox）实现上有个关键约束：
- 输入需要 reshape 成 (N*D, C, H, W)，其中 D = frame_depth，
  网络内部通过 BatchNorm/attention 机制处理帧间关系，
  因此送入模型的帧数量必须是 frame_depth 的整数倍。
- 训练标签是 BVP 的一阶差分，所以模型输出也是差分信号，
  需要在后处理阶段做 cumsum 积分还原波形。

本文件对以上约束做了封装，避免每次调用方都要记住这些细节。
"""
import logging
import sys
import os

import numpy as np
import torch

from rppg.measurement.preprocess import preprocess_clip_for_efficientphys

logger = logging.getLogger(__name__)


class EfficientPhysRunner:
    def __init__(self, model_py_path, weight_path, img_size=72, frame_depth=20,
                 chunk_length=180, stride=30, device="cuda"):
        self.img_size = img_size
        self.frame_depth = frame_depth
        self.chunk_length = self._validate_chunk_length(chunk_length, frame_depth)
        self.stride = stride
        self.device = device if torch.cuda.is_available() else "cpu"
        if device == "cuda" and self.device == "cpu":
            logger.warning("[EfficientPhysRunner] 未检测到可用CUDA，已回退到CPU推理。")

        self.model = self._load_model(model_py_path, weight_path)
        self.frame_buffer = []
        self._has_emitted_initial_bvp = False

    def _validate_chunk_length(self, chunk_length, frame_depth):
        """
        EfficientPhys.forward() 第一步是 torch.diff(x, dim=0)，T帧变成T-1帧，
        随后 TSM 模块要求 (T-1) 必须是 frame_depth 的整数倍才能 view() 成功
        (TSM内部用 nt // n_segment 向下取整，不整除时会直接shape mismatch报错)。
        约束: (chunk_length - 1) % frame_depth == 0
        这里自动向上取到最近一个满足条件的合法值。
        """
        remainder = (chunk_length - 1) % frame_depth
        if remainder != 0:
            valid_length = chunk_length - remainder + frame_depth
            logger.warning(
                f"[EfficientPhysRunner] chunk_length={chunk_length} 不满足 "
                f"(chunk_length-1) % frame_depth({frame_depth}) == 0 的约束"
                f"(EfficientPhys内部diff+TSM reshape要求)，已自动修正为 {valid_length}。"
                f"建议同步更新configs/inference_config.yaml，避免下次仍触发这条警告。"
            )
            return valid_length
        return chunk_length

    def _load_model(self, model_py_path, weight_path):
        # 动态导入用户工程里 models/EfficientPhys.py 中的模型类
        models_dir = os.path.dirname(model_py_path)
        if models_dir not in sys.path:
            sys.path.insert(0, os.path.dirname(models_dir))
        from models.EfficientPhys import EfficientPhys  # noqa: E402

        model = EfficientPhys(
            in_channels=3,
            frame_depth=self.frame_depth,
            img_size=self.img_size,
        )
        state_dict = torch.load(weight_path, map_location=self.device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # 兼容 DataParallel 保存时带 module. 前缀的情况
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=False)
        model.to(self.device)
        model.eval()
        logger.info(f"[EfficientPhysRunner] 权重加载完成: {weight_path} device={self.device}")
        return model

    def push_frame(self, roi_frame_bgr):
        """
        持续喂入单帧ROI图像，内部维护滑窗buffer，
        凑够chunk_length时自动触发一次推理并按stride滑动。
        return: np.ndarray 一维差分BVP信号（本次新产出的部分）或 None
        """
        self.frame_buffer.append(roi_frame_bgr)
        if len(self.frame_buffer) < self.chunk_length:
            return None

        clip = self.frame_buffer[-self.chunk_length:]
        pred = self._infer_clip(clip)
        self.frame_buffer = self.frame_buffer[self.stride:]
        # 只返回本次滑窗步长对应的新增部分，避免下游重复处理重叠段
        if pred is None:
            return None
        if not self._has_emitted_initial_bvp:
            # First prediction spans the complete model context; do not discard it.
            self._has_emitted_initial_bvp = True
            return pred
        # Later clips overlap, so append only the new tail.
        return pred[-self.stride:]

    def push_frame_legacy(self, roi_frame_bgr):
        """旧冷启动逻辑，仅保留供历史结果对照；新代码不要调用。

        原实现首次推理也只返回 ``stride`` 个样本，丢弃了前段有效 BVP，
        因而需要额外等待多个推理轮次才能填满分析窗。
        """
        self.frame_buffer.append(roi_frame_bgr)
        if len(self.frame_buffer) < self.chunk_length:
            return None
        clip = self.frame_buffer[-self.chunk_length:]
        pred = self._infer_clip(clip)
        self.frame_buffer = self.frame_buffer[self.stride:]
        return pred[-self.stride:] if pred is not None else None

    @torch.no_grad()
    def _infer_clip(self, clip_bgr):
        try:
            x = preprocess_clip_for_efficientphys(clip_bgr, img_size=self.img_size)  # (T,C,H,W)
            t = torch.from_numpy(x).float().to(self.device)
            out = self.model(t)
            out = out.detach().cpu().numpy().reshape(-1)
            return out
        except Exception as e:
            logger.error(f"[EfficientPhysRunner] 推理失败: {e}")
            return None

    def reset(self):
        self.frame_buffer = []
        self._has_emitted_initial_bvp = False
