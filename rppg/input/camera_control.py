import logging

import cv2

logger = logging.getLogger(__name__)


def configure_camera(cap, cfg, target_fps):
    """尝试配置摄像头并返回驱动实际报告的参数。"""
    width = int(cfg.get("width", 1280))
    height = int(cfg.get("height", 720))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, float(target_fps))
    fourcc = cfg.get("fourcc")
    if fourcc and len(fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if cfg.get("lock_auto_exposure"):
        # 不同后端取值不同；保留实际读取值供诊断，不假定 set 一定成功。
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    if cfg.get("lock_white_balance") and hasattr(cv2, "CAP_PROP_AUTO_WB"):
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    actual = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps_reported": float(cap.get(cv2.CAP_PROP_FPS)),
    }
    logger.info("摄像头实际配置: %s", actual)
    return actual
