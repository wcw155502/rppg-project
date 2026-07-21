"""
配置加载工具：统一读取 yaml 配置，做基础校验，避免各模块各写各的解析逻辑。
"""
import os
import yaml
import logging


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict):
    required_top_keys = ["face_detector", "rppg_model", "signal_processing", "fusion", "smoother"]
    for k in required_top_keys:
        if k not in cfg:
            raise ValueError(f"配置文件缺少必需字段: {k}")

    # 关键路径校验，工业部署时路径错误是最常见的低级故障
    for path_key in ["onnx_path"]:
        p = cfg["face_detector"].get(path_key)
        if p and not os.path.exists(p):
            logging.warning(f"[config] 人脸检测模型路径不存在(将在实际运行目录下重新校验): {p}")

    w = cfg["rppg_model"].get("weight_path")
    if w and not os.path.exists(w):
        logging.warning(f"[config] rPPG模型权重路径不存在(将在实际运行目录下重新校验): {w}")


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
