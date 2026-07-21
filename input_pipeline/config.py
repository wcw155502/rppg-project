from copy import deepcopy


DEFAULT_CONFIG = {
    "target_fps": 30.0,
    "queue_size": 3,
    "max_time_gap_ms": 150.0,
    "camera": {
        "width": 1280,
        "height": 720,
        "warmup_sec": 2.0,
        "detect_every_n_frames": 5,
        "max_flow_error": 20.0,
        "show_aligned_face": False,
        "max_abs_yaw": 0.45,
        "fourcc": "MJPG",
        "lock_auto_exposure": False,
        "lock_white_balance": False,
    },
    "face": {
        "aligned_size": 144,
        "model_bbox_scale": 1.5,
        "lost_timeout_sec": 0.35,
        "identity_iou_threshold": 0.30,
        "warmup_sec": 2.0,
    },
    "quality": {
        "min_brightness": 35.0,
        "max_brightness": 220.0,
        "max_overexposed_ratio": 0.12,
        "max_underexposed_ratio": 0.20,
        "min_sharpness": 20.0,
        "max_motion": 25.0,
        "min_skin_ratio": 0.15,
    },
    "diagnostics": {
        "enabled": False,
        "output_dir": "outputs/debug",
        "save_source_video": False,
        "save_aligned_face_video": True,
    },
}


def build_input_config(cfg):
    result = deepcopy(DEFAULT_CONFIG)
    supplied = cfg.get("input_pipeline", {})
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key].update(value)
        else:
            result[key] = value
    if result["target_fps"] <= 0:
        raise ValueError("input_pipeline.target_fps 必须大于0")
    if result["queue_size"] < 2:
        raise ValueError("input_pipeline.queue_size 必须至少为2")
    signal_fps = float(cfg.get("signal_processing", {}).get("fs", result["target_fps"]))
    if abs(float(result["target_fps"]) - signal_fps) > 1e-6:
        raise ValueError(
            "input_pipeline.target_fps 必须与 signal_processing.fs 完全一致，"
            "否则心率频率轴会产生系统性偏差"
        )
    return result
