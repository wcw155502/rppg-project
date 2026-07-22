"""第二阶段配置的局部默认值与校验。

部署配置仍集中在 ``configs/``，避免运行时需要在多个源码目录中搜索 YAML；
本模块只拥有测量相关配置的含义和约束。
"""
from copy import deepcopy


DEFAULT_MEASUREMENT_CONFIG = {
    "model": {"enabled": True},
    "signal_processing": {
        "fs": 30,
        "bandpass_low_hz": 0.7,
        "bandpass_high_hz": 2.5,
        "filter_order": 3,
        "sqi_threshold": 0.3,
        "bvp_window_sec": 10,
        "update_interval_sec": 2,
        "max_quality_gap_sec": 0.5,
    },
    "fusion": {
        "use_traditional_fallback": True,
        "rois": ["forehead", "cheek_l", "cheek_r"],
        "min_valid_rois": 1,
        "min_skin_ratio": 0.25,
        "min_valid_frame_ratio": 0.7,
    },
    "smoother": {"type": "kalman", "process_var": 0.5, "meas_var": 4.0},
}


def build_measurement_config(app_cfg: dict) -> dict:
    """提取并校验第二阶段配置，供 ``RPPGMeasurement`` 的组装层使用。"""
    result = deepcopy(DEFAULT_MEASUREMENT_CONFIG)
    sections = {
        "model": app_cfg.get("rppg_model", {}),
        "signal_processing": app_cfg.get("signal_processing", {}),
        "fusion": app_cfg.get("fusion", {}),
        "smoother": app_cfg.get("smoother", {}),
    }
    for name, supplied in sections.items():
        result[name].update(supplied)

    signal = result["signal_processing"]
    if signal["fs"] <= 0:
        raise ValueError("signal_processing.fs 必须大于 0")
    if not 0 < signal["bandpass_low_hz"] < signal["bandpass_high_hz"] < signal["fs"] / 2:
        raise ValueError("signal_processing 的频段必须位于 Nyquist 频率以内")
    if result["fusion"]["min_valid_rois"] < 1:
        raise ValueError("fusion.min_valid_rois 必须至少为 1")
    return result
