"""
信号后处理模块：积分重建、去趋势、带通滤波、HR估计、信号质量评估(SQI)。
职责单一，输入输出都是numpy一维数组，方便单测和替换实现。
"""
import numpy as np
from scipy.signal import butter, filtfilt


def reconstruct_bvp(diff_signal: np.ndarray) -> np.ndarray:
    """EfficientPhys输出的是BVP一阶差分，这里做积分还原成波形。"""
    return np.cumsum(diff_signal)


def detrend_signal(signal: np.ndarray, fs: int) -> np.ndarray:
    """简单滑动平均去趋势，比不做要好，比smoothness-prior detrending略粗糙但足够工业实时场景使用。"""
    window = max(3, int(fs))
    kernel = np.ones(window) / window
    trend = np.convolve(signal, kernel, mode="same")
    return signal - trend


def bandpass_filter(signal: np.ndarray, fs: int, low_hz: float, high_hz: float, order: int = 3) -> np.ndarray:
    nyq = fs / 2.0
    low, high = low_hz / nyq, high_hz / nyq
    if not (0 < low < high < 1):
        raise ValueError(f"非法滤波频段: low={low_hz}Hz high={high_hz}Hz fs={fs}Hz，请核对采样率配置。")
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def postprocess_diff_signal(diff_signal: np.ndarray, fs: int, low_hz: float, high_hz: float, order: int = 3):
    bvp = reconstruct_bvp(diff_signal)
    bvp = detrend_signal(bvp, fs)
    bvp = bandpass_filter(bvp, fs, low_hz, high_hz, order)
    return bvp


def estimate_hr_fft(bvp_filt: np.ndarray, fs: int, low_hz: float, high_hz: float) -> float:
    n = len(bvp_filt)
    if n < fs * 2:
        raise ValueError("信号长度过短，无法可靠估计心率，建议至少2秒以上数据。")
    windowed = bvp_filt * np.hanning(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not mask.any():
        raise ValueError("有效频段内无频谱能量，请检查采样率/滤波参数配置。")
    peak_freq = freqs[mask][np.argmax(power[mask])]
    return float(peak_freq * 60.0)


def signal_quality_index(bvp_detrended: np.ndarray, fs: int, low_hz: float, high_hz: float) -> float:
    """
    用去趋势、但尚未带通滤波的信号计算有效心率频段能量占比（SQI）。

    不能传入带通滤波后的信号，否则滤波器已经移除了频段外能量，
    随机噪声也会得到接近 1 的虚高 SQI，失去质量门控作用。

    SQI越接近1，说明信号能量越集中在合理心率频段内，越可信；
    SQI很低通常意味着运动伪影、遮挡或严重光照干扰。
    """
    signal = np.asarray(bvp_detrended, dtype=np.float64).reshape(-1)
    n = len(signal)
    if n < fs * 2:
        raise ValueError("信号长度过短，无法可靠评估信号质量，建议至少2秒以上数据。")
    if not np.all(np.isfinite(signal)):
        raise ValueError("信号包含 NaN 或无穷值，无法评估信号质量。")

    # 与 HR 估计保持一致地加窗，降低有限窗口造成的频谱泄漏。
    windowed = signal * np.hanning(n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    total = power.sum()
    if total <= np.finfo(np.float64).eps:
        return 0.0
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(power[band_mask].sum() / total)


# 旧版实现（保留用于结果对照，不再由推理管线调用）。
# 旧管线把已经带通滤波的信号传入这里，会使频段能量占比天然接近 1。
def signal_quality_index_legacy(bvp_filt: np.ndarray, fs: int, low_hz: float, high_hz: float) -> float:
    """旧版 SQI：对已带通的信号计算频段能量占比，仅用于历史结果对照。"""
    n = len(bvp_filt)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(bvp_filt)) ** 2
    total = power.sum() + 1e-9
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(power[band_mask].sum() / total)
