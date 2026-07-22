from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np


class InputStatus(str, Enum):
    NO_FACE = "no_face"
    WARMING_UP = "warming_up"
    MEASURING = "measuring"
    LOW_QUALITY = "low_quality"
    TIME_GAP = "time_gap"


@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp_ns: int
    frame_id: int
    dropped_frames: int = 0
    source: str = "camera"


@dataclass
class TimingStats:
    fps: float = 0.0
    interval_mean_ms: float = 0.0
    interval_std_ms: float = 0.0
    interval_p95_ms: float = 0.0
    sample_count: int = 0


@dataclass
class QualityReport:
    valid: bool
    reasons: Tuple[str, ...] = ()
    brightness: float = 0.0
    overexposed_ratio: float = 0.0
    underexposed_ratio: float = 0.0
    sharpness: float = 0.0
    motion: float = 0.0
    skin_ratio: float = 0.0


@dataclass
class ProcessedInput:
    timestamp_ns: int
    frame_id: int
    status: InputStatus
    frame: np.ndarray
    aligned_face: Optional[np.ndarray] = None
    model_input_face: Optional[np.ndarray] = None
    rois: Dict[str, np.ndarray] = field(default_factory=dict)
    # 矩形ROI保存(x1,y1,x2,y2)，动态脸颊ROI保存(N,2)多边形顶点。
    roi_boxes: Dict[str, object] = field(default_factory=dict)
    bbox: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None
    quality: Optional[QualityReport] = None
    timing: Optional[TimingStats] = None
    reset_required: bool = False
    reset_reason: Optional[str] = None
    track_id: int = 0
