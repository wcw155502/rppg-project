"""可信 rPPG 输入链路。

该包只负责采集、时间轴、人脸连续性、对齐、质量门控和诊断记录；
心率模型与信号算法仍位于原有 core/ 和 pipeline/ 中。
"""

from .types import FramePacket, InputStatus, ProcessedInput, QualityReport

__all__ = [
    "FramePacket",
    "InputStatus",
    "ProcessedInput",
    "QualityReport",
]
