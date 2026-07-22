# Context

## Glossary

- **rPPG measurement**: 从经过质量门控的人脸 ROI 序列中估计远程光电容积描记信号和心率的过程；不是对真实生理脉搏的直接测量。
- **RPPGMeasurement**: 负责 rPPG 窗口、候选信号、融合、平滑和重置的模块。它不负责人脸检测、摄像头采集或界面显示。
- **Measurement event**: 输入链路发送给 `RPPGMeasurement` 的有效 ROI 观测或中断/重置事件。
