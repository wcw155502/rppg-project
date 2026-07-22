# 第一、二阶段模块结构

## 决定

第一阶段的可信输入实现归入 `rppg.input`；第二阶段的 rPPG 估计、候选审查、融合与连续性归入 `rppg.measurement`。`runtime.rppg_pipeline` 是唯一负责组装两者的薄适配层。

`MeasurementEvent` 和 `RPPGMeasurementResult` 是两个阶段之间的 seam。输入模块拥有采集、时间戳、重采样、人脸对齐、质量门控与重置判断；测量模块拥有 BVP、候选、融合、平滑和测量重置。

部署用 YAML 继续集中在 `configs/`，以便一个运行配置能够完整复现；每个阶段在自己的 `config.py` 中拥有默认值和校验。

## 兼容性

旧的 `core`、`input_pipeline` 与 `pipeline` 保留为兼容导入。`legacy.roi_tracker` 只供旧的 `process_frame()` 适配路径使用，新代码不得依赖它。

## 后果

第二阶段新增 Welch PSD、谐波消歧、多 ROI 共识、异常拒绝时，只会进入 `rppg.measurement`，不会污染采集代码。真实准确率评估和训练适配尚未创建目录，待实际实施时再加入。
