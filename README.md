# rPPG Industrial Pipeline (SCRFD + EfficientPhys)

## 运行前必查清单（决定精度成败的3件事）

1. **`configs/inference_config.yaml` 中 `rppg_model.img_size` / `frame_depth` / `chunk_length` / `data_type`**
   必须与你训练 `SCAMPS_EfficientPhys.pth` 时用的 config 完全一致。这几项不匹配是最常见的"模型能跑但输出全错"的原因。

2. **`core/rppg_infer.py` 里 `EfficientPhys(...)` 构造参数**
   我按 rPPG-Toolbox 公开实现的常见接口写的（`in_channels, frame_depth, img_size`），
   你本地 `models/EfficientPhys.py` 如果签名不同（比如多了 `nb_filters1/nb_filters2/dropout_rate` 等），
   需要对应调整这里的实例化代码，否则 `load_state_dict` 会报 key 不匹配。

3. **`signal_processing.fs` 必须等于摄像头实际帧率**，不是配置里随便写的数字。
   `run_camera_demo.py` 启动时会自动检测并在不一致时打印警告，但请务必确认。

## 目录说明

```
configs/    推理与模型配置
onnx/       SCRFD人脸检测onnx模型
models/     EfficientPhys网络结构定义(已有)
weights/    SCAMPS预训练/微调后的权重(已有)
core/       各功能模块，互相解耦，可单独替换测试
pipeline/   总调度类，负责串联core下的各模块
scripts/    可执行入口
```

## 快速开始

```bash
pip install -r requirements.txt
python scripts/run_camera_demo.py --config configs/inference_config.yaml
```

默认入口现在使用 `input_pipeline/` 中的可信输入链路，包含异步摄像头采集、
固定时间轴重采样、SCRFD 五点对齐、人脸会话重置、曝光/清晰度/运动质量门控。

如需与修改前的同步采集和旧 ROITracker 对照，可运行：

```bash
python scripts/run_camera_demo.py --legacy_input --config configs/inference_config.yaml
```

固定视频可通过相同的新输入链路确定性回放：

```bash
python scripts/run_video_demo.py path/to/video.mp4 --config configs/inference_config.yaml
```

将 `input_pipeline.diagnostics.enabled` 设置为 `true` 后，每次运行会在
`outputs/debug/` 下保存配置快照、逐帧时间戳/人脸/质量指标、HR候选结果，
并可选保存原视频和对齐人脸视频。

## 可信输入链路目录

```text
input_pipeline/
├── types.py             统一帧、状态、质量数据结构
├── camera_capture.py    异步摄像头采集和有界缓冲
├── camera_control.py    分辨率、FPS、曝光/白平衡控制
├── timing.py            FPS统计、时间断点检测、固定帧率重采样
├── face_tracker.py      SCRFD检测和五点关键点光流传播
├── face_alignment.py    五点仿射对齐和固定ROI
├── face_session.py      新脸、丢脸、身份切换和预热状态机
├── quality_gate.py      亮度、曝光、清晰度、运动、肤色门控
├── diagnostics.py       输入与HR结果诊断记录
├── replay.py            固定视频确定性回放
└── processor.py         上述模块的统一入口
```

原有 `core/roi_tracker.py` 和 `RPPGPipeline.process_frame()` 均保留，作为 legacy
兼容路径。新链路通过 `RPPGPipeline.process_input()` 进入原有心率算法。

## 精度优化建议（优先级从高到低）

1. 先跑通全链路，确认没有崩溃、没有明显的心率跳变
2. 核对上面"运行前必查清单"里的三项配置一致性
3. **用真实场景数据微调模型**（这是收益最大的一步，SCAMPS是合成数据，存在明显domain gap，
   建议采集部署场景真实视频+同步血氧仪/心电数据做fine-tune）
4. 调整 `fusion.min_valid_rois` 和 `signal_processing.sqi_threshold`，
   在"输出覆盖率"和"输出可信度"之间找到适合你业务场景的平衡点
5. 边缘设备部署时再考虑 ONNX导出+量化，量化后务必重新跑一遍精度评估，不要假设量化无损

## 已知待完善项（工程后续要做的）

- `scripts/run_camera_demo.py` 只是可视化demo，工业交付还需要一套离线批量视频评估脚本
  （对齐真实心率标注，输出MAE/RMSE/Pearson r），建议在 `scripts/eval_offline.py` 中实现。
- `core/face_detector.py` 目前只保留面积最大的1张人脸，多人场景需要扩展为按 track_id 多路并行处理。
- 光照异常/严重遮挡场景目前依赖SQI阈值过滤，尚未做专门的遮挡检测（比如基于关键点可见性判断）。
