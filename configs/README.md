# 运行配置

`inference_config.yaml` 是可直接运行的完整配置。它按模块职责分段：

- `input` 与 `face_detector`：第一阶段可信输入；
- `rppg_model`、`signal_processing`、`fusion`、`smoother`：第二阶段测量；
- `runtime`：运行时展示与调试。

模块的默认值和校验位于其实现目录内的 `config.py`；本目录只保存可复现的部署配置，避免每个模块各自散落一份 YAML。
