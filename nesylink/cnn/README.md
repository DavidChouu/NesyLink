# nesylink.cnn

这里是从 `NesyLink-cnn/CNN` 迁移进来的 CNN 感知实验模块，只包含 CNN 本身：数据生成、标注、训练、checkpoint 和单图推理可视化。

当前模块不会自动替换 `nesylink.vision` 或 `student_agent` 的感知逻辑。后续如果要接 Perception，可以从 `model.py` 里的 `component_boxes_from_hybrid_output(...)` 输出开始写 adapter。

详细使用说明见 [USAGE.md](USAGE.md)。
