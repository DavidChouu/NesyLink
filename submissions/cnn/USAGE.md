# nesylink.cnn 使用说明

CNN 模块只负责感知：

```text
160x128 RGB 游戏像素图 -> ComponentBox 列表
```

它不负责规划，也不决定动作。

## 1. 输入

```text
图片大小: 160 x 128
通道: RGB
地图网格: 10 x 8
每个格子: 16 x 16 pixels
```

训练样本是一对文件：

```text
xxx.png   游戏画面
xxx.json  标注信息
```

当前训练集位置：

```text
nesylink/cnn/generated/retrain_aligned/train
nesylink/cnn/generated/retrain_aligned/test
```

## 2. 输出

模型输出是 `ComponentBox` 列表：

```python
ComponentBox(
    kind="monster",
    tiles=((8, 2),),
    bbox_px=(128, 32, 144, 48),
    score=0.98,
)
```

字段含义：

```text
kind     类别
tiles    对应 grid 坐标
bbox_px  像素框: (x1, y1, x2, y2)
score    置信度
```

当前类别必须和 `nesylink/cnn/components.py` 里的 `COMPONENT_CLASSES` 对齐：

```text
floor
wall
player
chest
monster
trap
abyss
button
switch
gap
bridge
exit_normal
exit_locked
exit_conditional
npc
unknown
```

输出规则：

```text
wall / chest / trap / abyss / button / switch / gap / bridge / npc
  一个格子一个框。

player / monster
  用动态头输出像素框，不要求在格子中心。

exit_normal / exit_locked / exit_conditional
  门是两格宽，两个门格合并成一个框。
```

## 3. 生成数据集

当前对齐版本建议生成 3000 张训练图和 300 张测试图：

```bash
python -m nesylink.cnn.generate_dataset \
  --train-dir nesylink/cnn/generated/retrain_aligned/train \
  --test-dir nesylink/cnn/generated/retrain_aligned/test \
  --train-count 3000 \
  --test-count 300 \
  --train-seed-start 150000 \
  --test-seed-start 180000 \
  --annotate-test \
  --labels \
  --sheet-out nesylink/cnn/generated/retrain_aligned/test_annotated_sheet.png
```

生成后先看：

```text
nesylink/cnn/generated/retrain_aligned/test_annotated_sheet.png
nesylink/cnn/generated/retrain_aligned/dataset_summary.json
```

`dataset_summary.json` 里要确认 `abyss`、`exit_conditional`、`exit_locked`、`npc` 都有数量。

## 4. 训练

```bash
python -m nesylink.cnn.train \
  --data-dir nesylink/cnn/generated/retrain_aligned/train \
  --pattern 'train_*.json' \
  --epochs 45 \
  --batch-size 8 \
  --out nesylink/cnn/checkpoints/tiny_hybrid_cnn_aligned.pt \
  --preview-out nesylink/cnn/checkpoints/tiny_hybrid_cnn_aligned_preview.png \
  --device auto
```

训练完成后会得到：

```text
nesylink/cnn/checkpoints/tiny_hybrid_cnn_aligned.pt
nesylink/cnn/checkpoints/tiny_hybrid_cnn_aligned.weights.pt
```

接 Perception 时一般加载：

```text
tiny_hybrid_cnn_aligned.weights.pt
```

注意：类表改过以后，旧的 `tiny_hybrid_cnn_retrain/quality/augmented` checkpoint 不能直接混用。

## 5. 接入 Perception

`nesylink.vision.classify_frame_cnn(...)` 默认加载：

```text
nesylink/cnn/checkpoints/tiny_hybrid_cnn_aligned.weights.pt
```

默认动态阈值是 `0.20`。最终接入输出是 `PixelObservation`，其中：

```text
静态 grid: 主要来自 CNN tile head
player / monster: 用 palette 锚点细化最终像素位置，避免多 player、丢 player 和 monster 假阳性
button: 用 palette 锚点确认，避免 floor 被误判成 button
```

## 6. 单张图片预测

```bash
python -m nesylink.cnn.infer_boxes \
  --image nesylink/cnn/generated/retrain_aligned/test/test_0000.png \
  --checkpoint nesylink/cnn/checkpoints/tiny_hybrid_cnn_aligned.weights.pt \
  --out nesylink/cnn/generated/retrain_aligned/test_0000_prediction.png \
  --threshold 0.50
```

如果预测框太少，降低 `--threshold`。如果错误框太多，提高 `--threshold`。

## 7. 单张图片标注

这一步不是模型预测，只是把 JSON 里的标准答案画出来：

```bash
python -m nesylink.cnn.annotate_scene \
  --image nesylink/cnn/generated/retrain_aligned/test/test_0000.png \
  --json nesylink/cnn/generated/retrain_aligned/test/test_0000.json \
  --out nesylink/cnn/generated/retrain_aligned/test/test_0000_label.png \
  --labels
```
