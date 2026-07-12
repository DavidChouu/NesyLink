# CNN 与规则策略测评报告（Task 1、2、5）

## 1. 测试口径

本报告依据助教新版 `evaluation.md` 与 `utils/evaluate_policy.py`，只测试
`mathematical_logic/task_1`、`task_2` 和 `task_5`。

- 端到端策略测评使用 `info_mode=safe`，策略只能读取 RGB 像素、上一动作奖励和物品栏。
- Task 1 直接使用 CNN 感知；Task 2、Task 5 通过 `utils/cnn_policy_adapter.py`
  将原有像素分类入口替换为 `classify_frame_cnn(..., fallback=False)`。
- CNN 未检测到玩家时只执行 `WAIT`，不会回退到朴素颜色识别，因此端到端结果是
  CNN-only 感知加规则策略的结果。
- 感知准确率由 `utils/evaluate_cnn_perception.py` 离线计算。该工具读取模拟器内部状态
  生成真值，但这些真值不会传给策略，只用于测试和统计，不能作为提交策略的一部分。

## 2. 新测评接口

新版测评的主要变化已经同步到本项目：

- 默认使用 `--info-mode safe`，不再向策略暴露坐标、血量、房间、事件和地图内部状态。
- 支持 `--task-policy TASK_ID=POLICY_SPEC`，可分别绑定各任务策略。
- `--robustness-suite` 固定按 60% 原地图、30% 空间变体、10% 颜色变体测评。
- 成功条件是 `world_completed`，中间里程碑不等同于通关。

本次端到端基线命令的核心参数为：

```bash
python utils/evaluate_policy.py \
  --tasks mathematical_logic/task_1 mathematical_logic/task_2 mathematical_logic/task_5 \
  --task-policy mathematical_logic/task_1=utils/cnn_policy_adapter.py:task1_policy \
  --task-policy mathematical_logic/task_2=utils/cnn_policy_adapter.py:task2_policy \
  --task-policy mathematical_logic/task_5=utils/cnn_policy_adapter.py:task5_policy \
  --info-mode safe --robustness-suite --num-envs 10 \
  --json-out /tmp/cnn_rules_robustness_10.json
```

## 3. 端到端通关结果

每个任务共 10 局：原地图 6 局、空间变体 3 局、颜色变体 1 局（灰度）。

| 任务 | 原地图 | 空间变体 | 颜色变体 | 合计成功率 |
| --- | ---: | ---: | ---: | ---: |
| Task 1 | 6/6（100%） | 3/3（100%） | 0/1（0%） | 9/10（90%） |
| Task 2 | 6/6（100%） | 3/3（100%） | 0/1（0%） | 9/10（90%） |
| Task 5 | 6/6（100%） | 0/3（0%） | 0/1（0%） | 6/10（60%） |
| 三任务合计 | 18/18（100%） | 6/9（66.7%） | 0/3（0%） | 24/30（80%） |

原地图平均步数分别为 Task 1：291、Task 2：265、Task 5：1199。Task 5
距离 1200 步上限仅余 1 步，属于极限通关，容错很低。

另外对五种颜色变体各测 1 局：

- Task 1 仅 `bright` 成功，颜色变体成功率为 1/5（20%）。
- Task 2、Task 5 五种颜色变体均失败，成功率均为 0/5。
- 因此，当前 CNN 并没有获得对新版颜色扰动的可靠不变性。

## 4. CNN 感知准确率

在默认地图、默认颜色的成功轨迹上，每 8 帧采样一次，共采样 221 帧：

| 指标 | Task 1（37 帧） | Task 2（34 帧） | Task 5（150 帧） |
| --- | ---: | ---: | ---: |
| 原始静态 tile 头准确率 | 99.966% | 99.743% | 99.892% |
| CNN + 后处理整图准确率 | 99.595% | 99.596% | 98.525% |
| 玩家 tile 完全正确率 | 83.784% | 94.118% | 68.667% |
| 玩家平均 tile 曼哈顿误差 | 0.162 | 0.059 | 0.313 |
| CNN 异常帧 | 0 | 0 | 0 |

动态目标的重点结果：

- Task 2 怪物 precision / recall 均为 100%。
- Task 5 怪物 precision / recall 均为 93.636%。
- Task 5 宝箱 precision 为 100%，recall 为 88.667%。
- Task 5 button precision 为 100%，recall 仅 22.059%，是明显弱项。

“原始静态 tile 头”只统计模型静态分类输出；“CNN + 后处理”还包含动态实体头和
确定性像素后处理，后者不能被表述为纯 CNN 准确率。玩家 tile 正确率较低也不一定代表
完全误识别，主要包含人物跨 tile 移动时相邻 tile 偏移；平均偏移仍小于一个 tile。

## 5. 可信结论

当前 CNN + 规则策略在默认画面和原地图上，Task 1、2、5 的小样本成功率均为 100%；
Task 1、2 也通过了三个空间变体。但这套方案还不能视为稳健或“百分之百正确”：

1. 三个任务在灰度观测下全部失败，扩展颜色测试也几乎全部失败。
2. Task 5 的三个空间变体全部失败，策略布局泛化不足。
3. Task 5 默认地图需要 1199/1200 步，任何识别抖动或多余动作都可能导致失败。
4. Task 5 的玩家定位、怪物检测和 button 召回均显著弱于 Task 1、2。

本次 `--num-envs 10` 是快速、可复现基线，不等同于助教默认的每任务 100 局正式结果。
按当前实测样本，所要求三个任务的固定套件总成功率是 **80%（24/30）**；在修复颜色
泛化和 Task 5 空间策略之前，不应宣称达到正式测评满分。
