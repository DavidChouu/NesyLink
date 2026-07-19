# 复现记录

本附件记录 2026-07-19 对 `submissions/` 最终提交目录进行的复验。所有命令均从
NesyLink 仓库根目录执行；正式性能统计只以同目录的
[`robustness_suite_eval.json`](robustness_suite_eval.json) 为数据源。

## 1. 代码版本与提交边界

- 仓库基线提交：`f285cb8864afc058fbe698ed93a9794590eeb9cb`
- 短提交：`f285cb8`
- 提交时间：`2026-07-19T22:32:26+08:00`
- 提交标题：`submission 2`
- `submissions/` 中最终文件的 SHA-256 见第 8 节；若工作树文件晚于上述 Git
  基线，以这些文件哈希为准。

本项目只提交并使用两个 Lean 文件：

1. `submissions/examples/NesyLinkEnvironment.lean`：主要形式化与证明文件，覆盖
   具体环境语义、五关策略、安全性、精化、条件完备性和公开地图通关证书；
2. `submissions/examples/BFSFormalization.lean`：辅助性的通用图搜索理论文件，
   独立补充 BFS 可靠性、完备性、最短性和里程碑分解证明。

两者抽象层次不同，辅助文件不替代主要文件对具体 NesyLink 环境和五关策略的
验证。除此之外没有其他 Lean 源文件或 Lean 文件依赖。

## 2. 软件与硬件

正式 500-episode 测评使用主报告记录的 WSL2/Ubuntu 22.04.5 LTS 环境：CPython
3.10.20、Gymnasium 1.3.0、NumPy 2.2.6、PyTorch 2.11.0+cu128、Torchvision
0.26.0+cu128 和 NVIDIA GeForce RTX 4060 Laptop GPU。最新提交验收另在 CPython
3.13.12、Gymnasium 1.3.0、NumPy 1.26.4、PyTorch 2.10.0+cu128、PyYAML 6.0.3
环境中复验 Python 语法、动态导入和 JSON 一致性。两套环境的用途不同，不把后者
误记为生成正式 JSON 的环境。

Lean 验收统一使用：

| 项目 | 版本 |
|---|---|
| Lean | 4.32.0 |
| Lake | 5.0.0 |
| 工具链 | `leanprover/lean4:v4.32.0` |

正式 Policy 调用分类器时使用代码默认的 CPU inference device；checkpoint 训练
元数据中的训练 device 为 CUDA。两者不是同一次运行配置。

## 3. CNN checkpoint

Policy 默认加载：

```text
nesylink/cnn/checkpoints/tiny_hybrid_cnn_button_pressed_all_v1.weights.pt
```

配套完整 checkpoint `tiny_hybrid_cnn_button_pressed_all_v1.pt` 的关键元数据为：

| 字段 | 值 |
|---|---|
| best epoch | 9 |
| 初始化权重 | `tiny_hybrid_cnn_targeted_v3.weights.pt` |
| component classes | floor, wall, player, chest, monster, trap, abyss, button, button_pressed, switch, gap, bridge, exit_normal, exit_locked, exit_conditional, npc, unknown |
| dynamic classes | player, monster |
| epochs / batch size | 14 / 128 |
| learning rate / weight decay | 0.001 / 0.0001 |
| validation ratio / seed | 0.12 / 41 |
| train tile / object tile accuracy | 0.9995545048 / 0.9991826100 |
| validation tile / object tile accuracy | 0.9995252418 / 0.9991286087 |

训练元数据记录的历史数据路径为
`nesylink/cnn/generated/button_pressed_all_v1/train`。该历史生成数据目录未作为作业
附件提交；报告不将其描述为已提交资产。

## 4. Python 语法、路径与导入检查

五个正式 Policy 的实际提交路径为：

```text
submissions/examples/task1_agent.py
submissions/examples/task2_agent.py
submissions/examples/task3_agent.py
submissions/examples/task4_agent.py
submissions/examples/task5_agent.py
```

执行：

```bash
python -m py_compile \
  submissions/examples/task1_agent.py \
  submissions/examples/task2_agent.py \
  submissions/examples/task3_agent.py \
  submissions/examples/task4_agent.py \
  submissions/examples/task5_agent.py
```

结果：退出码 0。五个文件还分别通过动态导入检查，均导出 `Policy`。每个 Policy
从其当前 `submissions/examples/` 位置使用 `Path(__file__).resolve().parents[2]`
定位仓库根目录，不依赖旧的 `docs/Mathematical_logic/examples/` 目录层级。

此外，使用上述五个当前提交路径，在 `--info-mode safe --num-envs 1 --seed 0`
口径下对 Task1–Task5 各执行一集原始地图冒烟检查，五关均以
`world_completed` 成功终止。步数依次为 291、165、543、1040 和 1176。

## 5. 500-episode 正式测评

按当前提交目录复现正式测评时使用以下命令：

```bash
python utils/evaluate_policy.py \
  --tasks mathematical_logic/task_1 mathematical_logic/task_2 \
          mathematical_logic/task_3 mathematical_logic/task_4 \
          mathematical_logic/task_5 \
  --task-policy mathematical_logic/task_1=submissions/examples/task1_agent.py \
  --task-policy mathematical_logic/task_2=submissions/examples/task2_agent.py \
  --task-policy mathematical_logic/task_3=submissions/examples/task3_agent.py \
  --task-policy mathematical_logic/task_4=submissions/examples/task4_agent.py \
  --task-policy mathematical_logic/task_5=submissions/examples/task5_agent.py \
  --info-mode safe \
  --robustness-suite \
  --num-envs 100 \
  --seed 0 \
  --json-out submissions/report_artifacts/robustness_suite_eval.json
```

未传 `--max-steps` 或 `--action-repeat`。正式结果文件写入时间为
`2026-07-19 21:55:07 +08:00`，大小为 510243 bytes。自动结构核对结果：

- 共 500 episodes、15 个 task-stage 汇总组；
- 每关 original/spatial/color 恰为 60/30/10；
- 500/500 的 `success=true`、`terminated=true` 且
  `terminal_reason="world_completed"`；
- 500/500 的 `truncated=false`。

正式 JSON 使用的五关控制逻辑、CNN 权重和测评参数与当前提交一致；提交整理阶段
仅把 Policy 路径统一到 `submissions/examples/`，并修正 `PROJECT_ROOT` 的目录层级，
未修改目标选择、BFS、动作过滤或战斗逻辑。路径修正后的五关冒烟检查结果见上一节。

详细均值、里程碑和事件累计见主报告；逐 episode 数据保留在 JSON 中。

## 6. Lean 编译

### 6.1 主要形式化文件

```bash
lean submissions/examples/NesyLinkEnvironment.lean
lake env lean submissions/examples/NesyLinkEnvironment.lean
```

两条命令均退出 0。语法错误、编译错误、未解证明目标和待决元变量均为 0；两份
输出各有 79 条 `unnecessarySimpa`、`unusedSimpArgs` 等非致命 linter 建议。

主文件按固定统计口径包含 472 条行首且无前置 attribute 的主要 `theorem`、202 条
带 attribute 的辅助 `theorem`、10 条 `private theorem` 和 19 个可计算 `example`。
472 条主要声明的完整名称与当前源码行号见
[`theorem_inventory.md`](theorem_inventory.md)。五关公开证书仍连接真实
`EngineExec`、`WorldCompleted`、`alive` 和 `ValidState`。

### 6.2 辅助 BFS 理论文件

```bash
lean submissions/examples/BFSFormalization.lean
lake env lean submissions/examples/BFSFormalization.lean
```

两条命令均退出 0。该文件包含 16 条主要 `theorem`，以有限图和精确 BFS 层为抽象
对象，独立验证路径、完备性、最短性与里程碑分解，不重复统计到主文件的 472 条
定理清单中。

## 7. Lean 禁止项扫描

```bash
rg -n '\b(sorry|admit|axiom|unsafe)\b|maxHeartbeats' \
  submissions/examples/NesyLinkEnvironment.lean \
  submissions/examples/BFSFormalization.lean
```

结果：0 个匹配。两个文件均未使用证明占位、额外公理、`unsafe` 证明逃逸或提高
heartbeat 的方式换取编译通过。

## 8. SHA-256

| 文件 | SHA-256 |
|---|---|
| `task1_agent.py` | `04c0b6c19e375bb969b67e72111cd98ac999e2386828945a2b92a4e169f43417` |
| `task2_agent.py` | `a0d84850cb92dc1c1723e2cbfd1470ac1417c56cb6a4b9948b5dfd6601c450de` |
| `task3_agent.py` | `697f4d40e8cd2a272ec7c5d06bb1bc0b29f4071d84880387d747f6f581ad1c81` |
| `task4_agent.py` | `aef6332234bde546ad8463258a43bf0e83821af886c707d94efc049859b246e6` |
| `task5_agent.py` | `20c9b9ee3e3665954f7390ffd1d5e2645cf9de58e2473c1981816e60d3a2e62c` |
| `NesyLinkEnvironment.lean` | `a511b2ceae14b13c11ea8ab631c216f9c0e056fcf6e14fe871ff7ecfc3630f0d` |
| `BFSFormalization.lean` | `16ac0e0d702657469eae0b835cb14a500f3c17bc598da2b3b069e9aa931fd034` |
| default weights checkpoint | `c145847a722b48d5f9d4199be2832007e2866b94a8047fb4e860c124c27440d0` |
| metadata checkpoint | `ddc9a88dd44b3394f23bb3bdacfc10e8f73d793fdff6fd3f41ea4535fb622c68` |
| `robustness_suite_eval.json` | `960778f42c068d555a980aae58d2ba482f8d9c1091121d4991612d2326be6037` |

## 9. 最终验收矩阵

| 检查 | 结果 |
|---|---|
| 五个 Policy `py_compile` 与动态导入 | PASS |
| Policy 文档路径与实际提交路径 | PASS |
| safe-mode robustness suite JSON | PASS，500 episodes / 15 groups / 60-30-10 |
| JSON 成功与终止字段 | PASS，500/500 world completed |
| 主文件主要定理声明/清单 | PASS，472/472 |
| 主文件可计算回归样例 | PASS，19 个 `example` |
| 两个 Lean 文件禁止项/`maxHeartbeats` 扫描 | PASS，0 个匹配 |
| `lean` 编译两个 Lean 文件 | PASS，均退出 0 |
| `lake env lean` 编译两个 Lean 文件 | PASS，均退出 0 |
| 语法/编译/未解目标/元变量错误 | PASS，0 个 |
| `git diff --check -- submissions` | PASS |

形式化编译与 Python 500/500 成功率是两类独立证据：前者验证明确陈述的逻辑
命题，后者测量 CNN、像素执行、控制器和环境随机性的实际组合效果。
