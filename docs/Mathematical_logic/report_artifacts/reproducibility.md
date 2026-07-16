# 复现记录

本附件记录 2026-07-16 在仓库根目录完成的正式测评、版本、文件指纹和验收结果。
正式性能统计只以同目录的 `robustness_suite_eval.json` 为数据源。

## 1. 代码版本与工作区边界

- Git 提交：`36e29e3a10b660e77d76ad4729d4ca889300bb6e`
- 短提交：`36e29e3`
- 分支：`main`
- 提交时间：`2026-07-16T03:43:36+08:00`
- 提交标题：`Lean`
- 当前工作树以该提交为基线，修复了 `NesyLinkEnvironment.lean` 的语法、Task4/5
  公开状态证书和 Task3/4/5 安全汇总接口，并同步报告附件；五个 Python
  Policy 和正式测评 JSON 未修改。

## 2. 软件与硬件

| 项目 | 版本或配置 |
|---|---|
| OS | Linux 6.6.87.2-microsoft-standard-WSL2 x86_64，glibc 2.35 |
| CPU | Intel Core i9-14900HX，32 logical CPUs，16 cores，2 threads/core |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU；CUDA 可用 |
| Python | 3.13.12 |
| Gymnasium | 1.3.0 |
| NumPy | 1.26.4 |
| PyTorch | 2.10.0+cu128；16 CPU threads |
| PyYAML | 6.0.3 |
| Lean | 4.32.0，commit `8c9756b28d64dab099da31a4c09229a9e6a2ef35` |
| Lake | 5.0.0-src+8c9756b |

正式 Policy 调用分类器时使用代码默认的 CPU inference device；checkpoint 的训练
元数据记录训练 device 为 CUDA。两者不是同一次运行配置，不应混淆。

## 3. CNN checkpoint

Policy 默认加载：

```text
nesylink/cnn/checkpoints/tiny_hybrid_cnn_button_pressed_all_v1.weights.pt
```

配套完整 checkpoint `tiny_hybrid_cnn_button_pressed_all_v1.pt` 内嵌元数据如下：

| 字段 | 值 |
|---|---|
| best epoch（保存的 `epoch`） | 9 |
| 初始化 | `tiny_hybrid_cnn_targeted_v3.weights.pt` |
| component classes | floor, wall, player, chest, monster, trap, abyss, button, button_pressed, switch, gap, bridge, exit_normal, exit_locked, exit_conditional, npc, unknown |
| dynamic classes | player, monster |
| epochs / batch size | 14 / 128 |
| learning rate / weight decay | 0.001 / 0.0001 |
| validation ratio / seed | 0.12 / 41 |
| train tile / object tile accuracy | 0.9995545048 / 0.9991826100 |
| validation tile / object tile accuracy | 0.9995252418 / 0.9991286087 |

训练元数据中的历史数据路径是
`nesylink/cnn/generated/button_pressed_all_v1/train`；该历史生成数据目录未作为本报告
附件提交，报告不把它虚构成可用资产。

## 4. Python 语法与导入检查

执行：

```bash
python -m py_compile \
  docs/Mathematical_logic/examples/task1_fsm_bfs_agent.py \
  docs/Mathematical_logic/examples/task2_fsm_bfs_agent.py \
  docs/Mathematical_logic/examples/task3_fsm_bfs_agent.py \
  docs/Mathematical_logic/examples/task4_fsm_bfs_agent.py \
  docs/Mathematical_logic/examples/task5_fsm_bfs_agent.py
```

结果：退出码 0。随后使用 `importlib.util.spec_from_file_location` 分别动态导入五个
文件，并将模块写入 `sys.modules` 后执行；五个模块均导入成功且均导出 `Policy`。

## 5. 500-episode 正式测评

执行命令：

```bash
python utils/evaluate_policy.py \
  --tasks mathematical_logic/task_1 mathematical_logic/task_2 \
          mathematical_logic/task_3 mathematical_logic/task_4 \
          mathematical_logic/task_5 \
  --task-policy mathematical_logic/task_1=docs/Mathematical_logic/examples/task1_fsm_bfs_agent.py \
  --task-policy mathematical_logic/task_2=docs/Mathematical_logic/examples/task2_fsm_bfs_agent.py \
  --task-policy mathematical_logic/task_3=docs/Mathematical_logic/examples/task3_fsm_bfs_agent.py \
  --task-policy mathematical_logic/task_4=docs/Mathematical_logic/examples/task4_fsm_bfs_agent.py \
  --task-policy mathematical_logic/task_5=docs/Mathematical_logic/examples/task5_fsm_bfs_agent.py \
  --info-mode safe \
  --robustness-suite \
  --num-envs 100 \
  --seed 0 \
  --json-out docs/Mathematical_logic/report_artifacts/robustness_suite_eval.json
```

未传 `--max-steps` 或 `--action-repeat`。正式重跑将标准输出重定向到 `/dev/null`
以避免终端输出阻塞；参数、Policy 输入、环境执行和 JSON 内容没有改变。命令退出码
为 0，JSON 写入时间为 `2026-07-16 06:10:05 +08:00`。

自动结构核对结果：

- 共 500 episodes、15 个 task-stage 汇总组；
- 每关 original/spatial/color 恰为 60/30/10；
- 500/500 的 `success=true`、`terminated=true`、
  `terminal_reason="world_completed"`；
- 500/500 的 `truncated=false`；
- JSON 大小 510244 bytes。

详细均值、里程碑和事件累计见主报告；逐 episode 数据保留在 JSON 中。

## 6. Lean 编译与静态扫描

依次执行：

```bash
lean docs/Mathematical_logic/examples/NesyLinkEnvironment.lean
lake env lean docs/Mathematical_logic/examples/NesyLinkEnvironment.lean
```

最新再次复验时，两条命令均退出 0。直接 `lean` 用时约 18.4 秒，
`lake env lean` 用时约 18.4 秒（同时启动，墙钟时间受当时负载影响）。
两份日志均有 79 条 Lean linter 的 `unnecessarySimpa`/`unusedSimpArgs` 等风格提示；
`error:`、`unsolved goals`、`unexpected token` 和未决元变量匹配数均为 0。

修复没有提高 heartbeat、删除定理、引入最终状态假设或将目标改为 `True`。
Task4 最终证书用五个房间快照组合；Task5 用阶段化 `currentRoom`/房间投影证书；
五关公开证书仍连接真实 `EngineExec`、`WorldCompleted`、`alive` 和
`ValidState`。

静态命令：

```bash
rg -n '\b(sorry|admit|axiom|unsafe)\b' \
  docs/Mathematical_logic/examples/NesyLinkEnvironment.lean
rg -n 'maxHeartbeats' \
  docs/Mathematical_logic/examples/NesyLinkEnvironment.lean
rg -c '^theorem ' \
  docs/Mathematical_logic/examples/NesyLinkEnvironment.lean
rg -c '^@\[[^]]+\] theorem ' \
  docs/Mathematical_logic/examples/NesyLinkEnvironment.lean
rg -c '^private theorem ' \
  docs/Mathematical_logic/examples/NesyLinkEnvironment.lean
rg -c '^example ' \
  docs/Mathematical_logic/examples/NesyLinkEnvironment.lean
```

结果：禁止项和 `maxHeartbeats` 匹配均为 0；按“行首且无前置 attribute 的
`theorem`”口径有 472 条
主要声明，定理清单同样有 472 个编号数据行。源码另外包含 202 条
带 attribute 的辅助 `theorem`、10 条 `private theorem` 和 19 个可计算 `example`；
这里披露其数量，但不混入
项目既定的 472 条主要定理清单。

## 7. SHA-256

| 文件 | SHA-256 |
|---|---|
| `task1_fsm_bfs_agent.py` | `fc8b8251d884cc1251c872a9b81b0b4604d4e8f3047e98d6ea81f87164a7ee18` |
| `task2_fsm_bfs_agent.py` | `0e94da5b0e3e8abec533afc3532355b0fd5c74d21cd93430d48fc47cd951f270` |
| `task3_fsm_bfs_agent.py` | `0547d4bbf524e1f6e0eed66be85325ae2c76331973966ec6696a7ae4a42478b2` |
| `task4_fsm_bfs_agent.py` | `e67386347c9b693fd8824655eeb571e2a5825abc3d255d0cbafb3efc33819ceb` |
| `task5_fsm_bfs_agent.py` | `7317291d1c54ba6f313097a9a17f236b27d09e8c897c9ea4256feb877b7d7030` |
| `NesyLinkEnvironment.lean` | `a511b2ceae14b13c11ea8ab631c216f9c0e056fcf6e14fe871ff7ecfc3630f0d` |
| default `.weights.pt` | `c145847a722b48d5f9d4199be2832007e2866b94a8047fb4e860c124c27440d0` |
| metadata `.pt` | `ddc9a88dd44b3394f23bb3bdacfc10e8f73d793fdff6fd3f41ea4535fb622c68` |
| `robustness_suite_eval.json` | `c9751a9f5703d545917faf0d3a28e24c8bde76ab2f964182f9ddbc54e70de028` |

## 8. 最终验收矩阵

| 检查 | 结果 |
|---|---|
| 五个 Policy `py_compile` | PASS，退出 0 |
| safe-mode robustness suite | PASS，退出 0 |
| JSON 500 episodes / 15 groups / 60-30-10 | PASS |
| JSON 成功与终止字段 | PASS，500/500 world completed |
| 主要定理声明/清单数量 | PASS，472/472；另披露 202 attribute + 10 private |
| 可计算回归样例 | PASS，19 个 `example` |
| Lean 禁止项/`maxHeartbeats` 扫描 | PASS，0 个匹配 |
| `lean` 独立编译 | PASS，退出 0 |
| `lake env lean` 编译 | PASS，退出 0 |
| 语法/编译/未解目标/元变量错误 | PASS，0 个匹配；79 条非致命 linter 提示 |
| `git diff --check` | PASS，退出 0 |

形式化编译与 Python 500/500 成功率仍作为两类独立证据；本附件记录实际命令和
退出码，不制作或伪造终端截图。
