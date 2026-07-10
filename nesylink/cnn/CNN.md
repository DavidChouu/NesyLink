我检查并跑完了基础测试，结论是：**队友这波 CNN 更新很有价值，Task2 已经能用 CNN-only 成功；但 Task5 还不行。**

**他主要改了什么**

1. 统一了 CNN 类别表，补上了之前缺失的：

```
abyss
exit_conditional
```

现在 CNN 类别已经和朴素像素识别基本对齐：

```
floor, wall, player, chest, monster, trap, abyss,
button, switch, gap, bridge,
exit_normal, exit_locked, exit_conditional,
npc, unknown
```

1. 修了条件门标注逻辑。

之前 `conditional` 门会被标成 `exit_locked`，现在改成：

```
if exit_type == "conditional":
    return "exit_conditional"
```

1. 修了 abyss/trap 标注逻辑。

现在 abyss 类型陷阱会标成 `abyss`，普通尖刺才标成 `trap`。

1. 加了新的 checkpoint：

```
nesylink/cnn/checkpoints/tiny_hybrid_cnn_aligned.weights.pt
```

1. 更新了 CNN 后处理，模型层面加入了动态框去重/NMS 相关逻辑。

不过注意：当前 `classify_frame_cnn` 默认 checkpoint 仍然是旧的：

```
tiny_hybrid_cnn_quality.weights.pt
```

这个旧模型和新类别数不匹配，直接会报错。测试新 CNN 时必须显式指定：

```
export NESYLINK_CNN_CHECKPOINT=/home/davidchou/nesylink/nesylink/cnn/checkpoints/tiny_hybrid_cnn_aligned.weights.pt
```

**静态感知测试**

Task2 首帧：

```
diff_count = 0
player 正确
monster 正确
chest 正确
exit_conditional 正确
```

Task5 首帧：

```
diff_count = 0
player 正确
monster 正确
chest 正确
exit_normal / exit_locked / exit_conditional 全部正确
npc 正确
```

这比之前好很多。之前 `exit_conditional` 是 0%，现在：

```
Task2 exit_conditional: 523 / 523 = 100%
Task5 exit_conditional: 1043 / 1045 = 99.81%
```

**轨迹级静态测试**

Task2：

```
tile_acc = 98.25%
player_exact = 95.09%
monster recall = 95.80%
chest recall = 100%
multi_player_frames = 118 / 265
```

Task2 虽然还有多 player 现象，但整体已经足够让 agent 成功。

Task5：

```
tile_acc = 98.87%
player_exact = 66.97%
monster recall = 86.09%
chest recall = 100%
cnn_fail_frames = 3
multi_player_frames = 22 / 1199
```

Task5 的主要问题仍然是：

```
player tile 偏移很多
偶发 detect 不到 player
monster 有漏检和误检
exit_normal 有时会被误成 exit_locked/monster
```

**Agent 测试**

我用临时 CNN-only agent 测了 Task2/Task5，只替换感知入口为：

```
classify_frame_cnn(obs, fallback=False)
```

Task2：

```
seed 0..9 全部 success=True
steps=263
reward=127.370
```

所以 Task2 现在 CNN-only 已经能跑通。

Task5：

默认 CNN-only：

```
失败：运行中 CNN did not detect player，直接抛异常
```

降低 player recovery threshold 到 `0.05` 后不再崩，但仍失败：

```
success=False
steps=761
terminal_reason=agent_dead
chest_opened=1
key_collected=1
button_pressed=1
room_changed=1
door_opened=0
monster_killed=0
world_completed=0
```

允许 fallback 到纯像素也还是同样失败，所以 Task5 不是单纯“偶尔丢玩家”导致的，主要还是 CNN 输出对当前 Task5 规划层不够稳定。

最终结论：

```
Task2：新 CNN 已经基本可用，CNN-only agent seed 0..9 全过。
Task5：新 CNN 明显进步，但还不能支撑当前 agent 通关。
```

我建议队友下一步优先修 Task5 的 `player tile` 稳定性，其次修 monster 的误检/漏检。当前最大的硬伤不是类别表了，类别表已经修得不错；现在是动态对象定位问题。