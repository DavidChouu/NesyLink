# TODO: 添加对于数理逻辑五个 Task 的使用与说明文档

本次数理逻辑项目一共包含了五个 Task，难度设计为 Easy、Easy、Medium、Medium、Hard。

可以采用各种方式去解决，hard coding、搜索算法、强化学习 等等都可以，如果使用强化学习的话，在`nesylink/rewards/` 目录下有一些参考的 reward function （不保证收敛，仅供参考）可以使用，当然你也可以自己修改 reward function 来训练你的 agent。

## Task 1: Key-Door Task
很简单的一个小任务，控制 Link 打开宝箱，拿到钥匙，打开门，OK，Mission Accomplished!

## Task 2: Kill-Monster Task
也是一个简单小任务，控制 Link 挥舞宝剑杀掉怪物再离开房间即可判定成功

## Task 3: Long horizon Task
算是前面两个任务的组合版本，只不过这次任务存在三个房间，Link 需要经过第二个房间（有一个 monster 干扰），前往第三个房间中拿到钥匙，最后回到第一个房间打开门

## Task 4: Map switch Task
从这个任务开始，环境中出现了可以改变地图结构的机关，Link 需要不断使用 switch 切换中央地图的结构从而完成任务

## Task 5: Map switch + Long horizon Task
最难的任务，欢迎大家来挑战！