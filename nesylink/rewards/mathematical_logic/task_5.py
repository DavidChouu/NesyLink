from __future__ import annotations

from typing import Any

import numpy as np

from .common import MathematicalLogicReward

_DRAIN_INTERVAL = 200 # 恭喜你找到了神秘参数 🤠 


class MathematicalLogicTask5Reward(MathematicalLogicReward):
    reward_name = "mathematical_logic/task_5"
    reward_weights = {
        **MathematicalLogicReward.reward_weights,
        "room_changed": 2.0,
        "button_pressed": 1.0,
        "talked_npc": 0.5,
        "chest_opened": 2.0,
        "gold_delta": 1.0,
        "key_collected": 5.0,
        "keys_delta": 5.0,
        "agent_healed": 2.0,
        "monster_hit": 1.0,
        "monster_kill": 5.0,
        "trap_triggered": -2.0,
        "hp_loss": -2.0,
        "door_opened": 5.0,
        "exit_reached": 5.0,
    }

    def __init__(self, **reward_kwargs: float):
        super().__init__(**reward_kwargs)
        self._call_count = 0
        self._engine: Any = None

    def set_engine_ref(self, engine) -> None:
        self._engine = engine

    def reset(self, obs: Any, info: dict[str, Any]) -> None:
        super().reset(obs, info)
        self._call_count = 0

    @property
    def _player(self):
        return self._engine.runtime.player if self._engine is not None else None

    def __call__(self, obs: Any, info: dict[str, Any], action: int | None = None) -> tuple[float, dict[str, Any]]:
        self._call_count += 1

        drained = False
        player = self._player
        if (
            player is not None
            and self._call_count % _DRAIN_INTERVAL == 0
            and player.health > 0
        ):
            player.health -= 1
            drained = True
            # Patch obs/info so downstream consumers see the drained HP.
            if isinstance(obs, dict) and "health" in obs:
                obs["health"] = np.array([player.health], dtype=np.int32)
            if isinstance(info, dict) and isinstance(info.get("agent"), dict):
                info["agent"]["hp"] = player.health

        reward, reward_info = super().__call__(obs, info, action)

        if drained and player is not None and player.health <= 0:
            reward_info["terminated"] = True
            reward_info["terminated_reason"] = "agent_dead"

        return reward, reward_info


def make_reward(**kwargs):
    return MathematicalLogicTask5Reward(**kwargs)
