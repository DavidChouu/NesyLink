from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    gym_id: str
    map_id: str | None = None
    map_path: str | Path | None = None
    reward_id: str | None = None
    reward_module: str | None = None
    reward_kwargs: Mapping[str, float] = field(default_factory=dict)
    max_steps: int = 500
    action_repeat: int = 1
    control_mode: str = "pixel"
    observation_mode: str = "full"
    monster_move_periods: Mapping[str, int] = field(default_factory=dict)
    max_monsters: int | None = None
    max_inventory: int = 2
    mission: str = ""
    player_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not self.gym_id.strip():
            raise ValueError("gym_id must be a non-empty string")
        if self.map_id is None and self.map_path is None:
            raise ValueError(f"task '{self.task_id}' must define map_id or map_path")
        if self.reward_id is not None and self.reward_module is not None:
            raise ValueError(f"task '{self.task_id}' cannot define both reward_id and reward_module")
        if self.max_steps < 1:
            raise ValueError(f"task '{self.task_id}' max_steps must be >= 1")
        if self.action_repeat < 1:
            raise ValueError(f"task '{self.task_id}' action_repeat must be >= 1")
        if self.control_mode not in {"pixel", "grid"}:
            raise ValueError(f"task '{self.task_id}' control_mode must be 'pixel' or 'grid'")
        if self.observation_mode not in {"full", "grid", "pixels"}:
            raise ValueError(f"task '{self.task_id}' observation_mode must be 'full', 'grid', or 'pixels'")
        if self.max_monsters is not None and self.max_monsters < 1:
            raise ValueError(f"task '{self.task_id}' max_monsters must be >= 1")
        if self.max_inventory < 1:
            raise ValueError(f"task '{self.task_id}' max_inventory must be >= 1")
        object.__setattr__(self, "reward_kwargs", MappingProxyType(dict(self.reward_kwargs)))
        object.__setattr__(
            self,
            "monster_move_periods",
            MappingProxyType(
                {
                    str(monster_type): max(1, int(period))
                    for monster_type, period in self.monster_move_periods.items()
                }
            ),
        )
        object.__setattr__(self, "player_config", MappingProxyType(dict(self.player_config)))
