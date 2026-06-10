from __future__ import annotations

from .builtin import BUILTIN_TASKS, register_builtin_tasks
from .config import MATHEMATICAL_LOGIC_CONFIG, mathematical_logic_config
from .registry import get_task, get_task_by_gym_id, list_tasks, register_task
from .specs import TaskSpec


register_builtin_tasks()


__all__ = [
    "BUILTIN_TASKS",
    "MATHEMATICAL_LOGIC_CONFIG",
    "TaskSpec",
    "get_task",
    "get_task_by_gym_id",
    "list_tasks",
    "mathematical_logic_config",
    "register_builtin_tasks",
    "register_task",
]
