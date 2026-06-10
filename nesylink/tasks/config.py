from __future__ import annotations

from importlib.resources import files
from typing import Any

import yaml


CONFIG_FILE = "task_config/mathematical_logic.yaml"


def load_task_config(filename: str = CONFIG_FILE) -> dict[str, Any]:
    config_path = files(__package__).joinpath(filename)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"task config '{filename}' must contain a mapping")
    return payload


MATHEMATICAL_LOGIC_CONFIG = load_task_config()


def mathematical_logic_config(task_id: str) -> dict[str, Any]:
    defaults = MATHEMATICAL_LOGIC_CONFIG.get("default", {})
    tasks = MATHEMATICAL_LOGIC_CONFIG.get("tasks", {})
    if not isinstance(defaults, dict) or not isinstance(tasks, dict):
        raise ValueError(f"task config '{CONFIG_FILE}' must define 'default' and 'tasks' mappings")
    config = dict(defaults)
    config.update(tasks.get(task_id, {}))
    return config
