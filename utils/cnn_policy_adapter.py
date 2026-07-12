"""Task-specific CNN-only adapters for evaluating the rule agents.

The original task modules are left unchanged.  Each adapter replaces only the
module-level perception function with ``classify_frame_cnn(..., fallback=False)``
and delegates reset/act to the original policy object.  Use these objects via
``utils/cnn_policy_adapter.py:task2_policy`` and the evaluator's
``--task-policy`` option.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from nesylink.vision import classify_frame_cnn


class CNNOnlyPolicyAdapter:
    def __init__(self, module_name: str):
        module = import_module(module_name)

        def classify(frame):
            return classify_frame_cnn(frame, fallback=False)

        # Task 1 already calls classify_frame_cnn directly.  Other task files
        # resolve classify_frame through their module global at action time.
        if hasattr(module, "classify_frame"):
            module.classify_frame = classify
        self._policy = module.make_policy()

    def reset(self) -> None:
        self._policy.reset()

    def act(self, obs, info: dict[str, Any] | None = None) -> int:
        try:
            return int(self._policy.act(obs, info))
        except RuntimeError as exc:
            # Official evaluation should count a perception failure as policy
            # failure, not abort every remaining episode in the batch.  WAIT
            # uses no alternate perception backend, so this remains CNN-only.
            if "CNN did not detect player" not in str(exc):
                raise
            return 0


task1_policy = CNNOnlyPolicyAdapter(
    "docs.Mathematical_logic.examples.task1_fsm_bfs_agent"
)
task2_policy = CNNOnlyPolicyAdapter(
    "docs.Mathematical_logic.examples.task2_fsm_bfs_agent"
)
task3_policy = CNNOnlyPolicyAdapter(
    "docs.Mathematical_logic.examples.task3_fsm_bfs_agent"
)
task4_policy = CNNOnlyPolicyAdapter(
    "docs.Mathematical_logic.examples.task4_fsm_bfs_agent"
)
task5_policy = CNNOnlyPolicyAdapter(
    "docs.Mathematical_logic.examples.task5_fsm_bfs_agent"
)
