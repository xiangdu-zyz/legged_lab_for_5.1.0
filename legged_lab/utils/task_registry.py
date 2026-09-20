# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

from typing import TYPE_CHECKING, Callable

from rsl_rl.env import VecEnv

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnvConfig
    from legged_lab.envs.base.base_env_config import BaseAgentConfig


class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
        self.lazy_factories = {}

    def register_lazy(self, name: str, factory: Callable[[], tuple]):
        """Defer optional task imports until that task is requested."""
        self.lazy_factories[name] = factory

    def _resolve(self, name: str):
        if name not in self.task_classes and name in self.lazy_factories:
            env_class, env_cfg, agent_cfg = self.lazy_factories[name]()
            self.register(name, env_class, env_cfg, agent_cfg)

    def register(self, name: str, task_class: VecEnv, env_cfg: "BaseEnvConfig", train_cfg: "BaseAgentConfig"):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        self._resolve(name)
        return self.task_classes[name]

    def get_cfgs(self, name) -> tuple["BaseEnvConfig", "BaseAgentConfig"]:
        self._resolve(name)
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        return env_cfg, train_cfg


# make global task registry
task_registry = TaskRegistry()
