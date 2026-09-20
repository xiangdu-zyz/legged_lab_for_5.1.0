"""Lazy registration of G1 motion imitation training tasks."""

from functools import partial


MIMIC_TASK_CONFIGS = {
    "g1_mimic": "G1BeyondMimicPlaneEnvCfg",
    "g1_mimic_stairs_up": "G1BeyondMimicStairsUpEnvCfg",
    "g1_mimic_stairs_down": "G1BeyondMimicStairsDownEnvCfg",
}


def make_mimic_task(name):
    from isaaclab.envs import ManagerBasedRLEnv

    from legged_lab.envs.g1 import g1_mimic_config

    env_cfg = getattr(g1_mimic_config, MIMIC_TASK_CONFIGS[name])()
    agent_cfg = g1_mimic_config.G1MimicAgentCfg()
    agent_cfg.experiment_name = agent_cfg.wandb_project = name
    return ManagerBasedRLEnv, env_cfg, agent_cfg


def register_mimic_tasks(registry):
    for name in MIMIC_TASK_CONFIGS:
        registry.register_lazy(name, partial(make_mimic_task, name))
