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

import argparse
from pathlib import Path
import sys

# Resolve local packages when this file is executed directly from any directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "rsl_rl")]

from isaaclab.app import AppLauncher
from rsl_rl.runners import OnPolicyRunner

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--task", type=str, default=None,
    help="Task: g1_flat, g1_rough, g1_mimic, g1_mimic_stairs_up, or g1_mimic_stairs_down.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

parser.add_argument("--motion_dir", type=str, default=None, help="Directory of retargeted G1 NPZ motion files (mimic).")
parser.add_argument("--motion_file", type=str, default=None, help="Select one retargeted G1 NPZ file (mimic).")
parser.add_argument(
    "--terrain", choices=("auto", "plane", "obstacles_179_305", "obstacles_405_581"), default="auto",
    help="Mimic terrain; auto selects the matching obstacle profile for a known motion file.",
)
parser.add_argument("--smoke_test", action="store_true", help="Validate finite simulation outputs and PPO parameter updates (mimic).")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.task and args_cli.task.startswith("g1_mimic"):
    local_motion_deps = REPOSITORY_ROOT / ".mimic_deps"
    if local_motion_deps.is_dir():
        sys.path.insert(2, str(local_motion_deps))

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import os
from datetime import datetime

import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def train():
    from legged_lab.envs.g1.mimic_tasks import MIMIC_TASK_CONFIGS

    if args_cli.task in MIMIC_TASK_CONFIGS:
        from legged_lab.utils.mimic.training import train_mimic
        return train_mimic(args_cli, app_launcher)

    runner: OnPolicyRunner

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    env_class = task_registry.get_task_class(env_class_name)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.device = args_cli.device
    agent_cfg.device = args_cli.device
    env_cfg.scene.seed = agent_cfg.seed

    if args_cli.distributed:
        env_cfg.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed

    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        train()
    except Exception:
        # Isaac Sim's shutdown can exit before Python prints a pending exception.
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close(wait_for_replicator=False)
