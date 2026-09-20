"""Train the ported G1 BeyondMimic task using this project's RSL-RL."""

from datetime import datetime
from pathlib import Path

import torch


def train_mimic(args, app_launcher):
    from isaaclab.utils.io import dump_yaml
    from isaaclab_tasks.utils import get_checkpoint_path
    from rsl_rl.runners import OnPolicyRunner

    from legged_lab.utils import task_registry
    from legged_lab.utils.cli_args import update_rsl_rl_cfg
    from legged_lab.utils.mimic.vecenv_wrapper import MimicVecEnvWrapper
    from legged_lab.terrains import G1_MOTION_TERRAIN_FILES

    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    env_cfg, agent_cfg = env_cfg.copy(), agent_cfg.copy()
    env_class = task_registry.get_task_class(args.task)
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args)
    device = args.device
    if args.distributed:
        device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.seed += app_launcher.local_rank
    env_cfg.sim.device = agent_cfg.device = device
    env_cfg.seed = agent_cfg.seed
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs

    motion_cfg = next(iter(env_cfg.scene.motion_reference.motion_buffers.values()))
    motion_dir = Path(args.motion_dir or motion_cfg.path).expanduser().resolve()
    motion_cfg.path = str(motion_dir)
    if not motion_dir.is_dir():
        raise FileNotFoundError(f"G1 motion directory does not exist: {motion_dir}. Supply --motion_dir.")
    motion_file = args.motion_file or env_cfg.motion_file
    if env_cfg.motion_file:
        if Path(motion_file).name != env_cfg.motion_file:
            raise ValueError(f"Task {args.task} requires motion {env_cfg.motion_file}.")
        if args.terrain not in ("auto", env_cfg.motion_terrain):
            raise ValueError(f"Task {args.task} requires terrain {env_cfg.motion_terrain}.")
    terrain_name = args.terrain
    if terrain_name == "auto":
        if not motion_file and any((motion_dir / name).is_file() for name in G1_MOTION_TERRAIN_FILES):
            raise ValueError("Select one obstacle clip with --motion_file to pair it with its terrain.")
        terrain_name = G1_MOTION_TERRAIN_FILES.get(Path(motion_file or "").name, "plane")
    if terrain_name != "plane":
        expected = G1_MOTION_TERRAIN_FILES.get(Path(motion_file or "").name)
        if terrain_name != expected:
            raise ValueError(f"Terrain {terrain_name} requires its matching obstacle --motion_file.")
        from legged_lab.envs.g1.g1_mimic_config import configure_g1_motion_terrain

        configure_g1_motion_terrain(env_cfg, terrain_name)
    env_cfg.motion_terrain = terrain_name
    print(f"[INFO] Mimic task: {args.task}; motion: {motion_file or 'directory selection'}")
    print(f"[INFO] Mimic terrain: {terrain_name}")
    if motion_file:
        if not (motion_dir / motion_file).is_file():
            raise FileNotFoundError(motion_dir / motion_file)
        env_cfg.motion_file = motion_file
        motion_cfg.selected_files = [motion_file]
        motion_cfg.filtered_motion_selection_filepath = None

    if args.smoke_test:
        agent_cfg.num_steps_per_env = 8
        agent_cfg.algorithm.num_learning_epochs = 1
        agent_cfg.algorithm.num_mini_batches = 1
        agent_cfg.max_iterations = args.max_iterations or 2
        agent_cfg.save_interval = 1
        # Exercise episode reset, reference resampling, and timeout bootstrapping.
        env_cfg.episode_length_s = 0.12

    log_root = Path("logs") / agent_cfg.experiment_name
    log_dir = log_root / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + agent_cfg.run_name)
    env = env_class(cfg=env_cfg, render_mode=None)
    try:
        wrapped = MimicVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=str(log_dir), device=device)
        if agent_cfg.resume:
            runner.load(get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint))
        dump_yaml(str(log_dir / "params/env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params/agent.yaml"), agent_cfg)
        before = None
        if args.smoke_test:
            obs, rewards, dones, _ = wrapped.step(torch.zeros(env.num_envs, wrapped.num_actions, device=device))
            assert all(torch.isfinite(v).all() for v in obs.values()), "Non-finite observations"
            assert rewards.shape == (env.num_envs,) and torch.isfinite(rewards).all(), "Invalid rewards"
            assert dones.shape == (env.num_envs,), "Invalid done flags"
            before = [p.detach().clone() for p in runner.alg.policy.parameters()]
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        if args.smoke_test:
            after = list(runner.alg.policy.parameters())
            assert all(torch.isfinite(p).all() for p in after), "Non-finite policy parameters"
            assert any(not torch.equal(a, b) for a, b in zip(before, after)), "PPO did not update parameters"
            assert list(log_dir.glob("model_*.pt")), "No model checkpoint saved"
            print(f"[SMOKE PASS] G1 mimic: finite steps, PPO update, checkpoint; {log_dir}")
    finally:
        env.close()
