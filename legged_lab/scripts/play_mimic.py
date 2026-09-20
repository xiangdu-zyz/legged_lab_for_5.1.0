"""Play or export a trained G1 terrain-mimic policy.

Examples:
    python legged_lab/scripts/play_mimic.py --task g1_mimic_stairs_up
    python legged_lab/scripts/play_mimic.py --task g1_mimic_stairs_down --export
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "rsl_rl")]
local_motion_deps = REPOSITORY_ROOT / ".mimic_deps"
if local_motion_deps.is_dir():
    sys.path.insert(2, str(local_motion_deps))

from isaaclab.app import AppLauncher


MIMIC_TASKS = ("g1_mimic_stairs_up", "g1_mimic_stairs_down")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=MIMIC_TASKS, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--load_run",
        type=Path,
        help="Run directory, either absolute or relative to logs/<task>.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint path or filename. By default, the highest-iteration checkpoint is used.",
    )
    parser.add_argument("--max_steps", type=int, help="Stop after this many 50 Hz policy steps.")
    parser.add_argument("--export", action="store_true", help="Export policy.pt, policy.onnx, and metadata.json.")
    parser.add_argument("--export_only", action="store_true", help="Export without stepping the simulation.")
    parser.add_argument(
        "--training_randomization",
        action="store_true",
        help="Keep observation noise and startup domain randomization during playback.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error("--num_envs must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max_steps must be positive")
    if args.headless and not args.export_only and args.max_steps is None:
        parser.error("Headless playback requires --max_steps (or use --export_only).")
    return args


def checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def resolve_checkpoint(task: str, load_run: Path | None, checkpoint: Path | None) -> Path:
    log_root = REPOSITORY_ROOT / "logs" / task
    if not log_root.is_dir():
        raise FileNotFoundError(f"Training log directory does not exist: {log_root}")

    run_dir = None
    if load_run is not None:
        run_dir = load_run.expanduser()
        if not run_dir.is_absolute():
            run_dir = log_root / run_dir
        run_dir = run_dir.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    if checkpoint is not None:
        candidate = checkpoint.expanduser()
        if candidate.is_absolute():
            candidates = [candidate]
        elif run_dir is not None:
            candidates = [run_dir / candidate]
        elif candidate.parent != Path("."):
            candidates = [REPOSITORY_ROOT / candidate, log_root / candidate]
        else:
            candidates = list(log_root.glob(f"*/{candidate.name}"))
        existing = [path.resolve() for path in candidates if path.is_file()]
        if not existing:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return max(existing, key=lambda path: (checkpoint_iteration(path), path.stat().st_mtime_ns))

    search_root = run_dir or log_root
    candidates = list(search_root.glob("model_*.pt")) if run_dir else list(search_root.glob("*/model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt checkpoint found below: {search_root}")
    return max(candidates, key=lambda path: (checkpoint_iteration(path), path.stat().st_mtime_ns)).resolve()


def tensor_values(value):
    import torch

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim > 1:
            value = value[0]
        return value.tolist()
    return value


def export_policy(runner, env, observation, checkpoint: Path, task: str):
    from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

    export_dir = checkpoint.parent / "exported"
    policy_nn = runner.alg.policy
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=str(export_dir), filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=str(export_dir), filename="policy.onnx")

    policy_obs = observation["policy"]
    observation_terms = [
        ("joint_pos_ref", 29),
        ("joint_vel_ref", 29),
        ("position_ref_robot_frame", 3),
        ("rotation_ref_tannorm_robot_frame", 6),
        ("base_lin_vel_body", 3),
        ("base_ang_vel_body", 3),
        ("joint_pos_relative_to_default", 29),
        ("joint_vel", 29),
        ("last_action", 29),
    ]
    offset = 0
    observation_layout = []
    for name, size in observation_terms:
        observation_layout.append({"name": name, "start": offset, "end": offset + size, "size": size})
        offset += size
    if offset != policy_obs.shape[-1]:
        raise RuntimeError(f"Documented observation size {offset} != runtime size {policy_obs.shape[-1]}")

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    action_term = base_env.action_manager.get_term("joint_pos")
    metadata = {
        "task": task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "motion_file": base_env.cfg.motion_file,
        "terrain": base_env.cfg.motion_terrain,
        "policy_hz": 1.0 / base_env.step_dt,
        "physics_dt_s": base_env.physics_dt,
        "decimation": base_env.cfg.decimation,
        "observation_dim": int(policy_obs.shape[-1]),
        "observation_layout": observation_layout,
        "action_dim": int(env.num_actions),
        "joint_names": list(robot.joint_names),
        "default_joint_pos": tensor_values(robot.data.default_joint_pos),
        "action_scale": tensor_values(action_term._scale),
        "action_offset": tensor_values(action_term._offset),
        "action_formula": "joint_position_target = action * action_scale + action_offset",
        "actor_normalizer_included_in_export": normalizer is not None,
        "sim2sim_note": (
            "The 160-D observation includes time-indexed motion-reference commands. A simulator runner must "
            "reproduce the reference clock, robot-frame position/rotation transforms, joint ordering, and PD model."
        ),
    }
    for name in ("joint_stiffness", "joint_damping", "joint_effort_limits", "joint_velocity_limits", "joint_armature"):
        if hasattr(robot.data, name):
            metadata[name] = tensor_values(getattr(robot.data, name))
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[EXPORT] TorchScript: {export_dir / 'policy.pt'}")
    print(f"[EXPORT] ONNX:        {export_dir / 'policy.onnx'}")
    print(f"[EXPORT] Metadata:    {export_dir / 'metadata.json'}")


def play(args, simulation_app):
    import torch
    from rsl_rl.runners import OnPolicyRunner

    from legged_lab.envs import task_registry
    from legged_lab.envs.g1.g1_mimic_config import configure_g1_motion_terrain
    from legged_lab.utils.mimic.vecenv_wrapper import MimicVecEnvWrapper

    checkpoint = resolve_checkpoint(args.task, args.load_run, args.checkpoint)
    env_cfg, agent_cfg = task_registry.get_cfgs(args.task)
    env_cfg, agent_cfg = env_cfg.copy(), agent_cfg.copy()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = agent_cfg.device = args.device
    env_cfg.seed = agent_cfg.seed
    configure_g1_motion_terrain(env_cfg, env_cfg.motion_terrain)

    if not args.training_randomization:
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.events.physics_material = None
        env_cfg.events.add_joint_default_pos = None
        env_cfg.events.base_com = None
        env_cfg.events.push_robot = None

    print(f"[INFO] Task:       {args.task}")
    print(f"[INFO] Checkpoint: {checkpoint}")
    print(f"[INFO] Terrain:    {env_cfg.motion_terrain}")
    print(f"[INFO] Motion:     {env_cfg.motion_file}")

    env_class = task_registry.get_task_class(args.task)
    raw_env = env_class(cfg=env_cfg, render_mode=None)
    try:
        env = MimicVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint), load_optimizer=False, map_location=agent_cfg.device)
        policy = runner.get_inference_policy(device=env.device)
        observation = env.get_observations()
        if tuple(observation["policy"].shape) != (args.num_envs, 160):
            raise RuntimeError(f"Unexpected policy observation shape: {tuple(observation['policy'].shape)}")
        if not torch.isfinite(observation["policy"]).all():
            raise RuntimeError("Initial policy observation contains NaN or Inf")

        if args.export or args.export_only:
            export_policy(runner, env, observation, checkpoint, args.task)
        if args.export_only:
            print("[PLAY PASS] Export completed; simulation stepping was skipped.")
            return

        steps = 0
        resets = 0
        while simulation_app.is_running() and (args.max_steps is None or steps < args.max_steps):
            with torch.inference_mode():
                actions = policy(observation)
                if not torch.isfinite(actions).all():
                    raise RuntimeError(f"Policy action contains NaN or Inf at step {steps}")
                observation, _, dones, _ = env.step(actions)
            resets += int(dones.sum().item())
            steps += 1
        print(f"[PLAY PASS] task={args.task}; steps={steps}; resets={resets}; checkpoint=model_{checkpoint_iteration(checkpoint)}.pt")
    finally:
        raw_env.close()


if __name__ == "__main__":
    args_cli = parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        play(args_cli, simulation_app)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close(wait_for_replicator=False)
