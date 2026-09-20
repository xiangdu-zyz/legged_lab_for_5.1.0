"""Replay a G1 motion clip on its matching terrain to inspect foot placement.

Example: python legged_lab/scripts/play_motion.py --clip=179_305 --paused
"""

import argparse
import json
from pathlib import Path
import math
import sys
import time
import traceback

# Direct execution adds scripts/ to sys.path, so include the checkout and its
# vendored RSL-RL source before importing project modules.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "rsl_rl")]

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", choices=("179_305", "405_581"), default="179_305")
    parser.add_argument("--motion_dir", type=Path,
                        default=REPOSITORY_ROOT / "resource/g1/data")
    parser.add_argument("--motion_file", type=Path, help="Override --clip with a motion filename or absolute NPZ path.")
    parser.add_argument("--terrain", choices=("auto", "plane", "obstacles_179_305", "obstacles_405_581"),
                        default="auto")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--paused", action="store_true", help="Start paused for initial foot placement inspection.")
    parser.add_argument("--once", action="store_true", help="Exit after displaying the last frame.")
    parser.add_argument("--max_steps", type=int, help="Exit after this many display updates (for smoke tests).")
    parser.add_argument("--report_every", type=int, default=30, help="Print foot gaps every N displayed source frames.")
    parser.add_argument("--report", type=Path, help="Save sampled foot gaps and worst penetration frames as JSON.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be positive and finite")
    if args.report_every <= 0 or (args.max_steps is not None and args.max_steps <= 0):
        parser.error("--report_every and --max_steps must be positive")
    if args.headless and (args.paused or not args.once) and args.max_steps is None:
        parser.error("Headless playback requires --once or --max_steps; paused playback requires --max_steps")
    return args


class PlaybackUI:
    def __init__(self, controls, terrain_name):
        import carb.input
        import omni.appwindow
        import omni.ui as ui

        self.controls = controls
        self.syncing = False
        self.window = ui.Window("G1 motion / terrain inspection", width=480, height=250)
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label(f"Terrain: {terrain_name}")
                self.status = ui.Label("")
                with ui.HStack():
                    ui.Button("Play / Pause", clicked_fn=self.toggle_pause)
                    ui.Button("Previous", clicked_fn=lambda: controls.step(-1))
                    ui.Button("Next", clicked_fn=lambda: controls.step(1))
                    ui.Button("Restart", clicked_fn=lambda: controls.seek(0))
                self.frame_slider = ui.IntSlider(min=0, max=controls.num_frames - 1)
                self.frame_slider.model.set_value(controls.index)
                self.frame_slider.model.add_value_changed_fn(self.seek)
                with ui.HStack():
                    ui.Label("Speed", width=70)
                    speed = ui.FloatSlider(min=0.05, max=max(3.0, controls.speed))
                    speed.model.set_value(controls.speed)
                    speed.model.add_value_changed_fn(lambda model: setattr(controls, "speed", model.as_float))
                    ui.Label("Loop", width=40)
                    loop = ui.CheckBox(width=25)
                    loop.model.set_value(controls.loop)
                    loop.model.add_value_changed_fn(lambda model: setattr(controls, "loop", model.as_bool))
                self.gaps = ui.Label("")
                ui.Label("Sole samples: green +/-2cm; red below surface; yellow above.")
                ui.Label("Space: pause | Left/Right: frame | R: restart | Esc: exit")
        self.input = carb.input.acquire_input_interface()
        self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self.subscription = self.input.subscribe_to_keyboard_events(self.keyboard, self.key_event)

    def toggle_pause(self):
        self.controls.paused = not self.controls.paused

    def seek(self, model):
        if not self.syncing:
            self.controls.seek(model.as_int)

    def key_event(self, event, *unused):
        import carb.input

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            actions = {"SPACE": self.toggle_pause, "LEFT": lambda: self.controls.step(-1),
                       "RIGHT": lambda: self.controls.step(1), "R": lambda: self.controls.seek(0),
                       "HOME": lambda: self.controls.seek(0),
                       "END": lambda: self.controls.seek(self.controls.num_frames - 1),
                       "ESCAPE": lambda: setattr(self.controls, "quit", True)}
            if event.input.name in actions:
                actions[event.input.name]()
        return True

    def refresh(self, displayed_frame, gaps):
        self.syncing = True
        try:
            self.frame_slider.model.set_value(self.controls.index)
        finally:
            self.syncing = False
        state = "Paused" if self.controls.paused else "Playing"
        self.status.text = f"{state} | frame {displayed_frame}/{self.controls.num_frames - 1} | {self.controls.speed:.2f}x"
        self.gaps.text = f"Minimum sampled vertical gap: L {gaps[0]:+.3f}m / R {gaps[1]:+.3f}m"

    def close(self):
        self.input.unsubscribe_to_keyboard_events(self.keyboard, self.subscription)
        self.window.destroy()


def play(args, simulation_app):
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils.math import quat_apply

    from legged_lab.assets.unitree import G1_29DOF_TORSOBASE_POPSICLE_CFG
    from legged_lab.terrains import G1_MOTION_TERRAIN_FILES, make_g1_motion_terrain
    from legged_lab.utils.mimic.playback import (
        MotionClip, PlaybackControls, foot_collision_samples, terrain_surface_heights,
    )

    filename = args.motion_file or Path(f"obstacles1_subject1_frames_{args.clip}_retargeted.npz")
    motion = MotionClip.load(filename if filename.is_absolute() else args.motion_dir / filename)
    if not 0 <= args.start_frame < len(motion.joint_pos):
        raise ValueError(f"--start_frame must be between 0 and {len(motion.joint_pos) - 1}")
    expected = G1_MOTION_TERRAIN_FILES.get(motion.path.name)
    terrain_name = expected if args.terrain == "auto" else args.terrain
    if terrain_name is None:
        raise ValueError("Unknown motion filename. Select --terrain=plane or use a registered obstacle clip.")
    if terrain_name != "plane" and terrain_name != expected:
        raise ValueError("The selected terrain does not match this motion clip.")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args.device))
    viewer = None
    try:
        scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0)
        scene_cfg.robot = G1_29DOF_TORSOBASE_POPSICLE_CFG.copy()
        scene_cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"
        scene_cfg.robot.spawn.rigid_props.disable_gravity = True
        scene_cfg.robot.spawn.activate_contact_sensors = False
        generator = make_g1_motion_terrain(terrain_name) if terrain_name != "plane" else None
        profile = next(iter(generator.sub_terrains.values())) if generator else None
        scene_cfg.terrain = TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="generator" if generator else "plane",
            terrain_generator=generator, max_init_terrain_level=None, collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0,
            ),
        )
        scene_cfg.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)),
        )
        scene = InteractiveScene(scene_cfg)
        markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/SoleSamples",
            markers={name: sim_utils.SphereCfg(
                radius=0.009, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ) for name, color in (("near_surface", (0.1, 0.9, 0.2)),
                                 ("below_surface", (1.0, 0.1, 0.1)),
                                 ("above_surface", (1.0, 0.8, 0.1)))},
        ))
        sim.reset()
        robot = scene["robot"]
        origin = scene.env_origins[0]
        target = origin.cpu().numpy() + [-1.0, 2.4, 0.8]
        sim.set_camera_view(eye=target + [4.0, -4.5, 2.6], target=target)
        joint_order = motion.joint_indices(robot.joint_names)
        joints = torch.as_tensor(motion.joint_pos[:, joint_order], device=sim.device)
        poses = torch.as_tensor(np.concatenate((motion.base_pos_w, motion.base_quat_w), axis=1), device=sim.device)
        poses[:, :3] += origin
        zeros_root = torch.zeros(1, 6, device=sim.device)
        zeros_joints = torch.zeros(1, robot.num_joints, device=sim.device)
        foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
        foot_ids, _ = robot.find_bodies(list(foot_names), preserve_order=True)
        samples = foot_collision_samples(scene_cfg.robot.spawn.asset_path, foot_names)
        centers = [torch.as_tensor(c, device=sim.device) for c, _ in samples]
        radii = [torch.as_tensor(r, device=sim.device) for _, r in samples]
        controls = PlaybackControls(len(joints), motion.framerate, frame=float(args.start_frame),
                                    speed=args.speed, paused=args.paused, loop=not args.once)
        if not args.headless:
            viewer = PlaybackUI(controls, terrain_name)
        print(f"[INFO] {motion.path.name}: {len(joints)} frames @ {motion.framerate:g}fps; terrain={terrain_name}")
        print("[INFO] Reference poses are written directly; physics is not advanced.")
        print("[INFO] Foot gaps are sampled vertical distances. Swing feet may have positive gaps.")
        last_report = -1
        displayed = set()
        measurements = {}
        updates = 0
        previous = time.perf_counter()
        while simulation_app.is_running() and not controls.quit and not controls.finished:
            tick = time.perf_counter()
            seconds = 1 / 60 if args.headless else min(0.1, tick - previous)
            previous = tick
            frame = controls.index
            robot.write_root_pose_to_sim(poses[frame:frame + 1])
            robot.write_root_velocity_to_sim(zeros_root)
            robot.write_joint_state_to_sim(joints[frame:frame + 1], zeros_joints)
            sim.forward()
            scene.update(1 / 60)
            positions, colors, gaps = [], [], []
            for i, body_id in enumerate(foot_ids):
                quaternion = robot.data.body_link_quat_w[0, body_id].expand(len(centers[i]), -1)
                points = quat_apply(quaternion, centers[i]) + robot.data.body_link_pos_w[0, body_id]
                points[:, 2] -= radii[i]
                local = (points - origin).cpu().numpy()
                heights = terrain_surface_heights(local, profile)
                distances = local[:, 2] - heights
                if not np.isfinite(distances).all():
                    raise ValueError(f"Foot samples leave the terrain tile at frame {frame}")
                gaps.append(float(distances.min()))
                positions.append(points)
                colors.append(torch.as_tensor(np.where(distances < -0.02, 1,
                                                        np.where(distances > 0.02, 2, 0)), device=sim.device))
            markers.visualize(torch.cat(positions), marker_indices=torch.cat(colors))
            if viewer:
                viewer.refresh(frame, gaps)
            if frame != last_report and (controls.paused or frame % args.report_every == 0 or frame == len(joints) - 1):
                print(f"[FRAME {frame:4d}] t={frame / motion.framerate:.3f}s | sole gap L={gaps[0]:+.3f}m R={gaps[1]:+.3f}m")
                last_report = frame
            displayed.add(frame)
            measurements[frame] = gaps
            sim.render()
            updates += 1
            if args.max_steps and updates >= args.max_steps:
                break
            controls.advance(seconds)
            if not args.headless:
                time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - tick)))
        print(f"[PLAYBACK DONE] terrain={terrain_name}; displayed={len(displayed)}/{len(joints)} frames; updates={updates}")
        if measurements:
            worst = [{"frame": min(measurements, key=lambda f: measurements[f][i]),
                      "gap_m": min(values[i] for values in measurements.values())} for i in range(2)]
            for name, item in zip(("L", "R"), worst):
                print(f"[ALIGNMENT] {name}: lowest sampled gap={item['gap_m']:+.3f}m at frame {item['frame']}")
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps({
                    "motion": str(motion.path), "terrain": terrain_name,
                    "measurement": "Sampled vertical foot gaps; physical contact is not simulated.",
                    "num_frames": len(joints), "displayed_frames": len(displayed),
                    "worst_sampled_gaps": dict(zip(("left", "right"), worst)),
                    "frames": [{"frame": f, "time_s": f / motion.framerate,
                                "left_gap_m": measurements[f][0], "right_gap_m": measurements[f][1]}
                               for f in sorted(measurements)],
                }, indent=2) + "\n")
    finally:
        if viewer:
            viewer.close()
        if sim.cfg.create_stage_in_memory:
            import omni.physx

            omni.physx.get_physx_simulation_interface().detach_stage()
            sim.stop()
            sim.clear()
        sim.clear_all_callbacks()
        sim.clear_instance()


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
