import os
from dataclasses import MISSING
from pathlib import Path

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp.mimic as instinct_mdp
import legged_lab.envs.g1.mimic_env_config as beyondmimic_cfg

##
# Pre-defined configs
##
from legged_lab.assets.unitree.unitree import (
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuators,
)
from legged_lab.utils.motion_reference import MotionReferenceManagerCfg
from legged_lab.utils.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from legged_lab.utils.motion_reference.utils import motion_interpolate_bilinear
from legged_lab.envs.base.base_env_config import BaseAgentCfg
from legged_lab.terrains import make_g1_motion_terrain

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG

# Configurable motion input; importing this module never writes temporary files.
MOTION_NAME = "G1Motion"


@configclass
class AmassMotionCfg(AmassMotionCfgBase):
    """AMASS motion configuration for BeyondMimic."""

    path = os.path.expanduser(os.environ.get("LEGGEDLAB_MOTION_DIR", "~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz"))
    retargetting_func = None
    filtered_motion_selection_filepath = os.environ.get("LEGGEDLAB_MOTION_SELECTION")
    motion_start_from_middle_range = [0.0, 0.8]
    motion_start_height_offset = 0.0
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontbackward"
    motion_bin_length_s = 1.0


motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    robot_model_path=G1_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference/torso_link",
    link_of_interests=[
        "pelvis",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    symmetric_augmentation_link_mapping=None,
    symmetric_augmentation_joint_mapping=None,
    symmetric_augmentation_joint_reverse_buf=None,
    frame_interval_s=0.0,
    update_period=0.02,
    num_frames=1,
    visualizing_robot_offset=(0.0, 1.5, 0.0),
    visualizing_robot_from="reference_frame",
    motion_buffers={
        MOTION_NAME: AmassMotionCfg(),
    },
    mp_split_method="Even",
)


@configclass
class G1BeyondMimicPlaneEnvCfg(beyondmimic_cfg.BeyondMimicEnvCfg):
    """G1 BeyondMimic plane environment configuration."""

    motion_file: str | None = None
    motion_terrain: str | None = None

    scene: beyondmimic_cfg.BeyondMimicSceneCfg = beyondmimic_cfg.BeyondMimicSceneCfg(
        num_envs=4096,
        robot=G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
        motion_reference=motion_reference_cfg,
    )

    def __post_init__(self):
        super().__post_init__()

        # add link_of_interests to the policy observation
        if self.observations.policy.__dict__.get("link_pos", None) is not None:
            self.observations.policy.link_pos.params["asset_cfg"].body_names = (
                self.scene.motion_reference.link_of_interests
            )
        if self.observations.policy.__dict__.get("link_rot", None) is not None:
            self.observations.policy.link_rot.params["asset_cfg"].body_names = (
                self.scene.motion_reference.link_of_interests
            )
        if hasattr(self.observations, "critic") and self.observations.critic is not None:
            if self.observations.critic.__dict__.get("link_pos", None) is not None:
                self.observations.critic.link_pos.params["asset_cfg"].body_names = (
                    self.scene.motion_reference.link_of_interests
                )
            if self.observations.critic.__dict__.get("link_rot", None) is not None:
                self.observations.critic.link_rot.params["asset_cfg"].body_names = (
                    self.scene.motion_reference.link_of_interests
                )

        self.scene.robot.actuators = beyondmimic_g1_29dof_actuators
        self.actions.joint_pos.scale = beyondmimic_action_scale

        assert (
            len(list(self.scene.motion_reference.motion_buffers.keys())) == 1
        ), "Only support single motion buffer for now"

        # self.events.push_robot = None
        # self.rewards.undesired_contacts = None

        self.run_name: str = "".join(
            [
                "G1BeyondMimic",
                (
                    "_linVelObs"
                    if getattr(self.observations.policy, "base_lin_vel", None) is not None
                    and self.observations.policy.base_lin_vel.scale != 0.0
                    else ""
                ),
                f"_{MOTION_NAME}",
                ("_noPush" if self.events.push_robot is None else ""),
                ("_noContactPenalty" if getattr(self.rewards, "undesired_contacts", None) is None else ""),
                "_GmrMotion",
            ]
        )


def configure_g1_motion_terrain(env_cfg, terrain_name):
    """Use a fixed obstacle layout and repeatable reference resets for calibration."""
    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = make_g1_motion_terrain(terrain_name, env_cfg.scene.num_envs)
    env_cfg.scene.terrain.max_init_terrain_level = None
    motion_cfg = next(iter(env_cfg.scene.motion_reference.motion_buffers.values()))
    motion_cfg.motion_start_from_middle_range = [0.0, 0.0]
    motion_cfg.motion_bin_length_s = None
    env_cfg.curriculum.beyond_adaptive_sampling = None
    env_cfg.events.bin_fail_counter_smoothing = None
    reset = env_cfg.events.reset_robot.params
    reset["randomize_pose_range"] = {}
    reset["randomize_velocity_range"] = {}
    reset["randomize_joint_pos_range"] = (0.0, 0.0)
    env_cfg.events.push_robot = None
    env_cfg.run_name = terrain_name


@configclass
class G1BeyondMimicStairsEnvCfg(G1BeyondMimicPlaneEnvCfg):
    """One obstacle clip bound to its matching editable terrain."""

    motion_file: str = MISSING
    motion_terrain: str = MISSING

    def __post_init__(self):
        super().__post_init__()
        motion_cfg = next(iter(self.scene.motion_reference.motion_buffers.values()))
        default_dir = Path(__file__).resolve().parents[3] / "resource/g1/data"
        motion_cfg.path = os.path.expanduser(os.environ.get("LEGGEDLAB_MOTION_DIR", str(default_dir)))
        motion_cfg.filtered_motion_selection_filepath = None
        motion_cfg.selected_files = [self.motion_file]
        configure_g1_motion_terrain(self, self.motion_terrain)


@configclass
class G1BeyondMimicStairsUpEnvCfg(G1BeyondMimicStairsEnvCfg):
    motion_file = "obstacles1_subject1_frames_179_305_retargeted.npz"
    motion_terrain = "obstacles_179_305"


@configclass
class G1BeyondMimicStairsDownEnvCfg(G1BeyondMimicStairsEnvCfg):
    motion_file = "obstacles1_subject1_frames_405_581_retargeted.npz"
    motion_terrain = "obstacles_405_581"


@configclass
class G1MimicAgentCfg(BaseAgentCfg):
    max_iterations = 30000
    save_interval = 1000

    def __post_init__(self):
        super().__post_init__()
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
