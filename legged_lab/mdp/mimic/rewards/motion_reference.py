from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

import legged_lab.utils.motion_reference.utils as motion_reference_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject

    from legged_lab.utils.motion_reference import MotionReferenceManager

"""
Reward functions
"""


def base_position_imitation_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    std: float = 0.1,
):
    motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]
    base_pos_ref = motion_reference.reference_frame.base_pos_w[:, 0]  # (batch_size, 3)
    base_pos = robot.data.root_pos_w  # (batch_size, 3)
    base_pos_diff = base_pos - base_pos_ref  # (batch_size, 3)
    rewards = torch.exp(-torch.sum(torch.square(base_pos_diff), dim=-1) / (std * std))  # (batch_size,)
    # Apply validity mask from the reference frame
    rewards *= motion_reference.reference_frame.validity[:, 0]  # (batch_size,)
    return rewards


def base_rot_imitation_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    difference_type: Literal["axis_angle", "box_minus"] = "axis_angle",
    std: float = 0.3,
):
    """Reward for base rotation imitation using a Gaussian kernel.
    This reward is designed to encourage the agent to match the base rotation of the reference motion in EVERY timestep.
    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
    """
    motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    quat_ref = motion_reference.reference_frame.base_quat_w[:, 0]  # (batch_size, 4)
    quat = asset.data.root_quat_w  # (batch_size, 4)
    if difference_type == "axis_angle":
        quat_diff = math_utils.quat_mul(quat_ref, math_utils.quat_conjugate(quat))  # (batch_size, 4)
        axisang = math_utils.axis_angle_from_quat(quat_diff)  # (batch_size, 3)
        rot_error = torch.norm(axisang, dim=-1)  # (batch_size,)
    elif difference_type == "box_minus":
        quat_box_minus = math_utils.quat_box_minus(quat_ref, quat)
        rot_error = torch.norm(quat_box_minus, dim=-1)  # (batch_size,)
    else:
        raise ValueError(f"Unsupported difference method: {difference_type}.")

    rewards = torch.exp(-torch.square(rot_error) / (std * std))

    return rewards


def link_pos_imitation_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    in_base_frame: bool = True,
    in_relative_world_frame: bool = False,
    std: float = 0.1,
    masked: bool = False,
    combine_method: Literal["prod", "sum", "mean_prod"] = "prod",
):
    """
    Reward for link position imitation using a Gaussian kernel.
    This reward is designed to encourage the agent to match the link positions of the reference motion in EVERY timestep.

    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
        in_base_frame: Whether to compute the link position in the base frame of the robot.
        in_relative_world_frame: Whether to compute the link position difference between the reference link pose,
            which are moved to robot's relative position.
        masked: If True, only compute rewards for links that are not masked in the reference motion.
    """
    motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    links = asset.find_bodies(motion_reference.cfg.link_of_interests, preserve_order=True)
    link_indices = links[0]
    links_pos_w = asset.data.body_link_pos_w[:, link_indices]  # (batch_size, num_links, 3)
    if in_base_frame:
        root_pos_w = asset.data.root_pos_w
        root_quat_w = asset.data.root_quat_w
        root_pos_w_inv, root_quat_w_inv = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w)
        links_pos = math_utils.transform_points(
            links_pos_w,
            root_pos_w_inv,
            root_quat_w_inv,
        )
        link_pos_ref = motion_reference.reference_frame.link_pos_b[:, 0]
    elif in_relative_world_frame:
        links_pos = links_pos_w
        link_pos_ref = motion_reference.reference_link_pos_relative_w
    else:
        links_pos = links_pos_w
        link_pos_ref = motion_reference.reference_frame.link_pos_w[:, 0]

    link_pos_diff = links_pos - link_pos_ref  # (batch_size, num_links, 3)
    link_pos_square = torch.sum(torch.square(link_pos_diff), dim=-1)  # (batch_size, num_links)
    if masked:
        # (batch_size, num_links_selected)
        link_mask = motion_reference.reference_frame.link_pos_mask[:, 0, reference_cfg.body_ids]
    else:
        link_mask = 1

    # select a subset of link_of_interests
    link_pos_square = link_pos_square[:, reference_cfg.body_ids]  # (batch_size, num_links_selected)

    if combine_method == "prod":
        # (batch_size, num_links_selected) -> (batch_size,)
        link_pos_square = torch.sum(link_pos_square * link_mask, dim=-1)
    if combine_method == "mean_prod":
        # (batch_size, num_links_selected) -> (batch_size,)
        link_pos_square = torch.mean(link_pos_square * link_mask, dim=-1)

    rewards = torch.exp(-link_pos_square / (std * std))

    if combine_method == "sum":
        # (batch_size, num_links_selected) -> (batch_size,)
        rewards = torch.sum(rewards * link_mask, dim=-1)

    return rewards


def link_rot_imitation_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    in_base_frame: bool = True,
    in_relative_world_frame: bool = False,
    std: float = 0.4,
    masked: bool = False,
    combine_method: Literal["prod", "sum", "mean_prod"] = "prod",
):
    """
    Reward for link rotation imitation using a Gaussian kernel.
    This reward is designed to encourage the agent to match the link rotations of the reference motion in EVERY timestep.
    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
        in_base_frame: Whether to compute the link rotation in the base frame of the robot.
        in_relative_world_frame: Whether to compute the link position difference between the reference link pose,
            which are moved to robot's relative position.
        masked: If True, only compute rewards for links that are not masked in the reference motion.
    """
    motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    links = asset.find_bodies(motion_reference.cfg.link_of_interests, preserve_order=True)  # type: ignore
    link_indices = links[0]
    links_rot_w = asset.data.body_link_quat_w[:, link_indices]  # (batch_size, num_links, 4)
    if in_base_frame:
        root_quat_w = asset.data.root_quat_w
        root_quat_w_inv = math_utils.quat_inv(root_quat_w)  # (batch_size, 4)
        link_rot = math_utils.quat_mul(root_quat_w_inv.unsqueeze(1).expand(-1, links_rot_w.shape[1], -1), links_rot_w)
        link_rot_ref = motion_reference.reference_frame.link_quat_b[:, 0]
    elif in_relative_world_frame:
        link_rot = links_rot_w
        link_rot_ref = motion_reference.reference_link_quat_relative_w
    else:
        link_rot = links_rot_w
        link_rot_ref = motion_reference.reference_frame.link_quat_w[:, 0]
    link_rot_error_magnitude = math_utils.quat_error_magnitude(
        link_rot.reshape(-1, 4), link_rot_ref.reshape(-1, 4)
    ).reshape(
        link_rot.shape[0], link_rot.shape[1]
    )  # (batch_size, num_links)

    # select a subset of link_of_interests
    link_rot_error_magnitude = link_rot_error_magnitude[:, reference_cfg.body_ids]
    link_rot_error_square = torch.square(link_rot_error_magnitude)  # (batch_size, num_links_selected)
    if masked:
        # (batch_size, num_links_selected)
        link_rot_mask = motion_reference.reference_frame.link_rot_mask[:, 0, reference_cfg.body_ids]
    else:
        link_rot_mask = 1

    if combine_method == "prod":
        # (batch_size, num_links_selected) -> (batch_size,)
        link_rot_error_square = torch.sum(link_rot_error_square * link_rot_mask, dim=-1)
    if combine_method == "mean_prod":
        # (batch_size, num_links_selected) -> (batch_size,)
        link_rot_error_square = torch.mean(link_rot_error_square * link_rot_mask, dim=-1)

    rewards = torch.exp(-link_rot_error_square / (std * std))

    if combine_method == "sum":
        # (batch_size, num_links_selected) -> (batch_size,)
        rewards = torch.sum(rewards * link_rot_mask, dim=-1)

    return rewards


def link_lin_vel_imitation_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    in_base_frame: bool = False,
    std: float = 0.4,
    masked: bool = False,
    combine_method: Literal["prod", "sum", "mean_prod"] = "prod",
):
    """
    Reward for link linear velocity imitation using a Gaussian kernel.
    This reward is designed to encourage the agent to match the link linear velocities of the reference motion in EVERY timestep.
    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
        in_base_frame: Whether to compute the link linear velocity in the base frame of the robot.
        masked: If True, only compute rewards for links that are not masked in the reference motion.
    """
    motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    links = asset.find_bodies(motion_reference.cfg.link_of_interests, preserve_order=True)  # type: ignore
    link_indices = links[0]
    links_lin_vel_w = asset.data.body_link_lin_vel_w[:, link_indices]  # (batch_size, num_links, 3)
    if in_base_frame:
        root_pos_w = asset.data.root_pos_w
        root_quat_w = asset.data.root_quat_w
        root_lin_vel_w = asset.data.root_lin_vel_w
        root_ang_vel_w = asset.data.root_ang_vel_w
        links_pos_w = asset.data.body_link_pos_w[:, link_indices]  # (batch_size, num_links, 3)
        link_pos_offset_w = links_pos_w - root_pos_w.unsqueeze(1)  # (batch_size, num_links, 3)
        link_lin_vel = math_utils.quat_apply_inverse(
            root_quat_w.unsqueeze(1).expand(-1, links_lin_vel_w.shape[1], -1),
            (
                links_lin_vel_w
                - root_lin_vel_w.unsqueeze(1)
                - torch.cross(root_ang_vel_w.unsqueeze(1), link_pos_offset_w, dim=-1)
            ),
        )  # (batch_size, num_links, 3)
        link_lin_vel_ref = motion_reference.reference_frame.link_lin_vel_b[:, 0]
    else:
        link_lin_vel = links_lin_vel_w
        link_lin_vel_ref = motion_reference.reference_frame.link_lin_vel_w[:, 0]
    link_lin_vel_diff = link_lin_vel - link_lin_vel_ref  # (batch_size, num_links, 3)
    link_lin_vel_square = torch.sum(torch.square(link_lin_vel_diff), dim=-1)  # (batch_size, num_links)
    if masked:
        # (batch_size, num_links_selected)
        link_lin_vel_mask = motion_reference.reference_frame.link_pos_mask[:, 0, reference_cfg.body_ids]
    else:
        link_lin_vel_mask = 1

    # select a subset of link_of_interests
    link_lin_vel_square = link_lin_vel_square[:, reference_cfg.body_ids]  # (batch_size, num_links_selected)
    if combine_method == "prod":
        # (batch_size, num_links_selected) -> (batch_size,)
        link_lin_vel_square = torch.sum(link_lin_vel_square * link_lin_vel_mask, dim=-1)
    if combine_method == "mean_prod":
        # (batch_size, num_links_selected) -> (batch_size,)
        link_lin_vel_square = torch.mean(link_lin_vel_square * link_lin_vel_mask, dim=-1)

    rewards = torch.exp(-link_lin_vel_square / (std * std))

    if combine_method == "sum":
        # (batch_size, num_links_selected) -> (batch_size,)
        rewards = torch.sum(rewards * link_lin_vel_mask, dim=-1)

    return rewards


def link_ang_vel_imitation_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    in_base_frame: bool = False,
    std: float = 0.4,
    masked: bool = False,
    combine_method: Literal["prod", "sum", "mean_prod"] = "prod",
):
    """
    Reward for link angular velocity imitation using a Gaussian kernel.
    This reward is designed to encourage the agent to match the link angular velocities of the reference motion in EVERY timestep.
    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
        in_base_frame: Whether to compute the link angular velocity in the base frame of the robot.
        masked: If True, only compute rewards for links that are not masked in the reference motion.
    """
    motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    links = asset.find_bodies(motion_reference.cfg.link_of_interests, preserve_order=True)  # type: ignore
    link_indices = links[0]
    links_ang_vel_w = asset.data.body_link_ang_vel_w[:, link_indices]  # (batch_size, num_links, 3)
    if in_base_frame:
        root_quat_w = asset.data.root_quat_w
        root_ang_vel_w = asset.data.root_ang_vel_w
        link_ang_vel = math_utils.quat_apply_inverse(
            root_quat_w.unsqueeze(1).expand(-1, links_ang_vel_w.shape[1], -1),
            links_ang_vel_w - root_ang_vel_w.unsqueeze(1),
        )  # (batch_size, num_links, 3)
        link_ang_vel_ref = motion_reference.reference_frame.link_ang_vel_b[:, 0]
    else:
        link_ang_vel = links_ang_vel_w
        link_ang_vel_ref = motion_reference.reference_frame.link_ang_vel_w[:, 0]
    link_ang_vel_diff = link_ang_vel - link_ang_vel_ref  # (batch_size, num_links, 3)
    link_ang_vel_square = torch.sum(torch.square(link_ang_vel_diff), dim=-1)  # (batch_size, num_links)
    if masked:
        # (batch_size, num_links_selected)
        link_ang_vel_mask = motion_reference.reference_frame.link_rot_mask[:, 0, reference_cfg.body_ids]
    else:
        link_ang_vel_mask = 1

    # select a subset of link_of_interests
    link_ang_vel_square = link_ang_vel_square[:, reference_cfg.body_ids]  # (batch_size, num_links_selected)
    if combine_method == "prod":
        # (batch_size, num_links_selected) -> (batch_size,)
        link_ang_vel_square = torch.sum(link_ang_vel_square * link_ang_vel_mask, dim=-1)
    if combine_method == "mean_prod":
        # (batch_size, num_links_selected) -> (batch_size,)
        link_ang_vel_square = torch.mean(link_ang_vel_square * link_ang_vel_mask, dim=-1)

    rewards = torch.exp(-link_ang_vel_square / (std * std))

    if combine_method == "sum":
        # (batch_size, num_links_selected) -> (batch_size,)
        rewards = torch.sum(rewards * link_ang_vel_mask, dim=-1)

    return rewards
