from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

import legged_lab.utils.motion_reference.utils as motion_reference_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from legged_lab.utils.motion_reference import MotionReferenceManager


def pos_far_from_ref(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    distance_threshold: float = 0.5,
    check_at_keyframe_threshold: float = 0.01,
    print_reason: bool = False,
    height_only: bool = False,
) -> torch.Tensor:
    """Check if the position is far from the reference position.

    Args:
        env: The environment object.
        distance_threshold: The distance threshold.
        check_at_keyframe: only check when the time is close to the keyframe.
        height_only: only check the height (z-axis) difference.

    Returns:
        True if the position is far from the reference position, False otherwise.
    """
    distance = motion_reference_utils.get_base_position_distance(
        env,
        asset_cfg,
        reference_cfg,
        masking=True,
        squared=False,
        return_diff=height_only,
    )
    if height_only:
        distance = distance[..., 2].abs()  # only check z-axis
    distance = motion_reference_utils.matching_reference_timing(
        env,
        distance,
        env.scene[reference_cfg.name],
        check_at_keyframe_threshold=check_at_keyframe_threshold,
        multiply_by_frame_interval=False,
    )

    return_ = distance > distance_threshold
    if print_reason and return_.any():
        motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]
        env_ids = torch.where(return_)[0]
        motion_identifiers = motion_reference.get_current_motion_identifiers(env_ids=env_ids)
        print(f"pos_far_from_ref: {return_.sum()} at {motion_identifiers}")
    return return_


def link_pos_far_from_ref(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    distance_threshold: float = 0.5,
    in_base_frame: bool = False,
    check_at_keyframe_threshold: float = 0.01,
    height_only: bool = False,  # recommend using with `in_base_frame=False`
    print_reason: bool = False,
) -> torch.Tensor:
    distance = motion_reference_utils.get_link_position_distance(
        env,
        asset_cfg,
        reference_cfg,
        masking=True,
        squared=False,
        in_base_frame=in_base_frame,
        return_diff=height_only,
    )  # (batch_size, num_links)
    if height_only:
        distance = distance[..., 2].abs()  # only check z-axis
    distance = distance.max(-1).values
    distance = motion_reference_utils.matching_reference_timing(
        env,
        distance,
        env.scene[reference_cfg.name],
        check_at_keyframe_threshold=check_at_keyframe_threshold,
        multiply_by_frame_interval=False,
    )

    return_ = distance > distance_threshold
    if print_reason and return_.any():
        motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]
        env_ids = torch.where(return_)[0]
        motion_identifiers = motion_reference.get_current_motion_identifiers(env_ids=env_ids)
        print(f"link_pos_far_from_ref: {return_.sum()} at {motion_identifiers}")
    return return_


def projected_gravity_far_from_ref(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    projected_gravity_threshold: float = 0.5,
    check_at_keyframe_threshold: float = 0.01,
    z_only: bool = True,
    print_reason: bool = False,
) -> torch.Tensor:
    """Check if the projected gravity is far from the reference projected gravity."""

    robot: Articulation = env.scene[asset_cfg.name]
    motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]

    # get the current rotation
    rot = robot.data.root_state_w[:, 3:7]  # shape: [N, 4]
    ref_rot = motion_reference.data.base_quat_w  # shape: [N, n_frames, 4]
    ref_rot = ref_rot[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]

    pg = math_utils.quat_apply_inverse(rot, robot.data.GRAVITY_VEC_W)
    ref_pg = math_utils.quat_apply_inverse(ref_rot, robot.data.GRAVITY_VEC_W)

    if z_only:
        diff = (pg[:, 2] - ref_pg[:, 2]).abs()
    else:
        diff = torch.norm(pg - ref_pg, dim=-1)

    if check_at_keyframe_threshold >= 0:
        near_keyframe = (motion_reference.time_to_aiming_frame <= check_at_keyframe_threshold).float()
        diff = diff * near_keyframe

    return_ = diff > projected_gravity_threshold
    if print_reason and return_.any():
        env_ids = torch.where(return_)[0]
        motion_identifiers = motion_reference.get_current_motion_identifiers(env_ids=env_ids)
        print(f"projected_gravity_far_from_ref: {return_.sum()} at {motion_identifiers}")
    return return_
