from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

import legged_lab.utils.mimic.math as instinct_math

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg


def link_pos_b(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), in_base_frame: bool = True
) -> torch.Tensor:
    """The link positions in the robot base frame.
    Returns:
        (num_envs, num_links, 3)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    if in_base_frame:
        link_pos = math_utils.transform_points(
            link_pos_w,
            *math_utils.subtract_frame_transforms(
                asset.data.root_link_pos_w,
                asset.data.root_link_quat_w,
            ),
        )
    else:
        link_pos = link_pos_w
    return link_pos


def link_tannorm_b(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), in_base_frame: bool = True
) -> torch.Tensor:
    """The link orientations in tangent-normal representation in the robot base frame.
    Returns:
        (num_envs, num_links, 6)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    link_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    if in_base_frame:
        link_quat = math_utils.quat_mul(
            math_utils.quat_inv(asset.data.root_link_quat_w).unsqueeze(1).expand(-1, link_quat_w.shape[1], -1),
            link_quat_w,
        )
    else:
        link_quat = link_quat_w
    link_tannorm = instinct_math.quat_to_tan_norm(link_quat)
    return link_tannorm
