""" Additinoal common termination functions that are not implemented in isaaclab. """

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

    from legged_lab.utils.motion_reference import MotionReferenceManager


def dataset_exhausted(
    env: ManagerBasedRLEnv,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    reset_without_notice: bool = False,
    print_reason: bool = False,
) -> torch.Tensor:
    """Check if the dataset is exhausted.

    Args:
        env: The environment object.
        reset_without_notice: whether to reset the environment without returning True.
    Returns:
        True if the dataset is exhausted, False otherwise.
    """
    motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]
    return_ = torch.logical_not(
        motion_reference.data.validity[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]
    )  # shape: [N,]
    if print_reason and return_.any():
        print("dataset_exhausted: ", return_.sum())
    if reset_without_notice:
        motion_reference.reset(env_ids=return_.nonzero(as_tuple=True)[0])
        return_[:] = False
    return return_


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_buffer: float = 3.0,
    print_reason: bool = False,
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return torch.zeros(
            (env.num_envs,), device=env.device, dtype=torch.bool
        )  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return_ = torch.logical_or(x_out_of_bounds, y_out_of_bounds)
        if print_reason and return_.any():
            print(f"The base is out of the terrain border:", return_.sum())
        return return_
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")
