from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, List, Literal

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .shadowing_command import (PositionRefCommand, RotationRefCommand, JointPosRefCommand, JointVelRefCommand, ShadowingCommandBase)


@configclass
class ShadowingCommandBaseCfg(CommandTermCfg):
    """Command space for the MDP."""

    class_type: type = ShadowingCommandBase

    motion_reference: SceneEntityCfg = SceneEntityCfg("motion_reference")
    """ The configuration for accessing motion reference """

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    """ The configuration for accessing the robot asset and its states. """

    resampling_time_range: List[float] = [1e4, 1e5]

    current_state_command: bool = False
    """ Whether to include the current state in the command space. If True,
    the command sequence will be 1-frame longer than the motion reference.
    """


@configclass
class PositionRefCommandCfg(ShadowingCommandBaseCfg):
    """Command for base position reference, w.r.t current robot/reference position."""

    class_type: type = PositionRefCommand

    anchor_frame: Literal["reference", "robot", "ref_rot_robot_pos", "ref_pos_robot_rot"] = "robot"
    """To generate local position reference, the reference can be represented in either the robot's local frame or the
    reference motion's local frame. The reference motion's local frame is read from the motion reference.
    """

    realtime_mode: bool = False
    """ Since this reference is represented in robot's local-frame, the reference can be updated
    in real-time or not.
    """

    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PositionRefCommand",
        markers={
            "point": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
        },
    )


@configclass
class RotationRefCommandCfg(ShadowingCommandBaseCfg):
    class_type: type = RotationRefCommand

    rotation_mode: Literal["quaterion", "axis_angle", "euler", "tannorm"] = "axis_angle"
    """ n_dims:
        - if rotation_mode is "quaternion", data_dim = 4
        - if rotation_mode is "axis_angle", data_dim = 3
        - if rotation_mode is "euler", data_dim = 3
        - if rotation_mode is "tannorm", data_dim = 6, where 6 consists of tangent and normal vectors
    """

    in_base_frame: bool = True
    """ Whether to represent the rotation in the robot base frame or or in the world frame.
    """

    realtime_mode: bool = False
    """ If this reference is represented in robot's local-frame (in_base_frame == True), the reference
    can be updated in real-time or not.
    """

    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/RotationRefCommand",
        markers={
            "pose": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.15, 0.15, 0.15),
            ),
        },
    )


@configclass
class JointPosRefCommandCfg(ShadowingCommandBaseCfg):
    class_type: type = JointPosRefCommand
    """To get the robot's default joint positions."""


@configclass
class JointVelRefCommandCfg(ShadowingCommandBaseCfg):
    class_type: type = JointVelRefCommand
