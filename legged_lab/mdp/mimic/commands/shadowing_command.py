from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm
from isaaclab.markers.visualization_markers import VisualizationMarkers

import legged_lab.utils.mimic.math as instinct_math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from legged_lab.utils.motion_reference import MotionReferenceManager

    from .commands_cfg import (PositionRefCommandCfg, RotationRefCommandCfg, JointPosRefCommandCfg, JointVelRefCommandCfg, ShadowingCommandBaseCfg)


class ShadowingCommandBase(CommandTerm):
    """ """

    cfg: ShadowingCommandBaseCfg

    def __init__(self, cfg: ShadowingCommandBaseCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # self.robot: Articulation = env.scene[cfg.asset_name]
        self._motion_reference: MotionReferenceManager = env.scene[cfg.motion_reference.name]
        self._command: torch.Tensor = None  # override in the child class # type: ignore
        self._mask: torch.Tensor = None  # override in the child class # type: ignore

    @property
    def command(self) -> torch.Tensor:
        """Return the command tensor (not flattened)."""
        if self.cfg.current_state_command:
            return torch.cat([self._command, self._current_state], dim=1)
        return self._command  # type: ignore

    @property
    def mask(self) -> torch.Tensor:
        """Return the mask for the command tensor.
        ## NOTE:
            The mask should return an additional ones in time dimension if the current_state_command is True.
        """
        raise NotImplementedError(
            "The mask property is not implemented in the base class.",
        )

    @property
    def ALL_INDICES(self):
        """A proxy property to access ALL_INDICES to read the target motion/command at current timestep."""
        return self._motion_reference.ALL_INDICES

    @property
    def aiming_frame_idx(self):
        """A proxy property to access the aiming frame index of the motion reference.
        ## NOTE:
            a fun feature, when aiming_frame_idx becomes -1 (invalid), command_term.command[ALL_INDICES, aiming_frame_idx]
            will return the last frame of the motion reference (aka current state).
        """
        return self._motion_reference.aiming_frame_idx

    @property
    def link_of_interests(self) -> Sequence[str] | None:
        """Return the link of interest in the motion reference."""
        return self._motion_reference.cfg.link_of_interests

    @property
    def time_to_aiming_frame(self) -> torch.Tensor:
        """A proxy property to access the time to the aiming frame of the motion reference."""
        return self._motion_reference.time_to_aiming_frame

    @property
    def _motion_reference_updated_env_ids(self) -> Sequence[int] | torch.Tensor:
        """Return the environment indices that need to be updated, considering the motion reference
        (as a sensor) might updates during simulation step, but each command term is updated once in
        each env step.
        """
        return torch.where(self._motion_reference.time_passed_from_update < (self._env.step_dt - 1e-6))[0]

    def _update_metrics(self):
        """Update the metrics based on the motion reference and time."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Does not depend on the command term. self._command always depend on motion reference."""
        pass

    def _update_command(self):
        """Update the command based on the motion reference and time."""
        self._motion_reference.data  # trigger the update of the motion reference data

        # update the real-time mode for the position
        if getattr(self.cfg, "realtime_mode", False):
            env_ids = self._motion_reference.ALL_INDICES
        else:
            env_ids = self._motion_reference_updated_env_ids
        if len(env_ids) > 0:
            self._update_command_by_env_ids(env_ids)
        if getattr(self, "_visualizer", None) is not None:
            self._compute_debug_vis_data()

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the command term for the given environment indices."""
        self._update_command_by_env_ids(env_ids)
        return {}  # no metrics to return, override if needed.

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """Update the command based on the motion reference and time, for the given environment indices."""
        raise NotImplementedError

    def _debug_vis_callback(self, event):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis and (getattr(self.cfg, "visualizer_cfg", None) is not None):
            if not hasattr(self, "_visualizer"):
                self._visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self._visualizer.set_visibility(True)
        else:
            if hasattr(self, "_visualizer"):
                self._visualizer.set_visibility(False)

    def _compute_debug_vis_data(self):
        """Compute and update the data for visualization."""
        pass


class PositionRefCommand(ShadowingCommandBase):
    cfg: PositionRefCommandCfg

    def __init__(self, cfg: PositionRefCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # update self.cfg.realtime_mode for backward compatibility
        if isinstance(self.cfg.realtime_mode, bool):
            self.cfg.realtime_mode = int(self.cfg.realtime_mode)
        # generate the command tensor buffer
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 3),
            device=self.device,
        )
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 2),  # (plane mask, height mask)
            device=self.device,
            dtype=torch.bool,
        )
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, 3),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, 2),  # (plane mask, height mask)
                device=self.device,
                dtype=torch.bool,
            )
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask[..., 0] = torch.logical_and(mask[..., 0], self._motion_reference.data.base_pos_plane_mask)
        mask[..., 1] = torch.logical_and(mask[..., 1], self._motion_reference.data.base_pos_height_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        if self.cfg.anchor_frame == "robot":
            anchor_pos_w_inv, anchor_quat_w_inv = math_utils.subtract_frame_transforms(
                self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w[env_ids],
                self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w[env_ids],
            )
        elif self.cfg.anchor_frame == "reference":
            anchor_pos_w_inv, anchor_quat_w_inv = math_utils.subtract_frame_transforms(
                self._motion_reference.reference_frame.base_pos_w[env_ids, 0],
                self._motion_reference.reference_frame.base_quat_w[env_ids, 0],
            )
        elif self.cfg.anchor_frame == "ref_rot_robot_pos":
            anchor_pos_w_inv, anchor_quat_w_inv = math_utils.subtract_frame_transforms(
                self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w[env_ids],
                self._motion_reference.data.base_quat_w[env_ids, 0],
            )
        elif self.cfg.anchor_frame == "ref_pos_robot_rot":
            anchor_pos_w_inv, anchor_quat_w_inv = math_utils.subtract_frame_transforms(
                self._motion_reference.data.base_pos_w[env_ids, 0],
                self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w[env_ids],
            )
        self._command[env_ids] = (
            math_utils.transform_points(
                self._motion_reference.data.base_pos_w[env_ids],
                anchor_pos_w_inv,
                anchor_quat_w_inv,
            )
            * torch.logical_or(
                self._motion_reference.data.base_pos_height_mask[env_ids],
                self._motion_reference.data.base_pos_plane_mask[env_ids],
            ).unsqueeze(-1)
            * self._mask[env_ids].any(dim=-1, keepdim=True)
        )

    def _debug_vis_callback(self, event):
        robot_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w
        robot_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w
        _vis_point_w: torch.tensor = math_utils.transform_points(
            self._command,  # (num_envs, num_frames, 3)
            robot_pos_w,
            robot_quat_w,
        )  # (num_envs, num_frames, 3)
        _scales = torch.ones_like(_vis_point_w)
        _scales[:, 1:] *= 0.5
        self._visualizer.visualize(
            translations=_vis_point_w.flatten(0, 1),
            scales=_scales.flatten(0, 1),
        )


class RotationRefCommand(ShadowingCommandBase):
    """The sequential command that only contains the rotation part of the pose command."""

    cfg: RotationRefCommandCfg

    def __init__(self, cfg: RotationRefCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class."""
        super().__init__(cfg, env)
        # update self.cfg.realtime_mode for backward compatibility
        if isinstance(self.cfg.realtime_mode, bool):
            self.cfg.realtime_mode = int(self.cfg.realtime_mode)
        # generate the command tensor buffer
        if self.cfg.rotation_mode == "quaternion":
            data_dims = (4,)
        elif self.cfg.rotation_mode in ["axis_angle", "euler"]:
            data_dims = (3,)
        elif self.cfg.rotation_mode == "tannorm":
            data_dims = (6,)
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )  # (n_envs, num_frames, 3 + (4 or 3))
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, 2),  # (orientation mask, heading mask)
            device=self.device,
            dtype=torch.bool,
        )
        # generate state buffer in case we need additional frame as a pseudo command frame
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, 2),  # (orientation mask, heading mask)
                device=self.device,
                dtype=torch.bool,
            )
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask[..., 0] = torch.logical_and(mask[..., 0], self._motion_reference.data.base_orientation_mask)
        mask[..., 1] = torch.logical_and(mask[..., 1], self._motion_reference.data.base_heading_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        _, robot_quat_w_inv = math_utils.subtract_frame_transforms(
            self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w[env_ids],
            self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w[env_ids],
        )

        if self.cfg.in_base_frame:
            quat_reference = math_utils.quat_mul(
                robot_quat_w_inv.unsqueeze(1).expand(-1, self._motion_reference.num_frames, -1),
                self._motion_reference.data.base_quat_w[env_ids],
            )
        else:
            quat_reference = self._motion_reference.data.base_quat_w[env_ids]

        if self.cfg.rotation_mode == "quaternion":
            self._command[env_ids] = quat_reference
        elif self.cfg.rotation_mode == "axis_angle":
            self._command[env_ids] = math_utils.axis_angle_from_quat(quat_reference)
        elif self.cfg.rotation_mode == "euler":
            quat_reference_ = quat_reference.flatten(0, 1)
            euler_ = torch.stack(math_utils.euler_xyz_from_quat(quat_reference_), dim=-1)
            self._command[env_ids] = euler_.view(self._command[env_ids].shape)
        elif self.cfg.rotation_mode == "tannorm":
            self._command[env_ids] = instinct_math_utils.quat_to_tan_norm(quat_reference)

        self._command[env_ids] *= torch.logical_or(
            self._motion_reference.data.base_orientation_mask[env_ids],
            self._motion_reference.data.base_heading_mask[env_ids],
        ).unsqueeze(-1) * self._mask[env_ids].any(dim=-1, keepdim=True)
        self._command[env_ids] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1)

        if self.cfg.current_state_command and self.cfg.in_base_frame:
            if self.cfg.rotation_mode == "quaternion":
                self._current_state[env_ids, :, 0] = 1.0
            elif self.cfg.rotation_mode in ["axis_angle", "euler"]:
                pass
            elif self.cfg.rotation_mode == "tannorm":
                self._current_state[env_ids, :, :] = 0
                self._current_state[env_ids, :, 0] = 1
                self._current_state[env_ids, :, 5] = 1
        if self.cfg.current_state_command and not self.cfg.in_base_frame:
            robot_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w[env_ids]
            if self.cfg.rotation_mode == "quaternion":
                self._current_state[env_ids, :, :] = robot_quat_w.unsqueeze(1)
            elif self.cfg.rotation_mode in "axis_angle":
                self._current_state[env_ids, :, :] = math_utils.axis_angle_from_quat(robot_quat_w).unsqueeze(1)
            elif self.cfg.rotation_mode == "euler":
                self._current_state[env_ids, :, :] = math_utils.euler_xyz_from_quat(robot_quat_w).unsqueeze(1)
            elif self.cfg.rotation_mode == "tannorm":
                self._current_state[env_ids, :, :] = instinct_math_utils.quat_to_tan_norm(robot_quat_w).unsqueeze(1)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "_vis_quat_reference_w"):
            return
        self._visualizer.visualize(
            translations=self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w,
            orientations=self._vis_quat_reference_w,
        )

    def _compute_debug_vis_data(self):
        root_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w

        # Since this command is defined in robot's local frame, we need to transform
        # the command data to the world frame. Even if it is not refreshed timely.
        aiming_frame_idx = self._motion_reference.aiming_frame_idx
        ALL_INDICES = self._motion_reference.ALL_INDICES
        keyframe_command = self._command[ALL_INDICES, aiming_frame_idx]
        if self.cfg.rotation_mode == "quaternion":
            quat_reference = keyframe_command[:]
        elif self.cfg.rotation_mode == "axis_angle":
            quat_reference = math_utils.quat_from_angle_axis(
                torch.norm(keyframe_command, dim=-1),
                keyframe_command,
            )
        elif self.cfg.rotation_mode == "euler":
            quat_reference = math_utils.quat_from_euler_xyz(
                keyframe_command[..., 0],
                keyframe_command[..., 1],
                keyframe_command[..., 2],
            )
        elif self.cfg.rotation_mode == "tannorm":
            quat_reference = instinct_math_utils.tan_norm_to_quat(keyframe_command)
        else:
            raise ValueError(f"Invalid rotation mode: {self.cfg.rotation_mode}")
        if self.cfg.in_base_frame:
            quat_reference = math_utils.quat_mul(
                root_quat_w,
                quat_reference,
            )
        self._vis_quat_reference_w = quat_reference


class JointPosRefCommand(ShadowingCommandBase):
    """ """

    cfg: JointPosRefCommandCfg

    def __init__(self, cfg: JointPosRefCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # generate the command tensor buffer
        data_dims = (self._motion_reference.data.joint_pos.shape[2],)
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, self._motion_reference.data.joint_pos.shape[2]),
            device=self.device,
            dtype=torch.bool,
        )  # (num_envs, num_frames, num_joints)
        # get and copy the default joint position
        # shape (num_envs, num_joints)
        self._default_joint_pos = self._env.scene[cfg.asset_cfg.name].data.default_joint_pos.clone()
        # generate state buffer in case we need additional frame as a pseudo command frame
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, self._motion_reference.data.joint_pos.shape[2]),
                device=self.device,
                dtype=torch.bool,
            )  # (num_envs, 1, num_joints)
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask = torch.logical_and(mask, self._motion_reference.data.joint_pos_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        self._command[env_ids, :] = self._motion_reference.data.joint_pos[env_ids] - self._default_joint_pos[
            env_ids
        ].unsqueeze(1)
        self._command[env_ids, :] *= self._motion_reference.data.joint_pos_mask[env_ids]
        self._command[env_ids, :] *= self._motion_reference.data.validity[env_ids].unsqueeze(-1)
        self._command[env_ids, :] *= self._mask[env_ids]
        if self.cfg.current_state_command:
            self._current_state[env_ids] = self._env.scene[self.cfg.asset_cfg.name].data.joint_pos[env_ids].unsqueeze(
                1
            ) - self._default_joint_pos[env_ids].unsqueeze(1)


class JointVelRefCommand(ShadowingCommandBase):
    """ """

    cfg: JointVelRefCommandCfg

    def __init__(self, cfg: JointVelRefCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # generate the command tensor buffer
        data_dims = (self._motion_reference.data.joint_vel.shape[2],)
        self._command = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, *data_dims),
            device=self.device,
        )
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, self._motion_reference.data.joint_vel.shape[2]),
            device=self.device,
            dtype=torch.bool,
        )  # (num_envs, num_frames, num_joints)
        # get and copy the default joint velocity
        # shape (num_envs, num_joints)
        self._default_joint_vel = self._env.scene[cfg.asset_cfg.name].data.default_joint_vel.clone()
        # generate state buffer in case we need additional frame as a pseudo command frame
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, self._motion_reference.data.joint_vel.shape[2]),
                device=self.device,
                dtype=torch.bool,
            )  # (num_envs, 1, num_joints)
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        mask = torch.logical_and(mask, self._motion_reference.data.joint_vel_mask)
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        self._command[env_ids, :] = self._motion_reference.data.joint_vel[env_ids] - self._default_joint_vel[
            env_ids
        ].unsqueeze(1)
        self._command[env_ids, :] *= (
            self._motion_reference.data.joint_vel_mask[env_ids]
            * self._motion_reference.data.validity[env_ids].unsqueeze(-1)
            * self._mask[env_ids]
        )
        if self.cfg.current_state_command:
            self._current_state[env_ids] = self._env.scene[self.cfg.asset_cfg.name].data.joint_vel[env_ids].unsqueeze(
                1
            ) - self._default_joint_vel[env_ids].unsqueeze(1)
