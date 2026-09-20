"""Motion loading, playback controls, and terrain measurements without simulator imports."""

from dataclasses import dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np


@dataclass
class MotionClip:
    path: Path
    framerate: float
    joint_names: list[str]
    joint_pos: np.ndarray
    base_pos_w: np.ndarray
    base_quat_w: np.ndarray

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        with np.load(path, allow_pickle=False) as data:
            required = {"framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{path.name}: missing fields {sorted(missing)}")
            fps = float(data["framerate"].item())
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in data["joint_names"].tolist()]
            joints, positions, quaternions = (
                np.asarray(data[key], dtype=np.float32) for key in ("joint_pos", "base_pos_w", "base_quat_w")
            )
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Motion framerate must be positive and finite.")
        if joints.ndim != 2 or joints.shape[1] != len(names) or joints.shape[0] == 0:
            raise ValueError("joint_pos must have shape (frames, joint_names), with at least one frame.")
        if len(set(names)) != len(names):
            raise ValueError("Motion joint names must be unique.")
        if positions.shape != (len(joints), 3) or quaternions.shape != (len(joints), 4):
            raise ValueError("Root poses must have shape (frames, 3) and (frames, 4), using wxyz quaternions.")
        if not all(np.isfinite(a).all() for a in (joints, positions, quaternions)):
            raise ValueError("Motion contains non-finite poses.")
        norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
        if (norms < 1e-8).any():
            raise ValueError("Motion contains a zero root quaternion.")
        return cls(path, fps, names, joints, positions, quaternions / norms)

    def joint_indices(self, robot_names):
        missing = set(robot_names).difference(self.joint_names)
        if missing:
            raise ValueError(f"Motion is missing robot joints: {sorted(missing)}")
        return [self.joint_names.index(name) for name in robot_names]


@dataclass
class PlaybackControls:
    num_frames: int
    framerate: float
    frame: float = 0.0
    speed: float = 1.0
    paused: bool = False
    loop: bool = True
    finished: bool = False
    quit: bool = False

    @property
    def index(self):
        return min(self.num_frames - 1, max(0, int(self.frame + 1e-6)))

    def seek(self, frame):
        self.frame = float(min(self.num_frames - 1, max(0, int(frame))))
        self.paused = True
        self.finished = False

    def step(self, direction):
        self.seek(self.index + direction)

    def advance(self, seconds):
        if self.paused:
            return
        next_frame = self.frame + seconds * self.framerate * self.speed
        if next_frame >= self.num_frames:
            if self.loop:
                self.frame = next_frame % self.num_frames
            elif self.index == self.num_frames - 1:
                self.finished = True
            else:
                self.frame = float(self.num_frames - 1)
        else:
            self.frame = next_frame


def terrain_surface_heights(points, profile):
    """Vertical surface heights for the same floor/treads used by the mesh generator."""
    points = np.asarray(points)
    heights = np.zeros(points.shape[0], dtype=np.float32)
    if profile is None:
        return heights
    angle = math.radians(profile.stair_yaw_deg)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    local = (points[:, :2] - profile.stair_offset_xy) @ rotation
    indices = np.searchsorted(profile.step_edges_y, local[:, 1], side="right") - 1
    on_stairs = ((local[:, 0] >= profile.x_range[0]) & (local[:, 0] <= profile.x_range[1])
                 & (indices >= 0) & (local[:, 1] < profile.platform_end_y))
    heights[on_stairs] = np.asarray(profile.step_heights)[indices[on_stairs]]
    outside = np.any(np.abs(points[:, :2]) > np.asarray(profile.size) / 2, axis=1)
    heights[outside] = np.nan
    return heights


def foot_collision_samples(urdf_path, foot_names):
    """Sample bottoms of the foot capsules created from this G1 URDF's cylinders.

    Each capsule's endpoints and center are sampled. These vertical distances
    help inspect placement; they do not replace physical contact detection.
    """
    urdf = ET.parse(urdf_path).getroot()
    result = []
    for name in foot_names:
        link = urdf.find(f"link[@name='{name}']")
        if link is None:
            raise ValueError(f"URDF has no foot link {name}")
        centers, radii = [], []
        for collision in link.findall("collision"):
            cylinder = collision.find("geometry/cylinder")
            if cylinder is None:
                raise ValueError(f"Unsupported foot collision geometry in {name}; expected G1 cylinders.")
            origin = collision.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            roll, pitch, yaw = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            # Rz(yaw) Ry(pitch) Rx(roll) applied to the cylinder's local z-axis.
            axis = np.array([math.cos(yaw) * math.sin(pitch) * math.cos(roll) + math.sin(yaw) * math.sin(roll),
                             math.sin(yaw) * math.sin(pitch) * math.cos(roll) - math.cos(yaw) * math.sin(roll),
                             math.cos(pitch) * math.cos(roll)])
            for fraction in (-0.5, 0.0, 0.5):
                centers.append(xyz + fraction * float(cylinder.get("length")) * axis)
                radii.append(float(cylinder.get("radius")))
        if not centers:
            raise ValueError(f"No collision samples found for {name}")
        result.append((np.asarray(centers, dtype=np.float32), np.asarray(radii, dtype=np.float32)))
    return result
