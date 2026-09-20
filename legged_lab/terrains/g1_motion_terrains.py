"""Editable stairs for the two obstacle motion clips in resource/g1/data.

Coordinates are relative to each environment origin, retaining the NPZ's
original axes. Heights and boundaries are initial fits to the foot trajectories,
not ground-truth terrain metadata. Tune each profile against reference playback.
"""

import copy
import math

import numpy as np
import trimesh
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass


def g1_motion_stairs(difficulty: float, cfg: "G1MotionStairsCfg"):
    """Generate a ground slab and six static stair treads, in Isaac Lab coordinates."""
    del difficulty  # The calibrated layout must remain fixed across difficulty levels.
    edges = (*cfg.step_edges_y, cfg.platform_end_y)
    heights = cfg.step_heights
    if len(cfg.step_edges_y) != len(heights) or not heights:
        raise ValueError("Each step needs one starting y-coordinate and one height.")
    if not all(math.isfinite(v) for v in (*edges, *heights, *cfg.x_range,
                                         *cfg.stair_offset_xy, cfg.stair_yaw_deg,
                                         cfg.ground_thickness)):
        raise ValueError("Terrain dimensions and offsets must be finite.")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("Step boundaries and platform_end_y must be strictly increasing.")
    if any(h <= 0 for h in heights) or any(b <= a for a, b in zip(heights, heights[1:])):
        raise ValueError("Step heights must be positive and strictly increasing.")
    if cfg.x_range[1] <= cfg.x_range[0] or cfg.ground_thickness <= 0:
        raise ValueError("Stair width and ground thickness must be positive.")

    size = np.asarray(cfg.size, dtype=float)
    center = np.array([size[0] / 2, size[1] / 2, 0.0])
    floor = trimesh.creation.box(extents=(*size, cfg.ground_thickness))
    floor.apply_translation(center + [0.0, 0.0, -cfg.ground_thickness / 2])
    meshes = [floor]
    yaw = math.radians(cfg.stair_yaw_deg)
    transform = trimesh.transformations.rotation_matrix(yaw, [0.0, 0.0, 1.0])
    transform[:2, 3] = cfg.stair_offset_xy
    for y0, y1, height in zip(edges, edges[1:], heights):
        step = trimesh.creation.box(extents=(cfg.x_range[1] - cfg.x_range[0], y1 - y0, height))
        step.apply_translation([(cfg.x_range[0] + cfg.x_range[1]) / 2, (y0 + y1) / 2, height / 2])
        step.apply_transform(transform)
        # Reject a layout that would overlap an adjacent environment's terrain tile.
        if np.any(np.abs(step.vertices[:, :2]) > size / 2 + 1e-6):
            raise ValueError("Stairs exceed the terrain tile. Increase size or adjust the layout.")
        step.apply_translation(center)
        meshes.append(step)

    # Keep z=0 even for the descending clip, which starts on the high platform.
    # The motion already contains its elevated starting root pose.
    return meshes, center


@configclass
class G1MotionStairsCfg(SubTerrainBaseCfg):
    function = g1_motion_stairs
    size = (10.0, 12.0)
    x_range: tuple[float, float] = (-1.8, -0.35)
    step_edges_y: tuple[float, ...] = (1.94, 2.24, 2.48, 2.70, 2.96, 3.20)
    platform_end_y: float = 4.20
    step_heights: tuple[float, ...] = (0.16, 0.31, 0.465, 0.605, 0.765, 0.917)
    stair_offset_xy: tuple[float, float] = (0.0, 0.0)
    stair_yaw_deg: float = 0.0
    ground_thickness: float = 0.1


G1_OBSTACLES_179_305_CFG = G1MotionStairsCfg()
"""Ascending clip: starts on the ground and climbs to the platform."""

G1_OBSTACLES_405_581_CFG = G1MotionStairsCfg(
    # Descending feet face -y. Boundaries must clear the whole trailing heel,
    # rather than reusing the ascending clip's foot-center estimates.
    step_edges_y=(2.08, 2.29, 2.54, 2.76, 3.03, 3.31),
    # The right heel reaches 0.757 m on this tread; retain a small clearance.
    step_heights=(0.15, 0.30, 0.46, 0.595, 0.756, 0.918),
)
"""Descending clip: tread edges fitted to the complete foot and heel trajectory."""

G1_MOTION_TERRAIN_PROFILES = {
    "obstacles_179_305": G1_OBSTACLES_179_305_CFG,
    "obstacles_405_581": G1_OBSTACLES_405_581_CFG,
}
G1_MOTION_TERRAIN_FILES = {
    "obstacles1_subject1_frames_179_305_retargeted.npz": "obstacles_179_305",
    "obstacles1_subject1_frames_405_581_retargeted.npz": "obstacles_405_581",
}


def make_g1_motion_terrain(name: str, num_envs: int = 1) -> TerrainGeneratorCfg:
    """Build an independent profile copy so calibration cannot mutate another run."""
    profile = copy.deepcopy(G1_MOTION_TERRAIN_PROFILES[name])
    # Use several identical tiles for parallel environments, without terrain curriculum.
    columns = min(8, max(1, math.ceil(math.sqrt(num_envs))))
    rows = min(8, max(1, math.ceil(num_envs / columns)))
    return TerrainGeneratorCfg(
        seed=42,
        size=profile.size,
        num_rows=rows,
        num_cols=columns,
        border_width=0.0,
        curriculum=False,
        use_cache=False,
        color_scheme="height",
        sub_terrains={name: profile},
    )
