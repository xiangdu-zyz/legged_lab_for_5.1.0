"""Generate a synthetic standing reference for integration testing, not training data."""

import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="logs/smoke_motion/standing_retargetted.npz")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    urdf = root / "legged_lab/assets/unitree/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf"
    names = [j.attrib["name"] for j in ET.parse(urdf).getroot().findall("joint") if j.attrib["type"] != "fixed"]
    assert len(names) == 29
    defaults = {
        ".*_hip_pitch_joint": -0.312, ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363, ".*_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2, "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2, "right_shoulder_pitch_joint": 0.2,
    }
    positions = np.zeros((250, len(names)), dtype=np.float32)
    for i, name in enumerate(names):
        for pattern, value in defaults.items():
            if re.fullmatch(pattern, name):
                positions[:, i] = value
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, framerate=50.0, joint_names=np.array(names), joint_pos=positions,
             base_pos_w=np.tile([0., 0., 0.82], (250, 1)).astype(np.float32),
             base_quat_w=np.tile([1., 0., 0., 0.], (250, 1)).astype(np.float32))
    print(output.resolve())


if __name__ == "__main__":
    main()
