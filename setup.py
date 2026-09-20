# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

from setuptools import find_packages, setup

setup(
    name="LeggedLab",
    packages=find_packages(),
    version="1.0.0",
    package_data={
        "legged_lab.assets.unitree": [
            "resources/unitree_g1/*",
            "resources/unitree_g1/meshes/*",
            "resources/unitree_g1/urdf/*",
            "resources/unitree_g1/launch/*",
        ],
        "legged_lab.utils.motion_reference": ["LICENSE"],
    },
    extras_require={"mimic": ["pytorch-kinematics==0.10.0"]},
    python_requires=">=3.11,<3.12",
    install_requires=[
        # Isaac Sim is installed separately; the Isaac Lab pip meta-package is
        # pinned here for the supported binary installation.
        "IsaacLab==2.3.0",
        "rsl-rl-lib==3.0.1",
        "tensordict>=0.7.0",
    ],
)
