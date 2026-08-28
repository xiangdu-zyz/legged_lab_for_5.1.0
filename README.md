# Legged Lab: Direct IsaacLab Workflow for Legged Robots

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab/v2.3.0/)
[![RSL_RL](https://img.shields.io/badge/RSL_RL-3.0.1-silver)](https://github.com/leggedrobotics/rsl_rl/releases/tag/v3.0.1)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

## Overview

This repository provides a direct workflow for training a legged robot using IsaacLab. It provides high transparency and low refactoring difficulty of the direct environment, and uses isaaclab components to simplify the workflow.

It has all the necessary conditions for sim-to-real and has been tested on real unitree g1 and h1 robots, [video available](https://www.bilibili.com/video/BV1tNRgYQEnr/).
Deploy Code: https://github.com/Hellod035/LeggedLabDeploy

**Maintainer**: Wandong Sun
**Contact**: 24b908020@stu.hit.edu.cn

**Key Features:**

- `Easy to Reorganize` Provides a direct workflow, allowing for fine-grained definition of environment logic.
- `Isolation` Work outside the core Isaac Lab repository, ensuring that the development efforts remain self-contained.
- `Long-term support` This repository will be updated with the updates of isaac sim and isaac lab, and will be supported for a long time.



## Installation

This revision targets the following version combination:

- Isaac Sim 5.1.0
- Isaac Lab 2.3.0
- RSL-RL 3.0.1
- Python 3.11

Do not use Python 3.10 from the Isaac Sim 4.x environment, and do not install Isaac Lab from the current `main` branch. The recommended pip installation is:

```bash
conda create -n leggedlab python=3.11
conda activate leggedlab
python -m pip install --upgrade pip

python -m pip install "isaaclab[isaacsim,all]==2.3.0" --extra-index-url https://pypi.nvidia.com
python -m pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

For binary or source-based installations, follow the [Isaac Lab 2.3.0 installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/index.html) and check out the `v2.3.0` tag. When Isaac Lab is installed from source, its internal Python distribution uses a component version instead of `2.3.0`; install this project with `python -m pip install -e . --no-deps` after running `./isaaclab.sh -i rsl_rl`.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
# Change this to the parent directory where LeggedLab should live.
LEGGEDLAB_PARENT=/absolute/path/to/workspace
mkdir -p "$LEGGEDLAB_PARENT"
cd "$LEGGEDLAB_PARENT"

# Option 1: HTTPS
git clone https://github.com/xiangdu-zyz/legged_lab_for_5.1.0.git LeggedLab

# Option 2: SSH (use this instead of Option 1)
git clone git@github.com:xiangdu-zyz/legged_lab_for_5.1.0.git LeggedLab
```

### Editable RSL-RL Source Installation

This repository includes the complete RSL-RL 3.0.1 source under `rsl_rl/`, vendored from upstream commit `2fc1f78`. The `isaaclab[isaacsim,all]` command installs an RSL-RL wheel first; replace it with this editable checkout so local source changes are imported immediately:

```bash
LEGGEDLAB_PARENT=/absolute/path/to/workspace
cd "$LEGGEDLAB_PARENT/LeggedLab/rsl_rl"
python -m pip install --no-deps --force-reinstall -e .
```

Verify that Python imports the bundled source rather than a copied package in `site-packages`:

```bash
python -c "from pathlib import Path; import rsl_rl; print(Path(rsl_rl.__file__).resolve())"
python -m pip show rsl-rl-lib
```

The first command should print `/absolute/path/to/workspace/LeggedLab/rsl_rl/rsl_rl/__init__.py`. If `isaaclab.sh -i rsl_rl` or a later pip upgrade replaces it, rerun the editable installation command.

- Install LeggedLab without resolving dependencies again, which guarantees that editable RSL-RL is preserved:

```bash
cd "$LEGGEDLAB_PARENT/LeggedLab"
python -m pip install --no-deps -e .
```

- Confirm the selected package versions:

```bash
python -c "from importlib.metadata import version; print('Isaac Lab:', version('isaaclab')); print('Isaac Sim:', version('isaacsim')); print('RSL-RL:', version('rsl-rl-lib'))"
```

- Verify that the extension is correctly installed by running the following command:

```bash
python legged_lab/scripts/train.py --task=g1_flat --headless --logger=tensorboard --num_envs=64
```


## Use Your Own Robot

Assets must be converted into USD format to be compatible with Legged Lab/IsaacLab. [Convert Tutorial](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/how-to/import_new_asset.html).


## Multi-GPU and Multi-Node Training

Legged Lab supports multi-GPU and multi-node reinforcement learning using rsl_rl, the usage is exactly the same as IsaacLab. [Detailed information](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/features/multi_gpu.html)

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}/legged_lab",
        "<path-to-IsaacLab>/source/isaaclab_tasks",
        "<path-to-IsaacLab>/source/isaaclab_mimic",
        "<path-to-IsaacLab>/source/isaaclab_assets",
        "<path-to-IsaacLab>/source/isaaclab_rl",
        "<path-to-IsaacLab>/source/isaaclab",
    ]
}
```

## References and Thanks
This project repository builds upon the shoulders of giants.
* [IsaacLab](https://github.com/isaac-sim/IsaacLab)   The various reusable practical components in IsaacLab greatly simplify the complexity of LeggedLab.
* [legged_gym](https://github.com/leggedrobotics/legged_gym)   We borrowed the code organization and environment definition logic of legged_gym and simplified it as much as possible.
* [Protomotions](https://github.com/NVlabs/ProtoMotions)   The motivation for building this repository comes from protomotions. For the first time, we realized that we could create our own environment using only IsaacLab components without inheriting 'DirectRLEnv' or 'ManagerBasedRLEnv'.

## Citation

If you use Legged Lab in your research, you can cite it as follows:

```bibtex
@software{LeggedLab,
  author = {Wandong, Sun},
  license = {BSD-3-Clause},
  title = {Legged Lab: Direct IsaacLab Workflow for Legged Robots},
  url = {https://github.com/Hellod035/LeggedLab},
  version = {1.0.0},
  year = {2025}
}
```
