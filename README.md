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


## G1 Assets and Training Tasks

The complete G1 resource directory and motor configurations are ported from
`/home/luvbot/InstinctLab` (commit `de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057`).
They live under `legged_lab/assets/unitree/`; the old G1 USD assets are replaced.
`G1_CFG` selects the torso-base simplified URDF with the source's delayed PD
motor configuration for `g1_flat` (and the retained `g1_rough`). The clog and
popsicle configurations, motor constants, symmetry mappings, and all source
G1 resource files are also available.

G1 BeyondMimic is integrated into `legged_lab/envs/g1/`, `legged_lab/mdp/mimic/`,
and `legged_lab/utils/motion_reference/`. It uses the popsicle URDF and the
source's BeyondMimic motor/action-scale configuration. Observations are
flattened for the bundled RSL-RL 3.0.1 PPO policy, and the source's single reward
group uses Isaac Lab's standard scalar reward manager. No separate InstinctLab
or Instinct-RL installation is required.

Install the additional motion kinematics dependency in your Isaac Lab environment:

```bash
python -m pip install pytorch-kinematics==0.10.0
```

Train with retargeted G1 motion data:

```bash
python legged_lab/scripts/train.py --task=g1_mimic --headless --logger=tensorboard \
    --motion_dir=/absolute/path/to/motions --motion_file=sprint1_subject2_retargetted.npz
```

Motion NPZ files must contain `framerate`, `joint_names`, `joint_pos`,
`base_pos_w`, and `base_quat_w` (quaternion order: w, x, y, z). Omitting
`--motion_file` loads all supported motions in the directory. These files must
be retargeted for the torso-base G1; human motion files are not interchangeable.

The two obstacle clips under `resource/g1/data/` select their editable stair
profiles automatically when passed with `--motion_file`. See
[G1 obstacle terrain calibration](legged_lab/terrains/README.md) for the layouts,
manual adjustment parameters, and commands. Other motion files retain the plane
terrain unless a terrain is explicitly selected.

Two registered mimic tasks bind the obstacle clips to their matching stair
profiles and use `resource/g1/data/` by default:

| Task | Motion frames | Terrain | Logs |
| --- | --- | --- | --- |
| `g1_mimic_stairs_up` | `179_305` | `obstacles_179_305` | `logs/g1_mimic_stairs_up/` |
| `g1_mimic_stairs_down` | `405_581` | `obstacles_405_581` | `logs/g1_mimic_stairs_down/` |

```bash
python legged_lab/scripts/train.py --task=g1_mimic_stairs_up --headless \
    --logger=tensorboard --num_envs=1024
python legged_lab/scripts/train.py --task=g1_mimic_stairs_down --headless \
    --logger=tensorboard --num_envs=1024
```

These tasks need no `--motion_file` or `--terrain`. Use `--motion_dir` to locate
copies of the same files elsewhere; conflicting clip or terrain overrides are
rejected. Terrain generation and reference playback share the editable profiles.
Finish terrain calibration before long training runs, especially for the
descending clip. `g1_mimic`, `g1_flat`, and `g1_rough` remain available.

To queue upstairs followed by downstairs training on one GPU, run this from
your Isaac Lab Python environment:

```bash
python legged_lab/scripts/queue_mimic.py --num_envs=4096 --max_iterations=10000 --detach
```

The background worker waits for each training process to exit before launching
the next. It prints its PID and queue directory under `logs/mimic_queue/`, where
`status.json`, `queue.log`, and each task's console log record progress. Models
and TensorBoard files retain their task-specific log folders. Completion requires
the final checkpoint; a failed task is recorded before the next task is attempted.

To inspect recorded motion poses on the matched terrain without a policy, run
`python legged_lab/scripts/play_motion.py --clip=179_305 --paused` or select
`--clip=405_581`. The playback window includes pause, frame stepping, a frame
slider, and colored foot samples for terrain alignment.

To play the trained policies in Isaac Sim, use the dedicated ManagerBased mimic
player. It selects the highest numbered checkpoint below the task log directory,
which is `model_9999.pt` for the completed 10,000-iteration runs:

```bash
python legged_lab/scripts/play_mimic.py --task=g1_mimic_stairs_up
python legged_lab/scripts/play_mimic.py --task=g1_mimic_stairs_down
```

Select a checkpoint explicitly with `--checkpoint=/absolute/path/model_9999.pt`,
or combine a run-directory name with `--load_run` and a checkpoint filename.
Playback uses one environment, disables observation/startup randomization, and
keeps the motion clock, termination conditions, and matched terrain active. A
bounded headless check can be run with `--headless --max_steps=100`.

Export the policy for a future sim-to-sim runner with:

```bash
python legged_lab/scripts/play_mimic.py --task=g1_mimic_stairs_up --headless --export_only
python legged_lab/scripts/play_mimic.py --task=g1_mimic_stairs_down --headless --export_only
```

Each run's `exported/` directory then contains `policy.pt`, `policy.onnx`, and
`metadata.json`. The metadata records the exact 160-element policy observation
layout, 29-joint order, default positions, action scale/offset, actuator values,
control rate, terrain, and motion file. The exports include the learned actor
observation normalizer.

There is currently no executable MuJoCo sim-to-sim runner in this repository.
The bundled `resources/unitree_g1/g1.xml` is not the model used for training: it
has 37 actuators including hand joints and only one torso joint, whereas this
policy uses the 29-DOF torso-base robot with three waist joints. A correct
MuJoCo port must first supply a matching 29-DOF MJCF and calibrated stair model,
then reproduce the 50 Hz reference clock and every observation transform listed
in `metadata.json`. Loading the exported network into the bundled XML would map
both observations and actions to the wrong joints.

Run a small integration smoke test with a synthetic standing reference:

```bash
python tools/create_g1_smoke_motion.py
python legged_lab/scripts/train.py --task=g1_mimic --headless --logger=tensorboard \
    --num_envs=16 --max_iterations=2 --motion_dir=logs/smoke_motion \
    --motion_file=standing_retargetted.npz --smoke_test
python legged_lab/scripts/train.py --task=g1_flat --headless --logger=tensorboard \
    --num_envs=16 --max_iterations=2 --run_name=asset_smoke
```

The mimic test checks finite observations and rewards, finite and updated PPO
parameters, episode resets, and checkpoint saving. The synthetic reference is
only for integration testing. Logs and models are saved under `logs/g1_mimic/`
and `logs/g1_flat/`. The migration smoke run on an RTX 4080 completed two PPO
iterations for each task; mimic also exercised timeouts and reference exhaustion.

For the local smoke run, dependencies were installed in `.mimic_deps/` rather
than changing the conda environment. The training script resolves the repository
and vendored RSL-RL from its own path and includes `.mimic_deps/` for mimic tasks
when present; no manual `PYTHONPATH` setup is needed. If Isaac Sim reports a
`CXXABI_1.3.15` error, launch with
`LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"` using the existing conda runtime.

播放上楼策略：

  python legged_lab/scripts/play_mimic.py \
      --task=g1_mimic_stairs_up

  播放下楼策略：

  python legged_lab/scripts/play_mimic.py \
      --task=g1_mimic_stairs_down

  脚本会自动选择对应任务中迭代次数最高的 checkpoint，目前两者都是 model_9999.pt。关闭 Isaac Sim 窗口或按 Ctrl+C 结束播放。

  无界面测试可以这样运行：

  python legged_lab/scripts/play_mimic.py \
      --task=g1_mimic_stairs_down \
      --headless \
      --max_steps=500

  如果只是看原始动作与地形是否对齐，不运行策略：

  # 上楼原始数据
  python legged_lab/scripts/play_motion.py --clip=179_305 --paused

  # 下楼原始数据
  python legged_lab/scripts/play_motion.py --clip=405_581 --paused

  sim2sim 前先导出：

  python legged_lab/scripts/play_mimic.py \
      --task=g1_mimic_stairs_up \
      --headless \
      --export_only

  python legged_lab/scripts/play_mimic.py \
      --task=g1_mimic_stairs_down \
      --headless \
      --export_only

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
