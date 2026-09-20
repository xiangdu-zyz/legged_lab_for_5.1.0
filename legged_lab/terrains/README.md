# G1 obstacle motion terrains

`g1_motion_terrains.py` contains two editable profiles matched to the clips in
`resource/g1/data/`. Both contain ground, six ascending treads in the +y direction,
and a top platform. The `179_305` motion ascends; the `405_581` motion starts on
the platform and descends in the -y direction.

These layouts are initial estimates from the G1 foot trajectories. The NPZ files
do not include the original terrain dimensions, so reference playback must be
used to finish calibration. Their top surfaces initially sit at 0.917 m and
0.918 m, respectively.

| Parameter | Meaning |
| --- | --- |
| `x_range` | Left/right bounds of the staircase in the motion's original axes |
| `step_edges_y` | Starting y-coordinate of each tread, ordered from low to high |
| `step_heights` | Absolute tread top heights above ground; these are not riser heights |
| `platform_end_y` | Far boundary of the top platform |
| `stair_offset_xy` | Horizontal translation of the staircase |
| `stair_yaw_deg` | Rotation about local (0, 0), applied before horizontal translation |
| `size` | Width and length of the complete terrain tile |

Edit `G1_OBSTACLES_179_305_CFG` or `G1_OBSTACLES_405_581_CFG` to adjust one clip's
profile. Editing the defaults of `G1MotionStairsCfg` changes both profiles except
for explicitly overridden fields. A profile is copied for each training run.

Inspect the reference motions with the GUI playback script. It writes recorded
poses directly and does not advance physics, so placement can be inspected even
without a trained policy:

```bash
python legged_lab/scripts/play_motion.py --clip=179_305 --paused
python legged_lab/scripts/play_motion.py --clip=405_581 --paused
```

The script locates the repository, vendored RSL-RL source, and default motion
directory from its own file path. It does not require setting `PYTHONPATH` or
installing this repository first. From another working directory, use the
absolute path to `play_motion.py` and the same arguments.

The window provides play/pause, frame stepping, a frame slider, playback speed,
and looping. Keyboard controls (with focus on the viewport) are Space to pause,
Left/Right for one frame, R/Home to return to frame zero, End for the last frame,
and Escape to exit. Seeking or stepping pauses playback. Omit `--paused` to play
immediately; use `--speed=0.5` for half speed. The terrain is selected using the
same filename mapping as training. Restart the script after editing a profile.

Colored points sample the bottoms of the foot collision capsules: green within
2 cm of the surface, red more than 2 cm below it, and yellow more than 2 cm above.
The GUI and console show each foot's minimum sampled vertical gap in meters.
Inspect support phases; a swing foot being above the surface is expected.
These geometric samples do not establish physical contact or exhaustively check
the foot's entire collision surface.

An alternative motion can be selected with `--motion_dir` and `--motion_file`.
Unknown filenames require `--terrain=plane`; the two named obstacle terrains
require their matching filenames. For a bounded full-clip smoke test, use
`--headless --once`. For a bounded paused GUI check, use
`--paused --max_steps=60`.

Use `--report=logs/motion_alignment.json` to save the displayed frames' foot gaps
and the frame with the lowest gap for each foot. Combine it with
`--headless --once` at the default speed to inspect every source frame. GUI
playback can skip frames at high speeds or when rendering is slow, and its
report only includes frames actually displayed.

The descending profile has its own tread boundaries, fitted to the complete
foot trajectory: `(2.08, 2.29, 2.54, 2.76, 3.03, 3.31)` m. Reusing the ascending
boundaries previously put the trailing heels inside the higher tread after the
feet had landed on the next tread. The former 0.76 m surface is now 0.756 m,
giving clearance for the right heel's lowest recorded position.

Full-clip URDF heel mesh vertex checks measured minimum clearances of +0.0027 m
for the left heel and +0.0012 m for the right heel. The complete capsule sample
check still has small forefoot deviations: approximately -0.0156 m on the left
at frame 104 and -0.0063 m on the right at frame 91. These geometric checks are
not physical contact measurements. Inspect the formerly affected frames:

```bash
python legged_lab/scripts/play_motion.py --clip=405_581 --start_frame=55 --paused
```

Also inspect frame 92 for the left heel, and frame 104 for the remaining
forefoot deviation. After any profile change, restart playback and check both
the initial frame and the complete motion.

Coordinates are local to each environment origin. The generator converts them
to Isaac Lab's positive tile coordinates internally; do not add half the tile
size yourself. Terrain origins remain at ground height, including the descending
clip. Its elevated starting pose is already present in the motion data.

From the repository root, launch either registered terrain mimic task. Each
selects its matching clip from `resource/g1/data/` automatically:

```bash
python legged_lab/scripts/train.py --task=g1_mimic_stairs_up --headless \
    --logger=tensorboard --num_envs=1024
python legged_lab/scripts/train.py --task=g1_mimic_stairs_down --headless \
    --logger=tensorboard --num_envs=1024
```

Logs and checkpoints are separated under `logs/g1_mimic_stairs_up/` and
`logs/g1_mimic_stairs_down/`. `--motion_dir` can override the dataset directory.
The registered tasks reject a motion filename or terrain belonging to the other
clip. Configurations are loaded lazily, so the motion kinematics dependency is
only needed for mimic training.

The general `g1_mimic` task also supports explicit clip selection:

```bash
python legged_lab/scripts/train.py --task=g1_mimic --headless --logger=tensorboard \
    --motion_dir=resource/g1/data \
    --motion_file=obstacles1_subject1_frames_179_305_retargeted.npz

python legged_lab/scripts/train.py --task=g1_mimic --headless --logger=tensorboard \
    --motion_dir=resource/g1/data \
    --motion_file=obstacles1_subject1_frames_405_581_retargeted.npz
```

The default `--terrain=auto` chooses the matching terrain. Explicit choices are
`--terrain=obstacles_179_305`, `--terrain=obstacles_405_581`, or `--terrain=plane`.
An explicit obstacle profile requires its matching motion filename, preventing
accidental pairing with the wrong clip. Select one clip per run; joint training
with a different terrain assigned to each sampled clip is not implemented.

Obstacle profiles initially start at frame zero, disable adaptive time sampling
and reset pose/velocity/joint perturbations, and disable pushes. This keeps
manual alignment reproducible. Initial motion velocities remain intact. Friction,
mass and default-joint randomization remain as configured by the mimic task.
To vary start times later, restore time sampling and its curriculum together in
`configure_g1_motion_terrain` in `legged_lab/envs/g1/g1_mimic_config.py` after
validating all sampled frames.

For a short integration smoke test, append
`--num_envs=4 --max_iterations=2 --smoke_test` to each command.
The existing `g1_flat` terrain is unchanged.
