# myCobot 280 eye-to-hand calibration — simulation and hardware

Recovers **where a fixed camera sits in the robot's base frame**, by moving an
ArUco target on the arm's flange through many poses and solving

```
T_ee_target = T_ee_base · T_base_cam · T_cam_target
```

for `T_base_cam`. The target is bolted to the flange, so its pose in the flange
frame is the same at every sample; the scatter of that quantity is the only
quality signal available on real hardware, and it is what everything here is
scored on.

The same code runs in simulation and on the physical rig. Only a small
parameter overlay differs.

---

## Quick start

### Simulation

```bash
# 1. the world, arm, camera and target
ros2 launch mycobot_gazebo mycobot_combined.launch.py \
    use_camera:=true use_aruco_marker:=true use_gripper:=false \
    world_file:=calibration.world use_rviz:=false

# 2. detector + pose sweep (~9 min of arm motion), writes a dataset
ros2 launch mycobot_calibration handeye_calibration.launch.py \
    use_sim_time:=true dataset_file:=results/my_dataset.json

# 3. solve
ros2 run mycobot_calibration handeye_solver \
    --dataset results/my_dataset.json --output results/my_result.yaml

# detection only, no arm motion
ros2 launch mycobot_calibration handeye_calibration.launch.py \
    use_sim_time:=true run_collector:=false
```

`scripts/run_demo.sh` walks all of it with narration. Use `--no-pause` to run
unattended.

### Hardware

```bash
# on the JetBot (camera host)
python3 jetbot_frame_server.py --device /dev/video3 --port 5555 --fps 5

# on the Orin
ros2 run mycobot_calibration jetbot_camera_bridge --ros-args -p host:=<jetbot-ip>

ros2 launch mycobot_gazebo mycobot_combined.launch.py \
    use_sim:=false use_rviz:=false use_gz_gui:=false \
    use_gripper:=false use_aruco_marker:=true

ros2 launch mycobot_calibration handeye_calibration.launch.py \
    use_sim_time:=false \
    overrides_file:=<share>/mycobot_calibration/config/hardware_overrides.yaml \
    dataset_file:=results/hw_dataset.json
```

`hardware_overrides.yaml` holds only the values that differ from simulation and
inherits everything else, so the two never drift apart silently.

---

## Viewing it

```bash
rviz2 -d mycobot_calibration/rviz/handeye_live.rviz   # robot, TF, detections
rviz2 -d mycobot_calibration/rviz/camera_pov.rviz     # raw camera only
```

Both are **derived from a config RViz itself wrote**, and that matters. An RViz
Image display lives in a Qt dock, and RViz positions its docks only from the
base64 `QMainWindow State` blob at the bottom of the file. A hand-written
config without that blob loads with no error, lists the display, subscribes to
the topic and receives every frame — and draws nothing, because the dock is
never placed. Keep the display named `Image`; the blob addresses it by name,
and layout changes belong in the GUI via *File > Save Config As*.

**RViz and Gazebo inside one launch file segfault on this Jetson**
(`MESA-LOADER: failed to open nvidia-drm`), and RViz's exit handler then tears
down the whole launch. Start the sim with `use_rviz:=false` and run `rviz2`
standalone, so a GL crash kills only RViz.

---

## The rig

As of 2026-09-03 the simulated camera is placed at the pose **measured on the
physical rig**, so sim and hardware see the same geometry from the same range.

| | sim | hardware |
|---|---|---|
| camera in `base_link` | `[0.167, 0.242, 0.433]` | `[0.1666, 0.2425, 0.4330]` |
| `base_link` → camera | 0.5235 m | 0.5235 m |
| workspace centre → camera | 0.4212 m | 0.4212 m |
| field of view | 87° (depth FOV) | 69.76° (colour FOV) |
| `fx` | 448 | 608 |
| detections stamped in | `camera_head_depth_optical_frame` | `camera_head_color_optical_frame` |

The FOV column is the **remaining honest difference**. Ignition renders the
simulated camera at the D435's *depth* FOV, so the same marker covers ~26%
fewer pixels than on the real colour camera at the same distance. Matching it
(sim `camera_hfov` → 1.2175) would also narrow the frame so the arm covers more
of it, and is the obvious next improvement.

The frame differs for a real reason, not an oversight: Ignition's `rgbd_camera`
renders the colour image from the sensor origin rather than the colour lens, so
stamping detections in the colour frame biases every one of them by the 15 mm
baseline. A real D435 genuinely images from the colour lens.

### The camera is a child of `base_link`

`base_link_to_torso_link` parents the stand to the robot, so **the spawn `z`
moves the arm and the camera together**. That is usually what you want —
`base_link → camera` is the only geometry the calibration sees, so lifting both
preserves it. Raising the arm *alone* means changing the spawn `z` **and**
lowering `camera_stand_z` by the same amount.

Two values must track the spawn `z` or the scene breaks:

- `camera_stand_drop` = spawn z − 0.425, else the stand floats above the table
  or sinks into it. Negative is legal.
- `robot_base_z` in `validate_calibration.launch.py`, which converts Gazebo
  world coordinates into `base_link`. If it disagrees, the validation object
  lands off by the difference and it reads as calibration error.

`base_link` is **not** the bottom of the chassis. The `g_shape_base` mesh is
millimetres (`<unit meter="0.001">`, Z_UP) spanning −0.055…+0.055, and its
visual origin adds another −0.03, so the chassis bottom is **0.085 m below
`base_link`**. Seating it exactly on the 0.425 table top needs spawn
`z = 0.510`; the current 0.460 leaves it 0.05 inside the table, which is
cosmetic only.

**Fifteen places carry the camera pose** — `camera_stand_x/y/z`,
`camera_tilt_deg`, `camera_pan_deg` in *both* launch files and the
`intel_rgbd_cam_d435` xacro, plus `nominal_camera_position` / `_rpy_deg` in
`pose_collector_node`, `present_target_node` and `handeye_calibration.yaml`.
The launch files declare their own defaults and pass them to xacro, so **the
launch value wins and editing only the xacro changes nothing**. This has been
got wrong three times.

---

## The target

**38 mm black square on a 45.6 mm plate**, DICT_6X6_250 id 3, on the flange.

`marker_size` is the one value that scales every distance the detector reports,
with nothing ever looking wrong. Told 0.100 against a 38 mm marker, the
detector reported every distance **2.63× long** and nothing failed or warned.

It is verified by a test that needs no external reference: move the arm, and
compare the marker's displacement in the camera frame against the same
displacement from forward kinematics. A rigid transform preserves distances, so
a *similarity* fit over those pairs must return scale 1. On 2026-09-03 it
returned **1.0254**, implying 39.0 mm against the 38.0 mm measured with a rule
— within hand-measurement tolerance.

To change the target, reprint **and** regenerate the asset:

```bash
python3 mycobot_calibration/scripts/generate_aruco_target.py \
    --id 3 --marker-size <measured> --border-ratio 0.10
```

### 38 mm is too small for hardware, and this is the central limitation

A planar marker's pose has two solutions, told apart only by perspective
foreshortening. Once the marker subtends too few degrees, `SOLVEPNP_IPPE_SQUARE`
flips between them. Measured on the physical rig with the arm **stationary**:

| distance | span | behaviour |
|---|---|---|
| 0.222 m | 106 px | 1.49° rotation spread over 89 frames |
| 0.18–0.29 m | | 0.17–0.45 mm position spread |
| 0.32 m | 72 px | **174.6°** spread, arm completely still |
| 0.287 m | | 20.8–32.4° for 8 s straight |
| 0.35 m | | 7–10 mm position spread |

**Set the cap from the rotation, not the position.** They fail at different
ranges and the collector gates on both: translation is still sub-millimetre at
0.287 m where rotation has already collapsed. A cap chosen from position data
alone (0.30) still let every pose fail; `max_marker_distance: 0.25` works.

**Tilt does not help.** Stepping J5 through 0–40° at 0.22 m passed at every
angle including face-on (0.25 mm / 0.91°); 50° lost detection outright. Range
is the whole story. Simulation needs no cap — its rendered corners are clean
enough to resolve the ambiguity at far smaller pixel counts, and its 2026-09-03
sweep had **zero** unstable poses.

The real fix is a bigger printed target. At 100 mm the same 106 px falls at
0.58 m, past the far end of the workspace. This rig cannot mount one.

---

## Results

### Simulation, 2026-09-03 — 40 samples

Sim carries `ground_truth_T_base_cam` from the URDF, so the solve can be scored
against truth rather than only for self-consistency.

```
method            consistency (RMS)        vs ground truth
TSAI             20.72 mm  7.98 deg     14.93 mm  5.49 deg
PARK             11.52 mm  7.43 deg     17.38 mm  2.65 deg
HORAUD           11.52 mm  7.43 deg     17.28 mm  2.61 deg
ANDREFF          72.89 mm  7.42 deg    140.19 mm  2.20 deg
DANIILIDIS       10.39 mm  7.53 deg     12.82 mm  1.32 deg

Selected: PARK (lowest residual)   error 17.38 mm, 2.646 deg
coverage: 6/12 cells (4/10 edge) | 0.18-0.63 m (3.48x) | axis conditioning 0.76
```

**The selection rule picked the wrong method here.** It chooses on lowest
consistency residual — the only signal hardware offers — but DANIILIDIS was
both more consistent *and* markedly closer to truth. Worth remembering before
trusting the same rule on a hardware result.

### Hardware, 2026-09-03 — three 40-sample sweeps

No ground truth exists, so the datasets are cross-checked against each other.
Each was solved by dropping its worst-agreeing sample and re-solving until the
residual stopped falling:

```
dataset   trans_rms  rot_rms   camera position
hw2         3.22 mm  1.06 deg  [0.165, 0.247, 0.429]
demo        2.70 mm  1.05 deg  [0.163, 0.241, 0.436]
hw3         6.34 mm  1.46 deg  [0.172, 0.239, 0.433]

consensus [0.1666, 0.2425, 0.4330]  rpy [-113.97, -2.42, 162.13]
agreement across datasets: 14.3 mm, 5.9 deg
```

Written to `results/handeye_result_2026-09-03_consensus.yaml`. Treat 14 mm /
5.9° as the uncertainty: good enough to aim a pose plan or grasp a
reasonably-sized object, not good enough for precision work.

**Read subset sizes carefully.** Sweeping hw2 from 40 samples down to 8 shows
exactly one knee, at 38 — two samples carry over half the error, and the rest
are merely noisy:

```
n=40  13.15 mm  8.11 deg      n=30   4.55 mm  2.45 deg
n=39  10.65 mm  6.55 deg      n=20   3.44 mm  1.70 deg
n=38   6.93 mm  4.08 deg  <-  n=12   3.22 mm  1.06 deg
                              n=8    3.62 mm  0.68 deg  <- stops improving
```

Minimising translation RMS over subset size is **not** a way to count good
samples: that score falls automatically as the subset shrinks. The reassuring
quantity is that camera position moves only ~10 mm across the whole range.

---

## Hardware notes

### Camera over the network

The D435 is on a JetBot (Jetson Nano, Ubuntu 18.04, ROS 2 Eloquent) on a
different subnet from the Humble stack that owns the arm, so the two ROS graphs
cannot usefully be joined. `scripts/jetbot_frame_server.py` is a plain TCP
frame pusher with no ROS dependency; `jetbot_camera_bridge` republishes on the
**simulation's own topic names**, so the detector, collector and solver run
unmodified.

- The D435's colour stream is an ordinary UVC node — `/dev/video3` on these
  units (`video1`/`video2` are depth/IR). No `pyrealsense2` needed.
- **Intrinsics are per-camera and hardcoded in the bridge.** Confirm the serial
  before trusting them. Note the USB descriptor serial differs from the
  librealsense serial on this firmware (`224123121992` vs `233622076860`) —
  check with `rs-enumerate-devices -s`, not `lsusb`.
- Turn WiFi power-save off on the JetBot: `sudo iw dev wlan0 set power_save off`.
  It cut median worst-frame-age from 513 ms to ~122 ms.
- Codec: colour JPEG q95 measured **0.019 mm** of distance bias against
  lossless, ~120× smaller than the arm's own execution error, and is cheaper on
  both wire and CPU (90 kB/frame at 6.8% Nano CPU, against 160 kB at 20.7% for
  mono PNG). q85 is already 15× worse than q95, so do not trade quality for
  bandwidth.

### Settling

`settle_time` must be **3.5 s** on hardware, not the simulator's 1.5 s. The
firmware services servo motion before it answers `GET_ANGLES` and keeps moving
for about a second after the trajectory reports done, so the arm is still
coasting when the collector starts sampling. Classifying one sweep's 77
rejected poses by *how* they failed separates the causes cleanly:

```
translation only :  0
rotation only    : 10   <- planar ambiguity (shares a translation, differs in rotation)
BOTH             : 67   <- a moving arm, which moves both
```

Settling, not ambiguity, was 87% of the loss. Raising it took captures from
5/146 to 40/121.

### Deriving `nominal_camera_position`

It only aims the pose plan and never enters the solve, so an error costs
coverage rather than accuracy — but a badly wrong guess wastes a whole sweep.
Correcting it from 69 mm off took the capture rate from 18% to 55%.

Do **not** derive it by asking for Cartesian marker positions and letting IK
find the joints. From the arm's rest posture (all joints near zero) it stands
fully extended and vertical, which is a wrist singularity: KDL will not
converge, `/compute_ik` returns `NO_IK_SOLUTION` even for the pose the arm is
*already* holding to within 8°, and OMPL reports *"Unable to sample any valid
states for goal tree"*. Search **joint** space and use `/compute_fk` to predict
where each candidate puts the marker; joint-space goals execute cleanly from
the same posture.

A position-only fit is tempting because positions survive the ambiguity that
corrupts rotations — but it misled us by 69 mm. Six points spanning 125 mm in Y
and only 50 mm in X and Z left rotation about the short axes weakly determined,
and `t = mean_base − R·mean_cam` carries any rotation error straight into the
translation. Its 3.47 mm residual proved the points were mutually consistent;
it never proved the transform was right.

---

## Known issues

| where | what |
|---|---|
| `handeye_solver.py:149` | `camera_frame` hardcoded to `camera_head_depth_optical_frame`. On hardware the detector stamps the **colour** frame, so every result YAML it writes for the real rig is mislabelled. |
| `pose_collector_node.py:450` | Same hardcoded name, used to record `ground_truth_T_base_cam`. Publishing an estimate under that name would make a sweep record a guess as measured truth — publish estimates as `camera_estimate_optical_frame` instead. |
| `present_target_node.py:151` | `is_reachable` returns `(ok, message)` but is used as a bare truth value. A non-empty tuple is always truthy, so the reachability gate never fires and the "try further out" fallback is dead. |
| `mycobot_calibration/README.md` | Sim-only and stale: still describes a 55 mm marker and older rig numbers. This file supersedes it for the rig, the target and the results. |
| coverage | The sim run reached only 6/12 image cells and 4/10 edge cells. Matching the sim FOV to the real colour camera would likely improve both. |

A per-pose stability gate **cannot** catch a consistently wrong answer: a
marker locked onto the wrong ambiguity branch produces 8 detections agreeing to
a fraction of a millimetre while sitting tens of degrees from what every other
pose implies. Only cross-sample agreement finds those.

---

## Layout

```
mycobot_calibration/
  config/handeye_calibration.yaml   simulation defaults, heavily commented
  config/hardware_overrides.yaml    only what differs on the real rig
  mycobot_calibration/              detector, collector, solver, validation
  rviz/                             handeye_live, camera_pov, validate
  scripts/generate_aruco_target.py  regenerates target + mesh together
  scripts/jetbot_frame_server.py    runs ON the JetBot, no ROS deps
  scripts/run_demo.sh               narrated end-to-end sim demo
mycobot_description/                URDF, meshes, camera stand
mycobot_gazebo/                     worlds and the combined launch
results/                            datasets and solved transforms
```

Datasets carry `ground_truth_T_base_cam` only in simulation; on hardware that
field is `null`, which is correct and expected.
