# mycobot_calibration

Eye-to-hand (fixed camera) calibration for the myCobot 280.

The camera is bolted to a stand next to the robot at base_link + (0.287,
−0.209, 0.281), aimed at the middle of the arm's workspace, and an ArUco target
rides on the flange. Sweeping the arm through a set of poses and watching the
target move recovers the pose of the camera in the robot base frame — the
transform that lets anything the camera sees be expressed in robot coordinates.

## What gets solved

| | |
|---|---|
| Known per pose | `T_base_ee` from forward kinematics (TF), `T_cam_target` from ArUco detection |
| Unknown, constant | `T_base_cam` (the answer) and `T_ee_target` (the marker's mounting offset) |
| Invariant exploited | `T_ee_target = T_ee_base · T_base_cam · T_cam_target` is the same at every pose |

`cv2.calibrateHandEye` solves the *eye-in-hand* problem. The eye-to-hand case
is the same equation with the robot poses inverted, so `solve()` passes
`inv(T_base_ee)` and reads the output as `T_base_cam`. The marker's mounting
offset never has to be measured — it cancels as the constant on the right.

## Running it as a demo

Two terminals. First the simulation, with the GUIs up so the arm can be seen
moving:

```bash
ros2 launch mycobot_gazebo mycobot_combined.launch.py use_camera:=true use_aruco_marker:=true use_gripper:=false world_file:=calibration.world
```

Then the demo, which walks the four workflow steps in order and pauses between
them so each can be explained while it is on screen:

```bash
ros2 run mycobot_calibration run_demo.sh
```

Or by path, `scripts/run_demo.sh`; add `--no-pause` to run it unattended.

It checks the clock, the camera and `move_group` before starting, and clears
the validation object out of the workspace — that object is a plain Gazebo
model with no planning-scene entry, so MoveIt cannot see it and it must not be
left in the way while the arm sweeps. Expect roughly 10 minutes end to end,
most of it the pose sweep — the runtime scales with `max_poses`, which is 40.

What to point at while it runs:

| stage | on screen |
|---|---|
| step 1 | the ArUco plate on the flange, in Gazebo |
| steps 2–3 | the arm working through the pose grid; `captured N/40` in the log, and the detections in RViz's camera viewport |
| step 4 | the five-method table, and that they agree |
| validation | the arm reaching for the spawned object |

## Quick start

Everything runs inside the `mycobot_dev` container.

**1. Start the simulation** with the camera and the target attached:

```bash
ros2 launch mycobot_gazebo mycobot_combined.launch.py use_camera:=true use_aruco_marker:=true use_gripper:=false use_gz_gui:=false world_file:=calibration.world use_rviz:=false
```

**2. Run the sweep.** The detector starts, the collector drives the arm through
its pose grid, and the launch exits when the dataset is written:

```bash
ros2 launch mycobot_calibration handeye_calibration.launch.py
```

**3. Solve.** Separate from collection, so one dataset can be re-run through
different methods without moving the robot again:

```bash
ros2 run mycobot_calibration handeye_solver --dataset /tmp/handeye_dataset.json
```

**4. Publish the result** as a TF frame, to compare against the URDF's camera
frame in RViz:

```bash
ros2 run mycobot_calibration calibration_publisher --ros-args -p result_file:=/tmp/handeye_result.yaml
```

## Checking the target before a sweep

To watch detection without moving the arm:

```bash
ros2 launch mycobot_calibration handeye_calibration.launch.py run_collector:=false
```

then view `/aruco_detector/debug_image`, which draws the detected marker and
its axes.

**`no detection` at the home configuration is normal.** With the arm parked the
plate's normal points along base +Y while the stand sits on the −Y side, so the
camera is looking at the plate's blank white back; it also projects 165 px above
the top of the frame. That is not a fault: the sweep's poses are planned to be
in frame and the plate comes into view as soon as the arm moves. Use this check
to confirm the detector is *running* and the camera is streaming.

To open on a live detection instead — worth it for a demo, where `no detection`
is the first thing an audience sees:

```bash
ros2 run mycobot_calibration present_target
```

This turns the plate to face the camera, centred, spanning ~55% of the frame.
It is presentation only and touches nothing the calibration uses.

**Re-aiming the camera at the marker instead does not work**, which is worth
knowing because it is the obvious thing to try. It produces no detection — the
plate is facing away, so aiming at it merely centres the blank back — and it
costs the sweep badly, because the home marker sits high and near the base:

| camera aimed at | candidates | cells | edge cells | best fill |
|---|---|---|---|---|
| workspace centre (**kept**) | 268 | 12/12 | 10/10 | 90% |
| the marker's home position | 86 | 6/12 | 4/10 | 50% |

Aiming there swings the camera from 23.2° *down* to 18.6° *up*, tipping the
arm's reachable volume out of the bottom of the frame. The camera's aim has one
job — put the workspace in the middle of the picture — and presentation is the
arm's job, since the arm is the thing that can move.

## How the target is mounted

The plate sits flat on the flange face and looks along the tool approach axis
(flange +Z), so it points wherever the end effector points. Aiming the tool at
the camera aims the target at it, and the plate stays in front of the arm
rather than behind it, so it is never self-occluded.

Defaults live in `mycobot_description/urdf/targets/aruco_marker.urdf.xacro`
and are all overridable:

| argument | default | |
|---|---|---|
| `aruco_marker_parent` | `link6_flange` | link the plate is fixed to |
| `aruco_marker_xyz` | `0 0 0.012` | offset in the parent frame |
| `aruco_marker_rpy` | `0 0 0` | plate +Z is the printed face normal |

The default assumes **no gripper** — the calibration launch uses
`use_gripper:=false`. The gripper body also extends along flange +Z from
z = 0.034, so with one fitted the plate must move clear of it, e.g.
`aruco_marker_xyz:="0 -0.05 0.01"` to put it on a bracket to the side, still
facing along the tool.

The plate is double-sided: the printed face at z = 0 exactly (so
`aruco_marker_link` coincides with the surface OpenCV reports a pose for) and
a blank white back 0.6 mm behind it. Without that back the plate is invisible
from behind, which reads as the target disappearing rather than as a plate
seen from the wrong side.

## Which markers are used

**One marker does the calibration.** That is the design, not a simplification:
hand-eye needs a single target of known size, and the variety the solve depends
on comes from moving the arm, not from more markers. A second marker appears
only because validation needs an *object* distinct from the target.

| role | id | set in |
|---|---|---|
| calibration target, on the flange | **3** | `marker_id` in `config/handeye_calibration.yaml`, and `aruco_marker_mesh` in the target xacro |
| validation object | **1** | `marker_id` argument of `validate_calibration.launch.py` |

Both are `DICT_6X6_250`. **The two ids must differ**: during validation both are
in the camera's view and the detector selects by id, so sharing one would
calibrate against whichever was returned first, with no error.

Changing the target's id takes two steps that have to agree, because the id is
baked into the rendered texture:

```bash
python3 scripts/generate_aruco_target.py --id N --marker-size 0.093
# then set aruco_marker_mesh to aruco_dict_6x6_250_idN.dae, and marker_id to N
```

`marker_size` is the **black square's** edge, not the white plate's. On real
hardware, measure the printed square and set that number — it is the one place
where a wrong value scales every distance the detector reports without ever
looking wrong.

## Regenerating the target

The marker texture and its plane mesh are generated, not hand-drawn:

```bash
python3 scripts/generate_aruco_target.py --marker-size 0.093 --id 3
```

This writes `aruco_dict_6x6_250_id3.{png,dae}` into
`mycobot_description/meshes/aruco_marker/`. Change the size or dictionary and
the same values must be set in `config/handeye_calibration.yaml` —
`marker_size` is the **black square's** edge, not the white plate's.

## Simulation quirks this package works around

Both of these produce a confident, wrong calibration rather than an error, so
they are handled explicitly rather than left to the caller.

**CameraInfo is not the camera.** Ignition Fortress advertises `CameraInfo`
for an `rgbd_camera` using its built-in defaults — K for 320×240 at 60° —
regardless of what the SDF `<camera>` block asks for, while delivering images
at the requested size. Trusting that K scales every translation by the ratio
of the two focal lengths (≈1.6× here). The detector therefore derives K from
the image size and the configured horizontal FOV by default
(`intrinsics_source: computed`), and refuses a `CameraInfo` whose principal
point is not near the image centre. **On real hardware set
`intrinsics_source: camera_info`.**

**The colour image is rendered at the depth frame.** The sensor renders from
its own origin, which the URDF places at `camera_head_link`, coincident with
the depth frame. `camera_head_color_optical_frame` sits 15 mm to the side, so
stamping detections there biases every one of them by that baseline — about
22 px at this working distance, verified by reprojecting TF ground truth. The
default `camera_frame` is therefore `camera_head_depth_optical_frame`.

## Result on this rig

> **Stale — do not quote these.** They were taken with the camera stand at
> 0.555 m from the shoulder and a 55 mm marker on an 82.5 mm plate. As of
> 2026-08-30 the stand is at 0.400 m and the marker is 93 mm on a 139.5 mm
> plate, so the working distances, the marker's pixel span and hence the
> detector's radial bias have all changed. The numbers below are kept only as a
> record of the method's behaviour, and the *reasoning* under them still holds;
> the figures do not. Re-run `handeye_solver` and
> `scripts/check_detection_accuracy.py` before quoting anything.

25 samples, 848×480, 0.055 m marker, solved against the URDF's ground truth:

```
method            consistency (RMS)        vs ground truth
TSAI              2.44 mm  0.57 deg      6.48 mm  0.41 deg
PARK              2.45 mm  0.57 deg      6.45 mm  0.44 deg
HORAUD            2.45 mm  0.57 deg      6.45 mm  0.44 deg
ANDREFF           5.42 mm  0.57 deg     10.40 mm  0.44 deg
DANIILIDIS        2.43 mm  0.57 deg      6.12 mm  0.41 deg

Selected: DANIILIDIS (lowest residual)
  T_base_camera   xyz = [+0.2223, -0.2926, +0.5050] m   rpy = [-115.40, +0.03, -0.09] deg
  ground truth    xyz = [+0.2200, -0.2900, +0.5000] m   rpy = [-115.00, +0.00, +0.00] deg
  error           6.12 mm, 0.408 deg
```

**The translation error is the detector's radial bias, not solver error.** The
dataset's mean working distance is 0.4007 m, and 1.43 % of that is 5.73 mm
against 6.12 mm observed. Publishing the result next to the true frame
confirms the direction too: the gap is essentially pure displacement along the
camera's optical axis. So the lever is more marker pixels — not more poses, not
a different method. See below.

Because the error scales with working distance, the headline number moves with
the pose set even when nothing about the method changes. Two earlier runs, for
context rather than as a like-for-like comparison:

| run | samples | marker | mean range | error |
|---|---|---|---|---|
| first pass | 8 | 0.040 m | — | 6.28 mm / 1.66° |
| target facing flange −Z | 25 | 0.055 m | 0.33 m | 4.65 mm / 0.55° |
| target facing along tool (current) | 25 | 0.055 m | 0.40 m | 6.12 mm / 0.41° |

Rotation accuracy improves monotonically with sample count and pose variety;
translation tracks `bias × distance` and says more about where the reachable
poses landed than about the calibration.

## Validating the result

A residual proves the observations agree with each other, not that the
transform is right — a transposed or inverted frame is absorbed identically at
every pose and leaves the residual small. The only real check is to use the
calibration for its purpose and compare against a known truth:

```bash
ros2 launch mycobot_calibration validate_calibration.launch.py
```

This spawns an ArUco-tagged plate (id 1) of known pose into the running
simulation, maps it into base coordinates through the calibrated transform,
and reaches for it. Measured:

```
Object seen at 0.612 m from the camera (10 detections averaged)

Object pose in base_link, via the calibration (DANIILIDIS):
  estimated   xyz = [+0.1207, +0.1002, +0.0466]   rpy = [+59.78, -0.10, -0.18]
  truth       xyz = [+0.1200, +0.1000, +0.0500]   rpy = [+60.00, +0.00, +0.00]
  error         3.48 mm    0.30 deg

Same detection through the TRUE camera pose from TF:
  detector only            8.39 mm    0.23 deg
  net change              -4.91 mm   +0.07 deg

  execution error          4.64 mm    0.92 deg   (tool vs commanded)
  END-TO-END MISS          1.83 mm
```

Across ten consecutive runs the perception half is **bit-identical** — the
scene is static and the renderer deterministic, so the detection and the
3.48 mm are the same every time. Everything that moves is the arm:

| term | value | varies? |
|---|---|---|
| detection + calibration | 3.48 mm / 0.30° | no, identical every run |
| detector alone (true camera pose) | 8.39 mm / 0.23° | no |
| execution, tool vs commanded | 2.0 – 4.9 mm / 0.9 – 1.6° | yes |
| end-to-end miss | 0.8 – 8.1 mm, mean ≈ 4.4 mm | yes |

The end-to-end figure swings because a 2 – 5 mm execution error combines with
a fixed calibration offset in a different direction each run. **Quote the
distribution, not a single run.** A best-case 0.83 mm has been observed and is
not reproducible: it is the same calibration with a favourable draw on arm
settling, and the next run may equally be 8 mm. The honest figure for this
system is ~4 mm typical with an 8 mm tail.

**The arm, not the calibration, is now the limit.** `arm_controller` in
`ros2_controllers_template.yaml` declares no goal tolerances at all, and sets
`allow_nonzero_velocity_at_trajectory_end: true`, so the trajectory simply
ends and the controller reports success wherever the joints have got to.
Tightening the calibration further buys nothing end-to-end until that is
addressed — add `constraints.goal_time` and per-joint `constraints.<joint>.goal`
entries, or dwell at the final point. That change affects every motion in the
workspace, including the hardware path, so it is deliberately left alone here.

**The calibrated transform beats the true one.** That is expected, not
suspicious. The calibration was fitted using this same biased detector, so it
absorbed the detector's systematic scale bias, and the two partly cancel when
the calibration is used with that detector. Two consequences:

- The transform's 6.12 mm error against geometric truth **overstates** how far
  wrong the robot actually ends up, which is 3.91 mm.
- The cancellation is specific to this camera and target. Change either and it
  disappears, so do not read the 3.91 mm as headroom.

This is why the validator reports both numbers. Quoting only the end-to-end
figure would hide a real bias; quoting only the transform error would send you
tuning something that is not costing you anything in practice.

## ChArUco: implemented, not yet worth using

A ChArUco board is available (`calibration_target:=charuco` on the sim launch,
`target_type: charuco` in the detector config). The reasoning for it is sound —
chessboard corners are saddle points, so blur biases them far less than the
outer edge of an ArUco square, and the pose is fitted from 16 corners instead
of 4. **On this rig it is measurably worse, and the honest answer is to keep
the single marker until the board spans more pixels.**

Measured at the same arm pose, against TF ground truth:

| target | translation error | rotation error |
|---|---|---|
| ArUco, 55 mm marker | **5.24 mm** | **0.33°** |
| ChArUco, 5×5 on a 121 mm plate | 12–22 mm | 22–27° |

The geometry is not the problem, and this was checked rather than assumed:
projecting the plate corners through the ground-truth pose lands exactly on the
rendered board, and the board-to-plate mapping is right to within a pixel for
most corners.

What actually goes wrong is resolution. At 848×480 the board spans ~180 px, so
each square is ~36 px and the embedded markers ~22 px. Only 8 of 12 markers
decode and only 9 of 16 corners get interpolated, and of those, three land 3–6
px from where ground truth says they should be while the rest are sub-pixel.
Because the sweep aims the board at the camera it is also near
fronto-parallel — measured 1.3° — where a planar target's out-of-plane
rotation is weakly observable, so those few bad corners swing the tilt by tens
of degrees. Neither an IPPE two-branch solve nor RANSAC rescued it.

That is the real lesson: **a ChArUco board needs substantially more pixels than
a single marker before it pays off.** Its accuracy comes from many corners, and
subdividing a target that is already small just makes every feature worse. The
single marker keeps its four corners large and unambiguous.

To revisit it, raise the camera to 1280×720 (~54 px per square) or drop to a
4×4 board with larger squares, then re-measure both targets at the same pose
with `scripts/check_detection_accuracy.py`.

**Narrowing the FOV would be a third way to revisit it, and the cheapest.** At
50° fx rises 2.03×, so at the same pose the board would span ~365 px instead of
~180 and each square ~73 px instead of ~36 — past the ~54 px named above,
without touching the render cost. The two failure modes it was losing to,
markers that would not decode and corners landing several pixels off, are both
pixel-starvation symptoms. The rig ships at 87°, so this remains untested; the
near-fronto-parallel ambiguity is a separate problem that more pixels do not
fix, and the single marker stays the default.

## Accuracy, and what limits it

`scripts/check_detection_accuracy.py` scores the detector against TF, which in
simulation knows the true marker pose. Measured on this rig at 848×480:

| marker size | marker span | radial bias | lateral bias | frame-to-frame |
|---|---|---|---|---|
| 0.040 m | ~58 px | +1.90 % of range | — | — |
| 0.055 m | ~80 px | +1.43 % of range | 0.39 mm | 0.000 mm std |
| 0.093 m (**current**) | ~74 px median, ~200 px at the near end | *not yet measured* | | |

The first two rows were taken at the old 0.555 m stand. The third has not been
re-measured since the 2026-08-30 stand move; the median span is *lower* than
the 0.055 m row despite the bigger marker only because the sweep now runs both
much closer and much further (0.22–0.55 m rather than 0.43–0.67 m), so the
median sits in a far wider distribution. Re-run
`scripts/check_detection_accuracy.py` to fill it in.

The first two rows imply the same thing: a fixed inward corner inset of
≈0.56 px per side. Anti-aliased marker edges bias subpixel refinement inward, the marker
reads slightly smaller than it is, and every distance comes out proportionally
long. Two consequences worth knowing:

- **It is bias, not noise.** Frame-to-frame scatter is *zero* — the scene is
  static and the renderer deterministic. Averaging more frames per pose cannot
  reduce it. `samples_per_pose` is kept at 8 anyway because it is the right
  behaviour on real hardware, where corner noise is real and zero-mean.
- **It shrinks with marker pixels, not with marker metres.** The lever is the
  span in pixels, so a bigger target, a closer camera, a higher-resolution
  sensor or a narrower lens all work. Narrowing the lens used to be the lever
  most worth reaching for; **moving the stand in and enlarging the plate has
  since bought the same thing** — the median span went from ~50 px to ~74 px,
  and the near end of the sweep now reaches ~200 px — without the cell coverage
  a narrow lens would now cost. Raising `camera_width`/`camera_height` to
  1280×720 would cut the bias further at roughly 2.3× the render cost — under
  software rendering in this container that is the trade to weigh.

  Because the bias contributes `bias × distance` to the final error, the near
  half of the sweep is now much better conditioned than anything the old rig
  could collect. Whether that shows up in the solved transform has not been
  measured yet.

Rotation is unaffected: 0.33° per detection, because an inset that is uniform
around the square moves the corners without turning it.

## What "the board must cover the image" actually means

The rule has two distinct readings, and this package only satisfied one of them
until 2026-08-30:

1. **The board's *positions* tile the frame across the sweep** — measured by
   `cells_covered` and `u/v_span_fraction`.
2. **The board is *large in frame* in at least some views** — which needs the
   arm close to the camera, and is measured by `fill_fraction_max`.

Only the first was implemented or measured, so a rig whose board never exceeded
**18% of the image height** scored full marks. The second is the one that limits
how precisely each individual view can be localised, because corner accuracy
depends on the board's span in pixels. Both are now scored, and the second is
why the stand moved from 0.555 m to 0.380 m and the plate grew from 82.5 mm to
139.5 mm.

**They trade against each other and cannot both be maximal in one frame.** The
border inset scales with the plate, so a board big enough to fill the view
cannot also sit at the frame edge without being clipped — and a clipped single
ArUco yields *no detection at all*, not a degraded one. The set satisfies both
across a range of distances, which is why "different distances from camera" is
a rule in its own right rather than a nicety.

What the shipped sweep delivers, on the 40 poses actually kept:

| rule | before | now |
|---|---|---|
| 20–40 images | 28 candidates | 40 views |
| different orientations | median 50°, conditioning 0.70 | median 50°, conditioning 0.69 |
| **board covering entire image** | **fills 18% of height** | **fills 90%** |
| board near edges as well as centre | 6/10 edge, 8/12 cells | **10/10 edge, 12/12 cells** |
| different distances | 0.43–0.67 m (1.6×) | **0.14–0.55 m (3.8×)** |
| marker span (median) | ~50 px | **~74 px** |

Four changes got there, in descending order of effect:

**1. The stand moved in, 0.555 m → 0.400 m.** This is the only one that lifts
the ceiling. At 0.555 m the arm could bring the plate no closer than 0.30 m, so
even a perfect planner topped out at 26% fill; and the reachable volume
subtended only 27.9° either side against the lens's 43.5°, capping the
workspace at 56% of the image width. At 0.400 m those become 62% and 90%.
Moving from 0.555 m to 0.500 m alone takes cell coverage from 8/12 to 12/12 and
edge cells from 6/10 to 10/10 — the edge-coverage problem was never a planner
bug, it was the camera being too far back.

Closer still is possible and was deliberately declined: D = 0.378 m is where the
workspace exactly fills the 87° lens, but it leaves only 120 mm between the
camera and the surface of the arm's reachable sphere. Free in simulation, thin
clearance for a physical arm swinging a 140 mm board.

**2. The plate grew, 82.5 mm → 139.5 mm** (93 mm black square). Fill scales
directly with it. On its own it *costs* edge coverage, because the border inset
scales with it too — at the old 6×5 grid the larger plate dropped cell coverage
from 12/12 to 6/12 — which is why `image_grid` went to 8×6, giving the planner
more positions to find one that fits at each distance.

**3. `samples_per_ray` 3 → 7.** The chord endpoints are dropped as tangent to
the reachable volume, so with N samples the nearest sits 1/(N+1) of the way
along it. At N = 3 that discarded the whole near quarter of the arm's reach, and
with it every close-up: the nearest planned pose was 0.43 m when the arm could
reach 0.30 m.

**4. Close-ups are planned explicitly**, because the image grid structurally
cannot produce them. Its border inset scales with the plate's on-screen size,
so `pixel_at` rejects any pose where the inset no longer fits the frame — with
`edge_safety` 1.6 and a 139.5 mm plate that is everything nearer than 0.208 m,
capping the board at 62% of the frame height while the arm can reach 0.140 m.
The views the supervisor's rule is actually about were being planned and then
thrown away.

`close_up_poses` (6) are therefore planned separately, **as close as the arm
can physically reach**, on a small ring near the centre
where nothing has to fit beside them, and placed at the **front** of the plan
so `max_poses` cannot discard them — six candidates against 262 would otherwise
get about one place in the stratified draw. Roll is suppressed for them: roll
is what makes a square's bounding box larger than the square, and at this size
a rolled plate overhangs the frame, which for a single ArUco means no detection
at all. Tilt is kept, so they still contribute rotation variety instead of a
stack of fronto-parallel views.

This is what makes the difference between 62% and 80%, and it is the one change
that implements the instruction literally.

The only thing allowed to hold them back is clipping. A single ArUco touching
the frame border yields no detection at all, so the plate keeps
`close_up_margin_px` (20) of border — and that margin is what absorbs the arm's
execution error, 6.3 mm of lateral miss at this working distance against the
2–5 mm the arm actually delivers. Stating it as a margin rather than as a
target fill keeps the trade in the units the risk lives in; an earlier cut of
this asked for 80% fill, which quietly reserved 17 mm of tolerance for a 5 mm
error and gave away 12 points of coverage for nothing.

**This makes the stand distance set the fill directly**, since the arm reaches
(D − 0.26) m: at D = 0.380 m that is 0.140 m and 90% of the frame height, and
every 10 mm further out costs about 6 points. Where the camera is bolted is now
the single number that decides whether this rule can be met.

A fifth change was needed to make the others land. The candidate pool went
from 28 to 262 while `max_poses` stayed at 40, so for the first time the planner
is *discarding* most of what it plans, and which candidates survive decides the
delivered coverage. A flat shuffle spreads them only in expectation, and taking
40 of 262 that way routinely missed edge cells — precisely the cells the fewest
candidates land in, so uniform sampling is least likely to draw them. Selection
is now round-robin over (image cell, distance band), which guarantees the rarest
strata appear in any prefix.

**The narrow-lens recommendation is now withdrawn.** It was the standing advice
while the stand sat at 0.555 m. At 0.400 m the workspace subtends 40.5° either
side, so a 50° lens is *narrower than the thing it is pointed at*: the target
can leave the frame, the border inset rejects most positions, and cell coverage
falls from 12/12 to 4/12. It still raises the marker span, but by ~1.45× rather
than the ~2× it was worth before. Pinned by
`test_a_narrower_lens_no_longer_helps_at_this_stand_position`.

## Image coverage, and the camera aim

A calibration dataset should put the target across the **whole image**, edges
included, at a range of distances and orientations — corner localisation and
lens distortion both vary across the field, so a fit made from one patch of
the image is only trustworthy in that patch.

Three separate things had to be fixed to get there, and only one of them was
in the pose planner. All the figures below are **geometry**, computed by
`test/test_pose_planning.py` without a simulator; what a sweep actually
delivers is lower, because IK and visibility still get a veto.

**1. Poses are planned in image space.** Each candidate starts as a position in
the image plus a distance, back-projected through the nominal camera to a 3D
point, so tiling the image grid tiles the picture by construction. The distance
along each ray comes from intersecting it with a coarse model of the arm's
reachable volume — a sphere of radius 0.26 m about the shoulder. That step
matters more than it sounds: the camera frustum is far larger than a 280 mm
arm's workspace, and a first attempt at three *fixed* distances left **6 of 60**
candidates reachable.

The earlier version instead sampled a **box in the robot's workspace** and
merely aimed the plate at the camera, which controls nothing about where the
target appears in the picture: 21% of the width, 4 of 12 cells, nothing near an
edge.

**2. The camera was aimed away from the robot.** With the original tilt of
25° and no pan, the optical axis points along base +Y and 25° down, while the
arm's workspace sits 45° down and 37° to the side. The entire reachable volume
projected to a disc centred on pixel **(189, 466)** of an 848×480 image — the
bottom-left corner, with more than half of it outside the frame. **No pose
planner can put the target in image regions the arm cannot reach.**

`camera_tilt_deg` and `camera_pan_deg` are xacro arguments, defaulting to the
aim that puts the optical axis exactly on the workspace centre. Pass
`camera_tilt_deg:=25 camera_pan_deg:=0` to restore the original framing. If you
change the aim, update `nominal_camera_rpy_deg` in **both** the collector
config and `pose_collector_node.py`'s declared default, or the image-space plan
back-projects through the wrong direction.

## Moving the camera

**Not in the Gazebo GUI.** The camera is a link of the robot model, welded on
through fixed joints (`base_link` → `torso_link` → `camera_head_link`).
Ignition's transform tool moves whole models, not links inside one, and even if
it could, `robot_state_publisher` publishes the camera's TF from the URDF — so
the images would come from the new place while TF went on reporting the old
one. Every detection would be stamped in a frame the camera is not at, and the
ground-truth comparison would agree with itself while being wrong.

Move it at launch instead. Six arguments reach the description:

| argument | default | |
|---|---|---|
| `camera_stand_x` | `0.287` | stand position in `base_link` |
| `camera_stand_y` | `-0.209` | |
| `camera_stand_z` | `0.281` | camera height above the stand's base |
| `camera_tilt_deg` | `23.16` | degrees below horizontal |
| `camera_pan_deg` | `55.26` | degrees about base +Z |
| `camera_hfov` | `1.5184` | radians (87°) |

**How far away the stand is, is the single most consequential number here.**
The arm can bring the plate no closer to the camera than (D − 0.26) m, where D
is the stand's distance from the shoulder, and the plate's span in pixels is
`fx × plate_size / distance`. So D sets a hard ceiling on how much of the frame
the target can ever fill, and no pose planner can lift it — see *What "the board
must cover the image" actually means* below.

```bash
ros2 launch mycobot_gazebo mycobot_combined.launch.py \
    use_camera:=true use_aruco_marker:=true use_gripper:=false \
    world_file:=calibration.world \
    camera_stand_x:=0.35 camera_stand_y:=-0.20 camera_stand_z:=0.62 \
    camera_tilt_deg:=30 camera_pan_deg:=20
```

These are the *ground truth*: the whole point of the calibration is to recover
them without being told, so moving the camera and checking the solver still
lands on it is the strongest test this package has.

Two things have to follow a move, neither of which affects accuracy but both of
which affect whether the sweep works at all:

- **`nominal_camera_position` and `nominal_camera_rpy_deg`** in the collector
  config. They only aim the pose plan, so an error costs coverage rather than
  accuracy — but a badly wrong guess plans the whole sweep against the wrong
  part of the frame and wastes it. For the example above they become
  `[0.35, -0.19, 0.62]` and `[-120.0, 0.0, 20.0]`; note the **+0.01 on Y**,
  which is the stand-to-camera offset baked into `camera_head_joint`.
- **`camera_hfov`** in *both* config blocks and in
  `validate_calibration.launch.py`, if you changed it.

Until this was plumbed through, `camera_tilt_deg` and `camera_pan_deg` existed
as xacro arguments that **no launch file forwarded**, so passing them on the
command line silently did nothing.

**3. The lens was wider than the thing it was pointed at — fixed by moving the
stand, not by changing the lens.** Aiming the camera correctly is not enough if
the workspace still only projects onto part of the frame. That used to be
described here as a property of the FOV alone, which was wrong: it is a
property of the FOV *and the distance*. The workspace sphere subtends
asin(0.26 / D) either side of the optical axis, so pulling the camera in widens
it against a fixed lens.

| camera | positions inside the workspace | width spanned | cells |
|---|---|---|---|
| stand D = 0.555 m, 87° FOV (old) | 12 / 30 | 52% | 8 / 12 |
| stand D = 0.500 m, 87° FOV | — | 64% | 12 / 12 |
| **stand D = 0.380 m, 87° FOV (current)** | **44 / 48** | **90%** | **12 / 12** |
| stand D = 0.378 m, 87° FOV | — | 100% | 12 / 12 |

`camera_hfov` stays at the D435's **87°** (1.5184 rad), because the simulated
camera should resemble the sensor it is named after — and, since the stand
move, because narrowing it would now make coverage *worse*. At D = 0.380 m the
workspace subtends 40.5° either side, so a 50° lens sees less than the arm can
reach and cell coverage falls to 4/12. The measured alternatives, taken at the
old 0.555 m stand, are still tabulated in the `camera_hfov` comment in
`mycobot_description/urdf/sensors/intel_rgbd_cam_d435.urdf.xacro`, labelled as
such.

If you do narrow it, `camera_hfov` must change in **both blocks** of
`config/handeye_calibration.yaml` and in `launch/validate_calibration.launch.py`
too, or K is built for one camera while the images come from another, which
silently scales every translation.

The width that is spanned is limited by the border inset as well as by reach.
The inset exists so the plate cannot hang off the edge — a clipped ArUco yields
no detection at all — and it is applied *per distance* rather than once for the
sweep, so a far sample, whose plate is smaller on screen, is planned closer to
the border than a near one (measured: the far half of the sweep gets within
56 px of the border, the near half only 73 px).

The collector reports the coverage it actually delivered — reachability and
visibility, not the plan, decide what is really collected — and warns once per
rule of thumb it fell short of. It is stored in the dataset under `coverage`:

| reported | rule it checks |
|---|---|
| `samples` | 20–40 views |
| `rotation_median_deg`, `axis_conditioning` | varied orientations |
| `cells_covered`, `u_span_fraction`, `v_span_fraction` | target *positions* over the whole image |
| `fill_fraction_max`, `fill_fraction_median` | target *large in frame* in at least some views |
| `edge_cells_covered`, `centre_cells_covered` | edges as well as the centre |
| `distance_ratio` | varied distances |

`fill_fraction_max` is the plate's span as a fraction of the image **height**,
in the view where it was largest. It is the half of "the board covers the image"
that position coverage does not check, and it was unmeasured until 2026-08-30 —
the rig scored full marks while its board never exceeded 18% of the frame. The
warning fires below 30%; the shipped sweep reaches 90%, via the
explicitly-planned close-ups described above.

`axis_conditioning` is the one that is not obvious. Hand-eye recovers rotation
from the *relative* motions between poses, and a set whose relative rotations
all share an axis is underdetermined however many samples it holds — while
still fitting its own observations perfectly, so the residual says nothing. The
number is the smallest singular value of the stacked rotation axes over the
largest: 0 for that degenerate set, and around 0.7 for the sweep this package
plans. It is why each grid point takes a tilt about **both** of the plate's
in-plane axes, not one, plus a roll about the viewing axis.

## Seeing the detections

The detector publishes an annotated view on `/aruco_detector/debug_image`
showing the detected marker, its outline, its pose axes and the distance. When
nothing is found it publishes the frame labelled `no detection` rather than
bare, so a quiet viewport is never ambiguous between "the target is out of
view" — which is normal while the arm is moving between poses — and "the
detector has died".

**Use a separate viewport, not the sim's main RViz.** `move_group.rviz`
deliberately carries no Image displays:

```bash
ros2 launch mycobot_calibration handeye_calibration.launch.py run_collector:=false show_detections:=true
# or, simplest of all:
ros2 run rqt_image_view rqt_image_view /aruco_detector/debug_image
```

`move_group.rviz` did carry two Image displays — "Calibration detections" on
`/aruco_detector/debug_image` and "Validation detections" on
`/aruco_object_detector/debug_image` — and they **segfaulted RViz on load**
(exit −11), which took Gazebo, `move_group` and the controller spawners down
with it through the launch teardown. The simulation appeared to die of a
graphics fault; the GL errors in the log were a red herring.

Bisected 2026-08-31: an Image display is harmless on its own —
`mycobot_calibration/rviz/handeye_calibration.rviz` carries one and runs
indefinitely, which is why `show_detections:=true` is the supported route. It
is the *combination* of an Image display with MoveIt's MotionPlanning display
that crashes, and supplying the `Median window` and `Transport Hint` properties
the hand-written blocks were missing does not help. Do not re-add them to
`move_group.rviz`.

Optionally it can also draw the **coverage accumulated so far** — every past
detection as a faint outline, the 4×3 cell grid, and a running
`coverage N/12 cells` count. That is off by default: the debug image's job is
to show the live detection, and the trail is clutter whenever coverage is not
the question being asked. Turn it on for a run meant to demonstrate that the
target reached the whole frame:

```bash
ros2 launch mycobot_calibration handeye_calibration.launch.py show_coverage:=true
```

Note the trail's `N views` counts detection *frames*, not calibration poses —
the detector keeps running while the arm moves between poses, so the trail is
denser than the pose set. Coverage is measured numerically by the collector
either way and stored in the dataset, so nothing depends on this being on.

If the simulation was launched with `use_rviz:=false`, each stage can open a
small RViz of its own with the camera viewport already configured:

```bash
ros2 launch mycobot_calibration handeye_calibration.launch.py show_detections:=true
ros2 launch mycobot_calibration validate_calibration.launch.py show_detections:=true
```

## Why the poses are shaped the way they are

Hand-eye calibration recovers rotation from the *relative* motions between
poses. A set that shares one rotation axis leaves the answer underdetermined
no matter how many samples it holds, and a grid of positions all aimed at the
camera is very nearly that degenerate set. So each grid point also gets a tilt
(`tilt_angles_deg`) and a roll about the viewing axis (`roll_angles_deg`).
`test_pure_translation_dataset_is_ill_conditioned` pins this down.

The tilt is applied about **both** of the plate's in-plane axes, giving
3 × 3 × 3 = 27 perturbations cycled across the grid rather than 9. One tilt
axis is not enough for the same reason: the relative rotations then span a
rank-deficient set of axes, which is what `axis_conditioning` measures and what
`test_tilt_spans_two_axes` guards. It also keeps the plate off
fronto-parallel, which is where a planar pose solve is most ambiguous — the
failure that cost the ChArUco path 20-odd degrees of orientation error.

The `nominal_camera_position` parameter is only used to aim the target while
generating poses. It does not enter the calibration, so an error there costs
visibility, not accuracy.

## Method selection

The solver runs all five OpenCV methods and reports two numbers for each:

- **consistency** — scatter of `T_ee_target` across the dataset. Available on
  real hardware, and what `--method auto` selects on.
- **vs ground truth** — only in simulation, where TF knows the true camera
  pose. Never used for selection, so the selection rule is identical on real
  hardware.

TSAI and DANIILIDIS lose conditioning as relative rotations approach 180°;
PARK, HORAUD and ANDREFF do not. On exact synthetic data with relative
rotations up to 178° the first two miss by hundreds of millimetres while the
other three are exact to 1e-9. This is a property of their rotation
parametrisations, and it is why the solver selects on residuals rather than
defaulting to a favourite. See
`test_tsai_degrades_near_180_degree_relative_rotations`.

## Tests

The maths is tested without ROS, Gazebo or a camera, so a bad result can be
attributed to the rig rather than to `solve()`:

```bash
cd mycobot_calibration && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/ -q
```

| file | covers |
|---|---|
| `test/test_handeye_solver.py` | the eye-to-hand reduction and the transform helpers |
| `test/test_pose_planning.py` | the pose plan and the coverage score |

The second exists because every rule a calibration dataset has to satisfy is
geometric, so all of them can be checked without moving a robot: that the plan
reaches all 12 image cells and both the edges and the middle, that the plate
never overhangs the frame, that the relative rotations do not share an axis,
and that the marker spans enough pixels to localise. Each metric also has a
test that makes it *fail* — a dataset stuck in one corner, one at a single
range, one rotating about a single axis — because a metric nobody has seen fail
is not a metric.

(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` sidesteps a pip-installed `anyio` in the
container image that is incompatible with the system pytest — unrelated to
this package.)

## Files

| Path | Role |
|---|---|
| `mycobot_calibration/aruco_detector_node.py` | Marker detection → `T_cam_target` |
| `mycobot_calibration/pose_collector_node.py` | Pose sweep, dataset capture |
| `mycobot_calibration/pose_planning.py` | Where to put the target, and scoring what came back |
| `mycobot_calibration/moveit_client.py` | Pose goals via `/move_action` + `/compute_ik` |
| `mycobot_calibration/handeye_solver.py` | Eye-to-hand reduction and scoring |
| `mycobot_calibration/calibration_publisher_node.py` | Result → static TF |
| `mycobot_calibration/transforms.py` | Rigid-transform helpers |
| `config/handeye_calibration.yaml` | All tunables |
| `scripts/generate_aruco_target.py` | Marker texture + mesh generator |
| `scripts/check_detection_accuracy.py` | Detector vs TF ground truth (sim only) |
| `mycobot_calibration/validate_calibration_node.py` | End-to-end check against a known object |
| `launch/validate_calibration.launch.py` | Spawns the object, detects, checks, reaches |

The target's URDF lives in `mycobot_description/urdf/targets/aruco_marker.urdf.xacro`.

## Note on ROS 2 Humble

This image ships no MoveIt Python binding — `moveit_py` arrived in Iron and
`moveit_commander` was ROS 1 only. `moveit_client.py` therefore talks directly
to `/move_action` and `/compute_ik`, the same interfaces `MoveGroupInterface`
uses underneath. The `/compute_ik` pre-check matters for unattended runs: a
sweep deliberately probes the edge of a 280 mm workspace, so many candidates
are unreachable, and rejecting those in milliseconds beats spending the full
planning timeout on each.
