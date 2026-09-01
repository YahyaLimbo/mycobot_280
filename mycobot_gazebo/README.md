# mycobot_gazebo #

Gazebo simulation support for myCobot robots including worlds and launch files

## mycobot_combined.launch.py

Brings up Gazebo, the ROS 2 controllers, MoveIt2 and RViz in one command.

| Argument | Default | Purpose |
|---|---|---|
| `use_sim` | `true` | Launch Gazebo. `false` uses the hardware path. |
| `use_rviz` | `true` | Start RViz alongside MoveIt2. |
| `use_gz_gui` | `true` | Show the Ignition GUI. See below. |
| `use_gripper` | `true` | Fit and drive the adaptive gripper. |
| `use_camera` | `false` | Enable the RealSense D435 on the fixed stand. |
| `use_aruco_marker` | `false` | Attach the ArUco calibration target to the flange. |
| `world_file` | `empty.world` | `calibration.world` for the tabletop rig. |
| `x` `y` `z` `roll` `pitch` `yaw` | `0 0 0.05 0 0 0` | Spawn pose. `z` auto-raises to `0.425` for `calibration.world`. |

### Headless operation

`ign gazebo` starts the GUI and the server as a single unit, so when the GUI
cannot obtain a working GL context it terminates and takes the server with it
— leaving the ROS bridges running and the simulation silently dead, with no
error on any ROS topic. That is what happens in a container with no
`/dev/dri`, where rendering falls back to software.

`use_gz_gui:=false` passes `-s` to run the server alone. Camera sensors keep
rendering: the Sensors system uses its own render context and does not depend
on the GUI. Anything camera-driven and unattended should run this way.

### Hand-eye calibration

```bash
ros2 launch mycobot_gazebo mycobot_combined.launch.py \
    use_camera:=true use_aruco_marker:=true use_gripper:=false \
    use_gz_gui:=false world_file:=calibration.world use_rviz:=false
```

This puts the robot on the calibration table with the D435 on its stand at
`base_link` + (0.22, −0.30, 0.50) and the ArUco target on the flange. See
`mycobot_calibration` for the sweep and the solver.

Note that `use_camera:=true` adds `torso_link` and the camera frames to the
robot description, so `base_link → camera_head_depth_optical_frame` is present
in TF as exact ground truth — useful for scoring a calibration in simulation,
and absent on real hardware by definition.
