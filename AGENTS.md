# AGENTS.md — myCobot 280 ROS2 workspace

## Project

ROS 2 Humble + Ignition Fortress simulation workspace for the myCobot 280 6-DOF robotic arm. Adapted from `automaticaddison/mycobot_ros2`.

Three packages:
- `mycobot_description` — URDF/XACRO, meshes, RVIZ config, robot state publisher
- `mycobot_gazebo` — Ignition Gazebo Fortress worlds, ROS<->Ignition bridge, combined launcher
- `mycobot_moveit_config` — MoveIt2 SRDF, kinematics, OMPL/pilz/STOMP planners, controllers

## Build

```sh
colcon build
```

Pre-built workspace — `build/` and `install/` exist. Source before launching:

```sh
source install/setup.bash
```

## Launch

| Command | What it does |
|---|---|
| `ros2 launch mycobot_gazebo mycobot_combined.launch.py` | Full sim: Gazebo + MoveIt2 + RViz |
| `ros2 launch mycobot_gazebo mycobot_combined.launch.py use_sim:=false` | Planning-only (hardware stub, no real driver) |
| `ros2 launch mycobot_gazebo mycobot_combined.launch.py use_rviz:=false` | Headless sim |
| `ros2 launch mycobot_gazebo mycobot_combined.launch.py use_gripper:=true` | With gripper |
| `ros2 launch mycobot_gazebo mycobot_combined.launch.py world_file:=calibration.world` | Tabletop calibration world |

## Key details

- **Default robot name**: `mycobot_280` (set via `robot_name` arg everywhere)
- **XACRO entrypoint**: `mycobot_description/urdf/robots/mycobot_280.urdf.xacro` — supports flags: `use_gripper`, `use_camera`, `use_gazebo`, `gripper_type`, `base_type`, etc.
- **Controllers** loaded via `ros2 control load_controller --set-state active` — sequenced: joint_state_broadcaster → arm_controller → (optional) gripper_action_controller
- **Planners**: OMPL (default), pilz_industrial_motion_planner, STOMP
- **Trajectory tolerance**: `allowed_start_tolerance: 0.2` (raised from default 0.01 to handle Gazebo drift)
- **Stale process cleanup**: combined launcher kills `parameter_bridge`, `image_bridge`, `ign gazebo` etc. on shutdown and pre-launch to avoid duplicate spawns
- **Calibration world auto-z**: when `world_file:=calibration.world`, z is auto-adjusted to 0.425 (tabletop height)
- **Hardware**: no real robot driver yet — `use_sim:=false` logs a stub message

## ROSA LLM agent

`rosa_local.py` at workspace root runs a ROSA agent (Gemma 4 via Ollama) connected to ROS2:

```sh
./rosa_local.py
```

## Controllers config

Template `ros2_controllers_template.yaml` in `mycobot_moveit_config/config/mycobot_280/` gets `${prefix}` and `${flange_link}` substituted on launch via `OpaqueFunction`. The output goes to `ros2_controllers.yaml` in the same directory.

## Launch ordering quirk

In `mycobot_combined.launch.py`, the `moveit_and_rviz` OpaqueFunction **must** be added to the launch description **before** `rsp_launch`. ROS2 launch shares `LaunchConfiguration` across `IncludeLaunchDescription` calls; RSP passes `use_rviz:=false` which would overwrite the shared value if read after. The combined launcher also does NOT nest RViz exit handlers inside the OpaqueFunction — that would silently block subsequent actions in ROS2 Humble.

## Tests

Only `ament_lint_auto` tests are configured (no functional tests). Run:
```sh
colcon test
```
