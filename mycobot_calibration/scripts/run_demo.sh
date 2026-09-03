#!/usr/bin/env bash
#
# End-to-end eye-to-hand calibration demo, run against a live simulation.
#
# Walks the four workflow steps in order and pauses between them so each can be
# explained while it is on screen:
#
#   1. target on the end effector    (already in the URDF, shown in Gazebo)
#   2. move the arm with MoveIt2     (~9 min, the arm sweeps the workspace)
#   3. record the pose pairs         (written as the sweep runs)
#   4. solve the hand-eye transform  (all five OpenCV methods, scored)
#   +  validate it on a known object (the part that proves it is usable)
#
# Start the simulation in another terminal first, with the GUIs up:
#
#   ros2 launch mycobot_gazebo mycobot_combined.launch.py \
#       use_camera:=true use_aruco_marker:=true use_gripper:=false \
#       world_file:=calibration.world
#
# then run this. Use --no-pause to run it unattended.

set -u -o pipefail

WORLD=${WORLD:-calibration_world}
DATASET=${DATASET:-/tmp/handeye_dataset.json}
RESULT=${RESULT:-/tmp/handeye_result.yaml}
PAUSE=1

for arg in "$@"; do
  case "$arg" in
    --no-pause) PAUSE=0 ;;
    -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

banner() {
  echo
  echo "=================================================================="
  echo "  $*"
  echo "=================================================================="
}

step() {
  if [ "$PAUSE" -eq 1 ]; then
    echo
    read -r -p "  [Enter] to continue, Ctrl-C to stop  "
  fi
}

# --- preflight ------------------------------------------------------------
# Fail here with something readable rather than 40 lines into a sweep. The
# most common way to start this demo wrong is to forget the simulation, and
# without it every later stage hangs waiting on a topic that never arrives.
banner "Preflight"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "  ros2 not on PATH -- source /opt/ros/humble/setup.bash and the workspace" >&2
  exit 1
fi

echo -n "  simulation clock ... "
if ! timeout 15 ros2 topic echo /clock --once >/dev/null 2>&1; then
  echo "NOT RUNNING"
  echo
  echo "  Start the simulation in another terminal first:" >&2
  echo >&2
  echo "    ros2 launch mycobot_gazebo mycobot_combined.launch.py \\" >&2
  echo "        use_camera:=true use_aruco_marker:=true use_gripper:=false \\" >&2
  echo "        world_file:=calibration.world" >&2
  exit 1
fi
echo "ok"

echo -n "  camera stream ...... "
if ! timeout 20 ros2 topic echo /camera_head/color/image_raw --once >/dev/null 2>&1; then
  echo "NOT PUBLISHING"
  echo "  the simulation is up but the camera is not -- relaunch with use_camera:=true" >&2
  exit 1
fi
echo "ok"

echo -n "  move_group ......... "
if ! timeout 20 ros2 action list 2>/dev/null | grep -q '/move_action'; then
  echo "NOT AVAILABLE"
  echo "  MoveIt2 is not up; the pose sweep cannot run" >&2
  exit 1
fi
echo "ok"

# The validation object is a plain Gazebo model, invisible to MoveIt's
# planning scene, so it must not be left in the workspace while the arm
# sweeps. Removing it is harmless when it is not there.
echo -n "  clearing workspace . "
ign service -s "/world/${WORLD}/remove" \
  --reqtype ignition.msgs.Entity --reptype ignition.msgs.Boolean \
  --timeout 3000 --req 'name: "aruco_target" type: MODEL' >/dev/null 2>&1
echo "ok"

# --- step 1 ---------------------------------------------------------------
banner "Step 1/4  --  calibration target on the end effector"
cat <<'EOF'

  The ArUco target (DICT_6X6_250 id 3, 38 mm black square on a 45.6 mm
  plate) is mounted on the flange and
  faces along the tool approach axis, so it points wherever the tool points.
  It is part of the robot description, enabled with use_aruco_marker:=true.

  Look at the Gazebo window: the plate is on the end of the arm.
EOF
step

# --- steps 2 and 3 --------------------------------------------------------
banner "Steps 2 & 3/4  --  MoveIt2 pose sweep, recording pose pairs"
cat <<'EOF'

  The collector generates candidate poses across the workspace, rejects the
  unreachable ones with /compute_ik, then plans and executes to the rest
  through MoveIt2's /move_action. At each stop it records:

      T_base_ee     the flange pose, from forward kinematics via TF
      T_cam_target  the marker pose, from ArUco detection

  Watch the arm move in Gazebo, and the detections in RViz's "Calibration
  detections" viewport. This takes about 9 minutes.

EOF
step

ros2 launch mycobot_calibration handeye_calibration.launch.py \
  dataset_file:="${DATASET}"

if [ ! -s "${DATASET}" ]; then
  echo "  no dataset was written -- the sweep did not collect anything" >&2
  exit 1
fi

# --- step 4 ---------------------------------------------------------------
banner "Step 4/4  --  solving the hand-eye transform"
cat <<'EOF'

  Eye-to-hand is the eye-in-hand problem with the robot poses inverted, so
  OpenCV's calibrateHandEye returns the camera pose in the robot base frame.
  All five methods are run and scored on how consistently each reproduces a
  fixed marker-to-flange offset; the winner is chosen on that alone, which is
  the same rule that would be used on real hardware.

EOF
step

ros2 run mycobot_calibration handeye_solver \
  --dataset "${DATASET}" --output "${RESULT}"

# --- validation -----------------------------------------------------------
banner "Validation  --  is the transform actually usable?"
cat <<'EOF'

  A small residual does not prove the transform is right: a transposed or
  inverted frame is absorbed identically at every pose. So an object of known
  pose is spawned, detected, mapped into base coordinates through the
  calibration, and reached for.

  The report separates the detector's own error from the calibration's, and
  the arm's settling error from both.

EOF
step

ros2 launch mycobot_calibration validate_calibration.launch.py \
  result_file:="${RESULT}"

banner "Demo complete"
echo
echo "  dataset  ${DATASET}"
echo "  result   ${RESULT}"
echo
echo "  To show the calibrated frame beside the true one in RViz:"
echo
echo "    ros2 run mycobot_calibration calibration_publisher \\"
echo "        --ros-args -p result_file:=${RESULT}"
echo
