#!/usr/bin/env python3
"""A minimal MoveIt2 pose-goal client built on the move_group action.

ROS 2 Humble's MoveIt ships no Python binding in this image: there is no
moveit_py (Iron and later) and no moveit_commander (ROS 1). Rather than pull
in a C++ node just to command poses, this talks to the same interfaces
MoveGroupInterface uses underneath:

    /compute_ik    (moveit_msgs/srv/GetPositionIK)   reachability pre-check
    /move_action   (moveit_msgs/action/MoveGroup)    plan and execute

The IK pre-check matters for unattended collection. Planning to an unreachable
pose costs the full planning timeout before it fails, and a calibration sweep
deliberately probes the edge of a 280 mm workspace, so a large share of
candidate poses are expected to be unreachable. compute_ik rejects those in
milliseconds.
"""

import time

import rclpy
from geometry_msgs.msg import Pose, PoseStamped, Vector3
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    OrientationConstraint,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from shape_msgs.msg import SolidPrimitive


class MoveItPoseClient:
    """Plans and executes Cartesian pose goals for one planning group."""

    # MoveItErrorCodes values worth naming in log output.
    ERROR_NAMES = {
        1: 'SUCCESS',
        -1: 'FAILURE',
        -2: 'PLANNING_FAILED',
        -3: 'INVALID_MOTION_PLAN',
        -4: 'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE',
        -5: 'CONTROL_FAILED',
        -6: 'UNABLE_TO_AQUIRE_SENSOR_DATA',
        -7: 'TIMED_OUT',
        -10: 'START_STATE_IN_COLLISION',
        -11: 'START_STATE_VIOLATES_PATH_CONSTRAINTS',
        -12: 'GOAL_IN_COLLISION',
        -13: 'GOAL_VIOLATES_PATH_CONSTRAINTS',
        -14: 'GOAL_CONSTRAINTS_VIOLATED',
        -15: 'INVALID_GROUP_NAME',
        -17: 'INVALID_LINK_NAME',
        -21: 'NO_IK_SOLUTION',
        -31: 'NO_IK_SOLUTION',
    }

    def __init__(self, node, group_name, end_effector_link, base_frame,
                 planning_time=5.0, planning_attempts=10,
                 velocity_scaling=0.3, acceleration_scaling=0.3,
                 position_tolerance=0.005, orientation_tolerance=0.02):
        self.node = node
        self.group_name = group_name
        self.end_effector_link = end_effector_link
        self.base_frame = base_frame
        self.planning_time = planning_time
        self.planning_attempts = planning_attempts
        self.velocity_scaling = velocity_scaling
        self.acceleration_scaling = acceleration_scaling
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance

        self.move_client = ActionClient(node, MoveGroup, '/move_action')
        self.ik_client = node.create_client(GetPositionIK, '/compute_ik')

    def wait_for_servers(self, timeout_sec=60.0):
        """Block until move_group is up. Returns False on timeout."""
        if not self.move_client.wait_for_server(timeout_sec=timeout_sec):
            self.node.get_logger().error('/move_action never appeared')
            return False
        if not self.ik_client.wait_for_service(timeout_sec=timeout_sec):
            self.node.get_logger().error('/compute_ik never appeared')
            return False
        return True

    def error_name(self, code):
        return self.ERROR_NAMES.get(code, f'UNKNOWN({code})')

    # -- reachability -------------------------------------------------------

    def is_reachable(self, pose, timeout_sec=5.0):
        """Ask MoveIt whether an IK solution exists for this pose."""
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.group_name
        request.ik_request.ik_link_name = self.end_effector_link
        request.ik_request.robot_state = RobotState()
        request.ik_request.robot_state.is_diff = True
        request.ik_request.avoid_collisions = True
        request.ik_request.timeout.sec = 1

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.base_frame
        pose_stamped.pose = pose
        request.ik_request.pose_stamped = pose_stamped

        future = self.ik_client.call_async(request)
        if not self._spin_until_done(future, timeout_sec):
            return False, 'IK service timed out'
        result = future.result()
        if result is None:
            return False, 'IK service returned nothing'
        code = result.error_code.val
        return code == 1, self.error_name(code)

    # -- motion -------------------------------------------------------------

    def build_goal(self, pose):
        """Encode a pose goal the way MoveGroupInterface does internally."""
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = self.base_frame
        position_constraint.link_name = self.end_effector_link
        position_constraint.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)

        region = SolidPrimitive()
        region.type = SolidPrimitive.SPHERE
        region.dimensions = [self.position_tolerance]
        position_constraint.constraint_region.primitives = [region]

        region_pose = Pose()
        region_pose.position = pose.position
        region_pose.orientation.w = 1.0
        position_constraint.constraint_region.primitive_poses = [region_pose]
        position_constraint.weight = 1.0

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = self.base_frame
        orientation_constraint.link_name = self.end_effector_link
        orientation_constraint.orientation = pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = self.orientation_tolerance
        orientation_constraint.absolute_y_axis_tolerance = self.orientation_tolerance
        orientation_constraint.absolute_z_axis_tolerance = self.orientation_tolerance
        orientation_constraint.weight = 1.0

        constraints = Constraints()
        constraints.name = 'pose_goal'
        constraints.position_constraints = [position_constraint]
        constraints.orientation_constraints = [orientation_constraint]

        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.goal_constraints = [constraints]
        goal.request.num_planning_attempts = self.planning_attempts
        goal.request.allowed_planning_time = self.planning_time
        goal.request.max_velocity_scaling_factor = self.velocity_scaling
        goal.request.max_acceleration_scaling_factor = self.acceleration_scaling
        # An empty start_state with is_diff set tells move_group to plan from
        # the live joint state rather than from an all-zeros configuration.
        goal.request.start_state.is_diff = True

        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        goal.planning_options.replan = False
        return goal

    def build_joint_goal(self, joint_names, positions, tolerance=0.01):
        """Encode a joint-space goal, for parking the arm at a known posture."""
        constraints = Constraints()
        constraints.name = 'joint_goal'
        constraints.joint_constraints = []
        for name, position in zip(joint_names, positions):
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = float(position)
            constraint.tolerance_above = tolerance
            constraint.tolerance_below = tolerance
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)

        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.goal_constraints = [constraints]
        goal.request.num_planning_attempts = self.planning_attempts
        goal.request.allowed_planning_time = self.planning_time
        goal.request.max_velocity_scaling_factor = self.velocity_scaling
        goal.request.max_acceleration_scaling_factor = self.acceleration_scaling
        goal.request.start_state.is_diff = True

        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        goal.planning_options.replan = False
        return goal

    def move_to_joints(self, joint_names, positions, timeout_sec=60.0):
        """Plan and execute to a joint configuration. Returns (ok, message)."""
        return self._execute(self.build_joint_goal(joint_names, positions),
                             timeout_sec)

    def move_to_pose(self, pose, timeout_sec=60.0):
        """Plan and execute to `pose`. Returns (ok, message)."""
        return self._execute(self.build_goal(pose), timeout_sec)

    def _execute(self, goal, timeout_sec):
        """Submit a MoveGroup goal and wait for the motion to finish."""
        send_future = self.move_client.send_goal_async(goal)
        if not self._spin_until_done(send_future, timeout_sec):
            return False, 'timed out submitting goal'

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False, 'move_group rejected the goal'

        result_future = goal_handle.get_result_async()
        if not self._spin_until_done(result_future, timeout_sec):
            # Leaving a goal running would let the arm keep moving while the
            # collector believes it has stopped, so cancel before giving up.
            goal_handle.cancel_goal_async()
            return False, 'timed out waiting for motion to finish'

        result = result_future.result()
        if result is None:
            return False, 'no result returned'
        code = result.result.error_code.val
        return code == 1, self.error_name(code)

    def _spin_until_done(self, future, timeout_sec):
        """Spin the node until `future` completes or the deadline passes.

        The collector runs as a single-threaded script, so this drives the
        executor by hand instead of relying on a background spin thread. The
        deadline is deliberately wall-clock: it supervises ROS plumbing, and
        on sim time it would hang forever if Gazebo were ever paused.
        """
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not future.done():
            if time.monotonic() > deadline:
                return False
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return future.done()
