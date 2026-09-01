#!/usr/bin/env python3
"""Prove a hand-eye calibration is usable, not merely self-consistent.

The solver's residual says the observations agree with each other. It does not
say the transform is correct: a frame that is transposed, inverted, or stamped
in the wrong optical convention can still produce a small residual, because
the error is absorbed identically at every pose. The only way to tell is to
use the calibration for what it is for -- turning something the camera sees
into a place the robot can go -- and check the answer against a known truth.

So this node:

  1. detects an object of known pose (ArUco id 1 on a plate),
  2. maps it into the robot base frame through the calibrated transform,
     T_base_object = T_base_cam . T_cam_object,
  3. reports the error against ground truth, and
  4. optionally commands the arm to reach for it and measures where the tool
     actually ended up.

Step 3 is reported twice: once through the calibrated transform, and once
through the true camera pose taken from TF. The second is the error the
detector contributes on its own, so the difference between them isolates what
the calibration is responsible for. Without that split, a detector bias of a
few millimetres reads as calibration error and sends you tuning the wrong
thing.
"""

import math
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener

from mycobot_calibration.calibration_publisher_node import parse_result
from mycobot_calibration.moveit_client import MoveItPoseClient
from mycobot_calibration.transforms import (
    average_transforms,
    euler_to_matrix,
    look_at_rotation,
    make_transform,
    matrix_to_euler,
    pose_from_msg,
    pose_to_msg,
    quaternion_to_matrix,
    transform_difference,
    transform_from_msg,
)


class ValidateCalibrationNode(Node):

    def __init__(self):
        super().__init__('validate_calibration')

        self.declare_parameter('result_file', '/tmp/handeye_result.yaml')
        self.declare_parameter('object_pose_topic',
                               '/aruco_object_detector/target_pose')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('end_effector_frame', 'link6_flange')
        self.declare_parameter('true_camera_frame',
                               'camera_head_depth_optical_frame')
        self.declare_parameter('planning_group', 'arm')

        # Ground truth, in the robot base frame. The object is spawned in
        # Gazebo world coordinates, and base_link sits at world z = 0.425 on
        # the calibration table, so a spawn at (0.12, 0.10, 0.475) is
        # (0.12, 0.10, 0.05) here.
        self.declare_parameter('object_xyz', [0.12, 0.10, 0.05])
        self.declare_parameter('object_rpy_deg', [60.0, 0.0, 0.0])

        self.declare_parameter('samples', 10)
        self.declare_parameter('sample_timeout', 15.0)
        self.declare_parameter('reach', True)
        self.declare_parameter('standoff', 0.10)

        # Park the arm clear of the camera's view of the object before looking.
        #
        # This is not optional housekeeping. Reaching for a marker means
        # approaching along its face normal, and that normal points at the
        # camera by construction -- so a successful reach leaves the tool
        # sitting squarely on the camera-to-object sightline. Without parking
        # first, the run after any reach sees nothing at all. Parking at the
        # start rather than only retreating at the end also makes the node
        # indifferent to whatever posture it inherits.
        self.declare_parameter('park_first', True)
        self.declare_parameter('park_joint_names', [
            'link1_to_link2', 'link2_to_link3', 'link3_to_link4',
            'link4_to_link5', 'link5_to_link6', 'link6_to_link6_flange'])
        self.declare_parameter('park_joint_positions', [0.0] * 6)

        self.base_frame = self.get_parameter('base_frame').value
        self.ee_frame = self.get_parameter('end_effector_frame').value
        self.true_camera_frame = self.get_parameter('true_camera_frame').value
        self.samples = int(self.get_parameter('samples').value)
        self.sample_timeout = float(self.get_parameter('sample_timeout').value)
        self.standoff = float(self.get_parameter('standoff').value)

        self.T_base_object_truth = make_transform(
            euler_to_matrix(*[math.radians(a) for a in
                              self.get_parameter('object_rpy_deg').value]),
            self.get_parameter('object_xyz').value)

        result = parse_result(self.get_parameter('result_file').value)
        self.T_base_cam = make_transform(
            quaternion_to_matrix(*result['rotation']), result['translation'])
        self.calibration_method = result['method']

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=20)
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.detections = []
        self.create_subscription(
            PoseStamped, self.get_parameter('object_pose_topic').value,
            lambda msg: self.detections.append(pose_from_msg(msg.pose)), qos)

        self.moveit = MoveItPoseClient(
            self,
            group_name=self.get_parameter('planning_group').value,
            end_effector_link=self.ee_frame,
            base_frame=self.base_frame,
            velocity_scaling=0.2,
            acceleration_scaling=0.2)

    def lookup(self, target_frame, source_frame, timeout_sec=10.0):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                tf = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, rclpy.time.Time())
                return transform_from_msg(tf.transform)
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.05)
        return None

    def observe_object(self):
        """Average a burst of detections of the object marker."""
        self.detections = []
        deadline = time.monotonic() + self.sample_timeout
        while (rclpy.ok() and len(self.detections) < self.samples
               and time.monotonic() < deadline):
            rclpy.spin_once(self, timeout_sec=0.05)
        if len(self.detections) < 3:
            return None
        return average_transforms(list(self.detections))

    def run(self):
        print('\n' + '=' * 68)
        print('  Hand-eye calibration validation')
        print('=' * 68)

        if self.get_parameter('park_first').value:
            if not self.moveit.wait_for_servers(timeout_sec=30.0):
                return 1
            ok, reason = self.moveit.move_to_joints(
                list(self.get_parameter('park_joint_names').value),
                list(self.get_parameter('park_joint_positions').value))
            if not ok:
                self.get_logger().warn(
                    f'could not park the arm ({reason}); the object may be '
                    'occluded by the tool')
            else:
                print('\nArm parked clear of the camera view')
            for _ in range(20):
                rclpy.spin_once(self, timeout_sec=0.05)

        T_cam_object = self.observe_object()
        if T_cam_object is None:
            self.get_logger().error(
                f'object marker not seen on '
                f'{self.get_parameter("object_pose_topic").value}')
            return 1

        distance = float(np.linalg.norm(T_cam_object[:3, 3]))
        print(f'\nObject seen at {distance:.3f} m from the camera '
              f'({len(self.detections)} detections averaged)')

        # -- step 1: does the calibration put the object in the right place? --
        T_base_object_est = self.T_base_cam @ T_cam_object
        calibrated_error = transform_difference(
            self.T_base_object_truth, T_base_object_est)

        print(f'\nObject pose in {self.base_frame}, via the calibration '
              f'({self.calibration_method}):')
        self.print_pose('  estimated  ', T_base_object_est)
        self.print_pose('  truth      ', self.T_base_object_truth)
        print(f'  error       {calibrated_error[0] * 1000:6.2f} mm   '
              f'{math.degrees(calibrated_error[1]):5.2f} deg')

        # -- step 2: how much of that is the detector, not the calibration? --
        T_base_cam_true = self.lookup(self.base_frame, self.true_camera_frame,
                                      timeout_sec=3.0)
        if T_base_cam_true is not None:
            T_base_object_ideal = T_base_cam_true @ T_cam_object
            detector_error = transform_difference(
                self.T_base_object_truth, T_base_object_ideal)
            print('\nSame detection through the TRUE camera pose from TF, '
                  'which isolates\nthe detector from the calibration:')
            print(f'  detector only          '
                  f'{detector_error[0] * 1000:6.2f} mm   '
                  f'{math.degrees(detector_error[1]):5.2f} deg')
            delta = (calibrated_error[0] - detector_error[0]) * 1000.0
            print(f'  net change             {delta:+6.2f} mm   '
                  f'{math.degrees(calibrated_error[1] - detector_error[1]):+5.2f} deg')
            if delta < 0.0:
                print('\n  The calibrated transform beats the true one, which is '
                      'expected rather\n  than suspicious: it was fitted using '
                      'this same detector, so it absorbed\n  the detector\'s '
                      'systematic bias and the two partly cancel in use. It is '
                      'the\n  reason the transform\'s error against geometric '
                      'truth overstates how far\n  wrong the robot actually '
                      'ends up. Swap in a different camera or target\n  and the '
                      'cancellation goes away.')

        if not self.get_parameter('reach').value:
            print()
            return 0

        # -- step 3: can the robot actually go there? ------------------------
        return self.reach_for(T_base_object_est)

    def reach_for(self, T_base_object_est):
        """Send the tool to a standoff in front of the estimated object pose.

        The tool is aimed at the object and stopped `standoff` metres short of
        it along the approach, rather than driven into contact: the interesting
        quantity is where the tool axis would meet the object, and stopping
        short measures that without the arm pushing a static model.
        """
        if not self.moveit.wait_for_servers(timeout_sec=30.0):
            return 1

        object_position = T_base_object_est[:3, 3]
        # Approach along the object's own face normal (+Z of the plate), so
        # the tool meets the marker square-on rather than edge-on.
        approach = T_base_object_est[:3, 2]
        tool_position = object_position + approach * self.standoff

        goal = make_transform(
            look_at_rotation(tool_position, object_position), tool_position)
        goal_pose = pose_to_msg(goal, Pose())

        print(f'\nReaching to {np.round(tool_position, 4).tolist()} '
              f'({self.standoff * 100:.0f} cm standoff, tool aimed at the object)')

        reachable, reason = self.moveit.is_reachable(goal_pose)
        if not reachable:
            print(f'  standoff pose is not reachable ({reason})')
            print('  the transform check above still stands; only the reach '
                  'is skipped\n')
            return 0

        ok, reason = self.moveit.move_to_pose(goal_pose)
        if not ok:
            print(f'  motion failed ({reason})')
            return 1

        time.sleep(1.0)
        for _ in range(40):
            rclpy.spin_once(self, timeout_sec=0.05)

        T_base_ee = self.lookup(self.base_frame, self.ee_frame)
        if T_base_ee is None:
            print('  could not read the flange pose afterwards')
            return 1

        # Where the tool axis actually points, projected out by the standoff.
        touch_point = T_base_ee[:3, 3] + T_base_ee[:3, 2] * self.standoff
        miss = float(np.linalg.norm(touch_point - self.T_base_object_truth[:3, 3]))

        print(f'  tool stopped at        '
              f'{np.round(T_base_ee[:3, 3], 4).tolist()}')
        print(f'  its axis meets         {np.round(touch_point, 4).tolist()}')
        print(f'  true object centre     '
              f'{np.round(self.T_base_object_truth[:3, 3], 4).tolist()}')

        # How much of the miss is the robot rather than the calibration. The
        # arm settles inside the controller's tolerance, not exactly on the
        # commanded pose, and projecting the tool axis out by the standoff
        # amplifies any orientation error -- 1 degree is 1.7 mm at 10 cm. This
        # term varies run to run on an unchanged detection, so folding it into
        # the headline figure would make the calibration look noisy when it is
        # not.
        execution = transform_difference(goal, T_base_ee)
        print(f'\n  execution error        '
              f'{execution[0] * 1000:6.2f} mm   '
              f'{math.degrees(execution[1]):5.2f} deg   (tool vs commanded, '
              'the robot\'s own\n                         '
              'settling, independent of the calibration)')
        print(f'\n  END-TO-END MISS        {miss * 1000:.2f} mm')
        print('  (camera detection -> calibration -> IK -> motion -> tool tip)\n')
        return 0

    @staticmethod
    def print_pose(label, T):
        rpy = [math.degrees(a) for a in matrix_to_euler(T[:3, :3])]
        print(f'{label} xyz = [{T[0, 3]:+.4f}, {T[1, 3]:+.4f}, {T[2, 3]:+.4f}] m'
              f'   rpy = [{rpy[0]:+.2f}, {rpy[1]:+.2f}, {rpy[2]:+.2f}] deg')


def main(args=None):
    rclpy.init(args=args)
    node = ValidateCalibrationNode()
    status = 1
    try:
        status = node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return status


if __name__ == '__main__':
    sys.exit(main())
