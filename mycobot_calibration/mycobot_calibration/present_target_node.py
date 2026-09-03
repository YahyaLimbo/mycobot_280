#!/usr/bin/env python3
"""Turn the plate to face the camera, so a demo opens on a live detection.

Not part of the calibration. At the home configuration the plate's normal
points along base +Y while the stand sits on the -Y side, so the camera is
looking at the plate's blank white back and the detector correctly reports
`no detection`. That is accurate and completely uninformative to watch, and it
is the first thing anyone sees when the simulation comes up.

This moves the arm to one pose where the plate squarely faces the camera,
centred in the frame, at a distance that makes it large without filling the
view. Run it after the simulation starts and before anything else.

    ros2 run mycobot_calibration present_target

Why the arm moves rather than the camera being re-aimed
-------------------------------------------------------
Pointing the camera at the marker's home position instead of at the middle of
the workspace is the obvious alternative, and it is worse on both counts.

It does not produce a detection: the plate is facing away, so aiming at it
just centres the blank back in the frame. And it wrecks the sweep. The home
marker sits high and near the base, so aiming there swings the camera from
23.2 deg DOWN to 18.6 deg UP -- a 41.8 deg change that tips the arm's
reachable volume out of the bottom of the frame. Measured, on the shipped rig:

    aim                      candidates   cells   edge cells   best fill
    workspace centre (kept)     268       12/12     10/10         90%
    marker home position         86        6/12      4/10         50%

The camera aim has one job, which is to put the arm's workspace in the middle
of the picture. Presentation is the arm's job, and the arm is the thing that
can move.
"""

import math
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node

from mycobot_calibration.moveit_client import MoveItPoseClient
from mycobot_calibration.pose_planning import nominal_intrinsics
from mycobot_calibration.transforms import (
    euler_to_matrix,
    look_at_rotation,
    make_transform,
    pose_to_msg,
    ray_sphere_intersection,
)


class PresentTargetNode(Node):

    def __init__(self):
        super().__init__('present_target')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('end_effector_frame', 'link6_flange')
        self.declare_parameter('planning_group', 'arm')

        # Same nominal camera the collector plans against, and for the same
        # reason: this only decides where to point the plate, so an error
        # costs a nice picture, never accuracy.
        self.declare_parameter('nominal_camera_position', [0.1666, 0.2425, 0.4330])
        self.declare_parameter('nominal_camera_rpy_deg', [-113.97, 0.0, 162.13])

        self.declare_parameter('image_width', 848)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('camera_hfov', 1.5184)

        # The whole plate, as elsewhere -- this is about its footprint in the
        # image, not about the black square the detector measures. 1.2x the
        # 0.038 m marker, matching marker_size in the collector block of
        # config/handeye_calibration.yaml and the generated mesh.
        #
        # This sat at 0.0960 (1.2x an 0.080 m marker) through two changes of
        # target and was only noticed when it reported the plate spanning 197 px
        # where the geometry gives 94. It is presentation only, so a stale value
        # never corrupted anything -- it just aimed the plate at the wrong
        # distance and then misreported the result.
        self.declare_parameter('marker_size', 0.0456)
        # Offset of the plate from the flange, along the flange's +Z. Must
        # match aruco_marker_xyz in the target xacro.
        self.declare_parameter('marker_offset_z', 0.012)

        # Fraction of the frame height the plate spans. Deliberately not the
        # 90% the calibration close-ups use: this pose is for looking at, and
        # a board that fills the view shows nothing of the arm holding it or
        # of the scene around it. 0.55 reads clearly and still leaves the
        # robot visible.
        self.declare_parameter('present_fill', 0.55)

        self.declare_parameter('workspace_centre', [0.0, 0.0, 0.1316])
        self.declare_parameter('workspace_radius_max', 0.26)
        self.declare_parameter('min_marker_height', 0.02)
        self.declare_parameter('velocity_scaling', 0.25)

    def parameter(self, name):
        return self.get_parameter(name).value

    def present(self):
        camera = np.asarray(self.parameter('nominal_camera_position'),
                            dtype=float)
        rpy = [math.radians(a)
               for a in self.parameter('nominal_camera_rpy_deg')]
        R_base_cam = euler_to_matrix(*rpy)

        width = int(self.parameter('image_width'))
        height = int(self.parameter('image_height'))
        fx, _, _ = nominal_intrinsics(width, height,
                                     float(self.parameter('camera_hfov')))
        plate = float(self.parameter('marker_size'))
        fill = float(self.parameter('present_fill'))

        workspace = np.asarray(self.parameter('workspace_centre'), dtype=float)
        radius = float(self.parameter('workspace_radius_max'))

        # Straight down the optical axis: the frame centre, so the plate is
        # squarely in the middle of the picture rather than off at an edge.
        axis = R_base_cam @ np.array([0.0, 0.0, 1.0])

        span = ray_sphere_intersection(camera, axis, workspace, radius)
        if span is None or span[1] <= 0.0:
            return False, ('the optical axis does not cross the arm\'s '
                           'reachable volume; check nominal_camera_position')

        client = MoveItPoseClient(
            self, self.parameter('planning_group'),
            self.parameter('end_effector_frame'), self.parameter('base_frame'),
            velocity_scaling=float(self.parameter('velocity_scaling')))
        if not client.wait_for_servers(timeout_sec=60.0):
            return False, 'move_group never appeared'

        offset = float(self.parameter('marker_offset_z'))
        wanted = plate * fx / (fill * height)

        # Preferred distance first, then progressively further out. Backing
        # off is always safe: the plate only gets smaller in frame.
        for distance in self._candidates(wanted, span):
            point = camera + axis * distance
            if point[2] < float(self.parameter('min_marker_height')):
                continue
            R = look_at_rotation(point, camera)
            flange = make_transform(R, point - R @ np.array([0.0, 0.0, offset]))

            pose = Pose()
            pose_to_msg(flange, pose)
            if not client.is_reachable(pose, timeout_sec=5.0):
                self.get_logger().info(
                    f'{distance:.3f} m unreachable, trying further out')
                continue

            ok, message = client.move_to_pose(pose, timeout_sec=90.0)
            if not ok:
                return False, f'planning to {distance:.3f} m failed: {message}'

            spans = plate * fx / distance
            self.get_logger().info(
                f'plate presented at {distance:.3f} m: '
                f'{spans:.0f} px, {spans / height * 100:.0f}% of frame height')
            return True, 'presented'

        return False, ('no reachable pose along the optical axis; the camera '
                       'may be too far from the arm')

    def _candidates(self, wanted, span):
        """The distance we want, then further ones inside the reachable chord."""
        near, far = max(span[0], 1e-3), span[1]
        yield float(np.clip(wanted, near, far))
        for fraction in (0.35, 0.5, 0.65, 0.8):
            yield near + fraction * (far - near)


def main(argv=None):
    rclpy.init(args=argv)
    node = PresentTargetNode()
    try:
        ok, message = node.present()
        if ok:
            node.get_logger().info(message)
        else:
            node.get_logger().error(message)
        return 0 if ok else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
