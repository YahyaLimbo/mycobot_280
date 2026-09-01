#!/usr/bin/env python3
"""Score the ArUco detector against simulation ground truth, in place.

Compares each detected marker pose with the same transform read from TF, which
the URDF knows exactly. Simulation only -- on real hardware there is nothing to
compare against, which is the whole reason hand-eye calibration exists.

Run it with the simulation and the detector already up:

    ros2 run mycobot_calibration aruco_detector --ros-args --params-file <config>
    python3 check_detection_accuracy.py

Use it to separate detector error from calibration error. If the per-detection
bias here is a few millimetres, no amount of pose collection will produce a
calibration better than that, and the fix belongs in the marker or the camera
(bigger target, more pixels), not in the solver.

The bias worth watching is the radial one. Anti-aliased marker edges pull
refined corners inward by a fraction of a pixel, the marker reads as slightly
smaller than it is, and every distance comes out proportionally long. That is
systematic, so averaging more frames will not remove it.
"""

import argparse
import math
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from mycobot_calibration.transforms import (
    pose_from_msg,
    transform_difference,
    transform_from_msg,
)


class AccuracyChecker(Node):

    def __init__(self, topic, camera_frame, marker_frame):
        super().__init__('check_detection_accuracy')
        self.camera_frame = camera_frame
        self.marker_frame = marker_frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.detections = []
        self.create_subscription(PoseStamped, topic, self.on_detection, 20)

    def on_detection(self, msg):
        self.detections.append(pose_from_msg(msg.pose))

    def truth(self):
        tf = self.tf_buffer.lookup_transform(
            self.camera_frame, self.marker_frame, rclpy.time.Time())
        return transform_from_msg(tf.transform)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', default='/aruco_detector/target_pose')
    parser.add_argument('--camera-frame',
                        default='camera_head_depth_optical_frame')
    parser.add_argument('--marker-frame', default='aruco_marker_link')
    parser.add_argument('--samples', type=int, default=15)
    args = parser.parse_args()

    rclpy.init()
    node = AccuracyChecker(args.topic, args.camera_frame, args.marker_frame)

    for _ in range(1500):
        rclpy.spin_once(node, timeout_sec=0.1)
        if len(node.detections) >= args.samples:
            break

    if not node.detections:
        print(f'no detections on {args.topic}', file=sys.stderr)
        return 1

    try:
        T_truth = node.truth()
    except Exception as exc:
        print(f'no TF {args.camera_frame} <- {args.marker_frame}: {exc}',
              file=sys.stderr)
        return 1

    print(f'\n{len(node.detections)} detections vs TF ground truth\n')
    print(f'  truth    xyz = {np.round(T_truth[:3, 3], 4).tolist()} m')

    translations = np.array([T[:3, 3] for T in node.detections])
    mean_translation = translations.mean(axis=0)
    print(f'  detected xyz = {np.round(mean_translation, 4).tolist()} m  (mean)')

    errors = [transform_difference(T_truth, T) for T in node.detections]
    translation_errors = np.array([e[0] for e in errors])
    rotation_errors = np.array([math.degrees(e[1]) for e in errors])

    print(f'\n  translation error : {translation_errors.mean() * 1000:6.2f} mm '
          f'mean, {translation_errors.max() * 1000:6.2f} mm max')
    print(f'  rotation error    : {rotation_errors.mean():6.2f} deg mean, '
          f'{rotation_errors.max():6.2f} deg max')

    # Split the error into the part along the viewing ray and the part across
    # it. A scale bias from edge blur shows up almost entirely in the radial
    # component, while corner noise spreads across both.
    truth_range = np.linalg.norm(T_truth[:3, 3])
    detected_range = np.linalg.norm(mean_translation)
    direction = T_truth[:3, 3] / truth_range
    offset = mean_translation - T_truth[:3, 3]
    radial = float(np.dot(offset, direction))
    lateral = float(np.linalg.norm(offset - radial * direction))

    print(f'\n  range             : {truth_range:.4f} m true, '
          f'{detected_range:.4f} m detected')
    print(f'  radial bias       : {radial * 1000:+6.2f} mm '
          f'({radial / truth_range * 100:+.2f} % of range)')
    print(f'  lateral bias      : {lateral * 1000:6.2f} mm')
    print(f'  frame-to-frame    : '
          f'{np.linalg.norm(translations - mean_translation, axis=1).std() * 1000:.3f}'
          ' mm std\n')

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
