#!/usr/bin/env python3
"""Broadcast a completed hand-eye calibration as a static TF.

    ros2 run mycobot_calibration calibration_publisher \\
        --ros-args -p result_file:=/tmp/handeye_result.yaml

Publishes the solved camera pose under its own frame name, by default
camera_calibrated_optical_frame, rather than overwriting the frame the URDF
already publishes. In simulation both exist at once, so RViz shows the
estimate and the truth side by side and any residual error is visible as the
gap between them. On real hardware, where nothing else publishes the camera
frame, set child_frame to the name the perception stack expects.
"""

import os
import re
import sys

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster


def parse_result(path):
    """Read the solver's YAML output.

    Deliberately a small hand-rolled reader rather than a PyYAML dependency:
    the file is written by this package in a fixed shape, and the parse is a
    handful of scalar lookups.
    """
    with open(path) as handle:
        text = handle.read()

    def scalar(key, default=None):
        match = re.search(rf'^\s*{key}:\s*([^\s#]+)\s*$', text, re.MULTILINE)
        if match is None:
            if default is None:
                raise ValueError(f'{path} has no {key!r} entry')
            return default
        return match.group(1)

    translation_block = text.split('translation:', 1)[1].split('rotation:', 1)[0]
    rotation_block = text.split('rotation:', 1)[1].split('rpy_deg:', 1)[0]

    def component(block, key):
        match = re.search(rf'^\s*{key}:\s*(\S+)\s*$', block, re.MULTILINE)
        if match is None:
            raise ValueError(f'{path}: missing {key} in transform')
        return float(match.group(1))

    return {
        'parent_frame': scalar('parent_frame', 'base_link'),
        'child_frame': scalar('child_frame', 'camera_head_depth_optical_frame'),
        'method': scalar('method', 'unknown'),
        'translation': [component(translation_block, axis) for axis in 'xyz'],
        'rotation': [component(rotation_block, axis) for axis in 'xyzw'],
    }


class CalibrationPublisherNode(Node):

    def __init__(self):
        super().__init__('calibration_publisher')

        self.declare_parameter('result_file', '/tmp/handeye_result.yaml')
        self.declare_parameter('parent_frame', '')
        self.declare_parameter('child_frame', 'camera_calibrated_optical_frame')

        path = self.get_parameter('result_file').value
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'no calibration at {path}; run handeye_solver first')

        result = parse_result(path)
        parent = self.get_parameter('parent_frame').value or result['parent_frame']
        child = self.get_parameter('child_frame').value

        message = TransformStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = parent
        message.child_frame_id = child
        message.transform.translation.x = result['translation'][0]
        message.transform.translation.y = result['translation'][1]
        message.transform.translation.z = result['translation'][2]
        message.transform.rotation.x = result['rotation'][0]
        message.transform.rotation.y = result['rotation'][1]
        message.transform.rotation.z = result['rotation'][2]
        message.transform.rotation.w = result['rotation'][3]

        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcaster.sendTransform(message)

        self.get_logger().info(
            f'broadcasting {parent} -> {child} from {path} '
            f'(method {result["method"]}): '
            f'xyz={[round(v, 4) for v in result["translation"]]}')


def main(args=None):
    rclpy.init(args=args)
    try:
        node = CalibrationPublisherNode()
    except (FileNotFoundError, ValueError) as exc:
        print(f'calibration_publisher: {exc}', file=sys.stderr)
        rclpy.shutdown()
        return 1
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
