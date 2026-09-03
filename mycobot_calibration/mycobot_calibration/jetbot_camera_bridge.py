#!/usr/bin/env python3
"""Republish the JetBot's D435 frames as a normal ROS 2 camera on this host.

The camera is mounted on the JetBot and cannot move (it is the mount). That
machine runs ROS 2 Eloquent on Ubuntu 18.04 and sits on a different subnet, so
joining the two ROS graphs is not on the table -- and it has no `aruco` in its
OpenCV, so it cannot run the detector either. This node closes the gap from the
other side: it pulls raw frames over a plain TCP socket from
`scripts/jetbot_frame_server.py` and publishes them here, where OpenCV 4.11 and
the whole tested calibration stack already live.

It deliberately publishes on the SAME topic names the simulation uses:

    /camera_head/color/image_raw     sensor_msgs/Image      (mono8)
    /camera_head/color/camera_info   sensor_msgs/CameraInfo

so `aruco_detector`, `pose_collector` and `handeye_solver` all run against the
physical camera completely unmodified. Only configuration changes.

INTRINSICS
----------
These default to the D435's own factory calibration for the colour stream at
848x480, read once off the device with

    rs-enumerate-devices -c

They are parameters rather than a live CameraInfo because librealsense and V4L2
capture are mutually exclusive on that device -- querying it while the frame
server streams would fail. Factory intrinsics do not drift, so reading them once
is not a compromise.

Note the FOV this implies. The simulation models the D435's DEPTH field of view,
87 degrees, because that is what the simulated rgbd_camera renders. The real
COLOUR stream is 69.7 degrees. The detector does not care once it is given a real
K, but the collector back-projects its pose plan through `camera_hfov`, so that
must be set to 1.2167 rad for hardware or the whole sweep is planned against a
frustum the camera does not have.
"""

import json
import socket
import struct
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

FRAME_HEADER = struct.Struct('>IQ')

# THESE ARE PER-CAMERA. They are the factory calibration of ONE D435, and every
# unit has its own -- the two on this bench differ by about 8 px in the
# principal point:
#
#   serial 233622076860  fx 608.238 fy 608.470  cx 422.118 cy 253.566  <- these
#   serial 238222072521  fx 608.715 fy 608.435  cx 430.063 cy 246.492
#
# Swapping the camera without changing these is a silent error of exactly the
# kind this package keeps warning about: nothing fails, and every reported
# translation is biased. Confirm the serial before trusting the defaults --
#
#   rs-enumerate-devices -s
#   rs-enumerate-devices -c | grep -A 9 'Intrinsic of "Color" / 848x480'
#
# Distortion comes back as Inverse Brown Conrady with every coefficient zero
# (the colour stream leaves the sensor already rectified), so the ROS model name
# is immaterial and plumb_bob with zeros is the honest translation.
DEFAULT_SERIAL = '233622076860'
DEFAULT_FX = 608.238098144531
DEFAULT_FY = 608.469970703125
DEFAULT_CX = 422.118041992188
DEFAULT_CY = 253.566452026367


class JetbotCameraBridge(Node):

    def __init__(self):
        super().__init__('jetbot_camera_bridge')

        self.declare_parameter('host', '172.22.133.39')
        self.declare_parameter('port', 5555)
        self.declare_parameter('image_topic', '/camera_head/color/image_raw')
        self.declare_parameter('camera_info_topic',
                               '/camera_head/color/camera_info')
        # The physical camera is not in the URDF, so this frame need not exist
        # in TF. Nothing in the calibration requires it to: the collector reads
        # the marker pose out of the detector's message and never looks the
        # camera up. Recovering where the camera is IS the job.
        self.declare_parameter('frame_id', 'camera_head_color_optical_frame')
        self.declare_parameter('fx', DEFAULT_FX)
        self.declare_parameter('fy', DEFAULT_FY)
        self.declare_parameter('cx', DEFAULT_CX)
        self.declare_parameter('cy', DEFAULT_CY)
        self.declare_parameter('reconnect_period', 2.0)

        self.host = self.get_parameter('host').value
        self.port = int(self.get_parameter('port').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.reconnect_period = float(
            self.get_parameter('reconnect_period').value)

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(
            Image, self.get_parameter('image_topic').value, 10)
        self.info_pub = self.create_publisher(
            CameraInfo, self.get_parameter('camera_info_topic').value, 10)

        self.camera_info = None
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    # -- intrinsics ---------------------------------------------------------

    def build_camera_info(self, width, height):
        fx = float(self.get_parameter('fx').value)
        fy = float(self.get_parameter('fy').value)
        cx = float(self.get_parameter('cx').value)
        cy = float(self.get_parameter('cy').value)

        info = CameraInfo()
        info.header.frame_id = self.frame_id
        info.width = width
        info.height = height
        info.distortion_model = 'plumb_bob'
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [fx, 0.0, cx,
                  0.0, fy, cy,
                  0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0,
                  0.0, fy, cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]

        hfov = 2.0 * np.arctan(width / (2.0 * fx))
        self.get_logger().info(
            'intrinsics %dx%d: fx=%.2f fy=%.2f cx=%.2f cy=%.2f '
            '(hfov %.2f deg = %.4f rad)'
            % (width, height, fx, fy, cx, cy,
               np.degrees(hfov), hfov))
        self.get_logger().info(
            'set camera_hfov to %.4f in BOTH blocks of '
            'handeye_calibration.yaml for hardware runs' % hfov)
        return info

    # -- socket -------------------------------------------------------------

    @staticmethod
    def recv_exactly(sock, count):
        chunks = []
        remaining = count
        while remaining > 0:
            chunk = sock.recv(min(remaining, 65536))
            if not chunk:
                raise ConnectionError('server closed the connection')
            chunks.append(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)

    def read_greeting(self, sock):
        """Read the newline-terminated JSON header the server opens with."""
        buf = b''
        while not buf.endswith(b'\n'):
            chunk = sock.recv(1)
            if not chunk:
                raise ConnectionError('server closed before sending a header')
            buf += chunk
            if len(buf) > 4096:
                raise ConnectionError('header did not terminate')
        return json.loads(buf.decode('utf-8'))

    def run(self):
        while self.running and rclpy.ok():
            try:
                self.stream_once()
            except (OSError, ConnectionError, ValueError) as exc:
                self.get_logger().warn(
                    'camera link to %s:%d down (%s); retrying in %.1fs'
                    % (self.host, self.port, exc, self.reconnect_period))
                time.sleep(self.reconnect_period)

    def stream_once(self):
        self.get_logger().info(
            'connecting to frame server at %s:%d' % (self.host, self.port))
        sock = socket.create_connection((self.host, self.port), timeout=10.0)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            greeting = self.read_greeting(sock)
            width = int(greeting['width'])
            height = int(greeting['height'])
            # Follow whatever the server says it is sending rather than
            # assuming. It defaults to mono8, because the detector converts to
            # grey before it looks at anything, but it can be asked for bgr8 --
            # which is worth it while a human is aiming the camera by eye.
            encoding = greeting.get('encoding', 'mono8')
            decode_flag = (cv2.IMREAD_COLOR if encoding == 'bgr8'
                           else cv2.IMREAD_GRAYSCALE)
            self.get_logger().info(
                'streaming %dx%d %s/%s'
                % (width, height, encoding, greeting.get('format')))
            self.camera_info = self.build_camera_info(width, height)

            # The wire carries the JetBot's capture time, but frames are stamped
            # with THIS host's clock: the two machines are not time-synced, and
            # every consumer here compares against this clock. The gap is
            # reported instead, because a growing one is the symptom that
            # matters -- it would mean detections were being attributed to the
            # pose after the one they were taken at.
            frames = 0
            worst_age_ms = 0.0
            last_report = time.time()

            while self.running and rclpy.ok():
                header = self.recv_exactly(sock, FRAME_HEADER.size)
                length, stamp_us = FRAME_HEADER.unpack(header)
                payload = self.recv_exactly(sock, length)

                image = cv2.imdecode(
                    np.frombuffer(payload, dtype=np.uint8), decode_flag)
                if image is None:
                    self.get_logger().warn('undecodable frame, skipping')
                    continue

                stamp = self.get_clock().now().to_msg()
                msg = self.bridge.cv2_to_imgmsg(image, encoding=encoding)
                msg.header.stamp = stamp
                msg.header.frame_id = self.frame_id
                self.image_pub.publish(msg)

                self.camera_info.header.stamp = stamp
                self.info_pub.publish(self.camera_info)

                frames += 1
                age_ms = (time.time() * 1e6 - stamp_us) / 1000.0
                worst_age_ms = max(worst_age_ms, age_ms)

                now = time.time()
                if now - last_report >= 10.0:
                    self.get_logger().info(
                        '%.1f Hz, %d kB/frame, worst frame age %.0f ms'
                        % (frames / (now - last_report), length // 1024,
                           worst_age_ms))
                    frames = 0
                    worst_age_ms = 0.0
                    last_report = now
        finally:
            sock.close()

    def stop(self):
        self.running = False


def main(args=None):
    rclpy.init(args=args)
    node = JetbotCameraBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
