#!/usr/bin/env python3
"""Detect the ArUco calibration target and publish its pose in the camera frame.

Subscribes to the colour stream, runs ArUco detection, and publishes the pose
of the requested marker as a PoseStamped stamped in the camera's optical
frame. This is the "camera to target" half of the hand-eye pair; the robot's
forward kinematics supply the other half.

Two properties of this simulation are handled explicitly, because both are
silent failures that produce a plausible but wrong calibration:

Intrinsics. Ignition Fortress advertises CameraInfo for the rgbd_camera
sensor with its built-in defaults (320x240, 60 degree FOV) regardless of what
the SDF <camera> block asks for, while the images themselves come out at the
requested size. Trusting that K scales every translation by roughly the ratio
of the two focal lengths. The node therefore derives K from the image size and
the configured horizontal FOV by default, and validates any CameraInfo it is
told to use.

Optical frame. The rgbd_camera renders from the sensor's own origin, which the
URDF places at camera_head_link, i.e. coincident with the depth frame. The
colour frame sits 15 mm to the side, so stamping detections in
camera_head_color_optical_frame biases every one of them by that baseline,
about 22 px at the working distance here. The default frame is therefore the
depth optical frame.
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

from mycobot_calibration.aruco_compat import (
    CharucoInterpolator,
    MarkerDetector,
    board_chessboard_corners,
    create_charuco_board,
    create_detector_parameters,
    estimate_marker_pose,
    get_dictionary,
)
from mycobot_calibration.transforms import matrix_to_quaternion

ARUCO_DICTIONARIES = {
    'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
    'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
    'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
    'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
    'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
    'DICT_ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class ArucoDetectorNode(Node):

    def __init__(self):
        super().__init__('aruco_detector')

        self.declare_parameter('image_topic', '/camera_head/color/image_raw')
        self.declare_parameter('camera_info_topic',
                               '/camera_head/color/camera_info')
        self.declare_parameter('camera_frame',
                               'camera_head_depth_optical_frame')
        self.declare_parameter('marker_frame', 'aruco_marker_detected')
        self.declare_parameter('dictionary', 'DICT_6X6_250')
        # id 3 is the flange target; the validation object is id 1.
        self.declare_parameter('marker_id', 3)
        self.declare_parameter('marker_size', 0.04)

        # 'aruco'   -- one marker, pose from its four outer corners
        # 'charuco' -- a board, pose fitted from every interior corner
        #
        # ChArUco attacks the error that dominates here. An ArUco corner sits
        # on the outer edge of a black square, and an anti-aliased edge biases
        # the refined corner inward by a fraction of a pixel: measured at 0.56
        # px on this rig, systematic, and immune to averaging. A chessboard
        # corner is a saddle point between two black and two white quadrants,
        # so blur is symmetric about it and the bias largely cancels. The pose
        # is also fitted from 16 corners rather than 4, and survives partial
        # visibility -- which matters because the sweep deliberately drives the
        # target to the edge of the frame, where a single marker that is
        # clipped yields nothing at all.
        self.declare_parameter('target_type', 'aruco')
        self.declare_parameter('squares_x', 5)
        self.declare_parameter('squares_y', 5)
        self.declare_parameter('square_length', 0.022)
        self.declare_parameter('charuco_marker_length', 0.0165)
        self.declare_parameter('min_charuco_corners', 6)
        self.declare_parameter('intrinsics_source', 'computed')
        # 1.5184 rad = 87 deg, the URDF camera's default. Must track it: this
        # is what K is built from when intrinsics_source is 'computed'.
        self.declare_parameter('camera_hfov', 1.5184)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('publish_tf', True)

        # Optionally draw every past detection as a faint outline, so the
        # frame shows the coverage accumulated over the sweep rather than just
        # the current view. Off by default: the debug image's job is to show
        # the live detection, and the trail is only wanted when demonstrating
        # that the target reached the whole frame. Coverage is measured
        # numerically by the collector either way, so nothing depends on this.
        self.declare_parameter('draw_coverage_trail', False)
        self.declare_parameter('coverage_trail_limit', 400)

        self.image_topic = self.get_parameter('image_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.marker_frame = self.get_parameter('marker_frame').value
        self.marker_id = int(self.get_parameter('marker_id').value)
        self.marker_size = float(self.get_parameter('marker_size').value)
        self.intrinsics_source = self.get_parameter('intrinsics_source').value
        self.camera_hfov = float(self.get_parameter('camera_hfov').value)
        self.publish_debug = bool(self.get_parameter('publish_debug_image').value)
        self.publish_tf = bool(self.get_parameter('publish_tf').value)

        dictionary_name = self.get_parameter('dictionary').value
        if dictionary_name not in ARUCO_DICTIONARIES:
            raise ValueError(f'unknown ArUco dictionary {dictionary_name!r}')
        self.dictionary = get_dictionary(ARUCO_DICTIONARIES[dictionary_name])

        self.target_type = self.get_parameter('target_type').value
        if self.target_type not in ('aruco', 'charuco'):
            raise ValueError(f'unknown target_type {self.target_type!r}')

        self.board = None
        self.board_centre_offset = None
        if self.target_type == 'charuco':
            squares_x = int(self.get_parameter('squares_x').value)
            squares_y = int(self.get_parameter('squares_y').value)
            square_length = float(self.get_parameter('square_length').value)
            self.min_charuco_corners = int(
                self.get_parameter('min_charuco_corners').value)
            self.board = create_charuco_board(
                squares_x, squares_y, square_length,
                float(self.get_parameter('charuco_marker_length').value),
                self.dictionary)
            # estimatePoseCharucoBoard reports the pose of the board ORIGIN,
            # which OpenCV puts at a corner of the squares region, not at its
            # middle. Publishing that directly would offset every detection by
            # half the board -- 78 mm here -- while still looking perfectly
            # self-consistent. The board's axes are +X right, +Y up, +Z out of
            # the printed face, the same convention estimatePoseSingleMarkers
            # uses, so only the translation needs correcting.
            self.board_centre_offset = np.array([
                squares_x * square_length / 2.0,
                squares_y * square_length / 2.0,
                0.0])

        # Subpixel corner refinement is the single highest-value detector
        # setting here. Without it corners land on integer pixels, and at the
        # ~58 px marker size of this rig that quantisation alone is worth
        # millimetres of translation error in the final calibration.
        self.detector_params = create_detector_parameters()
        self.detector_params.cornerRefinementMethod = \
            cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector_params.cornerRefinementWinSize = 5
        self.detector_params.cornerRefinementMaxIterations = 50
        self.detector_params.cornerRefinementMinAccuracy = 0.01

        self.marker_detector = MarkerDetector(self.dictionary,
                                              self.detector_params)
        self.charuco = (CharucoInterpolator(self.board, self.dictionary,
                                            self.detector_params)
                        if self.board is not None else None)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = np.zeros(5)
        self.info_warned = False
        self.detection_count = 0
        self.draw_trail = bool(self.get_parameter('draw_coverage_trail').value)
        self.trail_limit = int(self.get_parameter('coverage_trail_limit').value)
        self.trail = []
        self.last_charuco = None
        self.ambiguous_count = 0

        sensor_qos = QoSProfile(depth=1)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.pose_pub = self.create_publisher(PoseStamped, '~/target_pose', 10)
        self.debug_pub = (self.create_publisher(Image, '~/debug_image', 1)
                          if self.publish_debug else None)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value,
            self.on_camera_info, 10)
        self.create_subscription(
            Image, self.image_topic, self.on_image, sensor_qos)

        self.get_logger().info(
            f'{self.target_type} detector up: dictionary={dictionary_name}, '
            f'publishing poses in {self.camera_frame!r}')

    # -- intrinsics ---------------------------------------------------------

    def on_camera_info(self, msg):
        """Adopt or audit the published intrinsics, depending on the source."""
        if self.intrinsics_source == 'camera_info':
            K = np.array(msg.k, dtype=float).reshape(3, 3)
            if not self.plausible_intrinsics(K, msg.width, msg.height):
                if not self.info_warned:
                    self.get_logger().error(
                        'CameraInfo K is inconsistent with the image size it '
                        f'declares ({msg.width}x{msg.height}, K principal '
                        f'point ({K[0, 2]:.1f}, {K[1, 2]:.1f})). Refusing to '
                        'use it; set intrinsics_source:=computed.')
                    self.info_warned = True
                return
            self.camera_matrix = K
            self.dist_coeffs = np.array(msg.d, dtype=float) if msg.d else np.zeros(5)
        elif self.camera_matrix is None:
            self.set_computed_intrinsics(msg.width, msg.height)

    @staticmethod
    def plausible_intrinsics(K, width, height):
        """True when K's principal point sits near the middle of the image.

        A pinhole render always puts the principal point at the image centre.
        Ignition's stale default K claims (160, 120) for an 848x480 image,
        which this catches; a real miscalibrated camera would still pass, which
        is the intent -- this is a sanity gate, not a calibration check.
        """
        if width <= 0 or height <= 0 or K[0, 0] <= 0.0:
            return False
        return (abs(K[0, 2] - width / 2.0) < 0.25 * width and
                abs(K[1, 2] - height / 2.0) < 0.25 * height)

    def set_computed_intrinsics(self, width, height):
        """Derive a pinhole K from the image size and the configured FOV."""
        fx = (width / 2.0) / np.tan(self.camera_hfov / 2.0)
        self.camera_matrix = np.array([[fx, 0.0, width / 2.0],
                                       [0.0, fx, height / 2.0],
                                       [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.zeros(5)
        self.get_logger().info(
            f'Using intrinsics computed from {width}x{height} and hfov '
            f'{self.camera_hfov:.4f} rad: fx=fy={fx:.2f}, '
            f'cx={width / 2.0:.1f}, cy={height / 2.0:.1f}')

    # -- detection ----------------------------------------------------------

    def detect_single_marker(self, corners, ids):
        """Pose from one marker's four outer corners.

        Returns (rvec, tvec, outline) or None.
        """
        if ids is None:
            return None
        for corner_set, marker_id in zip(corners, ids.ravel()):
            if int(marker_id) != self.marker_id:
                continue
            rvec, tvec = estimate_marker_pose(
                corner_set, self.marker_size,
                self.camera_matrix, self.dist_coeffs)
            if rvec is None:
                return None
            return rvec, tvec, corner_set.reshape(-1, 2).astype(np.int32)
        return None

    def detect_charuco(self, gray, corners, ids):
        """Pose from every interior chessboard corner the board shows.

        The detected markers only serve to identify which board is in view and
        which corner is which; the corners themselves are then interpolated and
        refined on the chessboard, which is where the accuracy comes from.

        Returns (rvec, tvec, outline) or None. The translation is corrected
        from the board origin to the board centre so it can be compared with,
        and substituted for, a single-marker detection.
        """
        if ids is None or len(ids) == 0:
            return None

        count, charuco_corners, charuco_ids = self.charuco.interpolate(
            gray, corners, ids)
        if count < self.min_charuco_corners:
            return None

        # Deliberately NOT cv2.aruco.estimatePoseCharucoBoard.
        #
        # That helper runs solvePnP with the default iterative solver, which on
        # a planar target has two near-degenerate solutions and silently
        # returns whichever it converges to. The sweep aims the board straight
        # at the camera, so it is almost fronto-parallel -- measured at 1.3 deg
        # off -- which is exactly where the two branches are hardest to tell
        # apart. It picked the wrong one and produced a 23.7 deg orientation
        # error whose axis lay in the board plane, the signature of that flip,
        # while translation still looked plausible at 13 mm.
        #
        # IPPE is built for planar targets and returns BOTH solutions, so the
        # branch can be chosen on reprojection error rather than luck. The
        # single-marker path never showed this because
        # estimatePoseSingleMarkers uses IPPE_SQUARE internally.
        object_points = board_chessboard_corners(self.board)[
            charuco_ids.ravel()].reshape(-1, 1, 3).astype(np.float64)
        image_points = charuco_corners.reshape(-1, 1, 2).astype(np.float64)

        try:
            _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                object_points, image_points,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            return None
        if not rvecs:
            return None

        best = int(np.argmin(np.asarray(errors).ravel()))
        rvec, tvec = rvecs[best], tvecs[best]

        if len(rvecs) > 1:
            ordered = np.sort(np.asarray(errors).ravel())
            # A near-tie means the geometry itself cannot separate the two
            # branches, not that the solver misbehaved. Worth saying out loud,
            # because the fix is more tilt in the sweep, not more samples.
            if ordered[0] > 0.0 and ordered[1] / ordered[0] < 1.2:
                self.ambiguous_count += 1
                if self.ambiguous_count % 25 == 1:
                    self.get_logger().warn(
                        'planar pose is nearly ambiguous (reprojection errors '
                        f'{ordered[0]:.3f} vs {ordered[1]:.3f}); the board is '
                        'close to fronto-parallel. Increase tilt_angles_deg.')

        R, _ = cv2.Rodrigues(rvec)
        centre = tvec.reshape(3) + R @ self.board_centre_offset

        outline = cv2.convexHull(
            charuco_corners.reshape(-1, 2).astype(np.float32)
        ).reshape(-1, 2).astype(np.int32)
        self.last_charuco = (charuco_corners, charuco_ids, count)
        return rvec.reshape(3), centre, outline

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        if self.camera_matrix is None:
            if self.intrinsics_source == 'camera_info':
                return                       # still waiting for valid CameraInfo
            self.set_computed_intrinsics(msg.width, msg.height)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self.marker_detector.detect(gray)

        if self.target_type == 'charuco':
            found = self.detect_charuco(gray, corners, ids)
        else:
            found = self.detect_single_marker(corners, ids)

        if found is None:
            if self.debug_pub is not None:
                # Say so on the image rather than republishing the raw frame.
                # A live camera view with nothing drawn on it is ambiguous
                # between "the target is not visible" and "the detector has
                # died", and during a sweep the arm spends much of its time
                # legitimately in the first state.
                annotated = frame.copy()
                self.draw_coverage(annotated)
                cv2.putText(annotated, 'no detection', (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                self.publish_debug_image(annotated, msg.header)
            return

        rvec, tvec, outline = found
        R, _ = cv2.Rodrigues(rvec)

        # The detection is stamped with the camera frame this node was
        # configured with, not with the image header's frame_id. Ignition
        # stamps images with camera_head_link, which is not an optical frame:
        # consuming that directly would rotate every detection by 90 degrees
        # on two axes.
        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self.camera_frame
        quat = matrix_to_quaternion(R)
        pose.pose.position.x = float(tvec[0])
        pose.pose.position.y = float(tvec[1])
        pose.pose.position.z = float(tvec[2])
        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])
        self.pose_pub.publish(pose)

        if self.tf_broadcaster is not None:
            tf_msg = TransformStamped()
            tf_msg.header = pose.header
            tf_msg.child_frame_id = self.marker_frame
            tf_msg.transform.translation.x = pose.pose.position.x
            tf_msg.transform.translation.y = pose.pose.position.y
            tf_msg.transform.translation.z = pose.pose.position.z
            tf_msg.transform.rotation = pose.pose.orientation
            self.tf_broadcaster.sendTransform(tf_msg)

        self.detection_count += 1
        if self.detection_count % 50 == 1:
            self.get_logger().info(
                f'marker {self.marker_id} at '
                f'({tvec[0]:+.3f}, {tvec[1]:+.3f}, {tvec[2]:+.3f}) m '
                f'in {self.camera_frame}')

        if self.draw_trail:
            self.trail.append(outline)
            if len(self.trail) > self.trail_limit:
                self.trail.pop(0)

        if self.debug_pub is not None:
            annotated = frame.copy()
            self.draw_coverage(annotated)

            if self.target_type == 'charuco':
                charuco_corners, charuco_ids, count = self.last_charuco
                cv2.aruco.drawDetectedCornersCharuco(
                    annotated, charuco_corners, charuco_ids, (0, 255, 0))
                total_corners = board_chessboard_corners(self.board).shape[0]
                label = (f'charuco {count}/{total_corners}'
                         f' corners  {float(np.linalg.norm(tvec)):.3f} m')
                axis_length = float(self.get_parameter('square_length').value) * 2.0
            else:
                cv2.aruco.drawDetectedMarkers(
                    annotated, [outline.reshape(1, -1, 2).astype(np.float32)],
                    np.array([[self.marker_id]]))
                label = (f'id {self.marker_id}  '
                         f'{float(np.linalg.norm(tvec)):.3f} m')
                axis_length = self.marker_size * 0.75

            # Axes are drawn at the *reported* pose, which for ChArUco is the
            # board centre rather than the origin OpenCV returns. Drawing them
            # anywhere else would hide exactly the offset bug this correction
            # exists to prevent.
            cv2.drawFrameAxes(annotated, self.camera_matrix, self.dist_coeffs,
                              rvec, np.asarray(tvec, dtype=float).reshape(3, 1),
                              axis_length)
            cv2.putText(annotated, label, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            self.publish_debug_image(annotated, msg.header)

    def draw_coverage(self, frame):
        """Overlay past detections and how much of the frame they have reached.

        Drawn through a blended copy rather than straight onto the image, so a
        long trail cannot bury the live detection under solid ink.
        """
        if not self.trail:
            return

        overlay = frame.copy()
        for corners in self.trail:
            cv2.polylines(overlay, [corners], True, (255, 160, 0), 1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, frame)

        height, width = frame.shape[:2]
        for i in range(1, 4):
            cv2.line(frame, (width * i // 4, 0), (width * i // 4, height),
                     (70, 70, 70), 1)
        for i in range(1, 3):
            cv2.line(frame, (0, height * i // 3), (width, height * i // 3),
                     (70, 70, 70), 1)

        visited = set()
        for corners in self.trail:
            u, v = corners.mean(axis=0)
            visited.add((min(3, int(u / width * 4)), min(2, int(v / height * 3))))
        for cell_u, cell_v in visited:
            top_left = (cell_u * width // 4, cell_v * height // 3)
            bottom_right = ((cell_u + 1) * width // 4,
                            (cell_v + 1) * height // 3)
            cv2.rectangle(frame, top_left, bottom_right, (0, 200, 0), 2)

        cv2.putText(frame, f'coverage {len(visited)}/12 cells  '
                           f'{len(self.trail)} views',
                    (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 0), 2)

    def publish_debug_image(self, frame, header):
        debug_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        debug_msg.header = header
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
