#!/usr/bin/env python3
"""Drive the arm through a pose sweep and record hand-eye observation pairs.

At every pose this records the two halves the solver needs:

    T_base_ee    the flange pose in the robot base frame, from TF (forward
                 kinematics of the measured joint state)
    T_cam_target the marker pose in the camera frame, from the detector

and writes them to a JSON dataset for the solver to consume offline.

Pose generation aims the marker plate at a *nominal* camera location supplied
by configuration. That guess only decides where to point the target; it never
enters the calibration itself, which is what makes the result independent of
it. In a real cell the guess comes from a tape measure or the CAD model.

Conditioning drives the design of the sweep. Hand-eye calibration recovers
rotation from the relative motions between poses, and a set of poses that all
share one rotation axis leaves the solution underdetermined however many
samples it contains. Simply pointing the plate at the camera from a grid of
positions produces almost exactly that degenerate set, so each candidate also
receives a deliberate tilt about two axes and a roll about the viewing axis.

The geometry itself lives in pose_planning.py, which has no ROS dependency and
is therefore testable without a simulator. This node is the plumbing around
it: parameters in, MoveIt and TF in the middle, a JSON dataset out.
"""

import json
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener

from mycobot_calibration.moveit_client import MoveItPoseClient
from mycobot_calibration.pose_planning import (
    coverage_metrics,
    coverage_warnings,
    format_coverage,
    plan_marker_poses,
)
from mycobot_calibration.transforms import (
    average_transforms,
    invert_transform,
    pose_from_msg,
    pose_to_msg,
    transform_from_dict,
    transform_from_msg,
    transform_spread,
    transform_to_list,
)


class PoseCollectorNode(Node):

    def __init__(self):
        super().__init__('handeye_pose_collector')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('end_effector_frame', 'link6_flange')
        self.declare_parameter('marker_frame', 'aruco_marker_link')
        self.declare_parameter('planning_group', 'arm')
        self.declare_parameter('target_pose_topic', '/aruco_detector/target_pose')
        self.declare_parameter('output_file', '/tmp/handeye_dataset.json')

        # Nominal camera pose, used only to plan where the target will land in
        # the image. It is a rough prior -- from CAD or a tape measure on real
        # hardware -- and never enters the calibration itself.
        # These defaults are the calibration.world rig's actual optical frame,
        # i.e. camera_tilt_deg:=23.16 camera_pan_deg:=55.26 in the URDF, and
        # the stand at (0.287, -0.209, 0.281) plus the 0.01 m +Y offset that
        # camera_head_joint adds. They have twice been left behind when the
        # camera moved, which only bites a run started without the config
        # file: the plan then back-projects through the wrong direction and
        # the sweep quietly covers the wrong part of the frame. Update them
        # here AND in config/handeye_calibration.yaml whenever the stand moves.
        self.declare_parameter('nominal_camera_position', [0.287, 0.145, 0.450])
        self.declare_parameter('nominal_camera_rpy_deg', [-114.45, 0.0, 119.14])

        # Nominal intrinsics for back-projection. Must match the URDF camera.
        self.declare_parameter('image_width', 848)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('camera_hfov', 1.5184)

        # Coverage is specified in IMAGE space, not workspace space. Sampling a
        # box in the robot's frame and merely aiming the target at the camera
        # says nothing about where it lands in the picture: doing that here put
        # every sample in the left of the frame, spanning 21% of the width and
        # touching 4 of 12 image cells, with nothing near an edge. Corner
        # localisation error and any lens distortion vary across the field, so
        # a calibration fitted from one patch of the image is only valid there.
        self.declare_parameter('image_grid', [8, 6])

        # Border inset as a multiple of the plate's on-screen half-size. Above
        # 1.0 because a rolled plate's bounding box is larger than the square:
        # at the +/-30 deg of roll_angles_deg it grows by cos30 + sin30 = 1.37,
        # so 1.6 clears it with margin. The inset is applied per distance, so
        # raising this costs edge coverage at every range at once.
        self.declare_parameter('edge_safety', 1.6)

        # Coarse model of where the arm can put the marker, used to choose how
        # far along each camera ray to place it. Sampling fixed distances
        # instead wastes almost every candidate: the camera's frustum is far
        # larger than a 280 mm arm's workspace, and a first attempt at three
        # fixed distances left 6 of 60 candidates reachable. Intersecting each
        # ray with this sphere puts every candidate in the right neighbourhood
        # before /compute_ik is asked for a verdict.
        self.declare_parameter('workspace_centre', [0.0, 0.0, 0.1316])
        self.declare_parameter('workspace_radius_max', 0.26)
        self.declare_parameter('workspace_radius_min', 0.10)
        # The chord endpoints are dropped as tangent to the reachable volume,
        # so the nearest sample sits 1/(N+1) along the chord. At N=3 that threw
        # away the near quarter of the arm's reach and with it every close-up
        # view, which is where the board fills the frame.
        self.declare_parameter('samples_per_ray', 7)
        self.declare_parameter('min_marker_height', 0.02)

        # Applied about BOTH of the plate's in-plane axes, plus a roll about
        # the viewing axis. One tilt axis is not enough: the relative rotations
        # then span a rank-deficient set of axes, and a planar target held
        # nearly fronto-parallel is also where a planar pose solve is most
        # ambiguous. The collector reports the conditioning it achieved.
        self.declare_parameter('tilt_angles_deg', [-20.0, 0.0, 20.0])
        self.declare_parameter('roll_angles_deg', [-30.0, 0.0, 30.0])

        # Close-ups the image grid structurally cannot plan: its border inset
        # grows with the plate's on-screen size and rejects everything nearer
        # than 0.208 m, which caps the board at 62% of the frame. These are
        # planned separately and put at the front so max_poses keeps them.
        self.declare_parameter('close_up_poses', 6)
        self.declare_parameter('close_up_margin_px', 20.0)
        self.declare_parameter('max_poses', 40)
        # The whole PLATE, not the black square: this sizes the frame-border
        # inset and the fill-fraction score, both of which are about the
        # physical object's footprint in the image. The detector's own
        # marker_size is the black square (0.080) and is a different number.
        self.declare_parameter('marker_size', 0.0960)
        self.declare_parameter('shuffle_seed', 0)

        self.declare_parameter('settle_time', 1.5)
        self.declare_parameter('samples_per_pose', 8)
        self.declare_parameter('sample_timeout', 8.0)
        self.declare_parameter('no_detection_timeout', 2.0)
        self.declare_parameter('max_translation_spread', 0.004)
        self.declare_parameter('max_rotation_spread_deg', 3.0)
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('velocity_scaling', 0.3)

        self.base_frame = self.get_parameter('base_frame').value
        self.ee_frame = self.get_parameter('end_effector_frame').value
        self.marker_frame = self.get_parameter('marker_frame').value
        self.output_file = self.get_parameter('output_file').value
        self.settle_time = float(self.get_parameter('settle_time').value)
        self.samples_per_pose = int(self.get_parameter('samples_per_pose').value)
        self.sample_timeout = float(self.get_parameter('sample_timeout').value)
        self.no_detection_timeout = float(
            self.get_parameter('no_detection_timeout').value)
        self.max_t_spread = float(self.get_parameter('max_translation_spread').value)
        self.max_r_spread = math.radians(
            float(self.get_parameter('max_rotation_spread_deg').value))
        self.max_poses = int(self.get_parameter('max_poses').value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        detection_qos = QoSProfile(depth=20)
        detection_qos.reliability = ReliabilityPolicy.RELIABLE
        self.detections = []
        self.collecting = False
        self.create_subscription(
            PoseStamped, self.get_parameter('target_pose_topic').value,
            self.on_detection, detection_qos)

        self.moveit = MoveItPoseClient(
            self,
            group_name=self.get_parameter('planning_group').value,
            end_effector_link=self.ee_frame,
            base_frame=self.base_frame,
            planning_time=float(self.get_parameter('planning_time').value),
            velocity_scaling=float(self.get_parameter('velocity_scaling').value),
            acceleration_scaling=float(self.get_parameter('velocity_scaling').value),
        )

    # -- plumbing -----------------------------------------------------------

    def on_detection(self, msg):
        if self.collecting:
            self.detections.append(pose_from_msg(msg.pose))

    def sleep_sim(self, seconds):
        """Sleep while keeping callbacks flowing, on the wall clock."""
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)

    def lookup(self, target_frame, source_frame, timeout_sec=5.0):
        """T_target_source from TF, retrying until the timeout."""
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                tf = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, rclpy.time.Time())
                return transform_from_msg(tf.transform)
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError(f'TF {target_frame} <- {source_frame} unavailable')

    # -- pose generation ----------------------------------------------------

    def plan_poses(self):
        """Candidate marker poses from pose_planning, with the reach reported.

        The planner works in image space, so what it returns is only a plan:
        /compute_ik and the detector decide what is really collected. The
        numbers logged here are therefore about the *camera aim*, not about the
        dataset -- if much of the frame is out of the arm's reach, no choice of
        poses can cover the image and the camera has to move or re-aim first.
        """
        def parameter(name):
            return self.get_parameter(name).value

        poses, stats = plan_marker_poses(
            camera_position=parameter('nominal_camera_position'),
            camera_rpy_deg=parameter('nominal_camera_rpy_deg'),
            image_width=int(parameter('image_width')),
            image_height=int(parameter('image_height')),
            camera_hfov=float(parameter('camera_hfov')),
            image_grid=[int(v) for v in parameter('image_grid')],
            marker_size=float(parameter('marker_size')),
            edge_safety=float(parameter('edge_safety')),
            workspace_centre=parameter('workspace_centre'),
            workspace_radius_max=float(parameter('workspace_radius_max')),
            workspace_radius_min=float(parameter('workspace_radius_min')),
            samples_per_ray=int(parameter('samples_per_ray')),
            close_up_poses=int(parameter('close_up_poses')),
            close_up_margin_px=float(parameter('close_up_margin_px')),
            min_marker_height=float(parameter('min_marker_height')),
            tilt_angles_deg=parameter('tilt_angles_deg'),
            roll_angles_deg=parameter('roll_angles_deg'),
            shuffle_seed=int(parameter('shuffle_seed')),
        )

        reachable = stats['reachable_positions']
        total = stats['grid_positions']
        self.get_logger().info(
            f'{reachable}/{total} image positions fall inside the arm '
            f'workspace, giving {stats["candidates"]} candidate poses')

        # Warn well before half the frame is lost. The old threshold was "more
        # than half", which stayed silent at 16 of 30 -- a camera aim that
        # capped image coverage at 52% of the width reported nothing at all.
        if reachable < 0.75 * total:
            self.get_logger().warn(
                f'{total - reachable} of {total} image positions are out of '
                'the arm\'s reach, so that part of the frame cannot be '
                'covered by any pose set. Move the camera closer, re-aim it, '
                'or narrow its field of view so the workspace fills more of '
                'the view.')
        return poses

    def coverage_report(self, samples):
        """Where the accepted samples actually landed, scored and logged.

        The plan is only a plan: reachability and visibility decide what is
        really collected, so this measures the delivered coverage rather than
        the intended one, and checks it against every rule of thumb for a
        usable dataset rather than only the sample count.
        """
        if not samples:
            return {}

        report = coverage_metrics(
            [transform_from_dict(s['T_cam_target']) for s in samples],
            [transform_from_dict(s['T_base_ee']) for s in samples],
            int(self.get_parameter('image_width').value),
            int(self.get_parameter('image_height').value),
            float(self.get_parameter('camera_hfov').value),
            float(self.get_parameter('marker_size').value))

        self.get_logger().info(f'coverage: {format_coverage(report)}')
        for message in coverage_warnings(report):
            self.get_logger().warn(message)
        return report

    # -- collection ---------------------------------------------------------

    def collect_detections(self):
        """Gather a burst of detections and average them.

        Returns (T_cam_target, count) or (None, reason). Averaging beats a
        single frame because ArUco corner noise is close to zero-mean, and the
        spread check rejects a pose where the marker was only intermittently
        or marginally visible -- those produce outliers the solvers cannot
        reject on their own.
        """
        self.detections = []
        self.collecting = True
        start = time.monotonic()
        deadline = start + self.sample_timeout
        # Fail fast on a pose the camera simply cannot see. A reachable pose
        # that points the plate out of frame is common at the edges of the
        # candidate box, and waiting out the full sample timeout on each one
        # dominates the runtime of a sweep. If nothing at all has arrived
        # after several frame periods, nothing is going to.
        blind_deadline = start + self.no_detection_timeout
        while rclpy.ok() and len(self.detections) < self.samples_per_pose:
            now = time.monotonic()
            if now > deadline:
                break
            if not self.detections and now > blind_deadline:
                break
            rclpy.spin_once(self, timeout_sec=0.05)
        self.collecting = False

        samples = list(self.detections)
        if len(samples) < max(3, self.samples_per_pose // 2):
            return None, f'only {len(samples)} detections'

        t_spread, r_spread = transform_spread(samples)
        if t_spread > self.max_t_spread or r_spread > self.max_r_spread:
            return None, (f'unstable detection: {t_spread * 1000:.1f} mm / '
                          f'{math.degrees(r_spread):.1f} deg spread')

        return average_transforms(samples), len(samples)

    def run(self):
        self.get_logger().info('waiting for move_group and TF...')
        if not self.moveit.wait_for_servers():
            return 1

        T_ee_marker = self.lookup(self.ee_frame, self.marker_frame, timeout_sec=30.0)
        T_marker_ee = invert_transform(T_ee_marker)
        self.get_logger().info(
            f'marker is mounted at {np.round(T_ee_marker[:3, 3], 4).tolist()} '
            f'in {self.ee_frame}')

        candidates = self.plan_poses()
        self.get_logger().info(f'{len(candidates)} candidate marker poses')

        samples = []
        attempted = 0
        skipped_ik = 0
        failed_motion = 0
        failed_vision = 0

        for index, T_base_marker_goal in enumerate(candidates):
            if len(samples) >= self.max_poses:
                break
            attempted += 1

            # The planner is asked for a flange pose, not a marker pose: the
            # marker rides on the flange through a known fixed transform, and
            # constraining the tip link that the IK chain actually ends at
            # avoids depending on MoveIt's tip-substitution behaviour.
            T_base_ee_goal = T_base_marker_goal @ T_marker_ee
            goal_pose = pose_to_msg(T_base_ee_goal, Pose())

            reachable, reason = self.moveit.is_reachable(goal_pose)
            if not reachable:
                skipped_ik += 1
                continue

            ok, reason = self.moveit.move_to_pose(goal_pose)
            if not ok:
                failed_motion += 1
                self.get_logger().warn(f'pose {index}: motion failed ({reason})')
                continue

            self.sleep_sim(self.settle_time)

            T_cam_target, info = self.collect_detections()
            if T_cam_target is None:
                failed_vision += 1
                self.get_logger().warn(f'pose {index}: {info}')
                continue

            # Read FK *after* the detections so both describe the same settled
            # configuration. Reading it before would pair a pose from the tail
            # of the motion with images taken once the arm had stopped.
            T_base_ee = self.lookup(self.base_frame, self.ee_frame)

            samples.append({
                'index': index,
                'detections': info,
                'T_base_ee': transform_to_list(T_base_ee),
                'T_cam_target': transform_to_list(T_cam_target),
            })
            self.get_logger().info(
                f'captured {len(samples)}/{self.max_poses} '
                f'(pose {index}, {info} detections)')

        self.write_dataset(samples, T_ee_marker)
        self.get_logger().info(
            f'done: {len(samples)} samples from {attempted} candidates '
            f'({skipped_ik} unreachable, {failed_motion} motion failures, '
            f'{failed_vision} vision failures)')
        return 0 if len(samples) >= 3 else 1

    def write_dataset(self, samples, T_ee_marker):
        """Persist the observations plus enough context to interpret them."""
        ground_truth = None
        try:
            # Sim only. Recorded for scoring, never fed to the solver.
            camera_frame = 'camera_head_depth_optical_frame'
            ground_truth = transform_to_list(
                self.lookup(self.base_frame, camera_frame, timeout_sec=2.0))
        except RuntimeError:
            self.get_logger().info(
                'no camera frame in TF; dataset will carry no ground truth')

        payload = {
            'base_frame': self.base_frame,
            'end_effector_frame': self.ee_frame,
            'marker_frame': self.marker_frame,
            'T_ee_marker': transform_to_list(T_ee_marker),
            'ground_truth_T_base_cam': ground_truth,
            'coverage': self.coverage_report(samples),
            'samples': samples,
        }

        directory = os.path.dirname(os.path.abspath(self.output_file))
        os.makedirs(directory, exist_ok=True)
        with open(self.output_file, 'w') as handle:
            json.dump(payload, handle, indent=2)
        self.get_logger().info(f'dataset written to {self.output_file}')


def main(args=None):
    rclpy.init(args=args)
    node = PoseCollectorNode()
    status = 1
    try:
        status = node.run()
    except KeyboardInterrupt:
        node.get_logger().info('interrupted')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return status


if __name__ == '__main__':
    main()
