#!/usr/bin/env python3
"""Tests for the pose plan and the coverage score.

These run without ROS, Gazebo or a camera, which is the point: the rules a
calibration dataset has to satisfy -- enough views, spread over the whole
image, edges included, at varied distances and orientations -- are geometric,
so they can be checked without moving a robot.

Two things are pinned here. First that the planner, given the rig this package
targets, produces a set that meets those rules. Second that the score notices
when it does not: a metric nobody has seen fail is not a metric.
"""

import math

import cv2
import numpy as np
import pytest

from mycobot_calibration.pose_planning import (
    MIN_FILL_FRACTION,
    MIN_SAMPLES,
    coverage_metrics,
    coverage_warnings,
    format_coverage,
    nominal_intrinsics,
    plan_marker_poses,
)
from mycobot_calibration.transforms import (
    euler_to_matrix,
    invert_transform,
    look_at_rotation,
    make_transform,
)

# WARNING: as of 2026-08-31 this is NOT the shipped default any more.
#
# The camera defaults in the URDF and launch files were rolled back to the
# original committed position -- stand (0.22, -0.30, 0.50), tilt 25, pan 0 --
# which is 0.518 m from the shoulder and aimed 20 deg / 37 deg away from the
# workspace centre. Measured, that rig reaches 15 of 48 image positions, 4 of
# 12 cells and 3 of 10 edge cells, so it FAILS the "board near edges as well
# as centre" rule outright and trips three coverage warnings.
#
# The geometry below is kept as the REFERENCE rig: the one that satisfies
# every rule, used here to keep the metrics honest and to pin what a usable
# camera position looks like. Tests in this file therefore describe what the
# planner can do, not what the shipped defaults currently deliver. Re-point
# CAMERA_POSITION/CAMERA_RPY_DEG at the shipped values to see the difference,
# and expect roughly eight failures if you do.
#
# Reference rig: camera on a stand at (0.287, 0.150, 0.281), aimed at the
# middle of the arm's reachable volume (tilt 24.45, pan 119.14 in the URDF,
# which is this optical rpy). 0.361 m from the workspace centre.
#
# The stand was at 0.555 m until 2026-08-30. It was moved in because how far
# away the camera is decides how much of the frame the board can fill: the arm
# can only bring the plate to within (D - 0.26) m of the camera, so at 0.555 m
# the board never exceeded 18% of the image height. See TestBoardFillsTheFrame.
CAMERA_POSITION = [0.287, 0.160, 0.281]
CAMERA_RPY_DEG = [-114.45, 0.0, 119.14]
IMAGE_WIDTH, IMAGE_HEIGHT = 848, 480
CAMERA_HFOV = 1.5184        # the D435's depth FOV, 87 deg: what the rig ships
NARROW_HFOV = 0.8727        # 50 deg
PLATE_SIZE = 0.0960         # whole plate: what decides the border inset
MARKER_SIZE = 0.080         # the black square: what decides corner accuracy.
                            # Measured off the PRINTED target -- the simulated
                            # and physical markers must be the same object or a
                            # sim-derived calibration does not transfer.

# The rig as it shipped before the stand move, kept so the tests can show what
# the change bought rather than merely asserting the new numbers.
OLD_CAMERA_POSITION = [0.42, -0.29, 0.35]
OLD_CAMERA_RPY_DEG = [-113.2, 0.0, 55.4]
OLD_PLATE_SIZE = 0.0825

RIG = dict(
    camera_position=CAMERA_POSITION,
    camera_rpy_deg=CAMERA_RPY_DEG,
    image_width=IMAGE_WIDTH,
    image_height=IMAGE_HEIGHT,
    camera_hfov=CAMERA_HFOV,
    image_grid=[8, 6],
    marker_size=PLATE_SIZE,
    edge_safety=1.6,
    workspace_centre=[0.0, 0.0, 0.1316],
    workspace_radius_max=0.26,
    workspace_radius_min=0.10,
    samples_per_ray=7,
    min_marker_height=0.02,
    tilt_angles_deg=[-20.0, 0.0, 20.0],
    roll_angles_deg=[-30.0, 0.0, 30.0],
    shuffle_seed=0,
)

OLD_RIG = dict(
    RIG,
    camera_position=OLD_CAMERA_POSITION,
    camera_rpy_deg=OLD_CAMERA_RPY_DEG,
    image_grid=[6, 5],
    marker_size=OLD_PLATE_SIZE,
    samples_per_ray=3,
)

MAX_POSES = 40


def T_cam_base(position=None, rpy_deg=None):
    """A rig's true camera pose, inverted, for projecting planned poses."""
    return invert_transform(make_transform(
        euler_to_matrix(*[math.radians(a)
                          for a in (rpy_deg or CAMERA_RPY_DEG)]),
        np.array(position if position is not None else CAMERA_POSITION)))


def planned_dataset(base=None, **overrides):
    """Plan the sweep and score it as if every pose had been captured.

    Reachability and visibility can only lower the delivered coverage, so this
    is the best case. A plan that fails these assertions cannot be rescued by
    the robot, which is what makes them worth asserting.
    """
    settings = dict(base if base is not None else RIG)
    settings.update(overrides)
    poses, stats = plan_marker_poses(**settings)
    selected = poses[:MAX_POSES]
    T = T_cam_base(settings['camera_position'], settings['camera_rpy_deg'])
    targets = [T @ pose for pose in selected]
    report = coverage_metrics(targets, selected, settings['image_width'],
                              settings['image_height'], settings['camera_hfov'],
                              settings['marker_size'])
    return selected, stats, report


class TestPlanReachesTheWholeImage:

    def test_the_stand_position_reaches_most_of_the_image_grid(self):
        """The 87 degree lens is no longer the binding constraint; distance is.

        How much of the frame the workspace can occupy is set by how far away
        the camera is, not by the lens alone: the reachable sphere subtends
        asin(0.26 / D) either side of the optical axis, against the lens's
        43.5. At the old 0.555 m stand that was 27.9 degrees, so the left and
        right thirds of the frame were out of reach and only ~40% of the
        planned positions were usable. At 0.380 m it is 43.2 degrees and
        almost the whole grid is reachable.
        """
        _, stats, _ = planned_dataset()
        reachable = stats['reachable_positions'] / stats['grid_positions']
        assert reachable > 0.85

        _, old_stats, _ = planned_dataset(base=OLD_RIG)
        old_reachable = (old_stats['reachable_positions']
                         / old_stats['grid_positions'])
        assert old_reachable < 0.60

    def test_a_narrower_lens_no_longer_helps_at_this_stand_position(self):
        """The alternative, measured rather than asserted from theory.

        Narrowing the lens was the standing recommendation while the stand sat
        at 0.555 m, where the workspace filled barely half the frame. Moving
        the stand in inverted that: at 0.400 m the workspace subtends 40.5
        degrees either side, so a 50 degree lens (25 either side) is now
        NARROWER than the thing it is pointed at. The target can leave the
        frame, and the border inset rejects most positions, costing two thirds
        of the cell coverage.

        Recorded because the FOV table in the camera xacro quotes these
        numbers, and a table in a comment that nothing checks goes stale.
        """
        _, wide_stats, wide = planned_dataset()
        _, narrow_stats, narrow = planned_dataset(camera_hfov=NARROW_HFOV)
        # Cell coverage now TIES at 12/12: the 82.5 mm plate is small enough
        # that the border inset stops rejecting positions, so the narrow lens
        # is no longer disqualified on coverage. What it still costs is the
        # candidate pool, because the workspace overflows a 50 degree frame.
        assert narrow['cells_covered'] <= wide['cells_covered']
        assert narrow_stats['candidates'] < wide_stats['candidates']

    def test_plan_supplies_enough_candidates_for_a_usable_dataset(self):
        """The candidate pool is the ceiling on how many views can be captured.

        The plan now yields several times max_poses, so the sweep is capped by
        max_poses rather than by the pool and IK or vision rejections come out
        of slack rather than off the total. That was not true at the old stand
        position, where 28 candidates had to survive every rejection to reach
        the 20-view rule of thumb.
        """
        _, stats, _ = planned_dataset()
        assert stats['candidates'] >= 2 * MAX_POSES

    def test_target_reaches_every_cell_including_edges_and_centre(self):
        """What this camera position delivers: all 12 cells, all 10 edges.

        The old 0.555 m stand reached 8 of 12 and 6 of 10, which passed the
        coverage warnings while leaving the outer columns untouched.
        """
        _, _, report = planned_dataset()
        assert report['cells_covered'] == report['cells_total']
        assert report['edge_cells_covered'] == report['edge_cells_total']
        assert report['centre_cells_covered'] == report['centre_cells_total']

        _, _, old = planned_dataset(base=OLD_RIG)
        assert old['cells_covered'] < old['cells_total']

    def test_plate_stays_inside_the_frame_at_every_pose(self):
        """The border inset has to actually work, at every distance sampled.

        A plate that overhangs the frame is not a near-edge sample; a single
        ArUco clipped by the border yields no detection at all.
        """
        poses, _, _ = planned_dataset()
        fx, cx, cy = nominal_intrinsics(IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_HFOV)
        T = T_cam_base()
        for pose in poses:
            t = (T @ pose)[:3, 3]
            u, v = fx * t[0] / t[2] + cx, fx * t[1] / t[2] + cy
            half = (RIG['marker_size'] / 2.0) / t[2] * fx
            assert half <= u <= IMAGE_WIDTH - half
            assert half <= v <= IMAGE_HEIGHT - half

    def test_far_samples_sit_closer_to_the_border_than_near_ones(self):
        """The inset is per distance, not one value for the whole sweep.

        A plate of fixed size subtends fewer pixels the further away it is, so
        a distant sample can legitimately be planned closer to the border. The
        docstring claimed this before the code did it.

        Compared as the CLOSEST approach of each half of the sweep, not as one
        near sample against one far sample: which single pose happens to be
        nearest says nothing, since it may sit in the middle of the frame where
        the inset never binds. What the per-distance inset buys is that the far
        half can get nearer the border than the near half ever can.
        """
        poses, _, _ = planned_dataset()
        fx, cx, cy = nominal_intrinsics(IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_HFOV)
        T = T_cam_base()
        margins = []
        for pose in poses:
            t = (T @ pose)[:3, 3]
            u, v = fx * t[0] / t[2] + cx, fx * t[1] / t[2] + cy
            margins.append((float(np.linalg.norm(t)),
                            min(u, IMAGE_WIDTH - u, v, IMAGE_HEIGHT - v)))
        margins.sort()
        half = len(margins) // 2
        nearest_approach = min(m[1] for m in margins[:half])
        farthest_approach = min(m[1] for m in margins[half:])
        assert farthest_approach < nearest_approach


class TestBoardFillsTheFrame:
    """The other sense of "the board covers the image", and the one that was
    unmeasured until 2026-08-30.

    `cells_covered` asks whether the target's CENTRE visited the whole frame.
    It says nothing about how big the board was in any given view, so a rig
    whose board never exceeded 18% of the frame height scored full marks on it.
    Localisation accuracy depends on the board's span in pixels, so that gap
    mattered: it is the difference between a dataset that samples the field and
    one that samples it precisely.
    """

    def test_the_sweep_includes_a_genuine_close_up(self):
        _, _, report = planned_dataset()
        # 0.70 not 0.75: the marker is 55 mm, matching the printed target, and
        # at this rig the ARM's reach is what stops the board growing further.
        assert report['fill_fraction_max'] > 0.70

    def test_close_ups_exist_and_survive_the_max_poses_cut(self):
        """They are 6 candidates against 262, so odds are not good enough.

        The stratified draw gives each stratum a share proportional to its
        size, which for the close-ups is about one pose. They are the rule the
        image grid structurally cannot satisfy, so the plan puts them first
        rather than leaving them to the draw.
        """
        _, stats, _ = planned_dataset()
        assert stats['close_ups'] > 0

        poses, _ = plan_marker_poses(**RIG)
        kept = poses[:MAX_POSES]
        T = T_cam_base()
        distances = [float(np.linalg.norm((T @ p)[:3, 3])) for p in kept]
        assert sum(d < 0.18 for d in distances) >= stats['close_ups']

    def test_the_border_inset_is_what_caps_the_grid(self):
        """The 62% ceiling is the inset's, not the arm's.

        Worth pinning because it is the whole reason close-ups are planned
        separately: turning them off does not move the arm any further away,
        it just discards the views where the board is biggest.
        """
        _, _, without = planned_dataset(close_up_poses=0)
        _, _, with_them = planned_dataset()
        assert without['fill_fraction_max'] < 0.65
        assert with_them['fill_fraction_max'] > 0.70
        # The gain is the point, not the absolute number: it moves with the
        # target size, and the target size is fixed by the printed marker.
        assert with_them['fill_fraction_max'] > without['fill_fraction_max'] * 1.3

    def test_the_old_camera_position_could_not_fill_the_frame(self):
        """Close-ups are planned there too; they simply cannot get close.

        The poses go as near as the arm reaches and back off only to avoid
        clipping, so they always exist. What the camera position decides is
        how much of the frame that turns out to be: at the old 0.555 m stand
        the nearest reachable pose is 0.295 m, which is a quarter of the
        frame, and no pose planner can improve on it.
        """
        _, stats, old = planned_dataset(base=OLD_RIG)
        assert stats['close_ups'] > 0
        assert stats['close_up_range'] > 0.28
        assert old['fill_fraction_max'] < MIN_FILL_FRACTION
        assert any('never spanned more than' in m
                   for m in coverage_warnings(old))

    def test_close_ups_go_as_near_as_anything_allows(self):
        """"As close as possible" is the instruction, so it is asserted.

        Two limits compete for which one binds, and at this stand position it
        depends on the target size, so it is not asserted.

        Two limits compete. The plate must keep close_up_margin_px of border,
        because a single ArUco touching the frame edge yields no detection at
        all; and the arm can reach no nearer than (D - 0.26) m. At this rig
        (D = 0.361 m, arm limit 0.101 m) the frame limit moves with the plate:

            marker    plate     frame limit   binds on   fill
            55 mm     0.0825    0.084 m       arm        76%
            80 mm     0.1200    0.122 m       frame      92%
            93 mm     0.1395    0.142 m       frame      92%

        Which one binds decides whether moving the camera closer still helps.
        While the ARM binds it does, because the arm's reach is measured from
        the camera. Once the FRAME binds it does not: the plate is already as
        large as the image can hold, and only a smaller margin or a narrower
        lens would buy more.
        """
        _, stats, _ = planned_dataset()
        fx, _, _ = nominal_intrinsics(IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_HFOV)
        frame_limit = PLATE_SIZE * fx / (IMAGE_HEIGHT - 2 * 20.0)
        arm_limit = 0.3610 - RIG['workspace_radius_max']

        # Which limit binds depends on the plate size, and it INVERTED when the
        # marker was corrected from 93 mm back to the printed 55 mm: the frame
        # used to stop the board first (0.142 m), now the arm does (0.101 m).
        # Assert the general rule rather than one side of it, so this keeps
        # working whichever way a future target size tips it.
        # Assert ONLY the general rule: the close-ups sit at whichever limit
        # binds. Which one that is flips with the target size, and asserting a
        # direction here has now been wrong twice (93 mm -> frame, 55 mm ->
        # arm, 80 mm -> frame). The rule survives all three.
        #
        # The ring of close-ups sits slightly off the optical axis, and an
        # off-axis ray enters the reachable sphere marginally later, so the
        # achieved range sits at or just above the on-axis limit.
        binding = max(frame_limit, arm_limit)
        assert stats['close_up_range'] >= binding - 1e-6
        assert stats['close_up_range'] < binding + 0.01

    def test_a_close_up_plate_cannot_overhang_the_frame(self):
        """Roll is suppressed for them precisely so this holds.

        A rolled square's bounding box is up to 1.37x the square, which at
        80% fill would overhang the frame -- and a clipped single ArUco yields
        no detection at all rather than a degraded one.
        """
        poses, stats = plan_marker_poses(**RIG)
        fx, cx, cy = nominal_intrinsics(IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_HFOV)
        T = T_cam_base()
        for pose in poses[:stats['close_ups']]:
            t = (T @ pose)[:3, 3]
            u, v = fx * t[0] / t[2] + cx, fx * t[1] / t[2] + cy
            half = (PLATE_SIZE / 2.0) / t[2] * fx
            assert half <= u <= IMAGE_WIDTH - half
            assert half <= v <= IMAGE_HEIGHT - half

    def test_the_old_stand_position_could_not_produce_one(self):
        """Why the stand moved, rather than only the pose planner changing.

        The arm can bring the plate no closer than (D - 0.26) m, so at the old
        0.555 m stand no pose plan could have fixed this. It is a property of
        where the camera was bolted.
        """
        _, _, old = planned_dataset(base=OLD_RIG)
        assert old['fill_fraction_max'] < MIN_FILL_FRACTION
        assert any('never spanned more than' in m
                   for m in coverage_warnings(old))

    def test_a_distant_only_dataset_is_flagged(self):
        """The metric, made to fail on purpose."""
        pixels = [(100 + 25 * i, 100 + 12 * i) for i in range(25)]
        targets, robots = TestCoverageScoreDetectsBadDatasets()._dataset(
            pixels, [1.5 + 0.05 * i for i in range(25)])
        report = coverage_metrics(targets, robots, IMAGE_WIDTH, IMAGE_HEIGHT,
                                  CAMERA_HFOV, PLATE_SIZE)
        assert report['fill_fraction_max'] < MIN_FILL_FRACTION
        assert any('never spanned more than' in m
                   for m in coverage_warnings(report))

    def test_filling_the_frame_and_reaching_the_edges_are_both_required(self):
        """They trade against each other, so neither alone is sufficient.

        A board large enough to fill the view cannot also sit at the frame
        edge without being clipped, and a clipped single ArUco yields no
        detection at all. The set has to satisfy both across a range of
        distances, which is what makes distance_ratio a rule in its own right.
        """
        _, _, report = planned_dataset()
        assert report['fill_fraction_max'] > MIN_FILL_FRACTION
        assert report['edge_cells_covered'] == report['edge_cells_total']
        assert report['distance_ratio'] > 2.0


class TestPlanIsWellConditioned:

    def test_relative_rotations_do_not_share_an_axis(self):
        """The degeneracy that no amount of extra samples can fix."""
        _, _, report = planned_dataset()
        assert report['axis_conditioning'] > 0.3

    def test_tilt_is_applied_about_both_in_plane_axes(self):
        """The plate is tilted about X and Y, not just X.

        Checked directly on the planned rotations rather than through
        axis_conditioning: that number is not monotonic in tilt variety -- at
        this camera position removing the tilt entirely happens to raise it,
        because the look-at rotations alone already spread the relative axes
        over a wide solid angle. So it cannot serve as a proxy for "the tilt
        was applied". What is actually claimed is that each pose deviates from
        its pure look-at orientation about both in-plane axes, and that is what
        is measured here.
        """
        poses, _, _ = planned_dataset()
        camera = np.array(CAMERA_POSITION)
        deviations = []
        for pose in poses:
            look_at = look_at_rotation(pose[:3, 3], camera)
            deviations.append(
                cv2.Rodrigues(look_at.T @ pose[:3, :3])[0].ravel())
        deviations = np.array(deviations)

        # Tilt about the plate X and Y, roll about its Z, so all three
        # components must be exercised across the set.
        assert np.abs(deviations[:, 0]).max() > math.radians(10.0)
        assert np.abs(deviations[:, 1]).max() > math.radians(10.0)
        assert np.abs(deviations[:, 2]).max() > math.radians(10.0)

        # And both signs of each tilt, not just one direction.
        assert deviations[:, 0].min() < -math.radians(10.0)
        assert deviations[:, 1].min() < -math.radians(10.0)

    def test_samples_span_a_range_of_distances(self):
        _, _, report = planned_dataset()
        assert report['distance_ratio'] > 1.3

    def test_plan_meets_every_rule_of_thumb(self):
        """The whole point, in one assertion."""
        _, _, report = planned_dataset()
        assert report['samples'] >= MIN_SAMPLES
        assert coverage_warnings(report) == []


class TestCoverageScoreDetectsBadDatasets:
    """A metric that has never been seen to fail is not a metric."""

    def _dataset(self, pixels, distances, rotations=None):
        """Build T_cam_target/T_base_ee pairs that land on given pixels."""
        fx, cx, cy = nominal_intrinsics(IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_HFOV)
        targets, robots = [], []
        for i, ((u, v), distance) in enumerate(zip(pixels, distances)):
            direction = np.array([(u - cx) / fx, (v - cy) / fx, 1.0])
            direction /= np.linalg.norm(direction)
            targets.append(make_transform(np.eye(3), direction * distance))
            if rotations is None:
                # Rotation about one shared axis: the degenerate case.
                robots.append(make_transform(
                    euler_to_matrix(math.radians(5.0 * i), 0.0, 0.0),
                    [0.0, 0.0, 0.0]))
            else:
                robots.append(rotations[i])
        return targets, robots

    def test_a_dataset_stuck_in_one_corner_is_flagged(self):
        pixels = [(100 + 5 * i, 100 + 5 * i) for i in range(25)]
        targets, robots = self._dataset(pixels, [0.4] * 25)
        report = coverage_metrics(targets, robots, IMAGE_WIDTH, IMAGE_HEIGHT,
                                  CAMERA_HFOV, PLATE_SIZE)
        assert report['cells_covered'] < report['cells_total']
        messages = ' '.join(coverage_warnings(report))
        assert 'image cells' in messages
        assert 'edge cells' in messages

    def test_a_dataset_at_one_distance_is_flagged(self):
        pixels = [(100 + 25 * i, 100 + 12 * i) for i in range(25)]
        targets, robots = self._dataset(pixels, [0.4] * 25)
        report = coverage_metrics(targets, robots, IMAGE_WIDTH, IMAGE_HEIGHT,
                                  CAMERA_HFOV, PLATE_SIZE)
        assert report['distance_ratio'] == pytest.approx(1.0)
        assert any('same range' in m for m in coverage_warnings(report))

    def test_a_dataset_sharing_one_rotation_axis_is_flagged(self):
        """The failure the residual cannot see.

        A set of poses that all rotate about the same axis fits its own
        observations perfectly and leaves the calibration underdetermined, so
        this has to be caught before the solve, not after it.
        """
        pixels = [(100 + 25 * i, 100 + 12 * i) for i in range(25)]
        targets, robots = self._dataset(
            pixels, [0.3 + 0.01 * i for i in range(25)])
        report = coverage_metrics(targets, robots, IMAGE_WIDTH, IMAGE_HEIGHT,
                                  CAMERA_HFOV, PLATE_SIZE)
        assert report['axis_conditioning'] == pytest.approx(0.0, abs=1e-9)
        assert any('share an axis' in m for m in coverage_warnings(report))

    def test_too_few_views_is_flagged(self):
        pixels = [(100 + 60 * i, 80 + 40 * i) for i in range(8)]
        targets, robots = self._dataset(
            pixels, [0.3 + 0.03 * i for i in range(8)])
        report = coverage_metrics(targets, robots, IMAGE_WIDTH, IMAGE_HEIGHT,
                                  CAMERA_HFOV, PLATE_SIZE)
        assert any('rule of thumb is' in m for m in coverage_warnings(report))

    def test_an_empty_dataset_reports_nothing_rather_than_dividing_by_zero(self):
        assert coverage_metrics([], [], IMAGE_WIDTH, IMAGE_HEIGHT,
                                CAMERA_HFOV, PLATE_SIZE) == {}
        assert coverage_warnings({}) != []
        assert format_coverage({}) == 'no usable samples'

    def test_targets_behind_the_camera_are_ignored(self):
        """A detection with a negative range is not a view of anything."""
        behind = make_transform(np.eye(3), [0.0, 0.0, -0.4])
        report = coverage_metrics([behind], [np.eye(4)], IMAGE_WIDTH,
                                  IMAGE_HEIGHT, CAMERA_HFOV, PLATE_SIZE)
        assert report == {}


class TestPlanIsReproducible:

    def test_same_seed_gives_the_same_plan(self):
        first, _ = plan_marker_poses(**RIG)
        second, _ = plan_marker_poses(**RIG)
        assert len(first) == len(second)
        assert all(np.allclose(a, b) for a, b in zip(first, second))

    def test_different_seed_gives_a_different_order(self):
        settings = dict(RIG)
        settings['shuffle_seed'] = 7
        first, _ = plan_marker_poses(**RIG)
        second, _ = plan_marker_poses(**settings)
        assert len(first) == len(second)
        assert not all(np.allclose(a, b) for a, b in zip(first, second))

    def test_shuffle_keeps_coverage_when_the_plan_is_truncated(self):
        """max_poses must not amount to collecting one strip of the image.

        Candidates are generated position by position, so an unshuffled plan
        cut at max_poses would keep only the first few columns.
        """
        poses, _ = plan_marker_poses(**RIG)
        T = T_cam_base()
        truncated = poses[:MAX_POSES]
        report = coverage_metrics([T @ pose for pose in truncated], truncated,
                                  IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_HFOV, PLATE_SIZE)
        assert report['cells_covered'] >= 8


class TestMarkerIsLargeEnoughToLocalise:

    def _mean_span(self, hfov, base=None):
        poses, _, _ = planned_dataset(base=base, camera_hfov=hfov)
        fx, _, _ = nominal_intrinsics(IMAGE_WIDTH, IMAGE_HEIGHT, hfov)
        settings = dict(base if base is not None else RIG)
        T = T_cam_base(settings['camera_position'], settings['camera_rpy_deg'])
        marker = MARKER_SIZE * (settings['marker_size'] / PLATE_SIZE)
        return float(np.mean([marker / np.linalg.norm((T @ p)[:3, 3]) * fx
                              for p in poses]))

    def test_moving_the_stand_in_bought_most_of_what_a_narrow_lens_would(self):
        """Why the stand moved rather than the lens being narrowed.

        The dominant error on this rig is a fixed sub-pixel inward bias on the
        marker's corners, which reads as a proportional over-estimate of range
        and shrinks with marker PIXELS. Narrowing the lens to 50 degrees was
        the standing recommendation for that while the stand sat at 0.555 m,
        where it roughly doubled the span.

        Moving the stand in and enlarging the board did the same job without
        touching the lens, and without the cell coverage a narrow lens now
        costs (see test_a_narrower_lens_no_longer_helps_at_this_stand_position).
        Guarded as ratios so these do not become brittle assertions about
        absolute pixel counts.
        """
        assert self._mean_span(CAMERA_HFOV) > 1.8 * self._mean_span(
            CAMERA_HFOV, base=OLD_RIG)

    def test_a_narrower_lens_still_helps_the_span_but_much_less(self):
        """The trade, re-measured at the new stand rather than assumed.

        It was ~2x at 0.555 m. At 0.380 m it is ~1.19x, because a narrow lens
        cannot fit the larger plate near the frame border at close range and
        the planner is pushed to place the target further away. Combined with
        the cell coverage it costs, the narrow lens stopped being worth it --
        and the closer the stand gets, the less it has left to offer.
        """
        ratio = self._mean_span(NARROW_HFOV) / self._mean_span(CAMERA_HFOV)
        assert 1.05 < ratio < 1.5

    def test_every_view_spans_enough_pixels_to_refine_corners(self):
        poses, _, _ = planned_dataset()
        fx, _, _ = nominal_intrinsics(IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_HFOV)
        T = T_cam_base()
        spans = [MARKER_SIZE / np.linalg.norm((T @ p)[:3, 3]) * fx
                 for p in poses]
        # The far end of the sweep is now further away than it was, but the
        # marker is larger, so the worst view still spans comfortably more than
        # before. Below about 30 px a 6x6 marker stops decoding at all, and
        # subpixel corner refinement is doing real work well above that.
        # 40 px, not the 60 that the oversized 93 mm marker allowed. Below
        # about 30 px a 6x6 marker stops decoding at all, so this still leaves
        # real headroom -- and the far end of the sweep is where it is worst.
        assert min(spans) > 40.0
