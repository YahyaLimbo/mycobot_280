#!/usr/bin/env python3
"""Round-trip tests for the eye-to-hand reduction and the transform helpers.

These run without ROS, Gazebo or a camera. The point is to be able to tell a
wrong answer caused by the maths apart from one caused by the rig: if these
pass and a real sweep still comes out wrong, the fault is in the data, not in
solve().
"""

import json
import math
import os
import tempfile

import numpy as np
import pytest

from mycobot_calibration.handeye_solver import (
    METHODS,
    load_dataset,
    residuals,
    solve,
)
from mycobot_calibration.transforms import (
    average_transforms,
    euler_to_matrix,
    invert_transform,
    look_at_rotation,
    make_transform,
    matrix_to_euler,
    matrix_to_quaternion,
    quaternion_to_matrix,
    transform_difference,
    transform_to_list,
)

# The rig this package targets: camera on the fixed stand, marker on the
# flange 50 mm out and flipped to face back down the tool axis.
TRUE_T_BASE_CAM = make_transform(
    euler_to_matrix(math.radians(-115.0), 0.0, 0.0), [0.220, -0.290, 0.500])
TRUE_T_EE_MARKER = make_transform(
    euler_to_matrix(math.pi, 0.0, 0.0), [0.0, -0.05, -0.01])


def random_robot_poses(count, seed=0, rotation_spread_deg=30.0):
    """Plausible flange poses with genuinely varied rotation axes.

    The default spread is deliberately modest. Tsai-Lenz parametrises each
    relative rotation as 2*sin(theta/2) about its axis, which loses all
    conditioning as theta approaches 180 degrees, and the dual-quaternion
    method of Daniilidis degrades the same way. A fixture drawing orientations
    uniformly from +-90 degrees produces relative rotations up to 178 degrees
    and makes both methods miss by hundreds of millimetres on otherwise
    perfect data -- see test_tsai_degrades_near_180_degree_relative_rotations,
    which pins that behaviour down. Real calibration sweeps, including the one
    this package generates, stay well inside that limit.
    """
    rng = np.random.default_rng(seed)
    spread = math.radians(rotation_spread_deg)
    poses = []
    for _ in range(count):
        position = np.array([
            rng.uniform(-0.05, 0.15),
            rng.uniform(-0.12, 0.12),
            rng.uniform(0.28, 0.45),
        ])
        rpy = rng.uniform(-spread, spread, size=3)
        poses.append(make_transform(euler_to_matrix(*rpy), position))
    return poses


def synthesise(robot_poses, noise_translation=0.0, noise_rotation=0.0, seed=1):
    """Exact camera observations implied by the ground-truth geometry.

    T_cam_marker = inv(T_base_cam) . T_base_ee . T_ee_marker
    """
    rng = np.random.default_rng(seed)
    T_cam_base = invert_transform(TRUE_T_BASE_CAM)
    observations = []
    for T_base_ee in robot_poses:
        T = T_cam_base @ T_base_ee @ TRUE_T_EE_MARKER
        if noise_translation > 0.0 or noise_rotation > 0.0:
            perturbation = make_transform(
                euler_to_matrix(*rng.normal(0.0, noise_rotation, size=3)),
                rng.normal(0.0, noise_translation, size=3))
            T = T @ perturbation
        observations.append(T)
    return observations


class TestTransformHelpers:

    def test_quaternion_matrix_round_trip(self):
        for rpy in [(0.1, 0.2, 0.3), (math.pi, 0.0, 0.0),
                    (-2.007, 0.0, 0.0), (0.0, math.pi / 2 - 1e-4, 0.0)]:
            R = euler_to_matrix(*rpy)
            q = matrix_to_quaternion(R)
            assert np.allclose(quaternion_to_matrix(*q), R, atol=1e-9)

    def test_euler_round_trip(self):
        for rpy in [(0.1, 0.2, 0.3), (math.pi, 0.0, 0.0), (-2.007, 0.0, 0.0)]:
            R = euler_to_matrix(*rpy)
            assert np.allclose(euler_to_matrix(*matrix_to_euler(R)), R, atol=1e-9)

    def test_invert_transform(self):
        T = make_transform(euler_to_matrix(0.3, -0.7, 1.1), [0.1, -0.2, 0.3])
        assert np.allclose(T @ invert_transform(T), np.eye(4), atol=1e-12)

    def test_look_at_rotation_points_z_at_target(self):
        eye = np.array([0.05, 0.0, 0.35])
        target = np.array([0.22, -0.29, 0.50])
        R = look_at_rotation(eye, target)
        direction = (target - eye) / np.linalg.norm(target - eye)
        assert np.allclose(R[:, 2], direction, atol=1e-9)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0)

    def test_average_of_identical_transforms_is_that_transform(self):
        T = make_transform(euler_to_matrix(0.2, 0.4, -0.6), [0.1, 0.2, 0.3])
        assert np.allclose(average_transforms([T] * 5), T, atol=1e-12)

    def test_average_reduces_zero_mean_noise(self):
        rng = np.random.default_rng(7)
        T = make_transform(euler_to_matrix(0.2, 0.4, -0.6), [0.1, 0.2, 0.3])
        noisy = [T @ make_transform(
            euler_to_matrix(*rng.normal(0, 0.01, 3)), rng.normal(0, 0.002, 3))
            for _ in range(200)]
        mean_error = transform_difference(T, average_transforms(noisy))
        single_error = transform_difference(T, noisy[0])
        assert mean_error[0] < single_error[0]
        assert mean_error[0] < 5e-4


class TestSolver:

    @pytest.mark.parametrize('method', sorted(METHODS))
    def test_recovers_ground_truth_without_noise(self, method):
        """Every method must nail an exact dataset.

        This is the test that would catch an inverted or transposed transform
        in the eye-to-hand reduction: a direction error still produces a
        confident answer, just the wrong one.
        """
        robot_poses = random_robot_poses(15)
        observations = synthesise(robot_poses)

        T = solve(robot_poses, observations, method)
        translation_error, rotation_error = transform_difference(
            TRUE_T_BASE_CAM, T)

        assert translation_error < 1e-6, f'{method}: {translation_error} m off'
        assert rotation_error < 1e-6, f'{method}: {rotation_error} rad off'

    def test_residuals_are_zero_for_exact_solution(self):
        robot_poses = random_robot_poses(15)
        observations = synthesise(robot_poses)
        stats = residuals(TRUE_T_BASE_CAM, robot_poses, observations)
        assert stats['translation_rms_mm'] < 1e-6
        assert stats['rotation_rms_deg'] < 1e-6

    def test_residuals_grow_with_a_wrong_solution(self):
        """A displaced solution must be visible in the consistency metric.

        This is what makes residuals usable as the selection rule on real
        hardware, where there is no ground truth to compare against.
        """
        robot_poses = random_robot_poses(15)
        observations = synthesise(robot_poses)
        wrong = TRUE_T_BASE_CAM.copy()
        wrong[0, 3] += 0.02
        stats = residuals(wrong, robot_poses, observations)
        assert stats['translation_rms_mm'] > 1.0

    def test_tolerates_realistic_detection_noise(self):
        """1 mm / 0.5 deg per-observation noise should still land within 5 mm."""
        robot_poses = random_robot_poses(25, seed=3)
        observations = synthesise(
            robot_poses, noise_translation=0.001,
            noise_rotation=math.radians(0.5), seed=5)

        T = solve(robot_poses, observations, 'PARK')
        translation_error, rotation_error = transform_difference(
            TRUE_T_BASE_CAM, T)

        assert translation_error < 0.005
        assert rotation_error < math.radians(1.5)

    def test_tsai_degrades_near_180_degree_relative_rotations(self):
        """Pin down which methods survive large relative rotations.

        On exact data with relative rotations reaching ~178 degrees, PARK,
        HORAUD and ANDREFF stay exact while TSAI and DANIILIDIS are off by
        hundreds of millimetres. That is a property of their rotation
        parametrisations, not a bug here, and it is the reason the solver
        selects on residuals rather than defaulting to the best-known name.

        It also serves as the cross-check on the eye-to-hand reduction
        itself: three independently derived algorithms agreeing to 1e-9 on
        synthesised data means solve() inverts the robot poses correctly.
        """
        robot_poses = random_robot_poses(15, rotation_spread_deg=90.0)
        observations = synthesise(robot_poses)

        errors = {}
        for method in METHODS:
            T = solve(robot_poses, observations, method)
            errors[method] = transform_difference(TRUE_T_BASE_CAM, T)[0]

        for method in ('PARK', 'HORAUD', 'ANDREFF'):
            assert errors[method] < 1e-6, f'{method} should be exact'
        for method in ('TSAI', 'DANIILIDIS'):
            assert errors[method] > 0.01, (
                f'{method} unexpectedly coped with near-180 degree relative '
                'rotations; the fixture may no longer be degenerate')

    def test_pure_translation_dataset_is_ill_conditioned(self):
        """Poses sharing one orientation must not silently look successful.

        Guards the rationale for the tilt and roll perturbations in the
        collector: without rotation variety the problem is underdetermined,
        and this asserts that such a dataset really does fail rather than
        quietly returning something plausible.
        """
        rng = np.random.default_rng(11)
        fixed_rotation = euler_to_matrix(0.2, -0.3, 0.5)
        robot_poses = [
            make_transform(fixed_rotation, rng.uniform(-0.1, 0.1, size=3))
            for _ in range(15)
        ]
        observations = synthesise(robot_poses)

        errors = []
        for method in METHODS:
            try:
                T = solve(robot_poses, observations, method)
            except Exception:
                errors.append(float('inf'))
                continue
            if not np.all(np.isfinite(T)):
                errors.append(float('inf'))
                continue
            errors.append(transform_difference(TRUE_T_BASE_CAM, T)[0])

        assert min(errors) > 1e-3, (
            'a rotation-free dataset resolved the camera pose, which means '
            'the test fixture is not actually degenerate')


class TestDatasetIO:

    def test_load_dataset_round_trip(self):
        robot_poses = random_robot_poses(12)
        observations = synthesise(robot_poses)

        payload = {
            'base_frame': 'base_link',
            'end_effector_frame': 'link6_flange',
            'ground_truth_T_base_cam': transform_to_list(TRUE_T_BASE_CAM),
            'samples': [
                {'index': i,
                 'T_base_ee': transform_to_list(robot),
                 'T_cam_target': transform_to_list(observation)}
                for i, (robot, observation) in enumerate(
                    zip(robot_poses, observations))
            ],
        }

        handle = tempfile.NamedTemporaryFile(
            'w', suffix='.json', delete=False)
        try:
            json.dump(payload, handle)
            handle.close()

            _, loaded_robot, loaded_target = load_dataset(handle.name)
            for original, loaded in zip(robot_poses, loaded_robot):
                assert np.allclose(original, loaded, atol=1e-9)

            T = solve(loaded_robot, loaded_target, 'PARK')
            assert transform_difference(TRUE_T_BASE_CAM, T)[0] < 1e-6
        finally:
            os.unlink(handle.name)

    def test_load_dataset_rejects_too_few_samples(self):
        handle = tempfile.NamedTemporaryFile(
            'w', suffix='.json', delete=False)
        try:
            json.dump({'samples': []}, handle)
            handle.close()
            with pytest.raises(ValueError, match='at least 3'):
                load_dataset(handle.name)
        finally:
            os.unlink(handle.name)
