#!/usr/bin/env python3
"""Solve the eye-to-hand problem from a recorded dataset.

Usage:
    ros2 run mycobot_calibration handeye_solver --dataset /tmp/handeye_dataset.json

Reads the observation pairs written by the collector, runs every hand-eye
method OpenCV offers, scores them, and writes the winner to a YAML file that
calibration_publisher can broadcast.

The eye-to-hand reduction
-------------------------
cv2.calibrateHandEye solves the eye-in-hand problem: camera bolted to the
gripper, target fixed in the world. Its constraint is that

    T_base_target = T_base_gripper . X . T_cam_target       is constant,

where X = T_gripper_cam is what it returns.

Here the geometry is the other way round -- camera bolted to the world, target
bolted to the gripper -- and the invariant is

    T_gripper_target = T_gripper_base . X' . T_cam_target   is constant,

with X' = T_base_cam. The two equations are the same equation with
T_base_gripper replaced by its inverse and X replaced by X'. So feeding the
solver the *inverted* robot poses makes its output the camera pose in the
robot base frame, which is exactly the eye-to-hand answer. No other change is
needed, and nothing about the marker's mounting offset has to be known: it
cancels as the constant on the right-hand side.

OpenCV's own argument names use the opposite direction convention from this
package (its "gripper2base" is our T_base_gripper), so the crossing is done in
one place, in solve(), and nowhere else.
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

from mycobot_calibration.transforms import (
    invert_transform,
    make_transform,
    matrix_to_euler,
    matrix_to_quaternion,
    transform_difference,
    transform_from_dict,
)

METHODS = {
    'TSAI': cv2.CALIB_HAND_EYE_TSAI,
    'PARK': cv2.CALIB_HAND_EYE_PARK,
    'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
    'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
    'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def load_dataset(path):
    with open(path) as handle:
        payload = json.load(handle)

    samples = payload.get('samples', [])
    if len(samples) < 3:
        raise ValueError(
            f'{path} holds {len(samples)} samples; hand-eye calibration needs '
            'at least 3, and realistically 10 or more with varied rotations')

    robot_poses = [transform_from_dict(s['T_base_ee']) for s in samples]
    target_poses = [transform_from_dict(s['T_cam_target']) for s in samples]
    return payload, robot_poses, target_poses


def solve(robot_poses, target_poses, method):
    """Estimate T_base_cam. Inputs are T_base_ee and T_cam_target lists."""
    # The eye-to-hand reduction: invert the robot poses on the way in.
    inverted = [invert_transform(T) for T in robot_poses]

    R_base2gripper = [T[:3, :3] for T in inverted]
    t_base2gripper = [T[:3, 3].reshape(3, 1) for T in inverted]
    R_target2cam = [T[:3, :3] for T in target_poses]
    t_target2cam = [T[:3, 3].reshape(3, 1) for T in target_poses]

    R, t = cv2.calibrateHandEye(
        R_base2gripper, t_base2gripper,
        R_target2cam, t_target2cam,
        method=METHODS[method])
    return make_transform(R, t.reshape(3))


def residuals(T_base_cam, robot_poses, target_poses):
    """Consistency of the solution, without reference to ground truth.

    For the true X, the marker's pose in the flange frame

        T_ee_marker = T_ee_base . X . T_cam_target

    is the same at every sample, because the plate is bolted on. The scatter of
    that quantity across the dataset is therefore a direct measure of the
    calibration error, and it is the only quality signal available on real
    hardware. Reported as RMS deviation from the mean.
    """
    estimates = [invert_transform(robot) @ T_base_cam @ target
                 for robot, target in zip(robot_poses, target_poses)]

    translations = np.array([T[:3, 3] for T in estimates])
    mean_translation = translations.mean(axis=0)

    # Rotation scatter is measured against the sample whose rotation is
    # closest to the others, which avoids picking an outlier as the reference.
    best_index, best_score = 0, float('inf')
    for i, T_i in enumerate(estimates):
        score = sum(transform_difference(T_i, T_j)[1] for T_j in estimates)
        if score < best_score:
            best_index, best_score = i, score

    rotation_errors = [transform_difference(estimates[best_index], T)[1]
                       for T in estimates]

    return {
        'translation_rms_mm': float(
            np.sqrt(((translations - mean_translation) ** 2).sum(axis=1).mean())
            * 1000.0),
        'translation_max_mm': float(
            np.linalg.norm(translations - mean_translation, axis=1).max()
            * 1000.0),
        'rotation_rms_deg': float(
            math.degrees(np.sqrt(np.mean(np.square(rotation_errors))))),
        'rotation_max_deg': float(math.degrees(max(rotation_errors))),
    }


def format_transform(T):
    rpy = matrix_to_euler(T[:3, :3])
    return (f'xyz = [{T[0, 3]:+.4f}, {T[1, 3]:+.4f}, {T[2, 3]:+.4f}] m   '
            f'rpy = [{math.degrees(rpy[0]):+.2f}, '
            f'{math.degrees(rpy[1]):+.2f}, {math.degrees(rpy[2]):+.2f}] deg')


def write_result(path, T, method, stats, payload, accuracy):
    """Write the calibration as YAML, by hand to avoid a PyYAML dependency."""
    q = matrix_to_quaternion(T[:3, :3])
    rpy = matrix_to_euler(T[:3, :3])
    camera_frame = 'camera_head_depth_optical_frame'

    lines = [
        '# Eye-to-hand calibration result: pose of the camera optical frame',
        '# in the robot base frame. Generated by mycobot_calibration.',
        f'# method: {method}',
        f'# samples: {len(payload.get("samples", []))}',
        '',
        'handeye_calibration:',
        f'  parent_frame: {payload.get("base_frame", "base_link")}',
        f'  child_frame: {camera_frame}',
        f'  method: {method}',
        f'  samples: {len(payload.get("samples", []))}',
        '  transform:',
        '    translation:',
        f'      x: {T[0, 3]:.6f}',
        f'      y: {T[1, 3]:.6f}',
        f'      z: {T[2, 3]:.6f}',
        '    rotation:',
        f'      x: {q[0]:.6f}',
        f'      y: {q[1]:.6f}',
        f'      z: {q[2]:.6f}',
        f'      w: {q[3]:.6f}',
        '    rpy_deg:',
        f'      roll: {math.degrees(rpy[0]):.4f}',
        f'      pitch: {math.degrees(rpy[1]):.4f}',
        f'      yaw: {math.degrees(rpy[2]):.4f}',
        '  residuals:',
        f'    translation_rms_mm: {stats["translation_rms_mm"]:.4f}',
        f'    rotation_rms_deg: {stats["rotation_rms_deg"]:.4f}',
    ]
    if accuracy is not None:
        lines += [
            '  ground_truth_error:',
            f'    translation_mm: {accuracy[0] * 1000.0:.4f}',
            f'    rotation_deg: {math.degrees(accuracy[1]):.4f}',
        ]
    lines.append('')

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='/tmp/handeye_dataset.json')
    parser.add_argument('--output', default='/tmp/handeye_result.yaml')
    parser.add_argument('--method', default='auto',
                        help='One of TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS, '
                             'or auto to pick the lowest-residual method.')
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    payload, robot_poses, target_poses = load_dataset(args.dataset)

    truth = payload.get('ground_truth_T_base_cam')
    T_truth = transform_from_dict(truth) if truth else None

    print(f'\nDataset : {args.dataset}')
    print(f'Samples : {len(robot_poses)}')
    print(f'Solving : T_{payload.get("base_frame", "base_link")}_camera '
          '(eye-to-hand)\n')

    header = (f'{"method":<12} {"consistency (RMS)":>22} '
              f'{"vs ground truth":>22}')
    print(header)
    print('-' * len(header))

    results = {}
    for name in METHODS:
        try:
            T = solve(robot_poses, target_poses, name)
        except cv2.error as exc:
            print(f'{name:<12} failed: {str(exc).splitlines()[-1][:50]}')
            continue

        stats = residuals(T, robot_poses, target_poses)
        accuracy = transform_difference(T_truth, T) if T_truth is not None else None
        results[name] = (T, stats, accuracy)

        consistency = (f'{stats["translation_rms_mm"]:6.2f} mm '
                       f'{stats["rotation_rms_deg"]:5.2f} deg')
        if accuracy is None:
            truth_column = '            n/a'
        else:
            truth_column = (f'{accuracy[0] * 1000.0:6.2f} mm '
                            f'{math.degrees(accuracy[1]):5.2f} deg')
        print(f'{name:<12} {consistency:>22} {truth_column:>22}')

    if not results:
        print('\nEvery method failed. The dataset is unusable.')
        return 1

    if args.method == 'auto':
        # Chosen on consistency alone, so the selection rule is identical on
        # real hardware where no ground truth exists.
        chosen = min(results,
                     key=lambda n: (results[n][1]['translation_rms_mm'] / 1000.0
                                    + math.radians(results[n][1]['rotation_rms_deg'])))
    else:
        chosen = args.method.upper()
        if chosen not in results:
            print(f'\nmethod {chosen} is not among the successful solves')
            return 1

    T, stats, accuracy = results[chosen]
    print(f'\nSelected: {chosen}'
          f'{" (lowest residual)" if args.method == "auto" else ""}')
    print(f'  T_base_camera   {format_transform(T)}')
    if T_truth is not None:
        print(f'  ground truth    {format_transform(T_truth)}')
        print(f'  error           {accuracy[0] * 1000.0:.2f} mm, '
              f'{math.degrees(accuracy[1]):.3f} deg')

    write_result(args.output, T, chosen, stats, payload, accuracy)
    print(f'\nWritten to {args.output}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
