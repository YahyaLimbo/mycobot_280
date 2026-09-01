#!/usr/bin/env python3
"""Cross-check this package's hand-eye solve against easy_handeye2's backend.

WHAT THIS DOES AND DOES NOT PROVE
---------------------------------
easy_handeye2 calls cv2.calibrateHandEye, and so do we. This is therefore NOT
an independent check of the algorithm -- both stacks would inherit the same bug
if OpenCV had one. What it checks is the PLUMBING, which is where eye-to-hand
actually goes wrong in practice:

  * the direction convention of every stored transform,
  * the eye-to-hand reduction (inverting the robot poses),
  * OpenCV's gripper2base/target2cam argument ordering,
  * which optical frame the detections are stamped in.

The two stacks reach the reduction by visibly different routes:

  ours   handeye_solver.solve() inverts T_base_ee explicitly and feeds the
         result to OpenCV as gripper2base.
  theirs handeye_sampler._get_transforms() flips the TF lookup instead --
         lookup_transform(effector_frame, base_frame) rather than
         (base_frame, effector_frame) -- under a comment that reads
         "here we trick the library (it is actually made for eye_in_hand
         only)". Same matrix, obtained a different way.

Agreement to machine precision means both readings of the geometry coincide.

WHY WE REBUILD THEIR SAMPLES RATHER THAN RECORD THEM
----------------------------------------------------
Their sampler reads live TF, which is not available offline. We reproduce
exactly what it would have recorded for calibration_type 'eye_on_base' (their
name for eye-to-hand):

    robot    = inv(T_base_ee)      == lookup_transform(effector, base)
    tracking = T_cam_target        == lookup_transform(camera, marker)

That is a format conversion, not a borrowing of our solver. Their backend file
runs unmodified -- verify with the md5 this script prints.

LICENCE
-------
easy_handeye2 is LGPLv3, so its source is NOT vendored here. Clone it yourself:

    git clone https://github.com/marcoesposito1988/easy_handeye2

and pass --easy-handeye2 <path>. Cross-checked against upstream commit
29bd50a (2026-08-20). Needs `pip install transforms3d`, which the backend
imports; nothing else beyond this package's own dependencies.
"""

import argparse
import hashlib
import os
import shutil
import sys
import tempfile

import numpy as np

from mycobot_calibration.handeye_solver import load_dataset, residuals, solve
from mycobot_calibration.transforms import (
    invert_transform, make_transform, matrix_to_quaternion,
    quaternion_to_matrix, transform_difference, transform_from_dict)

# ours -> theirs. Both name the same five OpenCV constants.
ALGORITHMS = {
    'TSAI': 'Tsai-Lenz',
    'PARK': 'Park',
    'HORAUD': 'Horaud',
    'ANDREFF': 'Andreff',
    'DANIILIDIS': 'Daniilidis',
}

BACKEND_RELPATH = os.path.join(
    'easy_handeye2', 'easy_handeye2', 'handeye_calibration_backend_opencv.py')

# The real module pulls in easy_handeye2_msgs, which needs a built ROS message
# package. The backend only uses HandeyeCalibration as a container for its
# return value, so this stands in for it and lets the backend file run as-is.
SHIM = '''
class HandeyeCalibration:
    def __init__(self, parameters=None, transform=None):
        self.parameters = parameters
        self.transform = transform
'''


class _Logger:
    """The backend logs through node.get_logger(); we are not a ROS node."""

    def info(self, msg):
        pass

    def warn(self, msg):
        pass

    def warning(self, msg):
        pass

    def err(self, msg):
        print(f'easy_handeye2: {msg}', file=sys.stderr)


class _Node:
    def get_logger(self):
        return _Logger()


class _Sample:
    __slots__ = ('robot', 'tracking')


class _SampleList:
    def __init__(self, samples):
        self.samples = samples


def load_backend(source_root, workdir):
    """Import their backend from a clone, unmodified. Returns (class, md5)."""
    backend_src = os.path.join(source_root, BACKEND_RELPATH)
    if not os.path.isfile(backend_src):
        raise SystemExit(
            f'not found: {backend_src}\n'
            'Pass --easy-handeye2 pointing at a clone of\n'
            '  https://github.com/marcoesposito1988/easy_handeye2')

    with open(backend_src, 'rb') as handle:
        digest = hashlib.md5(handle.read()).hexdigest()

    pkg = os.path.join(workdir, 'easy_handeye2')
    os.makedirs(pkg, exist_ok=True)
    open(os.path.join(pkg, '__init__.py'), 'w').close()
    with open(os.path.join(pkg, 'handeye_calibration.py'), 'w') as handle:
        handle.write(SHIM)
    shutil.copyfile(
        backend_src, os.path.join(pkg, 'handeye_calibration_backend_opencv.py'))

    sys.path.insert(0, workdir)
    from easy_handeye2.handeye_calibration_backend_opencv import (
        HandeyeCalibrationBackendOpenCV)
    return HandeyeCalibrationBackendOpenCV, digest


def to_msg(T):
    """4x4 -> geometry_msgs/Transform, the way tf2 would hand it over."""
    from geometry_msgs.msg import Quaternion, Transform, Vector3
    qx, qy, qz, qw = matrix_to_quaternion(T[:3, :3])
    tx, ty, tz = T[:3, 3]
    return Transform(
        translation=Vector3(x=float(tx), y=float(ty), z=float(tz)),
        rotation=Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw)))


def from_msg(msg):
    q, t = msg.rotation, msg.translation
    return make_transform(quaternion_to_matrix(q.x, q.y, q.z, q.w),
                          [t.x, t.y, t.z])


def mm(T):
    t = T[:3, 3] * 1000.0
    return f'[{t[0]:8.2f} {t[1]:8.2f} {t[2]:8.2f}]'


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='/tmp/handeye_dataset.json',
                        help='dataset written by the pose collector')
    parser.add_argument('--easy-handeye2', required=True,
                        help='path to an easy_handeye2 clone')
    args = parser.parse_args()

    payload, robot_poses, target_poses = load_dataset(args.dataset)

    ground_truth = payload.get('ground_truth_T_base_cam')
    T_gt = transform_from_dict(ground_truth) if ground_truth else None

    with tempfile.TemporaryDirectory() as workdir:
        Backend, digest = load_backend(args.easy_handeye2, workdir)

        print(f'dataset          {args.dataset}   ({len(robot_poses)} samples)')
        print(f'their backend    md5 {digest} (unmodified)')
        if T_gt is not None:
            print(f'ground truth     {mm(T_gt)} mm')
        print()

        samples = []
        for T_base_ee, T_cam_target in zip(robot_poses, target_poses):
            sample = _Sample()
            sample.robot = to_msg(invert_transform(T_base_ee))
            sample.tracking = to_msg(T_cam_target)
            samples.append(sample)

        backend, node = Backend(), _Node()

        header = (f'{"method":<11}{"agreement":>12}{"":>10}'
                  f'{"ours vs truth":>16}{"":>9}{"residual":>11}{"":>9}')
        print(header)
        print(f'{"":<11}{"max|d|":>12}{"mm":>10}{"mm":>16}{"deg":>9}'
              f'{"mm":>11}{"deg":>9}')
        print('-' * len(header))

        worst = 0.0
        for ours_name, theirs_name in ALGORITHMS.items():
            T_ours = solve(robot_poses, target_poses, ours_name)
            result = backend.compute_calibration(
                node, None, _SampleList(samples), algorithm=theirs_name)
            if result is None:
                print(f'{ours_name:<11} easy_handeye2 declined to solve')
                continue
            T_theirs = from_msg(result.transform)

            elementwise = float(np.abs(T_ours - T_theirs).max())
            worst = max(worst, elementwise)
            d_m, _ = transform_difference(T_ours, T_theirs)
            res = residuals(T_ours, robot_poses, target_poses)

            row = f'{ours_name:<11}{elementwise:12.2e}{d_m * 1000:10.6f}'
            if T_gt is not None:
                g_m, g_rad = transform_difference(T_ours, T_gt)
                row += f'{g_m * 1000:16.4f}{np.degrees(g_rad):9.4f}'
            else:
                row += f'{"-":>16}{"-":>9}'
            row += (f'{res["translation_rms_mm"]:11.4f}'
                    f'{res["rotation_rms_deg"]:9.4f}')
            print(row)

        print()
        verdict = 'AGREE to machine precision' if worst < 1e-9 else 'DISAGREE'
        print(f'verdict: {verdict}  (worst elementwise {worst:.2e})')
        if worst >= 1e-9:
            print('The two stacks read the geometry differently. Investigate '
                  'the transform directions before trusting either result.')
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
