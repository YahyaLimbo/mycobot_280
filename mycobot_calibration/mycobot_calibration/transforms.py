#!/usr/bin/env python3
"""Rigid-transform helpers shared by the hand-eye calibration nodes.

Naming convention used throughout this package, without exception:

    T_a_b  is the pose of frame b expressed in frame a, equivalently the
           transform that maps a point given in b into a:  p_a = T_a_b * p_b

Hand-eye code drowns in transform-direction bugs, so every function here
states which direction it takes and returns, and the OpenCV boundary (which
uses its own "x2y" naming) is crossed in exactly one place, in solver.py.

Rotations are handled with numpy and cv2.Rodrigues rather than scipy: the
container ships scipy 1.8, which declares support only up to numpy 1.24 and
warns on import against the numpy 1.26 the rest of the stack needs.
"""

import math

import cv2
import numpy as np


def quaternion_to_matrix(x, y, z, w):
    """Rotation matrix from a quaternion in ROS (x, y, z, w) order."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        raise ValueError('cannot normalise a zero quaternion')
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quaternion(R):
    """Quaternion in ROS (x, y, z, w) order from a rotation matrix.

    Uses the branch with the largest denominator, which keeps the result
    well-conditioned for rotations near 180 degrees where the naive trace
    formula loses precision.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return np.array([(R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s,
                         (R[1, 0] - R[0, 1]) * s,
                         0.25 / s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([0.25 * s,
                         (R[0, 1] + R[1, 0]) / s,
                         (R[0, 2] + R[2, 0]) / s,
                         (R[2, 1] - R[1, 2]) / s])
    if R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 1] + R[1, 0]) / s,
                         0.25 * s,
                         (R[1, 2] + R[2, 1]) / s,
                         (R[0, 2] - R[2, 0]) / s])
    s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array([(R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s,
                     0.25 * s,
                     (R[1, 0] - R[0, 1]) / s])


def euler_to_matrix(roll, pitch, yaw):
    """Rotation matrix from fixed-axis RPY, matching tf2 and urdf convention.

    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def matrix_to_euler(R):
    """Fixed-axis RPY (roll, pitch, yaw) from a rotation matrix."""
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    if abs(sy) > 0.99999:
        # Gimbal lock: roll and yaw are not separable, fold everything on roll.
        return math.atan2(-R[1, 2], R[1, 1]), pitch, 0.0
    return (math.atan2(R[2, 1], R[2, 2]),
            pitch,
            math.atan2(R[1, 0], R[0, 0]))


def make_transform(R, t):
    """Assemble a 4x4 homogeneous transform from a 3x3 R and a 3-vector t."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def invert_transform(T):
    """Inverse of a homogeneous transform, exploiting R^-1 = R^T."""
    R = T[:3, :3]
    t = T[:3, 3]
    return make_transform(R.T, -R.T @ t)


def transform_from_msg(transform_msg):
    """4x4 matrix from a geometry_msgs/Transform.

    A geometry_msgs/TransformStamped with header.frame_id 'a' and child_frame_id
    'b' carries the pose of b in a, so this returns T_a_b.
    """
    t = transform_msg.translation
    q = transform_msg.rotation
    return make_transform(quaternion_to_matrix(q.x, q.y, q.z, q.w),
                          [t.x, t.y, t.z])


def pose_from_msg(pose_msg):
    """4x4 matrix from a geometry_msgs/Pose."""
    p = pose_msg.position
    q = pose_msg.orientation
    return make_transform(quaternion_to_matrix(q.x, q.y, q.z, q.w),
                          [p.x, p.y, p.z])


def pose_to_msg(T, pose_msg):
    """Fill a geometry_msgs/Pose in place from a 4x4 matrix."""
    q = matrix_to_quaternion(T[:3, :3])
    pose_msg.position.x = float(T[0, 3])
    pose_msg.position.y = float(T[1, 3])
    pose_msg.position.z = float(T[2, 3])
    pose_msg.orientation.x = float(q[0])
    pose_msg.orientation.y = float(q[1])
    pose_msg.orientation.z = float(q[2])
    pose_msg.orientation.w = float(q[3])
    return pose_msg


def transform_to_list(T):
    """Serialise a transform as {translation, quaternion, rpy_deg} for JSON."""
    q = matrix_to_quaternion(T[:3, :3])
    rpy = matrix_to_euler(T[:3, :3])
    return {
        'translation': [float(v) for v in T[:3, 3]],
        'quaternion_xyzw': [float(v) for v in q],
        'rpy_deg': [float(math.degrees(v)) for v in rpy],
    }


def transform_from_dict(data):
    """Inverse of transform_to_list, reading translation + quaternion_xyzw."""
    q = data['quaternion_xyzw']
    return make_transform(quaternion_to_matrix(*q), data['translation'])


def rotation_angle(R):
    """Magnitude of the rotation in radians, from the axis-angle form."""
    return float(np.linalg.norm(cv2.Rodrigues(R)[0]))


def rotation_difference(R_a, R_b):
    """Angle in radians between two rotations."""
    return rotation_angle(R_a.T @ R_b)


def transform_difference(T_a, T_b):
    """(translation error in metres, rotation error in radians) between poses."""
    return (float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])),
            rotation_difference(T_a[:3, :3], T_b[:3, :3]))


def average_transforms(transforms):
    """Average a list of 4x4 transforms.

    Translation is a plain mean. Rotation is the quaternion barycentre, taken
    as the dominant eigenvector of the accumulated outer products, which is
    the standard closed-form Markley average and avoids the drift of
    renormalised element-wise means. Quaternion sign is aligned to the first
    sample first, since q and -q are the same rotation but cancel if mixed.
    """
    if not transforms:
        raise ValueError('no transforms to average')

    translations = np.array([T[:3, 3] for T in transforms])
    quats = np.array([matrix_to_quaternion(T[:3, :3]) for T in transforms])

    reference = quats[0]
    for i in range(len(quats)):
        if np.dot(quats[i], reference) < 0.0:
            quats[i] = -quats[i]

    M = np.zeros((4, 4))
    for q in quats:
        M += np.outer(q, q)
    _, eigenvectors = np.linalg.eigh(M)
    q_mean = eigenvectors[:, -1]
    if q_mean[3] < 0.0:
        q_mean = -q_mean

    return make_transform(quaternion_to_matrix(*q_mean),
                          translations.mean(axis=0))


def transform_spread(transforms):
    """Scatter of a set of transforms about their average.

    Returns (max translation deviation in metres, max rotation deviation in
    radians). Used to reject a pose whose detections were unstable.
    """
    mean = average_transforms(transforms)
    max_t, max_r = 0.0, 0.0
    for T in transforms:
        dt, dr = transform_difference(mean, T)
        max_t, max_r = max(max_t, dt), max(max_r, dr)
    return max_t, max_r


def ray_sphere_intersection(origin, direction, centre, radius):
    """Where a ray enters and leaves a sphere, as (t_near, t_far), or None.

    Used to decide how far along a camera ray the calibration target can sit
    and still be somewhere the arm can reach. `direction` need not be a unit
    vector; the returned parameters are in units of its length.
    """
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    centre = np.asarray(centre, dtype=float)

    offset = origin - centre
    a = float(np.dot(direction, direction))
    if a < 1e-12:
        return None
    b = 2.0 * float(np.dot(offset, direction))
    c = float(np.dot(offset, offset)) - radius * radius

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None

    root = math.sqrt(discriminant)
    return ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))


def look_at_rotation(eye, target, up=(0.0, 0.0, 1.0)):
    """Rotation whose +Z axis points from eye toward target.

    Built for aiming the marker plate: the plate's +Z is its printed face
    normal, so a marker placed at `eye` with this rotation stares straight at
    `target`. The +X axis is chosen perpendicular to `up`, which fixes the
    otherwise free roll about the viewing axis.
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)

    z_axis = target - eye
    norm = np.linalg.norm(z_axis)
    if norm < 1e-9:
        raise ValueError('eye and target coincide, view direction undefined')
    z_axis /= norm

    up = np.asarray(up, dtype=float)
    if abs(np.dot(up, z_axis)) > 0.999:
        # Degenerate: looking straight along `up`. Any perpendicular will do.
        up = np.array([1.0, 0.0, 0.0])

    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    return np.column_stack([x_axis, y_axis, z_axis])
