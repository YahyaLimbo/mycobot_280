#!/usr/bin/env python3
"""Where to put the calibration target, and whether the set that came back is good.

Split out of the collector node so it can be tested without ROS, Gazebo or a
camera. Everything here is plain numpy: given a nominal camera and a coarse
model of the arm's reach it returns candidate marker poses, and given the
samples that were actually captured it scores them.

The scoring exists because a hand-eye dataset can be bad in ways that leave no
trace in the solver's residual. A set confined to one patch of the image, or
one sharing a single rotation axis, fits its own observations perfectly and
generalises to nothing. So the metrics here follow the standard rules of thumb
for a usable dataset, one metric per rule:

    20-40 views                    samples
    varied orientations            rotation_*  (spread and axis conditioning)
    target over the whole image    cells_covered, u/v_span_fraction
    target FILLING the image       fill_fraction_max
    edges as well as the centre    edge_cells_covered, centre_cells_covered
    varied distances               distance_ratio

`coverage_warnings` turns those into one message per rule that is not met.

Note the two distinct senses of "the board covers the image", which are
separate rules and separately measured. `cells_covered` asks whether the
target's CENTRE visited the whole frame across the sweep; `fill_fraction_max`
asks whether the board was ever BIG in frame, which needs the arm close to the
camera and is what limits how precisely each individual view can be localised.
A sweep can satisfy either one while failing the other, and only the first was
measured until 2026-08-30 -- a rig whose board never exceeded 18% of the frame
height scored full marks.

They also trade against each other and cannot both be maximal in one frame: the
border inset scales with the plate, so a board large enough to fill the view
cannot also sit at the edge without being clipped, and a clipped single ArUco
yields no detection at all rather than a degraded one. The set satisfies both
across a range of distances, which is why `distance_ratio` is a rule too.
"""

import itertools
import math
import random

import cv2
import numpy as np

from mycobot_calibration.transforms import (
    euler_to_matrix,
    look_at_rotation,
    make_transform,
    ray_sphere_intersection,
)

# The image is scored on a 4x3 grid of cells. The two cells in the middle row's
# middle columns are the "centre"; the ten around them are the "edge". A set
# that only fills the middle is the classic bad calibration dataset, and a set
# that only rides the border is no better, so both are counted.
CELLS_X, CELLS_Y = 4, 3
CENTRE_CELLS = frozenset({(1, 1), (2, 1)})
EDGE_CELLS = frozenset(
    cell for cell in itertools.product(range(CELLS_X), range(CELLS_Y))
    if cell not in CENTRE_CELLS)

# Thresholds the warnings fire on. Deliberately loose: they are meant to catch
# a dataset that is qualitatively wrong, not to grade a good one.
MIN_SAMPLES = 20
MAX_SAMPLES = 40
MIN_CELLS = 8
MIN_EDGE_CELLS = 6
MIN_CENTRE_CELLS = 1
MIN_SPAN_FRACTION = 0.50
MIN_DISTANCE_RATIO = 1.3
# At least one view in which the plate spans this fraction of the image HEIGHT
# (the smaller dimension, so the binding one for filling the frame). A close-up
# is where the corners are localised best, and it is the half of "board covers
# the image" that position coverage alone does not check. 0.30 is deliberately
# well under what the shipped rig delivers (0.90) -- it is meant to catch a
# camera mounted too far away to ever get a close-up, not to grade a good set.
# Since the close-ups go as near as the arm reaches, failing this says the
# stand is too far back, which is not something a pose plan can repair.
MIN_FILL_FRACTION = 0.30
MIN_ROTATION_MEDIAN_DEG = 15.0
MIN_AXIS_CONDITIONING = 0.20


def nominal_intrinsics(width, height, hfov):
    """(fx, cx, cy) for a pinhole camera of this size and horizontal FOV."""
    fx = (width / 2.0) / math.tan(hfov / 2.0)
    return fx, width / 2.0, height / 2.0


def cell_of(u, v, width, height):
    """Which cell of the CELLS_X x CELLS_Y grid a pixel falls in."""
    return (min(CELLS_X - 1, max(0, int(u / width * CELLS_X))),
            min(CELLS_Y - 1, max(0, int(v / height * CELLS_Y))))


DISTANCE_BANDS = 3


def _interleave_by_stratum(poses, strata, shuffle_seed):
    """Order candidates so that any prefix is spread over image and range.

    max_poses cuts this list, and the candidate pool is several times larger
    than the cut, so which candidates survive the cut is what decides the
    coverage a sweep actually delivers.

    A flat shuffle -- what this did while the pool was smaller than max_poses
    and nothing was ever discarded -- spreads them only in expectation. Taking
    40 of 262 that way routinely missed edge cells, because those are exactly
    the cells the fewest candidates land in, so uniform sampling is least
    likely to draw them. Round-robin over (image cell, distance band) instead
    guarantees the rarest strata appear in any prefix long before the common
    ones are exhausted.

    Deterministic for a given seed: the buckets and the order they are visited
    in are both shuffled with it.
    """
    if not poses:
        return []

    distances = [d for _, d in strata]
    low, high = min(distances), max(distances)
    span = (high - low) or 1.0

    groups = {}
    for pose, (cell, distance) in zip(poses, strata):
        band = min(DISTANCE_BANDS - 1,
                   int((distance - low) / span * DISTANCE_BANDS))
        groups.setdefault((cell, band), []).append(pose)

    rng = random.Random(int(shuffle_seed))
    for bucket in groups.values():
        rng.shuffle(bucket)
    order = sorted(groups)
    rng.shuffle(order)

    interleaved = []
    while len(interleaved) < len(poses):
        for key in order:
            if groups[key]:
                interleaved.append(groups[key].pop())
    return interleaved


def plan_marker_poses(camera_position, camera_rpy_deg, image_width, image_height,
                      camera_hfov, image_grid, marker_size, edge_safety,
                      workspace_centre, workspace_radius_max,
                      workspace_radius_min, samples_per_ray, min_marker_height,
                      tilt_angles_deg, roll_angles_deg, shuffle_seed,
                      close_up_poses=6, close_up_margin_px=20.0,
                      max_marker_distance=0.0):
    """Candidate marker poses, planned to tile the camera IMAGE.

    Each candidate starts as a position in the image and a distance rather than
    a point in the robot's workspace: the pixel is back-projected through the
    nominal camera and the marker centre is placed along that ray. Tiling the
    image grid therefore tiles the picture directly, edges and corners
    included, which is what a calibration dataset needs and what sampling a box
    in the workspace fails to give.

    The border inset is per distance, not global. It only has to keep the plate
    fully inside the frame, and a plate of fixed physical size subtends fewer
    pixels the further away it is, so a distant sample can sit closer to the
    border than a near one. That takes two passes: the first places the pixel
    using the inset at the farthest reachable distance in order to find where
    the ray enters and leaves the workspace, and the second re-places it using
    the inset at each distance actually sampled along that chord.

    `max_marker_distance` (0 = no limit) truncates every chord at that range.
    It exists for targets too small to stay unambiguous across the whole
    workspace. A planar marker's pose solve has two solutions that separate
    only through perspective foreshortening, so once the marker subtends too
    few degrees SOLVEPNP_IPPE_SQUARE flips between them; the collector's
    stability gate then throws the pose away. Measured on the 38 mm target
    against this rig: rock solid out to ~0.29 m (0.17-0.45 mm of position
    scatter), and 7-10 mm of scatter by 0.35 m -- with 174 deg of rotation
    scatter at 0.32 m with the arm completely STATIONARY. Left uncapped, the
    2026-09-02 sweep planned 0.154-0.41 m and threw away 43 of 48 poses at
    ~9 s each.

    Capping is strictly a throughput fix, and it costs conditioning: hand-eye
    recovers the translation partly from how the target's apparent size changes
    with range, so a sweep confined to a narrow distance band determines it
    less well. `distance_ratio` in the coverage report is what to watch. A
    bigger printed target is the better answer whenever one is available.

    Orientation faces the nominal camera, then takes a tilt about both of the
    plate's in-plane axes and a roll about the viewing axis. Two tilt axes
    rather than one: hand-eye recovers rotation from the relative motions
    between poses, so a set whose relative rotations share an axis leaves the
    answer underdetermined however many samples it holds, and a planar target
    seen nearly fronto-parallel is also where a planar pose solve is most
    ambiguous.

    Returns (poses, stats) where poses is a list of 4x4 T_base_marker and stats
    describes how much of the image grid the arm could actually reach.
    """
    camera = np.asarray(camera_position, dtype=float)
    workspace = np.asarray(workspace_centre, dtype=float)
    T_base_cam = make_transform(
        euler_to_matrix(*[math.radians(a) for a in camera_rpy_deg]), camera)
    fx, cx, cy = nominal_intrinsics(image_width, image_height, camera_hfov)

    # Distance to the middle of the reachable volume, and the farthest the
    # marker could possibly sit from the camera while still inside it.
    nominal_range = float(np.linalg.norm(workspace - camera))
    far_range = nominal_range + workspace_radius_max

    def inset_at(distance):
        """Half the plate's on-screen size at this distance, plus margin."""
        return (marker_size / 2.0) / distance * fx * edge_safety

    def pixel_at(fraction_u, fraction_v, distance):
        """Place a grid fraction inside the border inset for this distance."""
        inset = inset_at(distance)
        if 2.0 * inset >= min(image_width, image_height):
            return None                      # plate cannot fit in the frame
        return (inset + fraction_u * (image_width - 2.0 * inset),
                inset + fraction_v * (image_height - 2.0 * inset))

    def ray_through(u, v):
        """Unit ray in base coordinates through an image position."""
        ray_camera = np.array([(u - cx) / fx, (v - cy) / fx, 1.0])
        ray_camera /= np.linalg.norm(ray_camera)
        return T_base_cam[:3, :3] @ ray_camera

    # Cycle the perturbations across grid points rather than taking their
    # product with them: the product would be thousands of poses, while
    # cycling still gives every rotation offset a spread of positions, which is
    # what conditions the solve.
    tilts = [math.radians(a) for a in tilt_angles_deg]
    rolls = [math.radians(a) for a in roll_angles_deg]
    perturbations = itertools.cycle(
        [(tilt_x, tilt_y, roll)
         for tilt_x in tilts for tilt_y in tilts for roll in rolls])

    grid_x, grid_y = int(image_grid[0]), int(image_grid[1])
    fractions = list(itertools.product(np.linspace(0.0, 1.0, grid_x),
                                       np.linspace(0.0, 1.0, grid_y)))

    poses = []
    strata = []
    unreachable_positions = 0
    dropped_candidates = 0

    for fraction_u, fraction_v in fractions:
        # Pass 1: a provisional pixel at the smallest the plate can appear,
        # only to find where this ray crosses the reachable volume.
        provisional = pixel_at(fraction_u, fraction_v, far_range)
        if provisional is None:
            unreachable_positions += 1
            continue
        span = ray_sphere_intersection(camera, ray_through(*provisional),
                                       workspace, workspace_radius_max)
        if span is None or span[1] <= 0.0:
            unreachable_positions += 1
            continue

        near, far = max(span[0], 1e-3), span[1]
        if max_marker_distance > 0.0:
            far = min(far, max_marker_distance)
            if far <= near:
                unreachable_positions += 1
                continue
        accepted = 0
        # Endpoints dropped: the ray is tangent to the reachable volume there,
        # so those poses sit exactly on the boundary of what the arm can do.
        for distance in np.linspace(near, far, samples_per_ray + 2)[1:-1]:
            # Pass 2: re-place the pixel using the inset at THIS distance, then
            # re-cross the volume, since moving the pixel moved the ray.
            uv = pixel_at(fraction_u, fraction_v, distance)
            if uv is None:
                dropped_candidates += 1
                continue
            ray = ray_through(*uv)
            refined = ray_sphere_intersection(camera, ray, workspace,
                                              workspace_radius_max)
            if refined is None or refined[1] <= 0.0:
                dropped_candidates += 1
                continue

            ceiling = refined[1]
            if max_marker_distance > 0.0:
                ceiling = min(ceiling, max_marker_distance)
                if ceiling <= max(refined[0], 1e-3):
                    dropped_candidates += 1
                    continue
            distance = float(np.clip(distance, max(refined[0], 1e-3), ceiling))
            point_base = camera + ray * distance
            if np.linalg.norm(point_base - workspace) < workspace_radius_min:
                dropped_candidates += 1
                continue
            if point_base[2] < min_marker_height:
                dropped_candidates += 1
                continue

            tilt_x, tilt_y, roll = next(perturbations)
            # Tilt about the plate's own X and Y, roll about its viewing axis Z.
            rotation = (look_at_rotation(point_base, camera)
                        @ euler_to_matrix(tilt_x, tilt_y, roll))
            poses.append(make_transform(rotation, point_base))
            strata.append((cell_of(uv[0], uv[1], image_width, image_height),
                           distance))
            accepted += 1

        if accepted == 0:
            unreachable_positions += 1

    # Close-ups: the views the border inset necessarily throws away.
    #
    # The inset scales with the plate's on-screen size, so the nearer the
    # target the less freedom it has to move about the frame, and past a point
    # every candidate at that distance is rejected. Those rejected poses are
    # exactly the ones where the board covers most of the image, which is both
    # a rule a calibration dataset is supposed to satisfy and where the
    # corners are localised best -- the dominant error term here scales as
    # (sub-pixel corner bias x distance).
    #
    # With edge_safety at 1.6 and a 139.5 mm plate the grid cannot plan closer
    # than 0.208 m, capping the board at 62% of the frame height, while the arm
    # can reach 0.140 m. So a few poses are planned deliberately, AS CLOSE AS
    # THE ARM CAN REACH, near the centre where nothing has to fit beside them,
    # and with ROLL SUPPRESSED: roll is what makes a square's bounding box
    # larger than the square, and at this size a rolled plate would overhang
    # the frame. Tilt is kept, because it is what keeps the plate off
    # fronto-parallel where a planar pose solve is most ambiguous.
    #
    # The binding constraint is not the arm, it is clipping. A single ArUco
    # touching the frame border yields NO detection at all rather than a
    # degraded one, so the plate has to keep close_up_margin_px of border. That
    # margin is the whole design freedom here, and it buys tolerance to the
    # arm's execution error: at 20 px and this working distance it absorbs
    # 6.3 mm of lateral miss against the 2-5 mm the arm actually delivers,
    # while still filling ~92% of the frame. Expressing it as a margin rather
    # than as a target fill keeps the trade in the units the risk lives in --
    # an earlier version asked for 80% fill, which silently reserved 17 mm of
    # error budget for a 5 mm error.
    close_up_list = []
    close_up_ranges = []
    if close_up_poses > 0:
        # The closest the plate can sit without its edge entering the margin.
        usable = image_height - 2.0 * max(0.0, close_up_margin_px)
        if usable <= 0.0:
            usable = float(image_height)
        distance = marker_size * fx / usable
        half = (marker_size / 2.0) / distance * fx
        # Whatever room is left once the plate is placed; at high fill this is
        # small vertically and generous horizontally, since the frame is wide.
        free_u = max(0.0, image_width / 2.0 - half)
        free_v = max(0.0, image_height / 2.0 - half)
        tilt_cycle = itertools.cycle(
            [(tx, ty) for tx in tilts for ty in tilts if (tx, ty) != (0.0, 0.0)])

        for index in range(int(close_up_poses)):
            angle = 2.0 * math.pi * index / float(close_up_poses)

            # Spread them around the centre rather than stacking them on it,
            # using at most 80% of the free room so a small pose error cannot
            # clip the plate. Off-axis costs reach, though: a ray leaving the
            # optical axis enters the reachable sphere later, so the widest
            # ring is not always attainable and shrinks by camera placement in
            # a way that is easy to get wrong by hand. Rather than drop the
            # pose, walk the ring inward and take the first radius that the
            # arm can actually reach -- worst case dead centre.
            placed = None
            for scale in (0.8, 0.55, 0.3, 0.0):
                uv = (cx + scale * free_u * math.cos(angle),
                      cy + scale * free_v * math.sin(angle))
                ray = ray_through(*uv)
                span = ray_sphere_intersection(camera, ray, workspace,
                                               workspace_radius_max)
                if span is None or span[1] <= 0.0:
                    continue
                # As close as the arm can reach, but never nearer than the
                # clipping limit: back off to whichever binds on this ray.
                # Off-axis rays enter the reachable volume later, so which one
                # binds varies around the ring.
                reached = max(distance, span[0])
                if reached > span[1]:
                    continue                 # ray misses the reachable volume
                if 0.0 < max_marker_distance < reached:
                    # A 'close-up' past the ambiguity limit is not a close-up
                    # worth having: the plate is small enough there that the
                    # pose solve flips, and the collector would discard it.
                    continue
                point_base = camera + ray * reached
                if np.linalg.norm(point_base - workspace) < workspace_radius_min:
                    continue
                if point_base[2] < min_marker_height:
                    continue
                placed = (uv, point_base, reached)
                break

            if placed is None:
                continue                     # no close-up reachable here
            uv, point_base, reached = placed

            tilt_x, tilt_y = next(tilt_cycle)
            rotation = (look_at_rotation(point_base, camera)
                        @ euler_to_matrix(tilt_x, tilt_y, 0.0))
            close_up_list.append(make_transform(rotation, point_base))
            close_up_ranges.append(reached)

    # Close-ups go at the FRONT, ahead of the interleave, rather than being
    # mixed into it. The pool is several times max_poses, so anything left to
    # the stratified draw is only there in proportion to its stratum -- and
    # there are half a dozen close-ups against 262 grid poses. They are the
    # one rule the rest of the plan structurally cannot satisfy, so they are
    # given a guaranteed place instead of a share of the odds. It also means
    # the sweep probes its most demanding poses first, where a failure is
    # cheap to notice.
    poses = close_up_list + _interleave_by_stratum(poses, strata, shuffle_seed)

    stats = {
        'grid_positions': len(fractions),
        'reachable_positions': len(fractions) - unreachable_positions,
        'candidates': len(poses),
        'dropped_candidates': dropped_candidates,
        'nominal_range': nominal_range,
        # 0 means the arm cannot get close enough to fill the frame at all,
        # which is a property of where the camera is bolted, not of the plan.
        'close_ups': len(close_up_list),
        'close_up_range': min(close_up_ranges) if close_up_ranges else None,
    }
    return poses, stats


def coverage_metrics(target_transforms, robot_transforms, image_width,
                     image_height, camera_hfov, marker_size):
    """Score a captured dataset against the rules of thumb.

    `target_transforms` are T_cam_target, which is what decides where the
    target landed in the image and how far away it was. `robot_transforms` are
    T_base_ee, used for the rotation spread: they come from forward kinematics
    rather than from the detector, so the conditioning number is not polluted
    by detection noise.

    `marker_size` is the whole plate, not the black square, because what is
    being scored is how much of the frame the physical board occupied. It is
    the same number the planner insets the frame border by.

    Returns {} for an empty set, so a failed sweep reports nothing rather than
    dividing by zero.
    """
    fx, cx, cy = nominal_intrinsics(image_width, image_height, camera_hfov)

    pixels, distances, fills = [], [], []
    for T in target_transforms:
        t = np.asarray(T)[:3, 3]
        if t[2] <= 1e-6:
            continue                          # behind the camera; not visible
        pixels.append((fx * t[0] / t[2] + cx, fx * t[1] / t[2] + cy))
        distance = float(np.linalg.norm(t))
        distances.append(distance)
        # Plate span as a fraction of the image height. Measured against the
        # ray distance rather than the depth, matching the planner's border
        # inset, which errs slightly small and so never flatters the set.
        fills.append(fx * marker_size / distance / image_height)

    if not pixels:
        return {}

    pixels = np.array(pixels)
    cells = {cell_of(u, v, image_width, image_height) for u, v in pixels}

    report = {
        'samples': len(target_transforms),
        'fill_fraction_max': float(max(fills)),
        'fill_fraction_median': float(np.median(fills)),
        'u_span_fraction': float((pixels[:, 0].max() - pixels[:, 0].min())
                                 / image_width),
        'v_span_fraction': float((pixels[:, 1].max() - pixels[:, 1].min())
                                 / image_height),
        'cells_covered': len(cells),
        'cells_total': CELLS_X * CELLS_Y,
        'edge_cells_covered': len(cells & EDGE_CELLS),
        'edge_cells_total': len(EDGE_CELLS),
        'centre_cells_covered': len(cells & CENTRE_CELLS),
        'centre_cells_total': len(CENTRE_CELLS),
        'distance_min': min(distances),
        'distance_max': max(distances),
        'distance_ratio': max(distances) / min(distances),
    }
    report.update(rotation_metrics(robot_transforms))
    return report


def rotation_metrics(transforms):
    """Spread of the relative rotations, and whether they share an axis.

    Hand-eye recovers rotation from relative motions, so what matters is not
    how far the poses are from each other in absolute terms but how much, and
    about how many distinct axes, they rotate between. A set of relative
    rotations that all share one axis spans a rank-1 set of axes and leaves the
    solution underdetermined however many samples it holds -- which the
    smallest singular value of the stacked axes detects, while a plain angle
    spread does not.

    `axis_conditioning` is that smallest singular value over the largest: near
    0 for a degenerate set, and around 0.6-0.7 for the sweep this package
    plans.
    """
    rotations = [np.asarray(T)[:3, :3] for T in transforms]
    angles, axes = [], []
    for i in range(len(rotations)):
        for j in range(i + 1, len(rotations)):
            rotation_vector = cv2.Rodrigues(rotations[i].T @ rotations[j])[0].ravel()
            angle = float(np.linalg.norm(rotation_vector))
            if angle > 1e-6:
                angles.append(math.degrees(angle))
                axes.append(rotation_vector / angle)

    if not angles:
        return {'rotation_min_deg': 0.0, 'rotation_median_deg': 0.0,
                'rotation_max_deg': 0.0, 'axis_conditioning': 0.0}

    singular = np.linalg.svd(np.array(axes).T, compute_uv=False)
    return {
        'rotation_min_deg': float(min(angles)),
        'rotation_median_deg': float(np.median(angles)),
        'rotation_max_deg': float(max(angles)),
        'axis_conditioning': float(singular[2] / singular[0])
        if singular[0] > 1e-12 else 0.0,
    }


def coverage_warnings(report):
    """One message per rule of thumb the dataset fails. Empty when it passes."""
    if not report:
        return ['no usable samples: the target was never seen in front of the '
                'camera']

    warnings = []
    if report['samples'] < MIN_SAMPLES:
        warnings.append(
            f'only {report["samples"]} views; the rule of thumb is '
            f'{MIN_SAMPLES} to {MAX_SAMPLES}')

    if report['cells_covered'] < MIN_CELLS:
        warnings.append(
            f'the target reached {report["cells_covered"]} of '
            f'{report["cells_total"]} image cells; a calibration fitted from '
            'one region is only trustworthy in that region')
    if report['edge_cells_covered'] < MIN_EDGE_CELLS:
        warnings.append(
            f'only {report["edge_cells_covered"]} of '
            f'{report["edge_cells_total"]} edge cells were reached; corner '
            'localisation and lens distortion are worst near the border, which '
            'is exactly where this set says nothing')
    if report['centre_cells_covered'] < MIN_CENTRE_CELLS:
        warnings.append(
            'the target never crossed the middle of the frame')

    if (report['u_span_fraction'] < MIN_SPAN_FRACTION or
            report['v_span_fraction'] < MIN_SPAN_FRACTION):
        warnings.append(
            'the target spanned {u:.0f}% of the width and {v:.0f}% of the '
            'height; below {m:.0f}% the arm probably cannot reach far enough '
            'across the view, which no pose planner can fix'.format(
                u=report['u_span_fraction'] * 100,
                v=report['v_span_fraction'] * 100,
                m=MIN_SPAN_FRACTION * 100))

    if report['fill_fraction_max'] < MIN_FILL_FRACTION:
        warnings.append(
            'the board never spanned more than {f:.0f}% of the image height '
            '(want at least one view above {m:.0f}%); every view was a distant '
            'one, so the corners are localised no better than the worst of '
            'them. Move the camera closer to the arm or enlarge the board -- '
            'no pose planner can fix this'.format(
                f=report['fill_fraction_max'] * 100,
                m=MIN_FILL_FRACTION * 100))

    if report['distance_ratio'] < MIN_DISTANCE_RATIO:
        warnings.append(
            'all views were at nearly the same range ({d0:.2f}-{d1:.2f} m); '
            'without a spread of distances the scale of the solution is weakly '
            'observed'.format(d0=report['distance_min'], d1=report['distance_max']))

    if report['rotation_median_deg'] < MIN_ROTATION_MEDIAN_DEG:
        warnings.append(
            f'relative rotations are small (median '
            f'{report["rotation_median_deg"]:.0f} deg); hand-eye recovers '
            'rotation from relative motion, so a nearly-still wrist leaves it '
            'underdetermined')
    if report['axis_conditioning'] < MIN_AXIS_CONDITIONING:
        warnings.append(
            f'relative rotations nearly share an axis (conditioning '
            f'{report["axis_conditioning"]:.2f}); add tilt and roll variety, '
            'not more samples')

    return warnings


def format_coverage(report):
    """One-line summary of a coverage report, for the log."""
    if not report:
        return 'no usable samples'
    return (
        '{n} views | {c}/{ct} cells ({e}/{et} edge, {m}/{mt} centre) | '
        'centres over {u:.0f}% x {v:.0f}% of frame | board fills up to '
        '{f:.0f}% of height (median {fm:.0f}%) | {d0:.2f}-{d1:.2f} m ({r:.2f}x) '
        '| rel. rotation median {rot:.0f} deg, axis conditioning {ac:.2f}'.format(
            n=report['samples'], c=report['cells_covered'],
            ct=report['cells_total'], e=report['edge_cells_covered'],
            et=report['edge_cells_total'], m=report['centre_cells_covered'],
            mt=report['centre_cells_total'],
            u=report['u_span_fraction'] * 100, v=report['v_span_fraction'] * 100,
            f=report['fill_fraction_max'] * 100,
            fm=report['fill_fraction_median'] * 100,
            d0=report['distance_min'], d1=report['distance_max'],
            r=report['distance_ratio'], rot=report['rotation_median_deg'],
            ac=report['axis_conditioning']))
