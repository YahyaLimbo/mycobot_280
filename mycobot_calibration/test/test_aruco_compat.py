#!/usr/bin/env python3
"""Tests for the OpenCV ArUco compatibility layer.

OpenCV 4.7 deleted the free-function ArUco API rather than deprecating it, so
this package has to work against two incompatible versions: the ROS 2 Humble
container ships 4.5, a host install is likely 4.8 or newer. `aruco_compat`
hides that, and these tests check it renders and recovers a pose correctly on
*whichever* version is installed -- so the same file is the guard on both.

The check is end to end on purpose. Asserting that a shim calls the right
function proves nothing about whether the answer is right; rendering a marker
of known size at a known range and demanding the pose back does.
"""

import cv2
import numpy as np
import pytest

from mycobot_calibration.aruco_compat import (
    NEW_API,
    CharucoInterpolator,
    MarkerDetector,
    board_chessboard_corners,
    create_charuco_board,
    create_detector_parameters,
    estimate_marker_pose,
    generate_board_image,
    generate_marker_image,
    get_dictionary,
)

WIDTH, HEIGHT = 848, 480
FX = 909.23                     # 848 px at the rig's 50 degree horizontal FOV
CAMERA_MATRIX = np.array([[FX, 0.0, WIDTH / 2.0],
                          [0.0, FX, HEIGHT / 2.0],
                          [0.0, 0.0, 1.0]])
DIST_COEFFS = np.zeros(5)
MARKER_SIZE = 0.055
MARKER_ID = 0


def detector_parameters():
    """The detector's own settings, since refinement changes the answer."""
    parameters = create_detector_parameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 50
    parameters.cornerRefinementMinAccuracy = 0.01
    return parameters


def scene_with_marker(distance, centre_u, centre_v):
    """A marker of the right on-screen size, drawn fronto-parallel."""
    dictionary = get_dictionary(cv2.aruco.DICT_6X6_250)
    span = int(round(MARKER_SIZE / distance * FX))
    marker = generate_marker_image(dictionary, MARKER_ID, span)
    frame = np.full((HEIGHT, WIDTH), 255, np.uint8)
    u0, v0 = centre_u - span // 2, centre_v - span // 2
    frame[v0:v0 + span, u0:u0 + span] = marker
    return frame, dictionary, span


# Off the principal point on purpose. A square that is exactly fronto-parallel
# AND exactly centred is the degeneracy IPPE_SQUARE returns NaN for, which has
# its own test below; the accuracy tests should not be sitting on it.
OFF_CENTRE_U, OFF_CENTRE_V = WIDTH // 2 + 40, HEIGHT // 2 + 25


class TestSingleMarker:

    @pytest.mark.parametrize('distance', [0.35, 0.45, 0.60])
    def test_pose_of_a_rendered_marker_comes_back(self, distance):
        """Render at a known range, demand the range back.

        The tolerance is 2% because a rendered marker's anti-aliased edges bias
        the refined corners inward by a fraction of a pixel, which reads as a
        proportional over-estimate of range. That bias is the subject of the
        README's accuracy section; here it only has to stay small enough to
        prove the shim is not off by a scale factor.
        """
        frame, dictionary, _ = scene_with_marker(distance, OFF_CENTRE_U,
                                                 OFF_CENTRE_V)
        corners, ids = MarkerDetector(dictionary, detector_parameters()).detect(frame)

        assert ids is not None, 'nothing detected'
        assert MARKER_ID in ids.ravel()

        rvec, tvec = estimate_marker_pose(corners[0], MARKER_SIZE,
                                          CAMERA_MATRIX, DIST_COEFFS)
        assert rvec is not None
        assert tvec[2] == pytest.approx(distance, rel=0.02)

    def test_marker_off_centre_lands_where_it_was_drawn(self):
        """Guards the object-point ordering, which a centred marker cannot."""
        distance, u, v = 0.45, 300, 180
        frame, dictionary, _ = scene_with_marker(distance, u, v)
        corners, ids = MarkerDetector(dictionary, detector_parameters()).detect(frame)
        assert ids is not None

        _, tvec = estimate_marker_pose(corners[0], MARKER_SIZE,
                                       CAMERA_MATRIX, DIST_COEFFS)
        assert tvec[0] == pytest.approx((u - WIDTH / 2) / FX * distance, abs=0.005)
        assert tvec[1] == pytest.approx((v - HEIGHT / 2) / FX * distance, abs=0.005)

    def test_marker_faces_the_camera(self):
        """+Z out of the printed face, so a fronto-parallel plate reads ~zero.

        The old estimatePoseSingleMarkers fixed this convention and the rest of
        the package assumes it; the new API's solvePnP would happily return the
        other planar branch if the object points were ordered differently.
        """
        frame, dictionary, _ = scene_with_marker(0.45, OFF_CENTRE_U, OFF_CENTRE_V)
        corners, ids = MarkerDetector(dictionary, detector_parameters()).detect(frame)
        rvec, _ = estimate_marker_pose(corners[0], MARKER_SIZE, CAMERA_MATRIX,
                                       DIST_COEFFS)
        rotation, _ = cv2.Rodrigues(rvec)
        assert abs(rotation[2, 2]) == pytest.approx(1.0, abs=0.02)

    def test_a_degenerate_square_is_rejected_rather_than_returning_nan(self):
        """solvePnP reports success and hands back NaN for this one.

        A square exactly fronto-parallel and exactly centred on the principal
        point degenerates: the homography becomes a similarity and IPPE's
        decomposition divides by zero. `ok` stays True, so nothing downstream
        would notice -- the NaN would be averaged into a pose, written to the
        dataset and carried into the solve. 0.45 m is a size that triggers it
        on this intrinsic; the guard does not depend on that.
        """
        frame, dictionary, _ = scene_with_marker(0.45, WIDTH // 2, HEIGHT // 2)
        corners, ids = MarkerDetector(dictionary, detector_parameters()).detect(frame)
        assert ids is not None, 'the marker should still be DETECTED'

        rvec, tvec = estimate_marker_pose(corners[0], MARKER_SIZE,
                                          CAMERA_MATRIX, DIST_COEFFS)
        assert (rvec is None and tvec is None) or (
            np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))), \
            'a non-finite pose must be rejected, never published'

    def test_an_empty_scene_detects_nothing(self):
        dictionary = get_dictionary(cv2.aruco.DICT_6X6_250)
        blank = np.full((HEIGHT, WIDTH), 255, np.uint8)
        _, ids = MarkerDetector(dictionary, detector_parameters()).detect(blank)
        assert ids is None or len(ids) == 0


class TestCharucoBoard:

    def board_and_scene(self, squares=5, square_length=0.022,
                        marker_length=0.0165, pixels=480):
        dictionary = get_dictionary(cv2.aruco.DICT_6X6_250)
        board = create_charuco_board(squares, squares, square_length,
                                     marker_length, dictionary)
        image = generate_board_image(board, pixels, pixels, 10, 1)
        frame = np.full((HEIGHT, WIDTH), 255, np.uint8)
        frame[0:pixels, 200:200 + pixels] = image
        return board, dictionary, frame

    def test_board_geometry_is_the_requested_one(self):
        board, _, _ = self.board_and_scene()
        corners = board_chessboard_corners(board)
        assert corners.shape == (16, 3)          # a 5x5 board has 16 interior
        assert np.allclose(corners[:, 2], 0.0)   # and it is planar

    def test_interior_corners_are_recovered(self):
        board, dictionary, frame = self.board_and_scene()
        parameters = detector_parameters()
        corners, ids = MarkerDetector(dictionary, parameters).detect(frame)
        assert ids is not None and len(ids) > 0

        count, charuco_corners, charuco_ids = CharucoInterpolator(
            board, dictionary, parameters).interpolate(frame, corners, ids)
        assert count == 16
        assert charuco_corners.reshape(-1, 2).shape == (16, 2)
        assert sorted(charuco_ids.ravel()) == list(range(16))

    def test_interpolating_an_empty_scene_yields_nothing(self):
        board, dictionary, _ = self.board_and_scene()
        parameters = detector_parameters()
        blank = np.full((HEIGHT, WIDTH), 255, np.uint8)
        corners, ids = MarkerDetector(dictionary, parameters).detect(blank)
        count, _, _ = CharucoInterpolator(board, dictionary,
                                          parameters).interpolate(blank, corners, ids)
        assert count == 0


def test_the_api_in_use_is_reported():
    """Which branch ran, so a CI log says which OpenCV was actually exercised."""
    assert NEW_API == hasattr(cv2.aruco, 'ArucoDetector')
