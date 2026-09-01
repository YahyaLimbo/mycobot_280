#!/usr/bin/env python3
"""The one place the two OpenCV ArUco APIs are reconciled.

OpenCV 4.7 replaced the free-function ArUco API with objects and deleted the
old names rather than deprecating them, so code written against either version
raises AttributeError on the other. That matters here because this package is
run in two places: the ROS 2 Humble container, which ships OpenCV 4.5, and a
host install, which is likely 4.8 or newer.

Both are supported by dispatching on what cv2 actually exposes, not on a
version number -- distributions patch the aruco module independently of the
OpenCV version they claim. Everything ArUco-related that changed lives here, so
the detector node reads the same on both.

Behaviour is meant to be identical across the two paths, and where the old API
hid a choice the new one makes explicit, the old behaviour is reproduced:

- estimatePoseSingleMarkers is gone in the new API. It ran solvePnP with
  SOLVEPNP_IPPE_SQUARE against object points centred on the marker, +Z out of
  the printed face, and that is reproduced here exactly. Substituting the
  default iterative solver instead would reintroduce the planar two-branch
  ambiguity that the ChArUco path had to be rewritten to avoid.

- interpolateCornersCharuco is gone. CharucoDetector replaces it, and is
  constructed with the SAME DetectorParameters as the marker detector so
  corner refinement still applies; letting it build its own would silently drop
  the subpixel refinement that is worth millimetres here.
"""

import cv2
import numpy as np

# Dispatch on capability rather than on cv2.__version__.
NEW_API = hasattr(cv2.aruco, 'ArucoDetector')


def get_dictionary(dictionary_enum):
    """Predefined dictionary, by cv2.aruco.DICT_* enum."""
    if NEW_API:
        return cv2.aruco.getPredefinedDictionary(dictionary_enum)
    return cv2.aruco.Dictionary_get(dictionary_enum)


def create_detector_parameters():
    """Fresh detector parameters. Attribute names match across both APIs."""
    if NEW_API:
        return cv2.aruco.DetectorParameters()
    return cv2.aruco.DetectorParameters_create()


class MarkerDetector:
    """Detect ArUco markers, whichever API is installed.

    Wraps the dictionary and parameters together because the new API binds them
    at construction while the old one takes them per call.
    """

    def __init__(self, dictionary, parameters):
        self.dictionary = dictionary
        self.parameters = parameters
        self._detector = (cv2.aruco.ArucoDetector(dictionary, parameters)
                          if NEW_API else None)

    def detect(self, gray):
        """Returns (corners, ids); ids is None when nothing was found."""
        if NEW_API:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.parameters)
        return corners, ids


def estimate_marker_pose(corner_set, marker_size, camera_matrix, dist_coeffs):
    """Pose of one marker from its four outer corners, as (rvec, tvec).

    The marker frame is centred on the marker with +Z out of the printed face,
    which is what estimatePoseSingleMarkers used and what the rest of this
    package assumes.
    """
    if not NEW_API:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corner_set], marker_size, camera_matrix, dist_coeffs)
        return rvecs[0][0], tvecs[0][0]

    half = marker_size / 2.0
    object_points = np.array([[-half, half, 0.0],
                              [half, half, 0.0],
                              [half, -half, 0.0],
                              [-half, -half, 0.0]], dtype=np.float32)
    image_points = np.asarray(corner_set, dtype=np.float32).reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix,
                                  dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, None

    # IPPE_SQUARE can return NaN while still reporting success, for a square
    # that is exactly fronto-parallel and exactly symmetric about the principal
    # point: the homography degenerates to a similarity and the decomposition
    # divides by zero. Rare on real images, easy to hit on rendered ones, and
    # `ok` does not catch it. A NaN here would be averaged into a pose, written
    # to the dataset and carried into the solve without ever raising, so it is
    # rejected as a non-detection instead.
    if not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))):
        return None, None

    return rvec.reshape(3), tvec.reshape(3)


def create_charuco_board(squares_x, squares_y, square_length, marker_length,
                         dictionary):
    """A ChArUco board of the given geometry."""
    if NEW_API:
        return cv2.aruco.CharucoBoard((squares_x, squares_y), square_length,
                                      marker_length, dictionary)
    return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_length,
                                         marker_length, dictionary)


def board_chessboard_corners(board):
    """The board's interior corners in board coordinates, as an (N, 3) array."""
    if NEW_API:
        return board.getChessboardCorners()
    return board.chessboardCorners


class CharucoInterpolator:
    """Interior chessboard corners of a board that is partly in view.

    The new API's CharucoDetector is built once and reused; the old API's
    interpolateCornersCharuco is a free call. Both are given the marker
    detections already computed, so markers are not detected twice.
    """

    def __init__(self, board, dictionary, parameters):
        self.board = board
        self.dictionary = dictionary
        self._detector = (
            cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(),
                                      parameters)
            if NEW_API else None)

    def interpolate(self, gray, corners, ids):
        """Returns (count, charuco_corners, charuco_ids)."""
        # No markers is an ordinary frame, not an error: the board leaves the
        # view constantly while the arm moves between poses. The new API
        # returns empty for that, but interpolateCornersCharuco asserts on it
        # (`_markerIds.getMat().total() > 0`), so the old path has to be
        # guarded here or the detector dies on the first empty frame. Humble
        # ships OpenCV 4.x, so the old path is the one that actually runs.
        if ids is None or len(ids) == 0:
            return 0, None, None

        if not NEW_API:
            return cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board)

        charuco_corners, charuco_ids, _, _ = self._detector.detectBoard(
            gray, None, None, corners, ids)
        if charuco_ids is None:
            return 0, None, None
        return len(charuco_ids), charuco_corners, charuco_ids


def generate_marker_image(dictionary, marker_id, size_px):
    """Render a marker to a square image of the given pixel size."""
    if NEW_API:
        return cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    return cv2.aruco.drawMarker(dictionary, marker_id, size_px)


def generate_board_image(board, width_px, height_px, margin_px, border_bits):
    """Render a ChArUco board to an image."""
    if NEW_API:
        return board.generateImage((width_px, height_px), marginSize=margin_px,
                                   borderBits=border_bits)
    return board.draw((width_px, height_px), marginSize=margin_px,
                      borderBits=border_bits)
