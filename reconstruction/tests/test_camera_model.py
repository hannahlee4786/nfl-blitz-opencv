import cv2
import numpy as np
import pytest

from nfl_reconstruction.camera_model import (
    camera_matrix,
    fit_pose,
    fit_shared_focal,
    fov_from_focal,
    focal_from_fov,
    parameters_to_pose,
    pose_to_parameters,
    smooth_series,
)
from tests.fixtures import BLITZ_PARAMETERS, FOCAL, FRAME_SIZE, blitz_homography

PRINCIPAL_POINT = ((FRAME_SIZE[0] - 1) / 2, (FRAME_SIZE[1] - 1) / 2)


@pytest.mark.parametrize("up_sign", [1.0, -1.0])
def test_parameters_round_trip_through_a_proper_rotation(up_sign):
    parameters = np.array([83.0, 26.67, 15.7, 171.0, 27.0, 0.8])

    rotation, translation = parameters_to_pose(parameters, up_sign)

    assert np.allclose(rotation @ rotation.T, np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert np.allclose(pose_to_parameters(rotation, translation, up_sign), parameters)


def test_pose_fit_recovers_the_camera_behind_a_homography():
    matrix = camera_matrix(FOCAL, PRINCIPAL_POINT)

    rvec, tvec, error = fit_pose(blitz_homography(), matrix, FRAME_SIZE)
    parameters = pose_to_parameters(cv2.Rodrigues(rvec)[0], tvec, 1.0)

    pan_difference = (parameters[3] - BLITZ_PARAMETERS[3] + 180.0) % 360.0 - 180.0

    assert error < 1e-3
    assert pan_difference == pytest.approx(0.0, abs=1e-3)
    assert np.allclose(np.delete(parameters, 3), np.delete(BLITZ_PARAMETERS, 3), atol=1e-3)


def test_pose_fit_needs_the_field_in_view():
    matrix = camera_matrix(FOCAL, PRINCIPAL_POINT)
    away = np.array(((1.0, 0.0, 10 * FRAME_SIZE[0]), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))

    assert fit_pose(away @ blitz_homography(), matrix, FRAME_SIZE) is None


def test_shared_focal_fit_recovers_the_lens():
    homographies = [
        blitz_homography((83.0, 26.67, 15.7, 180.0, 27.0, 0.0)),
        blitz_homography((70.0, 20.0, 12.0, 170.0, 24.0, 0.0)),
        blitz_homography((90.0, 33.0, 18.0, 188.0, 30.0, 0.0)),
    ]

    focal, rms = fit_shared_focal(homographies, FRAME_SIZE, PRINCIPAL_POINT, 700.0)

    assert focal == pytest.approx(FOCAL, rel=1e-3)
    assert rms < 0.01


def test_field_of_view_and_focal_length_convert_both_ways():
    assert fov_from_focal(focal_from_fov(46.0, 1356), 1356) == pytest.approx(46.0)


def test_smoothing_fills_unweighted_frames_from_their_neighbors():
    values = np.array([0.0, 1.0, 2.0, 3.0, 100.0, 5.0, 6.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0])

    smoothed = smooth_series(values, weights, 2.0)

    assert np.allclose(smoothed, np.arange(7.0))
