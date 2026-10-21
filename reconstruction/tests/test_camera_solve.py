import json

import numpy as np
import pytest

from nfl_reconstruction.camera import field_homography
from nfl_reconstruction.camera_solve import (
    export_pose_json,
    find_sources,
    pose_to_state,
    smooth_parameters,
    solve_camera,
)
from tests.fixtures import BLITZ_PARAMETERS, FOCAL, FRAME_SIZE, blitz_homography

WHOLE_PICTURE = (0, FRAME_SIZE[0])


def camera_path(count, up_sign=1.0, focals=None):
    """A camera dollying back across the 70 while it pans over 180."""

    parameters = np.array(
        [
            (83.0 - 0.3 * frame, 26.67 + 0.1 * frame, 15.7, 175.0 + 0.4 * frame, 27.0, 0.0)
            for frame in range(count)
        ]
    )
    parameters[:, 3] = (parameters[:, 3] + 180.0) % 360.0 - 180.0
    focals = np.full(count, FOCAL) if focals is None else focals
    homographies = np.array(
        [
            blitz_homography(row, up_sign, focal)
            for row, focal in zip(parameters, focals)
        ]
    )
    return parameters, homographies


def angle_difference(first, second):
    return (np.asarray(first) - np.asarray(second) + 180.0) % 360.0 - 180.0


def test_sources_label_frames_by_how_far_they_can_be_trusted():
    sources = find_sources(
        ["GOOD", "GOOD", "FAIR", "LOST"], [True, False, True, True]
    )

    assert sources.tolist() == ["snapped", "predicted", "predicted", "filled"]


def test_solver_recovers_a_fixed_lens_and_the_camera_path():
    truth, homographies = camera_path(25)

    solution = solve_camera(
        homographies, ["GOOD"] * 25, [True] * 25, FRAME_SIZE, WHOLE_PICTURE
    )

    assert not solution.lens_changes
    assert solution.up_sign == 1.0
    assert np.allclose(solution.focals, FOCAL, rtol=1e-3)
    assert np.allclose(solution.parameters[:, :3], truth[:, :3], atol=0.05)
    assert np.allclose(angle_difference(solution.parameters[:, 3], truth[:, 3]), 0, atol=0.05)
    assert np.allclose(solution.parameters[:, 4:], truth[:, 4:], atol=0.05)
    assert np.nanmax(solution.fit_errors) < 0.01


def test_solver_fills_lost_frames_from_their_neighbors():
    truth, homographies = camera_path(25)
    statuses = ["GOOD"] * 25
    for frame in range(10, 15):
        statuses[frame] = "LOST"
        homographies[frame] = np.eye(3)

    solution = solve_camera(homographies, statuses, [True] * 25, FRAME_SIZE, WHOLE_PICTURE)

    assert solution.sources[10:15].tolist() == ["filled"] * 5
    assert all(solution.raw_homographies[frame] is None for frame in range(10, 15))
    assert np.allclose(solution.parameters[10:15, :3], truth[10:15, :3], atol=0.05)


def test_solver_coasts_then_holds_the_camera_when_the_clip_ends_lost():
    truth, homographies = camera_path(40)
    statuses = ["GOOD"] * 40
    for frame in range(20, 40):
        statuses[frame] = "LOST"
        homographies[frame] = np.eye(3)

    solution = solve_camera(homographies, statuses, [True] * 40, FRAME_SIZE, WHOLE_PICTURE)

    # The camera moves steadily, so coasting follows it for a while.
    assert np.allclose(solution.parameters[20:30, :3], truth[20:30, :3], atol=0.05)
    assert np.allclose(solution.parameters[30:], solution.parameters[29])


def test_smoothing_never_coasts_the_camera_under_the_field():
    # Clip 13 ends on a sideline close-up with no grid while the camera
    # comes down fast. Carrying that on put it under the field.
    count = 20
    parameters = np.array(
        [(50.0, 26.0, 10.0 - 1.5 * frame, 0.0, 27.0, 0.0) for frame in range(count)]
    )
    sources = np.array(["snapped"] * 6 + ["filled"] * (count - 6))

    smoothed = smooth_parameters(parameters, sources)

    assert (smoothed[:, 2] > 0).all()
    assert np.allclose(smoothed[-1], smoothed[-2])


def test_solver_lets_the_focal_length_change_when_the_lens_zooms():
    focals = np.linspace(800.0, 1000.0, 90)
    _, homographies = camera_path(90, focals=focals)

    solution = solve_camera(
        homographies, ["GOOD"] * 90, [True] * 90, FRAME_SIZE, WHOLE_PICTURE
    )

    # One focal length is fitted per 30-frame window and interpolated
    # between window centers, so only the middle of the clip follows the
    # zoom exactly.
    assert solution.lens_changes
    assert np.allclose(solution.focals[15:75], focals[15:75], rtol=0.01)
    assert solution.fovs[0] > solution.fovs[-1]


def test_solver_handles_a_mirrored_calibration():
    truth, homographies = camera_path(25, up_sign=-1.0)

    solution = solve_camera(
        homographies, ["GOOD"] * 25, [True] * 25, FRAME_SIZE, WHOLE_PICTURE
    )

    assert solution.up_sign == -1.0
    assert np.allclose(solution.parameters[:, :3], truth[:, :3], atol=0.05)


def test_solver_needs_a_snapped_frame():
    _, homographies = camera_path(5)

    with pytest.raises(RuntimeError, match="snapped"):
        solve_camera(homographies, ["GOOD"] * 5, [False] * 5, FRAME_SIZE, WHOLE_PICTURE)


@pytest.mark.parametrize("up_sign", [1.0, -1.0])
def test_camera_state_reproduces_the_solved_homography(up_sign):
    parameters = np.array([70.0, 31.0, 12.5, 166.0, 24.0, 0.6])

    state = pose_to_state(parameters, FOCAL, up_sign)
    homography = field_homography(state, FRAME_SIZE).detach().numpy()

    assert state.up_sign == int(up_sign)
    assert state.pitch_degrees == pytest.approx(24.0)
    assert np.allclose(homography, blitz_homography(parameters, up_sign), atol=1e-6)


def test_camera_state_needs_a_camera_looking_down():
    with pytest.raises(ValueError, match="looking down"):
        pose_to_state(np.array([70.0, 31.0, 12.5, 166.0, -5.0, 0.0]), FOCAL, 1.0)


def test_pose_export_lists_every_frame(tmp_path):
    _, homographies = camera_path(25)
    solution = solve_camera(
        homographies, ["GOOD"] * 25, [True] * 25, FRAME_SIZE, WHOLE_PICTURE
    )
    path = tmp_path / "camera-pose.json"

    export_pose_json(path, "clip.mp4", FRAME_SIZE, 30.0, list(range(10, 35)), solution)
    data = json.loads(path.read_text())

    assert data["up_sign"] == 1
    assert len(data["frames"]) == 25
    assert data["frames"][0]["frame_index"] == 10
    assert data["frames"][0]["time"] == pytest.approx(10 / 30)
    assert data["frames"][0]["position_yards"] == pytest.approx(
        list(BLITZ_PARAMETERS[:3]), abs=0.05
    )
