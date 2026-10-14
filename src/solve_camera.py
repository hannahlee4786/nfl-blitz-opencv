# Solves a pinhole camera for every tracked frame and exports it
import argparse
import json

import cv2
import numpy as np

from src.camera_model import (
    camera_matrix,
    find_up_sign,
    fit_pose,
    fit_shared_focal,
    focal_from_fov,
    fov_from_focal,
    parameters_to_pose,
    pose_homography,
    pose_to_parameters,
    smooth_series,
)
from src.line_score import (
    create_distance_map,
    find_painted_lines,
    measure_line_error,
)
from src.camera_view import render_camera_video
from src.preprocessing import preprocess_frame
from src.video_paths import (
    camera_path,
    list_input_videos,
    resolve_video,
    tracking_path,
)

# Starting guess for the focal length. Every clip so far is 44 to 47
# degrees wide.
INITIAL_FOV = 46.0

# Frames per window when checking whether the focal length changes.
FOCAL_WINDOW = 30

# Let the focal length change only when per-window focal lengths cut the
# camera's fit error by more than this fraction. More freedom always
# helps a little: clips with a fixed lens gain 12 to 18 percent.
ZOOM_IMPROVEMENT = 0.3

# How strongly to smooth the camera path. Higher is smoother but cuts
# corners on fast moves.
SMOOTHING = 2.0

# How much each frame counts when smoothing, by how it was tracked.
# Frames with weight 0 are filled in from their neighbors.
SOURCE_WEIGHTS = {
    "snapped": 1.0,
    "predicted": 0.2,
    "filled": 0.0,
}


def find_sources(tracking):
    """Label each frame by how far its homography can be trusted."""

    statuses = tracking["status"]
    snapped = tracking["snapped"]

    return np.where(
        statuses == "LOST",
        "filled",
        np.where(snapped & (statuses == "GOOD"), "snapped", "predicted"),
    )


def fit_focal_lengths(homographies, sources, frame_size, principal_point):
    """
    Return (focals, window_fovs, lens_changes): a focal length for every
    frame, the field of view fitted to each window, and whether the
    focal length had to change over the clip.
    """

    picture_width = frame_size[0]
    trusted = np.flatnonzero(sources == "snapped")

    shared_focal, _ = fit_shared_focal(
        homographies[trusted],
        frame_size,
        principal_point,
        focal_from_fov(INITIAL_FOV, picture_width),
    )

    window_centers = []
    window_focals = []

    for start in range(0, len(homographies), FOCAL_WINDOW):
        frames = trusted[
            (trusted >= start) & (trusted < start + FOCAL_WINDOW)
        ]

        if len(frames) < 5:
            continue

        window_focal, _ = fit_shared_focal(
            homographies[frames],
            frame_size,
            principal_point,
            shared_focal,
        )

        window_centers.append(frames.mean())
        window_focals.append(window_focal)

    window_fovs = [
        fov_from_focal(focal, picture_width) for focal in window_focals
    ]

    if len(window_focals) < 2:
        return np.full(len(homographies), shared_focal), window_fovs, False

    frame_numbers = np.arange(len(homographies))
    varying_focals = np.interp(frame_numbers, window_centers, window_focals)

    # Compare how well each choice reproduces the trusted frames.
    def fit_error(focals):
        errors = []

        for frame in trusted:
            pose = fit_pose(
                homographies[frame],
                camera_matrix(focals[frame], principal_point),
                frame_size,
            )

            if pose is not None:
                errors.append(pose[2])

        return np.sqrt(np.mean(np.square(errors)))

    shared_focals = np.full(len(homographies), shared_focal)
    lens_changes = (
        fit_error(varying_focals)
        < (1 - ZOOM_IMPROVEMENT) * fit_error(shared_focals)
    )

    if not lens_changes:
        return shared_focals, window_fovs, False

    return (
        smooth_series(varying_focals, np.ones(len(homographies)), SMOOTHING),
        window_fovs,
        True,
    )


def solve_poses(homographies, focals, sources, frame_size, principal_point):
    """
    Return the raw camera parameters of every frame, (x, y, height, pan,
    tilt, roll) per row, the fit error of each, and the up sign.
    Frames whose pose cannot be solved are marked "filled" in `sources`.
    """

    count = len(homographies)
    parameters = np.zeros((count, 6))
    errors = np.full(count, np.nan)
    poses = [None] * count

    for frame in range(count):
        pose = fit_pose(
            homographies[frame],
            camera_matrix(focals[frame], principal_point),
            frame_size,
        )

        if pose is None:
            sources[frame] = "filled"
            continue

        rvec, tvec, errors[frame] = pose
        poses[frame] = (cv2.Rodrigues(rvec)[0], tvec)

    # The camera stays on one side of the field for the whole clip.
    up_signs = [
        find_up_sign(*poses[frame])
        for frame in np.flatnonzero(sources == "snapped")
        if poses[frame] is not None
    ]

    up_sign = 1.0 if np.mean(up_signs) > 0 else -1.0

    for frame, pose in enumerate(poses):
        if pose is not None:
            parameters[frame] = pose_to_parameters(*pose, up_sign)

    return parameters, errors, up_sign


def smooth_parameters(parameters, sources):
    weights = np.array([SOURCE_WEIGHTS[source] for source in sources])
    used = weights > 0

    smoothed = np.zeros_like(parameters)

    for column in range(parameters.shape[1]):
        values = parameters[:, column].copy()

        # Pan wraps around at +-180 degrees. Unwrap it so smoothing does
        # not average 179 and -179 into 0.
        if column == 3:
            values[used] = np.degrees(np.unwrap(np.radians(values[used])))

        smoothed[:, column] = smooth_series(
            np.where(used, values, 0.0),
            weights,
            SMOOTHING,
        )

    smoothed[:, 3] = (smoothed[:, 3] + 180.0) % 360.0 - 180.0

    return smoothed


def measure_line_errors(input_path, frame_indices, active_area, cameras):
    """
    Return the line error of every frame for each list of homographies
    in `cameras`, one row per list. A None homography scores NaN.
    """

    capture = cv2.VideoCapture(str(input_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_indices[0]))

    errors = np.full((len(cameras), len(frame_indices)), np.nan)

    for index in range(len(frame_indices)):
        success, frame = capture.read()

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, field_mask = preprocess_frame(frame, active_area)
        distance_map = create_distance_map(
            find_painted_lines(gray, field_mask)
        )

        for row, homographies in enumerate(cameras):
            if homographies[index] is not None:
                errors[row, index] = measure_line_error(
                    distance_map,
                    field_mask,
                    homographies[index],
                )

    capture.release()

    return errors


def export_json(path, input_path, frame_size, fps, frames):
    data = {
        "video": str(input_path.resolve()),
        "fps": fps,
        "image_width": frame_size[0],
        "image_height": frame_size[1],
        "frames": frames,
    }

    with open(path, "w") as file:
        json.dump(data, file, indent=1)


def solve_camera(input_path):
    tracking = np.load(tracking_path(input_path))

    homographies = tracking["homography"]
    frame_indices = tracking["frame_index"]
    left, right = (int(value) for value in tracking["active_area"])

    capture = cv2.VideoCapture(str(input_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()

    # The principal point is the middle of the game picture, in OpenCV's
    # pixel-center coordinates.
    principal_point = ((left + right) / 2 - 0.5, height / 2 - 0.5)
    picture_size = (right - left, height)
    frame_size = (width, height)

    sources = find_sources(tracking)

    if not (sources == "snapped").any():
        raise RuntimeError(
            f"{input_path.name}: no frame was snapped to the painted "
            "lines, so there is nothing reliable to solve from."
        )

    focals, window_fovs, lens_changes = fit_focal_lengths(
        homographies,
        sources,
        frame_size,
        principal_point,
    )

    raw_parameters, fit_errors, up_sign = solve_poses(
        homographies,
        focals,
        sources,
        frame_size,
        principal_point,
    )

    parameters = smooth_parameters(raw_parameters, sources)

    poses = [parameters_to_pose(row, up_sign) for row in parameters]

    camera_homographies = np.array(
        [
            pose_homography(
                camera_matrix(focal, principal_point),
                rotation,
                translation,
            )
            for focal, (rotation, translation) in zip(focals, poses)
        ]
    )

    # The unsmoothed camera, to see what smoothing costs.
    raw_homographies = [
        None if source == "filled" else pose_homography(
            camera_matrix(focal, principal_point),
            *parameters_to_pose(row, up_sign),
        )
        for focal, row, source in zip(focals, raw_parameters, sources)
    ]

    tracked_errors, raw_errors, line_errors = measure_line_errors(
        input_path,
        frame_indices,
        (left, right),
        [list(homographies), raw_homographies, list(camera_homographies)],
    )

    fovs = np.array(
        [fov_from_focal(focal, picture_size[0]) for focal in focals]
    )

    output_path = camera_path(input_path)

    np.savez(
        output_path,
        frame_index=frame_indices,
        time=frame_indices / fps,
        source=sources,
        focal=focals,
        fov=fovs,
        position=parameters[:, :3],
        pan=parameters[:, 3],
        tilt=parameters[:, 4],
        roll=parameters[:, 5],
        rotation=np.array([rotation for rotation, _ in poses]),
        translation=np.array([translation for _, translation in poses]),
        homography=camera_homographies,
        line_error=line_errors,
        raw_parameters=raw_parameters,
        fit_error=fit_errors,
        principal_point=np.array(principal_point),
        active_area=np.array((left, right)),
        up_sign=up_sign,
        lens_changes=lens_changes,
    )

    frames = [
        {
            "frame_index": int(frame_indices[index]),
            "time": float(frame_indices[index] / fps),
            "source": str(sources[index]),
            "focal_px": float(focals[index]),
            "fov_deg": float(fovs[index]),
            "position_yards": [float(value) for value in parameters[index, :3]],
            "pan_deg": float(parameters[index, 3]),
            "tilt_deg": float(parameters[index, 4]),
            "roll_deg": float(parameters[index, 5]),
        }
        for index in range(len(frame_indices))
    ]

    json_path = output_path.with_suffix(".json")
    export_json(json_path, input_path, frame_size, fps, frames)

    video_path = output_path.with_suffix(".mp4")
    render_camera_video(input_path, output_path, video_path)

    trusted = sources == "snapped"
    window_text = " ".join(f"{fov:.0f}" for fov in window_fovs)

    if lens_changes:
        print(
            f"Lens: changes from {fovs.max():.1f} to {fovs.min():.1f} "
            f"degrees wide (per {FOCAL_WINDOW} frames: {window_text})"
        )
    else:
        print(
            f"Lens: fixed, {fovs[0]:.1f} degrees wide, focal length "
            f"{focals[0]:.0f} px (per {FOCAL_WINDOW} frames: {window_text})"
        )

    print(
        f"Camera vs. tracked grid: median {np.nanmedian(fit_errors[trusted]):.1f} px "
        f"on snapped frames. Roll: median {np.median(raw_parameters[trusted, 5]):+.2f} "
        f"degrees"
    )

    def describe(errors):
        return (
            f"{np.nanmedian(errors):.1f} / "
            f"{np.nanpercentile(errors, 95):.1f} px"
        )

    print(
        "Line error, median / 95th percentile: tracked grid "
        f"{describe(tracked_errors)}, camera {describe(raw_errors)}, "
        f"smoothed camera {describe(line_errors)}"
    )

    filled = int((sources == "filled").sum())

    if filled:
        print(
            f"{filled} frames had no usable grid and were filled in from "
            "their neighbors"
        )

    print(f"Saved camera to: {output_path} and {json_path.name}")
    print(f"Saved camera video to: {video_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Solve the camera of tracked videos.",
    )

    parser.add_argument(
        "videos",
        nargs="*",
        help="Video paths, or the start of file names in data/input. "
        "Leave empty to solve every tracked video.",
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()

    if arguments.videos:
        input_paths = [resolve_video(video) for video in arguments.videos]
    else:
        input_paths = [
            video for video in list_input_videos()
            if tracking_path(video).exists()
        ]

    if not input_paths:
        raise RuntimeError(
            "No tracked videos. Run python -m src.track_video first."
        )

    for input_path in input_paths:
        if not tracking_path(input_path).exists():
            print(
                f"Skipping {input_path.name}: not tracked. Run "
                f'python -m src.track_video "{input_path.stem}"'
            )
            continue

        print(f"Solving the camera of {input_path.name}...")
        solve_camera(input_path)


if __name__ == "__main__":
    main()
