# Snaps the projected field grid onto the painted yard lines
import cv2
import numpy as np

from src.field_model import LOWER_NUMBER_Y, UPPER_NUMBER_Y
from src.line_score import (
    LINE_BRIGHTNESS,
    LINE_SAMPLES,
    measure_line_error,
    project_field_points,
)


# Each refinement pass searches up to this many pixels on each side of a
# projected yard line. Later passes start closer, so they search less.
SEARCH_RADII = (20, 8, 4)

# Never search past this fraction of the gap to the next yard line, so a
# sample cannot jump onto the wrong line.
MAX_SEARCH_FRACTION = 0.4

# Painted yard lines are thin. Anything wider across is a number, logo,
# or player.
MAX_LINE_WIDTH = 12

# How far, in pixels, a sample may sit from its painted line before
# RANSAC treats it as an outlier.
SNAP_THRESHOLD = 2.0

# Keep a refit only when at least this many samples, spread over at
# least this many yard lines, agree with it.
MIN_SNAP_SAMPLES = 40
MIN_SNAP_LINES = 3


def create_snap_samples():
    """
    Return the line samples, minus the ones beside the painted numbers,
    whose digits sit right next to the line.
    """

    near_number = np.minimum(
        np.abs(LINE_SAMPLES[:, 1] - UPPER_NUMBER_Y),
        np.abs(LINE_SAMPLES[:, 1] - LOWER_NUMBER_Y),
    ) < 2.0

    return LINE_SAMPLES[~near_number]


SNAP_SAMPLES = create_snap_samples()


def find_line_centers(painted, points, normals, radii):
    """
    Walk along each normal from its point and return the offset, in
    pixels, to the center of the nearest painted line. Return NaN where
    no line is close enough.
    """

    max_radius = int(np.ceil(radii.max()))
    offsets = np.arange(-max_radius, max_radius + 1, dtype=np.float32)

    # One row per point, one column per offset along its normal.
    map_x = points[:, 0:1] + offsets[None, :] * normals[:, 0:1]
    map_y = points[:, 1:2] + offsets[None, :] * normals[:, 1:2]

    profiles = cv2.remap(
        painted,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    within_radius = np.abs(offsets)[None, :] <= radii[:, None]
    on_line = (profiles > LINE_BRIGHTNESS) & within_radius

    rows = np.arange(len(points))

    # The nearest bright pixel picks which line to use.
    distance = np.where(on_line, np.abs(offsets)[None, :], np.inf)
    nearest = distance.argmin(axis=1)
    found = np.isfinite(distance[rows, nearest])

    # Number the runs of bright pixels in each row, then keep the run
    # holding the nearest bright pixel.
    run_starts = on_line & ~np.pad(on_line, ((0, 0), (1, 0)))[:, :-1]
    run_ids = np.cumsum(run_starts, axis=1)
    in_run = on_line & (run_ids == run_ids[rows, nearest][:, None])

    # The brightness-weighted middle of the run is the line's center,
    # unlike its nearest edge.
    weights = np.where(in_run, profiles - LINE_BRIGHTNESS, 0.0)
    centers = (weights * offsets[None, :]).sum(axis=1) / np.maximum(
        weights.sum(axis=1),
        1e-6,
    )

    # A run cut off by the search window has an unknown center, so the
    # pixels just before and after it must be inside the window.
    first = np.where(in_run, offsets[None, :], np.inf).min(axis=1)
    last = np.where(in_run, offsets[None, :], -np.inf).max(axis=1)
    complete = (first - 1 >= -radii) & (last + 1 <= radii)
    thin = in_run.sum(axis=1) <= MAX_LINE_WIDTH

    centers[~(found & complete & thin)] = np.nan

    return centers


def refit_to_lines(homography, painted, field_mask, search_radius):
    """
    Move each projected sample to the center of the painted line beside
    it and fit a new homography to the moved samples. Return None when
    too few samples find a line.

    Samples move only across their line. Along the line they keep the
    position the current homography gives them, which comes from the
    tracked points.
    """

    height, width = painted.shape[:2]

    points, in_front = project_field_points(
        homography,
        SNAP_SAMPLES,
        painted.shape,
    )

    # A second point along each yard line gives its direction on screen,
    # and the same spot on the yard lines 5 yards away gives the gap.
    along, along_in_front = project_field_points(
        homography,
        SNAP_SAMPLES + (0.0, 0.5),
        painted.shape,
    )

    next_line, next_in_front = project_field_points(
        homography,
        SNAP_SAMPLES + (5.0, 0.0),
        painted.shape,
    )

    previous_line, previous_in_front = project_field_points(
        homography,
        SNAP_SAMPLES - (5.0, 0.0),
        painted.shape,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        direction = along - points
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        normals = np.column_stack([-direction[:, 1], direction[:, 0]])

        gap = np.minimum(
            np.abs(((next_line - points) * normals).sum(axis=1)),
            np.abs(((previous_line - points) * normals).sum(axis=1)),
        )

        radii = np.minimum(search_radius, MAX_SEARCH_FRACTION * gap)

        usable = (
            in_front
            & along_in_front
            & next_in_front
            & previous_in_front
            & np.isfinite(points).all(axis=1)
            & np.isfinite(normals).all(axis=1)
            & (radii >= 2)
            & (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height)
        )

    columns = points[usable, 0].astype(int)
    rows = points[usable, 1].astype(int)
    usable[usable] = field_mask[rows, columns] > 0

    if usable.sum() < MIN_SNAP_SAMPLES:
        return None

    centers = find_line_centers(
        painted,
        points[usable],
        normals[usable],
        radii[usable],
    )

    found = ~np.isnan(centers)

    if found.sum() < MIN_SNAP_SAMPLES:
        return None

    field_points = SNAP_SAMPLES[usable][found]
    moved_points = (
        points[usable][found]
        + centers[found, None] * normals[usable][found]
    )

    # Plain RANSAC, because the USAC methods reject these samples: every
    # sample lies on one of a few lines, which they treat as degenerate.
    refit, inlier_mask = cv2.findHomography(
        field_points,
        moved_points,
        cv2.RANSAC,
        SNAP_THRESHOLD,
    )

    if refit is None:
        return None

    inliers = inlier_mask.reshape(-1).astype(bool)
    line_count = len(np.unique(field_points[inliers, 0]))

    if inliers.sum() < MIN_SNAP_SAMPLES or line_count < MIN_SNAP_LINES:
        return None

    return refit


def snap_to_lines(homography, painted, distance_map, field_mask):
    """
    Refit the homography so the projected yard lines sit on the painted
    ones. Return None when the refit fails or makes the line error worse.
    """

    # Sample between pixels, so convert once instead of on every pass.
    painted = painted.astype(np.float32)
    snapped = None

    for search_radius in SEARCH_RADII:
        refit = refit_to_lines(
            homography if snapped is None else snapped,
            painted,
            field_mask,
            search_radius,
        )

        if refit is None:
            break

        snapped = refit

    if snapped is None:
        return None

    before = measure_line_error(distance_map, field_mask, homography)
    after = measure_line_error(distance_map, field_mask, snapped)

    if np.isnan(after) or (not np.isnan(before) and after > before):
        return None

    return snapped
