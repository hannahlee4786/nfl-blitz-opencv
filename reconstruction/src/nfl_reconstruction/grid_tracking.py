"""Follows the field grid through a clip, snapping it onto the painted lines.

Ported from the tracking loop in src/track_video.py on the camera-tracking
branch. Each frame, points tracked on the grass predict where the grid
moved, and the grid is then snapped onto the painted yard lines so errors
cannot build up. When the camera moves too fast for the tracked points,
features matched between the two frames give the prediction instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .line_score import (
    create_distance_map,
    find_painted_lines,
    get_tracking_status,
    measure_line_error,
    measure_line_fit,
    project_field_points,
)
from .line_snapping import snap_to_lines
from .point_tracking import match_frame_motion, track_screen_points
from .preprocessing import find_field_mask

# The clicked anchor points only set the starting homography. Tracking
# uses many automatic points on the field instead, so losing some to
# players or the screen edge does not matter.
MAX_FEATURES = 400

# Add new points when fewer than this many are still tracked.
MIN_FEATURES = 200

# Minimum distance, in pixels, between two tracked points.
FEATURE_SPACING = 12

# How far, in pixels, a point may be from where the homography places
# it before RANSAC treats it as an outlier.
RANSAC_THRESHOLD = 3.0

# After a turnover the camera swings around behind the new offense in a
# few frames, too fast for points or matched features to follow. To find
# the grid again, circle the last good camera around spots this far down
# the middle of the picture (as a fraction of its height), where the play
# usually is.
SWING_PIVOT_HEIGHTS = (0.5, 0.6, 0.7, 0.8, 0.9)
SWING_ANGLES = np.arange(-180.0, 180.0, 5.0)

# A grid turned 180 degrees fits the yard lines as well as the right one,
# so the turn must be pinned down some other way. Until the swing's
# direction is known, allow up to this many degrees either way: the first
# frame of a swing turns less than that.
SWING_FIRST_TURN = 90.0

# A turn bigger than this sets which way the camera is swinging. After
# that, allow turns up to 180 degrees that way but only this many back:
# on Clip 14 the 180-degree twin of the right grid turned the other way.
SWING_MIN_TURN = 10.0
SWING_BACKTRACK = 20.0

# Forget the direction after this many frames tracked without a search.
SWING_SETTLE_FRAMES = 15

# Then refine the best turns in steps this fine, so the snap starts close
# enough not to slide the grid along the yard lines.
SWING_FINE_ANGLES = np.arange(-4.0, 4.1, 1.0)

# How many of the best-scoring turns to try snapping.
SWING_SNAP_TRIES = 8

# Yard lines repeat every 5 yards and say nothing about where along them
# the grid sits, so several turns snap onto them equally well. Pick the
# one that best lines up the paint of the last good frame with this one:
# the numbers, hash marks, and sidelines then land where they should. The
# grass is no help; its texture looks different from every angle. Sample
# every this many pixels, blurred by this much so near misses still count.
SWING_MATCH_STEP = 4
SWING_MATCH_BLUR = 3.0

# Accept a found grid only when this many samples overlap and its match
# (see match_views) is at least this. Mid-swing on Clip 14 the right grids
# score only 0.11 to 0.32, the view changes so much; the swing's direction,
# not this, is what rules out the wrong ones.
SWING_MIN_MATCH_SAMPLES = 500
SWING_MIN_MATCH = 0.1

# Search when fewer than this fraction of the tracked points could be
# followed into the next frame. Then even a grid that snaps cleanly may be
# 5 yards off, so the search's grids compete with it.
SWING_POINTS_KEPT = 0.5


@dataclass(frozen=True)
class PaintedLines:
    """What one frame shows of the field: the grass and the paint on it."""

    field_mask: NDArray[np.uint8]
    painted: NDArray[np.uint8]
    distance_map: NDArray[np.float32]


@dataclass(frozen=True)
class GoodView:
    """The last frame whose grid sat on the painted lines."""

    homography: NDArray[np.float64]
    blurred: NDArray[np.float32]
    field_mask: NDArray[np.uint8]


@dataclass(frozen=True)
class GridTrack:
    """The tracked field-to-screen homography of every frame."""

    homographies: NDArray[np.float64]
    line_errors: NDArray[np.float64]
    statuses: NDArray[np.str_]
    snapped: NDArray[np.bool_]


def find_frame_lines(
    frame: NDArray[np.uint8],
    excluded: NDArray[np.bool_],
) -> PaintedLines:
    field_mask = find_field_mask(frame, excluded)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    painted = find_painted_lines(gray, field_mask)

    return PaintedLines(field_mask, painted, create_distance_map(painted))


def create_feature_mask(
    field_mask: NDArray[np.uint8],
    existing_points: NDArray[np.float32] | Sequence[tuple[float, float]],
) -> NDArray[np.uint8]:
    """Return the pixels where new feature points may be placed."""

    # Shrink the field so points are not placed on player outlines or
    # HUD edges, which do not move with the field.
    mask = cv2.erode(
        field_mask,
        np.ones((9, 9), np.uint8),
    )

    # Keep new points away from points that are already tracked.
    for x, y in existing_points:
        cv2.circle(
            mask,
            (int(round(x)), int(round(y))),
            FEATURE_SPACING,
            0,
            -1,
        )

    return mask


def screen_to_field(
    screen_points: NDArray[np.float32],
    homography: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Return where each screen point is on the field."""

    if len(screen_points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    field_points = cv2.perspectiveTransform(
        screen_points.reshape(-1, 1, 2).astype(np.float32),
        np.linalg.inv(homography),
    )

    return field_points.reshape(-1, 2).astype(np.float32)


def detect_new_points(
    gray: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    homography: NDArray[np.float64],
    count: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Find new feature points and record where each one is on the field.

    Returns (screen_points, field_points).
    """

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=count,
        qualityLevel=0.01,
        minDistance=FEATURE_SPACING,
        mask=mask,
    )

    if corners is None:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty

    screen_points = corners.reshape(-1, 2).astype(np.float32)

    # Each point stays tied to the same spot on the field. Look up that
    # spot with the current homography, which maps field to screen.
    return screen_points, screen_to_field(screen_points, homography)


def turn_about(
    homography: NDArray[np.float64],
    pivot: NDArray[np.float64],
    degrees: float,
) -> NDArray[np.float64]:
    """
    Return the homography of the camera circled `degrees` around the
    field point `pivot`, keeping its height and tilt.
    """

    # Circling the camera one way shows what turning the field the other
    # way about the same point would.
    angle = np.radians(-degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    shift = pivot - rotation @ pivot

    field_turn = np.array(
        (
            (cosine, -sine, shift[0]),
            (sine, cosine, shift[1]),
            (0.0, 0.0, 1.0),
        )
    )

    return homography @ field_turn


def blur_for_matching(painted: NDArray[np.uint8]) -> NDArray[np.float32]:
    return cv2.GaussianBlur(painted.astype(np.float32), (0, 0), SWING_MATCH_BLUR)


def match_views(
    earlier: GoodView,
    blurred: NDArray[np.float32],
    field_mask: NDArray[np.uint8],
    homography: NDArray[np.float64],
) -> float:
    """
    Return how much of this frame's field the earlier frame explains: how
    well its paint, carried over through the two grids, correlates with
    this frame's on the field both show, times the share of this frame's
    field that is. A wrong grid can correlate well on a small patch, but
    the camera circles the play, so the right one keeps most of the field
    in view. Return NaN when they share too little of it.
    """

    step = SWING_MATCH_STEP
    rows, columns = np.nonzero(field_mask[::step, ::step])
    rows, columns = rows * step, columns * step

    field_points = screen_to_field(
        np.column_stack([columns, rows]).astype(np.float32),
        homography,
    )
    earlier_points, in_front = project_field_points(
        earlier.homography,
        field_points.astype(np.float64),
        earlier.field_mask.shape,
    )

    height, width = earlier.field_mask.shape[:2]

    with np.errstate(invalid="ignore"):
        earlier_columns = np.round(earlier_points[:, 0])
        earlier_rows = np.round(earlier_points[:, 1])

        shared = (
            in_front
            & (earlier_columns >= 0)
            & (earlier_columns < width)
            & (earlier_rows >= 0)
            & (earlier_rows < height)
        )

    earlier_columns = earlier_columns[shared].astype(int)
    earlier_rows = earlier_rows[shared].astype(int)
    on_field = earlier.field_mask[earlier_rows, earlier_columns] > 0

    if on_field.sum() < SWING_MIN_MATCH_SAMPLES:
        return float("nan")

    then = earlier.blurred[earlier_rows[on_field], earlier_columns[on_field]]
    now = blurred[rows[shared][on_field], columns[shared][on_field]]

    then = then - then.mean()
    now = now - now.mean()
    spread = np.sqrt((then @ then) * (now @ now))

    if spread == 0:
        return float("nan")

    shared_fraction = on_field.sum() / len(rows)

    return float((then @ now) / spread * shared_fraction)


def fits_last_good(
    homography: NDArray[np.float64] | None,
    earlier: GoodView,
    blurred: NDArray[np.float32],
    lines: PaintedLines,
) -> bool:
    """
    Return whether a snapped grid both sits on the painted lines and lines
    up the last good frame with this one. A fast move can leave the
    tracked points with a wrong guess that still snaps onto lines 5 yards
    away; only the picture shows it.
    """

    if homography is None:
        return False

    error = measure_line_error(lines.distance_map, lines.field_mask, homography)

    if get_tracking_status(error) == "LOST":
        return False

    match = match_views(earlier, blurred, lines.field_mask, homography)

    # Too little field shared with the last good frame to judge; trust
    # the lines.
    return np.isnan(match) or match >= SWING_MIN_MATCH


def view_heading(
    homography: NDArray[np.float64],
    frame_shape: tuple[int, ...],
) -> float:
    """
    Return which way, in degrees on the field map, the camera looks: from
    the field under the bottom of the picture's middle toward the field
    under its center.
    """

    height, width = frame_shape[:2]
    near, far = screen_to_field(
        np.array(((width / 2, 0.9 * height), (width / 2, 0.5 * height)), dtype=np.float32),
        homography,
    ).astype(np.float64)
    along = far - near

    return float(np.degrees(np.arctan2(along[1], along[0])))


def turn_between(
    earlier: NDArray[np.float64],
    later: NDArray[np.float64],
    frame_shape: tuple[int, ...],
) -> float:
    """Return how many degrees the camera turned, from -180 to 180."""

    turn = view_heading(later, frame_shape) - view_heading(earlier, frame_shape)

    return float((turn + 180.0) % 360.0 - 180.0)


def turn_allowed(turn: float, direction: int) -> bool:
    """Whether a turn fits a swing going `direction` (0 when not known)."""

    if not np.isfinite(turn):
        return False

    if direction == 0:
        return abs(turn) <= SWING_FIRST_TURN

    return -SWING_BACKTRACK <= direction * turn


def find_swung_grid(
    earlier: GoodView,
    blurred: NDArray[np.float32],
    lines: PaintedLines,
    tracked: NDArray[np.float64] | None = None,
    direction: int = 0,
) -> tuple[NDArray[np.float64], float] | None:
    """
    Look for the grid after a fast camera swing, starting from the last
    good frame, and return it with how far the camera turned. The
    `tracked` grid, already snapped, competes with the turns found.
    Return None when no grid sits on the painted lines, turns the way the
    camera is swinging, and lines up that frame with this one.
    """

    height, width = lines.field_mask.shape[:2]
    shape = lines.field_mask.shape
    spots = np.array(
        [(width / 2, fraction * height) for fraction in SWING_PIVOT_HEIGHTS],
        dtype=np.float32,
    )
    pivots = screen_to_field(spots, earlier.homography).astype(np.float64)

    def score(pivot, degrees):
        candidate = turn_about(earlier.homography, pivot, degrees)
        fit = measure_line_fit(lines.distance_map, lines.field_mask, candidate)
        return fit, candidate

    coarse = []

    for pivot in pivots:
        if not np.isfinite(pivot).all():
            continue

        fits = []

        for degrees in SWING_ANGLES:
            fit, candidate = score(pivot, degrees)
            turn = turn_between(earlier.homography, candidate, shape)
            fits.append(fit if turn_allowed(turn, direction) else np.nan)

        fits = np.array(fits)
        padded = np.concatenate([[np.inf], np.nan_to_num(fits, nan=np.inf), [np.inf]])

        # Keep only the best angle of each dip, so one pivot's neighboring
        # angles do not crowd out the others.
        for position, degrees in enumerate(SWING_ANGLES):
            fit = padded[position + 1]

            if np.isfinite(fit) and fit <= padded[position] and fit <= padded[position + 2]:
                coarse.append((fit, pivot, degrees))

    coarse.sort(key=lambda item: item[0])

    refined = []

    for _, pivot, degrees in coarse[:SWING_SNAP_TRIES]:
        fits = [score(pivot, degrees + step) for step in SWING_FINE_ANGLES]
        fits = [item for item in fits if not np.isnan(item[0])]
        refined.append(min(fits, key=lambda item: item[0]))

    snapped_candidates = [
        snap_to_lines(
            candidate,
            lines.painted,
            lines.distance_map,
            lines.field_mask,
        )
        for _, candidate in refined
    ]

    best = None
    best_match = SWING_MIN_MATCH

    for snapped in [tracked, *snapped_candidates]:
        if snapped is None:
            continue

        turn = turn_between(earlier.homography, snapped, shape)

        if not turn_allowed(turn, direction):
            continue

        error = measure_line_error(
            lines.distance_map,
            lines.field_mask,
            snapped,
        )

        if get_tracking_status(error) != "GOOD":
            continue

        match = match_views(earlier, blurred, lines.field_mask, snapped)

        if match > best_match:
            best, best_match = (snapped, turn), match

    return best


def follow_grid(
    frames: Sequence[NDArray[np.uint8]],
    excluded: NDArray[np.bool_],
    homography: NDArray[np.float64],
) -> GridTrack:
    """
    Track the grid through `frames`, starting from `homography` on the
    first frame.
    """

    count = len(frames)
    homographies = np.zeros((count, 3, 3))
    line_errors = np.full(count, np.nan)
    statuses = np.full(count, "LOST", dtype="<U4")
    snapped = np.zeros(count, dtype=bool)

    previous_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    lines = find_frame_lines(frames[0], excluded)

    # The clicks only pin down the grid near the clicked points, so snap
    # the anchor onto the painted lines as well.
    current_homography = np.asarray(homography, dtype=np.float64)
    snapped_homography = snap_to_lines(
        current_homography,
        lines.painted,
        lines.distance_map,
        lines.field_mask,
    )

    snapped[0] = snapped_homography is not None

    if snapped[0]:
        current_homography = snapped_homography

    # Where to search from if the camera swings out of reach.
    last_good = GoodView(
        current_homography,
        blur_for_matching(lines.painted),
        lines.field_mask,
    )

    # Which way the camera is swinging (-1 or 1), 0 when not swinging.
    swing_direction = 0
    frames_since_search = 0

    screen_points, field_points = detect_new_points(
        previous_gray,
        create_feature_mask(lines.field_mask, []),
        current_homography,
        MAX_FEATURES,
    )

    homographies[0] = current_homography
    line_errors[0] = measure_line_error(
        lines.distance_map,
        lines.field_mask,
        current_homography,
    )
    statuses[0] = get_tracking_status(line_errors[0])

    for index in range(1, count):
        previous_homography = current_homography
        previous_field_mask = lines.field_mask

        current_gray = cv2.cvtColor(frames[index], cv2.COLOR_RGB2GRAY)
        lines = find_frame_lines(frames[index], excluded)

        valid_count = 0
        points_before = len(screen_points)

        if len(screen_points) > 0:
            (
                candidate_screen_points,
                valid_mask,
                _,
            ) = track_screen_points(
                previous_gray,
                current_gray,
                screen_points,
            )

            if candidate_screen_points is not None:
                valid_count = int(valid_mask.sum())

        if valid_count >= 4:
            valid_screen_points = (
                candidate_screen_points[valid_mask]
            ).astype(np.float32)

            valid_field_points = field_points[valid_mask]

            new_homography, inlier_mask = cv2.findHomography(
                valid_field_points,
                valid_screen_points,
                cv2.RANSAC,
                RANSAC_THRESHOLD,
            )

            if new_homography is not None:
                current_homography = new_homography
                inliers = inlier_mask.reshape(-1).astype(bool)

                # Drop points that no longer agree with the field,
                # such as ones dragged along by a player.
                screen_points = valid_screen_points[inliers]
                field_points = valid_field_points[inliers]
            else:
                valid_count = 0

        if valid_count < 4:
            # Keep the last homography and start over with fresh points.
            screen_points = np.empty((0, 2), dtype=np.float32)
            field_points = np.empty((0, 2), dtype=np.float32)

        # The tracked points only predict the grid. Correct it against
        # the painted yard lines so errors cannot build up.
        snapped_homography = snap_to_lines(
            current_homography,
            lines.painted,
            lines.distance_map,
            lines.field_mask,
        )

        if snapped_homography is None:
            # Fast camera moves lose most tracked points, so that
            # prediction can be far off. Predict again from features
            # matched between the two frames, and snap that instead.
            motion = match_frame_motion(
                previous_gray,
                current_gray,
                create_feature_mask(previous_field_mask, []),
                create_feature_mask(lines.field_mask, []),
            )

            if motion is not None:
                snapped_homography = snap_to_lines(
                    motion @ previous_homography,
                    lines.painted,
                    lines.distance_map,
                    lines.field_mask,
                )

        current_blurred = blur_for_matching(lines.painted)

        lost_most_points = valid_count < SWING_POINTS_KEPT * points_before

        if lost_most_points or not fits_last_good(
            snapped_homography,
            last_good,
            current_blurred,
            lines,
        ):
            # Too fast for the tracked points, as in the swing after a
            # turnover: search for the grid by turning the last good one,
            # and keep whichever grid lines up the pictures best.
            swung = find_swung_grid(
                last_good,
                current_blurred,
                lines,
                snapped_homography,
                swing_direction,
            )
            frames_since_search = 0

            # Outside a swing, leave small corrections to the tracked
            # points: the search knows nothing about where along the yard
            # lines the grid sits, and can slide it.
            if swung is not None and (
                swing_direction != 0 or abs(swung[1]) > SWING_MIN_TURN
            ):
                snapped_homography, turn = swung

                if abs(turn) > SWING_MIN_TURN:
                    swing_direction = 1 if turn > 0 else -1
        else:
            frames_since_search += 1

            if frames_since_search >= SWING_SETTLE_FRAMES:
                swing_direction = 0

        snapped[index] = snapped_homography is not None

        if snapped[index]:
            current_homography = snapped_homography

            # Tie the tracked points to the corrected grid, so the next
            # prediction starts from it.
            field_points = screen_to_field(
                screen_points,
                current_homography,
            )

        # Top up with new points as old ones are lost.
        if len(screen_points) < MIN_FEATURES:
            new_screen_points, new_field_points = detect_new_points(
                current_gray,
                create_feature_mask(lines.field_mask, screen_points),
                current_homography,
                MAX_FEATURES - len(screen_points),
            )

            screen_points = np.vstack(
                [screen_points, new_screen_points]
            )

            field_points = np.vstack(
                [field_points, new_field_points]
            )

        # Score the grid against the painted lines, not against the
        # tracked points, which drift along with the homography.
        homographies[index] = current_homography
        line_errors[index] = measure_line_error(
            lines.distance_map,
            lines.field_mask,
            current_homography,
        )
        statuses[index] = get_tracking_status(line_errors[index])

        if statuses[index] == "GOOD":
            last_good = GoodView(
                current_homography,
                current_blurred,
                lines.field_mask,
            )

        previous_gray = current_gray

    return GridTrack(homographies, line_errors, statuses, snapped)


def track_grid(
    frames: Sequence[NDArray[np.uint8]],
    excluded: NDArray[np.bool_],
    anchor_index: int,
    anchor_homography: NDArray[np.float64],
) -> GridTrack:
    """
    Track the grid from the anchor frame to both ends of the clip: forward
    through the frames after it, and backward through the ones before.
    """

    if not 0 <= anchor_index < len(frames):
        raise ValueError("Anchor index is outside the prepared frame sequence.")

    forward = follow_grid(frames[anchor_index:], excluded, anchor_homography)

    if anchor_index == 0:
        return forward

    backward = follow_grid(
        frames[anchor_index::-1], excluded, anchor_homography
    )

    # The backward walk starts on the anchor too; keep the forward copy.
    return GridTrack(
        *(
            np.concatenate([before[:0:-1], after])
            for before, after in (
                (backward.homographies, forward.homographies),
                (backward.line_errors, forward.line_errors),
                (backward.statuses, forward.statuses),
                (backward.snapped, forward.snapped),
            )
        )
    )


def measure_line_errors(
    frames: Sequence[NDArray[np.uint8]],
    excluded: NDArray[np.bool_],
    cameras: Sequence[Sequence[NDArray[np.float64] | None]],
) -> NDArray[np.float64]:
    """
    Return the line error of every frame for each list of homographies
    in `cameras`, one row per list. A None homography scores NaN.
    """

    errors = np.full((len(cameras), len(frames)), np.nan)

    for index, frame in enumerate(frames):
        lines = find_frame_lines(frame, excluded)

        for row, homographies in enumerate(cameras):
            if homographies[index] is not None:
                errors[row, index] = measure_line_error(
                    lines.distance_map,
                    lines.field_mask,
                    homographies[index],
                )

    return errors
