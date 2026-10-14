# Draws the solved camera moving over the field, beside the footage
import cv2
import numpy as np

from src.camera_model import camera_center, camera_matrix, view_rays
from src.field_model import (
    FIELD_LENGTH,
    FIELD_WIDTH,
    LOWER_NUMBER_Y,
    UPPER_NUMBER_Y,
    create_field_grid,
)
from src.visualization import draw_field_grid

# Size of each half of the video.
PANEL_WIDTH = 960
PANEL_HEIGHT = 720

# The overview camera: how wide it sees, how steeply it looks down, and
# how far it swings around to the side of the solved camera's view.
OVERVIEW_FOV = 45.0
OVERVIEW_ELEVATION = 30.0
OVERVIEW_SWING = 50.0

# Rays that never reach the field are drawn out to this many yards.
MAX_VIEW_DISTANCE = 120.0

# Ground drawn around the field, in yards, where views of the stands land.
SURROUND = 15.0

# Fraction of the panel kept clear around the field and the camera path.
FRAME_MARGIN = 0.06

# Colors, in BGR.
BACKGROUND_COLOR = (30, 30, 30)
SURROUND_COLOR = (40, 55, 40)
FIELD_COLOR = (45, 100, 45)
END_ZONE_COLOR = (35, 75, 35)
LINE_COLOR = (210, 210, 210)
CAMERA_COLOR = (0, 210, 255)
FILLED_CAMERA_COLOR = (150, 150, 150)
VIEW_COLOR = (255, 190, 60)
PATH_COLOR = (255, 0, 255)
SHADOW_COLOR = (25, 60, 25)


def look_at(eye, target, up):
    """Return the rotation of a camera at `eye` looking at `target`."""

    forward = target - eye
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    return np.array([right, np.cross(forward, right), forward])


def project(overview, points):
    """Return the panel pixel of each world point, and which are in front."""

    matrix, rotation, eye = overview

    camera_points = (rotation @ (np.asarray(points) - eye).T).T
    in_front = camera_points[:, 2] > 1e-3

    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = (matrix @ camera_points.T).T
        pixels = pixels[:, :2] / pixels[:, 2:3]

    return pixels, in_front


def to_pixel(point):
    return (int(round(point[0])), int(round(point[1])))


def draw_polyline(panel, overview, points, color, thickness):
    pixels, in_front = project(overview, points)

    if in_front.all():
        cv2.polylines(
            panel,
            [np.round(pixels).astype(np.int32)],
            False,
            color,
            thickness,
            cv2.LINE_AA,
        )


def look_at_field(matrix, rotation, translation, picture_box):
    """
    Return where the rays through the picture's corners end, on the field
    or MAX_VIEW_DISTANCE away, and the outline of the field area in view.
    """

    left, top, right, bottom = picture_box
    corners = np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]],
        dtype=np.float64,
    )

    # Walk around the picture's edge to outline what it sees.
    border = np.vstack(
        [
            np.linspace(corners[index], corners[(index + 1) % 4], 20, endpoint=False)
            for index in range(4)
        ]
    )

    center = camera_center(rotation, translation)

    def follow_rays(screen_points):
        rays = view_rays(matrix, rotation, screen_points)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)

        # The field is the plane z = 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            steps = -center[2] / rays[:, 2]

        hits = (steps > 0) & (steps < MAX_VIEW_DISTANCE)
        steps = np.where(hits, steps, MAX_VIEW_DISTANCE)

        return center + steps[:, None] * rays, hits

    corner_ends, _ = follow_rays(corners)
    border_ends, border_hits = follow_rays(border)

    return corner_ends, border_ends[border_hits]


def ground_rectangle(margin):
    """Return the corners of the field, grown by `margin` yards."""

    return np.array(
        [
            [-margin, -margin, 0.0],
            [FIELD_LENGTH + margin, -margin, 0.0],
            [FIELD_LENGTH + margin, FIELD_WIDTH + margin, 0.0],
            [-margin, FIELD_WIDTH + margin, 0.0],
        ]
    )


def create_overview(centers, rotations, up):
    """
    Place an overview camera above, behind, and to one side of the solved
    camera, just far enough back to see the field and the camera's path.
    Returns (matrix, rotation, eye).
    """

    scene = np.vstack([ground_rectangle(0.0), centers])

    target = (scene.min(axis=0) + scene.max(axis=0)) / 2
    target[2] = 0.0

    # The way the solved camera mostly looks, flattened onto the field and
    # swung around the vertical.
    heading = np.mean([rotation[2] for rotation in rotations], axis=0)
    heading -= (heading @ up) * up
    heading /= np.linalg.norm(heading)

    swing = np.radians(OVERVIEW_SWING)
    heading = heading * np.cos(swing) + np.cross(up, heading) * np.sin(swing)

    elevation = np.radians(OVERVIEW_ELEVATION)
    focal = (PANEL_WIDTH / 2) / np.tan(np.radians(OVERVIEW_FOV / 2))
    matrix = camera_matrix(focal, (PANEL_WIDTH / 2, PANEL_HEIGHT / 2))

    low = np.array([PANEL_WIDTH, PANEL_HEIGHT]) * FRAME_MARGIN
    high = np.array([PANEL_WIDTH, PANEL_HEIGHT]) * (1 - FRAME_MARGIN)

    # Back off until the whole scene fits inside the panel.
    distance = 10.0

    while True:
        eye = (
            target
            - distance * np.cos(elevation) * heading
            + distance * np.sin(elevation) * up
        )
        overview = (matrix, look_at(eye, target, up), eye)
        pixels, in_front = project(overview, scene)

        if in_front.all() and (pixels > low).all() and (pixels < high).all():
            return overview

        distance *= 1.05


def draw_field(overview):
    """Draw the field, which stays put, once."""

    panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), BACKGROUND_COLOR, np.uint8)

    def fill(corners, color):
        pixels, in_front = project(overview, corners)

        if in_front.all():
            cv2.fillPoly(panel, [np.round(pixels).astype(np.int32)], color)

    def strip(x_start, x_end):
        return np.array(
            [
                [x_start, 0.0, 0.0],
                [x_end, 0.0, 0.0],
                [x_end, FIELD_WIDTH, 0.0],
                [x_start, FIELD_WIDTH, 0.0],
            ]
        )

    fill(ground_rectangle(SURROUND), SURROUND_COLOR)
    fill(strip(0.0, FIELD_LENGTH), FIELD_COLOR)
    fill(strip(0.0, 10.0), END_ZONE_COLOR)
    fill(strip(FIELD_LENGTH - 10.0, FIELD_LENGTH), END_ZONE_COLOR)

    # Yard lines every 5 yards, the end lines, and the sidelines.
    lines = [((x, 0.0), (x, FIELD_WIDTH)) for x in range(0, 121, 5)]
    lines.append(((0.0, 0.0), (FIELD_LENGTH, 0.0)))
    lines.append(((0.0, FIELD_WIDTH), (FIELD_LENGTH, FIELD_WIDTH)))

    for start, end in lines:
        steps = np.linspace(0.0, 1.0, 30)[:, None]
        points = np.array(start) + steps * (np.array(end) - np.array(start))
        width = 2 if start[0] % 10 == 0 or start[1] == end[1] else 1

        draw_polyline(
            panel,
            overview,
            np.column_stack([points, np.zeros(len(points))]),
            LINE_COLOR,
            width,
        )

    # Yard numbers, to tell the two halves apart.
    for x in range(20, 101, 10):
        label = str(50 - abs(x - 60))

        for y in (UPPER_NUMBER_Y, LOWER_NUMBER_Y):
            pixel, in_front = project(overview, [[x, y, 0.0]])

            if in_front[0]:
                cv2.putText(
                    panel,
                    label,
                    to_pixel(pixel[0] - (10, -6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    LINE_COLOR,
                    1,
                    cv2.LINE_AA,
                )

    return panel


def draw_camera(panel, overview, matrix, rotation, translation, picture_box, size, color):
    """Draw the camera as a small pyramid with a triangle on its top."""

    left, top, right, bottom = picture_box
    center = camera_center(rotation, translation)

    corners = np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]],
        dtype=np.float64,
    )

    middle = (left + right) / 2
    above = top - 0.3 * (bottom - top)
    triangle = np.array(
        [
            [middle - 0.25 * (right - left), top],
            [middle + 0.25 * (right - left), top],
            [middle, above],
        ]
    )

    # Rays through these pixels have depth 1, so scaling gives depth `size`.
    picture = center + size * view_rays(matrix, rotation, corners)
    marker = center + size * view_rays(matrix, rotation, triangle)

    pixels, in_front = project(overview, np.vstack([[center], picture, marker]))

    if not in_front.all():
        return

    apex, picture, marker = pixels[0], pixels[1:5], pixels[5:]

    for corner in picture:
        cv2.line(panel, to_pixel(apex), to_pixel(corner), color, 2, cv2.LINE_AA)

    cv2.polylines(panel, [np.round(picture).astype(np.int32)], True, color, 2, cv2.LINE_AA)
    cv2.fillPoly(panel, [np.round(marker).astype(np.int32)], color)


def draw_label(panel, lines, color=(255, 255, 255)):
    """Write lines of text in a dark box at the top-left of a panel."""

    height = 28 * len(lines) + 12
    width = max(
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
        for line in lines
    ) + 24

    cv2.rectangle(panel, (8, 8), (8 + width, 8 + height), (0, 0, 0), -1)

    for index, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (20, 36 + 28 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )


def render_camera_video(input_path, camera_file, output_path):
    """
    Write a video of the footage with the solved camera's grid on the
    left, and an overview of that camera moving over the field on the
    right.
    """

    # Read every array once; an npz file rereads one on each access.
    camera = dict(np.load(camera_file))

    frame_indices = camera["frame_index"]
    rotations = camera["rotation"]
    translations = camera["translation"]
    focals = camera["focal"]
    sources = camera["source"]
    principal_point = tuple(camera["principal_point"])
    left, right = (int(value) for value in camera["active_area"])
    up = np.array([0.0, 0.0, float(camera["up_sign"])])

    capture = cv2.VideoCapture(str(input_path))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_indices[0]))

    picture_box = (left, 0, right - 1, height - 1)
    matrices = [camera_matrix(focal, principal_point) for focal in focals]

    centers = np.array(
        [
            camera_center(rotation, translation)
            for rotation, translation in zip(rotations, translations)
        ]
    )

    views = [
        look_at_field(matrix, rotation, translation, picture_box)
        for matrix, rotation, translation in zip(matrices, rotations, translations)
    ]

    overview = create_overview(centers, rotations, up)

    field = draw_field(overview)
    camera_size = 0.06 * np.linalg.norm(overview[2] - centers.mean(axis=0))
    shadows = centers - np.outer(centers @ up, up)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (2 * PANEL_WIDTH, PANEL_HEIGHT),
    )

    for index in range(len(frame_indices)):
        success, frame = capture.read()

        if not success:
            break

        rotation, translation = rotations[index], translations[index]
        corner_ends, outline = views[index]
        filled = sources[index] == "filled"
        color = FILLED_CAMERA_COLOR if filled else CAMERA_COLOR

        footage = draw_field_grid(
            frame,
            create_field_grid(),
            camera["homography"][index],
        )
        footage = cv2.resize(footage, (PANEL_WIDTH, PANEL_HEIGHT))
        draw_label(footage, ["Footage with the solved camera's grid"])

        panel = field.copy()

        # What the camera sees, shaded on the ground around the field.
        if len(outline) >= 3:
            _, clipped = cv2.intersectConvexConvex(
                outline[:, :2].astype(np.float32),
                ground_rectangle(SURROUND)[:, :2].astype(np.float32),
            )

        if len(outline) >= 3 and clipped is not None and len(clipped) >= 3:
            clipped = clipped.reshape(-1, 2)
            pixels, in_front = project(
                overview,
                np.column_stack([clipped, np.zeros(len(clipped))]),
            )

            if in_front.all():
                shaded = panel.copy()
                polygon = np.round(pixels).astype(np.int32)
                cv2.fillPoly(shaded, [polygon], VIEW_COLOR)
                panel = cv2.addWeighted(shaded, 0.35, panel, 0.65, 0)
                cv2.polylines(panel, [polygon], True, VIEW_COLOR, 1, cv2.LINE_AA)

        # The path so far, in the air and as a shadow on the field.
        if index > 0:
            draw_polyline(panel, overview, shadows[: index + 1], SHADOW_COLOR, 2)
            draw_polyline(panel, overview, centers[: index + 1], PATH_COLOR, 2)

        # A drop line shows the height.
        draw_polyline(
            panel,
            overview,
            np.array([centers[index], shadows[index]]),
            color,
            1,
        )

        # The rays through the picture's corners.
        for end in corner_ends:
            draw_polyline(
                panel,
                overview,
                np.array([centers[index], end]),
                VIEW_COLOR,
                1,
            )

        draw_camera(
            panel,
            overview,
            matrices[index],
            rotation,
            translation,
            picture_box,
            camera_size,
            color,
        )

        x, y, camera_height = camera["position"][index]

        draw_label(
            panel,
            [
                f"Frame {frame_indices[index]}  |  lens {camera['fov'][index]:.1f} deg  "
                f"|  height {camera_height:.1f} yd",
                f"pan {camera['pan'][index]:+.0f}  tilt {camera['tilt'][index]:.0f}  "
                f"roll {camera['roll'][index]:+.1f} deg  |  {sources[index]}",
            ],
            FILLED_CAMERA_COLOR if filled else (255, 255, 255),
        )

        writer.write(np.hstack([footage, panel]))

    capture.release()
    writer.release()
