# Collects the initial field-to-screen point matches
import cv2
import numpy as np


# Height of the instruction panel shown above the frame.
PANEL_HEIGHT = 80


def select_screen_points(frame, calibration_points):
    """
    Display a frame and ask the user to click each known field landmark.

    Returns the selected screen coordinates in original-image pixels.
    """

    original_height, original_width = frame.shape[:2]

    # Scale the display down if the video is larger than the screen.
    # Leave room for the instruction panel above the frame.
    display_scale = min(
        1.0,
        1100 / original_width,
        (800 - PANEL_HEIGHT) / original_height,
    )

    display_width = int(original_width * display_scale)
    display_height = int(original_height * display_scale)

    selected_points = []

    window_name = "Select field points"

    def create_display():
        display = cv2.resize(
            frame,
            (display_width, display_height),
        )

        # Redraw all previously selected points.
        for index, original_point in enumerate(selected_points):
            display_x = int(original_point[0] * display_scale)
            display_y = int(original_point[1] * display_scale)

            cv2.circle(
                display,
                (display_x, display_y),
                7,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                display,
                str(index + 1),
                (display_x + 10, display_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        if len(selected_points) < len(calibration_points):
            current = calibration_points[len(selected_points)]

            instruction = (
                f"Click point {len(selected_points) + 1}/"
                f"{len(calibration_points)}: {current['label']} "
                f"{current['coordinate']}"
            )
        else:
            instruction = "All points selected. Press ENTER to continue."

        # Add the panel above the frame instead of drawing it on top,
        # so landmarks near the top edge stay visible and clickable.
        display = cv2.copyMakeBorder(
            display,
            PANEL_HEIGHT,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        cv2.putText(
            display,
            instruction,
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display,
            "U = undo | R = reset | ESC = cancel",
            (15, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        return display

    def mouse_callback(event, display_x, display_y, flags, parameter):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(selected_points) >= len(calibration_points):
            return

        # Ignore clicks on the instruction panel.
        display_y -= PANEL_HEIGHT

        if display_y < 0:
            return

        # Convert the displayed location back into original-frame pixels.
        original_x = display_x / display_scale
        original_y = display_y / display_scale

        selected_points.append(
            (original_x, original_y)
        )

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display = create_display()
        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        # Enter
        if key in (10, 13):
            if len(selected_points) == len(calibration_points):
                break

        # U: undo the previous point
        elif key == ord("u"):
            if selected_points:
                selected_points.pop()

        # R: reset all points
        elif key == ord("r"):
            selected_points.clear()

        # Escape: cancel
        elif key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("Point selection cancelled.")

    cv2.destroyWindow(window_name)

    return np.array(
        selected_points,
        dtype=np.float32,
    )