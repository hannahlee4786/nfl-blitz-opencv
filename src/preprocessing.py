# Creates grayscale, blurred, and color-masked frames
import cv2
import numpy as np

# Columns that never get brighter than this are black bars, not game.
BAR_BRIGHTNESS = 40

def find_active_area(video_path, sample_count=10):
  """
  Return the first column of the game picture and the column just past
  its end.

  Widescreen captures pad the 4:3 picture with black bars on the left
  and right. Columns that stay dark in every sampled frame are bars.
  """

  capture = cv2.VideoCapture(str(video_path))
  frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
  brightest = None

  for index in np.linspace(0, max(frame_count - 1, 0), sample_count):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    success, frame = capture.read()

    if not success:
      continue

    column_max = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).max(axis=0)

    if brightest is None:
      brightest = column_max
    else:
      brightest = np.maximum(brightest, column_max)

  capture.release()

  if brightest is None:
    raise RuntimeError(f"Could not read video: {video_path}")

  columns = np.flatnonzero(brightest > BAR_BRIGHTNESS)

  if len(columns) == 0:
    return 0, len(brightest)

  return int(columns[0]), int(columns[-1]) + 1

def preprocess_frame(frame, active_area=None):
  height, width = frame.shape[:2]

  # Place the HUD boxes relative to the game picture, which is narrower
  # than the frame when the capture has black bars.
  left, right = active_area if active_area is not None else (0, width)
  picture_width = right - left

  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  lower_green = np.array([25, 30, 30])
  upper_green = np.array([100, 255, 255])

  green_mask = cv2.inRange(hsv, lower_green, upper_green)

  # Painted lines are white, not green. Include them so their edges
  # sit inside the mask instead of on its border.
  lower_white = np.array([0, 0, 150])
  upper_white = np.array([180, 70, 255])

  white_mask = cv2.inRange(hsv, lower_white, upper_white)

  field_mask = cv2.bitwise_or(green_mask, white_mask)

  # Remove the small scoreboard in the upper-left corner.
  cv2.rectangle(
    field_mask,
    (left, 0),
    (left + int(picture_width * 0.20), int(height * 0.18)),
    0,
    -1,
  )

  # Remove the large temporary message near the top-center.
  cv2.rectangle(
    field_mask,
    (left + int(picture_width * 0.40), 0),
    (left + int(picture_width * 0.88), int(height * 0.27)),
    0,
    -1,
  )

  # Remove the bottom HUD.
  cv2.rectangle(
    field_mask,
    (0, int(height * 0.78)),
    (width, height),
    0,
    -1,
  )

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)

  # Return the unmasked frame. Masking here would turn the HUD rectangles
  # into hard edges that Canny detects as fake horizontal lines.
  return blurred, field_mask