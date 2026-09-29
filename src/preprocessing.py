# Creates grayscale, blurred, and color-masked frames
import cv2
import numpy as np

def preprocess_frame(frame):
  height, width = frame.shape[:2]

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
    (0, 0),
    (int(width * 0.20), int(height * 0.18)),
    0,
    -1,
  )

  # Remove the large temporary message near the top-center.
  cv2.rectangle(
    field_mask,
    (int(width * 0.40), 0),
    (int(width * 0.88), int(height * 0.27)),
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