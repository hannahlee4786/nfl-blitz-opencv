# Creates grayscale, blurred, and color-masked frames
import cv2
import numpy as np

LOWER_GREEN = np.array([25, 30, 30])
UPPER_GREEN = np.array([100, 255, 255])


def get_hud_rectangles(width, height):
  # Regions covered by the scoreboard, the temporary message, and the
  # bottom HUD, as (top_left, bottom_right) pairs.
  return [
    ((0, 0), (int(width * 0.20), int(height * 0.18))),
    ((int(width * 0.40), 0), (int(width * 0.88), int(height * 0.27))),
    ((0, int(height * 0.78)), (width, height)),
  ]


def create_hud_mask(shape):
  # 255 where the HUD is, 0 elsewhere.
  height, width = shape[:2]
  mask = np.zeros((height, width), np.uint8)

  for top_left, bottom_right in get_hud_rectangles(width, height):
    cv2.rectangle(mask, top_left, bottom_right, 255, -1)

  return mask


def create_green_mask(hsv):
  return cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)


def preprocess_frame(frame):
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  green_mask = create_green_mask(hsv)

  # Painted lines are white, not green. Include them so their edges
  # sit inside the mask instead of on its border.
  lower_white = np.array([0, 0, 150])
  upper_white = np.array([180, 70, 255])

  white_mask = cv2.inRange(hsv, lower_white, upper_white)

  field_mask = cv2.bitwise_or(green_mask, white_mask)

  # Remove the scoreboard, the temporary message, and the bottom HUD.
  field_mask[create_hud_mask(frame.shape) > 0] = 0

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)

  # Return the unmasked frame. Masking here would turn the HUD rectangles
  # into hard edges that Canny detects as fake horizontal lines.
  return blurred, field_mask