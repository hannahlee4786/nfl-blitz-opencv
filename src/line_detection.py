# Runs Canny and the Hough Line Transform
import math

import cv2
import numpy as np

# Within one frame, yard lines stay within about 12 degrees of each other,
# while sidelines and other markings sit 34+ degrees away (measured on all
# clips in data/input).
ANGLE_TOLERANCE = 15


def angle_difference(a, b):
  # Smallest difference between two line angles, where 0 and 180 are the same.
  return abs((a - b + 90) % 180 - 90)


def filter_field_lines(lines, frame_shape):
  if lines is None:
    return None

  height, width = frame_shape[:2]
  candidates = []

  for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
    dx = x2 - x1
    dy = y2 - y1

    length = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dy, dx))

    # Treat a line pointing left the same as one pointing right.
    if angle > 90:
      angle -= 180
    elif angle < -90:
      angle += 180

    # Ignore HUD regions.
    midpoint_y = (y1 + y2) / 2

    if midpoint_y < height * 0.12:
      continue

    if midpoint_y > height * 0.88:
      continue

    # Ignore short segments.
    if length < 100:
      continue

    candidates.append(([x1, y1, x2, y2], angle))

  if not candidates:
    return None

  # The camera can rotate, so yard lines have no fixed angle on screen.
  # They are the largest group of lines sharing one direction, so pick
  # the angle that the most other lines agree with.
  angles = [angle for _, angle in candidates]

  dominant_angle = max(
    angles,
    key=lambda candidate: sum(
      angle_difference(candidate, angle) <= ANGLE_TOLERANCE
      for angle in angles
    ),
  )

  filtered = [
    line for line, angle in candidates
    if angle_difference(angle, dominant_angle) <= ANGLE_TOLERANCE
  ]

  return np.asarray(filtered, dtype=np.int32)

def detect_field_lines(processed_frame, field_mask):
  # Detect real image edges before applying the mask.
  edges = cv2.Canny(
    processed_frame,
    threshold1=75,
    threshold2=175,
  )

  # Shrink the mask so edges lying on its border (HUD boxes,
  # player outlines) are dropped too.
  inner_mask = cv2.erode(
    field_mask,
    np.ones((5, 5), np.uint8),
  )

  # Remove edges outside the allowed field regions.
  # Because this happens after Canny, the mask boundaries do not
  # create new artificial edges.
  masked_edges = cv2.bitwise_and(
    edges,
    edges,
    mask=inner_mask,
  )

  lines = cv2.HoughLinesP(
    masked_edges,
    rho=1,
    theta=np.pi / 180,
    threshold=80,
    minLineLength=100,
    maxLineGap=25,
  )

  return masked_edges, lines