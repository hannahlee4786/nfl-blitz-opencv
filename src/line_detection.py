# Runs Canny and the Hough Line Transform
import math

import cv2
import numpy as np


def filter_field_lines(lines, frame_shape):
  if lines is None:
    return None

  height, width = frame_shape[:2]
  filtered = []

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

    # For this clip, the main yard lines are approximately horizontal.
    if not -25 <= angle <= 25:
      continue

    filtered.append([x1, y1, x2, y2])

  if not filtered:
    return None

  return np.asarray(filtered, dtype=np.int32)

def detect_field_lines(processed_frame, field_mask):
  # Detect real image edges before applying the mask.
  edges = cv2.Canny(
    processed_frame,
    threshold1=75,
    threshold2=175,
  )

  # Remove edges outside the allowed field regions.
  # Because this happens after Canny, the mask boundaries do not
  # create new artificial edges.
  masked_edges = cv2.bitwise_and(
    edges,
    edges,
    mask=field_mask,
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