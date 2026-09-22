# Draws detected lines, points, and the field grid
import cv2
import numpy as np

def draw_detected_lines(frame, lines):
  result = frame.copy()

  if lines is None:
    return result

  # Handles both (N, 1, 4) and (N, 4) Hough-line shapes.
  normalized_lines = np.asarray(lines).reshape(-1, 4)

  for x1, y1, x2, y2 in normalized_lines:
    cv2.line(
      result,
      (int(x1), int(y1)),
      (int(x2), int(y2)),
      (0, 0, 255),
      2,
    )

  return result