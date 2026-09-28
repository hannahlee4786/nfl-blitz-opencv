# Runs the complete video-processing pipeline
from pathlib import Path
import cv2

# Importing functions from other files
from src.line_detection import detect_field_lines, filter_field_lines
from src.preprocessing import preprocess_frame
from src.visualization import draw_detected_lines

INPUT_PATH = Path("data/input/gameplay3.mp4")
OUTPUT_PATH = Path("data/output/detected_lines.mp4")

def main():
  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

  capture = cv2.VideoCapture(str(INPUT_PATH))

  if not capture.isOpened():
    raise RuntimeError(f"Could not open video: {INPUT_PATH}")

  width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = capture.get(cv2.CAP_PROP_FPS)

  writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
  )

  if not writer.isOpened():
    raise RuntimeError(f"Could not create video: {OUTPUT_PATH}")

  while True:
    success, frame = capture.read()

    if not success:
      break

    processed, field_mask = preprocess_frame(frame)
    edges, lines = detect_field_lines(processed, field_mask)
    filtered_lines = filter_field_lines(lines, frame.shape)

    output_frame = draw_detected_lines(frame, filtered_lines)
    writer.write(output_frame)

  capture.release()
  writer.release()

  print(f"Saved video to {OUTPUT_PATH}")

if __name__ == "__main__":
  main()