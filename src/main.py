# Runs the complete video-processing pipeline
import cv2

# Importing functions from other files
from src.line_detection import detect_field_lines, filter_field_lines
from src.preprocessing import preprocess_frame
from src.video_paths import INPUT_DIR, OUTPUT_DIR, list_input_videos
from src.visualization import draw_detected_lines

def process_video(input_path, output_path):
  capture = cv2.VideoCapture(str(input_path))

  if not capture.isOpened():
    raise RuntimeError(f"Could not open video: {input_path}")

  width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = capture.get(cv2.CAP_PROP_FPS)

  writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
  )

  if not writer.isOpened():
    raise RuntimeError(f"Could not create video: {output_path}")

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

def main():
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  input_paths = list_input_videos()

  if not input_paths:
    raise RuntimeError(f"No videos found in {INPUT_DIR}")

  for input_path in input_paths:
    output_path = OUTPUT_DIR / f"{input_path.stem}_lines.mp4"

    print(f"Processing {input_path.name}...")
    process_video(input_path, output_path)
    print(f"Saved video to {output_path}")

if __name__ == "__main__":
  main()
