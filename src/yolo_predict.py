
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def predict_video(model_path, video_path, output_path, conf=0.25):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create output video")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # results = model.predict(frame, conf=conf, verbose=False)

        # results = model.predict(
        #     frame,
        #     conf=conf,
        #     iou=0.45,
        #     verbose=False
        # )

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            iou=0.45,
            verbose=False
        )

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])

                # Thin bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    1,
                )

                # Small text label
                player_id = (
                    int(box.id.item())
                    if box.id is not None
                    else None
                )

                label = (
                    f"ID {player_id} ({confidence:.2f})"
                    if player_id is not None
                    else f"player ({confidence:.2f})"
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        writer.write(frame)

    cap.release()
    writer.release()

    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--output",
        default="runs/yolo_custom/gameplay3_thin.mp4",
    )
    parser.add_argument("--conf", type=float, default=0.25)

    args = parser.parse_args()

    predict_video(
        args.model,
        args.video,
        args.output,
        args.conf,
    )
