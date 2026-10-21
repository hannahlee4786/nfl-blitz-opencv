
import cv2
import argparse
from pathlib import Path


def extract_frames(video_path, output_dir, num_frames=100):
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Could not determine video frame count.")

    frame_indices = sorted(
        set(
            round(i * (total_frames - 1) / max(num_frames - 1, 1))
            for i in range(num_frames)
        )
    )

    saved = 0

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()

        if not success:
            continue

        filename = (
            f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
        )

        output_path = output_dir / filename

        cv2.imwrite(str(output_path), frame)
        saved += 1

    cap.release()

    print(f"Extracted {saved} frames from {video_path.name}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--output",
        default="dataset/raw_images"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100
    )

    args = parser.parse_args()

    extract_frames(
        args.video,
        args.output,
        args.frames
    )
