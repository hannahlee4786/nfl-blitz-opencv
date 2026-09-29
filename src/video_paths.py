# Finds input videos and names the files generated for each one
from pathlib import Path


INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def list_input_videos():
    return sorted(
        path for path in INPUT_DIR.iterdir()
        if path.suffix.lower() in VIDEO_EXTENSIONS
    )


def resolve_video(name):
    """
    Accept a video path, or the start of a file name in data/input
    (for example "Clip_1"), so names with spaces need no quoting.
    """

    path = Path(name)

    if path.exists():
        return path

    matches = [
        video for video in list_input_videos()
        if video.name.startswith(name)
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise RuntimeError(f"No video in {INPUT_DIR} matches {name!r}")

    raise RuntimeError(
        f"{name!r} matches several videos: "
        + ", ".join(video.name for video in matches)
    )


def calibration_path(video_path):
    return OUTPUT_DIR / f"{video_path.stem}_homography.npz"
