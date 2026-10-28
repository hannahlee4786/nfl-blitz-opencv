"""Click the painted yard numbers once per clip to fix the field coordinates.

    python -m fieldtrack.calibrate data/input/clip.mp4

Writes data/calibration/<clip>.npz and a preview image with the field grid.
Cherry-picked from the player-tracking branch (initialize_homography).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from fieldtrack.click_points import select_screen_points
from fieldtrack.field_model import (
    create_calibration_points,
    create_field_grid,
    get_calibration_field_points,
)
from fieldtrack.overlay import draw_calibration_points, draw_field_grid

CALIBRATION_DIR = Path("data/calibration")


def calibration_path(video_path):
    return CALIBRATION_DIR / f"{Path(video_path).stem}.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame to click on; pick one where the numbers are clear.")
    parser.add_argument("--yard-lines", nargs="+", default=["20", "30", "40"],
                        metavar="YARDS",
                        help="Painted yard numbers to click (10-50). Prefix F or N "
                        "to force the far or near half, e.g. 40 50 F40.")
    parser.add_argument("--swap-sides", action="store_true",
                        help="Camera faces the +x end zone.")
    parser.add_argument("--far-half", action="store_true",
                        help="The yard lines are on the half away from the x = 0 "
                        "(COWBOYS) end zone.")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.video)
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {args.frame} of {args.video}")

    points = create_calibration_points(args.yard_lines, args.far_half, args.swap_sides)
    field_points = get_calibration_field_points(points)
    print("Click these points in order:")
    for i, p in enumerate(points):
        print(f"{i + 1}. {p['label']} -> {p['coordinate']}")
    screen_points = select_screen_points(frame, points)

    # 8 px keeps careful clicks on wide painted numbers and still rejects a wrong one.
    homography, inliers = cv2.findHomography(field_points, screen_points, cv2.RANSAC, 8.0)
    if homography is None:
        raise RuntimeError("Could not compute a homography; check the clicks.")
    projected = cv2.perspectiveTransform(
        field_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
    errors = np.linalg.norm(projected - screen_points, axis=1)
    print(f"RANSAC inliers {int(inliers.sum())}/{len(field_points)}, "
          f"reprojection error mean {errors.mean():.2f}px max {errors.max():.2f}px")

    out = calibration_path(args.video)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, homography=homography, field_points=field_points,
             screen_points=screen_points, inlier_mask=inliers,
             frame_index=args.frame)
    preview = draw_calibration_points(
        draw_field_grid(frame, create_field_grid(), homography), screen_points)
    preview_path = out.with_suffix(".preview.png")
    cv2.imwrite(str(preview_path), preview)
    print(f"Saved {out} and {preview_path}")


if __name__ == "__main__":
    main()
