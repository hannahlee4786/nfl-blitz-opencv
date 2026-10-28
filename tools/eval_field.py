#!/usr/bin/env python3
"""Score the field fit against painted yard lines.

    .venv/bin/python tools/eval_field.py

For each labeled frame in data/labels/field_lines.json (screen rows of the
painted yard lines at one screen column) this projects the fit's yard lines
to that column and reports how far each labeled line is from the nearest
one, in yards. It scores the bare tracker (what the player-tracking branch
uses) and the refined fit side by side. The labels are drawn by eye, so
differences under about 0.15 yd are noise. Only downfield position is
scored; yard lines do not constrain sideline-to-sideline position.
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fieldtrack.refine import refine  # noqa: E402
from fieldtrack.tracker import track_field  # noqa: E402


def lines_at(H, u):
    rows = []
    for x in range(10, 111, 5):
        ys = np.linspace(-6, 60, 4000)
        pts = np.stack([np.full_like(ys, x), ys], 1).astype(np.float32)
        s = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H.astype(np.float32))[:, 0, :]
        i = int(np.argmin(abs(s[:, 0] - u)))
        if abs(s[i, 0] - u) < 4 and 0 < s[i, 1] < 1100:
            rows.append(s[i, 1])
    return np.array(rows)


def errors(H, u, labels):
    v = lines_at(H, u)
    out = []
    for lab in labels:
        j = int(np.argmin(abs(v - lab)))
        lo, hi = max(j - 1, 0), min(j + 1, len(v) - 1)
        px_per_yd = abs(v[hi] - v[lo]) / ((hi - lo) * 5.0)
        out.append(abs(v[j] - lab) / px_per_yd)
    return out


def summarize(name, e):
    e = np.array(e)
    print(f"{name:26s} n={len(e)} median {np.median(e):.2f} yd  mean {e.mean():.2f}  "
          f"p90 {np.percentile(e, 90):.2f}  max {e.max():.2f}")


def main():
    spec = json.load(open(os.path.join(ROOT, "data/labels/field_lines.json")))
    wanted = {}
    for item in spec["frames"]:
        wanted.setdefault(item["clip"], []).append(item)
    raw_err, fit_err = [], []
    for clip, items in wanted.items():
        video = os.path.join(ROOT, "data/input", clip + ".mp4")
        calib = os.path.join(ROOT, "data/calibration", clip + ".npz")
        if not (os.path.exists(video) and os.path.exists(calib)):
            print(f"skipping {clip}: video or calibration missing")
            continue
        by_frame = {it["frame"]: it for it in items}
        for state in track_field(video, calib):
            it = by_frame.get(state.frame_index)
            if it is None:
                continue
            fit, _, _ = refine(state.homography, state.frame)
            raw_err += errors(state.homography, it["u"], it["v"])
            fit_err += errors(fit, it["u"], it["v"])
            if state.frame_index >= max(by_frame):
                break
    summarize("tracker only (player-tracking)", raw_err)
    summarize("refined (opencv-pipeline)", fit_err)


if __name__ == "__main__":
    main()
