#!/usr/bin/env python3
"""Stage 3: detect players, project feet through homography onto the
reconstructed field map."""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_homography import analyze, hud_mask, PXY


def detect_players(frame):
    """Color-positive per-team masks. Returns list of detections."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hch, sch, vch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    keep = hud_mask(h, w, frame) > 0

    masks = {
        "red": (((hch < 12) | (hch > 168)) & (sch > 90) & (vch > 60)),
        "blue": ((hch > 100) & (hch < 135) & (sch > 90) & (vch > 70)),
        "white": ((sch < 55) & (vch > 150)),
    }
    out = []
    for team, m in masks.items():
        mm = (m & keep).astype(np.uint8) * 255
        if team == "white":
            # kill thin field lines, keep chunky bodies
            mm = cv2.erode(mm, np.ones((5, 5), np.uint8))
            mm = cv2.dilate(mm, np.ones((7, 7), np.uint8))
        # merge jersey/pants/helmet fragments of one body
        mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE,
                              np.ones((13, 13), np.uint8))
        mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8))
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(mm)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 200 or bw > w * 0.3 or bh > h * 0.55:
                continue
            out.append(dict(feet=(x + bw / 2.0, y + float(bh)),
                            team=team, bbox=(x, y, bw, bh),
                            area=int(area)))
    return out


def main(frame_path, anchor_yd=0.0):
    H = analyze(frame_path, frame_path.replace(".png", ""))
    if H is None:
        print("no homography")
        return
    frame = cv2.imread(frame_path)
    dets = detect_players(frame)

    # project feet to world yards
    pts = np.array([[d["feet"]] for d in dets], dtype=np.float32)
    wpts = cv2.perspectiveTransform(pts, H.astype(np.float32))[:, 0, :]

    field_map = cv2.imread("extracted/field_map.png")
    fh, fw = field_map.shape[:2]

    dbg = frame.copy()
    kept = 0
    for d, (xw, yw) in zip(dets, wpts):
        # world-size filter: a player footprint is small; painted logos
        # and markers project wide
        x0, y0, bw, bh = d["bbox"]
        corners = np.array([[[x0, y0 + bh]], [[x0 + bw, y0 + bh]]],
                           dtype=np.float32)
        wc = cv2.perspectiveTransform(corners, H.astype(np.float32))[:, 0, :]
        width_yd = float(np.linalg.norm(wc[1] - wc[0]))
        print(f"  det {d['team']:6s} area={d['area']:6d} "
              f"w_yd={width_yd:6.2f} xw={xw:7.1f} yw={yw:7.1f}")
        if not (0.3 <= width_yd <= 6.0) or not (-3 <= yw <= 45):
            continue
        kept += 1
        color = dict(red=(0, 0, 255), blue=(255, 80, 0),
                     white=(255, 255, 255),
                     other=(0, 255, 255))[d["team"]]
        cv2.rectangle(dbg, (x0, y0), (x0 + bw, y0 + bh), color, 2)
        cv2.circle(dbg, (int(d["feet"][0]), int(d["feet"][1])), 5,
                   color, -1)
        # composite coords: x: endzone offset + anchor + world x
        mx = int((10 + anchor_yd + xw) * PXY) + 0  # 256px EZ = 10yd
        my = int((yw + 2.5) * PXY)
        if 0 <= mx < fw and 0 <= my < fh:
            cv2.circle(field_map, (mx, my), 14, color, -1)
            cv2.circle(field_map, (mx, my), 14, (255, 255, 255), 2)
    print(f"players kept: {kept}/{len(dets)}")
    cv2.imwrite(frame_path.replace(".png", "_players.png"), dbg)
    cv2.imwrite(frame_path.replace(".png", "_map.png"), field_map)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)
