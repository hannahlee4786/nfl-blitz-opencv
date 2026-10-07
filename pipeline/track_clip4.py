#!/usr/bin/env python3
"""Full-clip player tracking for Clip 4 (camera swing + zoom test).

Reuses the Clip 3 pipeline (tools/track_clip.py) unchanged where possible.
Clip 4 specifics:
  * REF_FRAME = 0 (wide shot behind the offense at the 50).
  * Anchor: fresh x and absolute x share direction at frame 0; the orange
    line of scrimmage (absolute 50) lands at fresh x ~ 66 and the red
    first-down line (absolute 21) at fresh x ~ 37.5, so the translation
    snaps to -15 on the 5-yard grid (no axis flip needed at frame 0).
  * Fresh-fit x direction is NOT stable across frames (frame 40 indexes the
    opposite way), so reconcile() additionally tries axis flips.
  * Inter-frame homographies are sanity-gated (scale / rotation /
    translation limits) before being chained; failures hold the previous
    transform (STALE).
  * Flow mask additionally excludes orange overlays (LOS line, flame
    effects, football) that appear late in the clip.
"""
import csv
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import field_homography as fh
from track_clip import (fit_frame, inter_frame_h,
                        TEAM_BGR, STATUS_BGR, LEFT_SCALE, to_map)
from project_players import detect_players

CLIP = "Clip_4 Camera Swing and Distance Change Test.mp4"
REF_FRAME = 0
REF_ANCHOR = -15.0         # fresh x - 15 = absolute yards (verified vs the
                           # orange LOS at the 50 and red 1st-down at the 21)
PXY = fh.PXY
OUT_VIDEO = "data/output/tracked_players.mp4"
OUT_CSV = "data/output/positions.csv"
SAMPLE_AT = {0, 30, 60, 120, 180, 240}
SAMPLE_DIR = "data/output/frames"
HOMOGRAPHY_CACHE = "data/output/homography.npz"
FIELD_MAP = ""
# (x0, x1) of the picture inside black side bars. None until the first frame.
_CONTENT_X = None


def _content_x(frame):
    """The game picture, without the black bars a capture adds on the sides."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cols = gray.mean(axis=0)
    width = int(cols.shape[0])
    x0 = 0
    while x0 < width and cols[x0] < 8:
        x0 += 1
    x1 = width
    while x1 > x0 and cols[x1 - 1] < 8:
        x1 -= 1
    if x0 < 24 and width - x1 < 24:
        return 0, width
    return x0, x1


def _take(cap):
    """Read one frame and drop side bars that are not part of the field."""
    global _CONTENT_X
    ok, frame = cap.read()
    if not ok or frame is None:
        return ok, frame
    if _CONTENT_X is None:
        _CONTENT_X = _content_x(frame)
    x0, x1 = _CONTENT_X
    if x0 != 0 or x1 != frame.shape[1]:
        frame = frame[:, x0:x1]
    return True, frame


# ---------------------------------------------------------------- optical flow
def flow_mask4(frame):
    """Clip 4 flow mask: field prior minus players/HUD minus orange
    overlays (LOS line, flames, football) that move or sit off-plane."""
    h, w = frame.shape[:2]
    keep = fh.hud_mask(h, w, frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hch, sch, vch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    players = (((hch < 12) | (hch > 168)) & (sch > 90)) | \
              ((hch > 100) & (hch < 135) & (sch > 90)) | \
              ((sch < 55) & (vch > 150)) | \
              ((hch >= 8) & (hch <= 30) & (sch > 120) & (vch > 120))
    players = cv2.dilate(players.astype(np.uint8) * 255,
                         np.ones((15, 15), np.uint8))
    return cv2.bitwise_and(keep, cv2.bitwise_not(players))


def g_sane(G, w, h):
    """Reject inter-frame homographies with absurd scale/rotation/shift.
    At 30 fps even the big swing moves only a few deg / few % per frame."""
    if G is None:
        return False
    c = np.array([w / 2.0, h / 2.0])
    pts = np.array([[[c[0], c[1]]], [[c[0] + 100, c[1]]],
                    [[c[0], c[1] + 100]]], np.float32)
    q = cv2.perspectiveTransform(pts, G.astype(np.float32))[:, 0, :]
    dx = (q[1] - q[0]) / 100.0
    dy = (q[2] - q[0]) / 100.0
    sx, sy = np.linalg.norm(dx), np.linalg.norm(dy)
    if not (0.75 < sx < 1.33 and 0.75 < sy < 1.33):
        return False
    rot = np.degrees(np.arctan2(dx[1], dx[0]))
    if abs(rot) > 12.0:
        return False
    if np.linalg.norm(q[0] - c) > 220.0:
        return False
    return True


# ---------------------------------------------------------------- reconcile
FLIPS = (np.eye(3), np.diag([-1.0, 1.0, 1.0]),
         np.diag([1.0, -1.0, 1.0]), np.diag([-1.0, -1.0, 1.0]))


def reconcile4(A_chain, H_fresh, info, shape):
    """Clip 3 reconcile, but tries axis flips: fresh-fit x/y directions
    are arbitrary on Clip 4 (indexing direction flips between frames).
    Returns adjusted absolute H or None."""
    h, w = shape[:2]
    us = np.linspace(0.2 * w, 0.8 * w, 4)
    vs = np.linspace(0.35 * h, 0.75 * h, 3)
    P = np.array([[[u, v]] for u in us for v in vs], np.float32)
    Xc = cv2.perspectiveTransform(P, A_chain.astype(np.float32))[:, 0, :]
    best, best_rms = None, None
    for F in FLIPS:
        Hf = F @ H_fresh
        Xf = cv2.perspectiveTransform(P, Hf.astype(np.float32))[:, 0, :]
        dx = float(np.median(Xc[:, 0] - Xf[:, 0]))
        dy = float(np.median(Xc[:, 1] - Xf[:, 1]))
        dx_snap = round(dx / 5.0) * 5.0
        if abs(dx - dx_snap) > 1.5:
            continue
        if info["sideline"]:
            if abs(dy) >= 5.0:
                continue
            dy_use = 0.0
        else:
            dy_use = dy
        T = np.array([[1, 0, dx_snap], [0, 1, dy_use], [0, 0, 1.0]])
        A_new = T @ Hf
        Xn = cv2.perspectiveTransform(P, A_new.astype(np.float32))[:, 0, :]
        rms = float(np.sqrt(((Xn - Xc) ** 2).sum(axis=1).mean()))
        if rms > 3.0:
            continue
        if best is None or rms < best_rms:
            best, best_rms = A_new, rms
    return best


# ---------------------------------------------------------------- players
def _box_union(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return (x0, y0, x1 - x0, y1 - y0)


def _same_body(jersey, white):
    """White pants/helmet sitting on a colored jersey."""
    jx, jy, jw, jh = jersey
    wx, wy, ww, wh = white
    overlap = min(jx + jw, wx + ww) - max(jx, wx)
    if overlap < 0.35 * min(jw, ww):
        return False
    # pants hang below the jersey; helmets sit just above it
    return (jy - 12) <= (wy + wh) and (wy) <= (jy + jh + 18)


def merge_bodies(dets):
    """One player, one dot. Fold white pants and helmets into the jersey
    they belong to, and drop flat paint (yard digits, HUD strips)."""
    colored = [dict(d) for d in dets if d["team"] != "white"]
    white = [d for d in dets if d["team"] == "white"]
    used = set()
    for c in colored:
        for i, w in enumerate(white):
            if i in used or not _same_body(c["bbox"], w["bbox"]):
                continue
            used.add(i)
            c["bbox"] = _box_union(c["bbox"], w["bbox"])
        x, y, bw, bh = c["bbox"]
        c["feet"] = (x + bw / 2.0, y + float(bh))
    kept = []
    for c in colored:
        x, y, bw, bh = c["bbox"]
        if bh < 18 or bw > bh * 2.2:
            continue
        kept.append(c)
    for i, w in enumerate(white):
        if i in used:
            continue
        x, y, bw, bh = w["bbox"]
        if bh < 22 or bw > bh * 1.15:
            continue
        kept.append(w)
    return _merge_same_team(kept)


def _merge_same_team(dets):
    """Fold split pieces of one jersey (helmet, torso) into one box.
    Neighbors side by side barely overlap horizontally, so they stay apart."""
    dets = [dict(d) for d in dets]
    changed = True
    while changed:
        changed = False
        nxt, used = [], set()
        for i, a in enumerate(dets):
            if i in used:
                continue
            box = a["bbox"]
            for j in range(i + 1, len(dets)):
                if j in used or dets[j]["team"] != a["team"]:
                    continue
                if _same_body(box, dets[j]["bbox"]) or _same_body(dets[j]["bbox"], box):
                    box = _box_union(box, dets[j]["bbox"])
                    used.add(j)
                    changed = True
            x, y, bw, bh = box
            a = dict(a, bbox=box, feet=(x + bw / 2.0, y + float(bh)))
            nxt.append(a)
        dets = nxt
    return dets


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _color_frac(hsv, bbox, team):
    x, y, bw, bh = bbox
    roi = hsv[y:y + bh, x:x + bw]
    if roi.size == 0:
        return 0.0
    hch, sch, vch = roi[..., 0], roi[..., 1], roi[..., 2]
    if team == "blue":
        m = (hch > 100) & (hch < 135) & (sch > 90) & (vch > 70)
    else:
        m = ((hch < 12) | (hch > 168)) & (sch > 90) & (vch > 60)
    return float(np.mean(m))


def _team_masks(hsv):
    hch, sch, vch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((hch < 12) | (hch > 168)) & (sch > 75) & (vch > 50)
    blue = (hch > 100) & (hch < 135) & (sch > 75) & (vch > 55)
    return red, blue


def _mask_components(mask):
    mm = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                          np.ones((5, 5), np.uint8))
    mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mm)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = [int(t) for t in stats[i]]
        if area < 160 or bh < 12 or bw > 240:
            continue
        out.append(dict(bbox=(x, y, bw, bh), area=area))
    return out


def _split_wide(mask, bbox):
    """Two jerseys that touch become one wide blob. A valley in the
    column sum is the gap between them."""
    x, y, bw, bh = bbox
    if bw < max(100, int(bh * 1.8)) or bw < 8:
        return [bbox]
    col = mask[y:y + bh, x:x + bw].sum(axis=0).astype(np.float32)
    if col.size < 24:
        return [bbox]
    smooth = np.convolve(col, np.ones(7) / 7.0, mode="same")
    lo, hi = int(bw * 0.28), int(bw * 0.72)
    if hi <= lo + 4:
        return [bbox]
    valley = lo + int(np.argmin(smooth[lo:hi]))
    left = float(smooth[:valley].max()) if valley > 4 else 0.0
    right = float(smooth[valley:].max()) if valley < bw - 4 else 0.0
    if min(left, right) < 8:
        return [bbox]
    if smooth[valley] > 0.55 * min(left, right):
        return [bbox]
    if valley < 18 or bw - valley < 18:
        return [bbox]
    return [(x, y, valley, bh), (x + valley, y, bw - valley, bh)]


def _iou_over_smaller(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    smaller = min(aw * ah, bw * bh) or 1
    return inter / float(smaller)


def _ring_score(lab, index, stats):
    x, y, bw, bh, area = [int(t) for t in stats[index]]
    if area < 600 or bh < 16 or bw < 28 or bw / float(bh) < 1.25:
        return None
    ys, xs = np.where(lab[y:y + bh, x:x + bw] == index)
    if len(xs) < 40:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    radii = np.hypot(xs - cx, ys - cy)
    med = float(np.median(radii))
    if med < 14:
        return None
    rad = max(4, int(med * 0.45))
    height, width = lab.shape
    yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
    disk = xx * xx + yy * yy <= rad * rad
    y0, x0 = int(y + cy) - rad, int(x + cx) - rad
    sl_y = slice(max(0, y0), min(height, y0 + disk.shape[0]))
    sl_x = slice(max(0, x0), min(width, x0 + disk.shape[1]))
    sub = lab[sl_y, sl_x] == index
    ds = disk[(sl_y.start - y0):(sl_y.stop - y0), (sl_x.start - x0):(sl_x.stop - x0)]
    hole = 1.0 - (float(sub[ds].mean()) if ds.any() else 1.0)
    return dict(cx=x + cx, cy=y + cy, r=med, hole=hole,
                ratio=float(radii.std() / med),
                bbox=(x, y, bw, bh), area=area)


def _find_ring(hsv, hint=None, prev_r=None):
    """Saturated blue ring on the ground under the main player.
    The NFL shield is a filled logo and fails the hole test."""
    hch, sch, vch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    height, width = hch.shape
    mask = ((hch > 98) & (hch < 128) & (sch > 140) & (vch > 100)).astype(np.uint8) * 255
    if hint is not None:
        keep = np.zeros_like(mask)
        cv2.circle(keep, (int(hint[0]), int(hint[1])), 220, 255, -1)
        mask = cv2.bitwise_and(mask, keep)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, count):
        score = _ring_score(lab, i, stats)
        if score is None:
            continue
        if not (0.16 * height < score["cy"] < 0.90 * height):
            continue
        if hint is None:
            if score["hole"] < 0.55 or score["ratio"] > 0.40:
                continue
        else:
            # Feet stand in the ring, so the hole is partly filled.
            # The arrow beside the "1" is a smaller blue blob.
            if score["hole"] < 0.08 or score["ratio"] > 0.55:
                continue
            if prev_r and not (0.35 * prev_r <= score["r"] <= 2.4 * prev_r):
                continue
        if best is None or score["area"] > best["area"]:
            best = score
    if best is None and hint is not None and prev_r:
        best = _ring_at_hint(hsv, hint, prev_r)
    return best


def _ring_at_hint(hsv, hint, prev_r):
    """Feet can cover the hole. Keep the ring when the blue annulus is
    still there and the center is not a solid jersey."""
    hch, sch, vch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    height, width = hch.shape
    cx, cy, rad = float(hint[0]), float(hint[1]), float(prev_r)
    if not (0.16 * height < cy < 0.90 * height):
        return None
    yy, xx = np.ogrid[:height, :width]
    dist = np.hypot(xx - cx, yy - cy)
    blue = (hch > 98) & (hch < 128) & (sch > 130) & (vch > 90)
    annulus = (dist > rad * 0.62) & (dist < rad * 1.2)
    inner = dist < rad * 0.38
    if int(annulus.sum()) < 20:
        return None
    outer = float(blue[annulus].mean())
    hole = float(blue[inner].mean()) if inner.any() else 1.0
    if outer < 0.22 or outer < hole + 0.05:
        return None
    return dict(cx=cx, cy=cy, r=rad, hole=1.0 - hole, ratio=0.3,
                bbox=(int(cx - rad), int(cy - rad * 0.6), int(2 * rad), int(1.2 * rad)),
                area=int(outer * annulus.sum()), hinted=True)


def _arrow_candidates(hsv):
    """The blue downward arrow that travels with the quarterback.

    It is wide at the top and comes to a point, and it stays on screen
    after the ring under his feet is too small to score.
    """
    hch, sch, vch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    height = hch.shape[0]
    mask = ((hch > 100) & (hch < 130) & (sch > 145) & (vch > 70)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    hits = []
    for i in range(1, count):
        x, y, bw, bh, area = [int(t) for t in stats[i]]
        if not (400 <= area <= 4500) or bh < 22 or bw < 28 or bw > 120:
            continue
        cy = y + bh / 2.0
        if not (0.16 * height < cy < 0.88 * height):
            continue
        sub = lab[y:y + bh, x:x + bw] == i
        top = sub[:max(2, bh // 3)]
        bot = sub[-max(2, bh // 3):]
        top_w = float(np.median(top.sum(axis=1)))
        bot_w = float(np.median(bot.sum(axis=1)))
        if top_w < 12:
            continue
        taper = bot_w / max(top_w, 1.0)
        if taper > 0.58:
            continue
        hits.append(dict(cx=x + bw / 2.0, cy=cy, bh=float(bh),
                         area=area, taper=taper))
    return hits


def _flow_arrow(prev_gray, gray, arrow):
    """Carry the arrow a frame when the taper test misses."""
    cx, cy, bh = arrow
    pts = [[cx + fx * bh, cy + fy * bh]
           for fx in np.linspace(-0.3, 0.3, 5)
           for fy in np.linspace(-0.35, 0.15, 4)]
    pts = np.array(pts, np.float32).reshape(-1, 1, 2)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, pts, None, winSize=(31, 31), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    if nxt is None:
        return None
    back, st2, _ = cv2.calcOpticalFlowPyrLK(
        gray, prev_gray, nxt, None, winSize=(31, 31), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    if back is None:
        return None
    fb = np.linalg.norm(pts.reshape(-1, 2) - back.reshape(-1, 2), axis=1)
    ok = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 3.5)
    if int(ok.sum()) < 4:
        return None
    shift = np.median(nxt.reshape(-1, 2)[ok] - pts.reshape(-1, 2)[ok], axis=0)
    if float(np.hypot(shift[0], shift[1])) > 60:
        return None
    return (float(cx + shift[0]), float(cy + shift[1]), bh)


def _marked_arrow(hsv, hits, shape):
    """The single chevron that has the '1' plaque above it.

    Painted letters taper the same way. They do not have the plaque, so a
    wider taper is accepted only when that digit is there and it is the
    only one.
    """
    height = shape[0]
    marked = []
    for hit in hits:
        if hit["area"] < 900 or hit["taper"] > 0.55:
            continue
        if not (0.16 * height < hit["cy"] < 0.55 * height):
            continue
        arrow = (hit["cx"], hit["cy"], hit["bh"])
        if _arrow_has_digit(hsv, arrow):
            marked.append(hit)
    if len(marked) != 1:
        return None
    return marked[0]


def _pick_arrow(hits, prev, ring, shape, unsnap):
    """Prefer the arrow standing over the ring. Otherwise the one that
    continues the previous arrow, and after a gap the single large one."""
    height = shape[0]
    if ring is not None:
        above = [h for h in hits
                 if abs(h["cx"] - ring["cx"]) < 80
                 and 15 < ring["cy"] - h["cy"] < 170]
        if above:
            return max(above, key=lambda h: h["area"])
    base = prev
    if prev is not None and unsnap < 10:
        near = [h for h in hits if np.hypot(h["cx"] - base[0], h["cy"] - base[1]) < 80]
        if near:
            return max(near, key=lambda h: h["area"])
    if unsnap >= 10:
        big = [h for h in hits
               if h["area"] >= 900 and h["taper"] <= 0.45
               and 0.16 * height < h["cy"] < 0.55 * height]
        if big:
            return max(big, key=lambda h: h["area"])
    return None


def _find_marker(hsv, ring):
    """The black '1' plaque beside the ring. It is not a player."""
    if ring is None:
        return None
    sch, vch = hsv[..., 1], hsv[..., 2]
    dark = ((vch < 60) & (sch < 90)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    white = (sch < 60) & (vch > 180)
    count, lab, stats, _ = cv2.connectedComponentsWithStats(dark)
    best = None
    for i in range(1, count):
        x, y, bw, bh, area = [int(t) for t in stats[i]]
        if not (250 < area < 6000) or bh < 16 or bw < 16:
            continue
        if bw / float(bh) > 2.2 or bh / float(bw) > 2.2:
            continue
        inside = lab[y:y + bh, x:x + bw] == i
        if inside.sum() == 0 or float(white[y:y + bh, x:x + bw].mean()) < 0.04:
            continue
        cx, cy = x + bw / 2.0, y + bh / 2.0
        if np.hypot(cx - ring["cx"], cy - ring["cy"]) > 190:
            continue
        if best is None or area > best[0]:
            best = (area, (x, y, bw, bh))
    return None if best is None else best[1]


def _flow_box(prev_gray, gray, bbox, field, jersey):
    """Follow the jersey inside a box.

    Grass and the white yard paint move with the camera. A box that tracks
    those points slides off the player. If the jersey is not in the box,
    return nothing and let the body search place it again.
    """
    x, y, bw, bh = bbox
    height, width = prev_gray.shape[:2]
    if bw < 8 or bh < 8:
        return None
    if jersey is None:
        return None
    x0, x1 = max(0, int(x)), min(width, int(x + bw))
    y0, y1 = max(0, int(y)), min(height, int(y + bh * 0.85))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    ys, xs = np.nonzero(jersey[y0:y1, x0:x1])
    if xs.size < 24:
        return None
    take = np.linspace(0, xs.size - 1, min(24, xs.size)).astype(int)
    pts = np.stack([xs[take] + x0, ys[take] + y0], axis=1).astype(np.float32)
    pts = pts.reshape(-1, 1, 2)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    if nxt is None:
        return None
    back, st2, _ = cv2.calcOpticalFlowPyrLK(
        gray, prev_gray, nxt, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    if back is None:
        return None
    fb = np.linalg.norm(pts.reshape(-1, 2) - back.reshape(-1, 2), axis=1)
    ok = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 2.0)
    if int(ok.sum()) < 4:
        return None
    shift = np.median(nxt.reshape(-1, 2)[ok] - pts.reshape(-1, 2)[ok], axis=0)
    if float(np.hypot(shift[0], shift[1])) > 80:
        return None
    return float(shift[0]), float(shift[1])


def _fit_jersey(jersey, bbox, shape, search=8):
    """Shrink a box onto the jersey inside it.

    The search stays inside the box. A logo or the down marker outside
    the box cannot pull the box sideways.
    """
    if jersey is None:
        return None
    x, y, bw, bh = bbox
    height, width = shape[:2]
    x0 = max(0, int(x) + 2)
    x1 = min(width, int(x + bw) - 2)
    y0 = max(0, int(y) - search)
    y1 = min(height, int(y + bh * 0.72))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    ys, xs = np.nonzero(jersey[y0:y1, x0:x1])
    if ys.size < 70:
        return None
    px = xs.astype(np.int32) + x0
    py = ys.astype(np.int32) + y0
    cx = x + bw / 2.0
    near = np.abs(px - cx) < max(bw * 0.6, 30)
    if int(near.sum()) < 50:
        return None
    px, py = px[near], py[near]

    def _core(coords, gap):
        order = np.argsort(coords)
        c = coords[order]
        splits = np.where(np.diff(c) > gap)[0]
        starts = np.r_[0, splits + 1]
        ends = np.r_[splits + 1, c.size]
        best = max(range(len(starts)), key=lambda i: ends[i] - starts[i])
        keep = order[starts[best]:ends[best]]
        return coords[keep], keep

    py, keep = _core(py, 16)
    px = px[keep]
    if px.size < 40:
        return None
    px, keep = _core(px, 18)
    py = py[keep]
    if px.size < 40:
        return None
    jx0, jx1 = int(px.min()), int(px.max())
    jy0, jy1 = int(py.min()), int(py.max())
    jw, jh = jx1 - jx0 + 1, jy1 - jy0 + 1
    if jh < 14 or jw < 10 or jw > 160 or jy0 < height * 0.12:
        return None
    up = int(0.42 * jh)
    down = int(0.90 * jh)
    half = int(max(jw * 0.62, 16))
    nx = int(np.clip((jx0 + jx1) / 2.0 - half, -20, width - 8))
    nw = int(min(width + 20 - nx, half * 2))
    ny = int(np.clip(jy0 - up, -20, height - 8))
    nh = int(min(height * 0.90 - ny, (jy1 + down) - ny))
    if nw < 28 or nh < 34 or nw > 115 or nh > 175:
        return None
    if nw > bw * 1.15 or nh > bh * 1.2:
        return None
    cx = x + bw / 2.0
    if not (nx - 8 <= cx <= nx + nw + 8):
        return None
    return (nx, ny, nw, max(34, nh))


def _jersey_nudge(jersey, bbox):
    """Slide the box sideways onto the jersey that is still inside it."""
    if jersey is None:
        return 0.0
    x, y, bw, bh = bbox
    height, width = jersey.shape[:2]
    x0, x1 = max(0, int(x)), min(width, int(x + bw))
    y0, y1 = max(0, int(y)), min(height, int(y + bh * 0.75))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return 0.0
    xs = np.nonzero(jersey[y0:y1, x0:x1])[1]
    if xs.size < 40:
        return 0.0
    cx = float(xs.mean()) + x0
    du = cx - (x + bw / 2.0)
    if abs(du) < 6:
        return 0.0
    return float(np.clip(du, -16, 16))


def _grow_edge(hsv, mask, x, y, w, h, shape):
    """Extend a box cut by the frame edge until a solid shoe run is inside.

    A column counts when the jersey or the shoe forms a vertical run of at
    least eight pixels. A yard line is only a few pixels tall in each
    column, so it does not pull the box sideways.
    """
    height, width = shape[:2]
    x, y, w, h = int(x), int(y), int(w), int(h)
    right = x + w >= width - 8
    left = x <= 2
    if not right and not left:
        return x, y, w, h
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    field = (hh > 70) & (hh < 110) & (ss > 110)
    shoe = (ss < 90) & (vv > 90) & ~field
    keep = (mask > 0) | shoe
    y0 = max(0, y)
    y1 = min(height, y + h + 48)

    def solid(col):
        best = cur = 0
        for on in keep[y0:y1, col]:
            if on:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        return best >= 8

    if right:
        nx = x
        gap = 0
        for col in range(x - 1, max(0, x - 56), -1):
            if solid(col):
                nx = col
                gap = 0
            else:
                gap += 1
                if gap > 3:
                    break
        x0, x1 = nx, min(width, x + w)
    else:
        nx = min(width - 1, x + w - 1)
        gap = 0
        for col in range(x + w, min(width, x + w + 56)):
            if solid(col):
                nx = col
                gap = 0
            else:
                gap += 1
                if gap > 3:
                    break
        x0, x1 = x, nx + 1
    bot = min(height - 1, y + h)
    gap = 0
    for row in range(y + h, min(height - 1, y + h + 56)):
        if int(keep[row, x0:x1].sum()) >= 4:
            bot = row
            gap = 0
        else:
            gap += 1
            if gap > 5:
                break
    return x0, y, max(1, x1 - x0), max(1, bot - y + 1)


def _jersey_bodies(mask, shape, hsv=None):
    """Player-shaped jerseys on this frame, with a box down to the feet."""
    height, width = shape[:2]
    bodies = []
    for comp in _merge_vertical(_mask_components(mask), hsv=hsv):
        x, y, bw, bh = comp["bbox"]
        if comp["area"] < 240 or bh < 18 or bw < 16 or bw > 150:
            continue
        # The BLITZ letters share the blue mask. Their pixels are a lighter
        # hue than a jersey. A player standing on the letters still has
        # jersey pixels in the seed, so that seed stays. A far jersey is
        # small and high, and the same light blue shows up on it.
        far = y < height * 0.16 and bw < 45 and bh < 40
        if hsv is not None and not far:
            sub = mask[y:y + bh, x:x + bw]
            if int(sub.sum()) > 40:
                patch = hsv[y:y + bh, x:x + bw]
                hh, ss, vv = patch[..., 0], patch[..., 1], patch[..., 2]
                paint = ((hh > 96) & (hh < 112) & (ss > 140)
                         & (vv > 70) & (vv < 150) & sub)
                jersey = ((hh >= 114) & (hh < 130) & (ss > 80)
                          & (vv > 55) & sub)
                if int(paint.sum()) >= 80 and int(paint.sum()) > int(jersey.sum()) * 3:
                    continue
        # Far players rise above the usual top line once the camera zooms.
        # Anything above that line has to be a small jersey. The down marker
        # is taller, and the scoreboard is the top-left corner below.
        if y < height * 0.05:
            continue
        if y < height * 0.10 and not (bh < 40 and bw < 45):
            continue
        if y > height * 0.86:
            continue
        if y < height * 0.16 and x < width * 0.18:
            continue
        # The KAG logo sits in the bottom-right corner. Players do not.
        if x + bw / 2.0 > width * 0.75 and y > height * 0.68:
            continue
        if bw > bh * 2.2 and bh < 48:
            continue
        # A sideline pylon is a narrow fringe about fifty pixels tall.
        # A far player is shorter, and a player cut off by the frame is taller.
        if bw < 26 and 36 <= bh < 70 and x > 8 and x + bw < width - 8:
            continue
        if hsv is not None and bh >= 24:
            sub = mask[y:y + bh, x:x + bw]
            widths = sub.sum(axis=1).astype(np.float32)
            if widths.size >= 8 and float(widths.std()) > 1.0:
                corr = float(np.corrcoef(np.arange(widths.size), widths)[0, 1])
                patch = hsv[y:y + bh, x:x + bw]
                white = float(((patch[..., 1] < 70) & (patch[..., 2] > 150)).mean())
                # The red 20 marker is a triangle that widens downward
                # around a white digit. Jerseys do not do both.
                if corr > 0.72 and white > 0.07:
                    continue
            if y + bh < height * 0.84:
                y1 = min(height - 1, y + bh + int(max(16, bh * 0.8)))
                if y1 > y + bh:
                    sch, vch = hsv[..., 1], hsv[..., 2]
                    leg = float(((sch[y + bh:y1, x:x + bw] < 105)
                                 & (vch[y + bh:y1, x:x + bw] > 48)).mean())
                    if leg < 0.15:
                        continue
        if hsv is not None:
            sub = mask[y:y + bh, x:x + bw]
            if int(sub.sum()) > 30:
                vv = hsv[y:y + bh, x:x + bw, 2][sub]
                ss = hsv[y:y + bh, x:x + bw, 1][sub]
                if float(np.median(vv)) > 155 and float(np.median(ss)) > 185:
                    continue
        if hsv is not None and comp["area"] < 500 and bh < 48 and not far:
            y1b = min(height - 1, y + bh)
            y2b = min(height - 1, y1b + 16)
            if y2b > y1b:
                hch = hsv[..., 0]
                grass = ((hch[y1b:y2b, x:x + bw] > 65)
                         & (hch[y1b:y2b, x:x + bw] < 108)
                         & (hsv[y1b:y2b, x:x + bw, 1] > 70)
                         & (hsv[y1b:y2b, x:x + bw, 2] > 40))
                if float(grass.mean()) > 0.70:
                    continue
        down = 16 if bh >= 80 else int(0.45 * bh)
        up = min(48, int(0.70 * bh))
        half = int(max(bw * 0.55, 14))
        cx = x + bw / 2.0
        nx = int(round(cx - half))
        ny = int(round(y - up))
        nw = int(half * 2)
        feet = y + bh + down
        if hsv is not None:
            feet = max(feet, _pants_bottom(hsv, mask, x, y, bw, bh, shape))
            top_p, bot_p = _person_span(hsv, x, y, bw, bh, shape)
            ny = min(ny, top_p)
            feet = max(feet, bot_p)
            top_h, bot_h = _cover_head_feet(hsv, mask, nx, ny, nw, feet, shape)
            ny = min(ny, top_h)
            feet = max(feet, bot_h)
        nh = int(feet - ny)
        if nh > 220:
            nh = 220
        if ny + nh > height * 0.96:
            nh = int(height * 0.96 - ny)
        if hsv is not None and (nx <= 2 or nx + nw >= width - 8):
            nx, ny, nw, nh = _grow_edge(hsv, mask, nx, ny, nw, nh, shape)
            if nh > 220:
                nh = 220
            if ny + nh > height * 0.96:
                nh = int(height * 0.96 - ny)
        if nw < 24 or nh < 30:
            continue
        # The BLITZ letters and the NFL shield are flat blue paint. A box
        # on them has that paint and no jersey.
        if hsv is not None and (_on_field_logo(hsv, (nx, ny, nw, nh))
                                or _on_endzone_letters(hsv, (nx, ny, nw, nh))):
            continue
        bodies.append((nx, ny, nw, nh, cx, float(ny + nh)))
    return bodies


def _row_mid(arr):
    """Middle value of a short row, without the cost of np.median."""
    flat = np.asarray(arr).ravel()
    n = int(flat.size)
    if n == 0:
        return 0.0
    k = n // 2
    return float(np.partition(flat, k)[k])


def _on_grass(frame):
    """A gameplay picture is mostly green. The closing logo card is not."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = (hh > 65) & (hh < 110) & (ss > 55) & (vv > 40)
    return float(grass.mean()) >= 0.30


def _mostly_grass(hsv, box):
    """True when a box is sitting on empty field rather than a player."""
    height, width = hsv.shape[:2]
    x, y, w, h = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    x0, x1 = max(0, x), min(width, x + max(w, 1))
    y0, y1 = max(0, y), min(height, y + max(h, 1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return True
    patch = hsv[y0:y1, x0:x1]
    hh, ss, vv = patch[..., 0], patch[..., 1], patch[..., 2]
    field = (hh > 65) & (hh < 110) & (ss > 55) & (vv > 40)
    return float(field.mean()) > 0.82


def _on_ground_circle(hsv, box):
    """True when a box sits on the painted blue ring in empty grass.

    That ring is saturated blue, so it is not grass and the blue mask
    counts it as a jersey. A player standing in the ring still has more
    jersey than ring paint, and his box overlaps a body.
    """
    height, width = hsv.shape[:2]
    x, y, w, h = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    x0, x1 = max(0, x), min(width, x + max(w, 1))
    y0, y1 = max(0, y), min(height, y + max(h, 1))
    if x1 - x0 < 4 or y1 - y0 < 8:
        return False
    patch = hsv[y0:y1, x0:x1]
    hh, ss, vv = patch[..., 0], patch[..., 1], patch[..., 2]
    field = float(((hh > 65) & (hh < 110) & (ss > 55) & (vv > 40)).mean())
    jersey = float(((hh >= 114) & (hh < 130) & (ss > 80) & (vv > 55)).mean())
    ring = float(((hh > 98) & (hh < 128) & (ss > 140) & (vv > 100)).mean())
    return field > 0.72 and ring > 0.18 and ring > jersey


def _slide_onto_white(hsv, box, opp):
    """Step a tall empty box sideways onto a white jersey.

    The shiny jersey has left the team mask, so the box is on the grass
    beside the player. A step onto the other team is someone else. White
    in the top half is the jersey. White only along the bottom is a yard
    number.
    """
    height, width = hsv.shape[:2]
    x, y, w, h = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    if h < 80 or w * h <= 1500:
        return None

    def _top_white(bb):
        x0, x1 = max(0, int(bb[0])), min(width, int(bb[0] + bb[2]))
        y0 = max(0, int(bb[1]))
        y1 = min(height, int(bb[1] + bb[3] * 0.55))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        patch = hsv[y0:y1, x0:x1]
        return float(((patch[..., 1] < 80) & (patch[..., 2] > 140)).mean())

    best, best_w = None, 0.32
    for dx in range(-80, 81, 4):
        cand = (x + dx, y, w, h)
        if cand[0] < -8 or cand[0] + w > width + 8:
            continue
        if _mostly_grass(hsv, cand) or _on_field_logo(hsv, cand):
            continue
        if opp is not None and _jersey_count(opp, cand) >= 80:
            continue
        top = _top_white(cand)
        if top > best_w:
            best_w = top
            best = cand
    return best


def _lock_white(hsv, opp, origin, radius, shape):
    """A short white jersey near origin, with almost none of the other team.

    The search stays on the torso. A tall box beside a tackle counts the
    other jersey and has to be rejected. The whitest nearby torso wins,
    and a long step loses to a closer one.
    """
    height, width = shape[:2]
    ox, oy, bw, bh = origin
    bw = int(max(24, min(int(bw), 48)))
    bh = int(max(48, min(int(bh), 76)))
    best, best_s = None, -1.0
    for dy in range(-radius, radius + 1, 4):
        for dx in range(-radius, radius + 1, 4):
            x, y = int(ox + dx), int(oy + dy)
            if (x < 4 or y < int(height * 0.05)
                    or x + bw >= width - 4 or y + bh >= height - 4):
                continue
            cand = (x, y, bw, bh)
            if _mostly_grass(hsv, cand) or _on_field_logo(hsv, cand):
                continue
            y1 = min(height, y + int(bh * 0.55))
            patch = hsv[y:y1, x:x + bw]
            if patch.size == 0:
                continue
            white = float(((patch[..., 1] < 80) & (patch[..., 2] > 150)).mean())
            if white < 0.45:
                continue
            if opp is not None and _jersey_count(opp, cand) >= 40:
                continue
            score = white - float(np.hypot(dx, dy)) / 400.0
            if score > best_s:
                best_s = score
                best = cand
    return best


def _on_field_logo(hsv, box):
    """True when a box sits on the painted BLITZ letters or the NFL shield.

    That paint is a lighter blue than a jersey, and the box has no jersey
    pixels in it. A player standing on the logo still has jersey pixels.
    """
    height, width = hsv.shape[:2]
    x, y, w, h = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    x0, x1 = max(0, x), min(width, x + max(w, 1))
    y0, y1 = max(0, y), min(height, y + max(h, 1))
    if x1 - x0 < 4 or y1 - y0 < 8:
        return False
    patch = hsv[y0:y1, x0:x1]
    hh, ss, vv = patch[..., 0], patch[..., 1], patch[..., 2]
    logo = (hh > 96) & (hh < 112) & (ss > 140) & (vv > 70) & (vv < 150)
    jersey = (hh >= 114) & (hh < 130) & (ss > 80) & (vv > 55)
    logo_m = float(logo.mean())
    jersey_m = float(jersey.mean())
    if logo_m >= 0.18 and jersey_m < 0.08:
        return True
    # The letters often sit in the lower half. The rest of the box is grass,
    # so the full-box fraction stays under 0.18 and the box was kept. A
    # player standing on the letters still has a jersey in the box.
    if jersey_m < 0.03 and logo_m >= 0.12 and patch.shape[0] >= 16:
        field = (hh > 65) & (hh < 110) & (ss > 55) & (vv > 40)
        low = logo[patch.shape[0] // 2:]
        if float(field.mean()) > 0.85 and float(low.mean()) >= 0.25:
            return True
    return False


def _on_endzone_letters(hsv, box):
    """The team name painted across the end zone.

    That paint is a gray band much wider than a player. A jersey in the
    same row breaks the band, so a player standing there still counts.
    """
    height, width = hsv.shape[:2]
    x, y, w, h = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    if h < 20 or w < 8:
        return False
    row = int(np.clip(y + h * 0.45, 0, height - 1))
    cx = int(np.clip(x + w / 2.0, 0, width - 1))
    hue, sat, val = hsv[row, :, 0], hsv[row, :, 1], hsv[row, :, 2]

    def paint(col):
        grass = (65 < hue[col] < 105) and sat[col] > 90 and val[col] > 50
        return (not grass) and sat[col] < 140 and 25 < val[col] < 190

    if not paint(cx):
        return False
    left = cx
    while left > 0 and paint(left - 1):
        left -= 1
    right = cx
    while right < width - 1 and paint(right + 1):
        right += 1
    return (right - left) > 280


def _on_white_sideline(hsv, box):
    """A tall box on the painted sideline, not a white jersey.

    The sideline is a white band wider than the box. A jersey is only
    as wide as the player.
    """
    height, width = hsv.shape[:2]
    x, y, w, h = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    if h < 70 or w < 8:
        return False
    x0, x1 = max(0, x), min(width, x + max(w, 1))
    y0, y1 = max(0, y), min(height, y + max(h, 1))
    if x1 <= x0 or y1 <= y0:
        return False
    patch = hsv[y0:y1, x0:x1]
    white = float(((patch[..., 1] < 90) & (patch[..., 2] > 150)).mean())
    if white < 0.55:
        return False
    row = int(np.clip(y + h * 0.45, 0, height - 1))
    cx = int(np.clip(x + w / 2.0, 0, width - 1))
    sat, val = hsv[row, :, 1], hsv[row, :, 2]

    def is_white(col):
        return sat[col] < 90 and val[col] > 150

    if not is_white(cx):
        return False
    left = cx
    while left > 0 and is_white(left - 1):
        left -= 1
    right = cx
    while right < width - 1 and is_white(right + 1):
        right += 1
    return (right - left) > w + 40


def _cover_head_feet(hsv, mask, x, y, bw, foot, shape):
    """Pull the box up over a silver helmet and down over light shoes.

    A bright yard line is wider than the player, and the orange cone is
    a saturated red, so neither one moves the box.
    """
    height, width = shape[:2]
    x0, x1 = max(0, int(x)), min(width, int(x + bw))
    if x1 <= x0:
        return y, y
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    y0, y1 = max(0, int(y)), min(height, int(y) + 80)
    sub = mask[y0:y1, x0:x1]
    hue = 0.0
    if int(sub.sum()) > 12:
        hue = float(np.median(H[y0:y1, x0:x1][sub]))
    red_team = hue < 20 or hue > 160

    def kind(row):
        meds = _row_mid(S[row, x0:x1])
        medv = _row_mid(V[row, x0:x1])
        medh = _row_mid(H[row, x0:x1])
        if meds > 175 and medv > 140:
            return "stop"
        cone = float(((H[row, x0:x1] < 18) & (S[row, x0:x1] > 170) & (V[row, x0:x1] > 150)).mean())
        if cone > 0.12:
            return "stop"
        teamish = int(mask[row, x0:x1].sum()) >= 4
        other = (100 < medh < 135) if red_team else (medh < 15 or medh > 165)
        if other and meds > 35 and not teamish:
            return "stop"
        helmet = medv > 125 and meds < 100
        if helmet:
            wx0, wx1 = max(0, x0 - bw), min(width, x1 + bw)
            wide = float(((V[row, wx0:wx1] > 125) & (S[row, wx0:wx1] < 110)).mean())
            if wide > 0.55:
                helmet = False
        if teamish or helmet:
            return "keep"
        return "gap"

    top = int(y)
    for row in range(int(y) - 1, max(0, int(y) - 46), -1):
        got = kind(row)
        if got == "stop":
            break
        if got == "keep" and int(y) - row <= 28:
            top = row
    base = top
    for row in range(base - 1, max(0, base - 24), -1):
        if _row_mid(V[row, x0:x1]) <= 140 or _row_mid(S[row, x0:x1]) >= 160:
            continue
        arrow = float(((H[row, x0:x1] > 100) & (H[row, x0:x1] < 130) & (S[row, x0:x1] > 170)).mean())
        cone = float(((H[row, x0:x1] < 18) & (S[row, x0:x1] > 150) & (V[row, x0:x1] > 140)).mean())
        if arrow < 0.35 and cone < 0.08:
            top = min(top, row)
    bot = int(foot)
    if int(foot) - int(y) > 165:
        return top, bot
    gap = 0
    limit = min(height - 1, int(foot) + 40)
    for row in range(int(foot), limit):
        shoe = (S[row, x0:x1] < 100) & (V[row, x0:x1] > 85)
        n = int(shoe.sum())
        if n >= 6 and float(np.median(H[row, x0:x1][shoe])) < 103:
            bot = row
            gap = 0
        else:
            gap += 1
            if gap > 6:
                break
    return top, bot


def _person_span(hsv, x, y, bw, bh, shape):
    """Helmet and shoes are often outside the team-color mask."""
    height, width = shape[:2]
    x0, x1 = max(0, int(x)), min(width, int(x + bw))
    if x1 - x0 < 6:
        return y, y + bh
    hch, sch, vch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    field = (hch > 65) & (hch < 110) & (sch > 55) & (vch > 40)
    person = ~field & (vch > 42)
    top = y
    gap = 0
    for row in range(int(y) - 1, max(0, int(y) - 56), -1):
        if float(person[row, x0:x1].mean()) > 0.22:
            top = row
            gap = 0
        else:
            gap += 1
            if gap > 6:
                break
    bot = y + bh
    gap = 0
    for row in range(int(y + bh), min(height - 1, int(y + bh) + 72)):
        if float(np.median(vch[row, x0:x1])) < 42:
            break
        if float(person[row, x0:x1].mean()) > 0.18:
            bot = row
            gap = 0
        else:
            gap += 1
            if gap > 8:
                break
    return top, bot


def _pants_bottom(hsv, mask, x, y, bw, bh, shape):
    """Extend the box to the lowest same-team pixels under the jersey.

    A short gap is allowed where the other player covers the legs. Bright
    orange (the sideline cone) does not count as pants.
    """
    height, width = shape[:2]
    x0 = max(0, int(x) - 6)
    x1 = min(width, int(x + bw) + 18)
    if x1 - x0 < 6:
        return y + bh
    yb = min(height - 2, int(y + bh))
    limit = min(int(height * 0.92), yb + 68)
    sch, vch = hsv[..., 1], hsv[..., 2]
    last = yb
    gap = 0
    allow = min(52, max(28, int(1.3 * bh)))
    for row in range(yb, limit):
        band = mask[row, x0:x1]
        bright = (vch[row, x0:x1] > 155) & (sch[row, x0:x1] > 185)
        if int((band & ~bright).sum()) >= 3:
            last = row
            gap = 0
        else:
            gap += 1
            if gap > allow:
                break
    hch = hsv[..., 0]
    jersey = mask[y:y + bh, x0:x1]
    if int(jersey.sum()) > 20:
        jersey_h = float(np.median(hch[y:y + bh, x0:x1][jersey]))
    else:
        jersey_h = 0.0
    pad_limit = min(int(height * 0.90), last + 32)
    gap = 0
    row = last
    while row + 4 < pad_limit:
        hs = hch[row:row + 4, x0:x1]
        ss = sch[row:row + 4, x0:x1]
        vs = vch[row:row + 4, x0:x1]
        field = (hs > 65) & (hs < 110) & (ss > 55)
        light = (ss < 85) & (vs > 55) & ~field
        hue = float(np.median(hs))
        hue_ok = min(abs(hue - jersey_h), 180 - abs(hue - jersey_h)) < 28 or float(np.median(ss)) < 32
        if float(light.mean()) > 0.35 and hue_ok:
            row += 4
            last = row
            gap = 0
        else:
            gap += 4
            if gap > 8:
                break
            row += 4
    return last


def _jersey_count(jersey, bbox):
    if jersey is None:
        return 0
    x, y, bw, bh = bbox
    height, width = jersey.shape[:2]
    x0, x1 = max(0, int(x)), min(width, int(x + bw))
    y0, y1 = max(0, int(y)), min(height, int(y + bh * 0.8))
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(jersey[y0:y1, x0:x1].sum())


def _flow_ring(prev_gray, gray, ring):
    """Move the ring with the same forward-backward check used on the field."""
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts = np.stack([
        ring["cx"] + ring["r"] * np.cos(angles),
        ring["cy"] + ring["r"] * 0.62 * np.sin(angles),
    ], axis=1).astype(np.float32).reshape(-1, 1, 2)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    if nxt is None:
        return None
    back, st2, _ = cv2.calcOpticalFlowPyrLK(
        gray, prev_gray, nxt, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    if back is None:
        return None
    fb = np.linalg.norm(pts.reshape(-1, 2) - back.reshape(-1, 2), axis=1)
    ok = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 2.5)
    if int(ok.sum()) < 5:
        return None
    shift = np.median(nxt.reshape(-1, 2)[ok] - pts.reshape(-1, 2)[ok], axis=0)
    return (float(ring["cx"] + shift[0]), float(ring["cy"] + shift[1]))


def _propose_bodies(frame, hint=None, prev_r=None, leg_min=0.32):
    """One box per body. The blue ring and the '1' marker are labels,
    not players. A jersey seed grows by a fixed fraction of its own height
    so the box covers helmet and legs."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hch, sch, vch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    height, width = frame.shape[:2]
    ring = _find_ring(hsv, hint, prev_r)
    marker = _find_marker(hsv, ring)
    red, blue = _team_masks(hsv)
    avoid = np.zeros((height, width), np.uint8)
    if ring is not None:
        cv2.circle(avoid, (int(ring["cx"]), int(ring["cy"])),
                   int(max(10, ring["r"] * 1.3)), 255, -1)
    if marker is not None:
        x, y, bw, bh = marker
        avoid[max(0, y - 8):min(height, y + bh + 10),
              max(0, x - 8):min(width, x + bw + int(bw * 1.7))] = 255
    avoid[int(height * 0.88):, :] = 255
    # The scoreboard sits in the corner. Players who run downfield pass
    # through the rest of the top of the picture, so that area stays open.
    avoid[:int(height * 0.16), :int(width * 0.22)] = 255
    avoid[:int(height * 0.055), :] = 255
    dets = []
    for team, mask0 in (("red", red), ("blue", blue)):
        mask = mask0.copy()
        mask[avoid > 0] = False
        hard = mask.astype(np.uint8) * 255
        seeds = []
        for comp in _merge_fragments(_mask_components(mask)):
            seeds.extend(_split_wide(hard, comp["bbox"]))
        for x, y, bw, bh in seeds:
            if bw * bh < 700 or bh < 22 or bw > 170 or bw < 16:
                continue
            if y > height * 0.84:
                continue
            y1 = min(height - 1, y + bh + int(max(16, bh)))
            if y1 <= y + bh:
                continue
            leg = float(((sch[y + bh:y1, x:x + bw] < 105)
                         & (vch[y + bh:y1, x:x + bw] > 48)).mean())
            if leg < leg_min:
                continue
            up = int(0.48 * bh)
            down = int(1.35 * bh)
            cx = x + bw / 2.0
            half = int(max(bw * 0.52, 16))
            x0 = int(max(0, cx - half))
            x1 = int(min(width - 1, cx + half))
            y0 = int(max(0, y - up))
            yb = int(min(height * 0.90, y + bh + down))
            if x1 - x0 < 34 or yb - y0 < 36:
                continue
            if (x0 + x1) / 2 > 0.80 * width and (y0 + yb) / 2 > 0.75 * height:
                continue
            dets.append(dict(
                team=team, bbox=(x0, y0, x1 - x0, yb - y0),
                feet=((x0 + x1) / 2.0, float(yb)),
                main=False, frac=float(leg), seed=(x, y, bw, bh)))
    if ring is not None:
        best, best_d = None, 1e9
        for det in dets:
            # The ring marks the quarterback, on either team.
            if det["team"] not in ("red", "blue"):
                continue
            x, y, bw, bh = det["bbox"]
            if ring["cy"] < y + 0.35 * bh:
                continue
            dist = abs((x + bw / 2.0) - ring["cx"])
            if dist < best_d and dist < max(bw, 40):
                best, best_d = det, dist
        if best is not None:
            x, y, bw, bh = best["bbox"]
            yb = int(min(height * 0.90, max(y + bh, ring["cy"] + 2)))
            best["bbox"] = (x, y, bw, max(10, yb - y))
            best["feet"] = (float(ring["cx"]), float(ring["cy"]))
            best["main"] = True
    dets = sorted(dets, key=lambda d: -(d["bbox"][2] * d["bbox"][3]))
    kept = []
    for det in dets:
        if any(det["team"] == other["team"]
               and _iou_over_smaller(det["bbox"], other["bbox"]) > 0.55
               for other in kept):
            continue
        kept.append(det)
    return kept, ring


def _merge_fragments(comps, gap=8, max_union=110):
    """Join the two halves of one jersey (the number splits the color)
    and leave side-by-side teammates apart."""
    boxes = [dict(c) for c in comps]
    changed = True
    while changed:
        changed = False
        used, nxt = set(), []
        for i, comp in enumerate(boxes):
            if i in used:
                continue
            box, area = comp["bbox"], comp["area"]
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                other = boxes[j]["bbox"]
                gap_x = max(box[0], other[0]) - min(box[0] + box[2], other[0] + other[2])
                if gap_x > gap:
                    continue
                overlap = min(box[1] + box[3], other[1] + other[3]) - max(box[1], other[1])
                if overlap < 0.35 * min(box[3], other[3]):
                    continue
                x0 = min(box[0], other[0])
                y0 = min(box[1], other[1])
                x1 = max(box[0] + box[2], other[0] + other[2])
                y1 = max(box[1] + box[3], other[1] + other[3])
                if x1 - x0 > max_union:
                    continue
                box = (x0, y0, x1 - x0, y1 - y0)
                area += boxes[j]["area"]
                used.add(j)
                changed = True
            nxt.append(dict(bbox=box, area=area))
        boxes = nxt
    return boxes


def _merge_vertical(comps, gap_y=36, hsv=None):
    """A white jersey number splits one player into a top piece and a
    bottom piece. Those pieces share a column. Two teammates side by side
    do not."""
    boxes = [dict(c) for c in comps]
    changed = True
    while changed:
        changed = False
        used, nxt = set(), []
        for i, comp in enumerate(boxes):
            if i in used:
                continue
            box, area = comp["bbox"], comp["area"]
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                other = boxes[j]["bbox"]
                if min(box[2], other[2]) > 42:
                    continue
                gap = max(box[1], other[1]) - min(box[1] + box[3], other[1] + other[3])
                if gap > gap_y:
                    continue
                if gap > 6 and hsv is not None:
                    y_a = min(box[1] + box[3], other[1] + other[3])
                    y_b = max(box[1], other[1])
                    x_a = max(box[0], other[0])
                    x_b = min(box[0] + box[2], other[0] + other[2])
                    if y_b > y_a and x_b > x_a:
                        hch = hsv[y_a:y_b, x_a:x_b, 0]
                        sch = hsv[y_a:y_b, x_a:x_b, 1]
                        field = (hch > 60) & (hch < 110) & (sch > 40)
                        if field.size and float(field.mean()) > 0.55:
                            continue
                hover = (min(box[0] + box[2], other[0] + other[2])
                         - max(box[0], other[0]))
                if hover < 0.45 * min(box[2], other[2]):
                    continue
                x0 = min(box[0], other[0])
                y0 = min(box[1], other[1])
                x1 = max(box[0] + box[2], other[0] + other[2])
                y1 = max(box[1] + box[3], other[1] + other[3])
                if y1 - y0 > 170 or x1 - x0 > 90:
                    continue
                box = (x0, y0, x1 - x0, y1 - y0)
                area += boxes[j]["area"]
                used.add(j)
                changed = True
            nxt.append(dict(bbox=box, area=area))
        boxes = nxt
    return boxes


def _project_point(u, v, A):
    pt = cv2.perspectiveTransform(
        np.array([[[u, v]]], np.float32), A.astype(np.float32))[0, 0]
    if not np.isfinite(pt).all():
        return None
    return float(pt[0]), float(pt[1])


def _world_px(world, H):
    """Where a field point lands in this frame. Yard y can drift, so callers
    should trust the horizontal alignment more than the height."""
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    pt = cv2.perspectiveTransform(
        np.array([[[world[0], world[1]]]], np.float32),
        Hinv.astype(np.float32))[0, 0]
    if not np.isfinite(pt).all():
        return None
    return float(pt[0]), float(pt[1])


def _accepted_body(det, A):
    """A body whose feet land on the field. The downfield window is applied
    later, once every body on this frame is known."""
    x0, y0, bw, bh = det["bbox"]
    fx, fy = det["feet"]
    world = _project_point(fx, fy, A)
    if world is None:
        return None
    xw, yw = world
    corners = np.array([[[x0, y0 + bh]], [[x0 + bw, y0 + bh]]], np.float32)
    wc = cv2.perspectiveTransform(corners, A.astype(np.float32))[:, 0, :]
    width_yd = float(np.linalg.norm(wc[1] - wc[0]))
    if not (0.25 <= width_yd <= 6.5):
        return None
    if not (-10 <= yw <= 50):
        return None
    if det["team"] == "blue" and width_yd >= 3.2 and bw > bh * 1.15:
        return None
    return dict(team=det["team"], world=(float(xw), float(yw)),
                bbox=(int(x0), int(y0), int(bw), int(bh)),
                frac=det["frac"], width_yd=width_yd,
                main=bool(det["main"]))


def _on_this_field(items):
    """Keep feet on the Clip 4 field, and also the cluster standing at the
    line when a different camera puts that line outside those yards."""
    if not items:
        return []
    med = float(np.median([d["world"][0] for d in items]))
    kept = []
    for det in items:
        xw = det["world"][0]
        if -15 <= xw <= 125 or abs(xw - med) <= 18.0:
            kept.append(det)
    return kept


def _claim_arrow_body(frame, hsv, items, ring):
    """The player under the '1' is the main player when the ring is missing.

    The box starts under the chevron, so the digit stays above it.
    """
    if ring is not None or not items:
        return
    marked = _marked_arrow(hsv, _arrow_candidates(hsv), frame.shape)
    if marked is None:
        return
    ax, ay, abh = marked["cx"], marked["cy"], marked["bh"]
    best, best_d = None, 80.0
    for det in items:
        x, y, bw, bh = det["bbox"]
        dx = abs((x + bw / 2.0) - ax)
        if dx < best_d and y <= ay + abh and y + bh > ay:
            best, best_d = det, dx
    if best is None:
        return
    x, y, bw, bh = best["bbox"]
    ny = int(max(y, ay - abh * 0.55))
    nh = (y + bh) - ny
    if nh >= 36:
        best["bbox"] = (x, ny, bw, int(nh))
    best["main"] = True


# Set on the first frame of a clip. Clip 4's defense stands past midfield,
# so the flag stays off and the pant test is unchanged. A clip whose whole
# line is on the near side of that uses the wider pant test.
_LOOSE_LINE = None


def _bodies_of(frame, A, hint, prev_r, leg_min):
    bodies, ring = _propose_bodies(frame, hint, prev_r, leg_min=leg_min)
    items = [d for d in (_accepted_body(b, A) for b in bodies) if d is not None]
    items = _on_this_field(items)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    items = [d for d in items if not _on_field_logo(hsv, d["bbox"])]
    return items, ring, hsv


def _person_peaks(hsv, anchor):
    """Vertical bodies near a far player whose jersey has faded.

    Against the end zone the team color melts into the paint, so the
    usual seed disappears. A person is the column of non-field pixels
    under that paint. The flat end-zone stripe is a row of paint, and
    it is removed before the columns are measured.
    """
    height, width = hsv.shape[:2]
    ax, ay, aw, ah = anchor
    foot_x = ax + aw / 2.0
    foot_y = ay + ah
    x0 = max(0, int(foot_x - 110))
    x1 = min(width, int(foot_x + 110))
    y0 = max(int(height * 0.02), int(foot_y - 140))
    y1 = min(int(height * 0.42), int(foot_y + 20))
    if y1 - y0 < 30 or x1 - x0 < 24:
        return []
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    field = (hh > 65) & (hh < 110) & (ss > 55) & (vv > 40)
    person = (~field) & (vv > 48) & (ss > 22)
    paint = (hh > 96) & (hh < 115) & (ss > 100) & (vv > 60) & (vv < 170)
    sub = person[y0:y1, x0:x1].copy()
    row_paint = paint[y0:y1, x0:x1].mean(axis=1)
    sub[row_paint > 0.40] = False
    cols = sub.sum(axis=0).astype(np.float32)
    smooth = np.convolve(cols, np.ones(5) / 5.0, mode="same")
    boxes = []
    i = 0
    n = int(smooth.size)
    while i < n:
        if smooth[i] < 14:
            i += 1
            continue
        j = i
        while j < n and smooth[j] >= 10:
            j += 1
        seg = smooth[i:j]
        if float(seg.max()) < 20 or not (12 <= (j - i) <= 80):
            i = j
            continue
        if (j - i) > 36:
            thr = float(seg.max()) * 0.45
            cuts = [0]
            for k in range(2, len(seg) - 2):
                if seg[k] < thr and seg[k] <= seg[k - 1] and seg[k] <= seg[k + 1]:
                    if k - cuts[-1] > 14:
                        cuts.append(k)
            cuts.append(len(seg))
            spans = list(zip(cuts[:-1], cuts[1:]))
        else:
            spans = [(0, len(seg))]
        for a, b in spans:
            if b - a < 12 or float(smooth[i + a:i + b].max()) < 18:
                continue
            rows = np.where(sub[:, i + a:i + b].any(axis=1))[0]
            if rows.size < 16:
                continue
            box = (
                int(x0 + i + a),
                int(y0 + rows[0]),
                int(b - a),
                int(rows[-1] - rows[0]),
            )
            if box[3] < 28 or box[2] > 70 or box[3] > 170:
                continue
            if (_mostly_grass(hsv, box) or _on_endzone_letters(hsv, box)
                    or _on_field_logo(hsv, box)):
                continue
            boxes.append(box)
        i = j
    return boxes


def _closest_peak(hsv, anchor, occupied):
    """The body column nearest this track, not one already boxed."""
    ax = anchor[0] + anchor[2] / 2.0
    ay = anchor[1] + anchor[3]
    best, best_d = None, 110.0
    for box in _person_peaks(hsv, anchor):
        if any(_iou_over_smaller(box, other) > 0.20 for other in occupied):
            continue
        fx, fy = _foot_of(box)
        dist = float(np.hypot(fx - ax, fy - ay))
        if dist < best_d:
            best_d = dist
            best = box
    return best


def _far_players(hsv):
    """Small jerseys in the far part of the field.

    A player running away gets short and faint, and the usual seed rules
    leave him out. The scoreboard corner and the end-zone name are outside
    this search. A white jersey counts as blue: that is the away uniform.
    """
    height, width = hsv.shape[:2]
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    ycut = int(height * 0.38)
    specs = (
        ("red", ((hh < 15) | (hh > 165)) & (ss > 55) & (vv > 48)),
        ("blue", (hh > 100) & (hh < 132) & (ss > 55) & (vv > 48)),
        ("blue", (ss < 70) & (vv > 155)),
    )
    found = []
    for team, mask in specs:
        mask = mask.copy()
        mask[:int(height * 0.18), :int(width * 0.32)] = False
        mask[ycut:] = False
        opened = cv2.morphologyEx(
            mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(opened)
        for i in range(1, count):
            x, y, bw, bh, area = [int(v) for v in stats[i]]
            if area < 80 or bh < 16 or bw < 16 or bw > 48 or bh < bw:
                continue
            top, bot = _person_span(hsv, x, y, max(bw, 18), bh, hsv.shape)
            nh = int(bot - top)
            if nh < 44 or nh > 160 or top < 8:
                continue
            box = (int(x), int(top), int(max(bw, 18)), nh)
            cx = box[0] + box[2] / 2.0
            cy = box[1] + box[3] / 2.0
            if cx < width * 0.30 and cy < height * 0.20:
                continue
            if (_mostly_grass(hsv, box) or _on_field_logo(hsv, box)
                    or _on_endzone_letters(hsv, box)
                    or _on_white_sideline(hsv, box)):
                continue
            y1 = box[1] + int(box[3] * 0.55)
            patch = hsv[box[1]:y1, box[0]:box[0] + box[2]]
            if patch.size == 0:
                continue
            white = float(((patch[..., 1] < 70) & (patch[..., 2] > 150)).mean())
            colored = float(mask[box[1]:y1, box[0]:box[0] + box[2]].mean())
            if white < 0.28 and colored < 0.12:
                continue
            # A graphic has no grass under it. A player running away still
            # stands on the field.
            yb = min(height - 1, box[1] + box[3])
            yb2 = min(height, yb + 14)
            if yb2 <= yb:
                continue
            under = hsv[yb:yb2, box[0]:box[0] + box[2]]
            grass = float(((under[..., 0] > 65) & (under[..., 0] < 110)
                           & (under[..., 1] > 55) & (under[..., 2] > 40)).mean())
            if grass < 0.45:
                continue
            found.append((team, box))
    kept = []
    for team, box in found:
        if any(_iou_over_smaller(box, other) > 0.25 for _, other in kept):
            continue
        kept.append((team, box))
    return kept


def frame_candidates(frame, A, hint=None, prev_r=None):
    """Bodies for one frame, feet on the ground plane. The ring is returned
    beside the list so the main player's feet can follow it."""
    global _LOOSE_LINE
    items, ring, hsv = _bodies_of(frame, A, hint, prev_r, 0.32)
    if _LOOSE_LINE is None:
        back = sum(1 for d in items
                   if d["team"] == "blue" and d["world"][0] > 44)
        _LOOSE_LINE = back == 0 and len(items) >= 4
    if _LOOSE_LINE:
        items, ring, hsv = _bodies_of(frame, A, hint, prev_r, 0.15)
        _claim_arrow_body(frame, hsv, items, ring)
    return items, ring


def _merge_stacked(dets):
    """Torso and legs of one player overlap in the image and land a few
    yards apart on the field, because the torso is off the ground. Side by
    side teammates barely overlap horizontally, so they stay separate."""
    dets = [dict(d) for d in dets]
    changed = True
    while changed:
        changed = False
        nxt, used = [], set()
        for i, a in enumerate(dets):
            if i in used:
                continue
            box, world, frac = a["bbox"], a["world"], a["frac"]
            for j in range(i + 1, len(dets)):
                b = dets[j]
                if j in used or b["team"] != a["team"]:
                    continue
                ov = (min(box[0] + box[2], b["bbox"][0] + b["bbox"][2])
                      - max(box[0], b["bbox"][0]))
                if ov < 0.45 * min(box[2], b["bbox"][2]):
                    continue
                a_bot = box[1] + box[3]
                b_bot = b["bbox"][1] + b["bbox"][3]
                if max(box[1], b["bbox"][1]) - min(a_bot, b_bot) > 36:
                    continue
                if _dist(world, b["world"]) > 4.0:
                    continue
                # two full bodies lined up downfield also overlap in the
                # image. Only fold a smaller piece (legs, helmet) into a body.
                area_a = box[2] * box[3]
                area_b = b["bbox"][2] * b["bbox"][3]
                if min(area_a, area_b) > 0.5 * max(area_a, area_b):
                    continue
                if min(box[3], b["bbox"][3]) > 45:
                    continue
                box = _box_union(box, b["bbox"])
                if b_bot > a_bot:
                    world = b["world"]
                frac = max(frac, b["frac"])
                used.add(j)
                changed = True
            nxt.append(dict(a, bbox=box, world=world, frac=frac))
        dets = nxt
    return dets


def drop_stationary(per_frame, link=1.4, min_life=120, max_move=1.8):
    """Painted decals stay put in field yards. Players do not, once the
    play starts. A long chain that never moves is removed from every frame."""
    n = len(per_frame)
    claimed = [set() for _ in range(n)]
    kill = [set() for _ in range(n)]
    for t0 in range(n):
        for i, d in enumerate(per_frame[t0]):
            if i in claimed[t0]:
                continue
            chain = [(t0, i)]
            claimed[t0].add(i)
            last = d["world"]
            team = d["team"]
            tt = t0
            while tt + 1 < n:
                best, bd = None, link
                for j, e in enumerate(per_frame[tt + 1]):
                    if j in claimed[tt + 1] or e["team"] != team:
                        continue
                    dist = _dist(e["world"], last)
                    if dist < bd:
                        best, bd = j, dist
                if best is None:
                    break
                claimed[tt + 1].add(best)
                last = per_frame[tt + 1][best]["world"]
                chain.append((tt + 1, best))
                tt += 1
            if len(chain) < min_life:
                continue
            a = per_frame[chain[0][0]][chain[0][1]]["world"]
            b = per_frame[chain[-1][0]][chain[-1][1]]["world"]
            if _dist(a, b) <= max_move:
                for t, j in chain:
                    kill[t].add(j)
    removed = sum(len(s) for s in kill)
    cleaned = [[d for j, d in enumerate(frame) if j not in kill[t]]
               for t, frame in enumerate(per_frame)]
    return cleaned, removed


def _pick_spread(pool, sep, limit):
    """Greedy largest-first, dropping boxes that land on someone already
    picked. Leftover boxes closer than `sep` are the same body."""
    pool = sorted(pool, key=lambda d: -(d["frac"] * d["bbox"][2] * d["bbox"][3]))
    chosen = []
    for d in pool:
        if any(_dist(d["world"], c["world"]) < sep for c in chosen):
            continue
        chosen.append(d)
        if len(chosen) == limit:
            break
    return chosen


def select_roster(dets, per_team=7):
    """Lock the snap. Reds are the front (smaller x, toward the end zone
    the camera faces). Blues are behind them. Numbers run left to right.
    The blue body that owns the ground ring stays on the roster."""
    roster = []
    next_id = 1

    def _at_the_line(team, strict):
        tall = [d for d in dets if d["team"] == team and d["bbox"][3] >= 40]
        group = [d for d in tall if strict(d)]
        # Clip 4's line is the strict window. Another camera leaves that
        # window empty, and the players are the cluster around the median.
        if len(group) >= 4:
            return group
        if not tall:
            return []
        med = float(np.median([d["world"][0] for d in tall]))
        return [d for d in tall if abs(d["world"][0] - med) <= 18.0]

    reds = _pick_spread(
        _at_the_line("red", lambda d: d["world"][0] < 56), sep=1.5, limit=per_team)
    blues = _pick_spread(
        _at_the_line("blue", lambda d: d["world"][0] > 44), sep=1.7, limit=per_team)
    groups = {"red": reds, "blue": blues}
    for team, group in list(groups.items()):
        main = [d for d in dets if d.get("main") and d["team"] == team]
        if not main:
            continue
        have = any(_dist(main[0]["world"], b["world"]) < 0.8 for b in group)
        if not have:
            if len(group) >= per_team:
                group = group[:per_team - 1]
            group = group + [main[0]]
        groups[team] = group
    for group in (groups["red"], groups["blue"]):
        for d in sorted(group, key=lambda d: d["world"][1]):
            roster.append(dict(id=next_id, team=d["team"], world=d["world"],
                               bbox=d["bbox"], main=bool(d.get("main"))))
            next_id += 1
    return roster


def _feet_px(world, A):
    try:
        Hinv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None
    pt = cv2.perspectiveTransform(
        np.array([[[world[0], world[1]]]], np.float32),
        Hinv.astype(np.float32))[0, 0]
    if not np.isfinite(pt).all():
        return None
    return float(pt[0]), float(pt[1])


def _on_screen(world, A, shape):
    """Feet still inside the picture, above the bottom banner."""
    px = _feet_px(world, A)
    if px is None:
        return False
    h, w = shape[:2]
    u, v = px
    return -12 <= u <= w + 12 and -12 <= v <= h * 0.90


def _shift_box(bbox, old_world, new_world, A):
    a = _feet_px(old_world, A)
    b = _feet_px(new_world, A)
    if a is None or b is None:
        return bbox
    x, y, bw, bh = bbox
    return (int(round(x + b[0] - a[0])), int(round(y + b[1] - a[1])),
            int(bw), int(bh))


def _snap_body(frame, foot, team, reach=55.0):
    """A player-shaped blob near the predicted feet. Team color is tried
    first; any non-field body is accepted if the jersey is hidden."""
    u, v = foot
    h, w = frame.shape[:2]
    span = 70 if reach <= 70 else int(reach) + 20
    x0, x1 = max(0, int(u - span)), min(w, int(u + span))
    y0, y1 = max(0, int(v - span * 2)), min(h, int(v + 18))
    if x1 - x0 < 12 or y1 - y0 < 20:
        return None
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hch, sch, vch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    field = (hch > 60) & (hch < 110) & (sch > 40)
    if team == "blue":
        color = (hch > 90) & (hch < 145) & (sch > 35) & (vch > 40)
    else:
        color = ((hch < 18) | (hch > 160)) & (sch > 35) & (vch > 35)
    body = (~field) & (vch > 45) & (sch > 15)
    use = color.astype(np.uint8) * 255
    if int(use.sum()) < 400 * 255:
        if reach > 100:
            return None
        use = body.astype(np.uint8) * 255
    if int(use.sum()) < 500 * 255:
        return None
    use = cv2.morphologyEx(use, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(use)
    best, best_d = None, float(reach)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 280 or bh < 20 or bw > bh * 3.5 or area > 6500:
            continue
        if bw > 100 and bh < bw:
            continue
        fx = x0 + x + bw / 2.0
        fy = y0 + y + bh
        if fy > h * 0.88 or fy < h * 0.10:
            continue
        dist = float(np.hypot(fx - u, fy - v))
        if dist < best_d:
            best_d = dist
            best = (int(x0 + x), int(y0 + y), int(bw), int(bh), fx, fy)
    return best


def _assign_gated(preds, dets, A=None):
    """preds: (id, xy, team, gate). Nearest match first, one each.

    Position is the reference: world distance plus how close the blob is to
    the predicted spot in the image. Jersey color is only a tie-breaker.
    """
    pairs = []
    for tid, xy, team, gate in preds:
        pred_px = _feet_px(xy, A) if A is not None else None
        for j, d in enumerate(dets):
            dist = _dist(xy, d["world"])
            bx, by, bw, bh = d["bbox"]
            fx, fy = bx + bw / 2.0, by + bh
            img = 0.0
            if pred_px is not None:
                pix = float(np.hypot(fx - pred_px[0], fy - pred_px[1]))
                if pix > 160 and dist > 2.0:
                    continue
                img = pix / 45.0
            if dist > 8.0:
                continue
            if dist > gate and img > 1.2:
                continue
            if dist > gate + 3.0:
                continue
            penalty = 0.0 if d["team"] == team else 1.2
            pairs.append((dist + img + penalty, tid, j))
    pairs.sort()
    used_id, used_d = set(), set()
    out = {}
    for dist, tid, j in pairs:
        if tid in used_id or j in used_d:
            continue
        used_id.add(tid)
        used_d.add(j)
        out[tid] = dets[j]
    return out


def _field_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hch, sch, vch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (hch > 78) & (hch < 102) & (sch > 95) & (vch > 40)


def _foot_of(bbox):
    x, y, bw, bh = bbox
    return (x + bw / 2.0, y + float(bh))


def _assign_image(preds, dets, gate=85):
    """Match a number to a body by where the feet are in the picture."""
    pairs = []
    for tid, (u, v), team in preds:
        for j, det in enumerate(dets):
            fx, fy = _foot_of(det["bbox"])
            pix = float(np.hypot(fx - u, fy - v))
            if pix > gate:
                continue
            penalty = 0.0 if det["team"] == team else 28.0
            pairs.append((pix + penalty, tid, j))
    pairs.sort()
    used_id, used_d = set(), set()
    out = {}
    for _pix, tid, j in pairs:
        if tid in used_id or j in used_d:
            continue
        used_id.add(tid)
        used_d.add(j)
        out[tid] = dets[j]
    return out


def _clamp_step(prev, new, limit=6.0):
    if new is None or not np.isfinite(new).all():
        return prev
    dist = _dist(prev, new)
    if dist <= limit or dist < 1e-6:
        return (float(new[0]), float(new[1]))
    scale = limit / dist
    return (prev[0] + (new[0] - prev[0]) * scale,
            prev[1] + (new[1] - prev[1]) * scale)


def _on_label(box, ring, arrow, shape):
    """True when a box sits on the quarterback's circle, the '1' arrow, or the KAG logo.

    A teammate beside the quarterback is a player. The circle test is the
    ring itself, and the arrow test is a short box on the chevron.
    """
    height, width = shape[:2]
    fx, fy = _foot_of(box)
    cx = box[0] + box[2] / 2.0
    cy = box[1] + box[3] / 2.0
    bh = box[3]
    if cx > width * 0.78 and cy > height * 0.72:
        return True
    if ring is not None:
        rad = float(ring.get("r") or 36.0)
        if (cx - ring["cx"]) ** 2 + (cy - ring["cy"]) ** 2 <= (rad * 1.15) ** 2:
            return True
        if (fx - ring["cx"]) ** 2 + (fy - ring["cy"]) ** 2 <= (rad * 0.7) ** 2:
            return True
        # The digit above the ring is short. A player in that column is not.
        if (bh < 78 and abs(cx - ring["cx"]) < 72
                and ring["cy"] - 210 < cy < ring["cy"] - rad * 0.4):
            return True
    # A drifted arrow hint sits on the far pile. Only trust an arrow
    # that is actually above this frame's ring.
    if (arrow is not None and ring is not None
            and abs(arrow[0] - ring["cx"]) < 110
            and 10 < ring["cy"] - arrow[1] < 220):
        ax, ay = arrow[0], arrow[1]
        abh = float(arrow[2] if len(arrow) > 2 else 40.0)
        # The chevron box is short. A teammate under the marker is taller.
        if (bh < 80 and abs(cx - ax) < 42
                and ay - abh * 0.8 < cy < ay + abh * 0.45):
            return True
    return False


def _arrow_has_digit(hsv, arrow):
    """The real down marker has a dark plaque and a white digit above the chevron.

    A blue taper on a jersey, or the arrow drifted onto a white yard line,
    does not. White alone is the yard line. When the chevron is cut off at
    the right edge, the plaque sits just beside that fragment.
    """
    height, width = hsv.shape[:2]
    ax, ay, abh = float(arrow[0]), float(arrow[1]), float(arrow[2])

    def _plaque(x0, y0, x1, y1):
        x0, x1 = max(0, int(x0)), min(width, int(x1))
        y0, y1 = max(0, int(y0)), min(height, int(y1))
        if x1 - x0 < 20 or y1 - y0 < 20:
            return False
        patch = hsv[y0:y1, x0:x1]
        dark = (patch[..., 2] < 80) & (patch[..., 1] < 110)
        white = (patch[..., 1] < 80) & (patch[..., 2] > 160)
        return float(dark.mean()) >= 0.10 and 0.08 <= float(white.mean()) <= 0.70

    y1 = ay - abh * 0.4
    if _plaque(ax - 28, y1 - 55, ax + 28, y1):
        return True
    if ax > width * 0.88:
        # The clipped chevron is a small fragment. The plaque is a
        # tight patch beside it, so a wide mean washes the digit out.
        x0, x1 = int(ax - 130), int(min(width, ax + 20))
        y0, y1 = int(ay - 20), int(ay + 110)
        for yy in range(y0, y1 - 40, 8):
            for xx in range(x0, x1 - 40, 8):
                if _plaque(xx, yy, xx + 56, yy + 50):
                    return True
    return False


def _qb_body_under_arrow(hsv, mask, arrow, shape):
    """Jersey standing under the blue arrow. The '1' stays above the box.

    The arrow's own pixels are blue, so a box built on the chevron looks
    occupied. The player is the body below that chevron. Only the arrow
    that still has the digit above it is allowed to choose that body.
    """
    if arrow is None or not _arrow_has_digit(hsv, arrow):
        return None
    height, width = shape[:2]
    ax, ay, abh = float(arrow[0]), float(arrow[1]), float(arrow[2])
    best = None
    best_dx = 80.0
    for body in _jersey_bodies(mask, shape, hsv):
        bx, by, bw, bh, cx, fy = body
        cy = by + bh / 2.0
        if cy < height * 0.20 or cy > height * 0.90:
            continue
        # The grown box can reach up over the chevron. The player is the
        # one whose feet, and whose middle, sit under the arrow.
        if fy < ay + 40 or cy < ay - 10:
            continue
        dx = abs(cx - ax)
        if dx >= best_dx:
            continue
        box = (int(bx), int(by), int(bw), int(bh))
        if _on_field_logo(hsv, box):
            continue
        if _jersey_count(mask, box) < 40:
            continue
        best_dx = dx
        best = body
    if best is None:
        return None
    bx, by, bw, bh, cx, fy = best
    # The digit sits above the chevron. Do not let the box climb into it.
    ny = int(max(by, ay - abh * 0.55))
    nh = int(fy - ny)
    if nh < 70:
        return None
    nx = int(round(cx - bw / 2.0))
    nx = max(-20, min(width - 4, nx))
    return (nx, ny, int(bw), nh), (float(cx), float(fy))


def _qb_color(hch, sch, vch, team):
    """Jersey pixels for the quarterback's own team."""
    if team == "red":
        return ((hch < 18) | (hch > 160)) & (sch > 75) & (vch > 55)
    return (hch > 100) & (hch < 135) & (sch > 75) & (vch > 55)


def _qb_from_foot(hsv, foot, shape, team="blue"):
    """Box the quarterback upward from a foot point when the ring is gone.

    The walk stops at the saturated blue arrow so the '1' marker stays
    above the box.
    """
    height, width = shape[:2]
    cx, cy = int(round(foot[0])), int(round(foot[1]))
    half = 32
    x0, x1 = max(0, cx - half), min(width, cx + half)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = _qb_color(H, S, V, team)
    top = cy
    gap = 0
    gaps = 0
    for row in range(cy - 4, max(0, cy - 220), -2):
        if x1 <= x0:
            break
        if float(np.median(S[row, x0:x1])) > 180 and float(np.median(V[row, x0:x1])) > 140:
            top = max(0, row - 18)
            break
        if int(blue[row, x0:x1].sum()) >= 8:
            if gap > 24:
                gaps += 1
                if gaps > 1:
                    break
            top = row
            gap = 0
        else:
            gap += 2
            if gap > 70:
                break
    ny = max(0, top - 28)
    nh = int(np.clip(cy - ny, 100, 200))
    ny = max(0, int(cy - nh))
    bw = int(np.clip(nh * 0.46, 42, 90))
    nx = int(round(cx - bw / 2.0))
    nx = max(-20, min(width - 4, nx))
    return (nx, ny, bw, nh)


def _qb_box(hsv, ring, shape, team="blue"):
    """Box the quarterback from his helmet down to the blue ring.

    One gap is allowed for the silver pants between the jersey and the
    ring. A second gap is the arrow above the helmet, and the walk stops
    there so the box does not swallow the '1' marker.
    """
    height, width = shape[:2]
    cx = int(round(ring["cx"]))
    cy = int(round(ring["cy"]))
    rad = float(ring.get("r") or 36)
    half = int(max(24, min(36, rad * 0.9)))
    x0, x1 = max(0, cx - half), min(width, cx + half)
    hch, sch, vch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = _qb_color(hch, sch, vch, team)
    top = cy
    gap = 0
    gaps = 0
    row = cy - 4
    limit = max(0, cy - 240)
    while row > limit:
        n = int(blue[row, x0:x1].sum()) if x1 > x0 else 0
        if n >= 8:
            if gap > 24:
                gaps += 1
                if gaps > 1:
                    break
            top = row
            gap = 0
        else:
            gap += 2
            if gap > 84:
                break
        row -= 2
    ny = max(0, top - 44)
    foot = min(height - 4, int(cy + 0.22 * rad))
    nh = int(np.clip(foot - ny, 96, 220))
    ny = max(0, int(foot - nh))
    bw = int(np.clip(nh * 0.48, 42, 96))
    nx = int(round(cx - bw / 2.0))
    nx = max(-20, min(width - 4, nx))
    return (nx, ny, bw, nh)


def _box_on_ring(bbox, ring, shape):
    """Keep the box on the body, with its foot point on the ring."""
    _, _, bw, bh = bbox
    height, width = shape[:2]
    nx = int(round(ring["cx"] - bw / 2.0))
    ny = int(round(ring["cy"] - bh))
    nx = max(-20, min(width - 4, nx))
    ny = max(-20, min(height - 4, ny))
    return (nx, ny, int(bw), int(bh))


def _window_conf(fused, tid, t, n):
    """Agreement of this frame with the frames around it."""
    pts = []
    lo = max(0, t - 6)
    hi = min(n, t + 7)
    for k in range(lo, hi):
        item = fused[k].get(tid)
        if item is not None:
            pts.append(item["pos"])
    if len(pts) < 3 or tid not in fused[t]:
        return 0.35
    med = np.median(np.array(pts, np.float64), axis=0)
    gap = _dist(fused[t][tid]["pos"], med)
    cover = len(pts) / float(hi - lo)
    return float(min(1.0, (0.25 + 0.75 * cover) * (0.35 + 0.65 * _agree(gap))))


def track_bidirectional(per_frame, roster, homographies, shape, rings=None, arrows=None):
    """Numbers are assigned once, at the snap, and never reused.

    Forward, each frame is matched to where that number was on the previous
    frame. Backward, each frame is matched to where that number is on the
    next frame. Jersey color breaks ties; position is the reference.
    The yard position blends the detection with the step predicted from the
    previous frame. If no blob is found, the box keeps moving from that
    prediction until the feet leave the picture.
    """
    n = len(per_frame)
    if rings is None:
        rings = [None] * n
    if arrows is None:
        arrows = [None] * n
    ids = [r["id"] for r in roster]
    team_of = {r["id"]: r["team"] for r in roster}
    main_ids = [r["id"] for r in roster if r.get("main")]
    main_id = main_ids[0] if main_ids else None
    fwd = [dict() for _ in range(n)]
    fwd[0] = {r["id"]: dict(r) for r in roster}
    last = {r["id"]: r["world"] for r in roster}
    vel = {r["id"]: (0.0, 0.0) for r in roster}
    box_of = {r["id"]: r["bbox"] for r in roster}
    img = {r["id"]: _foot_of(r["bbox"]) for r in roster}
    img_vel = {r["id"]: (0.0, 0.0) for r in roster}
    alive = {tid: True for tid in ids}
    off = {tid: 0 for tid in ids}
    miss = {tid: 0 for tid in ids}
    bare = {tid: 0 for tid in ids}
    cruise = {tid: (0.0, 0.0) for tid in ids}
    solo = {item["id"]: item["bbox"] for item in roster}
    planted = {tid: img[tid] for tid in ids}
    # Last white jersey a track was published on, while his memory stays
    # on the grass. x, y, w, h, vx, vy, frame, locked.
    white_hold = {}
    good_world = dict(last)
    good_img = dict(img)
    held_world = dict(last)
    height, width = shape[:2]
    cap = cv2.VideoCapture(CLIP)
    ok0, prev_frame = _take(cap)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if ok0 else None
    prev_field = _field_mask(prev_frame) if ok0 else None
    if ok0:
        prev_red, prev_blue = _team_masks(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV))
    else:
        prev_red = prev_blue = None
    flow_hits = flow_tries = 0
    for t in range(1, n):
        ok, frame = _take(cap)
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur_red, cur_blue = _team_masks(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        held_at = {
            tid: (img[tid], last[tid], vel[tid], img_vel[tid], box_of[tid],
                  solo[tid], planted[tid], off[tid], bare[tid],
                  good_world[tid], good_img[tid])
            for tid in ids if alive.get(tid)}
        preds = []
        pred_of = {}
        mem_of = {}
        flowed = {}
        shift_of = {}
        for tid in ids:
            if not alive[tid]:
                continue
            jmask = prev_red if team_of[tid] == "red" else prev_blue
            src = solo[tid] if bare[tid] >= 1 else box_of[tid]
            flow_tries += 1
            shift = (_flow_box(prev_gray, gray, src, prev_field, jmask)
                     if prev_gray is not None else None)
            nudge = _jersey_nudge(jmask, box_of[tid])
            if shift is None:
                du, dv = img_vel[tid]
                du, dv = du * 0.85 + nudge, dv * 0.85
            else:
                du, dv = shift[0] + nudge, shift[1]
                flowed[tid] = True
                flow_hits += 1
            u, v = img[tid]
            pred = (u + du, v + dv)
            preds.append((tid, pred, team_of[tid]))
            pred_of[tid] = pred
            mem_of[tid] = (u, v)
            img_vel[tid] = (du, dv)
            shift_of[tid] = (du, dv)
        got = _assign_image(preds, per_frame[t], gate=48)
        ring = rings[t] if t < len(rings) else None
        arrow = arrows[t] if t < len(arrows) else None
        peak_locked = set()
        peak_undo = {}
        hsv_live = None
        for tid in ids:
            if not alive[tid]:
                continue
            guide = None
            if tid == main_id and ring is not None:
                guide = (float(ring["cx"]), float(ring["cy"]))
            elif tid == main_id and arrow is not None:
                guide = (float(arrow[0]), float(arrow[1] + 2.4 * arrow[2]))
            if guide is not None and 0.12 * height < guide[1] < 0.92 * height:
                u, v = img[tid]
                hsv_now = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                if ring is not None:
                    box = _qb_box(hsv_now, ring, shape, team_of[tid])
                    landed = None
                else:
                    box = _qb_from_foot(hsv_now, guide, shape, team_of[tid])
                    landed = _qb_body_under_arrow(hsv_now, cur_blue, arrow, shape)
                    if landed is not None:
                        box, guide = landed
                # A false arrow sits on empty grass. The real one stands over
                # the quarterback, so his box still holds a jersey. When the
                # tracked arrow is the false one, use an arrow candidate in
                # the upper field that still has a jersey under it.
                empty = ring is None and landed is None and (
                    _mostly_grass(hsv_now, box)
                    or _jersey_count(cur_blue, box) < 40
                    or (box[1] + box[3] / 2.0) < height * 0.20)
                if empty and ring is None:
                    best = None
                    best_j = 40
                    for hit in _arrow_candidates(hsv_now):
                        if hit["cy"] < height * 0.18 or hit["cy"] > height * 0.62:
                            continue
                        hit_arrow = (hit["cx"], hit["cy"], hit["bh"])
                        # A chevron without the digit is another player's blue.
                        if not _arrow_has_digit(hsv_now, hit_arrow):
                            continue
                        landed_alt = _qb_body_under_arrow(
                            hsv_now, cur_blue, hit_arrow, shape)
                        if landed_alt is None:
                            continue
                        alt_box, alt = landed_alt
                        if _mostly_grass(hsv_now, alt_box):
                            continue
                        jer = _jersey_count(cur_blue, alt_box)
                        if jer < best_j:
                            continue
                        best_j = jer
                        best = (alt, alt_box)
                    if best is not None:
                        guide, box = best
                        empty = False
                if not empty:
                    fx, fy = guide
                    img_vel[tid] = (fx - u, fy - v)
                    img[tid] = (fx, fy)
                    world = _clamp_step(last[tid], _project_point(fx, fy, homographies[t]))
                    vel[tid] = (world[0] - last[tid][0], world[1] - last[tid][1])
                    last[tid] = world
                    box_of[tid] = box
                    off[tid] = 0
                    own = cur_red if team_of[tid] == "red" else cur_blue
                    bare[tid] = 0 if _jersey_count(own, box) >= 40 else bare[tid] + 1
                    fwd[t][tid] = dict(
                        team=team_of[tid], world=world, bbox=box,
                        frac=0.9, width_yd=1.2, ring=True)
                    good_world[tid] = last[tid]
                    good_img[tid] = img[tid]
                    continue
            u, v = img[tid]
            du, dv = img_vel[tid]
            pred = (u + du, v + dv)
            jmask = cur_red if team_of[tid] == "red" else cur_blue
            held = (int(round(box_of[tid][0] + du)), int(round(box_of[tid][1] + dv)),
                    box_of[tid][2], box_of[tid][3])
            fit = _fit_jersey(jmask, held, shape)
            det = got.get(tid)
            use_det = False
            if det is not None and fit is None:
                if det.get("ring") or _iou_over_smaller(det["bbox"], held) > 0.15:
                    use_det = True
            if fit is not None:
                box = fit
                fx, fy = _foot_of(box)
                if (tid == main_id and ring is not None
                        and abs(ring["cx"] - fx) < 70
                        and 0 <= ring["cy"] - fy < 80):
                    box = _box_on_ring(box, ring, shape)
                    fx, fy = float(ring["cx"]), float(ring["cy"])
                img_vel[tid] = (fx - u, fy - v)
                img[tid] = (fx, fy)
                world = _clamp_step(last[tid], _project_point(fx, fy, homographies[t]))
                vel[tid] = (world[0] - last[tid][0], world[1] - last[tid][1])
                last[tid] = world
                box_of[tid] = box
                off[tid] = 0
                bare[tid] = 0 if _jersey_count(jmask, box) >= 40 else bare[tid] + 1
                fwd[t][tid] = dict(
                    team=team_of[tid], world=world, bbox=box,
                    frac=0.8, width_yd=1.2)
                good_world[tid] = last[tid]
                good_img[tid] = img[tid]
            elif use_det:
                det = got[tid]
                fx, fy = _foot_of(det["bbox"])
                if det.get("ring"):
                    fx, fy = float(ring["cx"]), float(ring["cy"])
                img_vel[tid] = (fx - u, fy - v)
                img[tid] = (fx, fy)
                world = _clamp_step(last[tid], _project_point(fx, fy, homographies[t]))
                vel[tid] = (world[0] - last[tid][0], world[1] - last[tid][1])
                last[tid] = world
                box_of[tid] = det["bbox"]
                off[tid] = 0
                fwd[t][tid] = dict(det, world=world)
                good_world[tid] = last[tid]
                good_img[tid] = img[tid]
            else:
                lost = _jersey_count(jmask, held) < 50
                reach = 55.0 if not lost else min(280.0, 80.0 + bare[tid] * 12.0)
                snapped = None if tid in flowed and not lost else _snap_body(
                    frame, pred, team_of[tid], reach=reach)
                if snapped is not None:
                    bx, by, bw, bh, fx, fy = snapped
                    img_vel[tid] = (fx - u, fy - v)
                    img[tid] = (fx, fy)
                    world = _clamp_step(last[tid], _project_point(fx, fy, homographies[t]))
                    vel[tid] = (world[0] - last[tid][0], world[1] - last[tid][1])
                    last[tid] = world
                    box_of[tid] = (bx, by, bw, bh)
                    off[tid] = 0
                    bare[tid] = 0
                    fwd[t][tid] = dict(
                        team=team_of[tid], world=world,
                        bbox=(bx, by, bw, bh), frac=0.2, width_yd=1.5)
                    good_world[tid] = last[tid]
                    good_img[tid] = img[tid]
                    continue
                nu, nv = pred
                cx = held[0] + held[2] / 2.0
                cy = held[1] + held[3] / 2.0
                # A player running away sits in the top of the picture, and
                # his jersey stops reading as team color. The body is still
                # the column of non-field pixels next to where he was.
                if tid != main_id and cy < height * 0.22:
                    if hsv_live is None:
                        hsv_live = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    weak = (_jersey_count(jmask, held) < 80
                            or _mostly_grass(hsv_live, held))
                    if weak:
                        occupied = [fwd[t][i]["bbox"] for i in fwd[t] if i != tid]
                        peak = _closest_peak(hsv_live, held, occupied)
                        if peak is not None:
                            peak_undo[tid] = (
                                img_vel[tid], img[tid], last[tid], vel[tid],
                                box_of[tid], solo[tid], planted[tid],
                                off[tid], bare[tid], good_world[tid], good_img[tid])
                            bx, by, bw, bh = peak
                            fx, fy = _foot_of(peak)
                            img_vel[tid] = (fx - u, fy - v)
                            img[tid] = (fx, fy)
                            world = _clamp_step(
                                last[tid], _project_point(fx, fy, homographies[t]))
                            vel[tid] = (
                                world[0] - last[tid][0], world[1] - last[tid][1])
                            last[tid] = world
                            box_of[tid] = peak
                            solo[tid] = peak
                            planted[tid] = (fx, fy)
                            off[tid] = 0
                            bare[tid] = 0
                            fwd[t][tid] = dict(
                                team=team_of[tid], world=world, bbox=peak,
                                frac=0.4, width_yd=1.2)
                            good_world[tid] = last[tid]
                            good_img[tid] = img[tid]
                            peak_locked.add(tid)
                            continue
                hud = cy < height * 0.12 or (cy < height * 0.16 and cx < width * 0.18)
                inside = ((-60 <= nu <= width + 60 and -30 <= nv <= height + 40)
                          and not hud)
                if inside:
                    img_vel[tid] = (du * 0.92, dv * 0.92)
                    img[tid] = (nu, nv)
                    off[tid] = 0
                    bare[tid] = bare[tid] + 1 if lost else 0
                    box_of[tid] = held
                    world = _clamp_step(last[tid], _project_point(nu, nv, homographies[t]))
                    vel[tid] = (world[0] - last[tid][0], world[1] - last[tid][1])
                    last[tid] = world
                    fwd[t][tid] = dict(
                        team=team_of[tid], world=world, bbox=held,
                        coasted=True, frac=0.0, width_yd=1.0)
                else:
                    off[tid] += 1
                    bare[tid] += 1
                    if off[tid] > 120:
                        alive[tid] = False
        cur_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        def _put(tid, box, fx, fy):
            u, v = img[tid]
            img_vel[tid] = (fx - u, fy - v)
            img[tid] = (fx, fy)
            world = _clamp_step(last[tid], _project_point(fx, fy, homographies[t]))
            vel[tid] = (world[0] - last[tid][0], world[1] - last[tid][1])
            last[tid] = world
            box_of[tid] = box
            solo[tid] = box
            planted[tid] = (fx, fy)
            bare[tid] = 0
            off[tid] = 0
            fwd[t][tid] = dict(
                team=team_of[tid], world=world, bbox=box,
                frac=0.5, width_yd=1.3)
            good_world[tid] = last[tid]
            good_img[tid] = img[tid]

        def _revert(tid):
            last[tid] = good_world[tid]
            img[tid] = good_img[tid]

        def _note_white(tid, box):
            prev = white_hold.get(tid)
            vx, vy = 4.0, 0.0
            if prev is not None and t - prev[6] <= 3:
                vx = float(box[0] - prev[0])
                vy = float(box[1] - prev[1])
                if abs(vx) > 24 or abs(vy) > 24:
                    vx, vy = prev[4], prev[5]
            white_hold[tid] = [
                float(box[0]), float(box[1]), int(box[2]), int(min(box[3], 76)),
                vx, vy, t, 0]

        def _solo_foot(tid):
            sx, sy, sw, sh = solo[tid]
            du, dv = shift_of.get(tid, (0.0, 0.0))
            return _foot_of((sx + du, sy + dv, sw, sh)), (du, dv)

        for team, mask in (("red", cur_red), ("blue", cur_blue)):
            bodies = _jersey_bodies(mask, shape, cur_hsv)
            taken = set()
            if (main_id is not None and team_of.get(main_id) == team
                    and main_id in fwd[t]):
                main_box = fwd[t][main_id]["bbox"]
                for j, body in enumerate(bodies):
                    if _iou_over_smaller(main_box, body[:4]) > 0.12:
                        taken.add(j)
            owners = {}
            claims = []
            for tid in ids:
                if not alive.get(tid) or team_of[tid] != team or tid == main_id:
                    continue
                if tid not in fwd[t]:
                    continue
                cur = fwd[t][tid]["bbox"]
                (sfx, sfy), _ = _solo_foot(tid)
                for j, body in enumerate(bodies):
                    if j in taken:
                        continue
                    if _iou_over_smaller(cur, body[:4]) <= 0.18:
                        continue
                    dist = float(np.hypot(body[4] - sfx, body[5] - sfy))
                    claims.append((dist, tid, j))
            claims.sort()
            owned_tid = set()
            for _gap, tid, j in claims:
                if tid in owned_tid or j in taken:
                    continue
                owned_tid.add(tid)
                taken.add(j)
                owners[tid] = j
            for tid, j in owners.items():
                bx, by, bw, bh, fx, fy = bodies[j]
                _put(tid, (int(bx), int(by), int(bw), int(bh)), fx, fy)
            for tid in ids:
                if not alive.get(tid) or team_of[tid] != team or tid == main_id:
                    continue
                if tid in owners or tid in peak_locked:
                    continue
                (sfx, sfy), (du, dv) = _solo_foot(tid)
                overlaps_taken = False
                if tid in fwd[t]:
                    cur = fwd[t][tid]["bbox"]
                    overlaps_taken = any(
                        _iou_over_smaller(cur, bodies[j][:4]) > 0.12
                        for j in taken if j < len(bodies))
                # A coast that has lost its jersey must not hop onto the next
                # player it slides past. A track still on a body may follow it.
                limit = 140.0 if bare[tid] == 0 else 60.0
                best, best_d = None, limit
                for j, body in enumerate(bodies):
                    if j in taken:
                        continue
                    dist = float(np.hypot(body[4] - sfx, body[5] - sfy))
                    if dist < best_d:
                        best_d = dist
                        best = j
                if best is not None:
                    taken.add(best)
                    bx, by, bw, bh, fx, fy = bodies[best]
                    _put(tid, (int(bx), int(by), int(bw), int(bh)), fx, fy)
                    continue
                sx, sy, sw, sh = solo[tid]
                # A box on empty grass stays put. Sliding it follows the
                # camera onto a different player.
                if bare[tid] >= 1 and _mostly_grass(cur_hsv, (sx, sy, sw, sh)):
                    du, dv = 0.0, 0.0
                held = (int(round(sx + du)), int(round(sy + dv)), int(sw), int(sh))
                fx, fy = _foot_of(held)
                bare[tid] += 1
                solo[tid] = held
                img[tid] = (fx, fy)
                box_of[tid] = held
                # Missed jersey color and a pile are not reasons to drop the
                # number. Drop it when the feet leave, or when the box has
                # slid onto empty grass.
                off_screen = (fy > height * 0.97 or fy < height * 0.04
                              or fx < -40 or fx > width + 40)
                grass = (not overlaps_taken and bare[tid] > 6
                         and _mostly_grass(cur_hsv, held))
                if tid in fwd[t] and (off_screen or grass):
                    if grass:
                        _revert(tid)
                    del fwd[t][tid]
                elif tid in fwd[t]:
                    world = _clamp_step(last[tid], _project_point(fx, fy, homographies[t]))
                    last[tid] = world
                    fwd[t][tid] = dict(
                        team=team, world=world, bbox=held,
                        coasted=True, frac=0.0, width_yd=1.0)
        red_bodies = _jersey_bodies(cur_red, shape, cur_hsv)
        blue_bodies = _jersey_bodies(cur_blue, shape, cur_hsv)
        republish = []
        far_now = _far_players(cur_hsv)

        def _near_far(tid, team):
            """A distant jersey of this team, just ahead of where he stood."""
            if t == 0 or tid not in fwd[t - 1]:
                return None
            px, py = _foot_of(fwd[t - 1][tid]["bbox"])
            best, best_d = None, 70.0
            for fteam, fbox in far_now:
                if fteam != team:
                    continue
                fx, fy = _foot_of(fbox)
                dist = float(np.hypot(fx - px, fy - py))
                if dist < best_d:
                    best_d = dist
                    best = fbox
            return best

        def _keep_far(tid, team):
            held = _near_far(tid, team)
            if held is None:
                return
            fx, fy = _foot_of(held)
            world = _clamp_step(
                fwd[t][tid]["world"],
                _project_point(fx, fy, homographies[t]))
            republish.append((tid, held, world))
        for tid in list(fwd[t]):
            if tid == main_id:
                box = fwd[t][tid]["bbox"]
                cy = box[1] + box[3] / 2.0
                on_qb = any(
                    _iou_over_smaller(box, b[:4]) > 0.12 for b in blue_bodies)
                # The ring path can sit low. An empty box on the bottom
                # banner cannot. A high box still counts when it is on his
                # body; the scoreboard has no jersey to stand on.
                if (not on_qb) and (
                        cy < height * 0.15
                        or _mostly_grass(cur_hsv, box)
                        or _on_ground_circle(cur_hsv, box)
                        or cy > height * 0.90):
                    _revert(tid)
                    fx, fy = good_img[tid]
                    if fy < height * 0.88:
                        restored = (int(fx) - 32, int(fy) - 150, 64, 150)
                        box_of[tid] = restored
                        solo[tid] = restored
                    del fwd[t][tid]
                continue
            box = fwd[t][tid]["bbox"]
            cx = box[0] + box[2] / 2.0
            cy = box[1] + box[3] / 2.0
            team = team_of[tid]
            bodies = red_bodies if team == "red" else blue_bodies
            if box[1] + box[3] > height * 0.96:
                nh = int(height * 0.96 - box[1])
                if nh >= 36:
                    box = (box[0], box[1], box[2], nh)
                    fwd[t][tid]["bbox"] = box
                    box_of[tid] = box
                else:
                    del fwd[t][tid]
                    continue
            main_box = fwd[t][main_id]["bbox"] if main_id in fwd[t] else None
            on_body = any(
                _iou_over_smaller(box, b[:4]) > 0.12
                and (main_box is None or _iou_over_smaller(b[:4], main_box) < 0.25)
                for b in bodies)
            # The peak was just placed on a far body. The team mask does
            # not see that jersey, so the later gates would call him gone.
            if tid in peak_locked:
                continue
            # A yard line through the BLITZ logo still has a few red pixels.
            # The box is grass. The letters themselves are a lighter blue
            # than a jersey, so a box full of that paint is the logo.
            # The scoreboard corner is never a player.
            if _on_field_logo(cur_hsv, box) or _on_endzone_letters(cur_hsv, box):
                # The box drifted onto the BLITZ paint or the down marker.
                # If last frame he was a player, a short step back onto
                # the white jersey is still him.
                if (_on_field_logo(cur_hsv, box)
                        and not _on_endzone_letters(cur_hsv, box)
                        and t > 0 and tid in fwd[t - 1]):
                    opp = cur_red if team == "blue" else cur_blue
                    cand = _slide_onto_white(cur_hsv, box, opp)
                    if cand is not None:
                        px, py = _foot_of(fwd[t - 1][tid]["bbox"])
                        fx = cand[0] + cand[2] / 2.0
                        fy = cand[1] + cand[3]
                        if float(np.hypot(fx - px, fy - py)) < 90:
                            world = _clamp_step(
                                fwd[t][tid]["world"],
                                _project_point(fx, fy, homographies[t]))
                            republish.append((tid, cand, world))
                            _note_white(tid, cand)
                _revert(tid)
                del fwd[t][tid]
                continue
            if (not on_body) and _mostly_grass(cur_hsv, box):
                # The jersey is the white shape one step to the side.
                # Publish that box, and leave the track memory on the grass.
                if (cy < height * 0.30 and box[2] * box[3] > 1500
                        and not _on_label(box, ring, arrow, shape)):
                    opp = cur_red if team == "blue" else cur_blue
                    slid = _slide_onto_white(cur_hsv, box, opp)
                    if slid is not None:
                        fx = slid[0] + slid[2] / 2.0
                        fy = slid[1] + slid[3]
                        world = _clamp_step(
                            fwd[t][tid]["world"],
                            _project_point(fx, fy, homographies[t]))
                        republish.append((tid, slid, world))
                        _note_white(tid, slid)
                _keep_far(tid, team)
                _revert(tid)
                del fwd[t][tid]
                continue
            # The painted sideline is white and wider than a player.
            # A white jersey is only as wide as the box.
            if not on_body and _on_white_sideline(cur_hsv, box):
                own_mask = cur_red if team == "red" else cur_blue
                opp_mask = cur_blue if team == "red" else cur_red
                if (_jersey_count(own_mask, box) < 24
                        and _jersey_count(opp_mask, box) < 24):
                    _revert(tid)
                    del fwd[t][tid]
                    continue
            # A thin box on the right edge, full of the other team, is that
            # player's edge. A hidden teammate is wider than this.
            if not on_body and box[2] < 50 and cx > width * 0.90:
                own_mask = cur_red if team == "red" else cur_blue
                opp_mask = cur_blue if team == "red" else cur_red
                own_n = _jersey_count(own_mask, box)
                opp_n = _jersey_count(opp_mask, box)
                if opp_n > 300 and own_n < 80:
                    _revert(tid)
                    del fwd[t][tid]
                    continue
            # The down marker stays on screen after the ring is gone.
            # A box whose center is on that chevron is the arrow, not a player.
            # The quarterback is the body underneath it.
            if (arrow is not None and not on_body
                    and _arrow_has_digit(cur_hsv, arrow)):
                ax, ay = float(arrow[0]), float(arrow[1])
                abh = float(arrow[2])
                if abs(cx - ax) < 50 and ay - abh < cy < ay + abh:
                    _revert(tid)
                    del fwd[t][tid]
                    continue
            if (main_id in fwd[t] and tid != main_id
                    and _iou_over_smaller(box, fwd[t][main_id]["bbox"]) > 0.35
                    and not on_body):
                _revert(tid)
                del fwd[t][tid]
                continue
            # The INCOMPLETE graphic and the scoreboard sit in this band.
            # A small far jersey up there is still a player.
            high = cy < height * 0.15 or (cy < height * 0.16 and cx < width * 0.20)
            far_jersey = on_body and box[2] <= 70 and box[3] >= 70
            if high and not far_jersey:
                # The box is high because he ran away, and it still holds
                # his jersey. The scoreboard corner does not.
                own_mask = cur_red if team == "red" else cur_blue
                scoreboard = cx < width * 0.22 and cy < height * 0.16
                if (_jersey_count(own_mask, box) >= 80 and not scoreboard
                        and not _on_field_logo(cur_hsv, box)
                        and not _on_endzone_letters(cur_hsv, box)):
                    stacked = any(
                        team_of.get(other) == team and other != tid
                        and _iou_over_smaller(box, fwd[t][other]["bbox"]) > 0.35
                        and any(_iou_over_smaller(fwd[t][other]["bbox"], b[:4]) > 0.12
                                for b in bodies)
                        for other in fwd[t])
                    if not stacked:
                        continue
                # A far player whose box is a few pixels short of the
                # usual cutoff is still him. Publish that box, and leave
                # the track memory on the last frame so he cannot walk
                # onto the player beside him.
                if (on_body and box[2] <= 70 and box[3] >= 60
                        and t > 0 and tid in fwd[t - 1]
                        and not _on_label(box, ring, arrow, shape)):
                    px, py = _foot_of(fwd[t - 1][tid]["bbox"])
                    fx, fy = _foot_of(box)
                    if float(np.hypot(fx - px, fy - py)) < 80:
                        republish.append((tid, tuple(box), fwd[t][tid]["world"]))
                _keep_far(tid, team)
                _revert(tid)
                del fwd[t][tid]
                continue
            # Empty paint under the scoreboard still has blue pixels.
            # A player in that part of the picture is on a jersey body.
            if (not on_body) and cy < height * 0.25 and cx < width * 0.32:
                _keep_far(tid, team)
                _revert(tid)
                del fwd[t][tid]
                continue
            jmask = cur_red if team == "red" else cur_blue
            still_there = _jersey_count(jmask, box) >= 24
            banner = _on_label(box, ring, arrow, shape) or (
                (not on_body) and (not still_there) and (
                    cy < height * 0.30 or (cy < height * 0.24 and cx < width * 0.40)
                    or box[2] * box[3] < 800))
            if not banner:
                continue
            # A shiny jersey reads white for a frame and the team mask
            # drops it. Remember the box, then put the track memory back
            # so the number cannot walk onto someone else.
            white_keep = False
            saved_box = box
            if (not on_body and not still_there and cy < height * 0.30
                    and box[2] * box[3] > 1500
                    and not _on_label(box, ring, arrow, shape)):

                def _white_frac(bb):
                    x0, y0 = max(0, int(bb[0])), max(0, int(bb[1]))
                    x1 = min(width, int(bb[0] + bb[2]))
                    y1 = min(height, int(bb[1] + bb[3]))
                    if x1 <= x0 or y1 <= y0:
                        return 0.0
                    patch = cur_hsv[y0:y1, x0:x1]
                    return float(((patch[..., 1] < 80) & (patch[..., 2] > 140)).mean())

                best_w = _white_frac(box)
                # The box can sit on the grass just beside a shiny jersey.
                # A short sideways step onto that white is still the player.
                if best_w <= 0.32 and box[3] >= 80:
                    for dx in range(-36, 37, 4):
                        if dx == 0:
                            continue
                        cand = (box[0] + dx, box[1], box[2], box[3])
                        if (_mostly_grass(cur_hsv, cand)
                                or _on_field_logo(cur_hsv, cand)):
                            continue
                        w = _white_frac(cand)
                        if w > best_w:
                            best_w = w
                            saved_box = cand
                white_keep = best_w > 0.32
            world = fwd[t][tid]["world"]
            if white_keep and saved_box is not box:
                fx = saved_box[0] + saved_box[2] / 2.0
                fy = saved_box[1] + saved_box[3]
                world = _clamp_step(world, _project_point(fx, fy, homographies[t]))
            saved = (tid, saved_box, world) if white_keep else None
            if saved is None:
                _keep_far(tid, team)
            _revert(tid)
            del fwd[t][tid]
            if saved is not None:
                republish.append(saved)
        for team, mask in (("red", cur_red), ("blue", cur_blue)):
            bodies = _jersey_bodies(mask, shape, cur_hsv)
            taken = set()
            for tid in ids:
                if tid not in fwd[t] or team_of[tid] != team:
                    continue
                cur = fwd[t][tid]["bbox"]
                for j, body in enumerate(bodies):
                    if _iou_over_smaller(cur, body[:4]) > 0.12:
                        taken.add(j)
            for tid in ids:
                if not alive.get(tid) or team_of[tid] != team or tid == main_id or tid in fwd[t]:
                    continue
                (sfx, sfy), _ = _solo_foot(tid)
                ix, iy = img[tid]
                px, py = planted.get(tid, (ix, iy))
                best, best_d = None, 1e9
                for j, body in enumerate(bodies):
                    if j in taken:
                        continue
                    on_edge = body[0] <= 2 or body[0] + body[2] >= width - 8
                    dist = min(
                        float(np.hypot(body[4] - sfx, body[5] - sfy)),
                        float(np.hypot(body[4] - ix, body[5] - iy)))
                    # A player cut off by the frame keeps the number that
                    # was last standing on that side of the picture.
                    if on_edge:
                        side = body[4] > width * 0.5
                        if (px > width * 0.5) == side:
                            dist = min(dist, float(np.hypot(body[4] - px, body[5] - py)))
                        limit = 450.0
                    else:
                        limit = 200.0
                    # The camera can carry a player across the picture while
                    # his image foot stays behind. Field y walks after the
                    # swing, so a matching column is the link. A body a long
                    # way above or below that column is someone else.
                    if dist >= limit:
                        # A track that was just here keeps the live foot.
                        wp = _world_px(last[tid], homographies[t])
                        if wp is not None and abs(body[4] - wp[0]) < 45:
                            if abs(body[5] - wp[1]) < 260:
                                dist = abs(body[4] - wp[0])
                        # After a long miss that foot has walked off. The last
                        # frame he stood on a jersey is the column to trust.
                        if dist >= limit and miss[tid] >= 20:
                            wp = _world_px(held_world[tid], homographies[t])
                            if wp is not None:
                                dx = abs(body[4] - wp[0])
                                dy = abs(body[5] - wp[1])
                                if (dx < 45 and dy < 160) or (dx < 200 and dy < 80):
                                    dist = dx
                        # A short gap after a real sighting is the camera
                        # zooming him larger, so the column can sit a bit low.
                        if dist >= limit and 0 < miss[tid] < 20:
                            wp = _world_px(last[tid], homographies[t])
                            if wp is not None:
                                dx = abs(body[4] - wp[0])
                                dy = abs(body[5] - wp[1])
                                if ((dx < 140 and dy < 190)
                                        or (dx < 200 and dy < 80)):
                                    dist = dx
                        # The other player in a pile can sit farther across
                        # the picture than the live foot. Require a teammate
                        # already on the neighboring body, a short gap, and
                        # no closer missing teammate.
                        if dist >= limit and 1 <= miss[tid] <= 10:
                            wp = _world_px(held_world[tid], homographies[t])
                            if wp is not None:
                                dx = abs(body[4] - wp[0])
                                dy = abs(body[5] - wp[1])
                                piled = False
                                for j2, b2 in enumerate(bodies):
                                    if j2 not in taken:
                                        continue
                                    if (abs(b2[4] - body[4]) < 160
                                            and abs(b2[5] - body[5]) < 120
                                            and abs(b2[4] - wp[0]) < 220
                                            and abs(b2[5] - wp[1]) < 140):
                                        piled = True
                                        break
                                nearer = False
                                if piled and dx < 300 and dy < 90:
                                    for other in ids:
                                        if (other == tid or other == main_id
                                                or other in fwd[t]
                                                or not alive.get(other)
                                                or team_of[other] != team):
                                            continue
                                        ow = _world_px(
                                            held_world[other], homographies[t])
                                        if ow is None:
                                            continue
                                        if (abs(body[4] - ow[0]) + 20 < dx
                                                and abs(body[5] - ow[1]) < 160):
                                            nearer = True
                                            break
                                if piled and dx < 300 and dy < 90 and not nearer:
                                    dist = min(dx, limit - 1)
                    if dist < limit and dist < best_d:
                        best_d = dist
                        best = j
                if best is None:
                    continue
                taken.add(best)
                bx, by, bw, bh, fx, fy = bodies[best]
                box = (int(bx), int(by), int(bw), int(bh))
                if _on_label(box, ring, arrow, shape):
                    continue
                _put(tid, box, fx, fy)
            # One wide blob is two players who just overlapped. Keep the
            # number that was on his own foot last frame, on his side of
            # that blob, for this frame only.
            for j, body in enumerate(bodies):
                if j not in taken:
                    continue
                bx, by, bw, bh, bcx, bfy = body
                if bw < 120:
                    continue
                owners = [tid for tid in ids
                          if tid in fwd[t] and team_of[tid] == team
                          and _iou_over_smaller(fwd[t][tid]["bbox"], body[:4]) > 0.12]
                if len(owners) != 1:
                    continue
                owner = owners[0]
                ox, oy = _foot_of(fwd[t][owner]["bbox"])
                inside = []
                for tid in ids:
                    if (tid == owner or tid == main_id or tid in fwd[t]
                            or not alive.get(tid) or team_of[tid] != team
                            or miss[tid] != 0):
                        continue
                    fx, fy = img[tid]
                    if not (bx - 70 <= fx <= bx + bw + 70
                            and by - 40 <= fy <= by + bh + 40):
                        continue
                    if abs(fx - ox) + abs(fy - oy) > 180:
                        continue
                    inside.append(tid)
                if len(inside) != 1:
                    continue
                tid = inside[0]
                fx, fy = img[tid]
                if abs(fx - ox) < 28:
                    fx = ox + (36 if fx >= ox else -36)
                fx = min(max(fx, bx + 24), bx + bw - 24)
                fy = min(max(fy, by + bh * 0.55), by + bh - 6)
                box = (int(fx - 28), int(fy - 100), 56, 104)
                if (_mostly_grass(cur_hsv, box) or _on_field_logo(cur_hsv, box)
                        or _on_label(box, ring, arrow, shape)):
                    continue
                _put(tid, box, fx, fy)
        for tid in ids:
            if alive.get(tid) and bare[tid] == 0:
                cruise[tid] = img_vel[tid]
            if tid in fwd[t]:
                miss[tid] = 0
                box = fwd[t][tid]["bbox"]
                bodies = red_bodies if team_of[tid] == "red" else blue_bodies
                on_body = any(
                    _iou_over_smaller(box, b[:4]) > 0.12 for b in bodies)
                if on_body:
                    held_world[tid] = last[tid]
            elif alive.get(tid):
                miss[tid] += 1
        # The white jersey leaves the grass anchor during a tackle, then
        # shows up again once he is clear. Follow that torso from the last
        # white box. A red number already sitting on it is the wrong player.
        followed = {}
        for tid in list(white_hold):
            if not alive.get(tid) or tid in fwd[t]:
                continue
            if any(item[0] == tid for item in republish):
                continue
            hold = white_hold[tid]
            if t - hold[6] > 36:
                del white_hold[tid]
                continue
            opp = cur_red if team_of[tid] == "blue" else cur_blue
            radius = 36 if hold[7] else 80
            found = _lock_white(
                cur_hsv, opp, (hold[0] + hold[4], hold[1] + hold[5], hold[2], hold[3]),
                radius, shape)
            if found is None:
                hold[0] += hold[4]
                hold[1] += hold[5]
                continue
            fx = found[0] + found[2] / 2.0
            fy = found[1] + found[3]
            world = _clamp_step(last[tid], _project_point(fx, fy, homographies[t]))
            last[tid] = world
            republish.append((tid, found, world))
            hold[4] = float(np.clip(found[0] - hold[0], -16, 16))
            hold[5] = float(np.clip(found[1] - hold[1], -16, 16))
            hold[0], hold[1] = float(found[0]), float(found[1])
            hold[2], hold[3] = int(found[2]), int(found[3])
            hold[6] = t
            hold[7] = 1
            followed[tid] = found
        for tid, found in followed.items():
            for other in list(fwd[t]):
                if team_of.get(other) == team_of.get(tid):
                    continue
                ob = fwd[t][other]["bbox"]
                if _iou_over_smaller(ob, found) < 0.25:
                    continue
                own = cur_red if team_of[other] == "red" else cur_blue
                bodies = red_bodies if team_of[other] == "red" else blue_bodies
                on_own = any(_iou_over_smaller(ob, b[:4]) > 0.12 for b in bodies)
                if on_own:
                    continue
                _revert(other)
                del fwd[t][other]
        # A red number on a white jersey in the distance is the blue player
        # who ran away. The red beside him still has his own jersey.
        for fteam, fbox in far_now:
            if fteam != "blue":
                continue
            for other in list(fwd[t]):
                if team_of.get(other) != "red" or other not in fwd[t]:
                    continue
                ob = fwd[t][other]["bbox"]
                if _iou_over_smaller(ob, fbox) < 0.25:
                    continue
                if any(_iou_over_smaller(ob, b[:4]) > 0.12 for b in red_bodies):
                    continue
                _revert(other)
                del fwd[t][other]
        for tid, box, world in republish:
            if tid not in fwd[t]:
                fwd[t][tid] = dict(
                    team=team_of[tid], world=world, bbox=box,
                    coasted=True, frac=0.4, width_yd=1.0)
        # A far-body guess that landed on a teammate already boxed is
        # dropped, and his memory goes back to where he was.
        for tid in list(peak_locked):
            if tid not in fwd[t]:
                continue
            box = fwd[t][tid]["bbox"]
            for other in fwd[t]:
                if (other == tid or other in peak_locked
                        or team_of.get(other) != team_of.get(tid)):
                    continue
                if _iou_over_smaller(box, fwd[t][other]["bbox"]) <= 0.30:
                    continue
                (iv, im, la, ve, bo, so, pl, of, ba, gw, gi) = peak_undo[tid]
                img_vel[tid], img[tid] = iv, im
                last[tid], vel[tid] = la, ve
                box_of[tid], solo[tid], planted[tid] = bo, so, pl
                off[tid], bare[tid] = of, ba
                good_world[tid], good_img[tid] = gw, gi
                del fwd[t][tid]
                break
        # A number that just came back on top of a teammate who never left
        # is the same body. The teammate keeps it.
        for tid in list(fwd[t]):
            if t == 0 or tid not in fwd[t]:
                continue
            box = fwd[t][tid]["bbox"]
            # Only the far end of the picture. A pile closer to the camera
            # is two players, and both numbers stay.
            if box[1] + box[3] / 2.0 > height * 0.28:
                continue
            # He stepped onto a teammate who has been here the whole time.
            # A one-frame gap is enough for the box to stick, so look back
            # a few frames.
            gone = any(t - k >= 0 and tid not in fwd[t - k] for k in range(1, 5))
            if not gone:
                continue
            for other in fwd[t]:
                if other == tid or team_of.get(other) != team_of.get(tid):
                    continue
                stayed = all(t - k >= 0 and other in fwd[t - k] for k in range(1, 4))
                if not stayed:
                    continue
                if _iou_over_smaller(box, fwd[t][other]["bbox"]) <= 0.45:
                    continue
                if tid in held_at:
                    (im, la, ve, iv, bo, so, pl, of, ba, gw, gi) = held_at[tid]
                    img[tid], last[tid], vel[tid], img_vel[tid] = im, la, ve, iv
                    box_of[tid], solo[tid], planted[tid] = bo, so, pl
                    off[tid], bare[tid] = of, ba
                    good_world[tid], good_img[tid] = gw, gi
                del fwd[t][tid]
                break
        if (main_id is not None and main_id not in fwd[t]
                and t > 0 and main_id in fwd[t - 1]):
            px, py = _foot_of(fwd[t - 1][main_id]["bbox"])
            if py > height * 0.88:
                best, best_d = None, 190.0
                for body in blue_bodies:
                    dist = float(np.hypot(body[4] - px, body[5] - py))
                    if dist >= best_d or body[5] > height * 0.97:
                        continue
                    cand = (int(body[0]), int(body[1]), int(body[2]), int(body[3]))
                    # Touching a teammate is not the same as standing on him.
                    # Only a box whose center is inside this body owns it.
                    owned = False
                    for i in list(fwd[t]):
                        ob = fwd[t][i]["bbox"]
                        ocx = ob[0] + ob[2] / 2.0
                        ocy = ob[1] + ob[3] / 2.0
                        if (cand[0] <= ocx <= cand[0] + cand[2]
                                and cand[1] <= ocy <= cand[1] + cand[3]):
                            owned = True
                            break
                    if owned:
                        continue
                    if _on_field_logo(cur_hsv, cand) or _mostly_grass(cur_hsv, cand):
                        continue
                    best_d = dist
                    best = (cand, float(body[4]), float(body[5]))
                if best is not None:
                    cand, cx, fy = best
                    fwd[t][main_id] = dict(
                        team=team_of[main_id],
                        world=_clamp_step(
                            last[main_id],
                            _project_point(cx, fy, homographies[t])),
                        bbox=cand, frac=0.5, width_yd=1.2)
        # Two numbers on one pair of feet. The one whose prediction was
        # already there keeps the body. The other keeps the step he was
        # making, so the next frame cannot carry him off on his teammate.
        if t > 0:
            feet = []
            for tid in fwd[t]:
                fx, fy = _foot_of(fwd[t][tid]["bbox"])
                feet.append((tid, fx, fy))
            parked = []
            for i, (tid, fx, fy) in enumerate(feet):
                if tid not in pred_of or tid not in fwd[t - 1]:
                    continue
                px, py = pred_of[tid]
                own = float(np.hypot(fx - px, fy - py))
                for other, ox, oy in feet[i + 1:]:
                    if (team_of[other] != team_of[tid] or other not in pred_of
                            or other not in fwd[t - 1]):
                        continue
                    if float(np.hypot(fx - ox, fy - oy)) > 12:
                        continue
                    opx, opy = pred_of[other]
                    d_oth = float(np.hypot(fx - opx, fy - opy))
                    if own > d_oth + 10:
                        parked.append(tid)
                    elif d_oth > own + 10:
                        parked.append(other)
            for tid in set(parked):
                if tid not in fwd[t] or tid not in pred_of or tid not in mem_of:
                    continue
                px, py = pred_of[tid]
                mx, my = mem_of[tid]
                prev = fwd[t - 1][tid]
                ox, oy = _foot_of(prev["bbox"])
                box = (int(round(prev["bbox"][0] + px - ox)),
                       int(round(prev["bbox"][1] + py - oy)),
                       int(prev["bbox"][2]), int(prev["bbox"][3]))
                world = _clamp_step(
                    prev["world"], _project_point(px, py, homographies[t]))
                img[tid] = (float(px), float(py))
                img_vel[tid] = (float(px - mx), float(py - my))
                last[tid] = world
                good_img[tid] = img[tid]
                good_world[tid] = world
                box_of[tid] = box
                solo[tid] = box
                fwd[t][tid] = dict(
                    team=team_of[tid], world=world, bbox=box,
                    coasted=True, frac=0.4, width_yd=1.0)
        prev_gray = gray
        prev_field = _field_mask(frame)
        prev_red, prev_blue = cur_red, cur_blue
    cap.release()
    print(f"body flow {flow_hits}/{flow_tries}")
    # A player who has left the picture stays gone. Do not coast them back.

    # Walk backward so the next frame is already decided and can correct
    # the frame before it. Frame 0 stays the roster.
    final = [dict() for _ in range(n)]
    final[0] = {r["id"]: dict(r) for r in roster}
    final[n - 1] = dict(fwd[n - 1])
    for t in range(n - 2, 0, -1):
        preds = []
        for tid in ids:
            fut = final[t + 1].get(tid)
            if fut is None:
                continue
            fut2 = final[t + 2].get(tid) if t + 2 < n else None
            if fut2 is not None:
                vx = fut2["world"][0] - fut["world"][0]
                vy = fut2["world"][1] - fut["world"][1]
            else:
                vx = vy = 0.0
            preds.append((tid,
                          (fut["world"][0] - vx, fut["world"][1] - vy),
                          team_of[tid], 4.0))
        got = _assign_gated(preds, per_frame[t], homographies[t]) if preds else {}
        for tid in ids:
            fut = final[t + 1].get(tid)
            fdet = fwd[t].get(tid)
            bdet = got.get(tid)
            def _fpix(det):
                return _foot_of(det["bbox"]) if det is not None else None
            if fdet is not None:
                chosen = fdet
            elif bdet is not None and fut is not None:
                ff = _foot_of(fut["bbox"])
                bf = _foot_of(bdet["bbox"])
                if (float(np.hypot(bf[0] - ff[0], bf[1] - ff[1])) <= 160
                        and (tid == main_id or not _on_label(
                            bdet["bbox"],
                            rings[t] if t < len(rings) else None,
                            arrows[t] if t < len(arrows) else None, shape))):
                    chosen = bdet
                else:
                    chosen = None
            else:
                chosen = None
            if chosen is not None:
                final[t][tid] = chosen

    # Position on this frame is a blend of the detection and the position
    # predicted from the previous frame (and, when seen, the next one).
    # Confidence is how well those agree. No detection means no dot.
    fused = [dict() for _ in range(n)]
    pos = {r["id"]: r["world"] for r in roster}
    vel = {r["id"]: (0.0, 0.0) for r in roster}
    conf = {r["id"]: 1.0 for r in roster}
    for r in roster:
        fused[0][r["id"]] = dict(pos=r["world"], conf=1.0,
                                 bbox=r["bbox"], world=r["world"])
    for t in range(1, n):
        for tid in ids:
            det = final[t].get(tid)
            if det is None:
                vx, vy = vel[tid]
                vel[tid] = (vx * 0.5, vy * 0.5)
                conf[tid] *= 0.7
                continue
            px, py = pos[tid]
            if det.get("coasted"):
                nx, ny = det["world"]
                conf[tid] = max(0.35, conf[tid] * 0.92)
                vel[tid] = (nx - px, ny - py)
                pos[tid] = (nx, ny)
                fused[t][tid] = dict(pos=(float(nx), float(ny)),
                                     conf=conf[tid], bbox=det["bbox"],
                                     world=(float(nx), float(ny)))
                continue
            vx, vy = vel[tid]
            pred = (px + vx, py + vy)
            meas = det["world"]
            gap = _dist(meas, pred)
            meas_w = _agree(gap)
            pred_w = conf[tid] * 0.85
            # a weak or stale prediction must not drag the player backward
            if pred_w < 0.3 or gap > 3.0:
                pred_w = 0.0
                meas_w = 1.0
            weights = [(pred, pred_w), (meas, max(meas_w, 0.2))]
            fut = final[t + 1].get(tid) if t + 1 < n else None
            if fut is not None and _dist(fut["world"], meas) <= 3.0:
                weights.append((fut["world"],
                                0.4 * _agree(_dist(fut["world"], meas))))
            wsum = sum(w for _, w in weights) or 1.0
            nx = sum(p[0] * w for p, w in weights) / wsum
            ny = sum(p[1] * w for p, w in weights) / wsum
            agree = _agree(gap) if pred_w > 0 else 0.6
            if fut is not None:
                agree = (0.7 * agree
                         + 0.3 * _agree(_dist(fut["world"], meas)))
            conf[tid] = float(min(1.0, 0.2 + 0.8 * agree))
            vel[tid] = (nx - px, ny - py)
            pos[tid] = (nx, ny)
            fused[t][tid] = dict(pos=(float(nx), float(ny)), conf=conf[tid],
                                 bbox=det["bbox"], world=meas)

    for t in range(n):
        for tid in list(fused[t]):
            pts = []
            lo, hi = max(0, t - 6), min(n, t + 7)
            for k in range(lo, hi):
                item = fused[k].get(tid)
                if item is not None:
                    pts.append(item["pos"])
            if len(pts) < 3:
                fused[t][tid]["conf"] = min(fused[t][tid]["conf"], 0.4)
                continue
            med = np.median(np.array(pts, np.float64), axis=0)
            gap = _dist(fused[t][tid]["pos"], med)
            if gap > 8.0:
                fused[t][tid]["pos"] = (float(med[0]), float(med[1]))
                gap = 0.0
            fused[t][tid]["conf"] = _window_conf(fused, tid, t, n)
    for t in range(1, n):
        for tid in list(fused[t]):
            prev = fused[t - 1].get(tid)
            if prev is None:
                continue
            px, py = prev["pos"]
            nx, ny = fused[t][tid]["pos"]
            step = float(np.hypot(nx - px, ny - py))
            if step > 6.0:
                scale = 6.0 / step
                fused[t][tid]["pos"] = (px + (nx - px) * scale,
                                        py + (ny - py) * scale)

    tracks = []
    for t in range(n):
        frame_tracks = []
        for tid in ids:
            if tid not in fused[t]:
                continue
            item = fused[t][tid]
            trail = []
            k = t
            while k >= max(0, t - 14) and tid in fused[k]:
                trail.append(fused[k][tid]["pos"])
                k -= 1
            trail.reverse()
            frame_tracks.append(dict(
                id=tid, team=team_of[tid], pos=item["pos"], conf=item["conf"],
                trail=trail, bbox=item["bbox"], world=item["world"]))
        tracks.append(frame_tracks)
    return tracks


def _agree(dist):
    """1 when a detection lands on the predicted spot, near 0 by 4 yards."""
    return float(np.exp(-0.5 * (dist / 1.5) ** 2))


# ---------------------------------------------------------------- rendering
def _label(img, text, org, scale):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def render_aligned(frame, tracks, dets, status, field_map):
    """Bird's eye panel facing the same way as the camera.

    The whole field stays fixed for the entire clip. Only the dots move.
    Measured on frame 0: image-up is decreasing field x (toward the end
    zone the camera looks at) and image-left is decreasing field y (far
    sideline). A counter-clockwise quarter turn plus a vertical flip puts
    that end zone at the top and the far sideline on the left, so forward
    in the clip is north on the map.
    """
    h, w = frame.shape[:2]
    left = frame.copy()
    for d in dets:
        x0, y0, bw, bh = d["bbox"]
        c = TEAM_BGR.get(d["team"], (0, 255, 255))
        cv2.rectangle(left, (x0, y0), (x0 + bw, y0 + bh), c, 2)
        foot = (int(x0 + bw / 2), int(y0 + bh))
        cv2.circle(left, foot, 5, c, -1)
        if d.get("id") is not None:
            text = str(d["id"])
            if d.get("conf") is not None:
                text = f"{d['id']} {d['conf']:.1f}"
            _label(left, text, (x0, max(18, y0 - 6)), 0.6)
    cv2.putText(left, status.upper(), (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                STATUS_BGR.get(status, (255, 255, 255)), 3)
    left = cv2.resize(left, (int(w * LEFT_SCALE), int(h * LEFT_SCALE)))
    if field_map is None:
        return left

    def _on_map(world):
        mx, my = to_map(world)
        return (int(mx), int(my))

    fmap = field_map.copy()
    for tr in tracks:
        c = TEAM_BGR.get(tr["team"], (0, 255, 255))
        pts = list(tr["trail"])
        for i in range(1, len(pts)):
            a = _on_map(pts[i - 1])
            b = _on_map(pts[i])
            fade = 0.3 + 0.7 * i / len(pts)
            cv2.line(fmap, a, b, tuple(int(ch * fade) for ch in c), 3)
        cv2.circle(fmap, _on_map(tr["pos"]), 14, c, -1)
        cv2.circle(fmap, _on_map(tr["pos"]), 14, (255, 255, 255), 2)
    # (mx, my) -> (my, mx): north (small x) at the top, far sideline at left
    fmap = cv2.rotate(fmap, cv2.ROTATE_90_COUNTERCLOCKWISE)
    fmap = cv2.flip(fmap, 0)
    scale = left.shape[0] / fmap.shape[0]
    fmap = cv2.resize(fmap, (int(fmap.shape[1] * scale), left.shape[0]))
    for tr in tracks:
        mx, my = _on_map(tr["pos"])
        _label(fmap, str(tr["id"]), (my * scale + 8, mx * scale - 6), 0.45)
    # north is the direction the camera faces
    cx = fmap.shape[1] // 2
    cv2.arrowedLine(fmap, (cx, 46), (cx, 16), (255, 255, 255), 2, tipLength=0.35)
    _label(fmap, "N", (cx + 8, 28), 0.6)
    return np.hstack([left, fmap])


# ---------------------------------------------------------------- main
def make_writer(shape):
    h, w = shape[:2]
    for stale in (OUT_VIDEO, OUT_VIDEO.replace(".mp4", ".avi")):
        if os.path.exists(stale):
            os.remove(stale)
    for fourcc, path in (("avc1", OUT_VIDEO),
                         ("MJPG", OUT_VIDEO.replace(".mp4", ".avi"))):
        wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc),
                             30.0, (w, h))
        if wr.isOpened():
            print(f"video writer: {fourcc} -> {path}")
            return wr, path
    raise RuntimeError("no video writer available")


# Field tracking follows nfl-blitz-opencv/src/tracking.py and track_video.py:
# fixed field landmarks are followed with pyramidal Lucas-Kanade, a
# forward-backward check rejects points that do not return, and the
# homography is refit from the landmarks that survive.
_FIELD_FLOW = dict(
    winSize=(31, 31),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def track_screen_points(previous_gray, current_gray, previous_points):
    """Same check as nfl-blitz-opencv/src/tracking.py:track_screen_points."""
    previous_points = previous_points.reshape(-1, 1, 2).astype(np.float32)
    current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, previous_points, None, **_FIELD_FLOW)
    if current_points is None:
        return None, None
    backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, current_points, None, **_FIELD_FLOW)
    if backward_points is None:
        return None, None
    forward_status = forward_status.reshape(-1).astype(bool)
    backward_status = backward_status.reshape(-1).astype(bool)
    backward_difference = np.linalg.norm(
        previous_points.reshape(-1, 2) - backward_points.reshape(-1, 2),
        axis=1)
    valid = forward_status & backward_status & (backward_difference < 2.0)
    return current_points.reshape(-1, 2), valid


def _seed_landmarks(A, shape):
    """Yard-line by cross-field grid, kept where it falls inside the picture."""
    h, w = shape[:2]
    world, screen = [], []
    for x in np.arange(10.0, 80.0, 5.0):
        for y in np.arange(-4.0, 40.0, 4.0):
            px = _feet_px((float(x), float(y)), A)
            if px is None:
                continue
            u, v = px
            if 30 < u < w - 30 and 0.08 * h < v < 0.84 * h:
                world.append((float(x), float(y)))
                screen.append((u, v))
    if not world:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    return np.array(world, np.float32), np.array(screen, np.float32)


def _homography_step_ok(prev_A, new_A, shape):
    """A field point should not leap across the picture in one frame."""
    height, width = shape[:2]
    worlds = np.array([[(x, y)] for x in np.linspace(20, 80, 7)
                       for y in np.linspace(0, 36, 5)], np.float32)
    try:
        inv0 = np.linalg.inv(prev_A).astype(np.float32)
        inv1 = np.linalg.inv(new_A).astype(np.float32)
    except np.linalg.LinAlgError:
        return False
    p0 = cv2.perspectiveTransform(worlds, inv0)[:, 0, :]
    p1 = cv2.perspectiveTransform(worlds, inv1)[:, 0, :]
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        return False
    on = ((p0[:, 0] > 0) & (p0[:, 0] < width)
          & (p0[:, 1] > 0.05 * height) & (p0[:, 1] < 0.92 * height))
    if int(on.sum()) < 4:
        return True
    shift = np.linalg.norm(p1[on] - p0[on], axis=1)
    return float(np.median(shift)) <= 28.0


def _homography_image_to_world(world, screen):
    """Their fit maps field -> screen. Ours maps screen -> field."""
    H, mask = cv2.findHomography(world, screen, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return None, None
    A = np.linalg.inv(H)
    A = A / A[2, 2]
    return A, mask.reshape(-1).astype(bool)


def track_field_homographies(A0, n):
    """Carry the frame-0 field homography by tracking its landmarks."""
    cap = cv2.VideoCapture(CLIP)
    ok, prev = _take(cap)
    if not ok:
        raise RuntimeError("could not read clip for field tracking")
    shape = prev.shape
    world, screen = _seed_landmarks(A0, shape)
    print(f"field landmarks at snap: {len(world)}")
    As = [np.array(A0, np.float64)]
    status = ["fresh"]
    prev_gray = cv2.GaussianBlur(cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    A = np.array(A0, np.float64)
    held = 0
    for t in range(1, n):
        ok, frame = _take(cap)
        if not ok:
            break
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        updated = False
        if len(screen) >= 4:
            cur, valid = track_screen_points(prev_gray, gray, screen)
            if cur is not None and int(valid.sum()) >= 4:
                new_A, inliers = _homography_image_to_world(world[valid], cur[valid])
                if (new_A is not None and int(inliers.sum()) >= 4
                        and _homography_step_ok(A, new_A, shape)):
                    A = new_A
                    world = world[valid][inliers]
                    screen = cur[valid][inliers]
                    updated = True
        if len(world) < 12:
            extra_w, extra_s = _seed_landmarks(A, shape)
            if len(extra_w):
                keep_w, keep_s = [world], [screen]
                for p, q in zip(extra_w, extra_s):
                    if len(screen) == 0 or np.linalg.norm(screen - q, axis=1).min() > 36:
                        keep_w.append(p.reshape(1, 2))
                        keep_s.append(q.reshape(1, 2))
                world = np.vstack(keep_w) if len(world) else extra_w
                screen = np.vstack(keep_s) if len(screen) else extra_s
        if not updated:
            held += 1
            if len(world):
                reproj = [_feet_px(tuple(p), A) for p in world]
                screen = np.array(
                    [p if p is not None else (0.0, 0.0) for p in reproj],
                    np.float32)
        As.append(A)
        status.append("chained" if updated else "stale")
        prev_gray = gray
        if t % 40 == 0:
            print(f"field {t}/{n} points={len(world)} updated={updated}")
    cap.release()
    print(f"field tracking held the previous homography on {held}/{n - 1} frames")
    return As, status


def load_homographies(n):
    if not os.path.exists(HOMOGRAPHY_CACHE):
        return None
    z = np.load(HOMOGRAPHY_CACHE)
    if z["A"].shape[0] != n:
        return None
    print(f"loaded homographies from {HOMOGRAPHY_CACHE}")
    return list(z["A"]), [str(s) for s in z["status"]]


def save_homographies(A, status):
    np.savez(HOMOGRAPHY_CACHE, A=np.stack(A), status=np.array(status))
    print(f"cached homographies to {HOMOGRAPHY_CACHE}")


def main():
    cap = cv2.VideoCapture(CLIP)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"frames: {n}", flush=True)
    cached = load_homographies(n)
    if cached is not None:
        cap.release()
        render_pass(n, cached[0], cached[1])
        return

    ok, frame = _take(cap)
    cap.release()
    if not ok:
        print("FATAL: could not read the opening frame")
        return
    try:
        H, info = fit_frame(frame)
    except Exception as e:
        H, info = None, str(e)
    if H is None:
        print(f"FATAL: no fresh fit at reference frame ({info})")
        return
    anchor = np.array([[1, 0, REF_ANCHOR], [0, 1, 0], [0, 0, 1.0]])
    A, status = track_field_homographies(anchor @ H, n)
    n_fresh = sum(1 for s in status if s == "fresh")
    n_chained = sum(1 for s in status if s == "chained")
    n_stale = sum(1 for s in status if s == "stale")
    print(f"status: fresh={n_fresh} chained={n_chained} stale={n_stale}")
    save_homographies(A, status)
    render_pass(n, A, status)


def _supported(per_frame, win=6):
    """A body has to show up across the frames around this one.
    One-frame paint does not get a number."""
    n = len(per_frame)
    kept = []
    for t, dets in enumerate(per_frame):
        row = []
        for det in dets:
            support = 0
            for k in range(max(0, t - win), min(n, t + win + 1)):
                if k == t:
                    continue
                for other in per_frame[k]:
                    if other["team"] != det["team"]:
                        continue
                    if _dist(other["world"], det["world"]) < 3.2:
                        support += 1
                        break
                    ax = det["bbox"][0] + det["bbox"][2] / 2.0
                    ay = det["bbox"][1] + det["bbox"][3]
                    bx = other["bbox"][0] + other["bbox"][2] / 2.0
                    by = other["bbox"][1] + other["bbox"][3]
                    if np.hypot(ax - bx, ay - by) < 60:
                        support += 1
                        break
            if support >= 2 or det.get("main"):
                row.append(det)
        kept.append(row)
    return kept


def render_pass(n, A, status):
    # Detect every frame, then lock the snap roster and carry those
    # numbers forward and backward. The blue ring is followed with the
    # same forward-backward point check as the field landmarks.
    global _LOOSE_LINE
    _LOOSE_LINE = None
    cap = cv2.VideoCapture(CLIP)
    per_frame = []
    rings = []
    arrows = []
    shape = None
    hint = None
    prev_r = None
    prev_gray = None
    arrow_pos = None
    arrow_gap = 99
    lost = 0
    i = 0
    playable = []
    while True:
        ok, frame = _take(cap)
        if not ok:
            break
        if shape is None:
            shape = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not _on_grass(frame):
            per_frame.append([])
            rings.append(None)
            arrows.append(None)
            playable.append(False)
            prev_gray = gray
            lost += 1
            arrow_gap += 1
            if i % 40 == 0:
                print(f"detect {i}/{n} cands=0 ring=False arrow=None", flush=True)
            i += 1
            continue
        playable.append(True)
        if hint is not None and prev_gray is not None and prev_r is not None:
            moved = _flow_ring(prev_gray, gray, dict(cx=hint[0], cy=hint[1], r=prev_r))
            if moved is not None:
                hint = moved
        use_hint = hint if lost < 40 else None
        dets, ring = frame_candidates(frame, A[i], use_hint, prev_r if use_hint else None)
        if ring is not None:
            hint = (ring["cx"], ring["cy"])
            prev_r = ring["r"]
            lost = 0
        else:
            lost += 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hits = _arrow_candidates(hsv)
        choice = _pick_arrow(hits, arrow_pos, ring, frame.shape, arrow_gap)
        if choice is None and _LOOSE_LINE:
            choice = _marked_arrow(hsv, hits, frame.shape)
        if choice is not None:
            arrow_pos = (choice["cx"], choice["cy"], choice["bh"])
            arrow_gap = 0
        elif arrow_pos is not None and prev_gray is not None:
            moved = _flow_arrow(prev_gray, gray, arrow_pos)
            if moved is not None:
                arrow_pos = moved
                arrow_gap += 1
            else:
                arrow_gap += 1
        else:
            arrow_gap += 1
        per_frame.append(dets)
        rings.append(ring)
        arrows.append(arrow_pos if arrow_gap < 18 else None)
        prev_gray = gray
        if i % 40 == 0:
            where = None if arrows[-1] is None else (
                round(arrows[-1][0]), round(arrows[-1][1]))
            print(f"detect {i}/{n} cands={len(dets)} ring={ring is not None} arrow={where}",
                  flush=True)
        i += 1
    cap.release()
    per_frame = _supported(per_frame)
    roster = select_roster(per_frame[0])
    per_frame, removed = drop_stationary(per_frame)
    print(f"removed {removed} stationary decal detections")
    n_red = sum(1 for r in roster if r["team"] == "red")
    n_blue = sum(1 for r in roster if r["team"] == "blue")
    print(f"opening roster: {n_red} red, {n_blue} blue")
    for r in roster:
        flag = " main" if r.get("main") else ""
        print(f"  {r['id']:2} {r['team']:5} ({r['world'][0]:5.1f}, {r['world'][1]:5.1f}){flag}")
    tracked = track_bidirectional(per_frame, roster, A, shape, rings, arrows)
    for t, ok in enumerate(playable):
        if not ok:
            tracked[t] = []

    os.makedirs(SAMPLE_DIR, exist_ok=True)
    field_map = cv2.imread(FIELD_MAP) if FIELD_MAP else None
    cap = cv2.VideoCapture(CLIP)
    writer, out_path = None, OUT_VIDEO
    rows = []
    i = 0
    while True:
        ok, frame = _take(cap)
        if not ok:
            break
        active = tracked[i]
        for tr in active:
            rows.append([i, tr["id"], tr["team"],
                         round(tr["pos"][0], 2), round(tr["pos"][1], 2),
                         round(tr["conf"], 2)])
        combo = render_aligned(frame, active, active, status[i], field_map)
        if writer is None:
            writer, out_path = make_writer(combo.shape)
        writer.write(combo)
        if i in SAMPLE_AT:
            cv2.imwrite(f"{SAMPLE_DIR}/composite_{i:04d}.png", combo)
        if i % 40 == 0:
            print(f"pass2 {i}/{n} tracks={len(active)}")
        i += 1
    writer.release()

    with open(OUT_CSV, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame", "id", "team", "x_yd", "y_yd", "conf"])
        wcsv.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")

    # verify playback
    chk = cv2.VideoCapture(out_path)
    nchk = int(chk.get(cv2.CAP_PROP_FRAME_COUNT))
    ok1, _ = chk.read()
    chk.set(cv2.CAP_PROP_POS_FRAMES, nchk - 1)
    ok2, _ = chk.read()
    print(f"playback check: frames={nchk} first={ok1} last={ok2}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Track the field and the players in a gameplay video.")
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--out-dir", default="data/output",
                        help="Directory for the video, CSV, and homography cache.")
    parser.add_argument("--anchor", type=float, default=-15.0,
                        help="Yards added on the reference frame so x matches the field.")
    parser.add_argument("--field-map",
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "field_map.png"),
                        help="Top-down field image for the vertical map panel.")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    CLIP = args.video
    OUT_VIDEO = os.path.join(args.out_dir, "tracked_players.mp4")
    OUT_CSV = os.path.join(args.out_dir, "positions.csv")
    HOMOGRAPHY_CACHE = os.path.join(args.out_dir, "homography.npz")
    SAMPLE_DIR = os.path.join(args.out_dir, "frames")
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    REF_ANCHOR = args.anchor
    FIELD_MAP = args.field_map
    main()
