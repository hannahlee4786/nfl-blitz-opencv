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
import json
import os
import sys
import cv2
import numpy as np

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, PROJECT_ROOT)
import field_homography as fh
from track_clip import (fit_frame, inter_frame_h,
                        TEAM_BGR, STATUS_BGR, LEFT_SCALE, to_map)
from project_players import detect_players
from src.team_classifier import (ColorPrototype, build_color_prototype,
                                 classify_jersey_region, crop_region,
                                 load_calibration_report, prototype_distance)

CLIP = "Clip_4 Camera Swing and Distance Change Test.mp4"
REF_FRAME = 0
REF_ANCHOR = -15.0         # fresh x - 15 = absolute yards (verified vs the
                           # orange LOS at the 50 and red 1st-down at the 21)
PXY = fh.PXY
OUT_VIDEO = "data/output/tracked_players.mp4"
OUT_CSV = "data/output/positions.csv"
OUT_BOXES = "data/output/boxes.csv"
SAMPLE_AT = {0, 30, 60, 120, 180, 240}
SAMPLE_DIR = "data/output/frames"
HOMOGRAPHY_CACHE = "data/output/homography.npz"
FIELD_MAP = ""
TEAM_PROTOTYPES = None
# External boxes CSV. None keeps the built-in detector and tracker.
PLAYER_BOXES = None
# Per-frame affine camera motion from the detector cache. This lets the
# renderer distinguish a field-fixed cone from a small moving player even
# when the field homography is stale.
CAMERA_MOTION = None
# (x0, x1) of the picture inside black side bars. None until the first frame.
_CONTENT_X = None
WRITE_VIDEO = True
STOP_AT = None
# Published-box relink. False leaves the ids track_bidirectional wrote.
RELINK = True
# 4b. False only builds the cuts and does not rename.
RELINK_LINKS = True
# Screen-fixed captions (banner, scoreboard text, WIN letters). Field
# paint is not in the mask. False leaves every box the tracker wrote.
OVERLAY_MASK = True
_OVERLAY_FRAMES = None
# Two ids on one body, removed after tracking. False writes every box.
SAME_BODY = True


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


# Dropped when the frame index changes. A capture can reuse one ndarray
# for the next frame, so identity alone would keep the previous picture.
_MEMO_AT = None
_MEMO = {}
_MISSING = object()
_FRAME_BUDGET = 1500 * 1024 * 1024


def _memo_reset():
    global _MEMO_AT, _MEMO
    _MEMO_AT = None
    _MEMO = {}


def _memo_begin(index):
    global _MEMO_AT, _MEMO
    if index != _MEMO_AT:
        _MEMO_AT = index
        _MEMO = {}


def _memo_get(key, objs):
    # The dict key is an address. The arrays are kept so that address
    # cannot be reused for a different array during this frame.
    if _MEMO_AT is None:
        return _MISSING
    hit = _MEMO.get(key)
    if hit is None:
        return _MISSING
    stored, value = hit
    if len(stored) != len(objs):
        return _MISSING
    for have, want in zip(stored, objs):
        if have is not want:
            return _MISSING
    return value


def _memo_put(key, objs, value):
    if _MEMO_AT is not None:
        _MEMO[key] = (tuple(objs), value)


def _hsv_of(frame):
    key = ("hsv", id(frame))
    objs = (frame,)
    hit = _memo_get(key, objs)
    if hit is not _MISSING:
        return hit
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _memo_put(key, objs, hsv)
    return hsv


def _load_frames(n):
    """Frames the detect, track, and render passes would each decode.

    Stored only when the cropped clip is under 1.5 GB. The copy is the
    picture _take returns, so a later read cannot overwrite it.
    """
    cap = cv2.VideoCapture(CLIP)
    frames = []
    while len(frames) < n:
        ok, frame = _take(cap)
        if not ok:
            break
        if not frames and frame.nbytes * n > _FRAME_BUDGET:
            cap.release()
            return None
        frames.append(frame.copy())
    cap.release()
    return frames


def _iter_frames(n, frames):
    if frames is not None:
        for i, frame in enumerate(frames):
            if i >= n:
                break
            yield i, frame
        return
    cap = cv2.VideoCapture(CLIP)
    i = 0
    while i < n:
        ok, frame = _take(cap)
        if not ok:
            break
        yield i, frame
        i += 1
    cap.release()


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
    key = ("teams", id(hsv))
    objs = (hsv,)
    hit = _memo_get(key, objs)
    if hit is not _MISSING:
        return hit
    hch, sch, vch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((hch < 12) | (hch > 168)) & (sch > 75) & (vch > 50)
    blue = (hch > 100) & (hch < 135) & (sch > 75) & (vch > 55)
    out = (red, blue)
    _memo_put(key, objs, out)
    return out


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


def _jersey_numbers(hsv, mask, bbox):
    """Centers of the numbers printed on this team's jerseys inside bbox.

    A number is a light hole in the jersey color. Closing the mask fills
    it, so the pixels that closing adds and that are light and not grass
    are the digits. Hands, gaps, and stripes are much smaller than a pair
    of digits.
    """
    height, width = hsv.shape[:2]
    x, y, bw, bh = [int(v) for v in bbox]
    x0, y0 = max(0, x - 4), max(0, y - 4)
    x1, y1 = min(width, x + bw + 4), min(height, y + bh + 4)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return []
    jer = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    filled = cv2.morphologyEx(jer, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)) > 0
    patch = hsv[y0:y1, x0:x1]
    hh, ss, vv = patch[..., 0], patch[..., 1], patch[..., 2]
    grass = (hh > 60) & (hh < 110) & (ss > 60)
    holes = filled & (jer == 0) & (ss < 115) & (vv > 125) & ~grass
    joined = cv2.dilate(holes.astype(np.uint8) * 255, np.ones((3, 5), np.uint8))
    count, _, stats, cent = cv2.connectedComponentsWithStats(joined)
    found = []
    for i in range(1, count):
        cw, ch, area = int(stats[i][2]), int(stats[i][3]), int(stats[i][4])
        if area < 150 or cw < 14 or ch < 10 or cw > 44 or ch > 34:
            continue
        found.append((x0 + float(cent[i][0]), y0 + float(cent[i][1])))
    return found


def _split_by_numbers(hsv, mask, bbox):
    """Two teammates side by side are one blob with two numbers on it.
    Cut the blob halfway between the numbers."""
    x, y, bw, bh = bbox
    nums = sorted(_jersey_numbers(hsv, mask, bbox))
    if len(nums) < 2:
        return [bbox]
    cuts = []
    for (ax, _), (bx, _) in zip(nums, nums[1:]):
        if bx - ax >= 18:
            cuts.append(int(round((ax + bx) / 2.0)))
    if not cuts:
        return [bbox]
    parts = []
    left = x
    for cut in cuts + [x + bw]:
        if cut - left >= 16:
            parts.append((left, y, cut - left, bh))
        left = cut
    return parts if len(parts) >= 2 else [bbox]


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
    objs = (mask, hsv)
    key = ("bodies", id(mask), shape[:2],
           None if hsv is None else id(hsv), _LOOSE_LINE)
    hit = _memo_get(key, objs)
    if hit is not _MISSING:
        return hit
    height, width = shape[:2]
    bodies = []
    comps = _merge_vertical(_mask_components(mask), hsv=hsv)
    if _LOOSE_LINE and hsv is not None:
        split = []
        for comp in comps:
            parts = _split_by_numbers(hsv, mask, comp["bbox"])
            if len(parts) == 1:
                split.append(comp)
                continue
            for px, py, pw, ph in parts:
                area = int(mask[py:py + ph, px:px + pw].sum())
                split.append(dict(bbox=(px, py, pw, ph), area=area, numbered=True))
        comps = split
    for comp in comps:
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
        # The KAG logo and the bottom banner sit in the corner.
        # A player on the sideline is above that corner.
        if y > height * 0.82 and x + bw / 2.0 > width * 0.82:
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
            corr = 0.0
            if widths.size >= 8 and float(widths.std()) > 1.0:
                corr = float(np.corrcoef(np.arange(widths.size), widths)[0, 1])
                patch = hsv[y:y + bh, x:x + bw]
                white = float(((patch[..., 1] < 70) & (patch[..., 2] > 150)).mean())
                # The red 20 marker is a triangle that widens downward
                # around a white digit. Jerseys do not do both.
                if corr > 0.65 and white > 0.07 and bw < 55:
                    continue
            if y + bh < height * 0.84:
                y1 = min(height - 1, y + bh + int(max(16, bh * 0.8)))
                if y1 > y + bh:
                    sch, vch = hsv[..., 1], hsv[..., 2]
                    leg = float(((sch[y + bh:y1, x:x + bw] < 105)
                                 & (vch[y + bh:y1, x:x + bw] > 48)).mean())
                    # He is on the white sideline, so the pants test sees
                    # the line instead of his legs. The jersey is still his.
                    edge = x + bw > width * 0.90
                    on_sideline = (
                        edge and comp["area"] >= 1800 and bh >= 70 and corr < 0.45)
                    if leg < 0.15 and not on_sideline:
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
        # on them has that paint and no jersey. The end-zone name is a
        # darker blue, and a wide short box on it is the same kind of paint.
        if hsv is not None and (_on_field_logo(hsv, (nx, ny, nw, nh))
                                or _on_endzone_letters(hsv, (nx, ny, nw, nh))
                                or _flat_on_paint(hsv, (nx, ny, nw, nh))):
            continue
        bodies.append((nx, ny, nw, nh, cx, float(ny + nh)))
    _memo_put(key, objs, bodies)
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
    hsv = _hsv_of(frame)
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


def _endzone_span(hsv, box):
    """How wide the saturated end-zone name is through this box.

    Those letters are the same blue as a jersey, so the pale-logo test
    misses them. A bright flare in the middle of a letter is still the
    name. The gaps between letters are short, and a jersey is one blob.
    """
    height, width = hsv.shape[:2]
    x, y, w, h = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    x0, x1 = max(0, x), min(width, x + max(w, 1))
    y0, y1 = max(0, y), min(height, y + max(h, 1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0
    best = 0
    for row in range(y0, y1, 2):
        hue, sat, val = hsv[row, :, 0], hsv[row, :, 1], hsv[row, :, 2]
        paint = ((hue > 105) & (hue < 132) & (val > 90)
                 & ((sat > 70) | (val > 170)))
        idx = np.where(paint)[0]
        if idx.size < 10:
            continue
        splits = np.where(np.diff(idx) > 42)[0]
        starts = np.r_[0, splits + 1]
        ends = np.r_[splits, idx.size - 1]
        for a, b in zip(starts, ends):
            left, right = int(idx[a]), int(idx[b])
            if min(right, x1) - max(left, x0) < 10:
                continue
            best = max(best, right - left)
    return best


def _flat_on_paint(hsv, box):
    """A wide, short box sitting on field paint.

    A tall box that still holds a jersey only overlaps the paint, so
    the name underneath it does not count. The painted name is wider
    than a player.
    """
    height, width = hsv.shape[:2]
    x, y, w, h = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    if h < 16 or w < 28 or h > 52 or w < int(h * 1.25):
        return False
    if _on_field_logo(hsv, box) or _on_endzone_letters(hsv, box):
        return True
    x0, x1 = max(0, x), min(width, x + max(w, 1))
    y0, y1 = max(0, y), min(height, y + max(h, 1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return False
    patch = hsv[y0:y1, x0:x1]
    hh, ss, vv = patch[..., 0], patch[..., 1], patch[..., 2]
    grass = (hh > 65) & (hh < 110) & (ss > 55) & (vv > 40)
    if float(grass.mean()) > 0.72:
        return False
    blue = (hh > 105) & (hh < 132) & (ss > 70) & (vv > 90)
    flare = (hh > 100) & (hh < 135) & (ss < 100) & (vv > 170)
    if float((blue | flare).mean()) < 0.34:
        return False
    return _endzone_span(hsv, box) >= 240


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
    # Same tests as a full-frame mask, on the rows this walk actually reads.
    y_lo = max(0, int(y) - 56)
    y_hi = min(height, int(y + bh) + 72)
    patch = hsv[y_lo:y_hi, x0:x1]
    hch, sch, vch = patch[..., 0], patch[..., 1], patch[..., 2]
    field = (hch > 65) & (hch < 110) & (sch > 55) & (vch > 40)
    person = ~field & (vch > 42)
    top = y
    gap = 0
    for row in range(int(y) - 1, max(0, int(y) - 56), -1):
        if float(person[row - y_lo].mean()) > 0.22:
            top = row
            gap = 0
        else:
            gap += 1
            if gap > 6:
                break
    bot = y + bh
    gap = 0
    for row in range(int(y + bh), min(height - 1, int(y + bh) + 72)):
        if float(np.median(vch[row - y_lo])) < 42:
            break
        if float(person[row - y_lo].mean()) > 0.18:
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
    hsv = _hsv_of(frame)
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
            for part in _split_wide(hard, comp["bbox"]):
                # Clip 4 never sets the loose-line flag, so its seeds stay.
                if _LOOSE_LINE:
                    seeds.extend(_split_by_numbers(hsv, mask, part))
                else:
                    seeds.append(part)
        for x, y, bw, bh in seeds:
            # A player in the back of the picture is a short jersey. The
            # usual 22-pixel seed drops him, then he never gets a number.
            far_seed = y < height * 0.42 and bh >= 16 and bw * bh >= 360
            if bw > 170 or bw < 16 or bh < 16:
                continue
            if not far_seed and (bw * bh < 700 or bh < 22):
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
            # A player running away gets a narrow jersey. The usual
            # 34-pixel box drops the far-right blue once he is small.
            # Clip 4 never sets the loose-line flag, so its cutoff stays.
            min_w = 24 if (_LOOSE_LINE and y < height * 0.42) else 34
            if x1 - x0 < min_w or yb - y0 < 36:
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


def _external_body(bbox, A):
    """A learned box whose foot projects onto the field.

    The -10 yard lower bound and the 0.25 yard minimum in _accepted_body
    match classical boxes. Learned feet on the quarterback land just past
    y=-10, at the point the classical track already publishes, and distant
    players are narrower than 0.25 yards. Width uses the centre 40% of the
    bottom edge so a wide torso still has a foot under the 6.5 yard cap.
    The upper y bound of 50 stays, which stops a bad foot from walking
    off the field under _clamp_step.
    """
    world = _project_point(*_foot_of(bbox), A)
    if world is None or not np.isfinite(world).all():
        return None
    xw, yw = float(world[0]), float(world[1])
    if yw > 50:
        return None
    x0, y0, bw, bh = bbox
    inset = 0.30
    corners = np.array(
        [[[x0 + bw * inset, y0 + bh]], [[x0 + bw * (1.0 - inset), y0 + bh]]],
        np.float32,
    )
    wc = cv2.perspectiveTransform(corners, A.astype(np.float32))[:, 0, :]
    if not np.isfinite(wc).all():
        return None
    width_yd = float(np.linalg.norm(wc[1] - wc[0]))
    if width_yd > 6.5:
        return None
    return {"world": (xw, yw), "width_yd": width_yd}


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
    trimmed = _trim_marker_box(hsv, best["bbox"])
    if trimmed != tuple(int(round(v)) for v in best["bbox"]):
        best["bbox"] = trimmed
        best["feet"] = _foot_of(trimmed)
        # The snap order uses the feet from before this cut. The roster
        # moves the world onto the new feet after the ids are assigned.
        best["marker_trim"] = True
    best["main"] = True


# Set on the first frame of a clip. Clip 4's defense stands past midfield,
# so the flag stays off and the pant test is unchanged. A clip whose whole
# line is on the near side of that uses the wider pant test.
_LOOSE_LINE = None


def _bodies_of(frame, A, hint, prev_r, leg_min):
    bodies, ring = _propose_bodies(frame, hint, prev_r, leg_min=leg_min)
    items = [d for d in (_accepted_body(b, A) for b in bodies) if d is not None]
    items = _on_this_field(items)
    hsv = _hsv_of(frame)
    items = [d for d in items
             if not _on_field_logo(hsv, d["bbox"])
             and not _flat_on_paint(hsv, d["bbox"])]
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


def _glyph_lines(glyphs):
    """Runs of similar glyphs on one horizontal line.

    A jersey is one blob. A caption is several blobs of one height
    spread across the screen.
    """
    glyphs = sorted(glyphs, key=lambda g: g[1] + g[3] / 2.0)
    used = [False] * len(glyphs)
    lines = []
    for i, glyph in enumerate(glyphs):
        if used[i]:
            continue
        cy = glyph[1] + glyph[3] / 2.0
        group = [i]
        used[i] = True
        for j, other in enumerate(glyphs):
            if used[j]:
                continue
            hy = other[1] + other[3] / 2.0
            if abs(hy - cy) < 34 and abs(other[3] - glyph[3]) < 36:
                group.append(j)
                used[j] = True
        if len(group) < 3:
            continue
        span = (max(glyphs[k][0] + glyphs[k][2] for k in group)
                - min(glyphs[k][0] for k in group))
        minh = min(glyphs[k][3] for k in group)
        count = len(group)
        if (count >= 4 and span >= 220) or (count >= 3 and span >= 310 and minh >= 75):
            lines.append([glyphs[k] for k in group])
    return lines


def _screen_overlay_mask(frame):
    """Pixels of a screen caption. The mask is not written into later."""
    key = ("overlay", id(frame))
    hit = _memo_get(key, (frame,))
    if hit is not _MISSING:
        return hit
    hsv = _hsv_of(frame)
    height, width = hsv.shape[:2]
    hue = hsv[..., 0].astype(np.int16)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    mask = np.zeros((height, width), np.uint8)
    blue = ((hue >= 112) & (hue <= 128) & (sat > 90) & (val > 60))
    blue = cv2.morphologyEx(
        blue.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blue)
    glyphs = []
    for i in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if area < 700 or bh < 36 or bw > 200 or bh < bw * 0.8:
            continue
        glyphs.append((x, y, bw, bh, i))
    for line in _glyph_lines(glyphs):
        for _x, _y, _bw, _bh, i in line:
            mask[labels == i] = 255
    # One wide caption (INCOMPLETE) is a single slab, not a row of narrow glyphs.
    for i in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if area < 5000 or bw < 220 or bh < 28 or bh > 130:
            continue
        if bh > bw * 0.55:
            continue
        mask[labels == i] = 255
    green = ((hue >= 48) & (hue <= 68) & (sat > 140) & (val > 80))
    green = cv2.morphologyEx(
        green.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    gcount, glabels, gstats, _ = cv2.connectedComponentsWithStats(green)
    green_glyphs = []
    for i in range(1, gcount):
        x, y, bw, bh, area = [int(v) for v in gstats[i]]
        if area < 600 or bh < 28 or bw > 120 or bh < bw * 0.55:
            continue
        green_glyphs.append((x, y, bw, bh, i))
    for line in _glyph_lines(green_glyphs):
        for _x, _y, _bw, _bh, i in line:
            mask[glabels == i] = 255
    # White caption letters. Pants are one blob, not a row of these.
    white = (sat < 55) & (val > 185)
    white = cv2.morphologyEx(
        white.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    wcount, wlabels, wstats, _ = cv2.connectedComponentsWithStats(white)
    white_glyphs = []
    for i in range(1, wcount):
        x, y, bw, bh, area = [int(v) for v in wstats[i]]
        if area < 500 or bh < 24 or bw > 80 or bh < bw * 0.7:
            continue
        white_glyphs.append((x, y, bw, bh, i))
    for line in _glyph_lines(white_glyphs):
        for _x, _y, _bw, _bh, i in line:
            mask[wlabels == i] = 255
    yellow = ((hue >= 18) & (hue <= 40) & (sat > 155) & (val > 165))
    yellow = cv2.dilate(yellow.astype(np.uint8) * 255, np.ones((3, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(yellow)
    for i in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        # A yard stripe is a hairline. A caption is a short wide block.
        if bw < 110 or bh < 16 or bh > 80 or area < 400:
            continue
        if bw > 650 and bh < 28:
            continue
        mask[labels == i] = 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    _memo_put(key, (frame,), mask)
    return mask


def _overlay_rejects(hsv, mask, box, frame_index, track_id, team, main=False):
    """True when the box is caption and no jersey is left outside it.

    Blue 11 is the clip 4 quarterback. On frames 0-160 his feet are
    still on screen, including where the bottom banner crosses him.
    """
    if mask is None:
        return False
    height, width = mask.shape[:2]
    feet = float(box[1]) + float(box[3])
    if (frame_index is not None and frame_index <= 160 and feet < height - 6
            and team == "blue"
            and (track_id == 11 or main)):
        return False
    x, y, bw, bh = [int(round(float(v))) for v in box]
    x0, y0 = max(0, x), max(0, y)
    x1 = min(width, x + max(bw, 1))
    y1 = min(height, y + max(bh, 1))
    if x1 - x0 < 4 or y1 - y0 < 8:
        return False
    covered = mask[y0:y1, x0:x1] > 0
    if float(covered.mean()) < 0.22:
        return False
    patch = hsv[y0:y1, x0:x1]
    bgr = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)
    hue = patch[..., 0].astype(np.int16)
    sat = patch[..., 1]
    val = patch[..., 2]
    bb = bgr[..., 0].astype(np.int16)
    gg = bgr[..., 1].astype(np.int16)
    rr = bgr[..., 2].astype(np.int16)
    outside = ~covered
    cyan = ((hue > 84) & (hue < 103) & (sat > 80) & (val > 50)
            & (bb > gg + 8) & (rr < 95) & (rr + 28 < gg))
    red = (((hue < 12) | (hue > 168)) & (sat > 70) & (val > 45) & (rr > gg + 8))
    jersey = (cyan | red) & outside
    hgt = jersey.shape[0]
    top = float(jersey[:max(1, hgt // 2)].mean())
    bot = float(jersey[int(hgt * 0.40):].mean())
    if top > 0.04 and bot > 0.03 and float(outside.mean()) > 0.25:
        return False
    return True


def frame_candidates(frame, A, hint=None, prev_r=None):
    """Bodies for one frame, feet on the ground plane. The ring is returned
    beside the list so the main player's feet can follow it."""
    global _LOOSE_LINE
    # The 0.32 pass only decides the flag. Once it is set, those boxes are discarded.
    if _LOOSE_LINE:
        items, ring, hsv = _bodies_of(frame, A, hint, prev_r, 0.15)
        _claim_arrow_body(frame, hsv, items, ring)
        return items, ring
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
                               bbox=d["bbox"], main=bool(d.get("main")),
                               marker_trim=bool(d.get("marker_trim"))))
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
    hsv = _hsv_of(frame)
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
            if det["team"] == team:
                penalty = 0.0
            elif det["team"] in ("red", "blue"):
                # A clear opponent has to be much closer than a teammate.
                penalty = 80.0
            else:
                penalty = 28.0
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


def _trim_marker_box(hsv, box):
    """Cut the '1' and its chevron off when they are the top of the box.

    The chevron is the same blue as the jersey, so a pad above the jersey
    swallows the digit. A chevron is a short band that is wider at the top,
    with the dark plaque above it. The column under that band is the
    helmet and the feet. A box that already starts at the helmet stays.
    """
    height, width = hsv.shape[:2]
    x, y, w, h = [int(round(v)) for v in box]
    if h < 70 or w < 16:
        return box
    x0, x1 = max(0, x), min(width, x + w)
    if x1 - x0 < 8:
        return box
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    y_hi = min(height - 1, y + min(h, 100))
    rows = []
    for row in range(max(0, y), y_hi):
        hh, ss, vv = hue[row, x0:x1], sat[row, x0:x1], val[row, x0:x1]
        blue = float(((hh > 105) & (hh < 130) & (ss > 150) & (vv > 90)).mean())
        if blue > 0.35:
            rows.append(row)
    if len(rows) < 5:
        return box
    run = [rows[0]]
    for row in rows[1:]:
        if row <= run[-1] + 4:
            run.append(row)
        else:
            break
    if not (10 <= run[-1] - run[0] <= 52):
        return box
    # The digit is a dark plaque with a white '1' in it. A silver helmet
    # is bright, so the same blue band on a jersey does not count.
    plaque = False
    if run[0] > y + 2:
        above = hsv[y:run[0], x0:x1]
        if above.size:
            dark = float(((above[..., 2] < 80) & (above[..., 1] < 110)).mean())
            white = float(((above[..., 1] < 80) & (above[..., 2] > 160)).mean())
            plaque = dark >= 0.28 and 0.08 <= white <= 0.45
    # A chevron at the top of the box is the marker. One farther down is
    # a jersey, unless the dark '1' plaque is what fills the gap.
    if run[0] > y + 40 and not plaque:
        return box
    if run[0] > y + 68:
        return box
    ny = min(height - 48, run[-1] + 1)
    x_lo = max(0, x - 30)
    x_hi = min(width, x + w + 30)
    band = hsv[ny:min(height, ny + 28), x_lo:x_hi]
    if band.size == 0:
        return box
    bh, bs, bv = band[..., 0], band[..., 1], band[..., 2]
    body = ~((bh > 65) & (bh < 108) & (bs > 50) & (bv > 40)) & (bv > 42)
    cols = body.sum(axis=0)
    xs = np.where(cols > 2)[0]
    nx, bw = x, w
    if xs.size >= 4:
        splits = np.where(np.diff(xs) > 6)[0]
        starts = np.r_[0, splits + 1]
        ends = np.r_[splits, xs.size - 1]
        best = max(range(len(starts)), key=lambda i: int(ends[i] - starts[i]))
        left = int(xs[starts[best]])
        right = int(xs[ends[best]])
        cx = x_lo + (left + right) / 2.0
        old = x + w / 2.0
        if abs(cx - old) <= 24:
            bw = int(max(w, min(70, right - left + 8)))
            nx = int(round(cx - bw / 2.0))
            nx = max(-8, min(width - 8, nx))
    half = max(8, bw // 3)
    a0 = max(0, int((nx + bw / 2.0) - half))
    a1 = min(width, int((nx + bw / 2.0) + half))
    if a1 - a0 < 6:
        return box
    last = ny
    gap = 0
    limit = min(height - 2, ny + 130)
    for row in range(ny + 2, limit):
        hh = hue[row, a0:a1]
        ss = sat[row, a0:a1]
        vv = val[row, a0:a1]
        field = (hh > 65) & (hh < 108) & (ss > 55) & (vv > 40)
        if float(((~field) & (vv > 40)).mean()) > 0.22:
            last = row
            gap = 0
        else:
            gap += 1
            if gap > 16 and last > ny + 36:
                break
    nh = last - ny + 1
    if nh < 56 or nh > 140:
        return box
    return (nx, ny, int(bw), int(nh))


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
    return _trim_marker_box(hsv, (nx, ny, bw, nh))


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
    return _trim_marker_box(hsv, (nx, ny, bw, nh))


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


def track_bidirectional(per_frame, roster, homographies, shape, rings=None, arrows=None,
                        frames=None):
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
    fresh = max(ids) + 1 if ids else 1
    roster_top = fresh
    born = {}
    team_of = {r["id"]: r["team"] for r in roster}
    main_ids = [r["id"] for r in roster if r.get("main")]
    main_id = main_ids[0] if main_ids else None
    fwd = [dict() for _ in range(n)]
    fwd[0] = {r["id"]: dict(r) for r in roster}
    last = {r["id"]: r["world"] for r in roster}
    vel = {r["id"]: (0.0, 0.0) for r in roster}
    box_of = {r["id"]: r["bbox"] for r in roster}
    qb_box = box_of[main_id] if main_id is not None else (0, 0, 1, 1)
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
    _memo_reset()
    if frames is not None:
        cap = None
        ok0 = len(frames) > 0
        prev_frame = frames[0] if ok0 else None
    else:
        cap = cv2.VideoCapture(CLIP)
        ok0, prev_frame = _take(cap)
    if ok0:
        _memo_begin(0)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if ok0 else None
    prev_field = _field_mask(prev_frame) if ok0 else None
    if ok0:
        prev_red, prev_blue = _team_masks(_hsv_of(prev_frame))
    else:
        prev_red = prev_blue = None
    flow_hits = flow_tries = 0
    for t in range(1, n):
        if frames is not None:
            if t >= len(frames):
                break
            frame = frames[t]
        else:
            ok, frame = _take(cap)
            if not ok:
                break
        _memo_begin(t)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur_red, cur_blue = _team_masks(_hsv_of(frame))
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
            use_ring = False
            if tid == main_id and ring is not None:
                ring_guide = (float(ring["cx"]), float(ring["cy"]))
                gx, gy = img[tid]
                # A ring that appears on another blue is not his circle.
                if float(np.hypot(ring_guide[0] - gx, ring_guide[1] - gy)) <= 100:
                    guide = ring_guide
                    use_ring = True
            if guide is None and tid == main_id and arrow is not None:
                arrow_guide = (float(arrow[0]), float(arrow[1] + 2.4 * arrow[2]))
                gx, gy = img[tid]
                if float(np.hypot(arrow_guide[0] - gx, arrow_guide[1] - gy)) <= 100:
                    guide = arrow_guide
            if guide is not None and 0.12 * height < guide[1] < 0.92 * height:
                u, v = img[tid]
                hsv_now = _hsv_of(frame)
                if use_ring:
                    box = _qb_box(hsv_now, ring, shape, team_of[tid])
                    landed = None
                else:
                    box = _qb_from_foot(hsv_now, guide, shape, team_of[tid])
                    landed = _qb_body_under_arrow(hsv_now, cur_blue, arrow, shape)
                    if landed is not None:
                        landed_box, landed_foot = landed
                        if float(np.hypot(landed_foot[0] - u, landed_foot[1] - v)) <= 100:
                            box, guide = landed_box, landed_foot
                        else:
                            landed = None
                box = _trim_marker_box(hsv_now, box)
                # A false arrow sits on empty grass. The real one stands over
                # the quarterback, so his box still holds a jersey. When the
                # tracked arrow is the false one, use an arrow candidate in
                # the upper field that still has a jersey under it.
                empty = (not use_ring) and landed is None and (
                    _mostly_grass(hsv_now, box)
                    or _jersey_count(cur_blue, box) < 40
                    or (box[1] + box[3] / 2.0) < height * 0.20)
                if empty and not use_ring:
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
                        # The "1" on the far side of the field is the next
                        # ball carrier. The quarterback is still where he was.
                        if float(np.hypot(alt[0] - u, alt[1] - v)) > 100:
                            continue
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
                    # He was already off the screen. A marker that no longer
                    # overlaps his last box is the player who ran into that spot.
                    stale = False
                    if tid == main_id and t > 0 and main_id not in fwd[t - 1]:
                        ox = qb_box[0] + qb_box[2] / 2.0
                        oy = qb_box[1] + qb_box[3] / 2.0
                        nx = box[0] + box[2] / 2.0
                        ny = box[1] + box[3] / 2.0
                        stale = (
                            _iou_over_smaller(box, qb_box) < 0.12
                            or ny < oy - 70
                            or float(np.hypot(nx - ox, ny - oy)) > 90)
                    if not stale:
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
            if tid == main_id and t > 0 and main_id not in fwd[t - 1]:
                ox = qb_box[0] + qb_box[2] / 2.0
                oy = qb_box[1] + qb_box[3] / 2.0

                def _left_him(bb):
                    nx = bb[0] + bb[2] / 2.0
                    ny = bb[1] + bb[3] / 2.0
                    return (
                        _iou_over_smaller(bb, qb_box) < 0.12
                        or ny < oy - 70
                        or float(np.hypot(nx - ox, ny - oy)) > 90)

                if fit is not None and _left_him(fit):
                    fit = None
                if use_det and _left_him(got[tid]["bbox"]):
                    use_det = False
            if fit is not None:
                box = fit
                if tid == main_id:
                    box = _trim_marker_box(_hsv_of(frame), box)
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
                if tid == main_id:
                    trimmed = _trim_marker_box(_hsv_of(frame), det["bbox"])
                    if trimmed != tuple(int(round(v)) for v in det["bbox"]):
                        det = dict(det, bbox=trimmed)
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
                if (tid == main_id and t > 0 and main_id not in fwd[t - 1]):
                    reach = 40.0
                snapped = None if tid in flowed and not lost else _snap_body(
                    frame, pred, team_of[tid], reach=reach)
                if snapped is not None:
                    bx, by, bw, bh, fx, fy = snapped
                    if (tid == main_id and t > 0 and main_id not in fwd[t - 1]):
                        oy = qb_box[1] + qb_box[3] / 2.0
                        if (by + bh / 2.0) < oy - 70:
                            snapped = None
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
                        hsv_live = _hsv_of(frame)
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
                # His feet already left the bottom. Coasting the old box
                # paints it on the banner in the corner he just left.
                exited = (
                    tid == main_id and t > 0 and main_id not in fwd[t - 1]
                    and qb_box[1] + qb_box[3] > height * 0.90)
                if inside and not exited:
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
        cur_hsv = _hsv_of(frame)

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
                # The far-right blue gets too narrow for the jersey-body
                # list, but the box is still on his jersey. Snapping to the
                # nearest body puts his number on the player beside him.
                if (tid in fwd[t] and _LOOSE_LINE
                        and _jersey_count(mask, fwd[t][tid]["bbox"]) >= 80):
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
                cur_foot = _foot_of(fwd[t][tid]["bbox"]) if tid in fwd[t] else None
                best, best_d = None, limit
                for j, body in enumerate(bodies):
                    if j in taken:
                        continue
                    dist = float(np.hypot(body[4] - sfx, body[5] - sfy))
                    # The box is still on this player. A body up in the
                    # pile can sit inside the search radius of a stale
                    # foot and steal the number.
                    if cur_foot is not None:
                        own = float(np.hypot(
                            body[4] - cur_foot[0], body[5] - cur_foot[1]))
                        if own > 55:
                            nearer = False
                            for other, det in fwd[t].items():
                                if other == tid or team_of.get(other) != team:
                                    continue
                                ox, oy = _foot_of(det["bbox"])
                                if float(np.hypot(
                                        body[4] - ox, body[5] - oy)) + 18 < own:
                                    nearer = True
                                    break
                            if nearer:
                                continue
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
                # The main player ran away. His box stayed low in the
                # same column, and the jersey is the short one above it.
                if (tid == main_id and _LOOSE_LINE
                        and abs(fx - px) < 42 and fy < py - 30):
                    dist = abs(fx - px)
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
                # The end-zone name is saturated blue, so the logo test
                # never sees it. A wide short box on that paint is the
                # letter. A tall box that still holds his jersey stays.
                if _flat_on_paint(cur_hsv, box):
                    _revert(tid)
                    del fwd[t][tid]
                    continue
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
                    if _LOOSE_LINE:
                        _keep_far(tid, team_of[tid])
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
            if (_on_field_logo(cur_hsv, box) or _on_endzone_letters(cur_hsv, box)
                    or _flat_on_paint(cur_hsv, box)):
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
                own_count = _jersey_count(own_mask, box)
                # On a clip where the line is close to the camera, a player
                # who runs away is a short box. Clip 4 never sets this flag.
                short_far = (
                    _LOOSE_LINE and on_body and box[3] >= 44 and box[2] <= 70
                    and own_count >= 36)
                if ((own_count >= 80 or short_far) and not scoreboard
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
        # Three reds run together on the right, and the snap roster
        # only carried one of them. A red jersey with its own printed
        # number and no box on that number gets one. Clip 4 never sets
        # the loose-line flag.
        if _LOOSE_LINE and fresh < roster_top + 3:
            for body in red_bodies:
                x, y, bw, bh = [int(v) for v in body[:4]]
                if bw < 36 or bh < 110 or bw > 110:
                    continue
                if y + bh > height * 0.94:
                    continue
                box = (x, y, bw, bh)
                if (_on_field_logo(cur_hsv, box)
                        or _on_endzone_letters(cur_hsv, box)
                        or _on_label(box, ring, arrow, shape)):
                    continue
                torso = (x, y, bw, int(bh * 0.6))
                nums = _jersey_numbers(cur_hsv, cur_red, torso)
                if not nums:
                    continue
                fx, fy = float(body[4]), float(body[5])
                taken = False
                revive = None
                revive_d = 110.0
                for other in list(ids):
                    if team_of.get(other) != "red" or not alive.get(other):
                        continue
                    if other in fwd[t]:
                        ob = fwd[t][other]["bbox"]
                        # One box per printed number. A teammate's box
                        # that reaches over this one but not onto its
                        # number belongs to the teammate.
                        if any(ob[0] <= nx <= ob[0] + ob[2]
                               and ob[1] <= ny <= ob[1] + ob[3]
                               for nx, ny in nums):
                            taken = True
                            break
                        ox, oy = _foot_of(ob)
                        if float(np.hypot(ox - fx, oy - fy)) < 20:
                            taken = True
                            break
                    else:
                        ox, oy = img[other]
                        dist = float(np.hypot(ox - fx, oy - fy))
                        if dist < revive_d:
                            revive, revive_d = other, dist
                if taken:
                    continue
                world = _project_point(fx, fy, homographies[t])
                if world is None:
                    continue
                if revive is not None:
                    tid = revive
                else:
                    if fresh >= roster_top + 3:
                        continue
                    tid = fresh
                    fresh += 1
                    ids.append(tid)
                    born[tid] = t
                    team_of[tid] = "red"
                    alive[tid] = True
                    off[tid] = miss[tid] = bare[tid] = 0
                    cruise[tid] = (0.0, 0.0)
                img[tid] = (fx, fy)
                img_vel[tid] = (0.0, 0.0)
                last[tid] = world
                vel[tid] = (0.0, 0.0)
                box_of[tid] = box
                solo[tid] = box
                planted[tid] = (fx, fy)
                good_world[tid] = world
                good_img[tid] = (fx, fy)
                held_world[tid] = world
                fwd[t][tid] = dict(
                    team="red", world=world, bbox=box,
                    frac=0.6, width_yd=1.2)
        if main_id is not None:
            if main_id in fwd[t]:
                qb_box = fwd[t][main_id]["bbox"]
            # Keep the search on the last box that was really him. Reverting
            # to an older foot puts the number on the blue who ran into
            # the spot he just left.
            img[main_id] = _foot_of(qb_box)
            good_img[main_id] = img[main_id]
        prev_field = _field_mask(frame)
        prev_red, prev_blue = cur_red, cur_blue
    if cap is not None:
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
            elif born.get(tid, 0) > t:
                chosen = None
            elif bdet is not None and fut is not None:
                ff = _foot_of(fut["bbox"])
                bf = _foot_of(bdet["bbox"])
                if tid == main_id:
                    # A gap fill has to be the same box. A blue standing
                    # where the quarterback just left is not him.
                    if _iou_over_smaller(bdet["bbox"], fut["bbox"]) > 0.12:
                        chosen = bdet
                    else:
                        chosen = None
                elif (float(np.hypot(bf[0] - ff[0], bf[1] - ff[1])) <= 160
                        and not _on_label(
                            bdet["bbox"],
                            rings[t] if t < len(rings) else None,
                            arrows[t] if t < len(arrows) else None, shape)):
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
            if tid not in pos:
                if det is None:
                    continue
                pos[tid] = det["world"]
                vel[tid] = (0.0, 0.0)
                conf[tid] = 0.8
                fused[t][tid] = dict(
                    pos=det["world"], conf=0.8,
                    bbox=det["bbox"], world=det["world"])
                continue
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


def _box_iou(a, b):
    """Intersection over union. 0 when either box is empty."""
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    left, top = max(ax, bx), max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _hungarian_assign(cost):
    """Minimum-cost pairs. Port of the dense Hungarian in tools/eval_tracking."""
    matrix = np.array(cost, dtype=np.float64, copy=True)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return []
    transposed = False
    if matrix.shape[0] > matrix.shape[1]:
        matrix = matrix.T.copy()
        transposed = True
    row_count, col_count = matrix.shape
    row_potential = np.zeros(row_count + 1, dtype=np.float64)
    col_potential = np.zeros(col_count + 1, dtype=np.float64)
    col_match = np.zeros(col_count + 1, dtype=np.int64)
    predecessor = np.zeros(col_count + 1, dtype=np.int64)
    huge = 1.0e12
    for row in range(1, row_count + 1):
        col_match[0] = row
        minimum = np.full(col_count + 1, huge, dtype=np.float64)
        used = np.zeros(col_count + 1, dtype=bool)
        current_col = 0
        guard = 0
        while True:
            guard += 1
            if guard > col_count + 2:
                break
            used[current_col] = True
            current_row = int(col_match[current_col])
            delta = huge
            next_col = 0
            for col in range(1, col_count + 1):
                if used[col]:
                    continue
                reduced = (
                    matrix[current_row - 1, col - 1]
                    - row_potential[current_row]
                    - col_potential[col]
                )
                if reduced < minimum[col]:
                    minimum[col] = reduced
                    predecessor[col] = current_col
                if minimum[col] < delta:
                    delta = float(minimum[col])
                    next_col = col
            for col in range(col_count + 1):
                if used[col]:
                    row_potential[col_match[col]] += delta
                    col_potential[col] -= delta
                else:
                    minimum[col] -= delta
            current_col = next_col
            if col_match[current_col] == 0:
                break
        while True:
            previous = int(predecessor[current_col])
            col_match[current_col] = col_match[previous]
            current_col = previous
            if current_col == 0:
                break
    pairs = []
    for col in range(1, col_count + 1):
        row = int(col_match[col])
        if row != 0:
            pairs.append((row - 1, col - 1))
    if transposed:
        pairs = [(col, row) for row, col in pairs]
    return pairs


def _hue_hist(hsv, box):
    """Normalized hue histogram of the torso. None when the patch is empty."""
    height, width = hsv.shape[:2]
    x, y, w, h = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    x0, x1 = max(0, x), min(width, x + max(w, 1))
    y0 = max(0, y)
    y1 = min(height, y + max(int(h * 0.5), 1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    patch = hsv[y0:y1, x0:x1]
    keep = patch[..., 1] > 40
    if int(keep.sum()) < 24:
        return None
    hist, _ = np.histogram(patch[..., 0][keep], bins=12, range=(0, 180))
    hist = hist.astype(np.float64)
    total = float(hist.sum())
    if total < 1:
        return None
    hist /= total
    norm = float(np.linalg.norm(hist))
    if norm < 1e-6:
        return None
    return hist / norm


def _number_sig(hsv, mask, box):
    """Small light/dark grid of the single readable jersey number."""
    nums = _jersey_numbers(hsv, mask, box)
    if len(nums) != 1:
        return None
    height, width = hsv.shape[:2]
    cx, cy = nums[0]
    x0, x1 = max(0, int(cx) - 10), min(width, int(cx) + 10)
    y0, y1 = max(0, int(cy) - 12), min(height, int(cy) + 12)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    patch = hsv[y0:y1, x0:x1]
    light = ((patch[..., 1] < 110) & (patch[..., 2] > 140)).astype(np.float64)
    small = cv2.resize(light, (6, 8), interpolation=cv2.INTER_AREA).ravel()
    norm = float(np.linalg.norm(small))
    if norm < 1e-6:
        return None
    return small / norm


def _relink_published(tracks, frames, shape, roster):
    """Cut each id into tracklets and relink them across a short gap.

    Frame-0 tracklets keep the roster ids. Boxes are renamed, not added
    or removed. Team is a hard gate. A link farther than 8 yards is refused.
    """
    main_ids = [r["id"] for r in roster if r.get("main")]
    main_id = main_ids[0] if main_ids else None
    by_id = {}
    team_of = {}
    for t, frame in enumerate(tracks):
        for tr in frame:
            tid = tr["id"]
            team_of[tid] = tr["team"]
            by_id.setdefault(tid, {})[t] = tr
    cuts = {tid: set() for tid in by_id}
    n_iou = n_yard = 0
    for tid, fmap in by_id.items():
        frames_of = sorted(fmap)
        for prev, nxt in zip(frames_of, frames_of[1:]):
            if nxt != prev + 1:
                continue
            io = _box_iou(fmap[prev]["bbox"], fmap[nxt]["bbox"])
            step_px = _dist(_foot_of(fmap[prev]["bbox"]), _foot_of(fmap[nxt]["bbox"]))
            step_yd = _dist(fmap[prev]["world"], fmap[nxt]["world"])
            if step_yd > 8.0:
                cuts[tid].add(nxt)
                n_yard += 1
            elif io < 0.2 and step_px > 48.0:
                cuts[tid].add(nxt)
                n_iou += 1
    ids = list(by_id)
    n_overlap = 0
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            if team_of[left] != team_of[right]:
                continue
            both = sorted(set(by_id[left]) & set(by_id[right]))
            began = None
            for fr in both:
                overlap = _box_iou(by_id[left][fr]["bbox"], by_id[right][fr]["bbox"])
                if overlap > 0.5 and began is None:
                    began = fr
                elif began is not None and overlap < 0.2:
                    if began > 0 and fr - began <= 18:
                        cuts[left].add(began)
                        cuts[right].add(began)
                        n_overlap += 1
                    began = None
                elif overlap <= 0.5:
                    began = None
    tracklets = []
    for tid, fmap in by_id.items():
        ordered = sorted(fmap)
        run = [ordered[0]]

        def _emit(run):
            tracklets.append(dict(
                tid=tid, team=team_of[tid], frames=list(run),
                pinned=run[0] == 0))

        for fr in ordered[1:]:
            if fr != run[-1] + 1 or fr in cuts[tid]:
                _emit(run)
                run = [fr]
            else:
                run.append(fr)
        _emit(run)
    n_cuts = n_iou + n_yard + n_overlap
    print(
        f"relink tracklets={len(tracklets)} cuts={n_cuts} "
        f"iou={n_iou} yard={n_yard} overlap={n_overlap}"
    )

    def _runs(frame_set):
        if not frame_set:
            return "()"
        ordered = sorted(frame_set)
        segs = []
        start = prev = ordered[0]
        for fr in ordered[1:]:
            if fr == prev + 1:
                prev = fr
                continue
            segs.append(f"({start}-{prev})")
            start = prev = fr
        segs.append(f"({start}-{prev})")
        return ",".join(segs)

    def _print_ids(published):
        before_sets = {tid: set(fmap) for tid, fmap in by_id.items()}
        after_sets = {}
        for fr, frame in enumerate(published):
            for tr in frame:
                after_sets.setdefault(tr["id"], set()).add(fr)
        created = sorted(set(after_sets) - set(before_sets))
        removed = sorted(set(before_sets) - set(after_sets))
        print("relink id runs:")
        for tid in sorted(set(before_sets) | set(after_sets)):
            before = before_sets.get(tid, set())
            after = after_sets.get(tid, set())
            mark = "" if before == after else " changed"
            print(f"  id {tid} before {_runs(before)} n={len(before)}")
            print(f"       after  {_runs(after)} n={len(after)}{mark}")
        print(f"relink ids created: {created or '(none)'}")
        print(f"relink ids removed: {removed or '(none)'}")
        return created, removed

    if not RELINK_LINKS or not tracklets:
        print("relink links=0 (link stage off)" if not RELINK_LINKS else "relink links=0")
        _print_ids(tracks)
        return tracks

    def _tail_vel(tl, kind):
        tail = tl["frames"][-4:]
        if len(tail) < 2:
            return (0.0, 0.0)
        if kind == "pix":
            p0 = _foot_of(by_id[tl["tid"]][tail[0]]["bbox"])
            p1 = _foot_of(by_id[tl["tid"]][tail[-1]]["bbox"])
        else:
            p0 = by_id[tl["tid"]][tail[0]]["world"]
            p1 = by_id[tl["tid"]][tail[-1]]["world"]
        dt = max(1, tail[-1] - tail[0])
        return ((p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt)

    cue = {}

    def _cue(tl):
        key = id(tl)
        if key in cue or frames is None:
            return cue.get(key, (None, None))
        hist = num = None
        sample = tl["frames"][:1] + tl["frames"][-1:]
        for fr in sample:
            if fr >= len(frames) or frames[fr] is None:
                continue
            hsv = _hsv_of(frames[fr])
            box = by_id[tl["tid"]][fr]["bbox"]
            if hist is None:
                hist = _hue_hist(hsv, box)
            if num is None:
                red, blue = _team_masks(hsv)
                mask = red if tl["team"] == "red" else blue
                num = _number_sig(hsv, mask, box)
        cue[key] = (hist, num)
        return cue[key]

    raw = []
    for ei, end in enumerate(tracklets):
        ef = end["frames"][-1]
        end_foot = _foot_of(by_id[end["tid"]][ef]["bbox"])
        end_world = by_id[end["tid"]][ef]["world"]
        end_pos = by_id[end["tid"]][ef]["pos"]
        vpix = _tail_vel(end, "pix")
        vyd = _tail_vel(end, "yd")
        for si, start in enumerate(tracklets):
            if start["pinned"] or si == ei or start["team"] != end["team"]:
                continue
            sf = start["frames"][0]
            dt = sf - ef
            if dt < 1 or dt > 15:
                continue
            pred_px = (end_foot[0] + vpix[0] * dt, end_foot[1] + vpix[1] * dt)
            start_foot = _foot_of(by_id[start["tid"]][sf]["bbox"])
            pix = _dist(pred_px, start_foot)
            pred_yd = (end_world[0] + vyd[0] * dt, end_world[1] + vyd[1] * dt)
            start_world = by_id[start["tid"]][sf]["world"]
            yards = _dist(pred_yd, start_world)
            if pix > 140.0 or yards > 8.0:
                continue
            if dt == 1 and _dist(end_pos, by_id[start["tid"]][sf]["pos"]) > 8.0:
                continue
            if (end["pinned"] and end["tid"] == main_id
                    and (pix > 110.0 or yards > 6.0)):
                continue
            cost = pix / 50.0 + 0.12 * yards
            if end["tid"] == start["tid"]:
                cost -= 0.15
            eh, en = _cue(end)
            sh, sn = _cue(start)
            if eh is not None and sh is not None:
                cost += 0.2 * (1.0 - float(np.dot(eh, sh)))
            if en is not None and sn is not None:
                agree = float(np.dot(en, sn))
                if agree > 0.8:
                    cost -= 0.2
                elif agree < 0.2:
                    cost += 0.45
            raw.append((cost, ei, si, pix, yards, dt))
    if not raw:
        print("relink links=0")
        _print_ids(tracks)
        return tracks
    end_ids = sorted(set(item[1] for item in raw))
    start_ids = sorted(set(item[2] for item in raw))
    end_at = {eid: i for i, eid in enumerate(end_ids)}
    start_at = {sid: i for i, sid in enumerate(start_ids)}
    n_end, n_start = len(end_ids), len(start_ids)
    matrix = np.full((n_end, n_start + n_end), 1.0e6, dtype=np.float64)
    for row in range(n_end):
        matrix[row, n_start + row] = 1.8
    best = {}
    for cost, ei, si, pix, yards, dt in raw:
        if cost > 1.8:
            continue
        key = (end_at[ei], start_at[si])
        if cost < best.get(key, (9e8,))[0]:
            best[key] = (cost, ei, si, pix, yards, dt)
            matrix[key] = cost
    chosen = []
    for row, col in _hungarian_assign(matrix):
        if col >= n_start or (row, col) not in best:
            continue
        cost, ei, si, pix, yards, dt = best[(row, col)]
        if cost > 1.8:
            continue
        chosen.append((cost, ei, si, pix, yards, dt))
    chosen.sort()
    kept = []
    seen = set()
    for cost, ei, si, pix, yards, dt in chosen:
        end, start = tracklets[ei], tracklets[si]
        # A cut with no id change stays the same id. Stitches do not rename.
        if end["tid"] == start["tid"]:
            continue
        recip = None
        for cost2, ej, sj, pix2, yards2, dt2 in chosen:
            other_end, other_start = tracklets[ej], tracklets[sj]
            if (other_end["frames"][-1] == end["frames"][-1]
                    and other_start["frames"][0] == start["frames"][0]
                    and other_end["tid"] == start["tid"]
                    and other_start["tid"] == end["tid"]):
                recip = (cost2, ej, sj, pix2, yards2, dt2)
                break
        # A one-way rename needs a fresh id for the displaced boxes.
        # Only a two-way swap frees the partner's id.
        if recip is None or dt > 2:
            continue
        pair = tuple(sorted((ei, recip[1])))
        if pair in seen:
            continue
        io = _box_iou(
            by_id[start["tid"]][start["frames"][0]]["bbox"],
            by_id[tracklets[recip[2]]["tid"]][start["frames"][0]]["bbox"])
        if io >= 0.4 or yards > 8.0 or recip[4] > 8.0:
            print(
                f"relink refuse swap f{end['frames'][-1]} "
                f"T{end['tid']}/T{start['tid']} iou={io:.2f} "
                f"yd={yards:.2f}/{recip[4]:.2f}"
            )
            seen.add(pair)
            continue
        seen.add(pair)
        kept.append((cost, ei, si, pix, yards, dt, "swap"))
        kept.append((recip[0], recip[1], recip[2], recip[3], recip[4], recip[5], "swap"))
    # A swap exchanges the two ids only while both of the linked tracklets
    # are still running. Later cuts keep the original id, so no fresh id
    # is needed and the chain does not stop at the first gap.
    pairs = {}
    for cost, ei, si, pix, yards, dt, _kind in kept:
        end, start = tracklets[ei], tracklets[si]
        key = (start["frames"][0], tuple(sorted((end["tid"], start["tid"]))))
        rec = pairs.setdefault(key, {
            "start": start["frames"][0],
            "end": start["frames"][-1],
            "ids": [],
            "cost": cost,
            "pix": pix,
            "yards": yards,
        })
        rec["end"] = max(rec["end"], start["frames"][-1])
        if end["tid"] not in rec["ids"]:
            rec["ids"].append(end["tid"])
        if start["tid"] not in rec["ids"]:
            rec["ids"].append(start["tid"])
        if cost < rec["cost"]:
            rec["cost"] = cost
            rec["pix"] = pix
            rec["yards"] = yards
    accepted = []
    used_ids = set()
    for rec in sorted(pairs.values(), key=lambda item: item["start"]):
        tid_a, tid_b = rec["ids"]
        if tid_a in used_ids or tid_b in used_ids:
            print(
                f"relink refuse swap f{rec['start']} T{tid_a}/T{tid_b} "
                f"id already exchanged"
            )
            continue
        used_ids.add(tid_a)
        used_ids.add(tid_b)
        accepted.append(rec)

    def _mapped(frame, orig):
        for rec in accepted:
            if frame < rec["start"] or frame > rec["end"]:
                continue
            tid_a, tid_b = rec["ids"]
            if orig not in (tid_a, tid_b):
                continue
            if frame not in by_id[tid_a] or frame not in by_id[tid_b]:
                continue
            return tid_b if orig == tid_a else tid_a
        return orig

    def _rewrite():
        out = [[] for _ in tracks]
        for tid, fmap in by_id.items():
            for fr, tr0 in fmap.items():
                tr = dict(tr0)
                tr["id"] = _mapped(fr, tid)
                out[fr].append(tr)
        return out

    def _pos_jumps(published):
        series = {}
        for fr, frame in enumerate(published):
            for tr in frame:
                series.setdefault(tr["id"], {})[fr] = tr["pos"]
        found = []
        for nid, fmap in series.items():
            ordered = sorted(fmap)
            for prev, nxt in zip(ordered, ordered[1:]):
                if nxt == prev + 1 and _dist(fmap[prev], fmap[nxt]) > 8.0:
                    found.append((nid, prev, nxt))
        return found

    while accepted:
        trial = _rewrite()
        jumps = _pos_jumps(trial)
        if not jumps:
            break
        nid, prev, nxt = jumps[0]
        dropped = None
        for item in accepted:
            if item["start"] <= nxt and nid in item["ids"]:
                dropped = item
        if dropped is None:
            print(f"relink refuse jump f{prev}->{nxt} id {nid}, original ids kept")
            accepted = []
            break
        accepted = [item for item in accepted if item is not dropped]
        print(
            f"relink refuse jump f{prev}->{nxt} "
            f"T{dropped['ids'][0]}/T{dropped['ids'][1]} "
            f"f{dropped['start']}-{dropped['end']}"
        )

    rewritten = [[] for _ in tracks]
    renamed = 0
    for tid, fmap in by_id.items():
        for fr, tr0 in fmap.items():
            tr = dict(tr0)
            new_id = _mapped(fr, tid)
            if new_id != tid:
                renamed += 1
            tr["id"] = new_id
            rewritten[fr].append(tr)
    for fr, frame in enumerate(rewritten):
        seen_ids = set()
        clean = []
        for tr in frame:
            if tr["id"] in seen_ids:
                print(f"relink duplicate id {tr['id']} on frame {fr}")
                print("relink original ids kept")
                return tracks
            seen_ids.add(tr["id"])
            clean.append(tr)
        rewritten[fr] = clean

    after_ids = {tr["id"] for frame in rewritten for tr in frame}
    created = sorted(after_ids - set(by_id))
    before_main = len(by_id.get(main_id, {})) if main_id is not None else 0
    after_main = sum(
        1 for frame in rewritten for tr in frame if tr["id"] == main_id)
    if created or (main_id is not None and after_main < before_main):
        print(
            f"relink rejected: created={created or '(none)'} "
            f"main {before_main} -> {after_main}"
        )
        print("relink links=0 renamed_frames=0")
        _print_ids(tracks)
        return tracks
    print(f"relink links={len(accepted)} renamed_frames={renamed}")
    for rec in accepted:
        tid_a, tid_b = rec["ids"]
        print(
            f"  swap f{rec['start']}-{rec['end']} T{tid_a}<->T{tid_b} "
            f"cost={rec['cost']:.2f} pix={rec['pix']:.0f} yd={rec['yards']:.2f}"
        )
    _print_ids(rewritten)
    return rewritten


# ---------------------------------------------------------------- rendering
def _label(img, text, org, scale):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = int(org[0]), int(org[1])
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _jersey_box(player_box):
    """Return the upper-torso portion of a full-player tracking box."""
    x, y, width, height = player_box
    left = int(round(x + 0.08 * width))
    top = int(round(y + 0.12 * height))
    jersey_width = max(1, int(round(0.84 * width)))
    jersey_height = max(1, int(round(0.50 * height)))
    return left, top, jersey_width, jersey_height


def _with_team_classification(frame, tracks, prototypes):
    """Add single-frame jersey evidence to already-associated tracks."""
    if prototypes is None:
        return tracks

    team_a, team_b = prototypes
    classified = []
    for track in tracks:
        item = dict(track)
        try:
            result = classify_jersey_region(
                crop_region(frame, _jersey_box(track["bbox"])),
                team_a,
                team_b,
            )
        except ValueError:
            item.update(
                team_name="UNKNOWN",
                team_conf=0.0,
                color_coverage=0.0,
            )
        else:
            item.update(
                team_name=result.label,
                team_conf=result.confidence,
                color_coverage=result.matched_fraction,
            )
        classified.append(item)
    return classified


def _box_iou(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _circular_median_hue(hues):
    """Median OpenCV hue, shifted so the wraparound sits away from the cluster."""
    hues = np.asarray(hues, dtype=np.float64)
    angles = hues * (2.0 * np.pi / 180.0)
    center = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    shifted = (angles - center + np.pi) % (2.0 * np.pi) - np.pi
    median_angle = float(np.median(shifted) + center)
    return float((median_angle % (2.0 * np.pi)) * 180.0 / (2.0 * np.pi))


def _hsv_to_bgr(hsv):
    pixel = np.array(
        [[[int(round(hsv[0])) % 180, int(round(hsv[1])), int(round(hsv[2]))]]],
        dtype=np.uint8,
    )
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(channel) for channel in bgr)


def _apply_track_team_votes(rows, prototypes):
    """Keep a stable team label, repairing only an opposite-colour prefix.

    A stitched path can begin on the other team and later settle onto the
    player that owns the track id. That happens to red number 72 at the start
    of Clip 4. Conversely, a tackle-pile crop can temporarily contain more of
    the opponent than the tracked player. Per-frame overrides therefore cause
    flicker and label the occluded red player blue. We find one offline colour
    change point only when the path opens with sustained opposing evidence;
    all later contradictory crops retain the stable tracker team.
    """
    by_id = {}
    for row in rows:
        by_id.setdefault(row[1], []).append(row)
    tracker_labels = sorted({str(row[2]).strip().lower() for row in rows})
    display_of = {}
    tracker_for_display = {}
    for tracker, proto in zip(tracker_labels, prototypes or ()):
        display = str(proto.label).strip()
        display_of[tracker] = display
        tracker_for_display[display.lower()] = tracker
    for group in by_id.values():
        group.sort(key=lambda row: int(row[0]))
        tracker_counts = {}
        for row in group:
            key = str(row[2]).strip().lower()
            tracker_counts[key] = tracker_counts.get(key, 0) + 1
        tracker = max(tracker_counts, key=tracker_counts.get)
        display = display_of.get(tracker, _display_team_name(tracker))
        agree = 0.0
        total = 0.0
        evidence = []
        for row in group:
            name = str(row[3]).strip()
            conf = row[7]
            coverage = row[8]
            if name == "" or name.upper() == "UNKNOWN" or conf == "":
                evidence.append((None, 0.0, 0.0))
                continue
            conf = float(conf)
            coverage = 0.0 if coverage == "" else float(coverage)
            if conf <= 0:
                evidence.append((None, 0.0, coverage))
                continue
            total += conf
            if name.lower() == display.lower():
                agree += conf
            evidence.append((tracker_for_display.get(name.lower()), conf, coverage))
        share = 0.0 if total == 0 else round(agree / total, 4)
        for row in group:
            row[3] = display
            row[7] = share

        # An opening segment needs repeated, usable opponent-colour evidence.
        # A single bad opening crop must not repaint the track.
        opening = {}
        for measured_tracker, conf, coverage in evidence[:7]:
            if (
                measured_tracker is None
                or measured_tracker == tracker
                or conf < 0.55
                or coverage < 0.20
            ):
                continue
            count, score = opening.get(measured_tracker, (0, 0.0))
            opening[measured_tracker] = (count + 1, score + conf)
        candidates = [
            (score, count, measured_tracker)
            for measured_tracker, (count, score) in opening.items()
            if count >= 3 and score >= 1.8
        ]
        if not candidates:
            continue
        _opening_score, _opening_count, opponent = max(candidates)

        # Maximise evidence for [opponent prefix | tracker suffix]. Short
        # contaminated runs inside the prefix cancel out when the original
        # colour returns, while a durable identity switch creates the peak.
        cumulative = 0.0
        best_score = 0.0
        cut = 0
        for index, (measured_tracker, conf, coverage) in enumerate(evidence):
            if conf >= 0.55 and coverage >= 0.20:
                if measured_tracker == opponent:
                    cumulative += conf
                elif measured_tracker == tracker:
                    cumulative -= conf
            if cumulative > best_score:
                best_score = cumulative
                cut = index + 1
        if cut == 0 or best_score < 1.8:
            continue

        prefix_votes = [
            conf for measured_tracker, conf, coverage in evidence[:cut]
            if measured_tracker == opponent and conf >= 0.55 and coverage >= 0.20
        ]
        prefix_conf = round(float(np.mean(prefix_votes)), 4) if prefix_votes else share
        opponent_display = display_of.get(opponent, _display_team_name(opponent))
        for row, (measured_tracker, conf, coverage) in zip(group[:cut], evidence[:cut]):
            row[3] = opponent_display
            if measured_tracker == opponent and conf > 0 and coverage >= 0.20:
                row[7] = round(conf, 4)
            else:
                row[7] = prefix_conf


def _team_choices_by_frame(rows):
    """Display-team choices keyed by frame and track, not only by track.

    A track can briefly follow a differently coloured player. Keying only by
    id made its final colour overwrite every earlier frame in the rendered
    video (for example, blue number 79 appeared red from the opening frame).
    """
    return {
        (int(row[0]), int(row[1])): (row[3], row[7])
        for row in rows
    }


def _display_team_name(tracker_label):
    text = str(tracker_label).strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def _auto_team_prototypes(video_path, boxes_path, team_names):
    """Build two jersey prototypes from tracker boxes.

    ``team_names`` is an optional pair of display names. The first name
    belongs to the alphabetically first tracker team label, and the second
    name belongs to the other label. When the pair is omitted, each display
    name is the tracker label with its first letter capitalised.
    """
    min_saturation = 55
    min_value = 35
    min_crops = 8
    max_crops = 40
    min_conf = 0.25
    min_width = 36.0
    min_height = 64.0
    max_iou = 0.15

    by_frame = {}
    with open(boxes_path, newline="") as handle:
        for row in csv.DictReader(handle):
            frame_i = int(float(row["frame"]))
            box = (
                float(row["x"]),
                float(row["y"]),
                float(row["w"]),
                float(row["h"]),
            )
            by_frame.setdefault(frame_i, []).append({
                "team": row["team"],
                "box": box,
                "conf": float(row["conf"]),
            })

    eligible = {}
    for frame_i, slots in by_frame.items():
        chosen = {}
        for slot in slots:
            width = slot["box"][2]
            height = slot["box"][3]
            if slot["conf"] < min_conf or width < min_width or height < min_height:
                continue
            if any(
                _box_iou(slot["box"], other["box"]) > max_iou
                for other in slots
                if other is not slot
            ):
                continue
            area = width * height
            current = chosen.get(slot["team"])
            if current is None or area > current[0]:
                chosen[slot["team"]] = (area, slot["box"])
        for team, (_area, box) in chosen.items():
            eligible.setdefault(team, []).append((frame_i, box))

    teams = sorted(eligible)
    if len(teams) < 2:
        counts = {team: len(eligible.get(team, [])) for team in teams}
        print(
            "WARNING: automatic team calibration found "
            f"{counts or 'no'} eligible crops. "
            "Running without team classification."
        )
        return None
    if len(teams) > 2:
        teams = sorted(teams, key=lambda team: len(eligible[team]), reverse=True)[:2]
        teams = sorted(teams)

    if team_names:
        names = [part.strip() for part in team_names.split(",")]
        if len(names) != 2 or not names[0] or not names[1] or names[0] == names[1]:
            raise ValueError(
                "--team-names needs two different labels, in alphabetical "
                "order of the tracker team column."
            )
    else:
        names = [_display_team_name(team) for team in teams]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(
            "WARNING: automatic team calibration could not open the video. "
            "Running without team classification."
        )
        return None

    per_team = {}
    for team in teams:
        picks = eligible[team]
        if len(picks) > max_crops:
            indexes = np.linspace(0, len(picks) - 1, max_crops)
            picks = [picks[int(round(index))] for index in indexes]
            # A shared frame can be chosen twice by rounding. Keep one box.
            unique = {}
            for frame_i, box in picks:
                unique[frame_i] = box
            picks = sorted(unique.items())
        per_team[team] = picks

    frames_needed = {}
    for team, picks in per_team.items():
        for frame_i, box in picks:
            frames_needed.setdefault(frame_i, []).append((team, box))

    built = {team: [] for team in teams}
    last_frame = max(frames_needed)
    frame_i = 0
    while frame_i <= last_frame:
        ok, frame = _take(cap)
        if not ok:
            break
        if frame_i in frames_needed:
            crop_x0 = 0 if _CONTENT_X is None else int(_CONTENT_X[0])
            for team, box in frames_needed[frame_i]:
                shifted = (box[0] - crop_x0, box[1], box[2], box[3])
                try:
                    prototype = build_color_prototype(
                        crop_region(frame, _jersey_box(shifted)),
                        names[teams.index(team)],
                        min_saturation=min_saturation,
                        min_value=min_value,
                    )
                except ValueError:
                    continue
                built[team].append(prototype)
        frame_i += 1
    cap.release()

    short = [team for team in teams if len(built[team]) < min_crops]
    if short:
        used = ", ".join(f"{team}={len(built[team])}" for team in teams)
        print(
            "WARNING: automatic team calibration needs at least "
            f"{min_crops} torso crops per team ({used}). "
            "Running without team classification."
        )
        return None

    prototypes = []
    crop_counts = {}
    for team, label in zip(teams, names):
        samples = built[team]
        hsv = (
            _circular_median_hue([sample.hsv[0] for sample in samples]),
            float(np.median([sample.hsv[1] for sample in samples])),
            float(np.median([sample.hsv[2] for sample in samples])),
        )
        crop_counts[team] = len(samples)
        prototypes.append(ColorPrototype(
            label=label,
            hsv=hsv,
            bgr=_hsv_to_bgr(hsv),
            sample_pixel_count=int(sum(sample.sample_pixel_count for sample in samples)),
        ))

    separation = prototype_distance(prototypes[0], prototypes[1])
    print(
        f"automatic team colors: {prototypes[0].label} from {crop_counts[teams[0]]} "
        f"{teams[0]} crops, {prototypes[1].label} from {crop_counts[teams[1]]} "
        f"{teams[1]} crops"
    )
    print(f"{prototypes[0].label} HSV: {tuple(round(v, 1) for v in prototypes[0].hsv)}")
    print(f"{prototypes[1].label} HSV: {tuple(round(v, 1) for v in prototypes[1].hsv)}")
    print(f"Team-color separation: {separation:.3f}")
    if separation < 0.15:
        print(
            "WARNING: The selected team colors are very similar. "
            "Classifications may be uncertain."
        )
    return tuple(prototypes), {
        "input": video_path,
        "frame_index": None,
        "calibration_report": None,
        "automatic": True,
        "crops_used": crop_counts,
        "team_a": {
            "label": prototypes[0].label,
            "hsv": [round(value, 2) for value in prototypes[0].hsv],
            "bgr": list(prototypes[0].bgr),
            "sample_pixel_count": prototypes[0].sample_pixel_count,
            "tracker_team": teams[0],
            "crops_used": crop_counts[teams[0]],
        },
        "team_b": {
            "label": prototypes[1].label,
            "hsv": [round(value, 2) for value in prototypes[1].hsv],
            "bgr": list(prototypes[1].bgr),
            "sample_pixel_count": prototypes[1].sample_pixel_count,
            "tracker_team": teams[1],
            "crops_used": crop_counts[teams[1]],
        },
        "prototype_distance": round(separation, 4),
        "classifications": [],
    }


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
        visible_team = str(d.get("team_name", "")).strip().lower()
        color_team = visible_team if visible_team in TEAM_BGR else d["team"]
        c = TEAM_BGR.get(color_team, (0, 255, 255))
        cv2.rectangle(left, (x0, y0), (x0 + bw, y0 + bh), c, 2)
        foot = (int(x0 + bw / 2), int(y0 + bh))
        cv2.circle(left, foot, 5, c, -1)
        if d.get("id") is not None:
            text = str(d["id"])
            if d.get("team_name") is not None:
                team_conf = round(100 * d.get("team_conf", 0.0))
                text = f"{d['id']} {d['team_name']} {team_conf}%"
                if d.get("conf") is not None:
                    text += f" | trk {round(100 * d['conf'])}%"
            elif d.get("conf") is not None:
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
        visible_team = str(tr.get("team_name", "")).strip().lower()
        color_team = visible_team if visible_team in TEAM_BGR else tr["team"]
        c = TEAM_BGR.get(color_team, (0, 255, 255))
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
    n_full = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"frames: {n_full}", flush=True)
    cached = load_homographies(n_full)
    n = n_full
    if STOP_AT is not None:
        n = min(n_full, max(0, int(STOP_AT)))
        print(f"stop-at: {n}", flush=True)
    if cached is not None:
        cap.release()
        publish(n, cached[0][:n], cached[1][:n])
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
    # A short run must not replace a full-clip homography cache.
    if n == n_full:
        save_homographies(A, status)
    publish(n, A, status)


def publish(n, A, status):
    if PLAYER_BOXES:
        render_external_boxes(n, A, status)
    else:
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


def _torso_peaks(mask, bbox):
    """How many team-color columns sit in the upper half of bbox.

    A digit dips the column sum without splitting the jersey. Two torsos
    leave a grass gap, so a valley has to fall under 30% of the shorter
    peak before it counts as a second body.
    """
    height, width = mask.shape[:2]
    x, y, w, h = [int(round(float(v))) for v in bbox]
    x0, y0 = max(0, x), max(0, y)
    x1 = min(width, x + max(w, 1))
    y1 = min(height, y + max(int(round(h * 0.55)), 8))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0
    cols = mask[y0:y1, x0:x1].sum(axis=0).astype(np.float32)
    smooth = np.convolve(cols, np.ones(7) / 7.0, mode="same")
    raw = []
    for i in range(1, len(smooth) - 1):
        if smooth[i] < 6:
            continue
        if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]:
            raw.append(i)
    merged = []
    for peak in raw:
        if merged and peak - merged[-1] < 14:
            if smooth[peak] > smooth[merged[-1]]:
                merged[-1] = peak
        else:
            merged.append(peak)
    kept = []
    for peak in merged:
        if not kept:
            kept.append(peak)
            continue
        prev = kept[-1]
        valley = float(smooth[prev:peak + 1].min())
        if valley > 0.30 * min(float(smooth[peak]), float(smooth[prev])):
            if smooth[peak] > smooth[prev]:
                kept[-1] = peak
            continue
        kept.append(peak)
    return len(kept)


def _one_torso(hsv, mask, box_a, box_b):
    """True when the union shows one jersey number or one torso peak."""
    union = _box_union(box_a, box_b)
    nums = len(_jersey_numbers(hsv, mask, union))
    if nums >= 2:
        return False
    if nums == 1:
        return True
    return _torso_peaks(mask, union) == 1


def _feet_stacked(box_a, box_b):
    """One body: the boxes nest and the feet stay in one column.

    Side-by-side linemen, and two players whose feet stay more than 36 px
    apart, fail this. A one- or two-frame cross in a pile fails the run
    length checked by the caller.
    """
    if _iou_over_smaller(box_a, box_b) < 0.88:
        return False
    ax, ay = _foot_of(box_a)
    bx, by = _foot_of(box_b)
    return abs(ax - bx) <= 14 and abs(ay - by) <= 36


def _tracklet_age(by_id, tid, frame):
    """Consecutive frames ending here whose feet do not jump."""
    boxes = by_id[tid]
    age = 0
    f = frame
    while f in boxes:
        age += 1
        prev = f - 1
        if prev not in boxes:
            break
        if _dist(_foot_of(boxes[f]), _foot_of(boxes[prev])) > 48:
            break
        f = prev
    return age


def _weaker_id(id_a, id_b, run, by_id, conf_of, main_ids):
    """The id with the shorter continuous history on this body.

    Close ages mean both have been here; the smaller box is the fragment.
    The main id is never the one returned.
    """
    ages = {
        id_a: _tracklet_age(by_id, id_a, run[-1]),
        id_b: _tracklet_age(by_id, id_b, run[-1]),
    }
    areas = {}
    confs = {}
    for tid in (id_a, id_b):
        areas[tid] = float(np.median(
            [by_id[tid][t][2] * by_id[tid][t][3] for t in run]))
        confs[tid] = float(np.median([conf_of[tid][t] for t in run]))
    if abs(ages[id_a] - ages[id_b]) >= 8:
        order = sorted((id_a, id_b), key=lambda tid: (ages[tid], areas[tid]))
    else:
        order = sorted(
            (id_a, id_b),
            key=lambda tid: (areas[tid], ages[tid], confs[tid]))
    drop = order[0]
    if drop in main_ids:
        drop = order[1]
    if drop in main_ids:
        return None
    return drop


def _suppress_same_body(tracks, frames, main_ids):
    """Drop a same-team box that has shared one torso for 6 frames or more."""
    if not SAME_BODY:
        return tracks
    by_id = {}
    team_of = {}
    conf_of = {}
    for t, frame_tracks in enumerate(tracks):
        for tr in frame_tracks:
            tid = tr["id"]
            team_of[tid] = tr["team"]
            by_id.setdefault(tid, {})[t] = tr["bbox"]
            conf_of.setdefault(tid, {})[t] = float(tr.get("conf") or 0.0)
    ids = sorted(by_id)
    geom = {}
    for i, id_a in enumerate(ids):
        for id_b in ids[i + 1:]:
            if team_of[id_a] != team_of[id_b]:
                continue
            both = sorted(set(by_id[id_a]) & set(by_id[id_b]))
            run = []
            runs = []
            for t in both:
                if _feet_stacked(by_id[id_a][t], by_id[id_b][t]):
                    if run and t != run[-1] + 1:
                        runs.append(run)
                        run = []
                    run.append(t)
                elif run:
                    runs.append(run)
                    run = []
            if run:
                runs.append(run)
            long = [item for item in runs if len(item) >= 6]
            if long:
                geom[(id_a, id_b)] = long
    needed = {t for runs in geom.values() for run in runs for t in run}
    color_ok = {}
    if needed:
        for t, frame in _iter_frames(len(tracks), frames):
            if t not in needed:
                continue
            _memo_begin(t)
            hsv = _hsv_of(frame)
            red, blue = _team_masks(hsv)
            red = red.copy()
            blue = blue.copy()
            for (id_a, id_b), runs in geom.items():
                if not any(t in run for run in runs):
                    continue
                mask = red if team_of[id_a] == "red" else blue
                color_ok[(id_a, id_b, t)] = _one_torso(
                    hsv, mask, by_id[id_a][t], by_id[id_b][t])
    drops = set()
    for (id_a, id_b), runs in geom.items():
        for run in runs:
            group = []
            groups = []
            for t in run:
                if color_ok.get((id_a, id_b, t)):
                    if group and t != group[-1] + 1:
                        groups.append(group)
                        group = []
                    group.append(t)
                elif group:
                    groups.append(group)
                    group = []
            if group:
                groups.append(group)
            for group in groups:
                if len(group) < 6:
                    continue
                drop = _weaker_id(
                    id_a, id_b, group, by_id, conf_of, main_ids)
                if drop is None:
                    continue
                for t in group:
                    drops.add((t, drop))
    before = {tid: len(fmap) for tid, fmap in by_id.items()}
    lose = {}
    for _t, tid in drops:
        lose[tid] = lose.get(tid, 0) + 1
    for tid, lost in lose.items():
        if before[tid] - lost <= 0:
            drops = {item for item in drops if item[1] != tid}
    out = []
    for t, frame_tracks in enumerate(tracks):
        out.append([tr for tr in frame_tracks if (t, tr["id"]) not in drops])
    after = {}
    for frame_tracks in out:
        for tr in frame_tracks:
            after[tr["id"]] = after.get(tr["id"], 0) + 1
    print(f"same-body dropped {len(drops)} boxes")
    for tid in sorted(before):
        kept = after.get(tid, 0)
        if kept == before[tid]:
            continue
        lost = 100.0 * (before[tid] - kept) / before[tid]
        print(f"  id {tid} frames {before[tid]} -> {kept}")
        if lost > 10.0:
            print(f"  id {tid} lost {lost:.1f}% of its frames")
    return out


def _load_box_lines(path):
    """Header plus data lines from a frame,id,team,x,y,w,h,conf CSV.

    Line endings stay as they are in the file so a full-length copy matches
    the tracker CSV byte for byte.
    """
    raw = open(path, "rb").read()
    if not raw:
        raise RuntimeError(f"empty player boxes: {path}")
    lines = raw.splitlines(keepends=True)
    header = lines[0]
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        frame = int(line.split(b",", 1)[0])
        rows.append((frame, line))
    return header, rows


def _warm_orange_fraction(frame, bbox):
    """Share of the inner box that is orange or red, not field green.

    OpenCV hue wraps at 0. A sideline cone sits in that wrap. A jersey can
    too, so this is only one of the cone tests.
    """
    height, width = frame.shape[:2]
    x, y, bw, bh = bbox
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(width, int(round(x + bw))), min(height, int(round(y + bh)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    crop = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    warm = ((hh <= 15) | (hh >= 170)) & (ss >= 50) & (vv >= 40)
    ih, iw = warm.shape
    ya, yb = ih // 5, ih - ih // 5
    xa, xb = iw // 5, iw - iw // 5
    inner = warm[ya:yb, xa:xb]
    if inner.size == 0:
        return 0.0
    return float(inner.mean())


def _camera_to_reference_transforms(motion, nframes):
    """Return transforms mapping each camera frame back to frame zero."""
    if motion is None or len(motion) == 0 or nframes <= 0:
        return None
    count = min(int(nframes), len(motion))
    transforms = [np.eye(3, dtype=np.float64)]
    for frame_i in range(1, count):
        step = np.eye(3, dtype=np.float64)
        step[:2] = np.asarray(motion[frame_i], dtype=np.float64)
        try:
            inverse = np.linalg.inv(step)
        except np.linalg.LinAlgError:
            inverse = np.eye(3, dtype=np.float64)
        transforms.append(transforms[-1] @ inverse)
    return transforms


def _transform_point(matrix, point):
    value = matrix @ np.array([point[0], point[1], 1.0], dtype=np.float64)
    if abs(value[2]) < 1e-9:
        return None
    value = value[:2] / value[2]
    if not np.isfinite(value).all():
        return None
    return float(value[0]), float(value[1])


def _sideline_cone_ids(samples, frame_hw):
    """Ids whose whole life is a stationary orange cone on the boundary.

    A player standing still before the snap is tall. A cone is a short
    orange box, and its foot stays put on the ground plane: the per-frame
    homographies map that foot to one place in field yards. Player-sized
    tracks set the sideline test, so a cone is the small warm track sitting
    past that group. A painted yard number is not required. Only the
    player-boxes path calls this.
    """
    if not samples or frame_hw is None:
        return {}
    frame_h, frame_w = frame_hw
    tall_y = []
    brief = {}
    for tid, rows in samples.items():
        if len(rows) < 12:
            continue
        widths = [r["bw"] for r in rows]
        heights = [r["bh"] for r in rows]
        med_w = float(np.median(widths))
        med_h = float(np.median(heights))
        small = med_h <= 0.060 * frame_h and med_w <= 0.040 * frame_w
        if not small:
            worlds = [r["world"] for r in rows if r["world"] is not None]
            if worlds:
                tall_y.append(float(np.median([point[1] for point in worlds])))
            continue
        warm = float(np.median([r["warm"] for r in rows]))
        if warm < 0.30:
            continue

        # Camera-compensated image coordinates are the reliable signal when
        # the field fit has gone stale. Cones remain fixed there; players do
        # not. Keep the older ground-plane test as a fallback for callers
        # without detector camera motion.
        cameras = [r.get("camera") for r in rows if r.get("camera") is not None]
        camera_fixed = False
        camera_info = None
        if len(cameras) >= 0.5 * len(rows):
            points = np.asarray(cameras, np.float64)
            center = np.median(points, axis=0)
            distance = np.linalg.norm(points - center, axis=1)
            inlier = distance <= 30.0
            if float(inlier.mean()) >= 0.60:
                spread = points[inlier].std(axis=0)
                camera_fixed = (
                    float(np.median(distance)) <= 18.0
                    and float(spread[0]) <= 24.0
                    and float(spread[1]) <= 14.0
                )
                camera_info = (center, spread, float(np.median(distance)))

        worlds = [r["world"] for r in rows if r["world"] is not None]
        world_fixed = False
        med_x = med_y = xstd = ystd = 0.0
        on_line = False
        if len(worlds) >= 0.5 * len(rows):
            xs = np.array([point[0] for point in worlds], np.float64)
            ys = np.array([point[1] for point in worlds], np.float64)
            med_x = float(np.median(xs))
            med_y = float(np.median(ys))
            dist = np.hypot(xs - med_x, ys - med_y)
            inlier = dist <= 8.0
            if float(inlier.mean()) >= 0.50:
                xstd = float(xs[inlier].std())
                ystd = float(ys[inlier].std())
                world_fixed = xstd <= 2.2 and ystd <= 2.2
            on_line = med_y <= 1.0 or med_y >= 52.3
        if not camera_fixed and not world_fixed:
            continue
        # tall_y is still growing; the formation test is applied after the loop.
        brief[tid] = {
            "n": len(rows),
            "frame0": rows[0]["frame"],
            "frame1": rows[-1]["frame"],
            "med_w": med_w,
            "med_h": med_h,
            "warm": warm,
            "med_x": med_x,
            "med_y": med_y,
            "xstd": xstd,
            "ystd": ystd,
            "on_line": on_line,
            "camera_fixed": camera_fixed,
            "camera_info": camera_info,
        }
    # Formation extent uses every player-sized track, including ones seen
    # after a cone in the dict walk.
    if tall_y:
        lo, hi = min(tall_y), max(tall_y)
    else:
        lo = hi = None
    cones = {}
    for tid, info in brief.items():
        past = False
        if lo is not None and not info["camera_fixed"]:
            past = info["med_y"] >= hi + 6.0 or info["med_y"] <= lo - 6.0
        if not (info["camera_fixed"] or info["on_line"] or past):
            continue
        if info["camera_fixed"]:
            _center, spread, med_distance = info["camera_info"]
            info["why"] = (
                f"camera-fixed; box {info['med_w']:.0f}x{info['med_h']:.0f} "
                f"warm {info['warm']:.2f}; compensated spread "
                f"({spread[0]:.1f},{spread[1]:.1f}) px, median {med_distance:.1f} px"
            )
        else:
            why = "outside y=0..53.3" if info["on_line"] else "past the player formation toward a sideline"
            info["why"] = (
                f"{why}; box {info['med_w']:.0f}x{info['med_h']:.0f} "
                f"warm {info['warm']:.2f}; foot ({info['med_x']:.1f},{info['med_y']:.1f}) "
                f"inlier std ({info['xstd']:.2f},{info['ystd']:.2f}) yd"
            )
        cones[tid] = info
    return cones


def render_external_boxes(n, A, status):
    """Project and draw boxes from PLAYER_BOXES. Field homographies stay as given."""
    header, box_lines = _load_box_lines(PLAYER_BOXES)
    frames = _load_frames(n)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    field_map = cv2.imread(FIELD_MAP) if FIELD_MAP else None
    writer, out_path = None, OUT_VIDEO
    rows = []
    trails = {}
    by_frame = {}
    for frame_i, line in box_lines:
        by_frame.setdefault(frame_i, []).append(line)
    read_n = 0
    last_world = {}
    cone_samples = {}
    frame_hw = None
    camera_transforms = _camera_to_reference_transforms(CAMERA_MOTION, n)
    for i, frame in _iter_frames(n, frames):
        if i >= len(A):
            break
        if frame_hw is None:
            frame_hw = frame.shape[:2]
        crop_x0 = 0 if _CONTENT_X is None else int(_CONTENT_X[0])
        for line in by_frame.get(i, []):
            parts = next(csv.reader([line.decode("utf-8").strip("\r\n")]))
            if len(parts) < 8:
                continue
            _frame, tid, _team, x, y, bw, bh, _conf = parts[:8]
            bbox = (
                int(round(float(x) - crop_x0)),
                int(round(float(y))),
                int(round(float(bw))),
                int(round(float(bh))),
            )
            # The published foot still uses _external_body, which drops a
            # point past y=50. A cone on a fresh fit lands in that region,
            # so the cone test keeps the raw ground-plane foot.
            foot_box = (
                float(x) - crop_x0,
                float(y),
                float(bw),
                float(bh),
            )
            raw = _project_point(*_foot_of(foot_box), A[i])
            if raw is not None and not np.isfinite(raw).all():
                raw = None
            camera = None
            if camera_transforms is not None and i < len(camera_transforms):
                camera = _transform_point(
                    camera_transforms[i],
                    _foot_of((float(x), float(y), float(bw), float(bh))),
                )
            cone_samples.setdefault(int(float(tid)), []).append({
                "frame": i,
                "bw": float(bw),
                "bh": float(bh),
                "world": None if raw is None else (float(raw[0]), float(raw[1])),
                "camera": camera,
                "warm": _warm_orange_fraction(frame, bbox),
            })
    cones = _sideline_cone_ids(cone_samples, frame_hw)
    if cones:
        print(f"sideline cones removed: {len(cones)}")
        for tid in sorted(cones):
            info = cones[tid]
            print(
                f"  id {tid} frames {info['frame0']}-{info['frame1']} "
                f"({info['n']}) {info['why']}"
            )
    else:
        print("sideline cones removed: 0")
    cone_ids = set(cones)
    pending_draw = []
    for i, frame in _iter_frames(n, frames):
        if i >= len(A) or i >= len(status):
            break
        read_n = i + 1
        crop_x0 = 0 if _CONTENT_X is None else int(_CONTENT_X[0])
        active = []
        mapped = []
        parsed = []
        accepted = []
        for line in by_frame.get(i, []):
            parts = next(csv.reader([line.decode("utf-8").strip("\r\n")]))
            if len(parts) < 8:
                continue
            _frame, tid, team, x, y, bw, bh, conf = parts[:8]
            if int(float(tid)) in cone_ids:
                continue
            x = float(x)
            y = float(y)
            bw = float(bw)
            bh = float(bh)
            bbox = (
                int(round(x - crop_x0)),
                int(round(y)),
                int(round(bw)),
                int(round(bh)),
            )
            body = _external_body(bbox, A[i])
            slot = {
                "id": int(float(tid)),
                "team": team,
                "bbox": bbox,
                "conf": float(conf),
                "body": body,
            }
            parsed.append(slot)
            if body is not None:
                accepted.append(body)
        on_field = {id(item) for item in _on_this_field(accepted)}
        for slot in parsed:
            body = slot["body"]
            world = None
            if body is not None and id(body) in on_field:
                prev = last_world.get(slot["id"])
                world = body["world"] if prev is None else _clamp_step(prev, body["world"])
                last_world[slot["id"]] = world
            if world is None:
                x_yd, y_yd = "", ""
            else:
                x_yd = round(world[0], 2)
                y_yd = round(world[1], 2)
            item = {
                "id": slot["id"],
                "team": slot["team"],
                "bbox": slot["bbox"],
                "conf": slot["conf"],
                "pos": world if world is not None else (0.0, 0.0),
            }
            rows.append([
                i, item["id"], slot["team"], "",
                x_yd, y_yd,
                round(slot["conf"], 4),
                "", "",
            ])
            active.append(item)
            if world is not None:
                hist = trails.setdefault(item["id"], [])
                hist.append(world)
                mapped.append(dict(item, trail=list(hist)))
        if TEAM_PROTOTYPES is not None:
            active = _with_team_classification(frame, active, TEAM_PROTOTYPES)
            by_id = {tr["id"]: tr for tr in active}
            for tr in mapped:
                src = by_id.get(tr["id"])
                if src is None:
                    continue
                tr["team_name"] = src.get("team_name")
                tr["team_conf"] = src.get("team_conf")
                tr["color_coverage"] = src.get("color_coverage")
                if "team_conf" in src:
                    for row in rows:
                        if row[0] == i and row[1] == tr["id"]:
                            row[3] = src.get("team_name", "")
                            row[7] = round(src["team_conf"], 4)
                            row[8] = round(src["color_coverage"], 4)
        if WRITE_VIDEO or i in SAMPLE_AT:
            drawn = []
            for tr in active:
                twin = dict(tr)
                twin["trail"] = []
                drawn.append(twin)
            pending_draw.append((i, mapped, drawn))
        if i % 40 == 0:
            print(f"boxes {i}/{n} tracks={len(active)}", flush=True)
    if TEAM_PROTOTYPES is not None:
        _apply_track_team_votes(rows, TEAM_PROTOTYPES)
        voted = _team_choices_by_frame(rows)
        for frame_i, mapped, drawn in pending_draw:
            for tr in mapped:
                choice = voted.get((frame_i, tr["id"]))
                if choice is None:
                    continue
                tr["team_name"], tr["team_conf"] = choice
            for tr in drawn:
                choice = voted.get((frame_i, tr["id"]))
                if choice is None:
                    continue
                tr["team_name"], tr["team_conf"] = choice
    writer = None
    pending_i = 0
    if pending_draw:
        for i, frame in _iter_frames(n, frames):
            if pending_i >= len(pending_draw):
                break
            if i != pending_draw[pending_i][0]:
                continue
            _frame_i, mapped, drawn = pending_draw[pending_i]
            pending_i += 1
            combo = render_aligned(frame, mapped, drawn, status[i], field_map)
            if i in SAMPLE_AT:
                cv2.imwrite(f"{SAMPLE_DIR}/composite_{i:04d}.png", combo)
            if WRITE_VIDEO:
                if writer is None:
                    writer, out_path = make_writer(combo.shape)
                writer.write(combo)
    if writer is not None:
        writer.release()

    with open(OUT_CSV, "w", newline="") as handle:
        wcsv = csv.writer(handle)
        wcsv.writerow([
            "frame", "id", "team", "team_name", "x_yd", "y_yd",
            "track_conf", "team_conf", "color_coverage",
        ])
        wcsv.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")

    written = 0
    with open(OUT_BOXES, "wb") as handle:
        handle.write(header if header.endswith(b"\n") else header + b"\n")
        for frame_i, line in box_lines:
            if frame_i >= read_n:
                continue
            tid = int(float(line.split(b",", 2)[1]))
            if tid in cone_ids:
                continue
            handle.write(line if line.endswith(b"\n") else line + b"\n")
            written += 1
    print(f"wrote {OUT_BOXES} ({written} rows)")

    if writer is not None:
        chk = cv2.VideoCapture(out_path)
        nchk = int(chk.get(cv2.CAP_PROP_FRAME_COUNT))
        ok1, _ = chk.read()
        chk.set(cv2.CAP_PROP_POS_FRAMES, max(0, nchk - 1))
        ok2, _ = chk.read()
        print(f"playback check: frames={nchk} first={ok1} last={ok2}")


def render_pass(n, A, status):
    # Detect every frame, then lock the snap roster and carry those
    # numbers forward and backward. The blue ring is followed with the
    # same forward-backward point check as the field landmarks.
    global _LOOSE_LINE, _OVERLAY_FRAMES
    _LOOSE_LINE = None
    _OVERLAY_FRAMES = []
    _memo_reset()
    frames = _load_frames(n)
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
    playable = []
    for i, frame in _iter_frames(n, frames):
        _memo_begin(i)
        if shape is None:
            shape = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not _on_grass(frame):
            per_frame.append([])
            rings.append(None)
            arrows.append(None)
            _OVERLAY_FRAMES.append(None)
            playable.append(False)
            prev_gray = gray
            lost += 1
            arrow_gap += 1
            if i % 40 == 0:
                print(f"detect {i}/{n} cands=0 ring=False arrow=None", flush=True)
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
        hsv = _hsv_of(frame)
        # The mask is applied to the published boxes. Filtering bodies here
        # changes which id follows which player.
        if OVERLAY_MASK:
            mask = _screen_overlay_mask(frame)
        else:
            mask = None
        _OVERLAY_FRAMES.append(mask)
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
    per_frame = _supported(per_frame)
    roster = select_roster(per_frame[0])
    # Ids are already assigned from the snap feet. Trimming the '1' off
    # the main box moves his feet onto his body without renumbering him.
    if frames and roster:
        hsv0 = _hsv_of(frames[0])
        for r in roster:
            had = bool(r.pop("marker_trim", False))
            if not r.get("main"):
                continue
            trimmed = _trim_marker_box(hsv0, r["bbox"])
            if trimmed != tuple(int(round(v)) for v in r["bbox"]):
                r["bbox"] = trimmed
                had = True
            if not had:
                continue
            fx, fy = _foot_of(r["bbox"])
            world = _project_point(fx, fy, A[0])
            if world is not None:
                r["world"] = world
    per_frame, removed = drop_stationary(per_frame)
    print(f"removed {removed} stationary decal detections")
    n_red = sum(1 for r in roster if r["team"] == "red")
    n_blue = sum(1 for r in roster if r["team"] == "blue")
    print(f"opening roster: {n_red} red, {n_blue} blue")
    for r in roster:
        flag = " main" if r.get("main") else ""
        print(f"  {r['id']:2} {r['team']:5} ({r['world'][0]:5.1f}, {r['world'][1]:5.1f}){flag}")
    tracked = track_bidirectional(
        per_frame, roster, A, shape, rings, arrows, frames)
    for t, ok in enumerate(playable):
        if not ok:
            tracked[t] = []
    if RELINK:
        tracked = _relink_published(tracked, frames, shape, roster)

    os.makedirs(SAMPLE_DIR, exist_ok=True)
    field_map = cv2.imread(FIELD_MAP) if FIELD_MAP else None
    writer, out_path = None, OUT_VIDEO
    rows = []
    box_rows = []
    # bbox is on the cropped frame. boxes.csv is source-video pixels.
    crop_x0 = 0 if _CONTENT_X is None else int(_CONTENT_X[0])

    def _source_px(value):
        number = round(float(value), 2)
        if number == int(number):
            return int(number)
        return number

    filtered = []
    for i, frame in _iter_frames(n, frames):
        row = tracked[i]
        if OVERLAY_MASK:
            mask = None
            if _OVERLAY_FRAMES is not None and i < len(_OVERLAY_FRAMES):
                mask = _OVERLAY_FRAMES[i]
            if mask is None and _on_grass(frame):
                mask = _screen_overlay_mask(frame)
            if mask is not None:
                hsv = _hsv_of(frame)
                row = [tr for tr in row
                       if not _overlay_rejects(
                           hsv, mask, tr["bbox"], i, tr["id"], tr["team"])]
        filtered.append(row)
    main_ids = {r["id"] for r in roster if r.get("main")}
    filtered = _suppress_same_body(filtered, frames, main_ids)

    for i, frame in _iter_frames(n, frames):
        row = filtered[i] if i < len(filtered) else []
        active = _with_team_classification(
            frame, row, TEAM_PROTOTYPES
        )
        for tr in active:
            rows.append([i, tr["id"], tr["team"],
                         tr.get("team_name", ""),
                         round(tr["pos"][0], 2), round(tr["pos"][1], 2),
                         round(tr["conf"], 4),
                         round(tr["team_conf"], 4)
                         if "team_conf" in tr else "",
                         round(tr["color_coverage"], 4)
                         if "color_coverage" in tr else ""])
            x, y, bw, bh = tr["bbox"]
            box_rows.append([
                i, tr["id"], tr["team"],
                _source_px(float(x) + crop_x0),
                _source_px(float(y)),
                _source_px(float(bw)),
                _source_px(float(bh)),
                round(float(tr["conf"]), 4),
            ])
        if WRITE_VIDEO or i in SAMPLE_AT:
            combo = render_aligned(frame, active, active, status[i], field_map)
            if i in SAMPLE_AT:
                cv2.imwrite(f"{SAMPLE_DIR}/composite_{i:04d}.png", combo)
            if WRITE_VIDEO:
                if writer is None:
                    writer, out_path = make_writer(combo.shape)
                writer.write(combo)
        if i % 40 == 0:
            print(f"pass2 {i}/{n} tracks={len(active)}")
    if writer is not None:
        writer.release()

    with open(OUT_CSV, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            "frame", "id", "team", "team_name", "x_yd", "y_yd",
            "track_conf", "team_conf", "color_coverage",
        ])
        wcsv.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")

    with open(OUT_BOXES, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame", "id", "team", "x", "y", "w", "h", "conf"])
        wcsv.writerows(box_rows)
    print(f"wrote {OUT_BOXES} ({len(box_rows)} rows)")

    if writer is not None:
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
    parser.add_argument(
        "--team-calibration",
        help=("JSON report created by src.classify_teams; enables team names "
              "and jersey-classification confidence in the video and CSV."))
    parser.add_argument(
        "--team-names",
        default=None,
        help=("Display names for the two tracker teams when colors are "
              "calibrated automatically from --player-boxes. The first name "
              "maps to the alphabetically first team label in that CSV, and "
              "the second name maps to the other label. Example: Blue,Red. "
              "The default is each tracker label with a capital first letter."))
    parser.add_argument("--no-video", action="store_true",
                        help="Skip writing tracked_players.mp4.")
    parser.add_argument(
        "--player-boxes",
        default=None,
        help=("CSV of frame,id,team,x,y,w,h,conf. Skips player detection "
              "and uses these boxes. Field tracking is unchanged."),
    )
    parser.add_argument(
        "--camera-motion",
        default=None,
        help=("NumPy array of per-frame affine camera motion. Used to reject "
              "small, field-fixed sideline cones when --player-boxes is set."),
    )
    parser.add_argument("--stop-at", type=int, default=None, metavar="N",
                        help="Process only frames 0..N-1. Frame 0 stays the roster frame.")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    CLIP = args.video
    OUT_VIDEO = os.path.join(args.out_dir, "tracked_players.mp4")
    OUT_CSV = os.path.join(args.out_dir, "positions.csv")
    OUT_BOXES = os.path.join(args.out_dir, "boxes.csv")
    HOMOGRAPHY_CACHE = os.path.join(args.out_dir, "homography.npz")
    SAMPLE_DIR = os.path.join(args.out_dir, "frames")
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    REF_ANCHOR = args.anchor
    FIELD_MAP = args.field_map
    WRITE_VIDEO = not args.no_video
    STOP_AT = args.stop_at
    PLAYER_BOXES = args.player_boxes
    CAMERA_MOTION = None
    if args.camera_motion:
        CAMERA_MOTION = np.load(args.camera_motion)
        print(f"loaded camera motion from {args.camera_motion}")
    if args.team_calibration:
        TEAM_PROTOTYPES = load_calibration_report(args.team_calibration)
        print(f"loaded team calibration from {args.team_calibration}")
    elif args.player_boxes:
        built = _auto_team_prototypes(
            args.video, args.player_boxes, args.team_names)
        if built is not None:
            TEAM_PROTOTYPES, report = built
            report_path = os.path.join(args.out_dir, "team_classification.json")
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")
            print(f"wrote {report_path}")
    main()
