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
                        TEAM_BGR, STATUS_BGR, MAP_CROP_X1, LEFT_SCALE, to_map)
from project_players import detect_players

CLIP = "Clip_4 Camera Swing and Distance Change Test.mp4"
REF_FRAME = 0
REF_ANCHOR = -15.0         # fresh x - 15 = absolute yards (verified vs the
                           # orange LOS at the 50 and red 1st-down at the 21)
PXY = fh.PXY
OUT_VIDEO = "data/output/tracked_players.mp4"
OUT_CSV = "data/output/positions.csv"
SAMPLE_AT = {0, 40, 90, 150, 185, 210, 240}
SAMPLE_DIR = "data/output/frames"
HOMOGRAPHY_CACHE = "data/output/homography.npz"
FIELD_MAP = ""


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


def frame_candidates(frame, A):
    """Red and blue bodies, with white pants folded in. Not yet 7 and 7."""
    h, w = frame.shape[:2]
    raw = detect_players(frame)
    colored = [dict(d) for d in raw if d["team"] in ("red", "blue")]
    white = [d for d in raw if d["team"] == "white"]
    used = set()
    for c in colored:
        for i, wd in enumerate(white):
            if i in used or not _same_body(c["bbox"], wd["bbox"]):
                continue
            ww = wd["bbox"][2] * wd["bbox"][3]
            if ww > c["bbox"][2] * c["bbox"][3]:
                continue
            used.add(i)
            c["bbox"] = _box_union(c["bbox"], wd["bbox"])
        x, y, bw, bh = c["bbox"]
        c["feet"] = (x + bw / 2.0, y + float(bh))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kept = []
    for c in colored:
        x, y, bw, bh = c["bbox"]
        if bh < 16 or bw > bh * 4.0 or y + bh > h * 0.92:
            continue
        frac = _color_frac(hsv, c["bbox"], c["team"])
        if frac < 0.12:
            continue
        c["frac"] = frac
        kept.append(c)
    # project feet; slightly looser bottom cutoff than Clip 3 so a player
    # standing just above the banner is kept
    out = []
    Af = A.astype(np.float32)
    for d in kept:
        x0, y0, bw, bh = d["bbox"]
        feet = np.array([[[x0 + bw / 2.0, y0 + float(bh)]]], np.float32)
        xw, yw = cv2.perspectiveTransform(feet, Af)[0, 0]
        corners = np.array([[[x0, y0 + bh]], [[x0 + bw, y0 + bh]]], np.float32)
        wc = cv2.perspectiveTransform(corners, Af)[:, 0, :]
        width_yd = float(np.linalg.norm(wc[1] - wc[0]))
        if not (0.3 <= width_yd <= 6.0):
            continue
        # players on the near edge of the picture project just past y=0
        if not (-8 <= yw <= 48) or not (-12 <= xw <= 122):
            continue
        # the midfield shield is a wide blue patch, wider in yards than a player
        if d["team"] == "blue" and width_yd >= 2.95 and bw > bh:
            continue
        out.append(dict(team=d["team"], world=(float(xw), float(yw)),
                        bbox=(int(x0), int(y0), int(bw), int(bh)),
                        frac=d["frac"], width_yd=width_yd))
    return _merge_stacked(out)


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
    """7 red and 7 blue at the snap. Reds are the front (smaller x, toward
    the end zone the camera faces). Blues are behind them. Numbers run
    left to right. The flat blue field logo is the widest leftover blue."""
    roster = []
    next_id = 1
    reds = [d for d in dets if d["team"] == "red"
            and d["bbox"][2] * d["bbox"][3] >= 1000 and d["bbox"][3] >= 34
            and d["world"][0] < 51]
    # the two halves of the far-left defender sit about 1.4 yd apart
    reds = _pick_spread(reds, sep=1.8, limit=per_team)
    blues = [d for d in dets if d["team"] == "blue"
             and d["bbox"][2] * d["bbox"][3] >= 1000 and d["bbox"][3] >= 34
             and d["world"][0] > 48]
    # torso and feet of the QB project about 2.8 yd apart
    blues = _pick_spread(blues, sep=3.0, limit=per_team + 2)
    while len(blues) > per_team:
        # the NFL shield covers more yards than a player's feet
        widest = max(range(len(blues)), key=lambda i: blues[i]["width_yd"])
        if blues[widest]["width_yd"] < 2.9:
            break
        del blues[widest]
    blues = blues[:per_team]
    for group in (reds, blues):
        for d in sorted(group, key=lambda d: d["world"][1]):
            roster.append(dict(id=next_id, team=d["team"], world=d["world"],
                               bbox=d["bbox"]))
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


def _snap_body(frame, foot, team):
    """A player-shaped blob near the predicted feet. Team color is tried
    first; any non-field body is accepted if the jersey is hidden."""
    u, v = foot
    h, w = frame.shape[:2]
    x0, x1 = max(0, int(u - 70)), min(w, int(u + 70))
    y0, y1 = max(0, int(v - 150)), min(h, int(v + 18))
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
        use = body.astype(np.uint8) * 255
    if int(use.sum()) < 500 * 255:
        return None
    use = cv2.morphologyEx(use, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(use)
    best, best_d = None, 55.0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 280 or bh < 20 or bw > bh * 3.5:
            continue
        fx = x0 + x + bw / 2.0
        fy = y0 + y + bh
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
            if dist > gate and img > 1.2:
                continue
            if dist > gate + 4:
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


def track_bidirectional(per_frame, roster, homographies, shape):
    """Numbers are assigned once, at the snap, and never reused.

    Forward, each frame is matched to where that number was on the previous
    frame. Backward, each frame is matched to where that number is on the
    next frame. Jersey color breaks ties; position is the reference.
    The yard position blends the detection with the step predicted from the
    previous frame. If no blob is found, the box keeps moving from that
    prediction until the feet leave the picture.
    """
    n = len(per_frame)
    ids = [r["id"] for r in roster]
    team_of = {r["id"]: r["team"] for r in roster}
    fwd = [dict() for _ in range(n)]
    fwd[0] = {r["id"]: dict(r) for r in roster}
    last = {r["id"]: r["world"] for r in roster}
    vel = {r["id"]: (0.0, 0.0) for r in roster}
    box_of = {r["id"]: r["bbox"] for r in roster}
    alive = {tid: True for tid in ids}
    off = {tid: 0 for tid in ids}
    cap = cv2.VideoCapture(CLIP)
    cap.read()
    for t in range(1, n):
        ok, frame = cap.read()
        if not ok:
            break
        preds = []
        for tid in ids:
            if not alive[tid]:
                continue
            px, py = last[tid]
            vx, vy = vel[tid]
            preds.append((tid, (px + vx, py + vy), team_of[tid], 5.0))
        got = _assign_gated(preds, per_frame[t], homographies[t])
        for tid in ids:
            if not alive[tid]:
                continue
            if tid in got:
                det = got[tid]
                nx, ny = det["world"]
                vel[tid] = (nx - last[tid][0], ny - last[tid][1])
                last[tid] = (nx, ny)
                box_of[tid] = det["bbox"]
                off[tid] = 0
                fwd[t][tid] = det
            else:
                px, py = last[tid]
                vx, vy = vel[tid]
                pred = (px + vx, py + vy)
                foot = _feet_px(pred, homographies[t])
                snapped = _snap_body(frame, foot, team_of[tid]) if foot else None
                if snapped is not None:
                    bx, by, bw, bh, fx, fy = snapped
                    world = cv2.perspectiveTransform(
                        np.array([[[fx, fy]]], np.float32),
                        homographies[t].astype(np.float32))[0, 0]
                    nx, ny = float(world[0]), float(world[1])
                    vel[tid] = (nx - px, ny - py)
                    last[tid] = (nx, ny)
                    box_of[tid] = (bx, by, bw, bh)
                    off[tid] = 0
                    fwd[t][tid] = dict(
                        team=team_of[tid], world=(nx, ny),
                        bbox=(bx, by, bw, bh), frac=0.2, width_yd=1.5)
                    continue
                if _on_screen(pred, homographies[t], shape):
                    spot = pred
                    vel[tid] = (vx * 0.92, vy * 0.92)
                    off[tid] = 0
                elif _on_screen((px, py), homographies[t], shape):
                    # the step would leave the picture; stay on the edge
                    spot = (px, py)
                    vel[tid] = (0.0, 0.0)
                    off[tid] = 0
                else:
                    off[tid] += 1
                    if off[tid] > 8:
                        alive[tid] = False
                    continue
                box = _shift_box(box_of[tid], (px, py), spot,
                                 homographies[t])
                fwd[t][tid] = dict(
                    team=team_of[tid], world=spot, bbox=box,
                    coasted=True, frac=0.0, width_yd=1.0)
                last[tid] = spot
                box_of[tid] = box
    cap.release()
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
            if (fut is not None and fdet is not None
                    and _dist(fdet["world"], fut["world"]) <= 4.5):
                chosen = fdet
            elif bdet is not None and fdet is not None and fut is not None:
                if (_dist(fdet["world"], fut["world"])
                        <= _dist(bdet["world"], fut["world"])):
                    chosen = fdet
                else:
                    chosen = bdet
            elif bdet is not None:
                chosen = bdet
            elif fdet is not None and fut is None:
                chosen = fdet
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

    crop_w = MAP_CROP_X1
    fmap = field_map[:, :crop_w].copy()
    for tr in tracks:
        c = TEAM_BGR.get(tr["team"], (0, 255, 255))
        pts = list(tr["trail"])
        for i in range(1, len(pts)):
            a = to_map(pts[i - 1])
            b = to_map(pts[i])
            fade = 0.3 + 0.7 * i / len(pts)
            cv2.line(fmap, a, b, tuple(int(ch * fade) for ch in c), 3)
        cv2.circle(fmap, to_map(tr["pos"]), 14, c, -1)
        cv2.circle(fmap, to_map(tr["pos"]), 14, (255, 255, 255), 2)
    # (mx, my) -> (my, mx): north (small x) at the top, far sideline at left
    fmap = cv2.rotate(fmap, cv2.ROTATE_90_COUNTERCLOCKWISE)
    fmap = cv2.flip(fmap, 0)
    scale = left.shape[0] / fmap.shape[0]
    fmap = cv2.resize(fmap, (int(fmap.shape[1] * scale), left.shape[0]))
    for tr in tracks:
        mx, my = to_map(tr["pos"])
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
    ok, prev = cap.read()
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
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        updated = False
        if len(screen) >= 4:
            cur, valid = track_screen_points(prev_gray, gray, screen)
            if cur is not None and int(valid.sum()) >= 4:
                new_A, inliers = _homography_image_to_world(world[valid], cur[valid])
                if new_A is not None and int(inliers.sum()) >= 4:
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
    print(f"frames: {n}")
    cached = load_homographies(n)
    if cached is not None:
        cap.release()
        # Frame 0 stays the anchored fit. Later frames follow field
        # landmarks the way nfl-blitz-opencv tracks them.
        A, status = track_field_homographies(cached[0][0], n)
        n_fresh = sum(1 for s in status if s == "fresh")
        n_chained = sum(1 for s in status if s == "chained")
        n_stale = sum(1 for s in status if s == "stale")
        print(f"status: fresh={n_fresh} chained={n_chained} stale={n_stale}")
        save_homographies(A, status)
        render_pass(n, A, status)
        return

    # ---- pass 1: fresh fits + flow
    fresh, G, infos = {}, {}, {}
    prev_gray, prev_mask = None, None
    shape = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if shape is None:
            shape = frame.shape
        try:
            H, info = fit_frame(frame)
        except Exception as e:
            H, info = None, f"err {e}"
        if H is not None:
            fresh[i] = H
            infos[i] = info
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                (5, 5), 0)
        m = flow_mask4(frame)
        if prev_gray is not None:
            g, ninl = inter_frame_h(prev_gray, gray, prev_mask)
            if g is not None and not g_sane(g, shape[1], shape[0]):
                g = None
            G[i - 1] = g
        prev_gray, prev_mask = gray, m
        if i % 20 == 0:
            print(f"pass1 {i}/{n} fresh={i in fresh}")
        i += 1
    n = i
    print(f"fresh fits: {len(fresh)}/{n}, flow ok: "
          f"{sum(1 for g in G.values() if g is not None)}/{n-1}")
    if REF_FRAME not in fresh:
        print("FATAL: no fresh fit at reference frame")
        return

    # ---- absolute homographies
    A = [None] * n
    status = ["chained"] * n
    T_anchor = np.array([[1, 0, REF_ANCHOR], [0, 1, 0], [0, 0, 1.0]])
    A[REF_FRAME] = T_anchor @ fresh[REF_FRAME]
    status[REF_FRAME] = "fresh"

    for i in range(REF_FRAME + 1, n):
        g = G.get(i - 1)
        if g is not None:
            A[i] = A[i - 1] @ np.linalg.inv(g)
            status[i] = "chained"
        else:
            A[i] = A[i - 1]
            status[i] = "stale"
        if i in fresh:
            adj = reconcile4(A[i], fresh[i], infos[i], shape)
            if adj is not None:
                A[i] = adj
                status[i] = "fresh"
    for i in range(REF_FRAME - 1, -1, -1):
        g = G.get(i)
        if g is not None:
            A[i] = A[i + 1] @ g
            status[i] = "chained"
        else:
            A[i] = A[i + 1]
            status[i] = "stale"
        if i in fresh:
            adj = reconcile4(A[i], fresh[i], infos[i], shape)
            if adj is not None:
                A[i] = adj
                status[i] = "fresh"
    n_fresh = sum(1 for s in status if s == "fresh")
    n_chained = sum(1 for s in status if s == "chained")
    n_stale = sum(1 for s in status if s == "stale")
    print(f"status: fresh={n_fresh} chained={n_chained} stale={n_stale}")
    save_homographies(A, status)
    render_pass(n, A, status)


def render_pass(n, A, status):
    # Detect every frame, then lock 7 red + 7 blue on the opening frame
    # and carry those numbers forward and backward.
    cap = cv2.VideoCapture(CLIP)
    per_frame = []
    shape = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if shape is None:
            shape = frame.shape
        per_frame.append(frame_candidates(frame, A[i]))
        if i % 40 == 0:
            print(f"detect {i}/{n} cands={len(per_frame[-1])}")
        i += 1
    cap.release()
    roster = select_roster(per_frame[0])
    per_frame, removed = drop_stationary(per_frame)
    print(f"removed {removed} stationary decal detections")
    n_red = sum(1 for r in roster if r["team"] == "red")
    n_blue = sum(1 for r in roster if r["team"] == "blue")
    print(f"opening roster: {n_red} red, {n_blue} blue")
    for r in roster:
        print(f"  {r['id']:2} {r['team']:5} ({r['world'][0]:5.1f}, {r['world'][1]:5.1f})")
    tracked = track_bidirectional(per_frame, roster, A, shape)

    os.makedirs(SAMPLE_DIR, exist_ok=True)
    field_map = cv2.imread(FIELD_MAP) if FIELD_MAP else None
    cap = cv2.VideoCapture(CLIP)
    writer, out_path = None, OUT_VIDEO
    rows = []
    i = 0
    while True:
        ok, frame = cap.read()
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
    parser.add_argument("--field-map", default="",
                        help="Optional top-down field image for the map panel.")
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
