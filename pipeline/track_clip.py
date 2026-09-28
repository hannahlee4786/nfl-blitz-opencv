#!/usr/bin/env python3
"""Full-clip player tracking for NFL Blitz 2000 footage.

Pass 1: per-frame fresh homography fits + optical-flow chaining.
Pass 2: player detection, temporal association, rendering, CSV.
"""
import csv
import os
import sys
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import field_homography as fh
from project_players import detect_players

CLIP = "Clip_3 Easiest QB Test.mp4"
REF_FRAME = 46
REF_ANCHOR = 10.0          # absolute yards offset established for frame 46
PXY = fh.PXY
OUT_VIDEO = "extracted/Clip_3_tracked.mp4"
OUT_CSV = "extracted/positions.csv"


# ---------------------------------------------------------------- fresh fit
def fit_frame(frame):
    """Fresh homography fit. Returns (H, info) or (None, reason)."""
    h, w = frame.shape[:2]
    mask = fh.ridge_mask(frame)
    segs = fh.get_segments(mask, w)
    if len(segs) < 8:
        return None, "few segments"
    cands = fh.merge_candidates(segs)
    _, vp1, m1 = fh.ransac_pencil(cands)
    if vp1 is None:
        return None, "no yard pencil"
    yard_all = [cands[k] for k in m1]
    yard_ang = np.median([c["angle"] for c in yard_all])
    rest = []
    for k, c in enumerate(cands):
        if k in m1:
            continue
        d = abs(c["angle"] - yard_ang) % 180
        if min(d, 180 - d) >= 20:
            rest.append(c)

    def vp_ok(vp):
        return vp[1] < 0.20 * h

    _, vp2, m2 = fh.ransac_pencil(rest, tol_deg=2.0, min_lines=2,
                                  vp_filter=vp_ok)
    down = [rest[k] for k in m2] if vp2 is not None else []
    yard = [c for c in yard_all
            if c["extent"] > w * 0.30 and c["support"] > w * 0.30]
    indexed = fh.index_yard_lines(yard, w, h)
    if len(indexed) < 4 or len(down) < 2:
        return None, f"structure ({len(indexed)} yd, {len(down)} down)"

    down.sort(key=lambda c: -c["support"])
    ref_line = indexed[len(indexed) // 2][0]["abc"]
    distinct = []
    for c in down:
        p = fh.line_cross(c["abc"], ref_line)
        if p is None:
            continue
        if all(np.linalg.norm(p - d[1]) > 40 for d in distinct):
            distinct.append((c, p))
    if len(distinct) < 2:
        return None, "one downfield line"

    (h1, _), (h2, _) = distinct[0], distinct[1]
    src, dst = [], []
    for c, xw in indexed:
        for dl, yy in ((h1, 0.0), (h2, fh.HASH_GAP)):
            p = fh.line_cross(c["abc"], dl["abc"])
            if p is not None:
                src.append(p)
                dst.append([xw, yy])
    if len(src) < 8:
        return None, "few points"
    H, inl = cv2.findHomography(np.array(src, np.float32),
                                np.array(dst, np.float32), cv2.RANSAC, 1.0)
    if H is None or inl.sum() < 8:
        return None, "bad fit"
    H = H / H[2, 2]

    def world_y(Hm, u, v):
        p = Hm @ np.array([u, v, 1.0])
        return p[1] / p[2]

    sideline = len(distinct) >= 3
    if sideline:
        pts = np.array(distinct[2][0]["pts"])
        c0 = float(np.median([world_y(H, px, py) for px, py in pts[:20]]))
        interior = world_y(H, w / 2, h * 0.6)
        if interior < c0:
            H = np.diag([1.0, -1.0, 1.0]) @ H
            c0 = -c0
        H = np.array([[1, 0, 0], [0, 1, -c0], [0, 0, 1.0]]) @ H
    else:
        if world_y(H, w / 2, h * 0.85) < world_y(H, w / 2, h * 0.3):
            H = np.diag([1.0, -1.0, 1.0]) @ H
        H = np.array([[1, 0, 0], [0, 1, fh.HASH_GAP / 2 + 13.0],
                      [0, 0, 1.0]]) @ H
    return H, dict(n_lines=len(indexed), n_inl=int(inl.sum()),
                   sideline=sideline)


# ---------------------------------------------------------------- optical flow
def flow_mask(frame):
    """Where to track field texture: field prior minus players/HUD."""
    h, w = frame.shape[:2]
    keep = fh.hud_mask(h, w, frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hch, sch, vch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    players = (((hch < 12) | (hch > 168)) & (sch > 90)) | \
              ((hch > 100) & (hch < 135) & (sch > 90)) | \
              ((sch < 55) & (vch > 150))
    players = cv2.dilate(players.astype(np.uint8) * 255,
                         np.ones((15, 15), np.uint8))
    return cv2.bitwise_and(keep, cv2.bitwise_not(players))


def inter_frame_h(gray_prev, gray_cur, mask_prev):
    pts = cv2.goodFeaturesToTrack(gray_prev, maxCorners=500,
                                  qualityLevel=0.01, minDistance=12,
                                  mask=mask_prev)
    if pts is None or len(pts) < 40:
        return None, 0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
        gray_prev, gray_cur, pts, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    st = st.reshape(-1).astype(bool)
    p0, p1 = pts.reshape(-1, 2)[st], nxt.reshape(-1, 2)[st]
    if len(p0) < 40:
        return None, 0
    G, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
    if G is None or inl.sum() < 30:
        return None, 0
    return G / G[2, 2], int(inl.sum())


# ---------------------------------------------------------------- reconcile
def reconcile(A_chain, H_fresh, info, shape):
    """Adjust fresh fit's translation to agree with chained prediction.
    Returns adjusted absolute H or None if inconsistent."""
    h, w = shape[:2]
    us = np.linspace(0.2 * w, 0.8 * w, 4)
    vs = np.linspace(0.35 * h, 0.75 * h, 3)
    P = np.array([[[u, v]] for u in us for v in vs], np.float32)
    Xc = cv2.perspectiveTransform(P, A_chain.astype(np.float32))[:, 0, :]
    Xf = cv2.perspectiveTransform(P, H_fresh.astype(np.float32))[:, 0, :]
    dx = float(np.median(Xc[:, 0] - Xf[:, 0]))
    dy = float(np.median(Xc[:, 1] - Xf[:, 1]))
    dx_snap = round(dx / 5.0) * 5.0
    if abs(dx - dx_snap) > 1.5:
        return None
    dy_use = dy if not info["sideline"] else \
        (REF_ANCHOR * 0 if abs(dy) < 5.0 else None)
    if dy_use is None:
        return None
    T = np.array([[1, 0, dx_snap], [0, 1, dy_use], [0, 0, 1.0]])
    A_new = T @ H_fresh
    Xn = cv2.perspectiveTransform(P, A_new.astype(np.float32))[:, 0, :]
    rms = float(np.sqrt(((Xn - Xc) ** 2).sum(axis=1).mean()))
    if rms > 3.0:
        return None
    return A_new


# ---------------------------------------------------------------- tracking
class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 0

    def update(self, frame_idx, dets):
        """dets: list of dict(team, world=(x,y)). Returns active tracks."""
        used = set()
        for tr in self.tracks:
            best, bd = None, 3.5
            for j, d in enumerate(dets):
                if j in used or d["team"] != tr["team"]:
                    continue
                dist = np.hypot(d["world"][0] - tr["pred"][0],
                                d["world"][1] - tr["pred"][1])
                if dist < bd:
                    best, bd = j, dist
            if best is not None:
                used.add(best)
                d = dets[best]
                tr["obs"].append(d["world"])
                sm = np.mean(tr["obs"], axis=0)
                tr["pos"] = tuple(sm)
                tr["trail"].append(tuple(sm))
                tr["last"] = frame_idx
                tr["pred"] = d["world"]
        for j, d in enumerate(dets):
            if j in used:
                continue
            self.tracks.append(dict(
                id=self.next_id, team=d["team"], pos=d["world"],
                pred=d["world"], obs=deque([d["world"]], maxlen=3),
                trail=deque([d["world"]], maxlen=15), last=frame_idx))
            self.next_id += 1
        self.tracks = [t for t in self.tracks if frame_idx - t["last"] <= 8]
        return [t for t in self.tracks if t["last"] == frame_idx]


def project_dets(frame, A, raw):
    """Filter raw detections and project to absolute world coords."""
    h, w = frame.shape[:2]
    out = []
    Af = A.astype(np.float32)
    for d in raw:
        x0, y0, bw, bh = d["bbox"]
        if y0 + bh > h * 0.855:  # feet cut off by frame/HUD edge
            continue
        feet = np.array([[[x0 + bw / 2.0, y0 + float(bh)]]], np.float32)
        xw, yw = cv2.perspectiveTransform(feet, Af)[0, 0]
        corners = np.array([[[x0, y0 + bh]], [[x0 + bw, y0 + bh]]],
                           np.float32)
        wc = cv2.perspectiveTransform(corners, Af)[:, 0, :]
        width_yd = float(np.linalg.norm(wc[1] - wc[0]))
        if not (0.3 <= width_yd <= 6.0):
            continue
        if not (-3 <= yw <= 45) or not (-15 <= xw <= 130):
            continue
        if yw < 0.8:   # sideline pylons / boundary decals
            continue
        out.append(dict(team=d["team"], world=(float(xw), float(yw)),
                        bbox=d["bbox"]))
    return out


# ---------------------------------------------------------------- rendering
TEAM_BGR = dict(red=(0, 0, 255), blue=(255, 100, 0),
                white=(255, 255, 255))
STATUS_BGR = dict(fresh=(80, 220, 80), chained=(60, 200, 255),
                  stale=(60, 60, 255))
MAP_CROP_X1 = 1792   # px: end zone + 60 yd
LEFT_SCALE = 0.5


def render(frame, tracks, dets, status, field_map):
    h, w = frame.shape[:2]
    left = frame.copy()
    for d in dets:
        x0, y0, bw, bh = d["bbox"]
        c = TEAM_BGR.get(d["team"], (0, 255, 255))
        cv2.rectangle(left, (x0, y0), (x0 + bw, y0 + bh), c, 2)
        cv2.circle(left, (int(x0 + bw / 2), int(y0 + bh)), 5, c, -1)
    cv2.putText(left, status.upper(), (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                STATUS_BGR.get(status, (255, 255, 255)), 3)
    left = cv2.resize(left, (int(w * LEFT_SCALE), int(h * LEFT_SCALE)))

    fmap = field_map[:, :MAP_CROP_X1].copy()
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
    scale = left.shape[0] / fmap.shape[0]
    fmap = cv2.resize(fmap, (int(fmap.shape[1] * scale), left.shape[0]))
    combo = np.hstack([left, fmap])
    return combo


def to_map(world):
    x, y = world
    return (int((10 + x) * PXY), int((y + 2.5) * PXY))


# ---------------------------------------------------------------- main
def main():
    cap = cv2.VideoCapture(CLIP)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"frames: {n}")

    # ---- pass 1: fresh fits + flow
    fresh, G, infos = {}, {}, {}
    prev_gray, prev_mask = None, None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        try:
            H, info = fit_frame(frame)
        except Exception as e:
            H, info = None, f"err {e}"
        if H is not None:
            fresh[i] = H
            infos[i] = info
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                (5, 5), 0)
        m = flow_mask(frame)
        if prev_gray is not None:
            g, ninl = inter_frame_h(prev_gray, gray, prev_mask)
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
    shape = (1016, 1356)

    n_fresh, n_chained, n_stale = 1, 0, 0
    for i in range(REF_FRAME + 1, n):
        g = G.get(i - 1)
        if g is not None:
            A[i] = A[i - 1] @ np.linalg.inv(g)
            status[i] = "chained"
        else:
            A[i] = A[i - 1]
            status[i] = "stale"
        if i in fresh:
            adj = reconcile(A[i], fresh[i], infos[i], shape)
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
            adj = reconcile(A[i], fresh[i], infos[i], shape)
            if adj is not None:
                A[i] = adj
                status[i] = "fresh"
    n_fresh = sum(1 for s in status if s == "fresh")
    n_chained = sum(1 for s in status if s == "chained")
    n_stale = sum(1 for s in status if s == "stale")
    print(f"status: fresh={n_fresh} chained={n_chained} stale={n_stale}")

    # ---- pass 2: detect, track, render
    field_map = cv2.imread("extracted/field_map.png")
    cap = cv2.VideoCapture(CLIP)
    tracker = Tracker()
    writer = None
    rows = []
    sample_at = {0, 30, 46, 93, 140, 185}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw = detect_players(frame)
        dets = project_dets(frame, A[i], raw)
        active = tracker.update(i, dets)
        for tr in active:
            rows.append([i, tr["id"], tr["team"],
                         round(tr["pos"][0], 2), round(tr["pos"][1], 2)])
        combo = render(frame, active, dets, status[i], field_map)
        if writer is None:
            writer = make_writer(combo.shape)
        writer.write(combo)
        if i in sample_at:
            cv2.imwrite(f"extracted/frames/composite_{i:04d}.png", combo)
        if i % 20 == 0:
            print(f"pass2 {i}/{n} dets={len(dets)} tracks={len(active)}")
        i += 1
    writer.release()

    with open(OUT_CSV, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame", "id", "team", "x_yd", "y_yd"])
        wcsv.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")

    # verify playback
    chk = cv2.VideoCapture(OUT_VIDEO)
    nchk = int(chk.get(cv2.CAP_PROP_FRAME_COUNT))
    ok1, _ = chk.read()
    chk.set(cv2.CAP_PROP_POS_FRAMES, nchk - 1)
    ok2, _ = chk.read()
    print(f"playback check: frames={nchk} first={ok1} last={ok2}")


def make_writer(shape):
    h, w = shape[:2]
    for fourcc, path in (("mp4v", OUT_VIDEO), ("avc1", OUT_VIDEO),
                         ("MJPG", OUT_VIDEO.replace(".mp4", ".avi"))):
        wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc),
                             30.0, (w, h))
        if wr.isOpened():
            print(f"video writer: {fourcc} -> {path}")
            return wr
    raise RuntimeError("no video writer available")


if __name__ == "__main__":
    main()
