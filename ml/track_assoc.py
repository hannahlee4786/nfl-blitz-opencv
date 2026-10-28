#!/usr/bin/env python3
"""Approach A: detection tracking with camera motion and a fixed roster.

Pipeline, all offline and zero-shot:

1. Load cached every-frame boxes (see ml/detect_all.py).
2. Drop HUD boxes, '1' player-indicator boxes, and partial duplicate boxes.
   Plaque boxes on wide shots stay in until the roster is chosen, then a
   chain that is mostly plaque is dropped and others lose those boxes.
   A player track that coasted onto a sideline pylon loses that run of
   small, warm, camera-fixed boxes after all repairs.
3. Associate forward and backward (ByteTrack-style high/low passes) in the
   camera-compensated frame. A box whose jersey is clearly the other team
   costs more to join a track.
4. Stitch tracklets of the same team when motion, appearance, the backward
   pass, or the quarterback marker agree. At most the long tracks are kept;
   weak duplicate tracks are removed.
5. Undo identity swaps where two same-team paths cross and swapping the
   suffix is smoother.
6. Interpolate short gaps in camera-compensated coordinates.
7. Number frame-0 red boxes left to right as 1..R, then blue left to right.

Clip 64 must be run with the same flags used for Clip 4.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.eval_tracking import hungarian  # noqa: E402


# Chosen on Clip 4. Do not retune per clip.
DEFAULTS = {
    "low": 0.07,
    "high": 0.18,
    "min_hits": 3,
    "coast": 14,
    "stitch_gap": 22,
    "interp_gap": 10,
    "contain": 0.70,
    "hud_max": 0.50,
    "min_h": 20.0,
    "min_w": 10.0,
    "max_aspect": 5.2,
    "min_aspect": 0.45,
    # A detector occasionally wraps a whole tackle pile in one person box.
    # Keep the established tracks moving through that short occlusion instead
    # of letting the group box replace one of them.
    "merge_area": 2.00,
    "merge_cover": 0.80,
    "merge_scale": 1.60,
    # A severe identity handoff can leave an established path following the
    # other team for dozens of frames.  Require both a long opposing-color run
    # and a poor re-ID match before splitting it; ordinary tackle/camera color
    # flicker is much shorter and remains untouched.
    "drift_trigger": 24,
    "drift_split": 4,
    "drift_similarity": 0.18,
    # Tracks jump onto an opponent in contact and keep their own team. When
    # the detection's jersey is clearly the other color, add this much on
    # top of the usual 0.25 team mismatch cost.
    "team_penalty": 0.5,
    "team_margin": 0.6,
    # A box shorter than this share of the track's recent height is a
    # partial body, often legs and the carried football, so its torso band
    # is not the jersey. Such a box gets no extra team penalty.
    "team_partial": 0.9,
    # limit_roster ranks chains over the whole clip. A chain past the top 7
    # is still kept when it has this many detections, one at least
    # roster_rescue_conf, and its team has fewer than 7 kept chains on every
    # frame it publishes. 0 turns the rescue off.
    "roster_rescue_len": 12,
    "roster_rescue_conf": 0.30,
    # After association, a chain at least this share '1' plaque boxes
    # (ICON_AFTER) is dropped; other chains only lose their plaque boxes.
    "icon_chain_share": 0.5,
    # A player track can coast onto a sideline pylon. A run of at least
    # cone_run_frames small, warm, camera-fixed boxes inside a player-sized
    # track is removed. 0 turns this off. Size and fixed-point limits match
    # the whole-track cone filter in pipeline/track_clip4.py.
    "cone_run_frames": 12,
    "cone_warm": 0.30,
}


class Det:
    __slots__ = ("uid", "frame", "x", "y", "w", "h", "conf", "team", "hist", "src", "emb")

    def __init__(self, uid, frame, x, y, w, h, conf, team, hist, src=None, emb=None):
        self.uid = uid
        self.frame = int(frame)
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        self.conf = float(conf)
        self.team = team
        self.hist = hist
        self.src = src
        self.emb = emb

    @property
    def cx(self) -> float:
        return self.x + 0.5 * self.w

    @property
    def cy(self) -> float:
        return self.y + 0.5 * self.h

    def clone_at(self, frame: int) -> "Det":
        copied = Det(
            self.uid, frame, self.x, self.y, self.w, self.h, self.conf, self.team, self.hist, src=self, emb=self.emb
        )
        return copied


class Track:
    def __init__(self, det: Det):
        self.obs: list[Det] = [det]
        self.vx = 0.0
        self.vy = 0.0
        self.misses = 0
        self.red = 1 if det.team == "red" else 0
        self.blue = 1 if det.team == "blue" else 0

    def team(self) -> str:
        return "red" if self.red >= self.blue else "blue"

    def last(self) -> Det:
        return self.obs[-1]

    def hist(self) -> np.ndarray:
        chunk = self.obs[-8:]
        return np.mean([det.hist for det in chunk], axis=0)

    def emb(self) -> np.ndarray | None:
        chunk = [det.emb for det in self.obs[-8:] if det.emb is not None]
        if not chunk:
            return None
        mean = np.mean(chunk, axis=0)
        norm = float(np.linalg.norm(mean))
        if norm < 1e-8:
            return None
        return mean / norm


def box_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def containment(a: Det, b: Det) -> float:
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.x + a.w, b.x + b.w)
    bottom = min(a.y + a.h, b.y + b.h)
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    smaller = min(a.w * a.h, b.w * b.h)
    if smaller <= 0:
        return 0.0
    return inter / smaller


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def invert_affine(matrix: np.ndarray) -> np.ndarray:
    linear = matrix[:, :2]
    shift = matrix[:, 2]
    try:
        inverse = np.linalg.inv(linear)
    except np.linalg.LinAlgError:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    out = np.zeros((2, 3), dtype=np.float64)
    out[:, :2] = inverse
    out[:, 2] = -inverse @ shift
    return out


def compose(motion: np.ndarray, frame0: int, frame1: int) -> np.ndarray:
    """Affine mapping coordinates on frame0 onto frame1."""
    acc = np.eye(3, dtype=np.float64)
    if frame1 >= frame0:
        for frame in range(frame0 + 1, frame1 + 1):
            step = np.eye(3, dtype=np.float64)
            step[:2] = motion[frame]
            acc = step @ acc
    else:
        for frame in range(frame0, frame1, -1):
            step = np.eye(3, dtype=np.float64)
            step[:2] = invert_affine(motion[frame])
            acc = step @ acc
    return acc[:2]


def apply_affine(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    return (
        float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]),
        float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]),
    )


def transform_box(matrix: np.ndarray, x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    corners = ((x, y), (x + w, y), (x, y + h), (x + w, y + h))
    points = [apply_affine(matrix, px, py) for px, py in corners]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    nx, ny = min(xs), min(ys)
    return nx, ny, max(xs) - nx, max(ys) - ny


def shift_velocity(matrix: np.ndarray, vx: float, vy: float) -> tuple[float, float]:
    return (
        float(matrix[0, 0] * vx + matrix[0, 1] * vy),
        float(matrix[1, 0] * vx + matrix[1, 1] * vy),
    )


def predict_box(last: Det, vx: float, vy: float, frame: int, motion: np.ndarray):
    matrix = compose(motion, last.frame, frame)
    x, y, w, h = transform_box(matrix, last.x, last.y, last.w, last.h)
    dt = frame - last.frame
    svx, svy = shift_velocity(matrix, vx, vy)
    return x + svx * dt, y + svy * dt, w, h, svx, svy


def hud_fraction(hud: np.ndarray, det: Det, width: int, height: int) -> float:
    mask_h, mask_w = hud.shape[:2]
    x1 = int(np.clip(det.x / width * mask_w, 0, mask_w - 1))
    y1 = int(np.clip(det.y / height * mask_h, 0, mask_h - 1))
    x2 = int(np.clip((det.x + det.w) / width * mask_w, 0, mask_w))
    y2 = int(np.clip((det.y + det.h) / height * mask_h, 0, mask_h))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = hud[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float((roi > 0).mean())


def suppress_duplicates(dets: list[Det], contain_thr: float) -> list[Det]:
    ordered = sorted(dets, key=lambda det: det.conf, reverse=True)
    kept: list[Det] = []
    for det in ordered:
        drop = False
        for index, other in enumerate(kept):
            # Different teams in a pile both stay. A near-total overlap is still
            # one body with a flickering team label.
            if det.team != other.team and containment(det, other) < 0.85:
                continue
            if containment(det, other) < contain_thr:
                continue
            det_area = det.w * det.h
            other_area = other.w * other.h
            if det_area > other_area * 1.15 and det.conf >= other.conf * 0.75:
                kept[index] = det
            drop = True
            break
        if not drop:
            kept.append(det)
    return kept


def load_cache(stem: Path):
    meta = json.loads(stem.with_suffix(".meta.json").read_text())
    rows = list(csv.DictReader(stem.with_suffix(".csv").open()))
    hist = np.load(stem.with_suffix(".hist.npy"))
    motion = np.load(stem.with_suffix(".motion.npy"))
    hud = np.load(stem.with_suffix(".hud.npy"))
    markers: dict[int, list[tuple[float, float]]] = {}
    with stem.with_suffix(".markers.csv").open() as handle:
        for row in csv.DictReader(handle):
            markers.setdefault(int(row["frame"]), []).append((float(row["x"]), float(row["y"])))
    if len(hist) != len(rows):
        raise RuntimeError(f"hist rows {len(hist)} != csv rows {len(rows)}")
    return meta, rows, hist, motion, hud, markers


# Icon levels written by ml/detect_all.py: ICON_BEFORE boxes are dropped
# before association, ICON_AFTER boxes are removed from chains afterwards.
ICON_AFTER = 1
ICON_BEFORE = 2


def load_icons(stem: Path, nrows: int) -> np.ndarray | None:
    """Per-row '1' plaque levels from ml/detect_all.py, or None for an older cache.

    A bool file from before the two levels existed marks ICON_BEFORE boxes.
    """
    path = stem.with_suffix(".icons.npy")
    if not path.is_file():
        return None
    icons = np.load(path)
    if len(icons) != nrows:
        raise RuntimeError(f"icon rows {len(icons)} != csv rows {nrows}")
    if icons.dtype == bool:
        return icons.astype(np.uint8) * ICON_BEFORE
    return icons


def load_warm(stem: Path, nrows: int) -> np.ndarray | None:
    """Per-row cone warm share from ml/detect_all.py, or None for an older cache."""
    path = stem.with_suffix(".warm.npy")
    if not path.is_file():
        return None
    warm = np.load(path)
    if len(warm) != nrows:
        raise RuntimeError(f"warm rows {len(warm)} != csv rows {nrows}")
    return warm


def reference_transforms(motion: np.ndarray) -> list[np.ndarray]:
    """3x3 transforms mapping each frame back onto frame zero."""
    transforms = [np.eye(3, dtype=np.float64)]
    for frame in range(1, len(motion)):
        step = np.eye(3, dtype=np.float64)
        step[:2] = motion[frame]
        try:
            inverse = np.linalg.inv(step)
        except np.linalg.LinAlgError:
            inverse = np.eye(3, dtype=np.float64)
        transforms.append(transforms[-1] @ inverse)
    return transforms


def remove_cone_runs(
    paths: list[dict[int, Det]],
    warm: np.ndarray | None,
    to_ref: list[np.ndarray],
    width: int,
    height: int,
    min_frames: int,
    min_warm: float,
) -> int:
    """Remove sideline-pylon runs from player-sized tracks; returns the box count.

    pipeline/track_clip4.py removes whole tracks that are a pylon. A player
    track that coasts onto a pylon keeps a player-sized median box and
    passes that test. Here the same tests run on each run of small boxes
    (at most 6% of the frame tall and 4% wide, gaps of at most 3 frames)
    inside a track whose median box is not small: at least min_frames
    boxes, median warm share of its detected boxes at least min_warm, and
    feet fixed after camera motion (60% within 30 px of the median, median
    distance at most 18 px, spread at most 24 px across and 14 px down).
    Pylon-sized tracks are left to the whole-track filter.
    """
    if warm is None or min_frames <= 0:
        return 0

    def small(det: Det) -> bool:
        return det.h <= 0.06 * height and det.w <= 0.04 * width

    removed = 0
    for path in paths:
        if not path:
            continue
        frames = sorted(path)
        median_h = float(np.median([path[f].h for f in frames]))
        median_w = float(np.median([path[f].w for f in frames]))
        if median_h <= 0.06 * height and median_w <= 0.04 * width:
            continue
        runs: list[list[int]] = []
        run: list[int] = []
        for frame in frames:
            if small(path[frame]) and (not run or frame - run[-1] <= 3):
                run.append(frame)
            else:
                if run:
                    runs.append(run)
                run = [frame] if small(path[frame]) else []
        if run:
            runs.append(run)
        for run in runs:
            if len(run) < min_frames:
                continue
            detected = [warm[path[f].uid] for f in run if path[f].uid >= 0]
            if not detected or float(np.median(detected)) < min_warm:
                continue
            feet = []
            for frame in run:
                det = path[frame]
                point = to_ref[frame] @ np.array([det.cx, det.y + det.h, 1.0])
                feet.append(point[:2] / point[2])
            feet = np.asarray(feet)
            center = np.median(feet, axis=0)
            distance = np.linalg.norm(feet - center, axis=1)
            inlier = distance <= 30.0
            if float(inlier.mean()) < 0.60 or float(np.median(distance)) > 18.0:
                continue
            spread = feet[inlier].std(axis=0)
            if float(spread[0]) > 24.0 or float(spread[1]) > 14.0:
                continue
            for frame in run:
                del path[frame]
            removed += len(run)
    return removed


def strip_icon_chains(
    chains: list[list[Tlet]], icons: np.ndarray | None, max_share: float
) -> tuple[list[tuple[list[Tlet], list[Det]]], int, int]:
    """Remove ICON_AFTER detections from chains after association.

    A chain whose detections are at least max_share icon is the icon itself;
    its few real boxes are the icon track drifting onto a player who already
    has his own track, so the whole chain goes. Other chains keep every real
    detection and lose only their icon ones. Returns (chain, observations)
    pairs, the dropped chain count, and the stripped detection count.
    """
    if icons is None:
        return [(chain, chain_obs(chain)) for chain in chains], 0, 0
    kept = []
    dropped = 0
    stripped = 0
    for chain in chains:
        obs = chain_obs(chain)
        real = [det for det in obs if icons[det.uid] != ICON_AFTER]
        if not real or len(obs) - len(real) >= max_share * len(obs):
            dropped += 1
            continue
        stripped += len(obs) - len(real)
        kept.append((chain, real))
    return kept, dropped, stripped


def build_frames(meta, rows, hist, hud, params, emb=None, icons=None) -> list[list[Det]]:
    width, height = int(meta["width"]), int(meta["height"])
    nframes = int(meta["nframes"])
    per_frame: list[list[Det]] = [[] for _ in range(nframes)]
    for uid, (row, feature) in enumerate(zip(rows, hist)):
        det = Det(
            uid,
            int(row["frame"]),
            float(row["x"]),
            float(row["y"]),
            float(row["w"]),
            float(row["h"]),
            float(row["conf"]),
            row["team"],
            feature,
            emb=None if emb is None else emb[uid],
        )
        if det.conf < params["low"]:
            continue
        if det.h < params["min_h"] or det.w < params["min_w"]:
            continue
        aspect = det.h / max(det.w, 1e-6)
        if aspect < params["min_aspect"] or aspect > params["max_aspect"]:
            continue
        if det.h > 0.9 * height or det.w > 0.7 * width:
            continue
        if hud_fraction(hud, det, width, height) > params["hud_max"]:
            continue
        if icons is not None and icons[uid] == ICON_BEFORE:
            continue
        if det.frame < 0 or det.frame >= nframes:
            continue
        per_frame[det.frame].append(det)
    for frame in range(nframes):
        per_frame[frame] = suppress_duplicates(per_frame[frame], params["contain"])
    return per_frame


def jersey_margin(det: Det) -> float:
    """How clearly the torso is one team color, 0 (mixed or none) to 1.

    Uses the cached 12x4 hue-saturation histogram from ml/detect_all.py:
    hue bins 0 and 11 are red, 7 and 8 are blue, and the lowest saturation
    bin is left out.
    """
    if det.hist is None:
        return 0.0
    counts = np.asarray(det.hist, dtype=np.float64).reshape(12, 4)
    red = float(counts[[0, 11], 1:].sum())
    blue = float(counts[[7, 8], 1:].sum())
    if red + blue < 0.08:
        return 0.0
    return abs(red - blue) / (red + blue)


def pair_cost(
    pred,
    det: Det,
    track: Track,
    min_iou: float,
    team_penalty: float = 0.0,
    team_margin: float = 1.0,
    team_partial: float = 0.0,
) -> float:
    px, py, pw, ph = pred
    iou = box_iou((px, py, pw, ph), (det.x, det.y, det.w, det.h))
    if min_iou > 0 and iou < min_iou:
        return 1e5
    dist = float(np.hypot(px + 0.5 * pw - det.cx, py + ph - (det.y + det.h)))
    dt = max(1, det.frame - track.last().frame)
    speed = float(np.hypot(track.vx, track.vy))
    height = max(28.0, 0.5 * (ph + det.h))
    # Standing players stay in a tight gate so a neighbor in the pile cannot
    # steal the id. A sprint widens the gate with the track's own speed.
    allow = 0.32 * height + 0.85 * speed * dt + 8.0
    if dist > allow and iou < 0.28:
        return 1e5
    app = cosine(track.hist(), det.hist)
    cost = dist / max(allow, 1.0) + 0.35 * (1.0 - iou) + 0.08 * (1.0 - app)
    if track.team() != det.team and (track.red + track.blue) >= 4:
        cost += 0.25
        if (
            team_penalty > 0
            and jersey_margin(det) >= team_margin
            and not partial_body(det, track, team_partial)
        ):
            cost += team_penalty
    return cost


def partial_body(det: Det, track: Track, ratio: float) -> bool:
    """True when the box is much shorter than the track's last five boxes."""
    if ratio <= 0:
        return False
    recent = float(np.median([obs.h for obs in track.obs[-5:]]))
    return det.h < ratio * recent


def match_tracks(
    tracks,
    preds,
    dets,
    min_iou: float,
    gate: float,
    reid_w: float = 0.0,
    team_penalty: float = 0.0,
    team_margin: float = 1.0,
    team_partial: float = 0.0,
):
    if not tracks or not dets:
        return []
    cost = np.full((len(tracks), len(dets)), 1e5, dtype=np.float64)
    for i, track in enumerate(tracks):
        if preds[i] is None:
            continue
        for j, det in enumerate(dets):
            cost[i, j] = pair_cost(preds[i], det, track, min_iou, team_penalty, team_margin, team_partial)
    if reid_w > 0:
        # A clear motion winner stays put: appearance gaps are ~0.2 cosine,
        # so this term only flips a match when two same-team boxes are close.
        for i, track in enumerate(tracks):
            track_emb = track.emb()
            if track_emb is None:
                continue
            for j, det in enumerate(dets):
                if cost[i, j] >= 1e4 or det.emb is None or det.team != track.team():
                    continue
                cost[i, j] += reid_w * (1.0 - cosine(track_emb, det.emb))
    row_count, col_count = cost.shape
    padded = np.full((row_count, col_count + row_count), gate, dtype=np.float64)
    padded[:, :col_count] = cost
    pairs = hungarian(padded)
    matched = []
    for row, col in pairs:
        if col < col_count and cost[row, col] < gate:
            matched.append((row, col))
    return matched


def update_track(track: Track, det: Det, motion: np.ndarray) -> None:
    last = track.last()
    dt = max(1, det.frame - last.frame)
    matrix = compose(motion, last.frame, det.frame)
    base_x, base_y = apply_affine(matrix, last.cx, last.cy)
    inst_vx = (det.cx - base_x) / dt
    inst_vy = (det.cy - base_y) / dt
    old_vx, old_vy = shift_velocity(matrix, track.vx, track.vy)
    if len(track.obs) == 1:
        track.vx, track.vy = inst_vx, inst_vy
    else:
        track.vx = 0.55 * old_vx + 0.45 * inst_vx
        track.vy = 0.55 * old_vy + 0.45 * inst_vy
    track.obs.append(det)
    track.misses = 0
    if det.team == "red":
        track.red += 1
    else:
        track.blue += 1


def associate(frames: list[list[Det]], motion: np.ndarray, params) -> list[Track]:
    alive: list[Track] = []
    finished: list[Track] = []
    tentative: list[Track] = []
    high = params["high"]
    coast = params["coast"]
    gate = 1.18
    for frame, dets in enumerate(frames):
        high_dets = [det for det in dets if det.conf >= high]
        low_dets = [det for det in dets if det.conf < high]
        preds = []
        for track in alive:
            box = predict_box(track.last(), track.vx, track.vy, frame, motion)
            x, y, w, h, _svx, _svy = box
            if w < 4 or h < 4:
                preds.append(None)
            else:
                preds.append((x, y, w, h))
        used_tracks = set()
        used_high = set()
        for row, col in match_tracks(alive, preds, high_dets, min_iou=0.0, gate=gate):
            update_track(alive[row], high_dets[col], motion)
            used_tracks.add(row)
            used_high.add(col)
        remain_idx = [i for i in range(len(alive)) if i not in used_tracks]
        remain_tracks = [alive[i] for i in remain_idx]
        remain_preds = [preds[i] for i in remain_idx]
        used_low = set()
        for row, col in match_tracks(remain_tracks, remain_preds, low_dets, min_iou=0.18, gate=1.05):
            update_track(remain_tracks[row], low_dets[col], motion)
            used_tracks.add(remain_idx[row])
            used_low.add(col)

        still = []
        for index, track in enumerate(alive):
            if index in used_tracks:
                still.append(track)
                continue
            track.misses += 1
            if track.misses <= coast:
                still.append(track)
            elif len(track.obs) >= 2 or track.obs[0].conf >= high:
                finished.append(track)
        alive = still

        matched_boxes = []
        for index in used_tracks:
            last = alive_last_box(alive, index, used_tracks, frames, frame)
            if last is not None:
                matched_boxes.append(last)
        # Spawn from high-confidence leftovers that are not a second box on a match.
        for col, det in enumerate(high_dets):
            if col in used_high:
                continue
            if overlaps_any(det, [high_dets[i] for i in used_high], 0.35):
                continue
            alive.append(Track(det))

        # Two-frame confirmation for a persistent low-score detection.
        tent_preds = []
        for track in tentative:
            box = predict_box(track.last(), track.vx, track.vy, frame, motion)
            tent_preds.append((box[0], box[1], box[2], box[3]))
        free_low = [low_dets[i] for i in range(len(low_dets)) if i not in used_low]
        used_tent = set()
        used_free = set()
        for row, col in match_tracks(tentative, tent_preds, free_low, min_iou=0.15, gate=1.1):
            det = free_low[col]
            update_track(tentative[row], det, motion)
            if len(tentative[row].obs) >= 2:
                alive.append(tentative[row])
            used_tent.add(row)
            used_free.add(col)
        new_tent = []
        for index, track in enumerate(tentative):
            if index not in used_tent and len(track.obs) < 2:
                continue
            if index not in used_tent:
                continue
            if track not in alive:
                new_tent.append(track)
        for col, det in enumerate(free_low):
            if col in used_free:
                continue
            if overlaps_any(det, high_dets, 0.3) or overlaps_any(det, [d for i, d in enumerate(low_dets) if i in used_low], 0.3):
                continue
            new_tent.append(Track(det))
        tentative = [track for track in new_tent if track not in alive and len(track.obs) < 2]

    for track in alive:
        if len(track.obs) >= 2 or track.obs[0].conf >= high:
            finished.append(track)
    return finished


def alive_last_box(alive, index, used_tracks, frames, frame):
    return None


def overlaps_any(det: Det, others: list[Det], iou_thr: float) -> bool:
    box = (det.x, det.y, det.w, det.h)
    for other in others:
        same = det.team == other.team
        iou = box_iou(box, (other.x, other.y, other.w, other.h))
        contain = containment(det, other)
        if not same and iou < 0.75 and contain < 0.85:
            continue
        if iou >= iou_thr or contain >= 0.62:
            return True
    return False


def prediction_coverage(pred, det: Det) -> float:
    """Share of a predicted player box covered by a detection."""
    px, py, pw, ph = pred
    left = max(px, det.x)
    top = max(py, det.y)
    right = min(px + pw, det.x + det.w)
    bottom = min(py + ph, det.y + det.h)
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    return inter / max(pw * ph, 1e-6)


def _match_options(params) -> dict:
    return {
        "reid_w": params.get("reid_assoc", 0.0),
        "team_penalty": params.get("team_penalty", 0.0),
        "team_margin": params.get("team_margin", 1.0),
        "team_partial": params.get("team_partial", 0.0),
    }


def associate_clean(frames: list[list[Det]], motion: np.ndarray, params) -> list[Track]:
    """Forward association. Tentative low-score tracks need a second hit."""
    alive: list[Track] = []
    finished: list[Track] = []
    tentative: list[Track] = []
    high = params["high"]
    for frame, dets in enumerate(frames):
        high_dets = [det for det in dets if det.conf >= high]
        low_dets = [det for det in dets if det.conf < high]
        preds = [predicted_tuple(track, frame, motion) for track in alive]
        used_tracks: set[int] = set()
        used_dets: set[int] = set()

        for row, col in match_tracks(alive, preds, high_dets, min_iou=0.0, gate=1.55, **_match_options(params)):
            update_track(alive[row], high_dets[col], motion)
            used_tracks.add(row)
            used_dets.add(id(high_dets[col]))

        remain_i = [i for i in range(len(alive)) if i not in used_tracks]
        remain = [alive[i] for i in remain_i]
        remain_preds = [preds[i] for i in remain_i]
        for row, col in match_tracks(remain, remain_preds, low_dets, min_iou=0.0, gate=1.45, **_match_options(params)):
            update_track(remain[row], low_dets[col], motion)
            used_tracks.add(remain_i[row])
            used_dets.add(id(low_dets[col]))

        still = []
        for index, track in enumerate(alive):
            if index in used_tracks:
                still.append(track)
                continue
            track.misses += 1
            if track.misses <= params["coast"]:
                still.append(track)
            elif _keep_finished(track, params):
                finished.append(track)
        alive = still

        taken = [det for det in dets if id(det) in used_dets]
        for det in high_dets:
            if id(det) in used_dets:
                continue
            if overlaps_any(det, taken, 0.32):
                continue
            alive.append(Track(det))
            used_dets.add(id(det))
            taken.append(det)

        tent_preds = [predicted_tuple(track, frame, motion) for track in tentative]
        free = [det for det in low_dets if id(det) not in used_dets and not overlaps_any(det, taken, 0.32)]
        used_tent = set()
        next_tent = []
        for row, col in match_tracks(tentative, tent_preds, free, min_iou=0.0, gate=1.45, **_match_options(params)):
            update_track(tentative[row], free[col], motion)
            used_tent.add(row)
            used_dets.add(id(free[col]))
            alive.append(tentative[row])
        for index, track in enumerate(tentative):
            if index in used_tent:
                continue
            # A one-frame tentative dies. Do not carry false boxes forward.
        for det in free:
            if id(det) in used_dets:
                continue
            if overlaps_any(det, taken, 0.32):
                continue
            next_tent.append(Track(det))
        tentative = next_tent

    for track in alive:
        if _keep_finished(track, params):
            finished.append(track)
    return finished


def predicted_tuple(track: Track, frame: int, motion: np.ndarray):
    if track.last().frame == frame:
        last = track.last()
        return (last.x, last.y, last.w, last.h)
    x, y, w, h, _vx, _vy = predict_box(track.last(), track.vx, track.vy, frame, motion)
    if w < 4 or h < 4:
        return None
    return (x, y, w, h)


def _keep_finished(track: Track, params) -> bool:
    if len(track.obs) >= params["min_hits"]:
        return True
    if len(track.obs) >= 2 and max(det.conf for det in track.obs) >= params["high"]:
        return True
    return False


def reverse_motion(motion: np.ndarray) -> np.ndarray:
    nframes = motion.shape[0]
    rev = np.zeros_like(motion)
    rev[0] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    for step in range(1, nframes):
        rev[step] = invert_affine(motion[nframes - step])
    return rev


def backward_ids(frames: list[list[Det]], motion: np.ndarray, params) -> dict[int, int]:
    nframes = len(frames)
    reversed_frames: list[list[Det]] = []
    for step in range(nframes):
        original = nframes - 1 - step
        reversed_frames.append([det.clone_at(step) for det in frames[original]])
    tracks = associate_clean(reversed_frames, reverse_motion(motion), params)
    mapping: dict[int, int] = {}
    for index, track in enumerate(tracks):
        for det in track.obs:
            source = det.src if det.src is not None else det
            mapping[source.uid] = index
    return mapping


def marker_near(det: Det, mx: float, my: float) -> bool:
    torso = det.y + 0.18 * det.h <= my <= det.y + 0.70 * det.h and det.x <= mx <= det.x + det.w
    if torso:
        return False
    foot = np.hypot(mx - det.cx, my - (det.y + det.h))
    head = np.hypot(mx - det.cx, my - det.y)
    return foot < 0.62 * det.h or head < 0.48 * det.h


def count_markers(obs: list[Det], markers: dict[int, list[tuple[float, float]]]) -> int:
    hits = 0
    for det in obs:
        for mx, my in markers.get(det.frame, []):
            if marker_near(det, mx, my):
                hits += 1
                break
    return hits


class Tlet:
    def __init__(self, obs: list[Det], team: str, vx: float, vy: float, marker_hits: int, bwd: set[int]):
        self.obs = sorted(obs, key=lambda det: det.frame)
        self.team = team
        self.vx = vx
        self.vy = vy
        self.marker_hits = marker_hits
        self.bwd = bwd
        feats = [det.hist for det in self.obs[-8:]]
        self.hist = np.mean(feats, axis=0) if feats else np.zeros(48, np.float32)
        embs = [det.emb for det in self.obs if det.emb is not None]
        if embs:
            mean = np.mean(embs, axis=0)
            norm = float(np.linalg.norm(mean))
            self.emb = None if norm < 1e-8 else mean / norm
        else:
            self.emb = None

    @property
    def start(self) -> int:
        return self.obs[0].frame

    @property
    def end(self) -> int:
        return self.obs[-1].frame

    def score(self) -> float:
        return float(sum(det.conf for det in self.obs))


def shares_backward(a: Tlet, b: Tlet) -> bool:
    return len(a.bwd & b.bwd) > 0


def link_cost(a: Tlet, b: Tlet, motion: np.ndarray, params) -> float | None:
    if a.team != b.team or a.end > b.start:
        # Allow a 2-frame overlap; caller requires a before b in start time.
        if not (a.team == b.team and a.start <= b.start and a.end <= b.end and b.start - a.end >= -2):
            return None
    gap = b.start - a.end - 1
    if gap > params["stitch_gap"]:
        return None
    if gap < -2:
        return None
    if gap < 0:
        shared = {}
        for det in a.obs:
            shared[det.frame] = det
        dists = []
        for det in b.obs:
            other = shared.get(det.frame)
            if other is None:
                continue
            dists.append(np.hypot(det.cx - other.cx, det.cy - other.cy) / max(24.0, det.h))
        if not dists or float(np.median(dists)) > 0.55:
            return None
    x, y, w, h, _vx, _vy = predict_box(a.obs[-1], a.vx, a.vy, b.start, motion)
    dist = float(np.hypot(x + 0.5 * w - b.obs[0].cx, y + h - (b.obs[0].y + b.obs[0].h)))
    norm = dist / max(30.0, 0.5 * (h + b.obs[0].h))
    marker_lock = a.marker_hits >= 4 and b.marker_hits >= 4
    linked = shares_backward(a, b) or marker_lock
    # A short gap that both directions already called the same person is a break
    # in the forward pass, not a different player.
    if shares_backward(a, b) and gap <= 12:
        limit = 6.0
    elif linked:
        limit = 2.8
    else:
        limit = 1.9
    if norm > limit:
        return None
    app = cosine(a.hist, b.hist)
    cost = norm + 0.55 * (1.0 - app) + 0.035 * max(0, gap)
    reid_w = float(params.get("reid_stitch", 0.0))
    reid_cos = None
    if reid_w > 0 and a.emb is not None and b.emb is not None:
        reid_cos = cosine(a.emb, b.emb)
        # Same-player cosine is about 0.4 and a teammate is about 0.2.
        cost += reid_w * (1.0 - reid_cos)
        if reid_cos < 0.12 and not marker_lock:
            return None
    # Do not let the backward pass glue two bodies the embedding says are different.
    if shares_backward(a, b) and (reid_cos is None or reid_cos >= 0.28):
        cost *= 0.45
    elif reid_cos is not None and reid_cos >= 0.50:
        cost *= 0.75
    if marker_lock:
        cost *= 0.35
    return cost


def chain_members(index: int, succ: dict[int, int], pred: dict[int, int]) -> list[int]:
    nodes = [index]
    cursor = index
    seen = {index}
    while cursor in pred and pred[cursor] not in seen:
        cursor = pred[cursor]
        seen.add(cursor)
        nodes.append(cursor)
    cursor = index
    while cursor in succ and succ[cursor] not in seen:
        cursor = succ[cursor]
        seen.add(cursor)
        nodes.append(cursor)
    return nodes


def frames_of(indices: list[int], tlets: list[Tlet]) -> set[int]:
    frames: set[int] = set()
    for index in indices:
        for det in tlets[index].obs:
            frames.add(det.frame)
    return frames


def stitch(tlets: list[Tlet], motion: np.ndarray, params) -> list[list[Tlet]]:
    order = sorted(range(len(tlets)), key=lambda i: (tlets[i].start, tlets[i].end))
    candidates = []
    for ia in order:
        for ib in order:
            if ia == ib:
                continue
            if tlets[ia].start > tlets[ib].start:
                continue
            cost = link_cost(tlets[ia], tlets[ib], motion, params)
            if cost is None:
                continue
            candidates.append((cost, ia, ib))
    candidates.sort(key=lambda item: item[0])
    succ: dict[int, int] = {}
    pred: dict[int, int] = {}
    for _cost, ia, ib in candidates:
        if ia in succ or ib in pred:
            continue
        left = frames_of(chain_members(ia, succ, pred), tlets)
        right = frames_of(chain_members(ib, succ, pred), tlets)
        overlap = left & right
        if len(overlap) > 2:
            continue
        succ[ia] = ib
        pred[ib] = ia
    chains = []
    seen = set()
    for index in range(len(tlets)):
        if index in seen or index in pred:
            continue
        chain = [tlets[index]]
        seen.add(index)
        cursor = index
        while cursor in succ:
            cursor = succ[cursor]
            chain.append(tlets[cursor])
            seen.add(cursor)
        chains.append(chain)
    for index in range(len(tlets)):
        if index not in seen:
            chains.append([tlets[index]])
    return chains


def chain_obs(chain: list[Tlet]) -> list[Det]:
    by_frame: dict[int, Det] = {}
    for tlet in chain:
        for det in tlet.obs:
            current = by_frame.get(det.frame)
            if current is None or det.conf > current.conf:
                by_frame[det.frame] = det
    return [by_frame[frame] for frame in sorted(by_frame)]


def chain_overlap_iou(a_obs: list[Det], b_obs: list[Det]) -> float:
    b_map = {det.frame: det for det in b_obs}
    values = []
    for det in a_obs:
        other = b_map.get(det.frame)
        if other is None:
            continue
        values.append(box_iou((det.x, det.y, det.w, det.h), (other.x, other.y, other.w, other.h)))
    if not values:
        return 0.0
    return float(np.median(values))


def drop_duplicate_chains(chains: list[list[Tlet]]) -> list[list[Tlet]]:
    obs = [chain_obs(chain) for chain in chains]
    teams = [chain[0].team for chain in chains]
    scores = [sum(det.conf for det in item) for item in obs]
    drop = set()
    for i in range(len(chains)):
        if i in drop:
            continue
        for j in range(i + 1, len(chains)):
            if j in drop or teams[i] != teams[j]:
                continue
            shared = len({det.frame for det in obs[i]} & {det.frame for det in obs[j]})
            if shared < 5:
                continue
            if chain_overlap_iou(obs[i], obs[j]) < 0.45:
                continue
            loser = i if scores[i] < scores[j] else j
            drop.add(loser)
    return [chain for index, chain in enumerate(chains) if index not in drop]


def published_frames(obs: list[Det], interp_gap: int) -> set[int]:
    """Frames a chain will have a box on: its observations plus gaps interpolate_chain fills."""
    frames = sorted(det.frame for det in obs)
    present = set(frames)
    for left, right in zip(frames, frames[1:]):
        if 1 < right - left <= interp_gap + 1:
            present.update(range(left + 1, right))
    return present


def limit_roster(
    chains: list[list[Tlet]],
    rescue_len: int = 0,
    rescue_conf: float = 1.0,
    interp_gap: int = 0,
) -> list[list[Tlet]]:
    """Drop weak extra chains so a team does not keep obvious false tracks.

    Long tracks are kept even if that leaves more than 7 ids. Forcing a merge
    or a drop of a real player costs more IDF1 than an extra short false track
    that already failed the duplicate test.

    Ranking is over the whole clip, so on a busy play a short chain of a real
    player can rank past 7 while far fewer teammates are on screen. With
    rescue_len > 0 such a chain is kept when it has at least rescue_len
    detections, one at rescue_conf or higher, and no frame where it would
    publish a box already holds 7 kept chains of its team.
    """
    kept = []
    by_team: dict[str, list[tuple[float, list[Tlet]]]] = {"red": [], "blue": []}
    for chain in chains:
        obs = chain_obs(chain)
        score = sum(det.conf for det in obs)
        by_team[chain[0].team].append((score, chain, obs))
    for team, items in by_team.items():
        items.sort(key=lambda item: item[0], reverse=True)
        occupied: collections.Counter = collections.Counter()
        for rank, (score, chain, obs) in enumerate(items):
            present = published_frames(obs, interp_gap)
            keep = rank < 7 or (len(obs) >= 20 and score >= 3.0)
            if not keep and rescue_len > 0:
                keep = (
                    len(obs) >= rescue_len
                    and max(det.conf for det in obs) >= rescue_conf
                    and all(occupied[frame] < 7 for frame in present)
                )
            if keep:
                kept.append(chain)
                occupied.update(present)
            # Otherwise an extra short or low-confidence chain: drop it.
    return kept


def smoothness(obs_by_frame: dict[int, Det]) -> float:
    frames = sorted(obs_by_frame)
    if len(frames) < 3:
        return 0.0
    cost = 0.0
    for index in range(1, len(frames) - 1):
        f0, f1, f2 = frames[index - 1], frames[index], frames[index + 1]
        a, b, c = obs_by_frame[f0], obs_by_frame[f1], obs_by_frame[f2]
        dt1 = max(1, f1 - f0)
        dt2 = max(1, f2 - f1)
        v1x = (b.cx - a.cx) / dt1
        v1y = (b.cy - a.cy) / dt1
        v2x = (c.cx - b.cx) / dt2
        v2y = (c.cy - b.cy) / dt2
        cost += (v2x - v1x) ** 2 + (v2y - v1y) ** 2
    return cost


def segments_cross(a0: Det, a1: Det, b0: Det, b1: Det) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    p1, p2 = (a0.cx, a0.cy), (a1.cx, a1.cy)
    p3, p4 = (b0.cx, b0.cy), (b1.cx, b1.cy)
    d1 = orient(p1, p2, p3)
    d2 = orient(p1, p2, p4)
    d3 = orient(p3, p4, p1)
    d4 = orient(p3, p4, p2)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    # Near miss: centers pass within a quarter of a body height while x-order flips.
    close = min(
        np.hypot(p2[0] - p4[0], p2[1] - p4[1]),
        np.hypot(p1[0] - p3[0], p1[1] - p3[1]),
    )
    height = max(20.0, 0.5 * (a1.h + b1.h))
    order0 = np.sign(p1[0] - p3[0])
    order1 = np.sign(p2[0] - p4[0])
    return bool(close < 0.85 * height and order0 != 0 and order1 != 0 and order0 != order1)


def unswap(paths: list[dict[int, Det]], teams: list[str]) -> int:
    swaps = 0
    for _pass in range(3):
        changed = False
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if teams[i] != teams[j]:
                    continue
                overlap = sorted(set(paths[i]) & set(paths[j]))
                if len(overlap) < 6:
                    continue
                base = smoothness(paths[i]) + smoothness(paths[j])
                best = None
                for k in range(1, len(overlap)):
                    prev, frame = overlap[k - 1], overlap[k]
                    if frame - prev > 2:
                        continue
                    if not segments_cross(paths[i][prev], paths[i][frame], paths[j][prev], paths[j][frame]):
                        continue
                    suffix = [f for f in overlap if f >= frame]
                    trial_i = dict(paths[i])
                    trial_j = dict(paths[j])
                    for f in suffix:
                        trial_i[f], trial_j[f] = paths[j][f], paths[i][f]
                    cost = smoothness(trial_i) + smoothness(trial_j)
                    if best is None or cost < best[0]:
                        best = (cost, suffix)
                if best is None:
                    continue
                if base - best[0] <= max(12.0, 0.08 * base):
                    continue
                for f in best[1]:
                    paths[i][f], paths[j][f] = paths[j][f], paths[i][f]
                swaps += 1
                changed = True
        if not changed:
            break
    return swaps


def interpolate_chain(obs: list[Det], motion: np.ndarray, hud: np.ndarray, width: int, height: int, params) -> list[Det]:
    if not obs:
        return []
    by_frame = {det.frame: det for det in obs}
    frames = sorted(by_frame)
    filled = dict(by_frame)
    for left, right in zip(frames, frames[1:]):
        gap = right - left - 1
        if gap <= 0 or gap > params["interp_gap"]:
            continue
        a, b = by_frame[left], by_frame[right]
        for frame in range(left + 1, right):
            alpha = (frame - left) / (right - left)
            ax, ay, aw, ah, _vx, _vy = predict_box(a, 0.0, 0.0, frame, motion)
            back = compose(motion, frame, right)
            bx, by, bw, bh = transform_box(invert_affine(back) if False else compose(motion, right, frame), b.x, b.y, b.w, b.h)
            # compose(motion, right, frame) maps right -> frame. Use that directly.
            bx, by, bw, bh = transform_box(compose(motion, right, frame), b.x, b.y, b.w, b.h)
            det = Det(
                -1,
                frame,
                (1 - alpha) * ax + alpha * bx,
                (1 - alpha) * ay + alpha * by,
                max(4.0, (1 - alpha) * aw + alpha * bw),
                max(4.0, (1 - alpha) * ah + alpha * bh),
                0.85 * min(a.conf, b.conf),
                a.team,
                a.hist,
                emb=a.emb,
            )
            if det.x + det.w < 8 or det.y + det.h < 8 or det.x > width - 8 or det.y > height - 8:
                continue
            if hud_fraction(hud, det, width, height) > params["hud_max"]:
                continue
            filled[frame] = det
    return [filled[frame] for frame in sorted(filled)]


def merged_path_frames(paths: list[dict[int, Det]], params) -> dict[int, set[int]]:
    """Find output boxes that enclose another track like a tackle-pile box."""
    suspicious: dict[int, set[int]] = {}
    frames = sorted({frame for path in paths for frame in path})
    for frame in frames:
        present = [(index, path[frame]) for index, path in enumerate(paths) if frame in path]
        for index, det in present:
            if det.uid < 0:
                continue
            det_area = det.w * det.h
            for other_index, other in present:
                if other_index == index:
                    continue
                other_area = other.w * other.h
                if det_area < params["merge_area"] * other_area:
                    continue
                if (
                    det.w < params["merge_scale"] * other.w
                    and det.h < params["merge_scale"] * other.h
                ):
                    continue
                other_box = (other.x, other.y, other.w, other.h)
                if prediction_coverage(other_box, det) < params["merge_cover"]:
                    continue
                suspicious.setdefault(index, set()).add(frame)
                break
    return suspicious


def _occluded_det(a: Det, b: Det, frame: int, motion: np.ndarray) -> Det:
    alpha = (frame - a.frame) / max(b.frame - a.frame, 1)
    ax, ay, aw, ah = transform_box(
        compose(motion, a.frame, frame), a.x, a.y, a.w, a.h
    )
    bx, by, bw, bh = transform_box(
        compose(motion, b.frame, frame), b.x, b.y, b.w, b.h
    )
    return Det(
        -2,
        frame,
        (1 - alpha) * ax + alpha * bx,
        (1 - alpha) * ay + alpha * by,
        max(4.0, (1 - alpha) * aw + alpha * bw),
        max(4.0, (1 - alpha) * ah + alpha * bh),
        0.05,
        a.team,
        a.hist,
        emb=a.emb,
    )


def repair_merged_occlusions(paths: list[dict[int, Det]], motion: np.ndarray, params) -> int:
    """Replace a group box and its following gap with per-track predictions."""
    suspicious = merged_path_frames(paths, params)
    repaired = 0
    for index, marked in suspicious.items():
        remaining = sorted(marked)
        groups = []
        for frame in remaining:
            if not groups or frame - groups[-1][-1] > 2:
                groups.append([frame])
            else:
                groups[-1].append(frame)

        path = paths[index]
        for group in groups:
            first, last = group[0], group[-1]
            before = sorted(frame for frame in path if frame < first)
            if not before:
                continue
            sample = before[-8:]
            baseline_area = float(
                np.percentile([path[frame].w * path[frame].h for frame in sample], 40)
            )
            clean_before = [
                frame
                for frame in sample
                if path[frame].w * path[frame].h <= 2.0 * baseline_area
            ]
            if not clean_before:
                continue
            left = max(clean_before)
            left_area = path[left].w * path[left].h
            clean_limit = 2.0 * max(baseline_area, left_area)
            rights = sorted(
                frame
                for frame in path
                if frame > last
                and frame not in marked
                and path[frame].w * path[frame].h <= clean_limit
            )
            if not rights:
                continue
            right = rights[0]
            # A group observation can precede a detector gap. Repair through
            # that gap only when the same identity returns inside the normal
            # coasting horizon.
            if right - last - 1 > params["coast"]:
                continue
            a, b = path[left], path[right]
            for frame in range(left + 1, right):
                path[frame] = _occluded_det(a, b, frame, motion)
                repaired += 1
    return repaired


def _mean_embedding(detections: list[Det]) -> np.ndarray | None:
    embeddings = [det.emb for det in detections if det.emb is not None]
    if not embeddings:
        return None
    mean = np.mean(embeddings, axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-8:
        return None
    return mean / norm


def repair_bad_handoffs(paths: list[dict[int, Det]], teams: list[str]) -> int:
    """Move a late continuation back onto a badly switched opening path.

    A long-lived roster path can jump to a nearby opponent while a new path
    starts on the original player. Re-ID makes that failure distinctive: the
    old path's suffix is nearly unrelated to its opening, while the nearby
    new path is a substantially better match. Transfer that continuation and
    discard the wrong suffix. Conservative thresholds limit this to severe
    handoff failures rather than ordinary appearance changes.
    """
    repaired = 0
    used = set()
    for main_i, main in enumerate(paths):
        if not main or min(main) != 0:
            continue
        prefix = [main[frame] for frame in sorted(main) if frame <= 60]
        reference = _mean_embedding(prefix)
        if reference is None:
            continue
        best = None
        for candidate_i, candidate in enumerate(paths):
            if candidate_i == main_i or candidate_i in used or not candidate:
                continue
            if teams[candidate_i] != teams[main_i] or len(candidate) < 20:
                continue
            start = min(candidate)
            if start < 30 or start > max(main):
                continue
            nearby = [
                main[frame]
                for frame in range(start - 3, start + 4)
                if frame in main
            ]
            if not nearby:
                continue
            first = candidate[start]
            spatial = min(
                np.hypot(
                    det.cx - first.cx,
                    det.y + det.h - (first.y + first.h),
                ) / max(30.0, 0.5 * (det.h + first.h))
                for det in nearby
            )
            if spatial > 0.65:
                continue
            candidate_head = _mean_embedding(
                [candidate[frame] for frame in sorted(candidate)[:15]])
            current_head = _mean_embedding(
                [main[frame] for frame in sorted(main) if frame >= start][:15])
            if candidate_head is None or current_head is None:
                continue
            candidate_match = cosine(reference, candidate_head)
            current_match = cosine(reference, current_head)
            margin = candidate_match - current_match
            if candidate_match < 0.18 or current_match > 0.08 or margin < 0.10:
                continue
            choice = (margin, candidate_match, -spatial, candidate_i, start)
            if best is None or choice > best:
                best = choice
        if best is None:
            continue
        _margin, _match, _distance, candidate_i, start = best
        candidate = paths[candidate_i]
        for frame in [frame for frame in main if frame >= start]:
            del main[frame]
        for frame, det in candidate.items():
            if frame >= start:
                main[frame] = det
        candidate.clear()
        used.add(candidate_i)
        repaired += 1
    return repaired


def split_sustained_team_drift(
    paths: list[dict[int, Det]],
    teams: list[str],
    trigger_frames: int = 24,
    split_frames: int = 4,
    max_similarity: float = 0.18,
) -> tuple[int, int]:
    """Split opponent handoffs without reacting to brief color mistakes.

    ``teams`` is the stable roster team for each path, while ``Det.team`` is
    still the detector's per-frame color evidence.  A path is considered
    corrupted only when it follows the opposite team for a long continuous
    run and that run is a poor re-ID match for the path's opening.  Once that
    strong signal exists, shorter opposing-team runs on the same corrupted
    path become new tracks of their observed team. This preserves a real
    player detection without drawing the old identity over that opponent.
    """

    moved = 0
    split_paths: list[dict[int, Det]] = []
    split_teams: list[str] = []
    for path, stable_team in list(zip(paths, teams)):
        if not path:
            continue
        frames = sorted(path)
        anchor_detections = [
            path[frame]
            for frame in frames[:40]
            if path[frame].team == stable_team
        ]
        anchor = _mean_embedding(anchor_detections)
        if anchor is None:
            continue

        runs: list[list[int]] = []
        run: list[int] = []
        for frame in frames:
            if path[frame].team != stable_team:
                if run and frame != run[-1] + 1:
                    runs.append(run)
                    run = []
                run.append(frame)
            elif run:
                runs.append(run)
                run = []
        if run:
            runs.append(run)

        corrupted = False
        for opposing_run in runs:
            if len(opposing_run) < trigger_frames:
                continue
            run_embedding = _mean_embedding(
                [path[frame] for frame in opposing_run]
            )
            if run_embedding is not None and cosine(anchor, run_embedding) < max_similarity:
                corrupted = True
                break
        if not corrupted:
            continue

        current_split: dict[int, Det] | None = None
        previous_end: int | None = None
        for opposing_run in runs:
            if len(opposing_run) < split_frames:
                continue
            # A one-frame color interruption commonly divides one continuous
            # opponent track. Keep those pieces under one new identity.
            if previous_end is None or opposing_run[0] - previous_end > 2:
                current_split = {}
                split_paths.append(current_split)
                split_teams.append(path[opposing_run[0]].team)
            for frame in opposing_run:
                current_split[frame] = path.pop(frame)
                moved += 1
            previous_end = opposing_run[-1]
    paths.extend(split_paths)
    teams.extend(split_teams)
    return moved, len(split_paths)


def suppress_interpolated_cross_team_duplicates(
    paths: list[dict[int, Det]], teams: list[str]
) -> int:
    """Drop an interpolated box nested inside an observed opponent box.

    Interpolation occasionally carries a blue path across red number 72 while
    the real red path is already present. Detector observations from two teams
    are kept for tackle piles; only a persistent run of synthetic ``uid ==
    -1`` duplicates is eligible for removal. Merge-repair boxes use ``uid ==
    -2`` and are also preserved.
    """
    candidates: dict[tuple[int, int], list[int]] = {}
    frames = sorted({frame for path in paths for frame in path})
    for frame in frames:
        present = [index for index, path in enumerate(paths) if frame in path]
        for offset, first_i in enumerate(present):
            first = paths[first_i][frame]
            for second_i in present[offset + 1:]:
                if teams[first_i] == teams[second_i]:
                    continue
                second = paths[second_i][frame]
                if containment(first, second) < 0.80:
                    continue
                if first.uid == -1 and second.uid >= 0:
                    candidates.setdefault((first_i, second_i), []).append(frame)
                elif second.uid == -1 and first.uid >= 0:
                    candidates.setdefault((second_i, first_i), []).append(frame)
    drops = set()
    for (drop_i, _keep_i), matching in candidates.items():
        run = []
        runs = []
        for frame in matching:
            if run and frame != run[-1] + 1:
                runs.append(run)
                run = []
            run.append(frame)
        if run:
            runs.append(run)
        # Real opponents can nest briefly while crossing or tackling. The bad
        # carried path stays nested frame after frame after its detector is
        # gone, so require persistence before removing it.
        for run in runs:
            if len(run) < 4:
                continue
            for frame in run:
                drops.add((drop_i, frame))
    for index, frame in drops:
        del paths[index][frame]
    return len(drops)


def per_frame_suppress(paths: list[dict[int, Det]], teams: list[str]) -> int:
    removed = 0
    frames = sorted({frame for path in paths for frame in path})
    for frame in frames:
        present = [i for i, path in enumerate(paths) if frame in path]
        present.sort(key=lambda i: paths[i][frame].conf * max(paths[i][frame].w * paths[i][frame].h, 1.0), reverse=True)
        keep = []
        for index in present:
            det = paths[index][frame]
            drop = False
            for other_i in keep:
                other = paths[other_i][frame]
                # These boxes are distinct identities carried through a known
                # multi-player detector merge. Their overlap is the reason the
                # repair exists, not evidence that either track is duplicate.
                if det.uid == -2 or other.uid == -2:
                    continue
                if teams[index] != teams[other_i]:
                    continue
                iou = box_iou((det.x, det.y, det.w, det.h), (other.x, other.y, other.w, other.h))
                if containment(det, other) >= 0.62 or iou >= 0.35:
                    drop = True
                    break
            if drop:
                del paths[index][frame]
                removed += 1
            else:
                keep.append(index)
    return removed


def renumber(paths: list[dict[int, Det]], teams: list[str]) -> list[int]:
    red, blue, later = [], [], []
    for index, path in enumerate(paths):
        if 0 in path:
            (red if teams[index] == "red" else blue).append(index)
        else:
            later.append(index)
    red.sort(key=lambda i: paths[i][0].cx)
    blue.sort(key=lambda i: paths[i][0].cx)
    later.sort(key=lambda i: (min(paths[i]), paths[i][min(paths[i])].cx))
    order = red + blue + later
    ids = [0] * len(paths)
    for number, index in enumerate(order, start=1):
        ids[index] = number
    return ids


def tracks_to_tlets(tracks: list[Track], markers, bwd: dict[int, int], params) -> list[Tlet]:
    tlets = []
    for track in tracks:
        if not _keep_finished(track, params):
            continue
        obs = sorted(track.obs, key=lambda det: det.frame)
        # Observations may be backward-clones. Use the source det so frame numbers
        # are the original video indices.
        original = []
        for det in obs:
            original.append(det.src if det.src is not None else det)
        original = sorted(original, key=lambda det: det.frame)
        bwd_ids = {bwd[det.uid] for det in original if det.uid in bwd}
        tlets.append(
            Tlet(
                original,
                track.team(),
                track.vx,
                track.vy,
                count_markers(original, markers),
                bwd_ids,
            )
        )
    return tlets


def export_paths(paths, teams, ids, width, height) -> list[dict]:
    rows = []
    for index, path in enumerate(paths):
        for frame in sorted(path):
            det = path[frame]
            x = min(max(det.x, 0.0), width - 1)
            y = min(max(det.y, 0.0), height - 1)
            w = min(det.w, width - x)
            h = min(det.h, height - y)
            if w < 2 or h < 2:
                continue
            rows.append(
                {
                    "frame": frame,
                    "id": ids[index],
                    "team": teams[index],
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "w": round(w, 2),
                    "h": round(h, 2),
                    "conf": round(float(det.conf), 6),
                }
            )
    rows.sort(key=lambda row: (row["frame"], row["id"]))
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "id", "team", "x", "y", "w", "h", "conf"])
        writer.writeheader()
        writer.writerows(rows)


def load_reid(stem: Path, weights: str | None) -> np.ndarray | None:
    if not weights:
        return None
    path = stem.with_suffix(".reid.npy")
    meta_path = stem.with_suffix(".reid.json")
    if not path.is_file() or not meta_path.is_file():
        raise RuntimeError(f"missing re-id cache for {stem}; run ml/reid/embed_cache.py")
    meta = json.loads(meta_path.read_text())
    if str(meta.get("weights")) != str(weights):
        raise RuntimeError(f"re-id cache {path} was built with {meta.get('weights')}, not {weights}")
    emb = np.load(path)
    return emb


def run(stem: Path, out: Path, params: dict) -> dict:
    started = time.perf_counter()
    meta, rows, hist, motion, hud, markers = load_cache(stem)
    emb = load_reid(stem, params.get("reid"))
    if emb is not None and len(emb) != len(rows):
        raise RuntimeError(f"reid rows {len(emb)} != csv rows {len(rows)}")
    icons = load_icons(stem, len(rows))
    if icons is None:
        print(f"no {stem.with_suffix('.icons.npy').name}; '1' plaque boxes are kept", flush=True)
    frames = build_frames(meta, rows, hist, hud, params, emb=emb, icons=icons)
    kept = sum(len(item) for item in frames)
    forward = associate_clean(frames, motion, params)
    bwd = backward_ids(frames, motion, params)
    tlets = tracks_to_tlets(forward, markers, bwd, params)
    chains = stitch(tlets, motion, params)
    chains = drop_duplicate_chains(chains)
    chains = limit_roster(
        chains,
        rescue_len=params["roster_rescue_len"],
        rescue_conf=params["roster_rescue_conf"],
        interp_gap=params["interp_gap"],
    )
    kept_chains, icon_chains, icon_stripped = strip_icon_chains(chains, icons, params["icon_chain_share"])
    width, height = int(meta["width"]), int(meta["height"])
    paths = []
    teams = []
    for chain, obs in kept_chains:
        filled = interpolate_chain(obs, motion, hud, width, height, params)
        paths.append({det.frame: det for det in filled})
        teams.append(chain[0].team)
    handoffs = repair_bad_handoffs(paths, teams)
    swaps = unswap(paths, teams)
    repaired = repair_merged_occlusions(paths, motion, params)
    removed = per_frame_suppress(paths, teams)
    cross_team_removed = suppress_interpolated_cross_team_duplicates(paths, teams)
    drift_reassigned, drift_tracks = split_sustained_team_drift(
        paths,
        teams,
        trigger_frames=params["drift_trigger"],
        split_frames=params["drift_split"],
        max_similarity=params["drift_similarity"],
    )
    # Splitting changes the stable team of these boxes, so repeat same-team
    # suppression in case an established red path already covers one.
    removed += per_frame_suppress(paths, teams)
    cone_run_boxes = remove_cone_runs(
        paths,
        load_warm(stem, len(rows)),
        reference_transforms(motion),
        width,
        height,
        params["cone_run_frames"],
        params["cone_warm"],
    )
    # The stable chain team is what gets exported. Preserve raw per-frame team
    # evidence only until all drift/occlusion repairs have consumed it.
    for path, team in zip(paths, teams):
        for det in path.values():
            det.team = team
    # Drop identities that lost every box.
    packed_paths, packed_teams = [], []
    for path, team in zip(paths, teams):
        if path:
            packed_paths.append(path)
            packed_teams.append(team)
    ids = renumber(packed_paths, packed_teams)
    out_rows = export_paths(packed_paths, packed_teams, ids, width, height)
    write_rows(out, out_rows)
    frame0 = [row for row in out_rows if row["frame"] == 0]
    summary = {
        "approach": "assoc",
        "cache": str(stem),
        "out": str(out),
        "params": params,
        "raw_boxes": len(rows),
        "icon_boxes_before": 0 if icons is None else int((icons == ICON_BEFORE).sum()),
        "icon_boxes_after": 0 if icons is None else int((icons == ICON_AFTER).sum()),
        "icon_chains_dropped": icon_chains,
        "icon_boxes_stripped": icon_stripped,
        "cone_run_boxes_removed": cone_run_boxes,
        "filtered_boxes": kept,
        "forward_tracklets": len(forward),
        "stitched_chains": len(chains),
        "output_boxes": len(out_rows),
        "output_ids": len(set(row["id"] for row in out_rows)),
        "frame0": len(frame0),
        "unswaps": swaps,
        "repaired_handoffs": handoffs,
        "merged_occlusion_boxes": repaired,
        "duplicate_boxes_removed": removed,
        "cross_team_interpolations_removed": cross_team_removed,
        "sustained_team_drift_reassigned": drift_reassigned,
        "sustained_team_drift_tracks": drift_tracks,
        "seconds": round(time.perf_counter() - started, 3),
        "nframes": int(meta["nframes"]),
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print("frame0", [(row["id"], row["team"], round(row["x"] + 0.5 * row["w"], 1)) for row in frame0], flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera-compensated offline tracker.")
    parser.add_argument("--cache", type=Path, required=True, help="Stem, e.g. /tmp/ml_trk/det/clip4_yolo11m")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reid", type=str, default="", help="Embedding weights. Requires a matching <cache>.reid.npy.")
    parser.add_argument("--reid-assoc", type=float, default=0.45, help="Weight on ambiguous same-team association ties.")
    parser.add_argument("--reid-stitch", type=float, default=0.85, help="Weight when stitching tracklets across gaps.")
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=float if isinstance(value, float) else int, default=value)
    return parser.parse_args(argv)


def params_from_args(args: argparse.Namespace) -> dict:
    params = {}
    for key, value in DEFAULTS.items():
        params[key] = getattr(args, key)
    params["reid"] = args.reid or None
    params["reid_assoc"] = float(args.reid_assoc) if args.reid else 0.0
    params["reid_stitch"] = float(args.reid_stitch) if args.reid else 0.0
    return params


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.cache, args.out, params_from_args(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
