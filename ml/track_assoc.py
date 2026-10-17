#!/usr/bin/env python3
"""Approach A: detection tracking with camera motion and a fixed roster.

Pipeline, all offline and zero-shot:

1. Load cached every-frame boxes (see ml/detect_all.py).
2. Drop HUD boxes and partial duplicate boxes.
3. Associate forward and backward (ByteTrack-style high/low passes) in the
   camera-compensated frame.
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


def build_frames(meta, rows, hist, hud, params, emb=None) -> list[list[Det]]:
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
        if det.frame < 0 or det.frame >= nframes:
            continue
        per_frame[det.frame].append(det)
    for frame in range(nframes):
        per_frame[frame] = suppress_duplicates(per_frame[frame], params["contain"])
    return per_frame


def pair_cost(pred, det: Det, track: Track, min_iou: float) -> float:
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
    return cost


def match_tracks(tracks, preds, dets, min_iou: float, gate: float, reid_w: float = 0.0):
    if not tracks or not dets:
        return []
    cost = np.full((len(tracks), len(dets)), 1e5, dtype=np.float64)
    for i, track in enumerate(tracks):
        if preds[i] is None:
            continue
        for j, det in enumerate(dets):
            cost[i, j] = pair_cost(preds[i], det, track, min_iou)
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

        for row, col in match_tracks(alive, preds, high_dets, min_iou=0.0, gate=1.55, reid_w=params.get("reid_assoc", 0.0)):
            update_track(alive[row], high_dets[col], motion)
            used_tracks.add(row)
            used_dets.add(id(high_dets[col]))

        remain_i = [i for i in range(len(alive)) if i not in used_tracks]
        remain = [alive[i] for i in remain_i]
        remain_preds = [preds[i] for i in remain_i]
        for row, col in match_tracks(remain, remain_preds, low_dets, min_iou=0.0, gate=1.45, reid_w=params.get("reid_assoc", 0.0)):
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
        for row, col in match_tracks(tentative, tent_preds, free, min_iou=0.0, gate=1.45, reid_w=params.get("reid_assoc", 0.0)):
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


def limit_roster(chains: list[list[Tlet]]) -> list[list[Tlet]]:
    """Drop weak extra chains so a team does not keep obvious false tracks.

    Long tracks are kept even if that leaves more than 7 ids. Forcing a merge
    or a drop of a real player costs more IDF1 than an extra short false track
    that already failed the duplicate test.
    """
    kept = []
    by_team: dict[str, list[tuple[float, list[Tlet]]]] = {"red": [], "blue": []}
    for chain in chains:
        obs = chain_obs(chain)
        score = sum(det.conf for det in obs)
        by_team[chain[0].team].append((score, chain, len(obs)))
    for team, items in by_team.items():
        items.sort(key=lambda item: item[0], reverse=True)
        for rank, (score, chain, length) in enumerate(items):
            if rank < 7:
                kept.append(chain)
                continue
            if length >= 20 and score >= 3.0:
                kept.append(chain)
                continue
            # Extra short or low-confidence chain: drop it.
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
    frames = build_frames(meta, rows, hist, hud, params, emb=emb)
    kept = sum(len(item) for item in frames)
    forward = associate_clean(frames, motion, params)
    bwd = backward_ids(frames, motion, params)
    tlets = tracks_to_tlets(forward, markers, bwd, params)
    chains = stitch(tlets, motion, params)
    chains = drop_duplicate_chains(chains)
    chains = limit_roster(chains)
    width, height = int(meta["width"]), int(meta["height"])
    paths = []
    teams = []
    for chain in chains:
        obs = chain_obs(chain)
        for det in obs:
            det.team = chain[0].team
        filled = interpolate_chain(obs, motion, hud, width, height, params)
        for det in filled:
            det.team = chain[0].team
        paths.append({det.frame: det for det in filled})
        teams.append(chain[0].team)
    swaps = unswap(paths, teams)
    removed = per_frame_suppress(paths, teams)
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
        "filtered_boxes": kept,
        "forward_tracklets": len(forward),
        "stitched_chains": len(chains),
        "output_boxes": len(out_rows),
        "output_ids": len(set(row["id"] for row in out_rows)),
        "frame0": len(frame0),
        "unswaps": swaps,
        "duplicate_boxes_removed": removed,
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
