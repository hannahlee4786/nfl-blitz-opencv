"""Temporal smoothing for the field fit and for player positions.

The field fit is refit from scratch on every frame and each player's foot is
the bottom edge of a detector box, so both wobble a little. Neither moves
fast, and the clip is processed offline, so a short centered filter removes the
wobble without lag.
"""
import cv2
import numpy as np
from scipy.signal import savgol_filter

# Image points used to summarize a homography. They sit on the field in the
# lower part of the picture, away from the horizon where projections blow up.
_ANCHORS = ((0.15, 0.45), (0.85, 0.45), (0.85, 0.92), (0.15, 0.92))


def _runs(valid):
    """Index ranges of consecutive True values."""
    start = None
    for i, ok in enumerate(list(valid) + [False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            yield start, i
            start = None


def smooth_series(values, window, order=2):
    """Savitzky-Golay along axis 0; short series are returned unchanged."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    window = min(window, n if n % 2 else n - 1)
    if window < order + 2:
        return values
    return savgol_filter(values, window, order, axis=0, mode="interp")


def smooth_homographies(A, shape, window=9):
    """Smooth image->world matrices by smoothing where fixed image points land."""
    h, w = shape[:2]
    src = np.array([[u * w, v * h] for u, v in _ANCHORS], np.float32)
    world = np.array([
        cv2.perspectiveTransform(src.reshape(-1, 1, 2), np.asarray(m, np.float32))[:, 0, :]
        for m in A
    ], np.float64)                                   # (frames, 4, 2)
    ok = np.isfinite(world).all(axis=(1, 2))
    smooth = world.copy()
    for a, b in _runs(ok):
        smooth[a:b] = smooth_series(world[a:b].reshape(b - a, -1), window).reshape(b - a, 4, 2)
    out = []
    for m, pts, good in zip(A, smooth, ok):
        if not good:
            out.append(m)
            continue
        out.append(cv2.getPerspectiveTransform(src, pts.astype(np.float32)).astype(np.float64))
    return out


def smooth_tracks(points, window=5, max_gap=1):
    """Smooth positions per track.

    points maps id -> {frame: (x, y)}. Runs break where more than max_gap
    frames are missing, so a player is never smoothed across a long absence.
    Returns the same structure.
    """
    out = {}
    for tid, by_frame in points.items():
        frames = sorted(by_frame)
        res = {}
        run = [frames[0]]
        runs = []
        for f in frames[1:]:
            if f - run[-1] <= max_gap + 1:
                run.append(f)
            else:
                runs.append(run)
                run = [f]
        runs.append(run)
        for run in runs:
            # Fill any one-frame holes by interpolation before filtering.
            full = np.arange(run[0], run[-1] + 1)
            xy = np.array([by_frame[f] for f in run], np.float64)
            fx = np.interp(full, run, xy[:, 0])
            fy = np.interp(full, run, xy[:, 1])
            sm = smooth_series(np.stack([fx, fy], axis=1), window)
            for f in run:
                res[f] = (float(sm[f - run[0], 0]), float(sm[f - run[0], 1]))
        out[tid] = res
    return out
