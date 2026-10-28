"""Pin the tracked field fit to the painted yard lines.

The tracker follows corner points frame to frame, so its error accumulates as
the camera moves. The painted layout is known exactly (a yard line every 5
yards, in field coordinates), so on every frame the fit is nudged until the
projected yard lines sit on the white lines in the picture. Nothing here
accumulates: each frame is corrected against the same geometry.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares

from fieldtrack.field_model import FIELD_WIDTH

# Yard lines every 5 yards from the goal line (field x = 10) to the other one.
LINE_XS = np.arange(10.0, 110.1, 5.0)
SAMPLES_ALONG = np.linspace(1.5, FIELD_WIDTH - 1.5, 40)

def line_distance_map(frame, cap=60.0):
    """Distance in pixels from each pixel to the nearest painted-line pixel."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ridge = cv2.subtract(gray, cv2.medianBlur(gray, 31)) > 14
    white = (hsv[..., 1] < 90) & (hsv[..., 2] > 130)
    mask = (ridge & white).astype(np.uint8)
    # Drop the screen graphics: score bug, play banner, bottom bars.
    mask[: int(0.18 * h), : int(0.30 * w)] = 0
    # Bottom banners (they sit between 0.90h and 0.98h) and the KAG logo.
    mask[int(0.90 * h):, :] = 0
    mask[int(0.82 * h): int(0.93 * h), int(0.86 * w):] = 0
    dist = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    return np.minimum(dist, cap).astype(np.float32), int(mask.sum())


def _project(H, pts):
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32),
                                    H.astype(np.float32))[:, 0, :]


def _model_points(H, shape):
    """Points along every yard line, kept where they land inside the picture."""
    h, w = shape[:2]
    xs, ys = np.meshgrid(LINE_XS, SAMPLES_ALONG)
    field = np.stack([xs.ravel(), ys.ravel()], axis=1)
    screen = _project(H, field)
    ok = (np.isfinite(screen).all(axis=1) & (screen[:, 0] > 5) & (screen[:, 0] < w - 5)
          & (screen[:, 1] > 0.12 * h) & (screen[:, 1] < 0.89 * h))
    return field[ok], ok


# Correction applied to the field plane before projecting, downfield only:
# a shift of x, a stretch of x, and a tilt of the yard lines (x depends a
# little on y). Yard lines cannot see sideways motion, so the correction never
# touches y: lateral position stays exactly what the tracker measured. (A
# version that also moved y drifted sideways by about 2 yards over a clip.)
_PRIOR_SCALE = np.array([0.25, 0.08, 0.15], np.float64)
# Hard limits: a shift past half the line spacing (2.5 yd) would let the fit
# slide onto the neighbouring yard line, which looks just as good to the cost.
_BOUND = np.array([2.0, 0.06, 0.6], np.float64)


def _correction(p, pivot):
    """x' = (1 + p1) x + shear * y + p0, about pivot; y unchanged."""
    sx, shear = 1.0 + p[1], np.radians(p[2])
    lin = np.array([[sx, shear], [0.0, 1.0]], np.float64)
    M = np.eye(3)
    M[:2, :2] = lin
    M[:2, 2] = pivot - lin @ pivot + np.array([p[0], 0.0])
    return M


def refine(H, frame, dist_map=None, weight=6.0):
    """Return (H', median residual px, line points used) for one frame.

    H maps field -> screen. The fit is corrected so projected yard lines land
    on the painted ones, with a prior that keeps the correction small.
    """
    if dist_map is None:
        dist_map, _ = line_distance_map(frame)
    field, ok = _model_points(H, frame.shape)
    if len(field) < 60:
        return H, float("nan"), len(field)
    pts = field.astype(np.float64)
    h, w = frame.shape[:2]
    c = np.linalg.inv(H) @ np.array([w / 2.0, 0.65 * h, 1.0])
    pivot = c[:2] / c[2]

    def build(p):
        return H @ _correction(p, pivot)

    def residuals_on(m):
        def residuals(p, cap):
            s = _project(build(p), pts)
            # A degenerate correction can throw points to infinity; treat them as off the lines.
            s = np.nan_to_num(s, nan=-1e4, posinf=1e4, neginf=-1e4).clip(-1e4, 1e4)
            d = cv2.remap(m, s[:, 0:1].astype(np.float32), s[:, 1:2].astype(np.float32),
                          cv2.INTER_LINEAR, borderValue=cap)[:, 0]
            # Keep the residual float32; mixing in float64 hid the gradient.
            return np.concatenate([np.minimum(d, cap),
                                   (weight * p / _PRIOR_SCALE).astype(np.float32)])
        return residuals

    p = np.zeros(3)
    # A bound-pinned parameter can give a zero finite-difference step; retry
    # with a different step before giving up on the frame.
    for step in (0.02, 0.05, 0.1):
        try:
            return _solve(residuals_on, dist_map, p, pts, build, step)
        except (ValueError, np.linalg.LinAlgError, cv2.error):
            continue
    return H, float("nan"), len(pts)


def _solve(residuals_on, dist_map, p, pts, build, step):
    for sigma, cap in ((8.0, 45.0), (3.0, 25.0), (0.0, 14.0)):
        m = cv2.GaussianBlur(dist_map, (0, 0), sigma) if sigma else dist_map
        res = least_squares(residuals_on(m), p, args=(cap,), loss="soft_l1", f_scale=3.0,
                            x_scale=_PRIOR_SCALE, diff_step=step, max_nfev=40,
                            bounds=(-_BOUND, _BOUND))
        p = res.x
    final = residuals_on(dist_map)(p, 25.0)[:len(pts)]
    return build(p), float(np.median(final)), len(pts)
