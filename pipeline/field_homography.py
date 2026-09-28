#!/usr/bin/env python3
"""Stage 2 v2: pencils -> 1D projective yard indexing -> homography.

World frame: x = yards downfield (zero at an arbitrary detected yard line
until absolute anchoring), y = yards from far sideline toward camera.
"""
import sys
import itertools
import cv2
import numpy as np

PXY = 25.6     # composite field-map pixels per yard
HASH_GAP = 9.2  # yards between hash rows (measured from GRASS2 texture)


def hud_mask(h, w, frame=None):
    keep = np.ones((h, w), np.uint8) * 255
    def block(x0, y0, x1, y1):
        keep[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 0
    block(0.00, 0.00, 0.30, 0.18)   # score bug (always present)
    block(0.05, 0.86, 0.50, 1.00)   # TURBO banner
    block(0.55, 0.86, 1.00, 1.00)   # INSERT COINS banner
    block(0.80, 0.80, 1.00, 0.95)   # KAG watermark
    # restrict to the field: only pixels whose neighborhood is mostly
    # field-colored (teal). Generic rejection of HUD boxes, banners,
    # stadium walls without hardcoded regions.
    if frame is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        field = ((hsv[..., 0] > 60) & (hsv[..., 0] < 110) &
                 (hsv[..., 1] > 40)).astype(np.uint8) * 255
        field = cv2.medianBlur(field, 9)
        field = cv2.dilate(field, np.ones((31, 31), np.uint8))
        keep &= field
    return keep


def ridge_mask(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray, 31)
    ridge = cv2.subtract(gray, bg)
    mask = (ridge > 10).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask & hud_mask(h, w, frame)


def get_segments(mask, w):
    lines = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=55,
                            minLineLength=w // 14, maxLineGap=30)
    if lines is None:
        return np.zeros((0, 4))
    return np.asarray(lines).reshape(-1, 4).astype(float)


def fit_line_tls(pts):
    mean = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean)
    a, b = vt[-1]
    c = -(a * mean[0] + b * mean[1])
    return np.array([a, b, c])


def seg_len(s):
    return float(np.hypot(s[2] - s[0], s[3] - s[1]))


def merge_candidates(segs, px_tol=8, ang_tol=6):
    cands = []
    order = np.argsort([-seg_len(s) for s in segs])
    for idx in order:
        s = segs[idx]
        p1, p2 = np.array(s[:2]), np.array(s[2:])
        ang = np.degrees(np.arctan2(*(p2 - p1)[::-1])) % 180
        placed = False
        for c in cands:
            a, b, cc = c["abc"]
            if abs(a * p1[0] + b * p1[1] + cc) < px_tol and \
               abs(a * p2[0] + b * p2[1] + cc) < px_tol:
                dd = abs(ang - c["angle"]) % 180
                if min(dd, 180 - dd) < ang_tol:
                    c["pts"].extend([p1, p2])
                    c["support"] += seg_len(s)
                    c["abc"] = fit_line_tls(np.array(c["pts"]))
                    placed = True
                    break
        if not placed:
            cands.append(dict(pts=[p1, p2], support=seg_len(s),
                              abc=fit_line_tls(np.array([p1, p2])),
                              angle=ang))
    for c in cands:
        a, b, _ = c["abc"]
        c["angle"] = np.degrees(np.arctan2(-a, b)) % 180
        pts = np.array(c["pts"])
        t = pts @ np.array([b, -a])
        c["extent"] = float(t.max() - t.min())
    return [c for c in cands if c["support"] > 60]


def line_cross(l1, l2):
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-12:
        return None
    return p[:2] / p[2]


def pencil_members(cands, vp, tol_deg):
    members, support = [], 0.0
    for k, c in enumerate(cands):
        mid = np.array(c["pts"]).mean(axis=0)
        v = vp - mid
        ang_v = np.degrees(np.arctan2(v[1], v[0])) % 180
        d = abs(ang_v - c["angle"]) % 180
        if min(d, 180 - d) < tol_deg:
            members.append(k)
            support += c["support"]
    return members, support


def ransac_pencil(cands, tol_deg=1.2, min_lines=3, vp_filter=None):
    best = (0, None, [])
    for i, j in itertools.combinations(range(len(cands)), 2):
        vp = line_cross(cands[i]["abc"], cands[j]["abc"])
        if vp is None or (vp_filter and not vp_filter(vp)):
            continue
        members, support = pencil_members(cands, vp, tol_deg)
        if len(members) >= min_lines and support > best[0]:
            best = (support, vp, members)
    return best


def solve_1d_proj(xs, ts):
    """Fit t = (a x + b) / (c x + 1), least squares if >3 points."""
    A = np.array([[x, 1.0, -x * t] for x, t in zip(xs, ts)])
    try:
        if len(xs) == 3:
            sol = np.linalg.solve(A, np.array(ts))
        else:
            sol, *_ = np.linalg.lstsq(A, np.array(ts), rcond=None)
    except np.linalg.LinAlgError:
        return None
    return sol  # a, b, c


def proj_1d(sol, x):
    a, b, c = sol
    return (a * x + b) / (c * x + 1.0)


def assign_slots(sol, ts, ok, yard):
    """Assign each line to its nearest integer 5-yd slot under sol."""
    a1, b1, c1 = sol
    assigned = {}
    for i in ok:
        den = a1 - c1 * ts[i]
        if abs(den) < 1e-12:
            continue
        x = (ts[i] - b1) / den
        k = round(x / 5.0)
        if abs(k) > 24:
            continue
        t_here = proj_1d(sol, k * 5.0)
        t_next = proj_1d(sol, (k + 1) * 5.0)
        tol = max(0.14 * abs(t_next - t_here), 2.5)
        if abs(ts[i] - t_here) < tol:
            if k not in assigned or \
                    yard[i]["support"] > yard[assigned[k]]["support"]:
                assigned[k] = i
    return assigned


def index_yard_lines(yard, w, h):
    """Assign integer 5-yard indices via 1D projective RANSAC along a
    transversal. Returns [(cand, index_float)] for inliers."""
    mean_ang = np.median([c["angle"] for c in yard])
    # transversal through image center, perpendicular to family
    theta = np.radians(mean_ang + 90)
    p0 = np.array([w / 2, h / 2])
    d = np.array([np.cos(theta), np.sin(theta)])
    # param t of each line's crossing with transversal
    ts = []
    for c in yard:
        a, b, cc = c["abc"]
        denom = a * d[0] + b * d[1]
        if abs(denom) < 1e-9:
            ts.append(None)
            continue
        t = -(a * p0[0] + b * p0[1] + cc) / denom
        ts.append(t)
    ok = [i for i, t in enumerate(ts) if t is not None]
    best = (0, None, {}, None)
    for combo in itertools.combinations(ok, 3):
        t3 = [ts[i] for i in combo]
        if len(set(np.round(t3, 3))) < 3:
            continue
        order = np.argsort(t3)
        t3s = [t3[k] for k in order]
        for spac in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (0, 2, 4)):
            sol = solve_1d_proj([s * 5.0 for s in spac], t3s)
            if sol is None:
                continue
            # guided matching: assign -> refit -> reassign
            assigned = {}
            for _ in range(3):
                assigned = assign_slots(sol, ts, ok, yard)
                if len(assigned) < 4:
                    break
                xs_fit = [k * 5.0 for k in assigned]
                ts_fit = [ts[i] for i in assigned.values()]
                new_sol = solve_1d_proj(xs_fit, ts_fit)
                if new_sol is None:
                    break
                sol = new_sol
            if len(assigned) < 4:
                continue
            ks = sorted(assigned)
            gaps = np.diff(ks)
            if gaps.max() > 3 or (ks[-1] - ks[0]) > 30:
                continue
            sup = sum(yard[i]["support"] for i in assigned.values())
            # prefer dense assignments (solid lines every slot), then
            # more lines, then painted support
            score = (len(assigned), -float(np.mean(gaps)), sup)
            if best[1] is None or score > best[3]:
                best = (sup, sol, dict(assigned), score)
    if best[1] is None:
        return []
    ks = sorted(best[2])
    k0 = ks[0]
    return [(yard[best[2][k]], (k - k0) * 5.0) for k in ks]


def analyze(path, out_prefix):
    frame = cv2.imread(path)
    h, w = frame.shape[:2]
    mask = ridge_mask(frame)
    segs = get_segments(mask, w)
    cands = merge_candidates(segs)
    print(f"{len(segs)} segments -> {len(cands)} candidate lines")

    _, vp1, m1 = ransac_pencil(cands)
    yard_all = [cands[k] for k in m1]
    yard_ang = np.median([c["angle"] for c in yard_all])
    rest = []
    for k, c in enumerate(cands):
        if k in m1:
            continue
        d = abs(c["angle"] - yard_ang) % 180
        if min(d, 180 - d) >= 20:  # must be clearly non-parallel to yards
            rest.append(c)

    # downfield pencil: vp must be well above the field (or far outside)
    def vp_ok(vp):
        # ground-plane vanishing points must sit near/above the horizon
        return vp[1] < 0.20 * h
    _, vp2, m2 = ransac_pencil(rest, tol_deg=2.0, min_lines=2,
                               vp_filter=vp_ok)
    down = [rest[k] for k in m2] if vp2 is not None else []
    print(f"pencil1: {len(yard_all)} lines vp={np.round(vp1,1) if vp1 is not None else None}")
    print(f"pencil2: {len(down)} lines vp={np.round(vp2,1) if vp2 is not None else None}")

    # solid yard lines only
    yard = [c for c in yard_all
            if c["extent"] > w * 0.30 and c["support"] > w * 0.30]
    print(f"solid yard lines: {len(yard)}")
    indexed = index_yard_lines(yard, w, h)
    print(f"indexed yard lines: {[(round(x,1)) for _, x in indexed]}")

    down.sort(key=lambda c: -c["support"])
    dbg = frame.copy()
    for c, xw in indexed:
        draw_line(dbg, c["abc"], (0, 0, 255), 2)
        lbl_at(dbg, c["abc"], w // 2, f"{xw:.0f}", (0, 255, 255))
    for i, c in enumerate(down[:6]):
        draw_line(dbg, c["abc"], (255, 0, 255), 2)
        mid = np.array(c["pts"]).mean(axis=0)
        cv2.putText(dbg, f"D{i}", (int(mid[0]) + 8, int(mid[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        print(f"  D{i}: angle={c['angle']:.1f} support={c['support']:.0f} "
              f"mid=({mid[0]:.0f},{mid[1]:.0f})")
    cv2.imwrite(f"{out_prefix}_pencils.png", dbg)

    if len(indexed) < 2 or len(down) < 2:
        print("not enough structure for homography")
        return None

    # dedupe downfield lines: distinct crossings on a middle yard line
    ref_line = indexed[len(indexed) // 2][0]["abc"]
    distinct = []
    for c in down:  # sorted by support desc
        p = line_cross(c["abc"], ref_line)
        if p is None:
            continue
        if all(np.linalg.norm(p - d[1]) > 40 for d in distinct):
            distinct.append((c, p))
    if len(distinct) < 2:
        print("need two distinct downfield lines")
        return None

    # two strongest = the two hash-mark columns, HASH_GAP yards apart
    (h1, p1), (h2, p2) = distinct[0], distinct[1]
    src, dst = [], []
    for c, xw in indexed:
        for dl, yy in ((h1, 0.0), (h2, HASH_GAP)):
            p = line_cross(c["abc"], dl["abc"])
            if p is not None:
                src.append(p)
                dst.append([xw, yy])
    src, dst = np.array(src), np.array(dst)
    H, inl = cv2.findHomography(src.astype(np.float32),
                                dst.astype(np.float32), cv2.RANSAC, 1.0)
    H = H / H[2, 2]
    print(f"homography inliers: {int(inl.sum())}/{len(src)}")

    def world_y(Hm, u, v):
        p = Hm @ np.array([u, v, 1.0])
        return p[1] / p[2]

    if len(distinct) >= 3:
        # anchor y=0 at the sideline, field interior on positive side
        pts = np.array(distinct[2][0]["pts"])
        c0 = float(np.median([world_y(H, px, py) for px, py in pts[:20]]))
        interior = world_y(H, w / 2, h * 0.6)
        if interior < c0:
            H = np.diag([1.0, -1.0, 1.0]) @ H
            c0 = -c0
            print("flipped cross-field axis")
        H = np.array([[1, 0, 0], [0, 1, -c0], [0, 0, 1.0]]) @ H
        print(f"sideline anchored (was at y={c0:.1f})")
    else:
        # no sideline: orient y toward camera, center via hash midpoint
        if world_y(H, w / 2, h * 0.85) < world_y(H, w / 2, h * 0.3):
            H = np.diag([1.0, -1.0, 1.0]) @ H
            print("flipped cross-field axis")
        H = np.array([[1, 0, 0], [0, 1, HASH_GAP / 2 + 13.0],
                      [0, 0, 1.0]]) @ H
        print("no sideline; centered via hash midpoint (approx)")

    world_w, world_h = int(80 * PXY), int(50 * PXY)
    S = np.diag([PXY, PXY, 1.0])
    T = np.array([[1, 0, 10 * PXY], [0, 1, 3 * PXY], [0, 0, 1.0]])
    warp = cv2.warpPerspective(frame, T @ S @ H, (world_w, world_h))
    for i in range(-2, 15):
        x = int((5 * i + 10) * PXY)
        cv2.line(warp, (x, 0), (x, world_h), (0, 255, 255), 1)
    cv2.line(warp, (0, int(3 * PXY)), (world_w, int(3 * PXY)),
             (255, 0, 255), 1)
    cv2.line(warp, (0, int((HASH_GAP + 3) * PXY)), (world_w,
             int((HASH_GAP + 3) * PXY)), (255, 0, 255), 1)
    cv2.imwrite(f"{out_prefix}_rectified.png", warp)
    return H


def draw_line(img, abc, color, thick=2):
    a, b, c = abc
    h, w = img.shape[:2]
    pts = []
    for x in (0.0, w - 1.0):
        if abs(b) > 1e-9:
            pts.append((x, -(a * x + c) / b))
    if abs(a) > 1e-9:
        for y in (0.0, h - 1.0):
            pts.append((-(b * y + c) / a, y))
    pts = [(int(x), int(y)) for x, y in pts
           if -w <= x <= 2 * w and -h <= y <= 2 * h]
    if len(pts) >= 2:
        cv2.line(img, pts[0], pts[-1], color, thick)


def lbl_at(img, abc, x, text, color):
    a, b, c = abc
    if abs(b) > 1e-9:
        y = int(-(a * x + c) / b)
        cv2.putText(img, text, (x, max(25, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"== {p}")
        analyze(p, p.replace(".png", ""))
