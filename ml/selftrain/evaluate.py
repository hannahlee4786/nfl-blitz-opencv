#!/usr/bin/env python3
"""Select a student checkpoint on Clip 4 and score Clip 4 and Clip 64.

Detection threshold and tracking low/high are chosen on Clip 4 only.
The scorer is tools/eval_tracking.py, run with the same Python that launched this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import cv2
import numpy as np

from ml.detect import clip_xyxy, labeled_frame_ids, read_frames
from ml.track_assoc import DEFAULTS, run as assoc_run
from tools.eval_tracking import box_iou, hungarian, load_boxes, load_labels
from ml.selftrain.cache_video import cache_one, load_model
from ml.selftrain.common import ART, REPO, SCRATCH, TEST, ensure_dirs

COARSE = [0.08, 0.12, 0.18, 0.25, 0.35, 0.45, 0.60]
BOX_FIELDS = ["frame", "id", "team", "x", "y", "w", "h", "conf"]
BASELINES = {
    "detection": {
        "gdino_0.30": {
            "clip4": {"coverage": 0.736, "precision": 0.809},
            "clip64": {"coverage": 0.749, "precision": 0.759},
        },
        "yolo11m_0.10": {
            "clip4": {"coverage": 0.702, "precision": 0.723},
            "clip64": {"coverage": 0.753, "precision": 0.686},
        },
    },
    "tracking": {
        "yolo11m_low0.07_high0.18": {
            "clip4": {"idf1": 0.512, "coverage": 0.727, "precision": 0.775},
            "clip64": {"idf1": 0.408, "coverage": 0.732, "precision": 0.771},
        },
        "gdino_low0.18_high0.35": {
            "clip4": {"idf1": 0.550, "coverage": 0.769, "precision": 0.769},
            "clip64": {"idf1": 0.352, "coverage": 0.749, "precision": 0.677},
        },
    },
}


def better_det(candidate: dict, incumbent: dict | None) -> bool:
    if incumbent is None:
        return True
    left = (candidate["f1"] or 0, candidate["coverage"] or 0, candidate["precision"] or 0)
    right = (incumbent["f1"] or 0, incumbent["coverage"] or 0, incumbent["precision"] or 0)
    return left > right


def f1_score(coverage, precision) -> float:
    if coverage is None or precision is None:
        return 0.0
    total = float(coverage) + float(precision)
    if total <= 0:
        return 0.0
    return 2.0 * float(coverage) * float(precision) / total


def write_boxes(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BOX_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in BOX_FIELDS})


def score_csv(labels: Path, boxes: Path, json_out: Path) -> dict:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/eval_tracking.py"),
            "--labels",
            str(labels),
            "--boxes",
            str(boxes),
            "--json-out",
            str(json_out),
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    json_out.with_suffix(".log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"scorer exit {proc.returncode} on {boxes.name}: {proc.stderr[-500:] or proc.stdout[-500:]}"
        )
    return json.loads(json_out.read_text())


def brief(result: dict) -> dict:
    return {
        "coverage": result["coverage"],
        "precision": result["precision"],
        "idf1": result["idf1"],
        "id_switches": result["id_switches"],
        "duplicates": result["duplicates"],
        "misses": result["misses"],
        "false_boxes": result["false_boxes"],
        "main_accuracy": result["main_accuracy"],
        "matches": result["matches"],
        "visible": result["visible"],
        "predicted": result["predicted"],
        "f1": round(f1_score(result["coverage"], result["precision"]), 4),
    }


def team_error_rate(labels_path: Path, boxes_path: Path, iou_threshold: float = 0.3) -> dict:
    """Fraction of IoU>=threshold matches, ignoring team, whose team disagrees."""
    labels = load_labels(str(labels_path))
    preds = load_boxes(str(boxes_path))
    spatial = 0
    wrong = 0
    for key, entries in labels["frames"].items():
        frame = int(key)
        ground = []
        for entry in entries:
            person = {
                "team": str(entry["team"]).strip().lower(),
                "box": [float(v) for v in entry["box"]],
                "hidden": bool(entry.get("hidden", False)),
            }
            if not person["hidden"]:
                ground.append(person)
        predictions = preds.get(frame, [])
        if not ground or not predictions:
            continue
        scores = np.zeros((len(ground), len(predictions)), dtype=np.float64)
        eligible = np.zeros((len(ground), len(predictions)), dtype=bool)
        for gt_index, gt in enumerate(ground):
            for pred_index, pred in enumerate(predictions):
                iou = box_iou(gt["box"], pred["box"])
                scores[gt_index, pred_index] = iou
                if iou >= iou_threshold:
                    eligible[gt_index, pred_index] = True
        if not bool(eligible.any()):
            continue
        cost = np.zeros_like(scores)
        cost[eligible] = -scores[eligible]
        for gt_index, pred_index in hungarian(cost):
            if not eligible[gt_index, pred_index]:
                continue
            spatial += 1
            if ground[gt_index]["team"] != predictions[pred_index]["team"]:
                wrong += 1
    rate = None if spatial == 0 else round(wrong / spatial, 4)
    return {"spatial_matches": spatial, "team_errors": wrong, "team_error_rate": rate}


def predict_labeled(model, video: Path, labels: Path, imgsz: int, conf: float) -> tuple[list[dict], dict]:
    frame_ids = labeled_frame_ids(labels)
    frames, _decode = read_frames(video, frame_ids)
    warmup = frames[frame_ids[0]]
    model.predict(
        warmup,
        imgsz=imgsz,
        conf=conf,
        iou=0.60,
        agnostic_nms=True,
        device="cpu",
        verbose=False,
        max_det=80,
    )
    rows = []
    times = []
    next_id = 1
    for frame_id in frame_ids:
        image = frames[frame_id]
        started = time.perf_counter()
        result = model.predict(
            image,
            imgsz=imgsz,
            conf=conf,
            iou=0.60,
            agnostic_nms=True,
            device="cpu",
            verbose=False,
            max_det=80,
        )[0]
        times.append(time.perf_counter() - started)
        height, width = image.shape[:2]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy()
        for box, score, cls_id in zip(xyxy, confs, clss):
            if int(cls_id) not in (0, 1):
                continue
            clipped = clip_xyxy(box, width, height)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            rows.append(
                {
                    "frame": frame_id,
                    "id": next_id,
                    "team": "red" if int(cls_id) == 0 else "blue",
                    "x": round(x1, 2),
                    "y": round(y1, 2),
                    "w": round(x2 - x1, 2),
                    "h": round(y2 - y1, 2),
                    "conf": round(float(score), 6),
                }
            )
            next_id += 1
    steady = times[1:] if len(times) > 1 else times
    timing = {
        "frames": len(frame_ids),
        "sec_per_frame": round(float(np.mean(times)), 4) if times else None,
        "sec_per_frame_steady": round(float(np.mean(steady)), 4) if steady else None,
    }
    return rows, timing


def fine_thresholds(center: float) -> list[float]:
    values = []
    cursor = center - 0.06
    while cursor <= center + 0.06 + 1e-9:
        if 0.02 <= cursor <= 0.85:
            values.append(round(cursor, 3))
        cursor += 0.02
    return values


def eval_rows(rows: list[dict], threshold: float, labels: Path, csv_path: Path, json_path: Path) -> dict:
    kept = [row for row in rows if row["conf"] + 1e-9 >= threshold]
    write_boxes(csv_path, kept)
    scored = brief(score_csv(labels, csv_path, json_path))
    scored.update(team_error_rate(labels, csv_path))
    scored["threshold"] = threshold
    scored["boxes"] = len(kept)
    return scored


def best_threshold(rows, labels, clip: str, tag: str, thresholds: list[float]) -> tuple[dict, list[dict]]:
    sweep = []
    best = None
    for threshold in thresholds:
        csv_path = ART / "reports" / "sweep" / f"{tag}_{clip}_t{threshold:.3f}.csv"
        json_path = csv_path.with_suffix(".json")
        scored = eval_rows(rows, threshold, labels, csv_path, json_path)
        sweep.append(scored)
        print(
            f"  {clip} {tag} t={threshold:.3f} F1 {scored['f1']:.3f} "
            f"cov {scored['coverage']} prec {scored['precision']}",
            flush=True,
        )
        if better_det(scored, best):
            best = scored
    return best, sweep


def weight_files(round_n: int, only: list[str] | None) -> list[Path]:
    directory = ART / "runs" / f"round{round_n}" / "weights"
    if only:
        paths = []
        for name in only:
            path = Path(name)
            if not path.is_file():
                path = directory / name
            if not path.is_file():
                raise RuntimeError(f"missing weights {name}")
            paths.append(path.resolve())
        # Identical copies (best.pt == last.pt == epoch6.pt) are scored once.
        unique = []
        seen = set()
        for path in paths:
            key = (path.stat().st_size, path.stat().st_mtime_ns)
            if key in seen:
                print(f"skip duplicate checkpoint {path.name}", flush=True)
                continue
            seen.add(key)
            unique.append(path)
        return unique
    if not directory.is_dir():
        raise RuntimeError(f"no weights directory {directory}")
    paths = sorted(path for path in directory.glob("*.pt") if path.name != "selected.pt" and path.stat().st_size > 1000)
    if not paths:
        raise RuntimeError(f"no checkpoints in {directory}")
    return paths


def report_name(round_n: int, tag: str, kind: str) -> Path:
    suffix = f"_{tag}" if tag else ""
    return ART / "reports" / f"round{round_n}{suffix}_{kind}"


def stage_select(round_n: int, reselect: bool, only: list[str] | None, tag: str) -> dict:
    dest = report_name(round_n, tag, "selection.json")
    if dest.is_file() and not reselect:
        print(f"using existing selection {dest}", flush=True)
        return json.loads(dest.read_text())
    labels4 = TEST["clip4"]["labels"]
    video4 = TEST["clip4"]["video"]
    candidates = []
    for weights in weight_files(round_n, only):
        print(f"checkpoint {weights.name}", flush=True)
        model, resolved = load_model(weights)
        rows, timing = predict_labeled(model, video4, labels4, 640, 0.05)
        best, sweep = best_threshold(rows, labels4, "clip4", f"r{round_n}_{weights.stem}_640", COARSE)
        fine = [value for value in fine_thresholds(best["threshold"]) if value not in COARSE]
        if fine:
            best_fine, sweep_fine = best_threshold(rows, labels4, "clip4", f"r{round_n}_{weights.stem}_640", fine)
            sweep.extend(sweep_fine)
            if better_det(best_fine, best):
                best = best_fine
        candidates.append(
            {
                "weights": str(resolved),
                "name": weights.name,
                "imgsz": 640,
                "timing": timing,
                "best": best,
                "sweep": sweep,
            }
        )
        del model
    candidates.sort(key=lambda item: (item["best"]["f1"], item["best"]["coverage"], item["best"]["precision"]), reverse=True)
    winner = candidates[0]
    print(f"best 640 checkpoint {winner['name']} F1 {winner['best']['f1']}", flush=True)
    model, resolved = load_model(Path(winner["weights"]))
    rows768, timing768 = predict_labeled(model, video4, labels4, 768, 0.05)
    best768, sweep768 = best_threshold(rows768, labels4, "clip4", f"r{round_n}_{Path(winner['weights']).stem}_768", COARSE)
    fine768 = [value for value in fine_thresholds(best768["threshold"]) if value not in COARSE]
    if fine768:
        best_fine, sweep_fine = best_threshold(
            rows768, labels4, "clip4", f"r{round_n}_{Path(winner['weights']).stem}_768", fine768
        )
        sweep768.extend(sweep_fine)
        if better_det(best_fine, best768):
            best768 = best_fine
    use_768 = best768["f1"] >= winner["best"]["f1"] + 0.005
    chosen_imgsz = 768 if use_768 else 640
    chosen_best = best768 if use_768 else winner["best"]
    chosen_timing = timing768 if use_768 else winner["timing"]
    selected_name = "selected.pt" if not tag else f"selected_{tag}.pt"
    selected = ART / "runs" / f"round{round_n}" / "weights" / selected_name
    shutil.copy2(winner["weights"], selected)

    print(f"scoring clip64 at imgsz {chosen_imgsz} t {chosen_best['threshold']}", flush=True)
    rows64, timing64 = predict_labeled(
        model, TEST["clip64"]["video"], TEST["clip64"]["labels"], chosen_imgsz, 0.05
    )
    clip64 = eval_rows(
        rows64,
        chosen_best["threshold"],
        TEST["clip64"]["labels"],
        report_name(round_n, tag, "det_clip64.csv"),
        report_name(round_n, tag, "det_clip64.json"),
    )
    # Re-score the chosen Clip 4 operating point to a stable path.
    rows4, timing4 = predict_labeled(model, video4, labels4, chosen_imgsz, 0.05)
    clip4 = eval_rows(
        rows4,
        chosen_best["threshold"],
        labels4,
        report_name(round_n, tag, "det_clip4.csv"),
        report_name(round_n, tag, "det_clip4.json"),
    )
    report = {
        "round": round_n,
        "selection_rule": "max Clip 4 detection F1; imgsz 768 kept only if it beats imgsz 640 by >= 0.005 F1",
        "weights": str(selected),
        "source_weights": winner["weights"],
        "source_name": winner["name"],
        "imgsz": chosen_imgsz,
        "threshold": chosen_best["threshold"],
        "clip4": clip4,
        "clip64": clip64,
        "clip4_sec_per_frame": timing4["sec_per_frame_steady"],
        "clip64_sec_per_frame": timing64["sec_per_frame_steady"],
        "imgsz768_best_f1": best768["f1"],
        "imgsz640_best_f1": winner["best"]["f1"],
        "checkpoints": [
            {"name": item["name"], "f1": item["best"]["f1"], "threshold": item["best"]["threshold"], "coverage": item["best"]["coverage"], "precision": item["best"]["precision"]}
            for item in candidates
        ],
    }
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"clip4": clip4, "clip64": clip64, "imgsz": chosen_imgsz}, indent=2), flush=True)
    print(f"wrote {dest}", flush=True)
    return report


def stage_track(round_n: int, selection: dict, tag: str) -> dict:
    weights = Path(selection["weights"])
    imgsz = int(selection["imgsz"])
    model, resolved = load_model(weights)
    stems = {}
    for name, spec in TEST.items():
        stem = ART / "student_cache" / f"round{round_n}{'_' + tag if tag else ''}_imgsz{imgsz}" / name
        print(f"caching {name} for tracking", flush=True)
        meta = cache_one(
            spec["video"],
            stem,
            "student",
            weights,
            imgsz,
            0.05,
            batch=4,
            model=model,
            resolved_weights=resolved,
        )
        stems[name] = {"stem": stem, "meta": meta}
    lows = [0.05, 0.10, 0.15, 0.25, 0.35]
    highs = [0.15, 0.25, 0.35, 0.50, 0.65]
    sweep = []
    best = None
    for low in lows:
        for high in highs:
            if high < low + 0.04:
                continue
            params = dict(DEFAULTS)
            params["low"] = low
            params["high"] = high
            out = ART / "tracks_test" / f"round{round_n}{'_' + tag if tag else ''}" / f"clip4_l{low:.2f}_h{high:.2f}.csv"
            assoc_run(stems["clip4"]["stem"], out, params)
            json_out = out.with_suffix(".score.json")
            scored = brief(score_csv(TEST["clip4"]["labels"], out, json_out))
            scored.update(team_error_rate(TEST["clip4"]["labels"], out))
            scored["low"] = low
            scored["high"] = high
            sweep.append(scored)
            print(
                f"  track low {low:.2f} high {high:.2f} IDF1 {scored['idf1']} "
                f"cov {scored['coverage']} prec {scored['precision']} "
                f"false {scored['false_boxes']} dup {scored['duplicates']}",
                flush=True,
            )
            key = (scored["idf1"] or 0, scored["coverage"] or 0, -(scored["false_boxes"] or 0), -(scored["duplicates"] or 0))
            if best is None or key > best[0]:
                best = (key, scored, params, out)
    assert best is not None
    _key, chosen, params, clip4_csv = best
    clip64_csv = ART / "tracks_test" / f"round{round_n}{'_' + tag if tag else ''}" / "clip64_chosen.csv"
    assoc_run(stems["clip64"]["stem"], clip64_csv, params)
    clip64 = brief(score_csv(TEST["clip64"]["labels"], clip64_csv, clip64_csv.with_suffix(".score.json")))
    clip64.update(team_error_rate(TEST["clip64"]["labels"], clip64_csv))
    # Stable copies of the chosen Clip 4 boxes (the sweep file is already the score source).
    report = {
        "round": round_n,
        "selection_rule": "max Clip 4 IDF1, then coverage, then fewer false boxes, then fewer duplicates",
        "low": params["low"],
        "high": params["high"],
        "other_params": {key: value for key, value in params.items() if key not in ("low", "high")},
        "clip4": chosen,
        "clip64": clip64,
        "clip4_csv": str(clip4_csv),
        "clip64_csv": str(clip64_csv),
        "sec_per_frame": {
            "clip4": stems["clip4"]["meta"].get("sec_per_frame_steady"),
            "clip64": stems["clip64"]["meta"].get("sec_per_frame_steady"),
        },
        "sweep_clip4": sweep,
    }
    dest = report_name(round_n, tag, "tracking.json")
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"clip4": chosen, "clip64": clip64, "low": params["low"], "high": params["high"]}, indent=2), flush=True)
    print(f"wrote {dest}", flush=True)
    return report


def grab(video: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    found = {}
    index = 0
    if not wanted:
        cap.release()
        return found
    last = max(wanted)
    while index <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            found[index] = frame
        index += 1
    cap.release()
    return found


def render_worst(round_n: int, kind: str, clip: str, boxes_csv: Path, score_json: Path, limit: int = 4) -> list[str]:
    result = json.loads(score_json.read_text())
    worst = result.get("worst_frames") or []
    worst = worst[:limit]
    if not worst:
        return []
    labels = load_labels(str(TEST[clip]["labels"]))
    preds = load_boxes(str(boxes_csv))
    frames = {int(record["frame"]) for record in worst}
    images = grab(TEST[clip]["video"], frames)
    out_dir = SCRATCH / "failures" / f"round{round_n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    gt_by_frame = {int(key): value for key, value in labels["frames"].items()}
    for record in worst:
        frame_id = int(record["frame"])
        image = images.get(frame_id)
        if image is None:
            continue
        vis = image.copy()
        for entry in gt_by_frame.get(frame_id, []):
            if entry.get("hidden"):
                continue
            x, y, w, h = (int(round(float(v))) for v in entry["box"])
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 0), 2)
            cv2.putText(
                vis,
                str(entry.get("person", "")),
                (x, max(14, y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 220, 0),
                1,
                cv2.LINE_AA,
            )
        for pred in preds.get(frame_id, []):
            x, y, w, h = (int(round(v)) for v in pred["box"])
            color = (0, 0, 255) if pred["team"] == "red" else (255, 40, 0)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            vis,
            f"{kind} {clip} f{frame_id} miss {record['misses']} false {record['false_boxes']}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        path = out_dir / f"{kind}_{clip}_f{frame_id:04d}.jpg"
        cv2.imwrite(str(path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        saved.append(str(path))
    return saved


REVIEW_FRAMES = {"clip4": [0, 60, 120, 240], "clip64": [0, 120, 240]}


def write_student_overlays(selection: dict, tag: str) -> list[str]:
    """Boxes at the Clip-4-chosen threshold on a fixed frame list."""
    model, _resolved = load_model(Path(selection["weights"]))
    imgsz = int(selection["imgsz"])
    threshold = float(selection["threshold"])
    out_dir = SCRATCH / "student_review" / (tag or "selected")
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for clip, frames in REVIEW_FRAMES.items():
        wanted = set(frames)
        images = grab(TEST[clip]["video"], wanted)
        for frame_id in frames:
            image = images.get(frame_id)
            if image is None:
                print(f"overlay missing {clip} frame {frame_id}", flush=True)
                continue
            result = model.predict(
                image, imgsz=imgsz, conf=threshold, iou=0.60, agnostic_nms=True, device="cpu", verbose=False, max_det=80
            )[0]
            vis = image.copy()
            height, width = image.shape[:2]
            n = 0
            if result.boxes is not None:
                for box, score, cls_id in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.cls.cpu().numpy(),
                ):
                    if int(cls_id) not in (0, 1):
                        continue
                    clipped = clip_xyxy(box, width, height)
                    if clipped is None:
                        continue
                    x1, y1, x2, y2 = (int(round(v)) for v in clipped)
                    color = (0, 0, 255) if int(cls_id) == 0 else (255, 40, 0)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    n += 1
            cv2.putText(
                vis,
                f"{tag or 'selected'} {clip} f{frame_id} t={threshold:.2f} n={n}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            path = out_dir / f"{clip}_f{frame_id:04d}.jpg"
            cv2.imwrite(str(path), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            saved.append(str(path))
            print(f"overlay {path} boxes {n}", flush=True)
    del model
    return saved


def stage_report(round_n: int, selection: dict, tracking: dict, tag: str) -> dict:
    det4 = selection["clip4"]
    det64 = selection["clip64"]
    tr4 = tracking["clip4"]
    tr64 = tracking["clip64"]
    failures = []
    failures += render_worst(round_n, "det", "clip4", report_name(round_n, tag, "det_clip4.csv"), report_name(round_n, tag, "det_clip4.json"))
    failures += render_worst(round_n, "det", "clip64", report_name(round_n, tag, "det_clip64.csv"), report_name(round_n, tag, "det_clip64.json"))
    failures += render_worst(round_n, "track", "clip4", Path(tracking["clip4_csv"]), Path(tracking["clip4_csv"]).with_suffix(".score.json"))
    failures += render_worst(round_n, "track", "clip64", Path(tracking["clip64_csv"]), Path(tracking["clip64_csv"]).with_suffix(".score.json"))
    report = {
        "round": round_n,
        "baselines": BASELINES,
        "targets": {
            "det_clip4_coverage_gt": 0.736,
            "det_clip4_precision_gt": 0.809,
            "det_clip64_coverage_gt": 0.749,
            "det_clip64_precision_gt": 0.759,
            "track_clip4_idf1_gt": 0.550,
            "track_clip64_idf1_gt": 0.408,
            "false_boxes_le": [62, 151],
            "duplicates_le": [11, 13],
        },
        "detection": {
            "weights": selection["weights"],
            "imgsz": selection["imgsz"],
            "threshold": selection["threshold"],
            "sec_per_frame": {"clip4": selection["clip4_sec_per_frame"], "clip64": selection["clip64_sec_per_frame"]},
            "clip4": det4,
            "clip64": det64,
            "beats_gdino_coverage_and_precision": bool(
                det4["coverage"] > 0.736
                and det4["precision"] > 0.809
                and det64["coverage"] > 0.749
                and det64["precision"] > 0.759
            ),
        },
        "tracking": {
            "low": tracking["low"],
            "high": tracking["high"],
            "sec_per_frame": tracking["sec_per_frame"],
            "clip4": tr4,
            "clip64": tr64,
            "beats_idf1": bool((tr4["idf1"] or 0) > 0.550 and (tr64["idf1"] or 0) > 0.408),
            "meets_false_and_dup_caps": bool(
                tr4["false_boxes"] <= 62
                and tr64["false_boxes"] <= 151
                and tr4["duplicates"] <= 11
                and tr64["duplicates"] <= 13
            ),
        },
        "failure_images": failures,
    }
    overlays = write_student_overlays(selection, tag or f"round{round_n}")
    report["student_overlays"] = overlays
    dest = report_name(round_n, tag, "summary.json")
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {dest}", flush=True)
    print(json.dumps({"detection_beats": report["detection"]["beats_gdino_coverage_and_precision"], "tracking_beats_idf1": report["tracking"]["beats_idf1"], "caps": report["tracking"]["meets_false_and_dup_caps"]}, indent=2), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select and evaluate the student.")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--stage", choices=["select", "track", "report", "all"], default="all")
    parser.add_argument("--reselect", action="store_true")
    parser.add_argument("--weights", nargs="*", default=None, help="Score only these checkpoints (paths or filenames).")
    parser.add_argument("--tag", default="", help="Suffix for report filenames, e.g. epoch7 or final.")
    args = parser.parse_args(argv)
    ensure_dirs()
    selection = None
    tracking = None
    if args.stage in ("select", "all"):
        selection = stage_select(args.round, args.reselect, args.weights, args.tag)
    else:
        path = report_name(args.round, args.tag, "selection.json")
        selection = json.loads(path.read_text())
    if args.stage in ("track", "all"):
        tracking = stage_track(args.round, selection, args.tag)
    elif args.stage == "report":
        tracking = json.loads(report_name(args.round, args.tag, "tracking.json").read_text())
    if args.stage in ("report", "all"):
        stage_report(args.round, selection, tracking, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
