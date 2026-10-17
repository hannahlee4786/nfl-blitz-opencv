#!/usr/bin/env python3
"""Score tracker boxes against labeled frames.

Ground truth
------------
A JSON object:

{
  "video": "<path>",
  "crop_x0": 0,
  "frames": {
    "<frame>": [
      {
        "person": "red40",
        "team": "red",
        "box": [x, y, w, h],
        "main": false,
        "hidden": false
      }
    ]
  }
}

Only frames listed under "frames" are scored. Each of those frames lists
every visible player whose feet are on screen. "person" is a stable name
for the same human across frames: team plus jersey number when the number
can be read, otherwise a tag such as "red_a".

Boxes are axis-aligned and in source-video pixels, the same coordinate
system as boxes.csv. "crop_x0" records the tracker's side-bar crop. It is
not applied again; both files are compared as stored.

"hidden": true means the player is fully hidden behind someone else. A
prediction matched to him is neither rewarded nor penalized, and missing
him is not a miss.

"main": true marks the main player (the quarterback the clip follows).

Predictions are rows of boxes.csv with columns
frame,id,team,x,y,w,h,conf. On each scored frame, predictions are matched
to same-team ground truth by an optimal assignment on IoU (Hungarian
algorithm implemented here). A pair is kept when its IoU is at least
--iou (default 0.3). The objective is the sum of those IoUs. Leaving a
box unmatched costs nothing, so one strong overlap is preferred to two
weaker overlaps whose IoUs add up to less.

Metrics
-------
coverage       matches / non-hidden ground truth
precision      matches / (predictions - predictions matched to a hidden player)
duplicates     for each visible person, how many extra same-team predictions
               overlap him at IoU >= --iou, summed over people. One
               prediction can count toward more than one person.
id switches    for each person, a change of matched track id between
               successive scored frames on which he is matched. A scored
               frame where he is missed is not itself a switch; the next
               matched frame is still compared with the previous match.
IDF1           2*IDTP / (2*IDTP + IDFP + IDFN). IDTP is the number of
               frame matches kept by the one-to-one person-to-track-id
               mapping that maximizes co-occurring matches across scored
               frames. Hidden ground truth, and predictions matched to a
               hidden player, are left out of IDTP, IDFP, and IDFN.
main accuracy  fraction of scored frames where the main person is listed,
               not hidden, and matched to the single track id that mapping
               gives him. Frames that do not list him, or list him hidden,
               are left out. None when the clip has no visible main.
               If several people are marked main, the fraction is pooled
               over all of their visible annotations.

Per scored frame the same counts are reported (switches are charged to the
later frame). The printed worst frames are those with the most misses plus
false boxes, with the missed person names and the false box coordinates.

--json-out writes the full result. --baseline compares that kind of JSON:
coverage, precision, IDF1, and main accuracy are worse when they fall by
more than --tolerance times the baseline value (a relative drop). Id
switches, false boxes, and misses are worse when they increase at all
(absolute tolerance 0; --tolerance does not loosen counts). Exit 1 if any
compared value is worse, otherwise exit 0.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np


HIGHER_IS_BETTER = ("coverage", "precision", "idf1", "main_accuracy")
LOWER_IS_BETTER = ("id_switches", "false_boxes", "misses")
BOX_COLUMNS = ("frame", "id", "team", "x", "y", "w", "h", "conf")


def box_iou(a, b):
    """Intersection over union of two [x, y, w, h] boxes. 0 if either is empty."""
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    inter_w = right - left
    inter_h = bottom - top
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def hungarian(cost):
    """Minimum-cost assignment.

    `cost` is a rectangular array. Returns min(rows, cols) pairs (row, col),
    one column per assigned row. This is the Jonker-Volgenant shortest
    augmenting path method (the dense Hungarian algorithm): each row is
    matched to a unique column, potentials keep the reduced costs nonnegative,
    and the result minimizes the sum of costs.
    """
    matrix = np.array(cost, dtype=np.float64, copy=True)
    if matrix.ndim != 2:
        raise ValueError("cost must be a 2-D array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return []
    transposed = False
    if matrix.shape[0] > matrix.shape[1]:
        matrix = matrix.T.copy()
        transposed = True
    row_count, col_count = matrix.shape
    # 1-indexed potentials and matching, matching the usual presentation.
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
                raise RuntimeError("Hungarian algorithm did not finish")
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


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _norm_gt(entry):
    if "box" not in entry or len(entry["box"]) != 4:
        raise ValueError(f"ground truth entry needs a box of 4 numbers: {entry!r}")
    if "person" not in entry or "team" not in entry:
        raise ValueError(f"ground truth entry needs person and team: {entry!r}")
    return {
        "person": str(entry["person"]),
        "team": str(entry["team"]).strip().lower(),
        "box": [float(v) for v in entry["box"]],
        "main": _as_bool(entry.get("main", False)),
        "hidden": _as_bool(entry.get("hidden", False)),
    }


def _norm_pred(entry):
    if "box" in entry:
        box = [float(v) for v in entry["box"]]
    else:
        box = [float(entry["x"]), float(entry["y"]), float(entry["w"]), float(entry["h"])]
    if len(box) != 4:
        raise ValueError(f"prediction needs a box of 4 numbers: {entry!r}")
    return {
        "id": int(entry["id"]),
        "team": str(entry["team"]).strip().lower(),
        "box": box,
        "conf": None if entry.get("conf") in (None, "") else float(entry["conf"]),
    }


def _rate(numerator, denominator):
    if denominator == 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def match_frame(ground_truth, predictions, iou_threshold):
    """Optimal same-team IoU assignment.

    Returns (gt_index, pred_index, iou) for pairs at or above the threshold.
    Ineligible pairs have cost 0, the same as leaving a box unmatched, so
    the solver maximizes the sum of accepted IoUs.
    """
    gt_count = len(ground_truth)
    pred_count = len(predictions)
    if gt_count == 0 or pred_count == 0:
        return []
    scores = np.zeros((gt_count, pred_count), dtype=np.float64)
    eligible = np.zeros((gt_count, pred_count), dtype=bool)
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):
            if gt["team"] != pred["team"]:
                continue
            score = box_iou(gt["box"], pred["box"])
            scores[gt_index, pred_index] = score
            if score >= iou_threshold:
                eligible[gt_index, pred_index] = True
    if not bool(eligible.any()):
        return []
    cost = np.zeros((gt_count, pred_count), dtype=np.float64)
    cost[eligible] = -scores[eligible]
    kept = []
    for gt_index, pred_index in hungarian(cost):
        if eligible[gt_index, pred_index]:
            kept.append((gt_index, pred_index, float(scores[gt_index, pred_index])))
    return kept


def _frame_dict(mapping):
    out = {}
    for key, value in mapping.items():
        out[int(key)] = value
    return out


def evaluate(labels, predictions, iou_threshold=0.3):
    """Score `labels` against `predictions` (frame -> list of boxes).

    Both arguments use the ground-truth / boxes.csv schemas documented above.
    Frame keys may be ints or strings. Returns a JSON-ready dict.
    """
    if not isinstance(labels, dict) or not isinstance(labels.get("frames"), dict):
        raise ValueError("labels must be an object with a frames object")
    gt_by_frame = _frame_dict(labels["frames"])
    pred_by_frame = _frame_dict(predictions)
    scored_frames = sorted(gt_by_frame)

    observations = {}
    cooccur = {}
    main_rows = []
    main_names = []
    seen_main = set()
    records = []

    for frame in scored_frames:
        ground_truth = [_norm_gt(entry) for entry in gt_by_frame[frame]]
        preds = [_norm_pred(entry) for entry in pred_by_frame.get(frame, [])]
        pairs = match_frame(ground_truth, preds, iou_threshold)
        matched_gt = {}
        matched_pred = set()
        for gt_index, pred_index, _score in pairs:
            matched_gt[gt_index] = pred_index
            matched_pred.add(pred_index)
            gt = ground_truth[gt_index]
            if gt["hidden"]:
                continue
            track_id = preds[pred_index]["id"]
            observations.setdefault(gt["person"], []).append((frame, track_id))
            key = (gt["person"], track_id)
            cooccur[key] = cooccur.get(key, 0) + 1

        visible_indexes = [i for i, gt in enumerate(ground_truth) if not gt["hidden"]]
        hidden_count = len(ground_truth) - len(visible_indexes)
        match_count = sum(1 for i in visible_indexes if i in matched_gt)
        missed = [ground_truth[i]["person"] for i in visible_indexes if i not in matched_gt]
        hidden_matched = sum(1 for i in matched_gt if ground_truth[i]["hidden"])
        false_coords = []
        for pred_index, pred in enumerate(preds):
            if pred_index in matched_pred:
                continue
            false_coords.append({
                "id": pred["id"],
                "team": pred["team"],
                "box": [float(v) for v in pred["box"]],
            })
        duplicates = 0
        for gt_index in visible_indexes:
            gt = ground_truth[gt_index]
            overlaps = 0
            for pred in preds:
                if pred["team"] != gt["team"]:
                    continue
                if box_iou(gt["box"], pred["box"]) >= iou_threshold:
                    overlaps += 1
            if overlaps > 1:
                duplicates += overlaps - 1
        for gt_index, gt in enumerate(ground_truth):
            if not gt["main"] or gt["person"] in seen_main:
                continue
            seen_main.add(gt["person"])
            main_names.append(gt["person"])
        for gt_index, gt in enumerate(ground_truth):
            if not gt["main"] or gt["hidden"]:
                continue
            matched_id = None
            if gt_index in matched_gt:
                matched_id = preds[matched_gt[gt_index]]["id"]
            main_rows.append((frame, gt["person"], matched_id))
        countable = len(preds) - hidden_matched
        records.append({
            "frame": frame,
            "gt": len(ground_truth),
            "visible": len(visible_indexes),
            "hidden": hidden_count,
            "predicted": len(preds),
            "matches": match_count,
            "misses": len(missed),
            "false_boxes": len(false_coords),
            "hidden_matched": hidden_matched,
            "duplicates": duplicates,
            "switches": 0,
            "coverage": _rate(match_count, len(visible_indexes)),
            "precision": _rate(match_count, countable),
            "missed_persons": missed,
            "false_box_coords": false_coords,
        })

    switch_details = []
    switch_at = {}
    for person, sequence in observations.items():
        by_frame = {}
        for frame, track_id in sequence:
            if frame not in by_frame:
                by_frame[frame] = track_id
        ordered = sorted(by_frame)
        for left, right in zip(ordered, ordered[1:]):
            if by_frame[left] == by_frame[right]:
                continue
            switch_at[right] = switch_at.get(right, 0) + 1
            switch_details.append({
                "person": person,
                "from_frame": left,
                "to_frame": right,
                "from_id": by_frame[left],
                "to_id": by_frame[right],
            })
    for record in records:
        record["switches"] = switch_at.get(record["frame"], 0)

    people = sorted({person for person, _track in cooccur})
    tracks = sorted({track for _person, track in cooccur})
    identity_map = {}
    idtp = 0
    if people and tracks:
        person_index = {person: index for index, person in enumerate(people)}
        track_index = {track: index for index, track in enumerate(tracks)}
        weight = np.zeros((len(people), len(tracks)), dtype=np.float64)
        for (person, track_id), count in cooccur.items():
            weight[person_index[person], track_index[track_id]] = count
        for person_i, track_i in hungarian(-weight):
            if weight[person_i, track_i] <= 0:
                continue
            idtp += int(round(weight[person_i, track_i]))
            identity_map[people[person_i]] = int(tracks[track_i])
    identity_map = {person: identity_map[person] for person in sorted(identity_map)}

    visible = sum(record["visible"] for record in records)
    hidden = sum(record["hidden"] for record in records)
    predicted = sum(record["predicted"] for record in records)
    matches = sum(record["matches"] for record in records)
    misses = sum(record["misses"] for record in records)
    false_boxes = sum(record["false_boxes"] for record in records)
    hidden_matched = sum(record["hidden_matched"] for record in records)
    duplicates = sum(record["duplicates"] for record in records)
    countable = predicted - hidden_matched
    idfn = visible - idtp
    idfp = countable - idtp
    idf1 = _rate(2 * idtp, 2 * idtp + idfp + idfn)

    main_correct = 0
    for _frame, person, matched_id in main_rows:
        mapped = identity_map.get(person)
        if mapped is not None and matched_id == mapped:
            main_correct += 1
    if len(main_names) == 1:
        main_person = main_names[0]
        main_track_id = identity_map.get(main_person)
    else:
        main_person = None
        main_track_id = None

    worst = [
        record for record in records
        if record["misses"] + record["false_boxes"] > 0
    ]
    worst.sort(key=lambda record: (-(record["misses"] + record["false_boxes"]), record["frame"]))

    return {
        "video": labels.get("video"),
        "crop_x0": labels.get("crop_x0", 0),
        "iou": iou_threshold,
        "frames_scored": len(records),
        "gt": visible + hidden,
        "visible": visible,
        "hidden": hidden,
        "predicted": predicted,
        "hidden_matched": hidden_matched,
        "countable_predictions": countable,
        "matches": matches,
        "misses": misses,
        "false_boxes": false_boxes,
        "duplicates": duplicates,
        "id_switches": len(switch_details),
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "idf1": idf1,
        "coverage": _rate(matches, visible),
        "precision": _rate(matches, countable),
        "main_person": main_person,
        "main_persons": main_names,
        "main_track_id": main_track_id,
        "main_frames": len(main_rows),
        "main_correct": main_correct,
        "main_accuracy": _rate(main_correct, len(main_rows)),
        "identity_map": identity_map,
        "switches": switch_details,
        "per_frame": records,
        "worst_frames": worst,
    }


def load_labels(path):
    with open(path) as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("frames"), dict):
        raise ValueError(f"{path} must be a JSON object with a frames object")
    return data


def load_boxes(path):
    """Read boxes.csv into frame -> list of {id, team, box, conf}."""
    by_frame = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = [name for name in BOX_COLUMNS if name not in fields]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        for row in reader:
            if row.get("frame") in (None, ""):
                continue
            frame = int(row["frame"])
            conf = row.get("conf")
            item = {
                "id": int(float(row["id"])),
                "team": str(row["team"]).strip().lower(),
                "box": [float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])],
                "conf": None if conf in (None, "") else float(conf),
            }
            by_frame.setdefault(frame, []).append(item)
    return by_frame


def regressions(current, baseline, tolerance):
    """Descriptions of metrics that got worse than `baseline`. Empty if none."""
    notes = []
    for key in HIGHER_IS_BETTER:
        if key not in baseline:
            notes.append(f"{key}: missing from baseline")
            continue
        base = baseline[key]
        cur = current.get(key)
        if base is None and cur is None:
            continue
        if base is None:
            continue
        if cur is None:
            notes.append(f"{key}: baseline {base} -> current n/a")
            continue
        base = float(base)
        cur = float(cur)
        drop = base - cur
        allowed = float(tolerance) * abs(base)
        if drop > allowed + 1e-9:
            relative = drop / abs(base) if base != 0 else float("inf")
            notes.append(
                f"{key}: baseline {base} -> current {cur} "
                f"(relative drop {relative:.4f} > tolerance {tolerance})"
            )
    for key in LOWER_IS_BETTER:
        if key not in baseline:
            notes.append(f"{key}: missing from baseline")
            continue
        base = int(baseline[key])
        cur = int(current[key])
        if cur > base:
            notes.append(f"{key}: baseline {base} -> current {cur} (count increased)")
    return notes


def _fmt_rate(value):
    if value is None:
        return "    n/a"
    return f"{value:7.3f}"


def _fmt_box(box):
    parts = []
    for value in box:
        number = float(value)
        if number == int(number):
            parts.append(str(int(number)))
        else:
            parts.append(f"{number:.2f}")
    return ", ".join(parts)


def format_report(result):
    """Readable per-clip summary, a per-frame table, and the worst frames."""
    lines = [
        f"video: {result.get('video') or '(unspecified)'}",
        f"iou threshold: {result['iou']}",
        f"frames scored: {result['frames_scored']}",
        "",
        f"coverage:      {_fmt_rate(result['coverage'])}  "
        f"({result['matches']} / {result['visible']} visible GT)",
        f"precision:     {_fmt_rate(result['precision'])}  "
        f"({result['matches']} / {result['countable_predictions']} countable predictions)",
        f"IDF1:          {_fmt_rate(result['idf1'])}  "
        f"(IDTP {result['idtp']}, IDFP {result['idfp']}, IDFN {result['idfn']})",
        f"id switches:   {result['id_switches']}",
        f"duplicates:    {result['duplicates']}",
        f"misses:        {result['misses']}",
        f"false boxes:   {result['false_boxes']}",
        f"main accuracy: {_fmt_rate(result['main_accuracy'])}  "
        f"(main={result['main_person'] or 'none'}, "
        f"track={result['main_track_id'] if result['main_track_id'] is not None else 'none'}, "
        f"{result['main_correct']}/{result['main_frames']} frames)",
        f"annotations:   {result['visible']} visible, {result['hidden']} hidden, "
        f"{result['predicted']} predictions",
    ]
    mapping = result["identity_map"]
    if mapping:
        pairs = ", ".join(f"{person}->{track}" for person, track in mapping.items())
        lines.append(f"identity map:  {pairs}")
    else:
        lines.append("identity map:  (none)")
    if result["switches"]:
        lines.append("id switch list:")
        shown = result["switches"][:20]
        for item in shown:
            lines.append(
                f"  {item['person']}: id {item['from_id']} on frame {item['from_frame']}"
                f" -> id {item['to_id']} on frame {item['to_frame']}"
            )
        extra = len(result["switches"]) - len(shown)
        if extra:
            lines.append(f"  ... {extra} more")
    lines.append("")
    lines.append(
        f"{'frame':>6} {'gt':>4} {'vis':>4} {'pred':>5} {'match':>5} "
        f"{'miss':>5} {'false':>5} {'dup':>4} {'sw':>3} {'cover':>7} {'prec':>7}"
    )
    for record in result["per_frame"]:
        lines.append(
            f"{record['frame']:6d} {record['gt']:4d} {record['visible']:4d} "
            f"{record['predicted']:5d} {record['matches']:5d} {record['misses']:5d} "
            f"{record['false_boxes']:5d} {record['duplicates']:4d} {record['switches']:3d} "
            f"{_fmt_rate(record['coverage'])} {_fmt_rate(record['precision'])}"
        )
    lines.append("")
    worst = result["worst_frames"]
    if not worst:
        lines.append("worst frames: none")
    else:
        lines.append("worst frames (misses + false boxes):")
        shown = worst[:12]
        for record in shown:
            lines.append(
                f"  frame {record['frame']}: misses={record['misses']} "
                f"false={record['false_boxes']}"
            )
            if record["missed_persons"]:
                lines.append("    missed: " + ", ".join(record["missed_persons"]))
            for false_box in record["false_box_coords"]:
                lines.append(
                    f"    false: id={false_box['id']} team={false_box['team']} "
                    f"box=({_fmt_box(false_box['box'])})"
                )
        extra = len(worst) - len(shown)
        if extra:
            lines.append(f"  ... {extra} more")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score tracker boxes.csv against a ground-truth JSON file."
    )
    parser.add_argument("--labels", required=True, help="Ground-truth JSON.")
    parser.add_argument("--boxes", required=True, help="Tracker boxes.csv.")
    parser.add_argument("--iou", type=float, default=0.3, help="Minimum IoU to accept a match.")
    parser.add_argument("--json-out", help="Write the full metric JSON here.")
    parser.add_argument("--baseline", help="Previous --json-out file to compare against.")
    parser.add_argument(
        "--tolerance", type=float, default=0.01,
        help="Allowed relative drop for coverage, precision, IDF1, and main accuracy.",
    )
    args = parser.parse_args(argv)
    if args.tolerance < 0:
        print("error: --tolerance must be >= 0", file=sys.stderr)
        return 2
    try:
        labels = load_labels(args.labels)
        predictions = load_boxes(args.boxes)
        result = evaluate(labels, predictions, args.iou)
    except (OSError, ValueError, json.JSONDecodeError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    crop = labels.get("crop_x0", 0)
    if crop not in (0, 0.0, None):
        print(
            f"note: crop_x0={crop} is recorded only; "
            "boxes are compared in the coordinates stored in each file"
        )
    print(format_report(result))
    if args.json_out:
        parent = os.path.dirname(os.path.abspath(args.json_out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.json_out, "w") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(f"wrote {args.json_out}")
    if not args.baseline:
        return 0
    try:
        with open(args.baseline) as handle:
            baseline = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not isinstance(baseline, dict):
        print("error: baseline must be a JSON object", file=sys.stderr)
        return 2
    notes = regressions(result, baseline, args.tolerance)
    if notes:
        print("baseline comparison: WORSE")
        for note in notes:
            print(f"  {note}")
        return 1
    print("baseline comparison: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
