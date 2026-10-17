#!/usr/bin/env python3
"""Checks for tools/eval_tracking.py. No pytest; prints PASS/FAIL per check."""

import csv
import itertools
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

import eval_tracking as ev


FAILURES = 0
HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "eval_tracking.py")


def check(name, ok, detail=""):
    global FAILURES
    if ok:
        print(f"PASS {name}", flush=True)
        return
    FAILURES += 1
    suffix = f": {detail}" if detail else ""
    print(f"FAIL {name}{suffix}", flush=True)


def run_check(name, fn):
    try:
        fn()
    except Exception as exc:
        check(name, False, f"{type(exc).__name__}: {exc}")
    else:
        check(name, True)


def gt(person, team, box, main=False, hidden=False):
    return {
        "person": person,
        "team": team,
        "box": list(box),
        "main": main,
        "hidden": hidden,
    }


def pred(track_id, team, box):
    return {"id": track_id, "team": team, "box": list(box), "conf": 1.0}


def labels_of(frames):
    return {
        "video": "synthetic",
        "crop_x0": 0,
        "frames": {str(frame): entries for frame, entries in frames.items()},
    }


def brute_min_cost(cost):
    matrix = np.asarray(cost, dtype=float)
    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        return 0.0
    best = None
    if rows <= cols:
        for chosen in itertools.permutations(range(cols), rows):
            total = float(sum(matrix[i, chosen[i]] for i in range(rows)))
            if best is None or total < best:
                best = total
    else:
        for chosen in itertools.permutations(range(rows), cols):
            total = float(sum(matrix[chosen[j], j] for j in range(cols)))
            if best is None or total < best:
                best = total
    return best


def assignment_cost(cost, pairs):
    return float(sum(cost[row, col] for row, col in pairs))


def test_hungarian():
    classic = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
    pairs = ev.hungarian(classic)
    assert set(map(tuple, pairs)) == {(0, 1), (1, 0), (2, 2)}, pairs
    assert abs(assignment_cost(classic, pairs) - 5.0) < 1e-9

    # The single best cell (cost -0.80) plus the leftover (-0.10) sums to
    # -0.90. The crossed pairing sums to -1.50 and is the optimum.
    greedy = np.array([[-0.80, -0.75], [-0.75, -0.10]])
    pairs = ev.hungarian(greedy)
    got = assignment_cost(greedy, pairs)
    assert abs(got - (-1.50)) < 1e-9, (pairs, got)

    wide = np.array([[9.0, 2.0, 7.0], [6.0, 4.0, 3.0]])
    pairs = ev.hungarian(wide)
    assert set(map(tuple, pairs)) == {(0, 1), (1, 2)}, pairs
    assert ev.hungarian(np.zeros((0, 3))) == []

    # One IoU of 1.0 beats two IoUs of ~0.35. A cardinality bonus would
    # keep both weak pairs; the IoU objective keeps the single strong pair.
    strong = labels_of({
        0: [
            gt("a", "red", [0, 0, 10, 10]),
            gt("b", "red", [4.8, 0, 10, 10]),
        ],
    })
    strong_preds = {
        0: [
            pred(1, "red", [0, 0, 10, 10]),
            pred(2, "red", [-4.8, 0, 10, 10]),
        ],
    }
    high = ev.box_iou([0, 0, 10, 10], [0, 0, 10, 10])
    weak_a = ev.box_iou([0, 0, 10, 10], [-4.8, 0, 10, 10])
    weak_b = ev.box_iou([4.8, 0, 10, 10], [0, 0, 10, 10])
    assert weak_a >= 0.3 and weak_b >= 0.3
    assert high > weak_a + weak_b
    scored = ev.evaluate(strong, strong_preds, 0.3)
    assert scored["matches"] == 1, scored["matches"]
    assert scored["misses"] == 1
    assert scored["per_frame"][0]["missed_persons"] == ["b"]

    rng = np.random.default_rng(0)
    shapes = [(1, 1), (2, 2), (3, 3), (2, 4), (4, 2), (3, 5), (5, 3), (6, 6), (7, 4), (7, 7)]
    for rows, cols in shapes:
        trials = 2 if rows * cols >= 36 else 4
        for _trial in range(trials):
            matrix = rng.integers(-5, 20, size=(rows, cols)).astype(float)
            pairs = ev.hungarian(matrix)
            assert len(pairs) == min(rows, cols), (rows, cols, pairs)
            assert len({row for row, _col in pairs}) == len(pairs)
            assert len({col for _row, col in pairs}) == len(pairs)
            got = assignment_cost(matrix, pairs)
            best = brute_min_cost(matrix)
            assert abs(got - best) < 1e-6, (matrix, got, best, pairs)


def test_coverage_precision():
    # B is covered by a blue box. Same-team matching must leave him a miss
    # and count that box, plus the stray red box, as false.
    frame = labels_of({
        0: [
            gt("A", "red", [0, 0, 10, 10]),
            gt("B", "red", [30, 0, 10, 10]),
        ],
    })
    preds = {
        0: [
            pred(1, "red", [0, 0, 10, 10]),
            pred(2, "blue", [30, 0, 10, 10]),
            pred(3, "red", [60, 0, 10, 10]),
        ],
    }
    result = ev.evaluate(frame, preds, 0.3)
    assert result["matches"] == 1, result["matches"]
    assert result["misses"] == 1
    assert result["false_boxes"] == 2
    assert result["visible"] == 2
    assert result["coverage"] == 0.5
    assert result["precision"] == round(1 / 3, 6)
    assert result["per_frame"][0]["missed_persons"] == ["B"]


def test_id_switch():
    # Consecutive scored frames. A goes 1 -> 2 and B goes 2 -> 1.
    frames, preds = _swap_case()
    result = ev.evaluate(frames, preds, 0.3)
    assert result["id_switches"] == 2, result["switches"]
    observed = {
        (item["person"], item["from_id"], item["to_id"]) for item in result["switches"]
    }
    assert observed == {("A", 1, 2), ("B", 2, 1)}, observed
    assert result["per_frame"][1]["switches"] == 2


def test_duplicate():
    frames = labels_of({0: [gt("A", "red", [0, 0, 10, 10])]})
    preds = {0: [pred(1, "red", [0, 0, 10, 10]), pred(2, "red", [2, 0, 10, 10])]}
    overlap = ev.box_iou([0, 0, 10, 10], [2, 0, 10, 10])
    assert 0.3 <= overlap < 0.9, overlap
    low = ev.evaluate(frames, preds, 0.3)
    assert low["matches"] == 1
    assert low["duplicates"] == 1, low["duplicates"]
    assert low["false_boxes"] == 1
    assert low["coverage"] == 1.0
    assert low["precision"] == 0.5
    high = ev.evaluate(frames, preds, 0.9)
    assert high["duplicates"] == 0
    assert high["matches"] == 1
    assert high["false_boxes"] == 1


def test_hidden():
    # Frame 0: a box on the hidden player is absorbed, and a stray box is false.
    # Frame 1: the hidden player has no box, which is not a miss.
    frames = labels_of({
        0: [
            gt("V", "red", [0, 0, 10, 10]),
            gt("H", "red", [40, 0, 10, 10], hidden=True),
        ],
        1: [
            gt("V", "red", [0, 0, 10, 10]),
            gt("H", "red", [40, 0, 10, 10], hidden=True),
        ],
    })
    preds = {
        0: [
            pred(1, "red", [0, 0, 10, 10]),
            pred(2, "red", [40, 0, 10, 10]),
            pred(3, "red", [80, 0, 10, 10]),
        ],
        1: [pred(1, "red", [0, 0, 10, 10])],
    }
    result = ev.evaluate(frames, preds, 0.3)
    assert result["matches"] == 2, result["matches"]
    assert result["misses"] == 0
    assert result["false_boxes"] == 1
    assert result["hidden_matched"] == 1
    assert result["visible"] == 2
    assert result["hidden"] == 2
    assert result["predicted"] == 4
    assert result["coverage"] == 1.0
    assert result["precision"] == round(2 / 3, 6)
    # The absorbed box is not an identity false positive.
    # IDTP 2, IDFP 1, IDFN 0 -> 4 / 5 = 0.8
    assert result["idtp"] == 2
    assert result["idfp"] == 1
    assert result["idfn"] == 0
    assert result["idf1"] == 0.8


def _swap_case():
    frames = labels_of({
        0: [
            gt("A", "red", [0, 0, 10, 10], main=True),
            gt("B", "red", [30, 0, 10, 10]),
        ],
        1: [
            gt("A", "red", [0, 0, 10, 10], main=True),
            gt("B", "red", [30, 0, 10, 10]),
        ],
    })
    preds = {
        0: [pred(1, "red", [0, 0, 10, 10]), pred(2, "red", [30, 0, 10, 10])],
        1: [pred(2, "red", [0, 0, 10, 10]), pred(1, "red", [30, 0, 10, 10])],
    }
    return frames, preds


def test_idf1():
    # Each person is matched on both frames, but the track ids swap.
    # Co-occurrence is 1 for every person/id pair, so the best one-to-one
    # map keeps 2 of the 4 frame matches.
    # IDTP 2, IDFP 2, IDFN 2 -> IDF1 = 4 / 8 = 0.5
    # The main person agrees with his mapped id on exactly one of two frames.
    frames, preds = _swap_case()
    result = ev.evaluate(frames, preds, 0.3)
    assert result["matches"] == 4
    assert result["misses"] == 0
    assert result["false_boxes"] == 0
    assert result["coverage"] == 1.0
    assert result["precision"] == 1.0
    assert result["idtp"] == 2, result["idtp"]
    assert result["idfp"] == 2
    assert result["idfn"] == 2
    assert result["idf1"] == 0.5
    assert set(result["identity_map"]) == {"A", "B"}
    assert len(set(result["identity_map"].values())) == 2
    assert result["main_person"] == "A"
    assert result["main_track_id"] in (1, 2)
    assert result["main_frames"] == 2
    assert result["main_correct"] == 1
    assert result["main_accuracy"] == 0.5


def write_case(directory, labels, predictions):
    label_path = os.path.join(directory, "labels.json")
    box_path = os.path.join(directory, "boxes.csv")
    with open(label_path, "w") as handle:
        json.dump(labels, handle)
    with open(box_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "id", "team", "x", "y", "w", "h", "conf"])
        for frame in sorted(predictions):
            for item in predictions[frame]:
                x, y, width, height = item["box"]
                writer.writerow([
                    frame, item["id"], item["team"], x, y, width, height, item["conf"],
                ])
    return label_path, box_path


def run_eval(args):
    proc = subprocess.run(
        [sys.executable, EVAL, *args],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def tweak(src, dst, **changes):
    with open(src) as handle:
        data = json.load(handle)
    data.update(changes)
    with open(dst, "w") as handle:
        json.dump(data, handle)


def test_baseline_exit_codes():
    coverage_labels = labels_of({
        0: [gt("A", "red", [0, 0, 10, 10]), gt("B", "red", [30, 0, 10, 10])],
    })
    coverage_preds = {
        0: [pred(1, "red", [0, 0, 10, 10]), pred(9, "red", [60, 0, 10, 10])],
    }
    swap_labels, swap_preds = _swap_case()
    with tempfile.TemporaryDirectory() as directory:
        labels_path, boxes_path = write_case(directory, coverage_labels, coverage_preds)
        report_path = os.path.join(directory, "report.json")
        code, output = run_eval([
            "--labels", labels_path, "--boxes", boxes_path, "--json-out", report_path,
        ])
        assert code == 0, output
        code, output = run_eval([
            "--labels", labels_path, "--boxes", boxes_path, "--baseline", report_path,
        ])
        assert code == 0, output
        assert "baseline comparison: ok" in output

        better = os.path.join(directory, "better.json")
        tweak(report_path, better, coverage=0.2, misses=5, false_boxes=5, id_switches=3)
        code, output = run_eval([
            "--labels", labels_path, "--boxes", boxes_path, "--baseline", better,
        ])
        assert code == 0, output
        assert "baseline comparison: ok" in output

        close = os.path.join(directory, "close.json")
        tweak(report_path, close, coverage=0.504)
        code, output = run_eval([
            "--labels", labels_path,
            "--boxes", boxes_path,
            "--baseline", close,
            "--tolerance", "0.01",
        ])
        assert code == 0, output

        dropped = os.path.join(directory, "dropped.json")
        tweak(report_path, dropped, coverage=0.8)
        code, output = run_eval([
            "--labels", labels_path, "--boxes", boxes_path, "--baseline", dropped,
        ])
        assert code == 1, output
        assert "baseline comparison: WORSE" in output
        assert "coverage" in output

        swap_labels_path, swap_boxes_path = write_case(
            directory, swap_labels, swap_preds
        )
        swap_report = os.path.join(directory, "swap.json")
        code, output = run_eval([
            "--labels", swap_labels_path,
            "--boxes", swap_boxes_path,
            "--json-out", swap_report,
        ])
        assert code == 0, output
        fewer_switches = os.path.join(directory, "fewer_switches.json")
        tweak(swap_report, fewer_switches, id_switches=0)
        code, output = run_eval([
            "--labels", swap_labels_path,
            "--boxes", swap_boxes_path,
            "--baseline", fewer_switches,
        ])
        assert code == 1, output
        assert "id_switches" in output
        assert "baseline comparison: WORSE" in output


def main():
    run_check("hungarian result", test_hungarian)
    run_check("coverage/precision", test_coverage_precision)
    run_check("id switch", test_id_switch)
    run_check("duplicate box", test_duplicate)
    run_check("hidden player", test_hidden)
    run_check("IDF1", test_idf1)
    run_check("baseline exit code", test_baseline_exit_codes)
    if FAILURES:
        print(f"FAILED {FAILURES}", flush=True)
        return 1
    print("ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
