#!/usr/bin/env python3
"""Make ground-truth tracking labels for a clip in the tools/eval_tracking.py format.

Three steps:

    draft  Pick every --step-th frame, pre-fill boxes from the raw YOLO detection
           cache (not the tracker), and write data/labels/<name>.draft.json plus
           numbered stills. Every box starts with an empty person name.
    edit   Open the draft in a window to fix boxes, add missing players, delete
           false boxes, and name each player. Names are kept across frames.
    check  Validate the draft, write per-player crop strips for an identity
           review, and with --final write data/labels/<name>.json.

A label frame must list every visible player whose feet are on screen. A
person name is the team plus jersey number when it can be read (red55,
blue8); otherwise team plus a letter (red_a). The same human keeps the same
name in every frame. "hidden" marks a player fully behind someone else, and
"main" marks the player the clip follows (the one under the blue ring).

Editor keys (window must have focus):

    click a box        select it            drag on empty field   add a box
    r then drag        redraw selected      x / Delete            delete selected
    i                  type a name, Enter   t                     toggle red/blue
    h                  toggle hidden        m                     toggle main
    c                  copy names from the previous labeled frame to unnamed boxes
    . or ]             next frame           , or [                previous frame
    u                  undo                 s                     save
    q                  save and quit        ?                     show these keys
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.paths import cache_root  # noqa: E402
from tools.eval_tracking import box_iou  # noqa: E402

TEAM_BGR = {"red": (60, 60, 235), "blue": (235, 140, 40)}
SELECTED_BGR = (0, 255, 255)
ROSTER_PER_TEAM = 7


# ----------------------------------------------------------------- drafting


def read_frames(video: Path, wanted: list[int]) -> dict[int, np.ndarray]:
    need = set(wanted)
    found = {}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    index = 0
    while need and index <= max(need):
        ok, frame = cap.read()
        if not ok:
            break
        if index in need:
            found[index] = frame
        index += 1
    cap.release()
    return found


def dedupe(boxes: list[dict]) -> list[dict]:
    """Keep the stronger of two boxes on one body (IoU >= 0.5 or 80% containment)."""
    kept: list[dict] = []
    for item in sorted(boxes, key=lambda b: -b["conf"]):
        x, y, w, h = item["box"]
        duplicate = False
        for other in kept:
            ox, oy, ow, oh = other["box"]
            inter = max(0, min(x + w, ox + ow) - max(x, ox)) * max(0, min(y + h, oy + oh) - max(y, oy))
            contain = inter / max(1, min(w * h, ow * oh))
            if box_iou(item["box"], other["box"]) >= 0.5 or contain >= 0.8:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def draft_frames(cache_stem: Path, frames: list[int], min_conf: float) -> dict[str, list[dict]]:
    """Boxes per frame from raw detections, with empty person names."""
    rows = list(csv.DictReader(cache_stem.with_suffix(".csv").open(newline="")))
    icons_path = cache_stem.with_suffix(".icons.npy")
    icons = np.load(icons_path) if icons_path.is_file() else np.zeros(len(rows))
    wanted = set(frames)
    per_frame: dict[int, list[dict]] = {f: [] for f in frames}
    for row, icon in zip(rows, icons):
        f = int(row["frame"])
        if f not in wanted or float(row["conf"]) < min_conf or icon:
            continue
        per_frame[f].append({
            "box": [int(round(float(row[k]))) for k in ("x", "y", "w", "h")],
            "team": row["team"],
            "conf": float(row["conf"]),
        })
    out = {}
    for f in frames:
        out[str(f)] = [
            {"person": "", "team": b["team"], "box": b["box"], "main": False, "hidden": False}
            for b in sorted(dedupe(per_frame[f]), key=lambda b: b["box"][0])
        ]
    return out


def draw(frame: np.ndarray, entries: list[dict], selected: int | None = None, title: str = "") -> np.ndarray:
    canvas = frame.copy()
    for index, entry in enumerate(entries):
        x, y, w, h = entry["box"]
        color = SELECTED_BGR if index == selected else TEAM_BGR.get(entry["team"], (200, 200, 200))
        thickness = 3 if index == selected else 2
        if entry.get("hidden"):
            for k in range(0, w, 12):
                cv2.line(canvas, (x + k, y), (x + min(k + 6, w), y), color, thickness)
                cv2.line(canvas, (x + k, y + h), (x + min(k + 6, w), y + h), color, thickness)
            cv2.line(canvas, (x, y), (x, y + h), color, thickness)
            cv2.line(canvas, (x + w, y), (x + w, y + h), color, thickness)
        else:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        name = entry["person"] or "?"
        tag = f"{index}:{name}" + (" MAIN" if entry.get("main") else "") + (" hid" if entry.get("hidden") else "")
        cv2.putText(canvas, tag, (x, max(16, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, tag, (x, max(16, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    if title:
        cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def write_stills(video: Path, labels: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(int(k) for k in labels["frames"])
    images = read_frames(video, frames)
    written = []
    for f in frames:
        if f not in images:
            continue
        path = out_dir / f"f{f:04d}.png"
        cv2.imwrite(str(path), draw(images[f], labels["frames"][str(f)], title=f"frame {f}"))
        written.append(path)
    return written


# ----------------------------------------------------------------- editing state


class LabelState:
    """All edits as plain methods so they can be tested without a window."""

    def __init__(self, labels: dict):
        self.labels = labels
        self.frames = sorted(int(k) for k in labels["frames"])
        self.position = 0
        self.selected: int | None = None
        self.history: list[tuple[int, list[dict]]] = []

    @property
    def frame(self) -> int:
        return self.frames[self.position]

    @property
    def entries(self) -> list[dict]:
        return self.labels["frames"][str(self.frame)]

    def _remember(self) -> None:
        self.history.append((self.frame, copy.deepcopy(self.entries)))
        del self.history[:-200]

    def undo(self) -> None:
        if not self.history:
            return
        frame, entries = self.history.pop()
        self.labels["frames"][str(frame)] = entries
        self.position = self.frames.index(frame)
        self.selected = None

    def move(self, step: int) -> None:
        self.position = int(np.clip(self.position + step, 0, len(self.frames) - 1))
        self.selected = None

    def select_at(self, x: float, y: float) -> int | None:
        """Smallest box containing the point, so a player in front of a pile is easy to pick."""
        hits = [(e["box"][2] * e["box"][3], i) for i, e in enumerate(self.entries)
                if e["box"][0] <= x <= e["box"][0] + e["box"][2] and e["box"][1] <= y <= e["box"][1] + e["box"][3]]
        self.selected = min(hits)[1] if hits else None
        return self.selected

    def add(self, box: list[int], team: str) -> int:
        self._remember()
        self.entries.append({"person": "", "team": team, "box": [int(v) for v in box], "main": False, "hidden": False})
        self.selected = len(self.entries) - 1
        return self.selected

    def replace_box(self, box: list[int]) -> None:
        if self.selected is None:
            return
        self._remember()
        self.entries[self.selected]["box"] = [int(v) for v in box]

    def delete(self) -> None:
        if self.selected is None:
            return
        self._remember()
        del self.entries[self.selected]
        self.selected = None

    def set_name(self, text: str) -> None:
        """'55' becomes <team>55; 'red55' or 'blue_a' is taken as written and also sets the team."""
        if self.selected is None or not text.strip():
            return
        self._remember()
        entry = self.entries[self.selected]
        text = text.strip().lower()
        for team in ("red", "blue"):
            if text.startswith(team):
                entry["team"] = team
                entry["person"] = text
                return
        entry["person"] = f"{entry['team']}{text}"

    def toggle(self, key: str) -> None:
        if self.selected is None:
            return
        self._remember()
        entry = self.entries[self.selected]
        if key == "team":
            entry["team"] = "blue" if entry["team"] == "red" else "red"
            if entry["person"].startswith(("red", "blue")):
                entry["person"] = entry["team"] + entry["person"].removeprefix("red").removeprefix("blue")
        else:
            entry[key] = not entry.get(key, False)
            if key == "main" and entry["main"]:
                for other in self.entries:
                    if other is not entry:
                        other["main"] = False

    def copy_previous(self) -> int:
        """Name unnamed boxes after the previous labeled frame's boxes they overlap most.

        Uses this file's own labels, never tracker ids. Ten frames apart a
        player moves, so the match is the closest same-team foot within one
        body height, then IoU. Every copied name should still be checked.
        """
        if self.position == 0:
            return 0
        previous = self.labels["frames"][str(self.frames[self.position - 1])]
        used = {e["person"] for e in self.entries if e["person"]}
        self._remember()
        filled = 0
        for entry in self.entries:
            if entry["person"]:
                continue
            x, y, w, h = entry["box"]
            best = None
            for prev in previous:
                if not prev["person"] or prev["person"] in used or prev["team"] != entry["team"]:
                    continue
                px, py, pw, ph = prev["box"]
                foot = float(np.hypot((x + w / 2) - (px + pw / 2), (y + h) - (py + ph)))
                if foot > max(h, ph):
                    continue
                score = (foot / max(h, ph), -box_iou(entry["box"], prev["box"]))
                if best is None or score < best[0]:
                    best = (score, prev)
            if best is not None:
                entry["person"] = best[1]["person"]
                entry["main"] = bool(best[1].get("main"))
                used.add(entry["person"])
                filled += 1
        return filled


# ----------------------------------------------------------------- checking


def problems(labels: dict) -> list[str]:
    found = []
    for key in sorted(labels["frames"], key=int):
        entries = labels["frames"][key]
        names = [e["person"] for e in entries]
        if any(not n for n in names):
            found.append(f"frame {key}: {sum(1 for n in names if not n)} box(es) without a person name")
        for name in {n for n in names if n and names.count(n) > 1}:
            found.append(f"frame {key}: {name} is used {names.count(name)} times")
        for e in entries:
            if e["person"] and not e["person"].startswith(e["team"]):
                found.append(f"frame {key}: {e['person']} is marked team {e['team']}")
            x, y, w, h = e["box"]
            if w < 4 or h < 4:
                found.append(f"frame {key}: {e['person'] or '?'} has a tiny box {e['box']}")
        for team in ("red", "blue"):
            visible = sum(1 for e in entries if e["team"] == team and not e.get("hidden"))
            if visible > ROSTER_PER_TEAM:
                found.append(f"frame {key}: {visible} visible {team} players (a team has 7)")
        if sum(1 for e in entries if e.get("main")) > 1:
            found.append(f"frame {key}: more than one main player")
    return found


def person_strips(video: Path, labels: dict, out_path: Path, cell: int = 96) -> None:
    """One row per person, one cell per labeled frame: a wrong name stands out in its row."""
    frames = sorted(int(k) for k in labels["frames"])
    images = read_frames(video, frames)
    people = sorted({e["person"] for entries in labels["frames"].values() for e in entries if e["person"]})
    if not people:
        return
    label_w = 120
    sheet = np.full((cell * len(people) + 24, label_w + cell * len(frames), 3), 30, np.uint8)
    for col, f in enumerate(frames):
        cv2.putText(sheet, str(f), (label_w + col * cell + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for row, person in enumerate(people):
        top = 24 + row * cell
        cv2.putText(sheet, person, (6, top + cell // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    TEAM_BGR.get("red" if person.startswith("red") else "blue"), 1)
        for col, f in enumerate(frames):
            entry = next((e for e in labels["frames"][str(f)] if e["person"] == person), None)
            if entry is None or f not in images:
                continue
            x, y, w, h = entry["box"]
            crop = images[f][max(0, y):y + h, max(0, x):x + w]
            if crop.size == 0:
                continue
            scale = (cell - 4) / max(crop.shape[:2])
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
            left = label_w + col * cell + 2
            sheet[top + 2:top + 2 + crop.shape[0], left:left + crop.shape[1]] = crop
            if entry.get("hidden"):
                cv2.putText(sheet, "hid", (left, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


# ----------------------------------------------------------------- window


def run_editor(video: Path, draft_path: Path, max_width: int) -> None:
    labels = json.loads(draft_path.read_text())
    state = LabelState(labels)
    images = read_frames(video, state.frames)
    shape = next(iter(images.values())).shape
    scale = min(1.0, max_width / shape[1])
    mode = {"redraw": False, "typing": None, "drag": None, "help": True, "message": ""}
    window = f"label {draft_path.name}"

    def to_frame(px, py):
        return px / scale, py / scale

    def on_mouse(event, px, py, _flags, _param):
        x, y = to_frame(px, py)
        if event == cv2.EVENT_LBUTTONDOWN:
            mode["drag"] = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and mode["drag"]:
            x0, y0, _, _ = mode["drag"]
            mode["drag"] = (x0, y0, x, y)
        elif event == cv2.EVENT_LBUTTONUP and mode["drag"]:
            x0, y0, x1, y1 = mode["drag"]
            mode["drag"] = None
            box = [min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)]
            if box[2] < 6 or box[3] < 6:
                state.select_at(x1, y1)
            elif mode["redraw"]:
                state.replace_box(box)
                mode["redraw"] = False
            else:
                team = state.entries[state.selected]["team"] if state.selected is not None else "red"
                state.add(box, team)
                mode["message"] = "new box: press i to name it, t to switch team"

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        frame = images[state.frame]
        title = f"frame {state.frame} ({state.position + 1}/{len(state.frames)})  ? for keys"
        canvas = draw(frame, state.entries, state.selected, title)
        if mode["drag"]:
            x0, y0, x1, y1 = (int(v) for v in mode["drag"])
            cv2.rectangle(canvas, (x0, y0), (x1, y1), SELECTED_BGR, 1)
        status = mode["message"]
        if mode["typing"] is not None:
            status = f"name: {mode['typing']}_   (Enter to set, Esc to cancel)"
        elif mode["redraw"]:
            status = "drag a new box for the selected player"
        unnamed = sum(1 for e in state.entries if not e["person"])
        status = f"{status}   unnamed here: {unnamed}"
        cv2.putText(canvas, status, (10, canvas.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, status, (10, canvas.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if mode["help"]:
            for row, line in enumerate(__doc__.split("Editor keys")[1].strip().splitlines()[2:]):
                cv2.putText(canvas, line.strip(), (10, 70 + 24 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(canvas, line.strip(), (10, 70 + 24 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            continue
        mode["message"] = ""
        if mode["typing"] is not None:
            if key in (13, 10):
                state.set_name(mode["typing"])
                mode["typing"] = None
            elif key == 27:
                mode["typing"] = None
            elif key in (8, 127):
                mode["typing"] = mode["typing"][:-1]
            elif chr(key).isalnum() or chr(key) == "_":
                mode["typing"] += chr(key)
            continue
        ch = chr(key)
        if ch in ".]":
            state.move(1)
        elif ch in ",[":
            state.move(-1)
        elif ch in ("x",) or key in (8, 127):
            state.delete()
        elif ch == "i" and state.selected is not None:
            mode["typing"] = ""
        elif ch == "t":
            state.toggle("team")
        elif ch == "h":
            state.toggle("hidden")
        elif ch == "m":
            state.toggle("main")
        elif ch == "r" and state.selected is not None:
            mode["redraw"] = True
        elif ch == "c":
            mode["message"] = f"copied {state.copy_previous()} name(s) from the previous frame; check them"
        elif ch == "u":
            state.undo()
        elif ch == "?":
            mode["help"] = not mode["help"]
        elif ch == "s":
            draft_path.write_text(json.dumps(labels, indent=1) + "\n")
            mode["message"] = f"saved {draft_path.name}"
        elif ch == "q":
            draft_path.write_text(json.dumps(labels, indent=1) + "\n")
            break
    cv2.destroyAllWindows()
    print(f"saved {draft_path}")


# ----------------------------------------------------------------- commands


def default_stem(video: Path) -> Path:
    return cache_root() / "det" / f"{video.stem}_yolo11m"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("draft", "edit", "check"):
        p = sub.add_parser(name)
        p.add_argument("--video", type=Path, required=True)
        p.add_argument("--name", required=True, help="Label name, e.g. clip1 -> data/labels/clip1.json")
    sub.choices["draft"].add_argument("--step", type=int, default=10, help="Label every Nth frame.")
    sub.choices["draft"].add_argument("--min-conf", type=float, default=0.30)
    sub.choices["draft"].add_argument("--cache", type=Path, default=None,
                                      help="Detection cache stem; default is the video's stem under $NFL_ML_CACHE/det.")
    sub.choices["draft"].add_argument("--force", action="store_true", help="Overwrite an existing draft.")
    sub.choices["edit"].add_argument("--max-width", type=int, default=1400)
    sub.choices["check"].add_argument("--final", action="store_true",
                                      help="Write data/labels/<name>.json when there are no problems.")
    args = parser.parse_args(argv)

    labels_dir = REPO / "data/labels"
    draft_path = labels_dir / f"{args.name}.draft.json"
    review_dir = REPO / "data/output/labels" / args.name
    video = args.video.expanduser().resolve()

    if args.command == "draft":
        if draft_path.exists() and not args.force:
            parser.error(f"{draft_path} exists; pass --force to start over")
        stem = args.cache or default_stem(video)
        if not stem.with_suffix(".csv").is_file():
            parser.error(f"no detection cache at {stem}.csv; run ml/run_clips.sh or ml/detect_all.py --video first")
        count = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT))
        frames = list(range(0, count, args.step))
        try:
            rel = str(video.relative_to(REPO))
        except ValueError:
            rel = str(video)
        labels = {"video": rel, "crop_x0": 0, "frames": draft_frames(stem, frames, args.min_conf)}
        draft_path.write_text(json.dumps(labels, indent=1) + "\n")
        stills = write_stills(video, labels, review_dir / "draft")
        boxes = sum(len(v) for v in labels["frames"].values())
        print(f"wrote {draft_path}: {len(frames)} frames, {boxes} draft boxes, every name empty")
        print(f"wrote {len(stills)} numbered stills to {review_dir / 'draft'}")
        return 0

    if args.command == "edit":
        if not draft_path.is_file():
            parser.error(f"no draft at {draft_path}; run the draft command first")
        run_editor(video, draft_path, args.max_width)
        return 0

    labels = json.loads(draft_path.read_text())
    found = problems(labels)
    people = sorted({e["person"] for v in labels["frames"].values() for e in v if e["person"]})
    boxes = sum(len(v) for v in labels["frames"].values())
    print(f"{draft_path.name}: {len(labels['frames'])} frames, {boxes} boxes, {len(people)} people")
    print("people:", ", ".join(people) if people else "(none named yet)")
    for line in found:
        print("  problem:", line)
    write_stills(video, labels, review_dir / "check")
    person_strips(video, labels, review_dir / "people.png")
    print(f"review stills in {review_dir / 'check'}; one row per person in {review_dir / 'people.png'}")
    if args.final:
        if found:
            print("not written: fix the problems above first")
            return 1
        final = labels_dir / f"{args.name}.json"
        final.write_text(json.dumps(labels, indent=2) + "\n")
        print(f"wrote {final}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
