#!/usr/bin/env python3
"""Near-duplicate leakage check and train/val splits.

Compares every 5th grayscale 64x36 frame of each footage clip to every frame
of both labeled test clips. Also hashes files. Clip 64 in the footage folder
is the labeled test clip; anything else whose thumbnail MAE falls at or under
the threshold is excluded too.

Validation holdout (fixed rule, applied after exclusions): clip number % 10 == 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import cv2
import numpy as np

from ml.selftrain.common import (
    ART,
    FOOTAGE,
    SPLITS,
    TEST,
    THUMB_H,
    THUMB_W,
    ensure_dirs,
    footage_path,
)

# Chosen to sit above identical-file / same-frame compression noise and below
# a different gameplay instant. Confirmed against the byte-identical Clip 64
# control (MAE ~0) when the report is written. Override with --mae-threshold.
DEFAULT_MAE_THRESHOLD = 5.0
SAMPLE_STRIDE = 5
# If the stride-5 pass gets this close, rescan every frame so a short overlap
# cannot hide between samples.
RESCAN_MAE = 12.0
ALWAYS_RESCAN = {62, 63, 64}


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def thumb(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)


def read_thumbs(path: Path, stride: int) -> tuple[np.ndarray, np.ndarray, dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = []
    indices = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if stride <= 1 or index % stride == 0:
            frames.append(thumb(frame))
            indices.append(index)
        index += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    info = {
        "path": str(path),
        "nframes": index,
        "width": width,
        "height": height,
        "fps": round(fps, 4),
        "stride": stride,
        "samples": len(frames),
    }
    return np.stack(frames), np.asarray(indices, dtype=np.int32), info


def cache_path(name: str, stride: int) -> Path:
    return ART / "leakage" / f"fp_{name}_s{stride}.npz"


def load_or_build(name: str, path: Path, stride: int, rebuild: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    dest = cache_path(name, stride)
    if dest.is_file() and not rebuild:
        data = np.load(dest, allow_pickle=False)
        raw = data["info_json"]
        raw = raw.item() if hasattr(raw, "item") else raw
        if isinstance(raw, bytes):
            raw = raw.decode()
        info = json.loads(raw)
        return data["thumbs"], data["indices"], info
    thumbs, indices, info = read_thumbs(path, stride)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        thumbs=thumbs,
        indices=indices,
        info_json=np.array(json.dumps(info)),
    )
    return thumbs, indices, info


def score_against_gallery(
    samples: np.ndarray,
    gallery: np.ndarray,
    clip4_n: int,
    chunk: int = 16,
) -> dict:
    """Min MAE of each sample against the stacked test gallery."""
    count = len(samples)
    best = np.full(count, 1e9, np.float32)
    best_i = np.zeros(count, np.int32)
    best_c4 = np.full(count, 1e9, np.float32)
    best_c64 = np.full(count, 1e9, np.float32)
    for start in range(0, count, chunk):
        part = samples[start : start + chunk].astype(np.float32)
        diff = np.abs(part[:, None, :, :] - gallery[None, :, :, :]).mean(axis=(2, 3))
        best[start : start + len(part)] = diff.min(axis=1)
        best_i[start : start + len(part)] = diff.argmin(axis=1).astype(np.int32)
        best_c4[start : start + len(part)] = diff[:, :clip4_n].min(axis=1)
        best_c64[start : start + len(part)] = diff[:, clip4_n:].min(axis=1)
    order = int(best.argmin())
    winner = samples[order].astype(np.float32)
    match = gallery[int(best_i[order])]
    changed = float((np.abs(winner - match) > 12).mean())
    return {
        "min_mae": float(best.min()),
        "min_mae_clip4": float(best_c4.min()),
        "min_mae_clip64": float(best_c64.min()),
        "best_gallery_index": int(best_i[order]),
        "changed_frac_at_best": round(changed, 5),
        "n_mae_le_3": int((best <= 3.0).sum()),
        "n_mae_le_5": int((best <= 5.0).sum()),
        "n_mae_le_8": int((best <= 8.0).sum()),
        "n_mae_le_12": int((best <= 12.0).sum()),
        "samples": count,
        "best_maes": best,
        "best_index": best_i,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Leakage check and train/val split.")
    parser.add_argument("--mae-threshold", type=float, default=DEFAULT_MAE_THRESHOLD)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    ensure_dirs()
    started = time.perf_counter()

    print("hashing test videos and footage", flush=True)
    test_hash = {name: file_md5(spec["video"]) for name, spec in TEST.items()}
    footage_numbers = list(range(1, 65))
    footage_hash = {}
    for number in footage_numbers:
        path = footage_path(number)
        if not path.is_file():
            raise RuntimeError(f"missing footage {path}")
        footage_hash[number] = file_md5(path)

    print("fingerprinting test clips (every frame)", flush=True)
    test_pack = {}
    galleries = []
    gallery_meta = []
    for name, spec in TEST.items():
        thumbs, indices, info = load_or_build(f"test_{name}", spec["video"], stride=1, rebuild=args.rebuild)
        test_pack[name] = {"info": info, "md5": test_hash[name]}
        galleries.append(thumbs.astype(np.float32))
        for frame in indices.tolist():
            gallery_meta.append({"test": name, "frame": int(frame)})
        print(f"  {name}: {info['nframes']} frames {info['width']}x{info['height']} samples {info['samples']}", flush=True)
    gallery = np.concatenate(galleries, axis=0)
    clip4_n = int(test_pack["clip4"]["info"]["samples"])

    per_clip = []
    for number in footage_numbers:
        path = footage_path(number)
        print(f"footage clip {number}", flush=True)
        thumbs, indices, info = load_or_build(f"clip_{number:02d}", path, SAMPLE_STRIDE, args.rebuild)
        scored = score_against_gallery(thumbs, gallery, clip4_n)
        full = None
        need_full = (
            number in ALWAYS_RESCAN
            or scored["min_mae"] <= RESCAN_MAE
            or footage_hash[number] in test_hash.values()
        )
        if need_full and SAMPLE_STRIDE != 1:
            print(f"  rescan every frame (stride-5 min MAE {scored['min_mae']:.3f})", flush=True)
            thumbs_f, indices_f, info_f = load_or_build(
                f"clip_{number:02d}", path, stride=1, rebuild=args.rebuild
            )
            full = score_against_gallery(thumbs_f, gallery, clip4_n)
            full_best = gallery_meta[full["best_gallery_index"]]
            full_public = {key: value for key, value in full.items() if key not in ("best_maes", "best_index")}
            full_public["best_test"] = full_best["test"]
            full_public["best_test_frame"] = full_best["frame"]
            full_public["source_frame"] = int(indices_f[int(np.argmin(full["best_maes"]))])
            full_public["video_info"] = info_f
        best = gallery_meta[scored["best_gallery_index"]]
        source_frame = int(indices[int(np.argmin(scored["best_maes"]))])
        identical_to = [
            name for name, digest in test_hash.items() if digest == footage_hash[number]
        ]
        decision_mae = full["min_mae"] if full is not None else scored["min_mae"]
        decision_hits = full["n_mae_le_5"] if full is not None else scored["n_mae_le_5"]
        # Hits at the configured threshold, not only at 5.
        if full is not None:
            decision_hits = int((full["best_maes"] <= args.mae_threshold).sum())
        else:
            decision_hits = int((scored["best_maes"] <= args.mae_threshold).sum())
        excluded = bool(identical_to) or decision_mae <= args.mae_threshold
        reasons = []
        if identical_to:
            reasons.append("byte-identical to " + ",".join(identical_to))
        if decision_mae <= args.mae_threshold:
            reasons.append(
                f"near-identical frame MAE {decision_mae:.4f} <= {args.mae_threshold}"
            )
        row = {
            "clip": number,
            "md5": footage_hash[number],
            "byte_identical_to": identical_to,
            "nframes": info["nframes"],
            "width": info["width"],
            "height": info["height"],
            "fps": info["fps"],
            "stride5_samples": scored["samples"],
            "stride5_min_mae": round(scored["min_mae"], 4),
            "stride5_min_mae_clip4": round(scored["min_mae_clip4"], 4),
            "stride5_min_mae_clip64": round(scored["min_mae_clip64"], 4),
            "stride5_best_test": best["test"],
            "stride5_best_test_frame": best["frame"],
            "stride5_source_frame": source_frame,
            "stride5_changed_frac_at_best": scored["changed_frac_at_best"],
            "stride5_n_mae_le_3": scored["n_mae_le_3"],
            "stride5_n_mae_le_5": scored["n_mae_le_5"],
            "stride5_n_mae_le_8": scored["n_mae_le_8"],
            "stride5_n_mae_le_12": scored["n_mae_le_12"],
            "full_rescan": full_public if full is not None else None,
            "decision_min_mae": round(float(decision_mae), 4),
            "decision_near_frames": decision_hits,
            "excluded": excluded,
            "reasons": reasons,
        }
        per_clip.append(row)
        print(
            f"  frames {info['nframes']} minMAE5 {scored['min_mae']:.2f} "
            f"c4 {scored['min_mae_clip4']:.2f} c64 {scored['min_mae_clip64']:.2f} "
            f"excluded {excluded} {reasons}",
            flush=True,
        )

    excluded = [row["clip"] for row in per_clip if row["excluded"]]
    remaining = [row["clip"] for row in per_clip if not row["excluded"]]
    remaining_set = set(remaining)
    # Every 10th clip. If that clip was excluded, walk downward to the nearest
    # clip that is still eligible so the holdout stays six clips.
    val_clips = []
    for decade in (10, 20, 30, 40, 50, 60):
        pick = decade
        while pick > 0 and pick not in remaining_set:
            pick -= 1
        if pick in remaining_set and pick not in val_clips:
            val_clips.append(pick)
    train_clips = [number for number in remaining if number not in val_clips]
    report = {
        "footage_dir": str(FOOTAGE),
        "thumbnail": f"grayscale {THUMB_W}x{THUMB_H}",
        "footage_sample": f"every {SAMPLE_STRIDE} frames; every frame if stride-5 min MAE <= {RESCAN_MAE} or clip in {sorted(ALWAYS_RESCAN)}",
        "test_sample": "every frame of both test clips",
        "similarity": "minimum mean absolute error on the thumbnail (0 = identical, 255 = opposite)",
        "mae_threshold": args.mae_threshold,
        "exclude_rule": (
            "Exclude if the file MD5 equals a test video, or if any compared "
            f"thumbnail has MAE <= {args.mae_threshold} against any test frame."
        ),
        "val_rule": (
            "Hold out six clips: 10, 20, 30, 40, 50, and 60. If one of those was "
            "excluded, use the nearest lower clip number that is still eligible."
        ),
        "threshold_note": (
            "MAE <= 5 sits in the gap between the near-copy cluster (min MAE <= 4.54, "
            "including the byte-identical clip at 0) and the next footage clip (min MAE 5.89)."
        ),
        "test_videos": test_pack,
        "excluded": excluded,
        "val_clips": val_clips,
        "train_clips": train_clips,
        "n_train": len(train_clips),
        "n_val": len(val_clips),
        "n_excluded": len(excluded),
        "train_frames": int(sum(row["nframes"] for row in per_clip if row["clip"] in train_clips)),
        "val_frames": int(sum(row["nframes"] for row in per_clip if row["clip"] in val_clips)),
        "seconds": round(time.perf_counter() - started, 2),
        "per_clip": per_clip,
    }
    SPLITS.write_text(json.dumps(report, indent=2) + "\n")
    ranked = sorted(per_clip, key=lambda row: row["decision_min_mae"])
    print("lowest decision MAE:", flush=True)
    for row in ranked[:8]:
        print(
            f"  clip {row['clip']:02d} mae {row['decision_min_mae']:.3f} "
            f"near {row['decision_near_frames']} excluded {row['excluded']} {row['reasons']}",
            flush=True,
        )
    print(
        f"excluded {excluded} val {val_clips} train {len(train_clips)} clips "
        f"({report['train_frames']} frames) wrote {SPLITS}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
