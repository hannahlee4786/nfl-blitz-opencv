#!/usr/bin/env python3
"""Same-team tracklet retrieval: rank-1 and mAP for an embedding and for HSV."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _ranks(queries: list[np.ndarray], gallery: list[np.ndarray], q_ids: list[str], g_ids: list[str]) -> tuple[float, float]:
    if not queries or not gallery:
        return float("nan"), float("nan")
    q = np.stack(queries)
    g = np.stack(gallery)
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-8, None)
    g = g / np.clip(np.linalg.norm(g, axis=1, keepdims=True), 1e-8, None)
    sim = q @ g.T
    ranks = []
    aps = []
    g_ids_arr = np.array(g_ids)
    for row, identity in enumerate(q_ids):
        order = np.argsort(-sim[row])
        ordered_ids = g_ids_arr[order]
        hit = np.where(ordered_ids == identity)[0]
        if len(hit) == 0:
            continue
        rank = int(hit[0]) + 1
        ranks.append(1.0 if rank == 1 else 0.0)
        # One relevant tracklet.
        aps.append(1.0 / rank)
    if not ranks:
        return float("nan"), float("nan")
    return float(np.mean(ranks)), float(np.mean(aps))


def tracklet_retrieval(records: list[dict], feature_key: str, gap_min: int = 10, gap_max: int = 40) -> dict:
    """Same-team retrieval at the gap the tracker stitches across.

    Each query crop is matched against one later crop of every same-team
    tracklet in that clip whose timestamp sits near the positive.
    """
    grouped: dict[tuple, list] = defaultdict(list)
    for row in records:
        feat = row.get(feature_key)
        if feat is None:
            continue
        grouped[(row["clip"], row["team"], row["identity"])].append(row)
    by_clip_team: dict[tuple, list] = defaultdict(list)
    for (clip, team, identity), rows in grouped.items():
        rows = sorted(rows, key=lambda item: item["frame"])
        if len(rows) >= 2:
            by_clip_team[(clip, team)].append((identity, rows))

    queries, gallery, q_ids, g_ids = [], [], [], []
    galleries = 0
    for members in by_clip_team.values():
        if len(members) < 2:
            continue
        for identity, rows in members:
            for index, query in enumerate(rows):
                later = [
                    row
                    for row in rows[index + 1 :]
                    if gap_min <= row["frame"] - query["frame"] <= gap_max
                ]
                if not later:
                    continue
                positive = later[0]
                window = []
                for other_id, other_rows in members:
                    if other_id == identity:
                        window.append(positive)
                        continue
                    near = min(other_rows, key=lambda row: abs(row["frame"] - positive["frame"]))
                    if abs(near["frame"] - positive["frame"]) <= 15:
                        window.append(near)
                if len(window) < 2:
                    continue
                queries.append(np.asarray(query[feature_key], dtype=np.float32))
                for item in window:
                    gallery.append(np.asarray(item[feature_key], dtype=np.float32))
                    g_ids.append(item["identity"])
                # _ranks expects one shared gallery. Score this query alone.
                rank1, ap = _ranks(
                    [queries[-1]],
                    [np.asarray(item[feature_key], dtype=np.float32) for item in window],
                    [identity],
                    [item["identity"] for item in window],
                )
                queries.pop()
                # stash by appending to parallel lists built below
                if rank1 == rank1:
                    q_ids.append((rank1, ap, len(window)))
                    galleries += len(window)
    if not q_ids:
        return {"queries": 0, "rank1": float("nan"), "mAP": float("nan"), "mean_gallery": 0}
    return {
        "queries": len(q_ids),
        "rank1": float(np.mean([item[0] for item in q_ids])),
        "mAP": float(np.mean([item[1] for item in q_ids])),
        "mean_gallery": float(np.mean([item[2] for item in q_ids])),
    }


def load_manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from ml.paths import ml_home

    root = ml_home() / "reid"
    records = load_manifest(root / "dataset" / "val.jsonl")
    emb_path = root / "val_embeddings.npy"
    ids_path = root / "val_emb_ids.json"
    if emb_path.is_file():
        matrix = np.load(emb_path)
        order = json.loads(ids_path.read_text())
        by_file = {row["file"]: row for row in records}
        packed = []
        for index, name in enumerate(order):
            row = dict(by_file[name])
            row["emb"] = matrix[index].tolist()
            packed.append(row)
        learned = tracklet_retrieval(packed, "emb")
    else:
        learned = {"queries": 0, "rank1": None, "mAP": None}
    hsv = tracklet_retrieval(records, "hist")
    report = {"learned": learned, "hsv": hsv}
    dest = root / "logs" / "val_retrieval.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
