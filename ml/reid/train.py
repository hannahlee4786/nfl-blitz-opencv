#!/usr/bin/env python3
"""Train a tracklet embedding. Checkpoints land under $NFL_ML_HOME/reid so a reboot can resume."""

from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml.reid.eval_retrieval import load_manifest, tracklet_retrieval  # noqa: E402
from ml.paths import ml_home  # noqa: E402
from ml.reid.model import Embedder, batch_hard_triplet  # noqa: E402

ROOT = ml_home() / "reid"
DATA = ROOT / "dataset"
CKPT = ROOT / "checkpoints"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_rows(split: str) -> list[dict]:
    rows = load_manifest(DATA / f"{split}.jsonl")
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["identity"]] += 1
    return [row for row in rows if counts[row["identity"]] >= 2]


class CropDataset(Dataset):
    def __init__(self, rows: list[dict], train: bool) -> None:
        self.rows = rows
        self.train = train
        self.root = DATA

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        bgr = cv2.imread(str(self.root / row["file"]))
        if bgr is None:
            raise RuntimeError(row["file"])
        image = augment(bgr) if self.train else torso(bgr)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - MEAN) / STD
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1))
        return tensor, row["identity"], row["file"]


def torso(bgr: np.ndarray) -> np.ndarray:
    """Jersey and helmet, excluding most of the grass and the feet."""
    height, width = bgr.shape[:2]
    y0, y1 = int(0.04 * height), int(0.70 * height)
    x0, x1 = int(0.15 * width), int(0.85 * width)
    patch = bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return bgr
    return cv2.resize(patch, (width, height), interpolation=cv2.INTER_AREA)


def augment(bgr: np.ndarray) -> np.ndarray:
    image = torso(bgr)
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
    scale = random.uniform(0.75, 1.0)
    height, width = image.shape[:2]
    crop_h, crop_w = max(8, int(height * scale)), max(8, int(width * scale))
    y0 = random.randint(0, height - crop_h)
    x0 = random.randint(0, width - crop_w)
    image = cv2.resize(image[y0 : y0 + crop_h, x0 : x0 + crop_w], (width, height), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.5:
        # Mild, so jersey shade stays a cue. Strong jitter made teammates identical.
        alpha = random.uniform(0.92, 1.08)
        beta = random.uniform(-6, 6)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        image = cv2.GaussianBlur(image, (3, 3), 0.6)
    return image


def make_batches(rows: list[dict], identities_per_team: int, shots: int, seed: int) -> list[list[int]]:
    by_id: dict[str, list[int]] = defaultdict(list)
    team_of: dict[str, str] = {}
    for index, row in enumerate(rows):
        by_id[row["identity"]].append(index)
        team_of[row["identity"]] = f"{row['clip']}_{row['team']}"
    groups: dict[str, list[str]] = defaultdict(list)
    for identity, team in team_of.items():
        groups[team].append(identity)
    rng = random.Random(seed)
    batches = []
    for _epoch_pass in range(2):
        for members in groups.values():
            if len(members) < 2:
                continue
            order = members[:]
            rng.shuffle(order)
            for start in range(0, len(order), identities_per_team):
                chosen = order[start : start + identities_per_team]
                if len(chosen) < 2:
                    continue
                batch = []
                for identity in chosen:
                    pool = by_id[identity][:]
                    rng.shuffle(pool)
                    # Prefer frames far apart: take ends of the time-sorted pool when possible.
                    timed = sorted(pool, key=lambda index: rows[index]["frame"])
                    if len(timed) >= shots:
                        picks = [timed[int(i)] for i in np.linspace(0, len(timed) - 1, shots)]
                    else:
                        picks = [rng.choice(timed) for _ in range(shots)]
                    batch.extend(int(index) for index in picks)
                batches.append(batch)
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def embed_split(model, rows: list[dict]) -> tuple[np.ndarray, list[str], list[dict]]:
    model.eval()
    loader_rows = rows
    dataset = CropDataset(loader_rows, train=False)
    matrix = np.zeros((len(dataset), 128), dtype=np.float32)
    names = []
    for start in range(0, len(dataset), 64):
        chunk = [dataset[i] for i in range(start, min(len(dataset), start + 64))]
        images = torch.stack([item[0] for item in chunk])
        pred = model(images).numpy()
        matrix[start : start + len(chunk)] = pred
        names.extend(item[2] for item in chunk)
    packed = []
    for index, row in enumerate(loader_rows):
        copied = dict(row)
        copied["emb"] = matrix[index]
        packed.append(copied)
    return matrix, names, packed


def save_ckpt(path: Path, model, optimizer, epoch: float, best: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_map": best},
        path,
    )


def main() -> int:
    torch.set_num_threads(6)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    train_rows = load_rows("train")
    val_rows = load_manifest(DATA / "val.jsonl")
    CKPT.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best = -1.0
    resume = CKPT / "last.pt"
    if resume.is_file():
        model = Embedder(pretrained=False)
    else:
        model = Embedder(pretrained=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    if resume.is_file():
        state = torch.load(resume, map_location="cpu")
        model.load_state_dict(state["model"])
        try:
            optimizer.load_state_dict(state["optimizer"])
        except ValueError as exc:
            print(f"optimizer reset: {exc}", flush=True)
        start_epoch = int(state["epoch"]) + 1
        best = float(state.get("best_map", -1.0))
        print(f"resume epoch {start_epoch} best_map {best:.4f}", flush=True)
    dataset = CropDataset(train_rows, train=True)
    epochs = 12
    started = time.perf_counter()
    log_path = ROOT / "logs" / "train.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    id_to_int = {identity: index for index, identity in enumerate(sorted({row["identity"] for row in train_rows}))}
    classifier = torch.nn.Linear(128, len(id_to_int))
    optimizer.add_param_group({"params": classifier.parameters(), "lr": 1e-3, "weight_decay": 1e-4})
    for epoch in range(start_epoch, epochs):
        model.train()
        batches = make_batches(train_rows, identities_per_team=8, shots=2, seed=epoch)
        losses = []
        for step, batch in enumerate(batches):
            images = torch.stack([dataset[index][0] for index in batch])
            labels = torch.tensor([id_to_int[dataset[index][1]] for index in batch])
            optimizer.zero_grad()
            features = model(images)
            logits = classifier(features) * 16.0
            loss = 0.5 * batch_hard_triplet(features, labels) + torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(batches)} loss {losses[-1]:.4f}", flush=True)
        _matrix, names, packed = embed_split(model, val_rows)
        metrics = tracklet_retrieval(packed, "emb")
        record = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else None,
            "minutes": round((time.perf_counter() - started) / 60, 2),
            "val": metrics,
            "train_images": len(train_rows),
            "steps": len(batches),
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        score = metrics["mAP"] if metrics["mAP"] == metrics["mAP"] else -1.0
        save_ckpt(CKPT / "last.pt", model, optimizer, epoch, max(best, score))
        if score >= best:
            best = score
            save_ckpt(CKPT / "best.pt", model, optimizer, epoch, best)
            np.save(ROOT / "val_embeddings.npy", _matrix)
            (ROOT / "val_emb_ids.json").write_text(json.dumps(names))
    elapsed = time.perf_counter() - started
    summary = {"minutes": round(elapsed / 60, 2), "best_map": best, "epochs": epochs, "resumed_from": start_epoch}
    (ROOT / "logs" / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
