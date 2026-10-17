"""Small ImageNet embedding for Blitz player crops."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Embedder(nn.Module):
    def __init__(self, backbone: str = "resnet18") -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")
        width = int(self.backbone.num_features)
        self.fc = nn.Linear(width, 128)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(self.backbone(images)), dim=1)


def batch_hard_triplet(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    dist = torch.cdist(emb, emb, p=2)
    same = labels.view(-1, 1).eq(labels.view(1, -1))
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    pos = dist.masked_fill(~same | eye, -1.0)
    neg = dist.masked_fill(same, 1e6)
    hardest_pos, _ = pos.max(dim=1)
    hardest_neg, _ = neg.min(dim=1)
    valid = hardest_pos > 0
    if int(valid.sum()) == 0:
        return emb.sum() * 0.0
    return F.relu(hardest_pos[valid] - hardest_neg[valid] + margin).mean()
