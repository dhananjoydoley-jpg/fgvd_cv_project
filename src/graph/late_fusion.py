"""MLP head that fuses pooled graph embedding with detector feature vectors."""

import torch
import torch.nn as nn


class LateFusionMLP(nn.Module):
    def __init__(
        self,
        graph_dim: int,
        det_dim: int,
        num_classes: int,
        hidden: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.det_dim = det_dim
        self.net = nn.Sequential(
            nn.Linear(graph_dim + det_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, graph_emb: torch.Tensor, det_feat: torch.Tensor | None) -> torch.Tensor:
        if det_feat is None:
            det_feat = torch.zeros(
                graph_emb.size(0), self.det_dim, device=graph_emb.device, dtype=graph_emb.dtype
            )
        if det_feat.dim() == 1:
            det_feat = det_feat.unsqueeze(0)
        return self.net(torch.cat([graph_emb, det_feat], dim=-1))
