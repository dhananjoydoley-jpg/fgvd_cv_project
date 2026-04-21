"""
src/graph/models/gat.py
Graph Attention Network (GAT) — improved alternative to SGCN.

Motivation: GAT learns adaptive attention weights over neighbours,
potentially capturing more discriminative spatial context than the
fixed distance-weighting in SGCN.

Architecture:
    • N × GATConv(in_ch → hidden, heads=4) + ELU + Dropout
    • Global mean pool
    • Linear(hidden → num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

from src.graph.late_fusion import LateFusionMLP


class GATClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.5,
        det_feat_dim: int = 0,
        fusion_hidden: int = 64,
    ):
        super().__init__()
        self.dropout = dropout
        self.det_feat_dim = int(det_feat_dim)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels * heads
            # Last layer: single head, concat=False
            if i == num_layers - 1:
                self.convs.append(
                    GATConv(in_ch, hidden_channels, heads=1, concat=False, dropout=dropout)
                )
            else:
                self.convs.append(
                    GATConv(in_ch, hidden_channels, heads=heads, concat=True, dropout=dropout)
                )

        if self.det_feat_dim > 0:
            self.fusion = LateFusionMLP(
                graph_dim=hidden_channels,
                det_dim=self.det_feat_dim,
                num_classes=num_classes,
                hidden=fusion_hidden,
                dropout=dropout,
            )
            self.classifier = None
        else:
            self.fusion = None
            self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else \
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        det_feat = getattr(data, "det_feat", None)
        if self.det_feat_dim > 0 and self.fusion is not None:
            return self.fusion(x, det_feat)
        return self.classifier(x)
