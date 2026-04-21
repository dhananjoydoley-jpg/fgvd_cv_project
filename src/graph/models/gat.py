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


class GATClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout

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
        return self.classifier(x)
