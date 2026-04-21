"""
src/graph/models/sgcn.py
Spatial Graph Convolutional Network (SGCN) — paper §III-B-3.

Reference:
    Danel et al., "Spatial graph convolutional networks,"
    Neural Information Processing, Springer, 2020.

SGCN layer update rule (eq. 3):
    H' = D̂^{-1/2} Â D̂^{-1/2} H W + b

where Â incorporates both connectivity and spatial proximity weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops, degree

from src.graph.late_fusion import LateFusionMLP


class SGCNConv(MessagePassing):
    """
    One SGCN layer: spatially-weighted graph convolution.

    The edge weights (spatial distances) stored in edge_attr are used to
    scale messages, implementing the spatial weighting in Â.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, edge_attr=None):
        # Add self-loops
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr.squeeze() if edge_attr is not None else None,
            fill_value=0.0, num_nodes=x.size(0)
        )

        # Symmetric normalisation: D̂^{-1/2} Â D̂^{-1/2}
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Incorporate spatial distance weights from edge_attr
        if edge_attr is not None:
            # Invert distance so closer neighbours get higher weight
            spatial_w = torch.exp(-edge_attr)          # shape: (E,)
            norm = norm * spatial_w

        x = self.lin(x)
        out = self.propagate(edge_index, x=x, norm=norm)
        return out + self.bias

    def message(self, x_j, norm):
        return norm.unsqueeze(-1) * x_j


class SGCN(nn.Module):
    """
    Full SGCN classifier for FGVD object classification.

    Architecture (paper §IV):
        • N × SGCNConv(in_ch → 64) + ReLU + Dropout
        • Global mean pool
        • Linear(64 → num_classes)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.5,
        det_feat_dim: int = 0,
        fusion_hidden: int = 64,
    ):
        super().__init__()
        self.dropout = dropout
        self.det_feat_dim = int(det_feat_dim)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.convs.append(SGCNConv(in_ch, hidden_channels))

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
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else \
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)

        det_feat = getattr(data, "det_feat", None)
        if self.det_feat_dim > 0 and self.fusion is not None:
            return self.fusion(x, det_feat)
        return self.classifier(x)
