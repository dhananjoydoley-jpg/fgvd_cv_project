"""
src/graph/graph_builder.py
Convert a feature map (H×W×C) into a PyG Data object representing a 2D grid graph.

Paper §III-B-2:
  - Each pixel → node
  - Edges connect spatially adjacent nodes (4- or 8-neighbourhood)
  - Edge weights = Euclidean distance between pixel intensities (RGB channel)
"""

import numpy as np
import torch
from torch_geometric.data import Data


def build_grid_graph(
    feature_map: np.ndarray,          # H×W×C float32
    connectivity: int = 8,            # 4 or 8
    label: int | None = None,
) -> Data:
    """
    Build a PyG Data graph from a 2D feature map.

    Args:
        feature_map : H×W×C float32 array (output of features.extract_features).
        connectivity : 4 (N/S/E/W) or 8 (+ diagonals).
        label       : integer class label (optional).

    Returns:
        torch_geometric.data.Data with:
            x       : (H*W, C)  node features
            edge_index : (2, E) COO edge indices
            edge_attr  : (E, 1) Euclidean distance weights
            y          : scalar label (if provided)
    """
    H, W, C = feature_map.shape
    N = H * W

    # Flatten nodes: node id = row * W + col
    x = torch.tensor(
        feature_map.reshape(N, C), dtype=torch.float32
    )

    # Build adjacency in COO format
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),           ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ]
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    src_list, dst_list, weight_list = [], [], []

    for r in range(H):
        for c in range(W):
            u = r * W + c
            feat_u = feature_map[r, c]   # C-dim vector
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    v = nr * W + nc
                    feat_v = feature_map[nr, nc]
                    # Edge weight: Euclidean distance on RGB (first 3 channels) or all channels
                    dist = float(np.linalg.norm(feat_u[:3] - feat_v[:3]))
                    src_list.append(u)
                    dst_list.append(v)
                    weight_list.append(dist)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(weight_list, dtype=torch.float32).unsqueeze(1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.y = torch.tensor(label, dtype=torch.long)
    return data


def build_grid_graph_fast(
    feature_map: np.ndarray,
    connectivity: int = 8,
    label: int | None = None,
) -> Data:
    """
    Vectorised version of build_grid_graph — much faster for 64×64 maps.
    Uses numpy index arithmetic instead of nested Python loops.
    """
    H, W, C = feature_map.shape
    N = H * W

    x = torch.tensor(feature_map.reshape(N, C), dtype=torch.float32)

    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]

    rows = np.arange(H).repeat(W)
    cols = np.tile(np.arange(W), H)
    src_ids = rows * W + cols

    all_src, all_dst, all_w = [], [], []

    feat_flat = feature_map.reshape(N, C)

    for dr, dc in offsets:
        nr = rows + dr
        nc = cols + dc
        valid = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        s = src_ids[valid]
        d = (nr[valid] * W + nc[valid])

        diff = feat_flat[s, :3] - feat_flat[d, :3]
        w = np.linalg.norm(diff, axis=1)

        all_src.append(s)
        all_dst.append(d)
        all_w.append(w)

    src = np.concatenate(all_src)
    dst = np.concatenate(all_dst)
    edge_index = torch.from_numpy(np.vstack((src, dst))).long()
    edge_attr = torch.from_numpy(np.concatenate(all_w)).float().unsqueeze(1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.y = torch.tensor(label, dtype=torch.long)
    return data
