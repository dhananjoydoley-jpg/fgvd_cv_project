"""
src/utils/dataset.py
PyTorch Dataset that reads 64×64 crops, extracts features, and builds grid graphs.
Supports on-the-fly or pre-cached graph construction.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.graph.features import extract_features
from src.graph.graph_builder import build_grid_graph_fast


CLASS_NAMES = ["car", "motorcycle", "scooter", "truck", "autorickshaw", "bus"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


class FGVDGraphDataset(Dataset):
    """
    Dataset for Stage 2 classification.

    Two init modes:
      1. From a metadata JSON (produced by infer_yolo.py) — for test inference.
      2. From a crops directory structured as class/image.png — for training.

    Args:
        root         : path to crops directory (class subdirs).
        feature_types: list of feature names, e.g. ['rgb', 'gabor', 'sobel'].
        connectivity : 4 or 8 for grid graph neighbourhood.
        cache_dir    : if provided, save/load pre-built .pt graph files here.
        metadata_json: if provided, use this file instead of scanning root.
    """

    def __init__(
        self,
        root: str,
        feature_types: list[str] = ("rgb", "gabor", "sobel"),
        connectivity: int = 8,
        cache_dir: str | None = None,
        metadata_json: str | None = None,
    ):
        self.root = Path(root)
        self.feature_types = list(feature_types)
        self.connectivity = connectivity
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Build index: list of (img_path, label_idx)
        if metadata_json:
            with open(metadata_json) as f:
                meta = json.load(f)
            self.samples = [
                (Path(m["crop_path"]), CLASS_TO_IDX[m["class_name"]])
                for m in meta
            ]
        else:
            self.samples = []
            for cls_name, cls_idx in CLASS_TO_IDX.items():
                cls_dir = self.root / cls_name
                if not cls_dir.exists():
                    continue
                for img_file in cls_dir.glob("*.png"):
                    self.samples.append((img_file, cls_idx))
                for img_file in cls_dir.glob("*.jpg"):
                    self.samples.append((img_file, cls_idx))

        print(f"[FGVDGraphDataset] {len(self.samples)} samples from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        img_path, label = self.samples[idx]

        # Try cache first
        if self.cache_dir:
            cache_key = f"{img_path.stem}_{'_'.join(self.feature_types)}_c{self.connectivity}.pt"
            cache_file = self.cache_dir / cache_key
            if cache_file.exists():
                try:
                    return torch.load(cache_file, weights_only=False, map_location="cpu")
                except TypeError:
                    return torch.load(cache_file, map_location="cpu")

        # Load and build graph
        img = cv2.imread(str(img_path))
        assert img is not None, f"Could not read {img_path}"

        feat = extract_features(img, self.feature_types)      # 64×64×C
        graph = build_grid_graph_fast(feat, self.connectivity, label=label)

        if self.cache_dir:
            torch.save(graph, cache_file)

        return graph
