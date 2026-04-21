"""
Detector-side features for late fusion with the graph embedding.

Vector (default 9-D): log relative box area, log aspect ratio, YOLO confidence,
one-hot over K YOLO class ids (FGVD uses 6 vehicle classes).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import torch


DEFAULT_DET_FEAT_DIM = 9


def fusion_model_kwargs(cfg: dict) -> dict:
    """Keyword args for SGCN / GATClassifier late-fusion head."""
    lf = cfg.get("late_fusion") or {}
    if not lf.get("enabled", False):
        return {"det_feat_dim": 0, "fusion_hidden": int(lf.get("fusion_hidden", 64))}
    return {
        "det_feat_dim": int(lf.get("det_feat_dim", DEFAULT_DET_FEAT_DIM)),
        "fusion_hidden": int(lf.get("fusion_hidden", 64)),
    }


def detector_feat_tensor(
    bbox_xyxy: list[float] | list[int],
    image_hw: tuple[int, int],
    confidence: float,
    yolo_cls_id: int,
    num_classes: int = 6,
) -> torch.Tensor:
    """Return shape [1, 3 + num_classes] float32 on CPU."""
    x1, y1, x2, y2 = (float(t) for t in bbox_xyxy)
    H, W = float(image_hw[0]), float(image_hw[1])
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    box_area = bw * bh
    img_area = max(H * W, 1.0)
    rel_area = box_area / img_area
    aspect = bw / (bh + 1e-6)
    fa = math.log(rel_area + 1e-8)
    fasp = math.log(aspect + 1e-8)
    fconf = float(confidence)
    oh = [0.0] * num_classes
    if 0 <= int(yolo_cls_id) < num_classes:
        oh[int(yolo_cls_id)] = 1.0
    vec = [fa, fasp, fconf] + oh
    return torch.tensor(vec, dtype=torch.float32).view(1, -1)


def build_crop_path_to_det_feat(
    metadata: list[dict[str, Any]],
    num_classes: int = 6,
) -> dict[str, torch.Tensor]:
    """
    Map resolved crop_path string -> det_feat tensor [1, D].
    Fills image_hw from metadata or reads each unique source_image once.
    """
    hw_cache: dict[str, tuple[int, int]] = {}
    out: dict[str, torch.Tensor] = {}

    for m in metadata:
        crop_path = str(Path(m["crop_path"]).resolve())
        if "image_height" in m and "image_width" in m:
            hw = (int(m["image_height"]), int(m["image_width"]))
        else:
            src = str(Path(m["source_image"]).resolve())
            if src not in hw_cache:
                im = cv2.imread(src)
                if im is None:
                    hw_cache[src] = (1, 1)
                else:
                    hw_cache[src] = (im.shape[0], im.shape[1])
            hw = hw_cache[src]

        bbox = m["bbox_xyxy"]
        conf = float(m["confidence"])
        cid = int(m["class_id"])
        out[crop_path] = detector_feat_tensor(bbox, hw, conf, cid, num_classes=num_classes)

    return out
