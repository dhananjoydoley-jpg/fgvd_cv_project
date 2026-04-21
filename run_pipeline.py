"""
run_pipeline.py
End-to-end inference: given an image (or directory), detect vehicles with
YOLOv8-nano and classify them with SGCN.

Usage:
    python run_pipeline.py \
        --yolo_weights runs/detect/fgvd_yolov8n/weights/best.pt \
        --gnn_weights  runs/gnn/fgvd_sgcn/best.pt \
        --source       path/to/image_or_dir \
        [--save_vis]   # draw bounding boxes + class labels
"""

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torch_geometric.data import Batch

from src.graph.features import extract_features, feature_dim
from src.graph.graph_builder import build_grid_graph_fast
from src.graph.models.sgcn import SGCN
from src.utils.dataset import CLASS_NAMES

CROP_SIZE = 64
DEFAULT_FEATURES = ["rgb", "gabor", "sobel"]
PALETTE = [
    (220, 50, 50), (50, 180, 50), (50, 50, 220),
    (200, 130, 0), (140, 60, 200), (0, 180, 200),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yolo_weights", required=True)
    p.add_argument("--gnn_weights",  required=True)
    p.add_argument("--source",       required=True)
    p.add_argument("--features",     nargs="+", default=DEFAULT_FEATURES)
    p.add_argument("--gnn_batch_size", type=int, default=16)
    p.add_argument("--save_vis",     action="store_true")
    p.add_argument("--out_dir",      default="runs/inference")
    p.add_argument("--conf",         type=float, default=0.25)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_gnn(weights: str, feature_types: list[str], device: str) -> SGCN:
    in_ch = feature_dim(feature_types)
    model = SGCN(in_ch, num_classes=len(CLASS_NAMES), dropout=0.0)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.to(device).eval()


def classify_crops(crops: list[np.ndarray], feature_types: list[str],
                   model: SGCN, device: str, batch_size: int = 16) -> list[int]:
    """Convert a list of BGR crops to graph batch and run GNN."""
    predictions = []
    for start in range(0, len(crops), batch_size):
        graphs = []
        for crop in crops[start:start + batch_size]:
            feat = extract_features(crop, feature_types)
            graphs.append(build_grid_graph_fast(feat, connectivity=8))

        batch = Batch.from_data_list(graphs).to(device)
        with torch.inference_mode():
            logits = model(batch)
        predictions.extend(logits.argmax(1).cpu().tolist())

    return predictions


def run(args):
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    detector = YOLO(args.yolo_weights)
    classifier = load_gnn(args.gnn_weights, args.features, args.device)

    source = Path(args.source).expanduser()
    if not source.exists():
        raise FileNotFoundError(
            f"Source not found: {source}\n"
            "Pass a real image path or folder (not the placeholder /full/path/...)."
        )
    img_paths = (
        [source] if source.is_file()
        else sorted(source.glob("**/*.jpg")) + sorted(source.glob("**/*.png"))
    )
    if not img_paths:
        raise FileNotFoundError(
            f"No images found under: {source}\n"
            "Use a .jpg/.png file, or a directory that contains images."
        )

    all_results = []

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        det_results = detector.predict(str(img_path), conf=args.conf, verbose=False)[0]

        crops, boxes, confs = [], [], []
        for box in det_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(cv2.resize(crop, (CROP_SIZE, CROP_SIZE)))
            boxes.append([x1, y1, x2, y2])
            confs.append(float(box.conf.item()))

        if not crops:
            continue

        class_ids = classify_crops(crops, args.features, classifier, args.device, batch_size=args.gnn_batch_size)

        img_results = []
        for (x1, y1, x2, y2), cls_id, conf in zip(boxes, class_ids, confs):
            img_results.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "det_conf": conf,
            })

        all_results.append({"image": str(img_path), "detections": img_results})

        if args.save_vis:
            vis = img.copy()
            for det in img_results:
                x1, y1, x2, y2 = det["bbox"]
                color = PALETTE[det["class_id"] % len(PALETTE)]
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                label = f"{det['class_name']} {det['det_conf']:.2f}"
                cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.imwrite(str(out_path / img_path.name), vis)

    # Save JSON results
    results_file = out_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    total = sum(len(r["detections"]) for r in all_results)
    print(f"\n✅ Processed {len(img_paths)} image(s), {total} detection(s).")
    print(f"   Results → {results_file}")
    if args.save_vis:
        print(f"   Visualisations → {out_path}")


if __name__ == "__main__":
    run(parse_args())
