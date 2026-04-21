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
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from torch_geometric.data import Batch

from src.graph.detector_feats import detector_feat_tensor
from src.graph.features import extract_features, feature_dim
from src.graph.graph_builder import build_grid_graph_fast
from src.graph.model_factory import build_gnn_classifier as build_model
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
    p.add_argument("--config",       default="configs/gnn_config.yaml", help="GNN config (late_fusion must match checkpoint)")
    p.add_argument("--features",     nargs="+", default=DEFAULT_FEATURES)
    p.add_argument("--gnn_batch_size", type=int, default=16)
    p.add_argument("--save_vis",     action="store_true")
    p.add_argument("--out_dir",      default="runs/inference")
    p.add_argument("--conf",         type=float, default=0.25)
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_gnn(weights: str, feature_types: list[str], device: str, config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    in_ch = feature_dim(feature_types)
    eval_cfg = {**cfg, "dropout": 0.0}
    model = build_model("sgcn", in_ch, num_classes=len(CLASS_NAMES), cfg=eval_cfg)
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    return model.to(device).eval()


def classify_crops(
    crops: list[np.ndarray],
    det_infos: list[dict],
    feature_types: list[str],
    model,
    device: str,
    batch_size: int = 16,
) -> list[int]:
    """Build graphs + optional detector vectors, batch-infer with SGCN (+ late fusion)."""
    predictions: list[tuple[int, float]] = []
    for start in range(0, len(crops), batch_size):
        graphs = []
        for crop, dinfo in zip(crops[start:start + batch_size], det_infos[start:start + batch_size]):
            feat = extract_features(crop, feature_types)
            g = build_grid_graph_fast(feat, connectivity=8)
            g.det_feat = detector_feat_tensor(
                dinfo["bbox_xyxy"],
                dinfo["image_hw"],
                dinfo["confidence"],
                dinfo["yolo_cls_id"],
            )
            graphs.append(g)

        batch = Batch.from_data_list(graphs).to(device)
        with torch.inference_mode():
            logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        pred_ids = logits.argmax(1)
        for i in range(pred_ids.shape[0]):
            pid = int(pred_ids[i].item())
            predictions.append((pid, float(probs[i, pid].item())))

    return predictions


def run(args):
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    detector = YOLO(args.yolo_weights)
    classifier = load_gnn(args.gnn_weights, args.features, args.device, args.config)

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

        crops, boxes, confs, det_infos = [], [], [], []
        ih, iw = img.shape[0], img.shape[1]
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
            det_infos.append({
                "bbox_xyxy": [x1, y1, x2, y2],
                "image_hw": (ih, iw),
                "confidence": float(box.conf.item()),
                "yolo_cls_id": int(box.cls.item()),
            })

        if not crops:
            continue

        id_and_gnn_conf = classify_crops(
            crops,
            det_infos,
            args.features,
            classifier,
            args.device,
            batch_size=args.gnn_batch_size,
        )

        img_results = []
        for (x1, y1, x2, y2), (cls_id, gnn_conf), det_conf in zip(
            boxes, id_and_gnn_conf, confs
        ):
            img_results.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "det_conf": det_conf,
                "gnn_conf": gnn_conf,
            })

        all_results.append({"image": str(img_path), "detections": img_results})

        if args.save_vis:
            vis = img.copy()
            for det in img_results:
                x1, y1, x2, y2 = det["bbox"]
                color = PALETTE[det["class_id"] % len(PALETTE)]
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                # det_conf = YOLO "this box is an object" score, not accuracy.
                # gnn_conf = softmax probability for the displayed fine-grained class.
                label = (
                    f"{det['class_name']} gnn:{det['gnn_conf']:.2f} det:{det['det_conf']:.2f}"
                )
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
