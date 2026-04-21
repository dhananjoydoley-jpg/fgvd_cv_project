"""
src/training/evaluate.py
Evaluate the full two-stage pipeline on FGVD test set.
Reproduces the binary classification cases from Tables II–V:
  - Two-wheelers vs All
  - Three-wheelers vs All
  - Two+Three-wheelers vs All
  - All classes (multiclass)

Usage:
    python src/training/evaluate.py \
        --yolo_weights runs/detect/<your_yolo_run>/weights/best.pt \
        --gnn_weights  runs/gnn/fgvd_sgcn/best.pt \
        --test_images  data/raw/images/test \
        --config       configs/gnn_config.yaml

    # Tip: find YOLO best.pt with: find runs/detect -name best.pt
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch
import yaml
from torch_geometric.loader import DataLoader

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.detection.infer_yolo import extract_crops
from src.graph.detector_feats import DEFAULT_DET_FEAT_DIM
from src.utils.dataset import FGVDGraphDataset
from src.graph.features import feature_dim
from src.graph.model_factory import build_gnn_classifier as build_model
from src.utils.metrics import compute_metrics, binary_accuracy, print_summary

# Two-wheeler class ids: motorcycle (1), scooter (2)
TWO_WHEELER_IDS  = [1, 2]
# Three-wheeler: autorickshaw (4)
THREE_WHEELER_IDS = [4]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yolo_weights", required=True)
    p.add_argument("--gnn_weights",  required=True)
    p.add_argument("--test_images",  required=True)
    p.add_argument("--config",       default="configs/gnn_config.yaml")
    p.add_argument("--model",        default="sgcn", choices=["sgcn", "gat"])
    p.add_argument("--crops_out",    default="data/processed/crops/test")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out_report",
        default="runs/evaluation/pipeline_eval_latest.txt",
        help="Path to write the full text report (parent dirs created as needed).",
    )
    return p.parse_args()


def run_pipeline(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    feature_types = cfg.get("features", ["rgb", "gabor", "sobel"])
    conn = 8 if "8" in str(cfg.get("graph_connectivity", "8")) else 4

    # ── Stage 1: Detect & crop ───────────────────────────────────────────
    print("\n🔍 Stage 1: YOLOv8-nano detection…")
    extract_crops(
        weights=args.yolo_weights,
        source=args.test_images,
        out_dir=args.crops_out,
        conf=0.25, iou=0.45, device=args.device,
    )

    # ── Stage 2: GNN classification ──────────────────────────────────────
    print(f"\n📐 Stage 2: {args.model.upper()} classification…")
    lf = cfg.get("late_fusion") or {}
    det_dim = int(lf.get("det_feat_dim", DEFAULT_DET_FEAT_DIM)) if lf.get("enabled", False) else 0
    meta_file = Path(args.crops_out) / "crop_metadata.json"
    if meta_file.is_file():
        test_ds = FGVDGraphDataset(
            args.crops_out,
            feature_types,
            conn,
            cache_dir="data/processed/graphs/test",
            metadata_json=str(meta_file),
            det_feat_dim=det_dim,
            detector_metadata_json=str(meta_file),
        )
    else:
        test_ds = FGVDGraphDataset(
            args.crops_out,
            feature_types,
            conn,
            cache_dir="data/processed/graphs/test",
            det_feat_dim=det_dim,
            detector_metadata_json=None,
        )
    num_workers = int(cfg.get("num_workers", 0))
    loader = DataLoader(
        test_ds,
        batch_size=cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=num_workers > 0,
    )

    in_ch = feature_dim(feature_types)
    eval_cfg = {**cfg, "dropout": 0.0}
    model = build_model(args.model, in_ch, num_classes=6, cfg=eval_cfg)

    model.load_state_dict(torch.load(args.gnn_weights, map_location="cpu"), strict=True)
    model = model.to(args.device).eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(args.device)
            logits = model(batch)
            all_pred.extend(logits.argmax(1).cpu().tolist())
            all_true.extend(batch.y.cpu().tolist())

    # ── Report ───────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  FGVD Test Results — Full Multiclass (L-1)")
    print("="*55)
    metrics = compute_metrics(all_true, all_pred)
    print_summary(metrics)

    print("\n" + "="*55)
    print("  Binary Cases")
    print("="*55)
    acc_2w   = binary_accuracy(all_true, all_pred, TWO_WHEELER_IDS)
    acc_3w   = binary_accuracy(all_true, all_pred, THREE_WHEELER_IDS)
    acc_23w  = binary_accuracy(all_true, all_pred, TWO_WHEELER_IDS + THREE_WHEELER_IDS)
    print(f"  Two-wheelers vs All          : {acc_2w*100:.2f}%")
    print(f"  Three-wheelers vs All        : {acc_3w*100:.2f}%")
    print(f"  Two+Three-wheelers vs All    : {acc_23w*100:.2f}%")

    report_path = Path(args.out_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"FGVD two-stage pipeline evaluation",
        f"Written: {stamp}",
        "",
        "Arguments:",
        f"  yolo_weights: {args.yolo_weights}",
        f"  gnn_weights:  {args.gnn_weights}",
        f"  test_images:  {args.test_images}",
        f"  config:       {args.config}",
        f"  model:        {args.model}",
        f"  crops_out:    {args.crops_out}",
        f"  device:       {args.device}",
        f"  features:     {feature_types}",
        "",
        f"Samples (graphs): {len(test_ds)}",
        "",
        "=" * 55,
        "  FGVD Test Results — Full Multiclass",
        "=" * 55,
        "",
        f"  Accuracy : {metrics['accuracy']*100:.2f}%",
        f"  Precision: {metrics['precision']*100:.2f}%",
        f"  Recall   : {metrics['recall']*100:.2f}%",
        f"  F1       : {metrics['f1']*100:.2f}%",
        "",
        metrics["report"],
        "",
        "=" * 55,
        "  Binary Cases",
        "=" * 55,
        f"  Two-wheelers vs All          : {acc_2w*100:.2f}%",
        f"  Three-wheelers vs All        : {acc_3w*100:.2f}%",
        f"  Two+Three-wheelers vs All    : {acc_23w*100:.2f}%",
        "",
    ]
    text = "\n".join(lines)
    report_path.write_text(text, encoding="utf-8")
    print(f"\n📄 Report saved to: {report_path.resolve()}")


if __name__ == "__main__":
    run_pipeline(parse_args())
