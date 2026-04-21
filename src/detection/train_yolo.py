"""
src/detection/train_yolo.py
Stage 1 — Train YOLOv8-nano on the FGVD dataset.

Usage:
    python src/detection/train_yolo.py
    python src/detection/train_yolo.py --config configs/detection.yaml --resume runs/detect/fgvd_yolov8n/weights/last.pt
"""

import argparse
from pathlib import Path

import yaml
import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/detection.yaml")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def resolve_device(device_value):
    if device_value is None:
        return "cpu"

    if isinstance(device_value, str):
        device_value = device_value.strip()
        if device_value.lower() == "cpu":
            return "cpu"
        if not torch.cuda.is_available():
            return "cpu"
        return device_value

    if isinstance(device_value, int):
        if torch.cuda.is_available() and device_value < torch.cuda.device_count():
            return device_value
        return "cpu"

    return "cpu"


def train(cfg_path: str, resume: str | None = None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Load YOLOv8-nano (pretrained on COCO; fine-tune on FGVD)
    model_weights = resume if resume else cfg["model"]
    if resume is not None and not Path(resume).is_file():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {resume}. "
            "Pass a valid `last.pt` path or omit --resume to start from the base model."
        )

    model = YOLO(model_weights)
    device = resolve_device(cfg.get("device", 0))

    results = model.train(
        data=cfg["data"],
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=device,
        workers=cfg.get("workers", 2),
        amp=cfg.get("amp", True) and device != "cpu",
        optimizer=cfg.get("optimizer", "AdamW"),
        lr0=cfg.get("lr0", 0.001),
        lrf=cfg.get("lrf", 0.01),
        momentum=cfg.get("momentum", 0.937),
        weight_decay=cfg.get("weight_decay", 0.0005),
        warmup_epochs=cfg.get("warmup_epochs", 3.0),
        warmup_momentum=cfg.get("warmup_momentum", 0.8),
        warmup_bias_lr=cfg.get("warmup_bias_lr", 0.1),
        hsv_h=cfg.get("hsv_h", 0.015),
        hsv_s=cfg.get("hsv_s", 0.7),
        hsv_v=cfg.get("hsv_v", 0.4),
        degrees=cfg.get("degrees", 0.0),
        translate=cfg.get("translate", 0.1),
        scale=cfg.get("scale", 0.5),
        flipud=cfg.get("flipud", 0.0),
        fliplr=cfg.get("fliplr", 0.5),
        mosaic=cfg.get("mosaic", 1.0),
        mixup=cfg.get("mixup", 0.1),
        project=cfg.get("project", "runs/detect"),
        name=cfg.get("name", "fgvd_yolov8n"),
        save=cfg.get("save", True),
        save_period=cfg.get("save_period", 10),
        resume=resume is not None,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n✅ Training complete. Best weights saved to: {best_weights}")
    return best_weights


if __name__ == "__main__":
    args = parse_args()
    train(args.config, args.resume)
