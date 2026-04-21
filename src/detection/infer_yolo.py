"""
src/detection/infer_yolo.py
Stage 1 — Run YOLOv8-nano on FGVD images and save 64×64 crops for Stage 2.

Usage:
    python src/detection/infer_yolo.py \
        --weights runs/detect/fgvd_yolov8n/weights/best.pt \
        --source data/raw/images/test \
        --out data/processed/crops
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

CROP_SIZE = 64  # paper §IV: resized to 64×64 before feature extraction

CLASS_NAMES = {
    0: "car",
    1: "motorcycle",
    2: "scooter",
    3: "truck",
    4: "autorickshaw",
    5: "bus",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to YOLOv8-nano best.pt")
    p.add_argument("--source", required=True, help="Directory of input images")
    p.add_argument("--out", default="data/processed/crops")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--device", default="0")
    return p.parse_args()


def extract_crops(weights: str, source: str, out_dir: str,
                  conf: float, iou: float, device: str):
    wpath = Path(weights)
    if not wpath.is_file():
        raise FileNotFoundError(
            f"YOLO weights not found: {wpath.resolve()}\n"
            "Train Stage 1 (YOLO) first, or pass a valid --yolo_weights path "
            "to best.pt (search under runs/detect/ for your run folder)."
        )
    model = YOLO(str(wpath))
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Create per-class subdirectories
    for name in CLASS_NAMES.values():
        (out_path / name).mkdir(exist_ok=True)

    metadata: list[dict] = []
    img_paths = list(Path(source).glob("**/*.jpg")) + list(Path(source).glob("**/*.png"))

    for img_path in tqdm(img_paths, desc="Detecting & cropping"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )[0]

        h, w = img.shape[:2]
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                                      interpolation=cv2.INTER_AREA)

            cls_name = CLASS_NAMES[cls_id]
            crop_name = f"{img_path.stem}_det{i:03d}.png"
            crop_save = out_path / cls_name / crop_name
            cv2.imwrite(str(crop_save), crop_resized)

            metadata.append({
                "source_image": str(img_path),
                "crop_path": str(crop_save),
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": confidence,
                "bbox_xyxy": [x1, y1, x2, y2],
                "image_height": h,
                "image_width": w,
            })

    meta_file = out_path / "crop_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Extracted {len(metadata)} crops → {out_path}")
    print(f"   Metadata saved to: {meta_file}")
    return metadata


if __name__ == "__main__":
    args = parse_args()
    extract_crops(args.weights, args.source, args.out, args.conf, args.iou, args.device)
