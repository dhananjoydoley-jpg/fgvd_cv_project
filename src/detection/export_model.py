"""
src/detection/export_model.py
Export a trained YOLOv8-nano checkpoint to ONNX or TensorRT.

Usage:
    # ONNX (default)
    python src/detection/export_model.py --weights runs/detect/fgvd_yolov8n/weights/best.pt

    # TensorRT FP16
    python src/detection/export_model.py \
        --weights runs/detect/fgvd_yolov8n/weights/best.pt \
        --format engine --half
"""

import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument(
        "--format",
        default="onnx",
        choices=["onnx", "engine", "torchscript", "openvino"],
        help="Export format",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", help="FP16 (TRT only)")
    p.add_argument("--dynamic", action="store_true", help="Dynamic axes (ONNX)")
    p.add_argument("--simplify", action="store_true", help="Simplify ONNX graph")
    return p.parse_args()


def export(args):
    model = YOLO(args.weights)
    out = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )
    print(f"✅ Model exported to: {out}")


if __name__ == "__main__":
    export(parse_args())
