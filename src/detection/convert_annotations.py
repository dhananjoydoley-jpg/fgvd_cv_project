"""Convert VOC-style XML annotations to YOLO text labels.

The FGVD annotations in this workspace are stored as XML files under
data/raw/labels/{train,val,test}. This script converts them into YOLO label
files next to the XML files so Ultralytics can train from the same dataset
layout.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import xml.etree.ElementTree as ET


CLASS_TO_ID = {
    "car": 0,
    "motorcycle": 1,
    "scooter": 2,
    "truck": 3,
    "autorickshaw": 4,
    "bus": 5,
    "mini-bus": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-root",
        default="data/raw/labels",
        help="Root directory containing train/val/test annotation folders.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".xml", ".sml"],
        help="Annotation file extensions to convert.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print conversion statistics without writing label files.",
    )
    return parser.parse_args()


def normalize_bbox(xmin: float, ymin: float, xmax: float, ymax: float, width: float, height: float) -> tuple[float, float, float, float]:
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def convert_file(annotation_path: Path, dry_run: bool = False) -> tuple[int, int]:
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    size_node = root.find("size")
    if size_node is None:
        raise ValueError(f"Missing <size> node in {annotation_path}")

    width = float(size_node.findtext("width", default="0"))
    height = float(size_node.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {annotation_path}: {width}x{height}")

    label_lines: list[str] = []
    skipped = 0

    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip()
        class_name = raw_name.split("_", 1)[0]
        class_id = CLASS_TO_ID.get(class_name)
        if class_id is None:
            skipped += 1
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            skipped += 1
            continue

        xmin = float(bbox.findtext("xmin", default="0"))
        ymin = float(bbox.findtext("ymin", default="0"))
        xmax = float(bbox.findtext("xmax", default="0"))
        ymax = float(bbox.findtext("ymax", default="0"))

        x_center, y_center, box_width, box_height = normalize_bbox(xmin, ymin, xmax, ymax, width, height)
        label_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )

    label_path = annotation_path.with_suffix(".txt")
    if not dry_run:
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    return len(label_lines), skipped


def main() -> None:
    args = parse_args()
    labels_root = Path(args.labels_root)
    extensions = {ext if ext.startswith(".") else f".{ext}" for ext in args.extensions}

    annotation_files = [
        path for path in sorted(labels_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in extensions
    ]

    total_files = 0
    total_labels = 0
    total_skipped = 0

    for annotation_path in annotation_files:
        label_count, skipped = convert_file(annotation_path, dry_run=args.dry_run)
        total_files += 1
        total_labels += label_count
        total_skipped += skipped

    mode = "Dry run" if args.dry_run else "Conversion"
    print(
        f"{mode} complete: {total_files} annotation files processed, "
        f"{total_labels} labels written, {total_skipped} objects skipped."
    )


if __name__ == "__main__":
    main()