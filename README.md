# FGVD CV project

Two-stage pipeline: **YOLOv8-nano** detection → **SGCN/GAT** graph classification on FGVD-style vehicle crops.

This README lists the **commands that reproduce the workflow**, including the **exact arguments** captured in `runs/evaluation/pipeline_eval_saved.txt` for the reported **test-set pipeline accuracy** (multiclass and binary cases).

---

## What is “accuracy” here?

End-of-pipeline metrics come from **`src/training/evaluate.py`**, which runs YOLO on the test images, builds graphs from crops, runs the GNN, then prints sklearn-style metrics.

A fixed snapshot of one run is in:

- `runs/evaluation/pipeline_eval_saved.txt`

That file records (among other things):

- **Accuracy (multiclass): 65.59%**
- **Precision / Recall / F1** and per-class breakdown
- **Binary cases** (two-wheelers vs all, etc.)

---

## 0. Shell and project root

Commands assume the repository root (adjust the path if yours differs):

```bash
cd "/home/cse/Documents/cv project (Copy)"
```

Your session used a virtualenv (prompt showed `(.venv)`). Typical activation:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 1. Install dependencies

Follow the **install order** in `requirements.txt` (PyTorch first, then the rest; optional PyG wheels if you use CUDA).

**GPU (example from comments in `requirements.txt`, CUDA 12.4 wheels):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**CPU-only:**

```bash
pip install torch torchvision
pip install -r requirements.txt
```

---

## 2. Prepare YOLO labels from VOC XML (if not already done)

```bash
python src/detection/convert_annotations.py
```

Uses defaults: `--labels-root data/raw/labels` and writes YOLO `.txt` labels next to the XML layout under `data/raw/labels/{train,val,test}`.

---

## 3. Stage 1 — Train YOLO (`configs/detection.yaml`)

```bash
python src/detection/train_yolo.py
```

Optional resume:

```bash
python src/detection/train_yolo.py --config configs/detection.yaml --resume runs/detect/<your_run>/weights/last.pt
```

Training hyperparameters and Ultralytics `project` / `name` come from `configs/detection.yaml`. The **weights path used in the saved evaluation** (below) is:

`runs/detect/runs/detect/fgvd_yolov8n8/weights/best.pt`

If your new run writes under a different folder, pass that `best.pt` into the next steps instead.

---

## 4. Build crops + `crop_metadata.json` for train/val (late fusion)

`configs/gnn_config.yaml` enables **late fusion** and expects detector metadata JSON for train/val. After you have YOLO `best.pt`, generate crops for **train** and **val** image roots:

```bash
python src/detection/infer_yolo.py \
  --weights runs/detect/runs/detect/fgvd_yolov8n8/weights/best.pt \
  --source data/raw/images/train \
  --out data/processed/crops/train \
  --device 0

python src/detection/infer_yolo.py \
  --weights runs/detect/runs/detect/fgvd_yolov8n8/weights/best.pt \
  --source data/raw/images/val \
  --out data/processed/crops/val \
  --device 0
```

(`--device` follows `infer_yolo.py`: default is `"0"` for first GPU; use `cpu` if needed.)

---

## 5. Stage 2 — Train the GNN

```bash
python src/training/train_gnn.py --device cuda
```

Other documented variants from the script header:

```bash
python src/training/train_gnn.py --config configs/gnn_config.yaml --model gat
python src/training/train_gnn.py --resume runs/gnn/fgvd_sgcn/last.pt
```

Checkpoints are written under `runs/gnn/<run_name>/` per `project` / `name` in `configs/gnn_config.yaml` (overridable with `--project` / `--name`). The **weights file referenced by the saved evaluation** is:

`runs/gnn/fgvd_sgcn/best.pt`

---

## 6. Full-pipeline test evaluation (this produces the “accuracy” report)

These are the **arguments recorded** in `runs/evaluation/pipeline_eval_saved.txt` (manual snapshot, 2026-04-21) for the metrics quoted there:

```bash
python src/training/evaluate.py \
  --yolo_weights runs/detect/runs/detect/fgvd_yolov8n8/weights/best.pt \
  --gnn_weights runs/gnn/fgvd_sgcn/best.pt \
  --test_images data/raw/images/test \
  --config configs/gnn_config.yaml \
  --device cuda \
  --out_report runs/evaluation/pipeline_eval_latest.txt
```

Notes:

- `--model` defaults to `sgcn` (matches the saved snapshot workflow unless you override it).
- Stage 1 inside this script writes crops under `--crops_out` (default `data/processed/crops/test`) and uses `conf=0.25`, `iou=0.45` as implemented in `evaluate.py`.

To save a second copy for a paper or handoff, point `--out_report` to a new filename (e.g. `runs/evaluation/pipeline_eval_saved.txt`).

---

## 7. Optional — YOLO-only batch predict (seen in your terminal log)

Your terminal session showed Ultralytics **predict** over **1083** test images at **640×640**, with results saved under `runs/detect/predict`. The equivalent CLI is typically:

```bash
yolo predict \
  model=runs/detect/runs/detect/fgvd_yolov8n8/weights/best.pt \
  source=data/raw/images/test \
  project=runs/detect \
  name=predict
```

This is **detection inference**, not the GNN pipeline accuracy in §6.

---

## 8. Optional — Export YOLO to ONNX

You have `best.onnx` next to `best.pt`; the matching helper is:

```bash
python src/detection/export_model.py \
  --weights runs/detect/runs/detect/fgvd_yolov8n8/weights/best.pt
```

---

## Honest note on “every command you ever ran”

This README ties together:

1. **Exact `evaluate.py` flags** from `runs/evaluation/pipeline_eval_saved.txt` (authoritative for the quoted accuracy).
2. **The intended training / crop / install sequence** from the repo’s scripts and configs.
3. **One recovered inference command** consistent with your terminal output (`runs/detect/predict`, 1083 test images).

Shell history in the captured Cursor terminal did **not** include the full original `train_yolo.py` / `train_gnn.py` invocations; if you used extra flags (`--name`, different `--config`, etc.), add them here alongside your run folders under `runs/detect/` and `runs/gnn/`.
