"""
src/training/train_gnn.py
Stage 2 — Train the SGCN (or GAT) classifier on FGVD crop graphs.

Usage:
    python src/training/train_gnn.py
    python src/training/train_gnn.py --device cuda
    python src/training/train_gnn.py --config configs/gnn_config.yaml --model gat
    python src/training/train_gnn.py --resume runs/gnn/fgvd_sgcn/last.pt
"""

import argparse
import time
from contextlib import nullcontext
from pathlib import Path
import sys

import yaml
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.utils.dataset import FGVDGraphDataset
from src.graph.features import feature_dim
from src.graph.models.sgcn import SGCN
from src.graph.models.gat import GATClassifier
from src.utils.metrics import compute_metrics, print_summary

CLASS_NAMES = ["car", "motorcycle", "scooter", "truck", "autorickshaw", "bus"]


def resolve_training_device(device: str) -> str:
    """
    Normalize CLI device choice. Use --device cuda to require a GPU (fails loudly if
    PyTorch cannot see CUDA). Use auto to prefer CUDA when available.
    """
    raw = (device or "auto").strip()
    d = raw.lower()

    def _require_cuda() -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build, verify drivers with nvidia-smi, "
                "and run outside environments that hide the GPU (or pass --device cpu)."
            )

    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d == "cpu":
        return "cpu"
    if d == "gpu":
        _require_cuda()
        return "cuda"
    if d == "cuda" or d.startswith("cuda:"):
        _require_cuda()
        return raw
    raise ValueError(f"Unknown --device {device!r} (expected auto, cuda, cuda:N, gpu, or cpu)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/gnn_config.yaml")
    p.add_argument("--model", default="sgcn", choices=["sgcn", "gat"])
    p.add_argument("--epochs", type=int, default=None, help="Override config epochs for a shorter smoke test")
    p.add_argument("--project", default=None, help="Override the output project directory")
    p.add_argument("--name", default=None, help="Override the run name")
    p.add_argument("--max_train_samples", type=int, default=None, help="Limit train samples for a quick smoke test")
    p.add_argument("--max_val_samples", type=int, default=None, help="Limit validation samples for a quick smoke test")
    p.add_argument("--train_crops", default="data/processed/crops/train")
    p.add_argument("--val_crops",   default="data/processed/crops/val")
    p.add_argument("--cache_dir",   default="data/processed/graphs")
    p.add_argument(
        "--device",
        default="auto",
        help="auto (prefer CUDA), cuda (require GPU), or cpu",
    )
    p.add_argument("--resume",      default=None, help="Path to a checkpoint to resume from, preferably last.pt")
    return p.parse_args()


def build_model(model_name: str, in_channels: int, num_classes: int, cfg: dict):
    if model_name == "sgcn":
        return SGCN(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_channels=cfg.get("hidden_channels", 64),
            num_layers=cfg.get("num_layers", 3),
            dropout=cfg.get("dropout", 0.5),
        )
    else:  # gat
        return GATClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_channels=cfg.get("hidden_channels", 64),
            num_layers=cfg.get("num_layers", 3),
            dropout=cfg.get("dropout", 0.5),
        )


def maybe_subset_dataset(dataset, max_items: int | None, seed: int = 0):
    if max_items is None or max_items <= 0 or max_items >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_items].tolist()
    return Subset(dataset, indices)


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_training_checkpoint(checkpoint_path: Path, model, optimizer, scaler, device: str, use_amp: bool):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            move_optimizer_state_to_device(optimizer, device)

        if use_amp and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_acc = float(checkpoint.get("best_acc", 0.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        return start_epoch, best_acc, best_epoch

    # Backward compatibility: allow resuming from an old plain state_dict.
    model.load_state_dict(checkpoint)
    return 1, 0.0, 0


def train_one_epoch(model, loader, optimizer, criterion, device, accumulation_steps=1, use_amp=False, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(loader)

    for step, batch in enumerate(loader, start=1):
        batch = batch.to(device)

        amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
        with amp_context:
            logits = model(batch)
            loss = criterion(logits, batch.y)
            scaled_loss = loss / accumulation_steps

        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if step % accumulation_steps == 0 or step == num_batches:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * batch.num_graphs
        correct += (logits.argmax(1) == batch.y).sum().item()
        total += batch.num_graphs
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        all_pred.extend(logits.argmax(1).cpu().tolist())
        all_true.extend(batch.y.cpu().tolist())
    return compute_metrics(all_true, all_pred, CLASS_NAMES)


def train(args):
    args.device = resolve_training_device(args.device)
    if str(args.device).startswith("cuda"):
        print(f"[train] Using GPU: {torch.cuda.get_device_name(0)} ({args.device})")
    else:
        print(f"[train] Using device: {args.device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    feature_types = cfg.get("features", ["rgb", "gabor", "sobel"])
    connectivity   = cfg.get("graph_connectivity", "8-neighbour")
    conn = 8 if "8" in str(connectivity) else 4
    target_epochs = int(args.epochs) if args.epochs is not None else int(cfg["epochs"])

    # Datasets
    train_ds = FGVDGraphDataset(args.train_crops, feature_types, conn,
                                cache_dir=args.cache_dir + "/train")
    val_ds   = FGVDGraphDataset(args.val_crops,   feature_types, conn,
                                cache_dir=args.cache_dir + "/val")

    train_ds = maybe_subset_dataset(train_ds, args.max_train_samples, seed=0)
    val_ds = maybe_subset_dataset(val_ds, args.max_val_samples, seed=1)
    print(f"[FGVDGraphDataset] using {len(train_ds)} train samples and {len(val_ds)} val samples")

    num_workers = int(cfg.get("num_workers", 0))
    accumulation_steps = max(1, int(cfg.get("accumulation_steps", 1)))
    use_amp = bool(cfg.get("amp", True)) and str(args.device).startswith("cuda") and torch.cuda.is_available()
    pin_memory = str(args.device).startswith("cuda")
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    in_ch = feature_dim(feature_types)
    model = build_model(args.model, in_ch, num_classes=len(CLASS_NAMES), cfg=cfg)
    model = model.to(args.device)
    print(f"\n🔧 Model: {args.model.upper()} | in_channels={in_ch} | params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg["lr"],
                                 weight_decay=cfg.get("weight_decay", 0.0))
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_project = args.project if args.project is not None else cfg.get("project", "runs/gnn")
    run_name = args.name if args.name is not None else cfg.get("name", "fgvd_sgcn")
    if args.name is None and run_name == "fgvd_sgcn" and args.model != "sgcn":
        run_name = f"fgvd_{args.model}"

    save_dir = Path(run_project) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_period = int(cfg.get("save_period", 0))
    start_epoch = 1
    best_acc, best_epoch = 0.0, 0

    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch, best_acc, best_epoch = load_training_checkpoint(
            resume_path,
            model,
            optimizer,
            scaler,
            args.device,
            use_amp,
        )
        print(f"↩️  Resumed from {resume_path} at epoch {start_epoch - 1} | best_acc={best_acc*100:.2f}%")

    if start_epoch > target_epochs:
        print(f"Checkpoint epoch {start_epoch - 1} is already past the requested target of {target_epochs} epochs.")

    for epoch in range(start_epoch, target_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            args.device,
            accumulation_steps=accumulation_steps,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_metrics = evaluate(model, val_loader, args.device)
        val_acc = val_metrics["accuracy"]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{target_epochs} | "
            f"loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_acc={val_acc*100:.2f}% | {elapsed:.1f}s"
        )

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), save_dir / "best.pt")
            print(f"   💾 Saved best checkpoint (val_acc={best_acc*100:.2f}%)")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "model_name": args.model,
            "feature_types": feature_types,
            "connectivity": conn,
            "cfg": cfg,
        }
        torch.save(checkpoint, save_dir / "last.pt")

        if save_period > 0 and epoch % save_period == 0:
            torch.save(checkpoint, save_dir / f"checkpoint_epoch_{epoch:03d}.pt")

    print(f"\n✅ Training done. Best val_acc={best_acc*100:.2f}% at epoch {best_epoch}")
    print(f"   Weights: {save_dir / 'best.pt'}")
    print(f"   Last checkpoint: {save_dir / 'last.pt'}")

    # Final val report
    model.load_state_dict(torch.load(save_dir / "best.pt", map_location="cpu"))
    final = evaluate(model, val_loader, args.device)
    print_summary(final)


if __name__ == "__main__":
    train(parse_args())
