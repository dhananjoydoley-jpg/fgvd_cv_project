"""Construct SGCN / GAT from configs (shared by train, evaluate, run_pipeline)."""

from src.graph.detector_feats import fusion_model_kwargs
from src.graph.models.gat import GATClassifier
from src.graph.models.sgcn import SGCN


def build_gnn_classifier(model_name: str, in_channels: int, num_classes: int, cfg: dict):
    fk = fusion_model_kwargs(cfg)
    if model_name == "sgcn":
        return SGCN(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_channels=cfg.get("hidden_channels", 64),
            num_layers=cfg.get("num_layers", 3),
            dropout=cfg.get("dropout", 0.5),
            **fk,
        )
    if model_name == "gat":
        return GATClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_channels=cfg.get("hidden_channels", 64),
            num_layers=cfg.get("num_layers", 3),
            dropout=cfg.get("dropout", 0.5),
            **fk,
        )
    raise ValueError(f"Unknown model_name {model_name!r}")
