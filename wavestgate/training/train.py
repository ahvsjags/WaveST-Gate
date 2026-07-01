"""Toy training CLI for the WaveST-Gate MVP."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from wavestgate.data.synthetic import make_synthetic_batch
from wavestgate.evaluation.metrics import summarize_proportion_metrics
from wavestgate.models.types import WaveSTGateConfig
from wavestgate.models.wavestgate import WaveSTGate
from wavestgate.training.losses import compute_losses


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        loaded = yaml.safe_load(text)
    except Exception:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError("Config must parse to a dictionary")
    return loaded


def train_from_config(path: str | Path) -> dict[str, float]:
    raw = load_config(path)
    model_config = WaveSTGateConfig.from_dict(raw.get("model", {}))
    train_cfg = raw.get("training", {})
    seed = int(train_cfg.get("seed", 0))
    torch.manual_seed(seed)

    requested_device = train_cfg.get("device", "cpu")
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
    model = WaveSTGate(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("learning_rate", 1e-3)))
    batch = make_synthetic_batch(
        model_config,
        batch_size=int(train_cfg.get("batch_size", 8)),
        seed=seed,
        device=device,
    )

    steps = int(train_cfg.get("steps", 5))
    history = []
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss, loss_dict = compute_losses(output, batch, model_config)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        optimizer.step()
        row = {"step": float(step)}
        row.update({key: value.item() for key, value in loss_dict.items()})
        history.append(row)

    model.eval()
    with torch.no_grad():
        output = model(batch)
        metrics = summarize_proportion_metrics(output.proportions, batch.proportion_gt, output.spot_uncertainty)

    checkpoint_path = Path(train_cfg.get("checkpoint_path", "checkpoint.pt"))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model_config.__dict__,
            "metrics": metrics,
        },
        checkpoint_path,
    )

    metrics_path = Path(train_cfg.get("metrics_path", "metrics.csv"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(history[-1].keys()) + list(metrics.keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        final_row = history[-1].copy()
        final_row.update(metrics)
        writer.writerow(final_row)

    return {**history[-1], **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WaveST-Gate on a synthetic MVP batch.")
    parser.add_argument("--config", required=True, help="Path to JSON-compatible YAML config.")
    args = parser.parse_args()
    metrics = train_from_config(args.config)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
