"""Train WaveST-Gate from a prepared real-data batch."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from wavestgate.data.assemble import BatchMetadata
from wavestgate.data.prepare_dataset import load_prepared_dataset
from wavestgate.evaluation.metrics import summarize_proportion_metrics
from wavestgate.models.types import WaveSTGateConfig
from wavestgate.models.types import WaveSTGateBatch, WaveSTGateOutput
from wavestgate.models.wavestgate import WaveSTGate
from wavestgate.training.losses import compute_losses
from wavestgate.training.train import load_config


def _infer_model_config(
    raw_model: dict,
    num_genes: int,
    num_cell_types: int,
    niche_gt: torch.Tensor | None = None,
) -> WaveSTGateConfig:
    values = dict(raw_model)
    values.setdefault("num_genes", num_genes)
    values.setdefault("num_cell_types", num_cell_types)
    if "niche_classes" not in values and niche_gt is not None and niche_gt.numel() > 0:
        values["niche_classes"] = int(niche_gt.max().item()) + 1
    config = WaveSTGateConfig.from_dict(values)
    if config.num_genes != num_genes:
        raise ValueError(f"model.num_genes={config.num_genes} does not match prepared data genes={num_genes}")
    if config.num_cell_types != num_cell_types:
        raise ValueError(
            f"model.num_cell_types={config.num_cell_types} does not match prepared data cell types={num_cell_types}"
        )
    return config


def _expression_rmse(predicted: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(F.mse_loss(torch.log1p(predicted.clamp_min(0.0)), torch.log1p(target.clamp_min(0.0)))).item()


def _niche_accuracy(logits: torch.Tensor | None, target: torch.Tensor | None) -> float | None:
    if logits is None or target is None:
        return None
    if target.ndim == 2:
        target = target.argmax(dim=1)
    return (logits.argmax(dim=1) == target.long()).float().mean().item()


def _slice_optional(value: torch.Tensor | None, indices: torch.Tensor) -> torch.Tensor | None:
    return value.index_select(0, indices) if value is not None else None


def _slice_edges(
    edge_index: torch.Tensor | None,
    edge_weight: torch.Tensor | None,
    indices: torch.Tensor,
    n_total: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if edge_index is None or edge_index.numel() == 0:
        return None, None
    device = edge_index.device
    indices = indices.to(device=device)
    lookup = torch.full((n_total,), -1, dtype=torch.long, device=device)
    lookup[indices] = torch.arange(indices.numel(), device=device)
    src = lookup[edge_index[0].long()]
    dst = lookup[edge_index[1].long()]
    keep = (src >= 0) & (dst >= 0)
    if not keep.any():
        return torch.empty(2, 0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.float32, device=device)
    sliced_edge_index = torch.stack([src[keep], dst[keep]], dim=0)
    sliced_edge_weight = edge_weight[keep] if edge_weight is not None else None
    return sliced_edge_index, sliced_edge_weight


def _slice_batch(batch: WaveSTGateBatch, indices: torch.Tensor) -> WaveSTGateBatch:
    n_total = batch.st_expression.size(0)
    edge_index, edge_weight = _slice_edges(batch.edge_index, batch.edge_weight, indices, n_total)
    return WaveSTGateBatch(
        image_patches=batch.image_patches.index_select(0, indices),
        st_expression=batch.st_expression.index_select(0, indices),
        reference_prototypes=batch.reference_prototypes,
        coords=_slice_optional(batch.coords, indices),
        edge_index=edge_index,
        edge_weight=edge_weight,
        proportion_gt=_slice_optional(batch.proportion_gt, indices),
        niche_gt=_slice_optional(batch.niche_gt, indices),
    )


def _slice_metadata(metadata: BatchMetadata, indices: torch.Tensor) -> BatchMetadata:
    idx = indices.detach().cpu().tolist()
    return BatchMetadata(
        spot_ids=[metadata.spot_ids[int(i)] for i in idx],
        gene_names=metadata.gene_names,
        cell_types=metadata.cell_types,
    )


def _as_split_list(value) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _split_indices(
    metadata: BatchMetadata,
    split_path: str | Path | None,
    splits,
    device: torch.device,
    *,
    label: str,
) -> torch.Tensor:
    requested = _as_split_list(splits)
    if requested is None:
        return torch.arange(len(metadata.spot_ids), device=device)
    if split_path is None:
        raise ValueError(f"{label} splits were requested but data.split_path is missing")
    split_frame = pd.read_csv(split_path)
    required = {"spot_id", "split"}
    if not required.issubset(split_frame.columns):
        raise ValueError(f"Split file must contain columns {sorted(required)}")
    split_frame["spot_id"] = split_frame["spot_id"].astype(str)
    split_by_spot = split_frame.set_index("spot_id")["split"].astype(str)
    selected = [
        idx
        for idx, spot_id in enumerate(metadata.spot_ids)
        if str(split_by_spot.get(str(spot_id), "")) in set(requested)
    ]
    if not selected:
        raise ValueError(f"No spots matched {label} splits: {requested}")
    return torch.as_tensor(selected, dtype=torch.long, device=device)


def _concat_optional(values: list[torch.Tensor | None]) -> torch.Tensor | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return torch.cat(present, dim=0)


def _concat_outputs(outputs: list[WaveSTGateOutput]) -> WaveSTGateOutput:
    return WaveSTGateOutput(
        proportions=torch.cat([output.proportions for output in outputs], dim=0),
        reconstructed_expression=torch.cat([output.reconstructed_expression for output in outputs], dim=0),
        gate_weights=torch.cat([output.gate_weights for output in outputs], dim=0),
        agent_attention=torch.cat([output.agent_attention for output in outputs], dim=0),
        image_feature=torch.cat([output.image_feature for output in outputs], dim=0),
        expression_feature=torch.cat([output.expression_feature for output in outputs], dim=0),
        agent_feature=torch.cat([output.agent_feature for output in outputs], dim=0),
        fused_feature=torch.cat([output.fused_feature for output in outputs], dim=0),
        niche_logits=_concat_optional([output.niche_logits for output in outputs]),
        raw_gate_weights=_concat_optional([output.raw_gate_weights for output in outputs]),
        spot_uncertainty=_concat_optional([output.spot_uncertainty for output in outputs]),
        modality_uncertainty=_concat_optional([output.modality_uncertainty for output in outputs]),
        modality_reliability=_concat_optional([output.modality_reliability for output in outputs]),
        morphology_feature=_concat_optional([output.morphology_feature for output in outputs]),
    )


def _run_model_batched(model: WaveSTGate, batch: WaveSTGateBatch, batch_size: int) -> WaveSTGateOutput:
    outputs = []
    n_spots = batch.st_expression.size(0)
    for start in range(0, n_spots, batch_size):
        indices = torch.arange(start, min(start + batch_size, n_spots), device=batch.st_expression.device)
        outputs.append(model(_slice_batch(batch, indices)))
    return _concat_outputs(outputs)


def _write_predictions(
    output,
    metadata,
    predictions_path: str | Path | None,
    gates_path: str | Path | None,
    uncertainty_path: str | Path | None = None,
    modality_reliability_path: str | Path | None = None,
    niche_logits_path: str | Path | None = None,
    reconstructed_expression_path: str | Path | None = None,
    agent_attention_path: str | Path | None = None,
    raw_gates_path: str | Path | None = None,
) -> None:
    if predictions_path is not None:
        predictions_path = Path(predictions_path)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        prop_df = pd.DataFrame(
            output.proportions.detach().cpu().numpy(),
            index=metadata.spot_ids,
            columns=metadata.cell_types,
        )
        prop_df.to_csv(predictions_path, index_label="spot_id")

    if gates_path is not None:
        gates_path = Path(gates_path)
        gates_path.parent.mkdir(parents=True, exist_ok=True)
        gate_df = pd.DataFrame(
            output.gate_weights.detach().cpu().numpy(),
            index=metadata.spot_ids,
            columns=["image", "expression", "reference"],
        )
        gate_df.to_csv(gates_path, index_label="spot_id")

    if raw_gates_path is not None and output.raw_gate_weights is not None:
        raw_gates_path = Path(raw_gates_path)
        raw_gates_path.parent.mkdir(parents=True, exist_ok=True)
        raw_gate_df = pd.DataFrame(
            output.raw_gate_weights.detach().cpu().numpy(),
            index=metadata.spot_ids,
            columns=["image", "expression", "reference"],
        )
        raw_gate_df.to_csv(raw_gates_path, index_label="spot_id")

    if uncertainty_path is not None and output.spot_uncertainty is not None:
        uncertainty_path = Path(uncertainty_path)
        uncertainty_path.parent.mkdir(parents=True, exist_ok=True)
        uncertainty_df = pd.DataFrame(
            output.spot_uncertainty.detach().cpu().reshape(-1).numpy(),
            index=metadata.spot_ids,
            columns=["spot_uncertainty"],
        )
        uncertainty_df.to_csv(uncertainty_path, index_label="spot_id")

    if modality_reliability_path is not None and output.modality_reliability is not None:
        modality_reliability_path = Path(modality_reliability_path)
        modality_reliability_path.parent.mkdir(parents=True, exist_ok=True)
        reliability_df = pd.DataFrame(
            output.modality_reliability.detach().cpu().numpy(),
            index=metadata.spot_ids,
            columns=["image", "expression", "reference"],
        )
        reliability_df.to_csv(modality_reliability_path, index_label="spot_id")

    if niche_logits_path is not None and output.niche_logits is not None:
        niche_logits_path = Path(niche_logits_path)
        niche_logits_path.parent.mkdir(parents=True, exist_ok=True)
        columns = [f"niche_{idx}" for idx in range(output.niche_logits.size(1))]
        niche_df = pd.DataFrame(
            output.niche_logits.detach().cpu().numpy(),
            index=metadata.spot_ids,
            columns=columns,
        )
        niche_df.to_csv(niche_logits_path, index_label="spot_id")

    if reconstructed_expression_path is not None:
        reconstructed_expression_path = Path(reconstructed_expression_path)
        reconstructed_expression_path.parent.mkdir(parents=True, exist_ok=True)
        expression_df = pd.DataFrame(
            output.reconstructed_expression.detach().cpu().numpy(),
            index=metadata.spot_ids,
            columns=metadata.gene_names,
        )
        expression_df.to_csv(reconstructed_expression_path, index_label="spot_id")

    if agent_attention_path is not None:
        agent_attention_path = Path(agent_attention_path)
        agent_attention_path.parent.mkdir(parents=True, exist_ok=True)
        agent_df = pd.DataFrame(
            output.agent_attention.detach().cpu().numpy(),
            index=metadata.spot_ids,
            columns=metadata.cell_types,
        )
        agent_df.to_csv(agent_attention_path, index_label="spot_id")


def train_real_from_config(path: str | Path) -> dict[str, float]:
    """Train on a prepared real-data batch and return final metrics."""

    raw = load_config(path)
    data_cfg = raw.get("data", {})
    if "prepared_path" not in data_cfg:
        raise ValueError("train_real config requires data.prepared_path")

    train_cfg = raw.get("training", {})
    seed = int(train_cfg.get("seed", 0))
    torch.manual_seed(seed)
    requested_device = train_cfg.get("device", "cpu")
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")

    batch, metadata, extra = load_prepared_dataset(data_cfg["prepared_path"], device=device)
    init_checkpoint_path = train_cfg.get("init_checkpoint")
    raw_model = dict(raw.get("model", {}))
    init_checkpoint = None
    if init_checkpoint_path is not None:
        init_checkpoint = torch.load(init_checkpoint_path, map_location=device, weights_only=False)
        checkpoint_config = dict(init_checkpoint.get("config", {}))
        checkpoint_config.update(raw_model)
        raw_model = checkpoint_config
    model_config = _infer_model_config(
        raw_model,
        num_genes=batch.st_expression.size(1),
        num_cell_types=batch.reference_prototypes.size(0),
        niche_gt=batch.niche_gt,
    )
    model = WaveSTGate(model_config).to(device)
    if init_checkpoint is not None:
        model.load_state_dict(
            init_checkpoint["model_state_dict"],
            strict=bool(train_cfg.get("init_strict", True)),
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    split_path = data_cfg.get("split_path")
    train_indices = _split_indices(
        metadata,
        split_path,
        train_cfg.get("train_splits", data_cfg.get("train_splits")),
        device,
        label="training",
    )
    eval_cfg = raw.get("evaluation", {})
    eval_indices = _split_indices(
        metadata,
        split_path,
        eval_cfg.get("eval_splits", train_cfg.get("eval_splits", data_cfg.get("eval_splits"))),
        device,
        label="evaluation",
    )
    training_source = _slice_batch(batch, train_indices)
    eval_batch = _slice_batch(batch, eval_indices)
    eval_metadata = _slice_metadata(metadata, eval_indices)

    steps = int(train_cfg.get("steps", 100))
    batch_size = int(train_cfg.get("batch_size", training_source.st_expression.size(0)))
    eval_batch_size = int(train_cfg.get("eval_batch_size", batch_size))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    history = []
    n_train_spots = training_source.st_expression.size(0)
    n_eval_spots = eval_batch.st_expression.size(0)
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if batch_size >= n_train_spots:
            train_batch = training_source
        else:
            indices = torch.randperm(n_train_spots, device=device)[:batch_size]
            train_batch = _slice_batch(training_source, indices)
        output = model(train_batch)
        loss, loss_dict = compute_losses(output, train_batch, model_config)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        row = {"step": float(step)}
        row.update({key: value.item() for key, value in loss_dict.items()})
        history.append(row)

    model.eval()
    with torch.no_grad():
        output = _run_model_batched(model, eval_batch, eval_batch_size) if eval_batch_size < n_eval_spots else model(eval_batch)
        metrics = {"expression_log1p_rmse": _expression_rmse(output.reconstructed_expression, eval_batch.st_expression)}
        if eval_batch.proportion_gt is not None:
            metrics.update(summarize_proportion_metrics(output.proportions, eval_batch.proportion_gt, output.spot_uncertainty))
        niche_accuracy = _niche_accuracy(output.niche_logits, eval_batch.niche_gt)
        if niche_accuracy is not None:
            metrics["niche_accuracy"] = niche_accuracy
        metrics["num_train_spots"] = float(n_train_spots)
        metrics["num_eval_spots"] = float(n_eval_spots)

    checkpoint_path = Path(train_cfg.get("checkpoint_path", "real_checkpoint.pt"))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model_config.__dict__,
            "metrics": metrics,
            "metadata": {
                "spot_ids": metadata.spot_ids,
                "gene_names": metadata.gene_names,
                "cell_types": metadata.cell_types,
            },
            "prepared_extra": extra,
            "training": {
                "prepared_path": str(data_cfg["prepared_path"]),
                "split_path": str(split_path) if split_path is not None else "",
                "train_splits": _as_split_list(train_cfg.get("train_splits", data_cfg.get("train_splits"))),
                "eval_splits": _as_split_list(
                    eval_cfg.get("eval_splits", train_cfg.get("eval_splits", data_cfg.get("eval_splits")))
                ),
                "init_checkpoint": str(init_checkpoint_path) if init_checkpoint_path is not None else "",
                "num_train_spots": int(n_train_spots),
                "num_eval_spots": int(n_eval_spots),
            },
        },
        checkpoint_path,
    )

    predictions_cfg = raw.get("predictions", {})
    _write_predictions(
        output,
        eval_metadata,
        predictions_cfg.get("proportions_path"),
        predictions_cfg.get("gates_path"),
        uncertainty_path=predictions_cfg.get("uncertainty_path"),
        modality_reliability_path=predictions_cfg.get("modality_reliability_path"),
        niche_logits_path=predictions_cfg.get("niche_logits_path"),
        reconstructed_expression_path=predictions_cfg.get("reconstructed_expression_path"),
        agent_attention_path=predictions_cfg.get("agent_attention_path"),
        raw_gates_path=predictions_cfg.get("raw_gates_path"),
    )

    metrics_path = Path(train_cfg.get("metrics_path", "real_metrics.csv"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    final_row = {**history[-1], **metrics}
    history_path_value = train_cfg.get("history_path")
    if history_path_value is not None:
        history_path = Path(history_path_value)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[-1].keys()))
            writer.writeheader()
            writer.writerows(history)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_row.keys()))
        writer.writeheader()
        writer.writerow(final_row)
    return final_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WaveST-Gate from a prepared real-data batch.")
    parser.add_argument("--config", required=True, help="Path to train_real YAML/JSON config.")
    args = parser.parse_args()
    metrics = train_real_from_config(args.config)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
