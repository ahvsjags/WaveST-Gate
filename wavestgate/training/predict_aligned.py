"""Cross-dataset prediction with checkpoint gene/cell-type alignment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from wavestgate.data.assemble import BatchMetadata
from wavestgate.data.prepare_dataset import load_prepared_dataset
from wavestgate.models.types import WaveSTGateBatch, WaveSTGateConfig
from wavestgate.models.wavestgate import WaveSTGate
from wavestgate.training.train_real import _slice_batch


def _load_checkpoint_model(checkpoint_path: str | Path, device: torch.device) -> tuple[WaveSTGate, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = WaveSTGateConfig.from_dict(checkpoint["config"])
    model = WaveSTGate(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _align_matrix(
    values: torch.Tensor,
    source_names: list[str],
    target_names: list[str],
    dim: int,
) -> torch.Tensor:
    if dim not in {0, 1}:
        raise ValueError("dim must be 0 or 1")
    source_lookup = {name: idx for idx, name in enumerate(source_names)}
    shape = list(values.shape)
    shape[dim] = len(target_names)
    aligned = values.new_zeros(shape)
    for target_idx, name in enumerate(target_names):
        source_idx = source_lookup.get(name)
        if source_idx is None:
            continue
        if dim == 0:
            aligned[target_idx] = values[source_idx]
        else:
            aligned[:, target_idx] = values[:, source_idx]
    return aligned


def align_batch_to_checkpoint(
    batch: WaveSTGateBatch,
    metadata: BatchMetadata,
    checkpoint: dict,
) -> tuple[WaveSTGateBatch, list[str], list[str], list[str]]:
    ckpt_meta = checkpoint.get("metadata", {})
    target_genes = list(ckpt_meta.get("gene_names", []))
    target_cell_types = list(ckpt_meta.get("cell_types", []))
    if not target_genes or not target_cell_types:
        raise ValueError("Checkpoint metadata must include gene_names and cell_types")

    expression = _align_matrix(batch.st_expression, metadata.gene_names, target_genes, dim=1)
    reference_by_cell = _align_matrix(batch.reference_prototypes, metadata.cell_types, target_cell_types, dim=0)
    reference = _align_matrix(reference_by_cell, metadata.gene_names, target_genes, dim=1)
    missing_genes = [gene for gene in target_genes if gene not in set(metadata.gene_names)]
    missing_cell_types = [cell_type for cell_type in target_cell_types if cell_type not in set(metadata.cell_types)]
    aligned_batch = WaveSTGateBatch(
        image_patches=batch.image_patches,
        st_expression=expression,
        reference_prototypes=reference,
        coords=batch.coords,
        edge_index=batch.edge_index,
        edge_weight=batch.edge_weight,
        proportion_gt=None,
        niche_gt=None,
    )
    return aligned_batch, target_genes, target_cell_types, missing_genes + missing_cell_types


def _predict_batched_to_cpu(
    model: WaveSTGate,
    batch: WaveSTGateBatch,
    batch_size: int,
) -> dict[str, torch.Tensor | None]:
    """Run memory-bounded inference and retain only exportable spot-level outputs."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    chunks: dict[str, list[torch.Tensor]] = {
        "proportions": [],
        "gates": [],
        "uncertainty": [],
        "agent_attention": [],
        "modality_reliability": [],
        "niche_logits": [],
    }
    squared_error = 0.0
    error_elements = 0
    n_spots = batch.st_expression.size(0)

    with torch.inference_mode():
        for start in range(0, n_spots, batch_size):
            indices = torch.arange(start, min(start + batch_size, n_spots), device=batch.st_expression.device)
            chunk = _slice_batch(batch, indices)
            output = model(chunk)
            chunks["proportions"].append(output.proportions.detach().cpu())
            chunks["gates"].append(output.gate_weights.detach().cpu())
            chunks["agent_attention"].append(output.agent_attention.detach().cpu())
            if output.spot_uncertainty is not None:
                chunks["uncertainty"].append(output.spot_uncertainty.detach().reshape(-1).cpu())
            if output.modality_reliability is not None:
                chunks["modality_reliability"].append(output.modality_reliability.detach().cpu())
            if output.niche_logits is not None:
                chunks["niche_logits"].append(output.niche_logits.detach().cpu())

            squared_error += float(F.mse_loss(
                torch.log1p(output.reconstructed_expression.clamp_min(0.0)),
                torch.log1p(chunk.st_expression.clamp_min(0.0)),
                reduction="sum",
            ).item())
            error_elements += output.reconstructed_expression.numel()
            del output, chunk
            if batch.st_expression.device.type == "cuda":
                torch.cuda.empty_cache()

    values: dict[str, torch.Tensor | None] = {
        key: torch.cat(value, dim=0) if value else None
        for key, value in chunks.items()
    }
    values["expression_log1p_rmse"] = torch.tensor((squared_error / max(error_elements, 1)) ** 0.5)
    return values


def predict_aligned(
    checkpoint_path: str | Path,
    prepared_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict[str, object]:
    run_device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    model, checkpoint = _load_checkpoint_model(checkpoint_path, run_device)
    batch, metadata, extra = load_prepared_dataset(prepared_path, device=run_device)
    aligned_batch, target_genes, target_cell_types, missing = align_batch_to_checkpoint(batch, metadata, checkpoint)
    output = _predict_batched_to_cpu(model, aligned_batch, batch_size)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    proportions = output["proportions"]
    gates = output["gates"]
    uncertainty = output["uncertainty"]
    attention = output["agent_attention"]
    if proportions is None or gates is None or uncertainty is None or attention is None:
        raise RuntimeError("WaveST-Gate did not return the required prediction outputs")

    pd.DataFrame(proportions.numpy(), index=metadata.spot_ids, columns=target_cell_types).to_csv(
        output_dir / "predicted_proportions.csv",
        index_label="spot_id",
    )
    pd.DataFrame(gates.numpy(), index=metadata.spot_ids, columns=["image", "expression", "reference"]).to_csv(
        output_dir / "gate_weights.csv",
        index_label="spot_id",
    )
    pd.DataFrame(uncertainty.numpy(), index=metadata.spot_ids, columns=["spot_uncertainty"]).to_csv(
        output_dir / "spot_uncertainty.csv",
        index_label="spot_id",
    )
    pd.DataFrame(attention.numpy(), index=metadata.spot_ids, columns=target_cell_types).to_csv(
        output_dir / "agent_attention.csv",
        index_label="spot_id",
    )
    modality_reliability = output["modality_reliability"]
    if modality_reliability is not None:
        pd.DataFrame(modality_reliability.numpy(), index=metadata.spot_ids, columns=["image", "expression", "reference"]).to_csv(
            output_dir / "modality_reliability.csv",
            index_label="spot_id",
        )
    niche_logits = output["niche_logits"]
    if niche_logits is not None:
        pd.DataFrame(niche_logits.numpy(), index=metadata.spot_ids).to_csv(
            output_dir / "niche_logits.csv",
            index_label="spot_id",
        )
    metrics = {
        "prepared_path": str(prepared_path),
        "num_spots": len(metadata.spot_ids),
        "num_target_genes": len(target_genes),
        "num_missing_alignment_items": len(missing),
        "batch_size": batch_size,
        "device": str(run_device),
        "expression_log1p_rmse": float(output["expression_log1p_rmse"].item()),
        "mean_spot_uncertainty": float(uncertainty.mean().item()),
        "mean_image_gate": float(gates[:, 0].mean().item()),
        "mean_expression_gate": float(gates[:, 1].mean().item()),
        "mean_reference_gate": float(gates[:, 2].mean().item()),
    }
    with (output_dir / "aligned_prediction_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "prepared_path": str(prepared_path),
        "output_dir": str(output_dir),
        "missing_alignment_items": missing,
        "prepared_extra": extra,
        "metrics": metrics,
    }
    (output_dir / "aligned_prediction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run no-retuning cross-dataset WaveST-Gate prediction.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    metrics = predict_aligned(args.checkpoint, args.prepared, args.output_dir, args.batch_size, args.device)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
