"""Robustness stress tests for trained WaveST-Gate checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import torch

from wavestgate.data.prepare_dataset import load_prepared_dataset
from wavestgate.evaluation.metrics import summarize_proportion_metrics
from wavestgate.evaluation.nature_analysis import MARKER_SETS
from wavestgate.training.predict_aligned import _load_checkpoint_model, align_batch_to_checkpoint
from wavestgate.training.train_real import _run_model_batched


GENE_DROPOUT_RATES = [0.1, 0.3, 0.5]
GENE_PANEL_SIZES = [200, 100]
PROTOTYPE_NOISE_LEVELS = [0.05, 0.10]
PROTOTYPE_DROPOUT_RATES = [0.10, 0.30]
STAIN_PERTURBATIONS = {
    "stain_darken": (0.75, 0.0),
    "stain_brighten": (1.25, 0.0),
    "he_noise": (1.0, 0.08),
}


def _evaluate(model, batch, target, batch_size: int) -> dict[str, float]:
    with torch.no_grad():
        output = _run_model_batched(model, batch, batch_size)
    metrics = summarize_proportion_metrics(output.proportions.cpu(), target.cpu(), output.spot_uncertainty.cpu())
    metrics["mean_spot_uncertainty"] = float(output.spot_uncertainty.mean().item())
    metrics["mean_image_gate"] = float(output.gate_weights[:, 0].mean().item())
    metrics["mean_expression_gate"] = float(output.gate_weights[:, 1].mean().item())
    metrics["mean_reference_gate"] = float(output.gate_weights[:, 2].mean().item())
    del output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def _evaluate_output_subset(output, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    if mask.device != target.device:
        mask = mask.to(target.device)
    metrics = summarize_proportion_metrics(
        output.proportions.detach().cpu()[mask.cpu()],
        target.detach().cpu()[mask.cpu()],
        output.spot_uncertainty.detach().cpu()[mask.cpu()],
    )
    metrics["num_subset_spots"] = float(mask.sum().item())
    return metrics


def _clone_batch(batch):
    return type(batch)(
        image_patches=batch.image_patches.clone(),
        st_expression=batch.st_expression.clone(),
        reference_prototypes=batch.reference_prototypes.clone(),
        coords=batch.coords,
        edge_index=batch.edge_index,
        edge_weight=batch.edge_weight,
        proportion_gt=batch.proportion_gt,
        niche_gt=batch.niche_gt,
    )


def _append_row(rows: list[dict[str, float | str]], scenario: str, level: str, metrics: dict[str, float], **extra: float | str) -> None:
    row: dict[str, float | str] = {"scenario": scenario, "level": level, **extra}
    row.update(metrics)
    rows.append(row)


def _top_gene_indices(batch, panel_size: int) -> torch.Tensor:
    variances = batch.st_expression.float().var(dim=0)
    n_keep = min(int(panel_size), int(batch.st_expression.size(1)))
    return torch.topk(variances, k=n_keep).indices


def _marker_gene_indices(gene_names: list[str], device: torch.device) -> torch.Tensor:
    marker_genes = {gene for genes in MARKER_SETS.values() for gene in genes}
    indices = [idx for idx, gene in enumerate(gene_names) if gene in marker_genes]
    return torch.as_tensor(indices, dtype=torch.long, device=device)


def _zero_non_panel_genes(batch, keep_idx: torch.Tensor) -> None:
    mask = torch.zeros(batch.st_expression.size(1), dtype=torch.bool, device=batch.st_expression.device)
    mask[keep_idx] = True
    batch.st_expression[:, ~mask] = 0.0
    batch.reference_prototypes[:, ~mask] = 0.0


def _load_split_table(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    split_path = Path(path)
    if not split_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(split_path)
    if "spot_id" not in frame.columns or "split" not in frame.columns:
        return pd.DataFrame()
    frame["spot_id"] = frame["spot_id"].astype(str)
    frame["split"] = frame["split"].astype(str)
    return frame


def _load_cell_counts(path: str | Path | None, spot_ids: list[str]) -> pd.Series:
    if path is None or not Path(path).exists():
        return pd.Series(dtype=float)
    frame = pd.read_csv(path)
    if "spot_id" not in frame.columns:
        return pd.Series(dtype=float)
    frame["spot_id"] = frame["spot_id"].astype(str)
    values = frame.set_index("spot_id").drop(columns=[], errors="ignore")
    numeric = values.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return numeric.sum(axis=1).reindex(spot_ids).fillna(0.0)


def run_robustness(
    checkpoint_path: str | Path,
    prepared_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 512,
    device: str = "cuda",
    splits_path: str | Path | None = None,
    xenium_counts_path: str | Path | None = None,
) -> list[dict[str, float | str]]:
    run_device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    model, checkpoint = _load_checkpoint_model(checkpoint_path, run_device)
    raw_batch, metadata, _ = load_prepared_dataset(prepared_path, device=run_device)
    if raw_batch.proportion_gt is None:
        raise ValueError("Robustness evaluation requires proportion_gt")
    batch, _, _, missing = align_batch_to_checkpoint(raw_batch, metadata, checkpoint)
    target = raw_batch.proportion_gt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    _append_row(rows, "clean", "0", _evaluate(model, batch, target, batch_size))

    generator = torch.Generator(device=run_device).manual_seed(2026)
    n_genes = batch.st_expression.size(1)
    for rate in GENE_DROPOUT_RATES:
        perturbed = _clone_batch(batch)
        n_drop = max(int(round(n_genes * rate)), 1)
        drop_idx = torch.randperm(n_genes, generator=generator, device=run_device)[:n_drop]
        perturbed.st_expression[:, drop_idx] = 0.0
        _append_row(rows, "gene_dropout", str(rate), _evaluate(model, perturbed, target, batch_size), num_genes_dropped=float(n_drop))

    for panel_size in GENE_PANEL_SIZES:
        perturbed = _clone_batch(batch)
        keep_idx = _top_gene_indices(batch, panel_size)
        _zero_non_panel_genes(perturbed, keep_idx)
        _append_row(
            rows,
            "gene_panel",
            f"top{panel_size}_variance",
            _evaluate(model, perturbed, target, batch_size),
            num_genes_kept=float(len(keep_idx)),
        )
    marker_idx = _marker_gene_indices(metadata.gene_names, run_device)
    if len(marker_idx) > 0:
        perturbed = _clone_batch(batch)
        _zero_non_panel_genes(perturbed, marker_idx)
        _append_row(
            rows,
            "gene_panel",
            "marker_only",
            _evaluate(model, perturbed, target, batch_size),
            num_genes_kept=float(len(marker_idx)),
        )

    for idx, cell_type in enumerate(checkpoint["metadata"]["cell_types"]):
        if idx >= batch.reference_prototypes.size(0):
            continue
        perturbed = _clone_batch(batch)
        perturbed.reference_prototypes[idx] = 0.0
        metrics = _evaluate(model, perturbed, target, batch_size)
        _append_row(rows, "reference_missing_celltype", str(cell_type), metrics)

    prototype_generator = torch.Generator(device=run_device).manual_seed(99)
    proto_std = batch.reference_prototypes.std().clamp_min(1e-8)
    for noise_level in PROTOTYPE_NOISE_LEVELS:
        perturbed = _clone_batch(batch)
        noise = torch.randn(perturbed.reference_prototypes.shape, generator=prototype_generator, device=run_device) * proto_std * noise_level
        perturbed.reference_prototypes = (perturbed.reference_prototypes + noise).clamp_min(0.0)
        _append_row(rows, "prototype_perturbation", f"gaussian_noise_{noise_level}", _evaluate(model, perturbed, target, batch_size))
    for dropout_rate in PROTOTYPE_DROPOUT_RATES:
        perturbed = _clone_batch(batch)
        mask = torch.rand(perturbed.reference_prototypes.shape, generator=prototype_generator, device=run_device) < dropout_rate
        perturbed.reference_prototypes[mask] = 0.0
        _append_row(rows, "prototype_perturbation", f"entry_dropout_{dropout_rate}", _evaluate(model, perturbed, target, batch_size))

    for name, (scale, noise) in STAIN_PERTURBATIONS.items():
        perturbed = _clone_batch(batch)
        perturbed.image_patches = (perturbed.image_patches * scale).clamp(0.0, 1.0)
        if noise > 0:
            perturbed.image_patches = (perturbed.image_patches + noise * torch.randn_like(perturbed.image_patches)).clamp(0.0, 1.0)
        _append_row(rows, "he_perturbation", name, _evaluate(model, perturbed, target, batch_size))

    with torch.no_grad():
        clean_output = _run_model_batched(model, batch, batch_size)

    supervised = target.sum(dim=1) > 1e-8
    expression_total = batch.st_expression.sum(dim=1).detach().cpu()
    if supervised.any():
        low_expr_threshold = torch.quantile(expression_total[supervised.cpu()], 0.25).item()
        high_expr_threshold = torch.quantile(expression_total[supervised.cpu()], 0.75).item()
        for label, mask in [
            ("low_expression_spots", supervised.cpu() & (expression_total <= low_expr_threshold)),
            ("high_expression_spots", supervised.cpu() & (expression_total >= high_expr_threshold)),
        ]:
            _append_row(rows, "subgroup", label, _evaluate_output_subset(clean_output, target, mask))

        counts = _load_cell_counts(xenium_counts_path, metadata.spot_ids)
        if not counts.empty:
            count_tensor = torch.as_tensor(counts.values, dtype=torch.float32)
            covered = supervised.cpu() & (count_tensor > 0)
            if covered.any():
                low_count_threshold = torch.quantile(count_tensor[covered], 0.25).item()
                high_count_threshold = torch.quantile(count_tensor[covered], 0.75).item()
                for label, mask in [
                    ("low_cell_count_spots", covered & (count_tensor <= low_count_threshold)),
                    ("high_cell_count_spots", covered & (count_tensor >= high_count_threshold)),
                ]:
                    _append_row(rows, "subgroup", label, _evaluate_output_subset(clean_output, target, mask))

        split_table = _load_split_table(splits_path)
        if not split_table.empty:
            split_by_spot = split_table.set_index("spot_id")["split"].reindex(metadata.spot_ids)
            for split_name in sorted(split_by_spot.dropna().unique()):
                split_mask = torch.as_tensor((split_by_spot == split_name).to_numpy(), dtype=torch.bool)
                mask = supervised.cpu() & split_mask
                if bool(mask.sum()):
                    _append_row(rows, "split", f"manifest_{split_name}", _evaluate_output_subset(clean_output, target, mask))

        supervised_indices = torch.where(supervised.cpu())[0]
        random_generator = torch.Generator().manual_seed(314159)
        for seed_idx, holdout_fraction in enumerate([0.2, 0.3, 0.4], start=1):
            n_holdout = max(int(round(len(supervised_indices) * holdout_fraction)), 1)
            order = supervised_indices[torch.randperm(len(supervised_indices), generator=random_generator)]
            mask = torch.zeros_like(supervised.cpu())
            mask[order[:n_holdout]] = True
            _append_row(rows, "split", f"random_holdout_{holdout_fraction}", _evaluate_output_subset(clean_output, target, mask), holdout_fraction=float(holdout_fraction))

    del clean_output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_path = output_dir / "robustness_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "prepared_path": str(prepared_path),
        "output_dir": str(output_dir),
        "missing_alignment_items": missing,
        "summary_path": str(out_path),
        "gene_dropout_rates": GENE_DROPOUT_RATES,
        "gene_panel_sizes": GENE_PANEL_SIZES,
        "prototype_noise_levels": PROTOTYPE_NOISE_LEVELS,
        "prototype_dropout_rates": PROTOTYPE_DROPOUT_RATES,
        "stain_perturbations": list(STAIN_PERTURBATIONS),
        "splits_path": str(splits_path) if splits_path is not None else "",
        "xenium_counts_path": str(xenium_counts_path) if xenium_counts_path is not None else "",
    }
    (output_dir / "robustness_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return rows


def list_default_robustness_scenarios() -> dict[str, list[float | int | str]]:
    return {
        "gene_dropout": list(GENE_DROPOUT_RATES),
        "gene_panel": ["top200_variance", "top100_variance", "marker_only"],
        "prototype_perturbation": [f"gaussian_noise_{value}" for value in PROTOTYPE_NOISE_LEVELS]
        + [f"entry_dropout_{value}" for value in PROTOTYPE_DROPOUT_RATES],
        "he_perturbation": list(STAIN_PERTURBATIONS),
        "reference_missing_celltype": ["each_checkpoint_cell_type"],
        "subgroup": ["low_expression_spots", "high_expression_spots", "low_cell_count_spots", "high_cell_count_spots"],
        "patch_size": [32, 64, 128, 256],
        "split": ["manifest_train", "manifest_val", "manifest_test", "random_holdout_0.2", "random_holdout_0.3", "random_holdout_0.4"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WaveST-Gate robustness stress tests.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--splits", default=None, help="Optional spot_splits.csv for split-specific robustness metrics.")
    parser.add_argument("--xenium-counts", default=None, help="Optional Xenium cell-count table for low/high cell-count subgroup metrics.")
    args = parser.parse_args()
    rows = run_robustness(
        args.checkpoint,
        args.prepared,
        args.output_dir,
        args.batch_size,
        args.device,
        splits_path=args.splits,
        xenium_counts_path=args.xenium_counts,
    )
    print(json.dumps(rows[:5], indent=2), flush=True)
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
