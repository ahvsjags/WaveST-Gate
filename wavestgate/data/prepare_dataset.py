"""CLI and helpers for preparing real-data WaveST-Gate batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from wavestgate.data.assemble import BatchMetadata, assemble_wavestgate_batch
from wavestgate.data.extract_he_patches import extract_spot_patches
from wavestgate.data.ground_truth import count_cells_in_spots, proportions_from_counts
from wavestgate.data.preprocess_scrna import build_reference_prototypes, load_reference_prototypes_table
from wavestgate.data.preprocess_st import attach_coordinates, load_spot_coordinates, load_spot_expression, read_delimited_table
from wavestgate.models.types import WaveSTGateBatch


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON-compatible YAML config."""

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


def _optional_path(value: str | None) -> str | None:
    return value if value not in {None, ""} else None


def batch_to_state(batch: WaveSTGateBatch, metadata: BatchMetadata, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert a batch plus metadata to a portable state dictionary."""

    return {
        "image_patches": batch.image_patches.cpu(),
        "st_expression": batch.st_expression.cpu(),
        "reference_prototypes": batch.reference_prototypes.cpu(),
        "coords": batch.coords.cpu() if batch.coords is not None else None,
        "edge_index": batch.edge_index.cpu() if batch.edge_index is not None else None,
        "edge_weight": batch.edge_weight.cpu() if batch.edge_weight is not None else None,
        "proportion_gt": batch.proportion_gt.cpu() if batch.proportion_gt is not None else None,
        "niche_gt": batch.niche_gt.cpu() if batch.niche_gt is not None else None,
        "metadata": {
            "spot_ids": list(metadata.spot_ids),
            "gene_names": list(metadata.gene_names),
            "cell_types": list(metadata.cell_types),
        },
        "extra": extra or {},
    }


def state_to_batch(state: dict[str, Any], device: torch.device | str = "cpu") -> tuple[WaveSTGateBatch, BatchMetadata, dict[str, Any]]:
    """Reconstruct a batch from `batch_to_state` output."""

    batch = WaveSTGateBatch(
        image_patches=state["image_patches"],
        st_expression=state["st_expression"],
        reference_prototypes=state["reference_prototypes"],
        coords=state.get("coords"),
        edge_index=state.get("edge_index"),
        edge_weight=state.get("edge_weight"),
        proportion_gt=state.get("proportion_gt"),
        niche_gt=state.get("niche_gt"),
    ).to(device)
    meta = state["metadata"]
    metadata = BatchMetadata(
        spot_ids=list(meta["spot_ids"]),
        gene_names=list(meta["gene_names"]),
        cell_types=list(meta["cell_types"]),
    )
    return batch, metadata, dict(state.get("extra", {}))


def save_prepared_dataset(path: str | Path, batch: WaveSTGateBatch, metadata: BatchMetadata, extra: dict[str, Any] | None = None) -> None:
    """Save a prepared dataset to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(batch_to_state(batch, metadata, extra=extra), path)


def load_prepared_dataset(path: str | Path, device: torch.device | str = "cpu") -> tuple[WaveSTGateBatch, BatchMetadata, dict[str, Any]]:
    """Load a prepared dataset created by this module."""

    state = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        raise ValueError("Prepared dataset must be a state dictionary")
    return state_to_batch(state, device=device)


def _build_xenium_proportions(
    cells_path: str | Path,
    coords_df: pd.DataFrame,
    spot_ids: list[str],
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = read_delimited_table(cells_path)
    spot_table = coords_df.loc[spot_ids].reset_index()
    spot_id_col = cfg.get("spot_id_col", "spot_id")
    if spot_id_col not in spot_table.columns:
        spot_table = spot_table.rename(columns={spot_table.columns[0]: spot_id_col})
    counts = count_cells_in_spots(
        cells=cells,
        spots=spot_table,
        spot_radius=float(cfg["spot_radius"]),
        cell_type_col=cfg.get("cell_type_col", "cell_type"),
        cell_x_col=cfg.get("cell_x_col", "x"),
        cell_y_col=cfg.get("cell_y_col", "y"),
        spot_id_col=spot_id_col,
        spot_x_col=cfg.get("spot_x_col", "x"),
        spot_y_col=cfg.get("spot_y_col", "y"),
    )
    return counts, proportions_from_counts(counts)


def _load_niche_labels(
    labels_path: str | Path,
    spot_ids: list[str],
    spot_id_col: str = "spot_id",
    niche_col: str = "niche",
) -> tuple[pd.Series, dict[str, int]]:
    labels = read_delimited_table(labels_path)
    for col in [spot_id_col, niche_col]:
        if col not in labels.columns:
            raise ValueError(f"niche label table is missing column: {col}")
    labels[spot_id_col] = labels[spot_id_col].astype(str)
    missing_spots = [spot_id for spot_id in spot_ids if spot_id not in set(labels[spot_id_col])]
    if missing_spots:
        raise ValueError(f"Niche label table is missing spot ids: {missing_spots[:5]}")

    raw_values = labels.set_index(spot_id_col).loc[spot_ids, niche_col]
    if pd.api.types.is_numeric_dtype(raw_values):
        encoded = raw_values.astype(int)
        mapping = {str(value): int(value) for value in sorted(encoded.unique())}
    else:
        unique_labels = sorted(raw_values.astype(str).unique().tolist())
        mapping = {label: idx for idx, label in enumerate(unique_labels)}
        encoded = raw_values.astype(str).map(mapping).astype(int)
    encoded.index = spot_ids
    return encoded, mapping


def prepare_from_config(path: str | Path) -> tuple[WaveSTGateBatch, BatchMetadata, dict[str, Any]]:
    """Prepare a WaveST-Gate real-data batch from a config file."""

    raw = load_config(path)
    data_cfg = raw.get("data", {})
    required = ["spot_expression_path", "spot_coords_path", "he_image_path", "output_path"]
    missing = [key for key in required if key not in data_cfg]
    if missing:
        raise ValueError(f"Prepare config is missing required data keys: {missing}")
    if not data_cfg.get("scrna_expression_path") and not data_cfg.get("reference_prototypes_path"):
        raise ValueError("Prepare config requires either data.scrna_expression_path or data.reference_prototypes_path")

    columns = raw.get("columns", {})
    spots = load_spot_expression(
        data_cfg["spot_expression_path"],
        spot_id_col=columns.get("spot_id_col", "spot_id"),
    )
    coords = load_spot_coordinates(
        data_cfg["spot_coords_path"],
        spot_id_col=columns.get("spot_id_col", "spot_id"),
        x_col=columns.get("spot_x_col", "x"),
        y_col=columns.get("spot_y_col", "y"),
    )
    spots = attach_coordinates(spots, coords, x_col=columns.get("spot_x_col", "x"), y_col=columns.get("spot_y_col", "y"))

    if data_cfg.get("reference_prototypes_path"):
        reference = load_reference_prototypes_table(
            data_cfg["reference_prototypes_path"],
            cell_type_col=columns.get("cell_type_col", "cell_type"),
            target_genes=spots.gene_names,
        )
    else:
        reference = build_reference_prototypes(
            data_cfg["scrna_expression_path"],
            labels_path=_optional_path(data_cfg.get("scrna_labels_path")),
            cell_id_col=columns.get("cell_id_col", "cell_id"),
            cell_type_col=columns.get("cell_type_col", "cell_type"),
            target_genes=spots.gene_names,
        )

    patch_cfg = raw.get("patches", {})
    patch_size = int(patch_cfg.get("patch_size", raw.get("model", {}).get("patch_size", 256)))
    patch_output_dir = _optional_path(patch_cfg.get("output_dir"))
    patch_result = extract_spot_patches(
        data_cfg["he_image_path"],
        coords.loc[spots.spot_ids],
        patch_size=patch_size,
        spot_ids=spots.spot_ids,
        x_col=columns.get("spot_x_col", "x"),
        y_col=columns.get("spot_y_col", "y"),
        output_dir=None,
    )
    image_patches = patch_result

    saved_patch_paths = None
    if patch_output_dir is not None:
        saved_patch_paths = extract_spot_patches(
            data_cfg["he_image_path"],
            coords.loc[spots.spot_ids],
            patch_size=patch_size,
            spot_ids=spots.spot_ids,
            x_col=columns.get("spot_x_col", "x"),
            y_col=columns.get("spot_y_col", "y"),
            output_dir=patch_output_dir,
        )

    proportions = None
    counts = None
    xenium_cfg = raw.get("xenium", {})
    if data_cfg.get("xenium_cells_path"):
        if "spot_radius" not in xenium_cfg:
            raise ValueError("xenium.spot_radius is required when data.xenium_cells_path is provided")
        counts, proportions = _build_xenium_proportions(
            data_cfg["xenium_cells_path"],
            coords,
            spots.spot_ids,
            {
                **columns,
                **xenium_cfg,
                "spot_id_col": columns.get("spot_id_col", "spot_id"),
            },
        )
        if data_cfg.get("xenium_counts_output_path"):
            counts.to_csv(data_cfg["xenium_counts_output_path"], index_label="spot_id")
        if data_cfg.get("xenium_proportions_output_path"):
            proportions.to_csv(data_cfg["xenium_proportions_output_path"], index_label="spot_id")

    niche_labels = None
    niche_label_mapping = None
    if data_cfg.get("niche_labels_path"):
        niche_labels, niche_label_mapping = _load_niche_labels(
            data_cfg["niche_labels_path"],
            spots.spot_ids,
            spot_id_col=columns.get("spot_id_col", "spot_id"),
            niche_col=columns.get("niche_col", "niche"),
        )

    graph_cfg = raw.get("graph", {})
    graph_k = graph_cfg.get("k", 6)
    graph_k = None if graph_k is None else int(graph_k)
    batch, metadata = assemble_wavestgate_batch(
        spots,
        reference,
        image_patches=image_patches,
        proportions=proportions,
        niche_labels=niche_labels,
        graph_k=graph_k,
    )
    extra = {
        "source_config": str(path),
        "patch_size": patch_size,
        "patch_paths": [str(p) for p in saved_patch_paths] if saved_patch_paths is not None else None,
        "has_xenium_ground_truth": proportions is not None,
        "num_spots": len(metadata.spot_ids),
        "num_genes": len(metadata.gene_names),
        "num_cell_types": len(metadata.cell_types),
        "has_niche_labels": niche_labels is not None,
        "niche_label_mapping": niche_label_mapping,
    }
    save_prepared_dataset(data_cfg["output_path"], batch, metadata, extra=extra)
    return batch, metadata, extra


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WaveST-Gate tensors from real-data tables and H&E image.")
    parser.add_argument("--config", required=True, help="Path to prepare_dataset YAML/JSON config.")
    args = parser.parse_args()
    batch, metadata, extra = prepare_from_config(args.config)
    print(f"prepared_spots: {len(metadata.spot_ids)}")
    print(f"prepared_genes: {len(metadata.gene_names)}")
    print(f"prepared_cell_types: {len(metadata.cell_types)}")
    print(f"image_patches: {tuple(batch.image_patches.shape)}")
    print(f"has_xenium_ground_truth: {extra['has_xenium_ground_truth']}")
    print(f"has_niche_labels: {extra['has_niche_labels']}")


if __name__ == "__main__":
    main()
