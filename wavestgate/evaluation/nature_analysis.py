"""Submission-oriented analyses for reliability, boundaries, and niches."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from wavestgate.data.build_spatial_graph import build_knn_graph
from wavestgate.data.prepare_dataset import load_prepared_dataset
from wavestgate.evaluation.metrics import per_spot_jensen_shannon_divergence


MARKER_SETS = {
    "tumor_epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "ERBB2"],
    "stromal": ["COL1A1", "COL1A2", "DCN", "LUM", "ACTA2"],
    "immune": ["CD3D", "CD3E", "CD8A", "MS4A1", "PTPRC"],
    "macrophage": ["CD68", "CD163", "MRC1", "C1QA", "C1QB"],
    "t_cell": ["CD3D", "CD3E", "CD4", "CD8A", "GZMB"],
    "her2_tumor": ["ERBB2", "GRB7", "MUC1", "KRT19", "EPCAM"],
}

CELL_TYPE_GROUPS = {
    "immune": ["B Cells", "CD4+ T Cells", "CD8+ T Cells", "IRF7+ DCs", "LAMP3+ DCs", "Macrophages 1", "Macrophages 2", "Mast Cells"],
    "tumor": ["DCIS 1", "DCIS 2", "Invasive Tumor", "Prolif Invasive Tumor", "T Cell & Tumor Hybrid"],
    "ductal": ["DCIS 1", "DCIS 2", "Myoepi ACTA2+", "Myoepi KRT15+"],
    "stromal": ["Endothelial", "Perivascular-Like", "Stromal", "Stromal & T Cell Hybrid"],
}

NICHE_VALIDATION_GROUPS = {
    "tumor": ["DCIS 1", "DCIS 2", "Invasive Tumor", "Prolif Invasive Tumor", "T Cell & Tumor Hybrid"],
    "immune": ["B Cells", "CD4+ T Cells", "CD8+ T Cells", "IRF7+ DCs", "LAMP3+ DCs", "Macrophages 1", "Macrophages 2", "Mast Cells"],
    "stromal": ["Endothelial", "Perivascular-Like", "Stromal", "Stromal & T Cell Hybrid", "Myoepi ACTA2+", "Myoepi KRT15+"],
}


def _read_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "spot_id" in df.columns:
        df = df.set_index("spot_id")
    else:
        df = df.set_index(df.columns[0])
    return df


def _aligned_tensor(df: pd.DataFrame, spot_ids: list[str], columns: list[str]) -> torch.Tensor:
    aligned = df.reindex(index=spot_ids, columns=columns, fill_value=0.0)
    return torch.as_tensor(aligned.values, dtype=torch.float32)


def _write_spatial_map(
    output_dir: Path,
    name: str,
    coords: pd.DataFrame,
    values: pd.Series,
    cmap: str = "viridis",
) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    path = output_dir / f"{name}.png"
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    sc = ax.scatter(coords["x"], coords["y"], c=values.reindex(coords.index), s=8, cmap=cmap)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_").lower()


def _write_spatial_panel(
    output_dir: Path,
    name: str,
    coords: pd.DataFrame,
    values_by_name: dict[str, pd.Series],
    cmap: str = "viridis",
) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    if not values_by_name:
        return ""
    n_items = len(values_by_name)
    n_cols = min(4, n_items)
    n_rows = int(np.ceil(n_items / n_cols))
    path = output_dir / f"{name}.png"
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.9 * n_rows), dpi=180, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, values) in zip(axes.ravel(), values_by_name.items()):
        sc = ax.scatter(coords["x"], coords["y"], c=values.reindex(coords.index), s=4, cmap=cmap, vmin=0.0, vmax=max(float(values.max()), 1e-8))
        ax.set_title(title, fontsize=8)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        ax.axis("on")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _write_xy_plot(
    output_dir: Path,
    name: str,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    path = output_dir / f"{name}.png"
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    ax.plot(frame[x_col], frame[y_col], marker="o", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _write_boundary_he_overlay(
    output_dir: Path,
    coords: pd.DataFrame,
    image_rgb: pd.DataFrame,
    edge_rows: pd.DataFrame,
    name: str = "boundary_he_overlay",
) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    path = output_dir / f"{name}.png"
    rgb = image_rgb.reindex(coords.index)[["patch_red_mean", "patch_green_mean", "patch_blue_mean"]].fillna(0.5).to_numpy()
    rgb = np.clip(rgb, 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=180)
    ax.scatter(coords["x"], coords["y"], c=rgb, s=10, linewidths=0)
    if len(edge_rows):
        threshold = float(edge_rows["predicted_l1_jump"].quantile(0.90))
        top_edges = edge_rows[edge_rows["predicted_l1_jump"] >= threshold].nlargest(1500, "predicted_l1_jump")
        coord_lookup = coords[["x", "y"]].to_dict("index")
        for _, row in top_edges.iterrows():
            src = coord_lookup.get(row["src"])
            dst = coord_lookup.get(row["dst"])
            if src is None or dst is None:
                continue
            ax.plot([src["x"], dst["x"]], [src["y"], dst["y"]], color="#00FFFF", alpha=0.25, linewidth=0.6)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _permutation_mean_gap_pvalue(
    low_values: np.ndarray,
    high_values: np.ndarray,
    permutations: int = 10000,
    seed: int = 19,
) -> float:
    observed = float(np.mean(high_values) - np.mean(low_values))
    values = np.concatenate([low_values, high_values])
    labels = np.concatenate([np.zeros(len(low_values), dtype=bool), np.ones(len(high_values), dtype=bool)])
    if len(low_values) == 0 or len(high_values) == 0 or not np.isfinite(observed):
        return 1.0
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(permutations):
        shuffled = rng.permutation(labels)
        diff = float(values[shuffled].mean() - values[~shuffled].mean())
        if diff >= observed:
            count += 1
    return float((count + 1) / (permutations + 1))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) <= 1e-8 or np.std(b) <= 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _spearman_like(a: np.ndarray, b: np.ndarray) -> float:
    return _safe_corr(pd.Series(a).rank(method="average").to_numpy(), pd.Series(b).rank(method="average").to_numpy())


def _cell_type_group(cell_type: str) -> str:
    for group, values in CELL_TYPE_GROUPS.items():
        if cell_type in values:
            return group
    return "other"


def _group_fraction_frame(frame: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = {}
    used: set[str] = set()
    for group, cell_types in groups.items():
        present = [cell_type for cell_type in cell_types if cell_type in frame.columns]
        used.update(present)
        rows[group] = frame[present].sum(axis=1) if present else pd.Series(0.0, index=frame.index)
    other_cols = [cell_type for cell_type in frame.columns if cell_type not in used]
    rows["other"] = frame[other_cols].sum(axis=1) if other_cols else pd.Series(0.0, index=frame.index)
    grouped = pd.DataFrame(rows, index=frame.index)
    totals = grouped.sum(axis=1).replace(0.0, np.nan)
    return grouped.div(totals, axis=0).fillna(0.0)


def _row_entropy(frame: pd.DataFrame) -> pd.Series:
    values = frame.to_numpy(dtype=float)
    values = values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)
    entropy = -(values * np.log(np.clip(values, 1e-12, 1.0))).sum(axis=1)
    return pd.Series(entropy, index=frame.index)


def _edge_type(src_type: str, dst_type: str) -> str:
    src_group = _cell_type_group(src_type)
    dst_group = _cell_type_group(dst_type)
    groups = {src_group, dst_group}
    if src_type == dst_type:
        return f"within_{src_group}"
    if src_group == "ductal" or dst_group == "ductal" or "DCIS" in src_type or "DCIS" in dst_type or "Myoepi" in src_type or "Myoepi" in dst_type:
        return "ductal_boundary"
    if groups == {"tumor", "stromal"}:
        return "tumor_stroma_boundary"
    if groups == {"tumor", "immune"}:
        return "immune_infiltration_edge"
    if groups == {"immune", "stromal"}:
        return "immune_stromal_edge"
    return "_".join(sorted(groups))


def _infer_niche_label(row: pd.Series) -> str:
    dominant = str(row.get("dominant_cell_type", ""))
    tumor_score = float(row.get("marker_tumor_epithelial", 0.0))
    her2_score = float(row.get("marker_her2_tumor", 0.0))
    stromal_score = float(row.get("marker_stromal", 0.0))
    immune_score = float(row.get("marker_immune", 0.0))
    macrophage_score = float(row.get("marker_macrophage", 0.0))
    t_cell_score = float(row.get("marker_t_cell", 0.0))
    if her2_score >= max(stromal_score, immune_score, macrophage_score, t_cell_score) and ("Tumor" in dominant or "DCIS" in dominant or tumor_score > 0):
        return "HER2/tumor-associated niche"
    if "Tumor" in dominant or "DCIS" in dominant or tumor_score >= max(stromal_score, immune_score):
        if stromal_score > 0 and stromal_score >= 0.25 * max(tumor_score, 1e-8):
            return "tumor-stroma interface niche"
        return "tumor epithelial niche"
    if "Macrophages" in dominant or macrophage_score >= max(tumor_score, stromal_score, immune_score):
        return "macrophage-rich immune niche"
    if "T Cells" in dominant or t_cell_score >= max(tumor_score, stromal_score, macrophage_score):
        return "T cell-rich immune niche"
    if immune_score >= max(tumor_score, stromal_score):
        return "immune infiltration niche"
    if "Stromal" in dominant or "Endothelial" in dominant or stromal_score >= max(tumor_score, immune_score):
        return "stromal remodeling niche"
    return "mixed tumor-immune-stromal niche"


def proportion_map_analysis(
    prepared_path: str | Path,
    predictions_path: str | Path,
    output_dir: Path,
    top_n: int = 8,
) -> dict[str, object]:
    batch, metadata, _ = load_prepared_dataset(prepared_path)
    if batch.coords is None:
        raise ValueError("Proportion map analysis requires spot coordinates")
    map_dir = output_dir / "proportion_maps"
    map_dir.mkdir(parents=True, exist_ok=True)

    pred_df = _read_table(predictions_path)
    pred = pred_df.reindex(index=metadata.spot_ids, columns=metadata.cell_types, fill_value=0.0).astype(float)
    pred[pred < 0] = 0.0
    pred = pred.div(pred.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(1.0 / max(len(metadata.cell_types), 1))
    coords = pd.DataFrame(batch.coords.cpu().numpy(), index=metadata.spot_ids, columns=["x", "y"])

    summary_rows = []
    for cell_type in metadata.cell_types:
        values = pred[cell_type]
        summary_rows.append(
            {
                "cell_type": cell_type,
                "group": _cell_type_group(cell_type),
                "mean_fraction": float(values.mean()),
                "median_fraction": float(values.median()),
                "max_fraction": float(values.max()),
                "num_spots_gt_0_1": int((values > 0.1).sum()),
                "num_spots_gt_0_25": int((values > 0.25).sum()),
                "num_spots_gt_0_5": int((values > 0.5).sum()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["mean_fraction", "max_fraction"], ascending=False)
    summary_path = map_dir / "cell_type_proportion_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    priority = [
        "Invasive Tumor",
        "Prolif Invasive Tumor",
        "DCIS 1",
        "DCIS 2",
        "Stromal",
        "Macrophages 1",
        "Macrophages 2",
        "CD8+ T Cells",
        "CD4+ T Cells",
        "B Cells",
        "Endothelial",
    ]
    selected: list[str] = []
    for cell_type in priority:
        if cell_type in pred.columns and cell_type not in selected:
            selected.append(cell_type)
    for cell_type in summary_df["cell_type"].tolist():
        if cell_type not in selected:
            selected.append(cell_type)
        if len(selected) >= top_n:
            break

    cell_type_map_paths: dict[str, str] = {}
    selected_series: dict[str, pd.Series] = {}
    for cell_type in selected:
        name = f"predicted_{_safe_name(cell_type)}_proportion_map"
        cell_type_map_paths[cell_type] = _write_spatial_map(map_dir, name, coords, pred[cell_type], cmap="magma")
        selected_series[cell_type] = pred[cell_type]
    panel_path = _write_spatial_panel(map_dir, "predicted_top_celltypes_panel", coords, selected_series, cmap="magma")

    group_frame = _group_fraction_frame(pred, {k: v for k, v in CELL_TYPE_GROUPS.items() if k in {"tumor", "immune", "stromal", "ductal"}})
    group_map_paths: dict[str, str] = {}
    for group in ["tumor", "immune", "stromal", "ductal", "other"]:
        if group in group_frame.columns:
            group_map_paths[group] = _write_spatial_map(map_dir, f"predicted_{group}_fraction_map", coords, group_frame[group], cmap="viridis")
    group_panel_path = _write_spatial_panel(
        map_dir,
        "predicted_tumor_immune_stromal_group_panel",
        coords,
        {group: group_frame[group] for group in group_frame.columns},
        cmap="viridis",
    )

    manifest = {
        "num_spots": int(len(metadata.spot_ids)),
        "num_cell_types": int(len(metadata.cell_types)),
        "selected_cell_types": selected,
        "summary_path": str(summary_path),
        "top_celltype_panel": panel_path,
        "group_panel": group_panel_path,
        "cell_type_map_paths": cell_type_map_paths,
        "group_map_paths": group_map_paths,
    }
    manifest_path = map_dir / "proportion_map_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def reliability_analysis(
    prepared_path: str | Path,
    predictions_path: str | Path,
    uncertainty_path: str | Path,
    gates_path: str | Path,
    output_dir: Path,
    modality_reliability_path: str | Path | None = None,
    n_bins: int = 10,
) -> dict[str, object]:
    batch, metadata, _ = load_prepared_dataset(prepared_path)
    if batch.proportion_gt is None:
        raise ValueError("Reliability analysis requires prepared proportion_gt")
    pred_df = _read_table(predictions_path)
    uncertainty_df = _read_table(uncertainty_path)
    gates_df = _read_table(gates_path)
    modality_df = _read_table(modality_reliability_path) if modality_reliability_path is not None and Path(modality_reliability_path).exists() else None
    pred = _aligned_tensor(pred_df, metadata.spot_ids, metadata.cell_types)
    target = batch.proportion_gt.cpu()
    mask = target.sum(dim=1) > 1e-8
    jsd = per_spot_jensen_shannon_divergence(pred[mask], target[mask]).numpy()
    supervised_ids = np.array(metadata.spot_ids)[mask.numpy()]
    uncertainty = uncertainty_df.reindex(supervised_ids)["spot_uncertainty"].astype(float).to_numpy()
    order = np.argsort(uncertainty)

    spot_rows = pd.DataFrame({"spot_id": supervised_ids, "jsd": jsd, "spot_uncertainty": uncertainty}).set_index("spot_id")
    spot_rows = spot_rows.join(gates_df, how="left")
    if modality_df is not None:
        spot_rows = spot_rows.join(modality_df.add_prefix("reliability_"), how="left")
    expression_totals = batch.st_expression.cpu().sum(dim=1).numpy()
    expression_zero_fraction = (batch.st_expression.cpu().numpy() <= 1e-8).mean(axis=1)
    image_mean = batch.image_patches.cpu().float().mean(dim=(1, 2, 3)).numpy()
    image_std = batch.image_patches.cpu().float().std(dim=(1, 2, 3)).numpy()
    spot_rows = spot_rows.join(
        pd.DataFrame(
            {
                "total_expression": expression_totals,
                "expression_zero_fraction": expression_zero_fraction,
                "image_mean": image_mean,
                "image_std": image_std,
            },
            index=metadata.spot_ids,
        ),
        how="left",
    )
    target_df = pd.DataFrame(target.numpy(), index=metadata.spot_ids, columns=metadata.cell_types)
    pred_aligned = pred_df.reindex(index=metadata.spot_ids, columns=metadata.cell_types, fill_value=0.0)
    spot_rows["dominant_gt"] = target_df.loc[supervised_ids].idxmax(axis=1)
    spot_rows["dominant_pred"] = pred_aligned.loc[supervised_ids].idxmax(axis=1)
    spot_rows["dominant_gt_fraction"] = target_df.loc[supervised_ids].max(axis=1)
    spot_rows["dominant_pred_fraction"] = pred_aligned.loc[supervised_ids].max(axis=1)
    spot_rows.to_csv(output_dir / "reliability_spot_errors.csv", index_label="spot_id")
    if batch.coords is not None:
        coords = pd.DataFrame(batch.coords.cpu().numpy(), index=metadata.spot_ids, columns=["x", "y"])
        uncertainty_all = uncertainty_df.reindex(metadata.spot_ids)["spot_uncertainty"].astype(float)
        _write_spatial_map(output_dir, "spot_uncertainty_map", coords, uncertainty_all, cmap="magma")
        for gate_name in ["image", "expression", "reference"]:
            _write_spatial_map(output_dir, f"{gate_name}_gate_map", coords, gates_df.reindex(metadata.spot_ids)[gate_name], cmap="viridis")

    coverages = np.linspace(0.1, 1.0, 10)
    risk_rows = []
    for coverage in coverages:
        n_keep = max(int(round(len(order) * coverage)), 1)
        keep = order[:n_keep]
        risk_rows.append({"coverage": float(coverage), "mean_jsd": float(jsd[keep].mean()), "num_spots": int(n_keep)})
    pd.DataFrame(risk_rows).to_csv(output_dir / "risk_coverage_curve.csv", index=False)
    risk_path = _write_xy_plot(output_dir, "risk_coverage_curve", pd.DataFrame(risk_rows), "coverage", "mean_jsd", "Coverage kept from lowest uncertainty", "Mean JSD")

    if len(np.unique(uncertainty)) < 2:
        bin_ids = np.zeros_like(uncertainty, dtype=int)
    else:
        bin_ids = pd.qcut(uncertainty, q=min(n_bins, len(np.unique(uncertainty))), labels=False, duplicates="drop")
    calibration_rows = []
    for bin_id in sorted(pd.Series(bin_ids).dropna().unique()):
        in_bin = np.asarray(bin_ids == bin_id)
        calibration_rows.append(
            {
                "bin": int(bin_id),
                "num_spots": int(in_bin.sum()),
                "mean_uncertainty": float(uncertainty[in_bin].mean()),
                "median_uncertainty": float(np.median(uncertainty[in_bin])),
                "mean_jsd": float(jsd[in_bin].mean()),
                "median_jsd": float(np.median(jsd[in_bin])),
            }
        )
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(output_dir / "uncertainty_calibration_bins.csv", index=False)
    calibration_path = _write_xy_plot(output_dir, "uncertainty_calibration", calibration_df, "mean_uncertainty", "mean_jsd", "Mean spot uncertainty", "Observed mean JSD")

    failure_cols = [
        "jsd",
        "spot_uncertainty",
        "total_expression",
        "expression_zero_fraction",
        "image_mean",
        "image_std",
        "dominant_gt",
        "dominant_pred",
        "dominant_gt_fraction",
        "dominant_pred_fraction",
        "image",
        "expression",
        "reference",
    ]
    failure_cols.extend([col for col in spot_rows.columns if col.startswith("reliability_")])
    failure_cases = spot_rows.sort_values("jsd", ascending=False).head(50)[failure_cols]
    failure_cases.to_csv(output_dir / "failure_case_candidates.csv", index_label="spot_id")

    q = max(int(np.ceil(len(order) * 0.25)), 1)
    low = jsd[order[:q]]
    high = jsd[order[-q:]]
    risk_df = pd.DataFrame(risk_rows)
    summary = {
        "num_supervised_spots": int(len(supervised_ids)),
        "uncertainty_error_pearson": _safe_corr(uncertainty, jsd),
        "uncertainty_error_spearman": _spearman_like(uncertainty, jsd),
        "low_uncertainty_jsd": float(low.mean()),
        "high_uncertainty_jsd": float(high.mean()),
        "risk_gap": float(high.mean() - low.mean()),
        "high_vs_low_uncertainty_permutation_p": _permutation_mean_gap_pvalue(low, high),
        "risk_coverage_auc": float(np.trapezoid(risk_df["mean_jsd"], risk_df["coverage"])),
        "calibration_bin_pearson": _safe_corr(calibration_df["mean_uncertainty"].to_numpy(), calibration_df["mean_jsd"].to_numpy()),
        "calibration_bin_spearman": _spearman_like(calibration_df["mean_uncertainty"].to_numpy(), calibration_df["mean_jsd"].to_numpy()),
        "mean_image_gate": float(gates_df["image"].mean()),
        "mean_expression_gate": float(gates_df["expression"].mean()),
        "mean_reference_gate": float(gates_df["reference"].mean()),
        "failure_case_candidates_path": str(output_dir / "failure_case_candidates.csv"),
        "map_paths": {
            "spot_uncertainty_map": str(output_dir / "spot_uncertainty_map.png"),
            "image_gate_map": str(output_dir / "image_gate_map.png"),
            "expression_gate_map": str(output_dir / "expression_gate_map.png"),
            "reference_gate_map": str(output_dir / "reference_gate_map.png"),
            "risk_coverage_curve": risk_path,
            "uncertainty_calibration": calibration_path,
        },
    }
    (output_dir / "reliability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def boundary_analysis(
    prepared_path: str | Path,
    predictions_path: str | Path,
    output_dir: Path,
    comparison_predictions_path: str | Path | None = None,
    k: int = 6,
) -> dict[str, object]:
    batch, metadata, _ = load_prepared_dataset(prepared_path)
    pred_df = _read_table(predictions_path)
    pred = _aligned_tensor(pred_df, metadata.spot_ids, metadata.cell_types)
    coords = batch.coords.cpu()
    edge_index, edge_weight = build_knn_graph(coords, k=k)
    src, dst = edge_index.long()
    pred_jump = (pred[src] - pred[dst]).abs().sum(dim=1).numpy()
    src_ids = np.array(metadata.spot_ids)[src.numpy()]
    dst_ids = np.array(metadata.spot_ids)[dst.numpy()]
    pred_df_aligned = pred_df.reindex(index=metadata.spot_ids, columns=metadata.cell_types, fill_value=0.0)
    pred_dominant = pred_df_aligned.idxmax(axis=1)
    rows = pd.DataFrame(
        {
            "src": src_ids,
            "dst": dst_ids,
            "spatial_weight": edge_weight.numpy(),
            "predicted_l1_jump": pred_jump,
        }
    )
    if batch.proportion_gt is not None:
        target = batch.proportion_gt.cpu()
        target_df = pd.DataFrame(target.numpy(), index=metadata.spot_ids, columns=metadata.cell_types)
        gt_dominant = target_df.idxmax(axis=1)
        valid = (target[src].sum(dim=1) > 1e-8) & (target[dst].sum(dim=1) > 1e-8)
        gt_jump = (target[src] - target[dst]).abs().sum(dim=1).numpy()
        rows["ground_truth_l1_jump"] = gt_jump
        rows["has_gt_pair"] = valid.numpy()
        threshold = np.quantile(gt_jump[valid.numpy()], 0.75) if valid.any() else np.inf
        rows["boundary_edge"] = rows["ground_truth_l1_jump"] >= threshold
        rows["src_dominant_gt"] = gt_dominant.reindex(src_ids).to_numpy()
        rows["dst_dominant_gt"] = gt_dominant.reindex(dst_ids).to_numpy()
        rows["boundary_type"] = [
            _edge_type(src_type, dst_type) if is_valid else _edge_type(pred_dominant.loc[src_id], pred_dominant.loc[dst_id])
            for src_type, dst_type, is_valid, src_id, dst_id in zip(
                rows["src_dominant_gt"], rows["dst_dominant_gt"], rows["has_gt_pair"], rows["src"], rows["dst"]
            )
        ]
    else:
        rows["boundary_type"] = [_edge_type(pred_dominant.loc[src_id], pred_dominant.loc[dst_id]) for src_id, dst_id in zip(src_ids, dst_ids)]
    if comparison_predictions_path is not None:
        cmp_df = _read_table(comparison_predictions_path)
        cmp = _aligned_tensor(cmp_df, metadata.spot_ids, metadata.cell_types)
        rows["comparison_l1_jump"] = (cmp[src] - cmp[dst]).abs().sum(dim=1).numpy()
    rows.to_csv(output_dir / "boundary_edge_jumps.csv", index=False)

    if "boundary_edge" in rows.columns:
        boundary = rows[rows["boundary_edge"]]
        interior = rows[~rows["boundary_edge"]]
    else:
        boundary = rows.nlargest(max(int(len(rows) * 0.25), 1), "predicted_l1_jump")
        interior = rows.drop(boundary.index)
    summary = {
        "num_edges": int(len(rows)),
        "mean_boundary_jump": float(boundary["predicted_l1_jump"].mean()),
        "mean_interior_jump": float(interior["predicted_l1_jump"].mean()) if len(interior) else 0.0,
        "boundary_to_interior_jump_ratio": float(boundary["predicted_l1_jump"].mean() / max(interior["predicted_l1_jump"].mean(), 1e-8))
        if len(interior)
        else 0.0,
    }
    if "comparison_l1_jump" in rows.columns:
        summary["comparison_mean_boundary_jump"] = float(boundary["comparison_l1_jump"].mean())
        summary["comparison_mean_interior_jump"] = float(interior["comparison_l1_jump"].mean()) if len(interior) else 0.0

    type_rows = []
    for boundary_type, group in rows.groupby("boundary_type"):
        type_boundary = group[group["boundary_edge"]] if "boundary_edge" in group.columns else group
        type_rows.append(
            {
                "boundary_type": boundary_type,
                "num_edges": int(len(group)),
                "num_boundary_edges": int(len(type_boundary)),
                "mean_predicted_l1_jump": float(group["predicted_l1_jump"].mean()),
                "mean_boundary_predicted_l1_jump": float(type_boundary["predicted_l1_jump"].mean()) if len(type_boundary) else 0.0,
                "mean_ground_truth_l1_jump": float(group["ground_truth_l1_jump"].mean()) if "ground_truth_l1_jump" in group.columns else 0.0,
                "mean_comparison_l1_jump": float(group["comparison_l1_jump"].mean()) if "comparison_l1_jump" in group.columns else 0.0,
                "boundary_preservation_delta_vs_comparison": float(group["predicted_l1_jump"].mean() - group["comparison_l1_jump"].mean())
                if "comparison_l1_jump" in group.columns
                else 0.0,
            }
        )
    boundary_type_df = pd.DataFrame(type_rows).sort_values("num_edges", ascending=False)
    boundary_type_df.to_csv(output_dir / "boundary_type_summary.csv", index=False)

    expression = pd.DataFrame(batch.st_expression.cpu().numpy(), index=metadata.spot_ids, columns=metadata.gene_names)
    marker_rows = []
    for boundary_type, group in rows.groupby("boundary_type"):
        edge_spots = pd.Index(np.unique(np.concatenate([group["src"].to_numpy(), group["dst"].to_numpy()])))
        for marker_set, genes in MARKER_SETS.items():
            present = [gene for gene in genes if gene in expression.columns]
            if present:
                marker_rows.append(
                    {
                        "boundary_type": boundary_type,
                        "marker_set": marker_set,
                        "present_genes": ";".join(present),
                        "num_edge_spots": int(len(edge_spots)),
                        "mean_expression": float(expression.loc[edge_spots, present].mean(axis=1).mean()),
                    }
                )
    pd.DataFrame(marker_rows).to_csv(output_dir / "boundary_marker_validation.csv", index=False)

    image_patches = batch.image_patches.cpu().float()
    image_rgb = pd.DataFrame(
        image_patches.mean(dim=(2, 3)).numpy(),
        index=metadata.spot_ids,
        columns=["patch_red_mean", "patch_green_mean", "patch_blue_mean"],
    )
    image_rgb["patch_intensity_mean"] = image_patches.mean(dim=(1, 2, 3)).numpy()
    image_rgb["patch_texture_std"] = image_patches.std(dim=(1, 2, 3)).numpy()
    he_proxy_rows = []
    for boundary_type, group in rows.groupby("boundary_type"):
        edge_spots = pd.Index(np.unique(np.concatenate([group["src"].to_numpy(), group["dst"].to_numpy()])))
        features = image_rgb.loc[edge_spots]
        he_proxy_rows.append(
            {
                "boundary_type": boundary_type,
                "num_edge_spots": int(len(edge_spots)),
                "mean_patch_red": float(features["patch_red_mean"].mean()),
                "mean_patch_green": float(features["patch_green_mean"].mean()),
                "mean_patch_blue": float(features["patch_blue_mean"].mean()),
                "mean_patch_intensity": float(features["patch_intensity_mean"].mean()),
                "mean_patch_texture_std": float(features["patch_texture_std"].mean()),
                "mean_predicted_l1_jump": float(group["predicted_l1_jump"].mean()),
            }
        )
    pd.DataFrame(he_proxy_rows).to_csv(output_dir / "boundary_he_pathology_proxy.csv", index=False)

    if batch.coords is not None:
        coords_df = pd.DataFrame(batch.coords.cpu().numpy(), index=metadata.spot_ids, columns=["x", "y"])
        spot_boundary_score = pd.Series(0.0, index=metadata.spot_ids)
        for spot_id, value in pd.concat(
            [
                rows[["src", "predicted_l1_jump"]].rename(columns={"src": "spot_id"}),
                rows[["dst", "predicted_l1_jump"]].rename(columns={"dst": "spot_id"}),
            ]
        ).groupby("spot_id")["predicted_l1_jump"].max().items():
            spot_boundary_score.loc[spot_id] = float(value)
        summary["boundary_map"] = _write_spatial_map(output_dir, "boundary_sharpness_map", coords_df, spot_boundary_score, cmap="inferno")
        summary["boundary_he_overlay"] = _write_boundary_he_overlay(output_dir, coords_df, image_rgb, rows)
    summary["boundary_type_summary_path"] = str(output_dir / "boundary_type_summary.csv")
    summary["boundary_marker_validation_path"] = str(output_dir / "boundary_marker_validation.csv")
    summary["boundary_he_pathology_proxy_path"] = str(output_dir / "boundary_he_pathology_proxy.csv")
    (output_dir / "boundary_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def niche_analysis(
    prepared_path: str | Path,
    predictions_path: str | Path,
    gates_path: str | Path,
    agent_attention_path: str | Path,
    output_dir: Path,
    n_niches: int = 5,
    uncertainty_path: str | Path | None = None,
    modality_reliability_path: str | Path | None = None,
) -> dict[str, object]:
    batch, metadata, _ = load_prepared_dataset(prepared_path)
    pred_df = _read_table(predictions_path)
    gates_df = _read_table(gates_path)
    agent_df = _read_table(agent_attention_path)
    uncertainty_df = _read_table(uncertainty_path) if uncertainty_path is not None and Path(uncertainty_path).exists() else None
    modality_df = _read_table(modality_reliability_path) if modality_reliability_path is not None and Path(modality_reliability_path).exists() else None
    pred = pred_df.reindex(index=metadata.spot_ids, columns=metadata.cell_types, fill_value=0.0)
    features = pred.to_numpy(dtype=float)
    n_niches = max(1, min(int(n_niches), len(features)))
    labels = KMeans(n_clusters=n_niches, random_state=0, n_init=20).fit_predict(features)
    assignments = pd.DataFrame({"spot_id": metadata.spot_ids, "niche": labels}).set_index("spot_id")

    composition_rows = []
    gate_rows = []
    agent_rows = []
    for niche in range(n_niches):
        ids = assignments.index[assignments["niche"] == niche]
        mean_props = pred.loc[ids].mean(axis=0)
        row = {"niche": niche, "num_spots": int(len(ids)), "dominant_cell_type": str(mean_props.idxmax())}
        row.update({f"mean_{cell_type}": float(value) for cell_type, value in mean_props.items()})
        composition_rows.append(row)
        gate_mean = gates_df.loc[ids].mean(axis=0)
        gate_rows.append({"niche": niche, "num_spots": int(len(ids)), **{f"mean_{k}_gate": float(v) for k, v in gate_mean.items()}})
        agent_mean = agent_df.loc[ids].mean(axis=0)
        agent_rows.append({"niche": niche, "top_agent": str(agent_mean.idxmax()), "top_agent_attention": float(agent_mean.max())})
    composition_df = pd.DataFrame(composition_rows)
    gate_df = pd.DataFrame(gate_rows)
    agent_summary_df = pd.DataFrame(agent_rows)

    expression = pd.DataFrame(batch.st_expression.cpu().numpy(), index=metadata.spot_ids, columns=metadata.gene_names)
    marker_rows = []
    for marker_set, genes in MARKER_SETS.items():
        present = [gene for gene in genes if gene in expression.columns]
        if not present:
            continue
        values = expression[present].mean(axis=1)
        for niche in range(n_niches):
            ids = assignments.index[assignments["niche"] == niche]
            marker_rows.append(
                {
                    "marker_set": marker_set,
                    "niche": niche,
                    "present_genes": ";".join(present),
                    "mean_expression": float(values.loc[ids].mean()) if len(ids) else 0.0,
                }
            )
    marker_df = pd.DataFrame(marker_rows, columns=["marker_set", "niche", "present_genes", "mean_expression"])
    marker_df.to_csv(output_dir / "niche_marker_enrichment.csv", index=False)

    if marker_df.empty:
        marker_wide = pd.DataFrame({"niche": list(range(n_niches))})
    else:
        marker_wide = marker_df.pivot(index="niche", columns="marker_set", values="mean_expression").add_prefix("marker_").reset_index()
    biological_df = composition_df.merge(gate_df, on=["niche", "num_spots"], how="left")
    biological_df = biological_df.merge(agent_summary_df, on="niche", how="left")
    biological_df = biological_df.merge(marker_wide, on="niche", how="left").fillna(0.0)
    if uncertainty_df is not None:
        uncertainty_rows = []
        for niche in range(n_niches):
            ids = assignments.index[assignments["niche"] == niche]
            uncertainty_rows.append(
                {
                    "niche": niche,
                    "mean_spot_uncertainty": float(uncertainty_df.loc[ids, "spot_uncertainty"].mean()),
                    "median_spot_uncertainty": float(uncertainty_df.loc[ids, "spot_uncertainty"].median()),
                }
            )
        biological_df = biological_df.merge(pd.DataFrame(uncertainty_rows), on="niche", how="left")
    if modality_df is not None:
        modality_rows = []
        for niche in range(n_niches):
            ids = assignments.index[assignments["niche"] == niche]
            means = modality_df.loc[ids].mean(axis=0)
            modality_rows.append({"niche": niche, **{f"mean_{col}_reliability": float(value) for col, value in means.items()}})
        biological_df = biological_df.merge(pd.DataFrame(modality_rows), on="niche", how="left")
    biological_df["niche_label"] = biological_df.apply(_infer_niche_label, axis=1)
    biological_df.to_csv(output_dir / "niche_biological_summary.csv", index=False)

    label_map = biological_df.set_index("niche")["niche_label"].to_dict()
    assignments["niche_label"] = assignments["niche"].map(label_map)
    assignments.to_csv(output_dir / "niche_assignments.csv", index_label="spot_id")
    composition_df = composition_df.merge(biological_df[["niche", "niche_label"]], on="niche", how="left")
    gate_df = gate_df.merge(biological_df[["niche", "niche_label"]], on="niche", how="left")
    agent_summary_df = agent_summary_df.merge(biological_df[["niche", "niche_label"]], on="niche", how="left")
    composition_df.to_csv(output_dir / "niche_composition.csv", index=False)
    gate_df.to_csv(output_dir / "gate_reliability_by_niche.csv", index=False)
    agent_summary_df.to_csv(output_dir / "agent_attention_by_niche.csv", index=False)

    coords = pd.DataFrame(batch.coords.cpu().numpy(), index=metadata.spot_ids, columns=["x", "y"])
    map_paths = {
        "niche_map": _write_spatial_map(output_dir, "niche_map", coords, assignments["niche"], cmap="tab10"),
    }
    neighborhood_summary_path = ""
    if batch.proportion_gt is not None and batch.coords is not None:
        target_df = pd.DataFrame(batch.proportion_gt.cpu().numpy(), index=metadata.spot_ids, columns=metadata.cell_types)
        has_gt = target_df.sum(axis=1) > 1e-8
        gt_groups = _group_fraction_frame(target_df, NICHE_VALIDATION_GROUPS)
        pred_groups = _group_fraction_frame(pred, NICHE_VALIDATION_GROUPS)
        edge_index, _ = build_knn_graph(batch.coords.cpu(), k=6)
        neighbor_lookup: dict[str, set[str]] = {spot_id: {spot_id} for spot_id in metadata.spot_ids}
        src_ids = np.array(metadata.spot_ids)[edge_index[0].long().numpy()]
        dst_ids = np.array(metadata.spot_ids)[edge_index[1].long().numpy()]
        for src_id, dst_id in zip(src_ids, dst_ids):
            neighbor_lookup[src_id].add(dst_id)
        neighborhood_rows = []
        for spot_id in metadata.spot_ids:
            supervised_neighbors = [neighbor_id for neighbor_id in neighbor_lookup[spot_id] if bool(has_gt.loc[neighbor_id])]
            if supervised_neighbors:
                neighborhood = gt_groups.loc[supervised_neighbors].mean(axis=0)
                neighborhood_entropy = float(_row_entropy(pd.DataFrame([neighborhood], index=[spot_id])).iloc[0])
                neighborhood_dominant = str(neighborhood.idxmax())
            else:
                neighborhood = pd.Series(0.0, index=gt_groups.columns)
                neighborhood_entropy = 0.0
                neighborhood_dominant = "none"
            predicted_dominant = str(pred_groups.loc[spot_id].idxmax())
            spot_dominant = str(gt_groups.loc[spot_id].idxmax()) if bool(has_gt.loc[spot_id]) else "none"
            row = {
                "spot_id": spot_id,
                "niche": int(assignments.loc[spot_id, "niche"]),
                "niche_label": str(assignments.loc[spot_id, "niche_label"]),
                "has_xenium_ground_truth": bool(has_gt.loc[spot_id]),
                "num_xenium_neighbors": int(len(supervised_neighbors)),
                "predicted_dominant_group": predicted_dominant,
                "xenium_spot_dominant_group": spot_dominant,
                "xenium_neighborhood_dominant_group": neighborhood_dominant,
                "spot_group_agreement": bool(predicted_dominant == spot_dominant) if bool(has_gt.loc[spot_id]) else False,
                "neighborhood_group_agreement": bool(predicted_dominant == neighborhood_dominant) if supervised_neighbors else False,
                "xenium_neighborhood_entropy": neighborhood_entropy,
            }
            row.update({f"predicted_{group}_fraction": float(pred_groups.loc[spot_id, group]) for group in pred_groups.columns})
            row.update({f"xenium_spot_{group}_fraction": float(gt_groups.loc[spot_id, group]) for group in gt_groups.columns})
            row.update({f"xenium_neighborhood_{group}_fraction": float(neighborhood[group]) for group in gt_groups.columns})
            neighborhood_rows.append(row)
        neighborhood_df = pd.DataFrame(neighborhood_rows)
        neighborhood_df.to_csv(output_dir / "niche_xenium_neighborhood_validation.csv", index=False)
        summary_rows = []
        for niche, group in neighborhood_df.groupby("niche"):
            valid_neighbors = group["num_xenium_neighbors"] > 0
            summary_rows.append(
                {
                    "niche": int(niche),
                    "niche_label": str(group["niche_label"].iloc[0]),
                    "num_spots": int(len(group)),
                    "num_spots_with_xenium_gt": int(group["has_xenium_ground_truth"].sum()),
                    "num_spots_with_xenium_neighbors": int(valid_neighbors.sum()),
                    "spot_group_agreement_rate": float(group.loc[group["has_xenium_ground_truth"], "spot_group_agreement"].mean())
                    if group["has_xenium_ground_truth"].any()
                    else 0.0,
                    "neighborhood_group_agreement_rate": float(group.loc[valid_neighbors, "neighborhood_group_agreement"].mean()) if valid_neighbors.any() else 0.0,
                    "dominant_xenium_neighborhood_group": str(
                        group.loc[valid_neighbors, "xenium_neighborhood_dominant_group"].mode().iloc[0] if valid_neighbors.any() else "none"
                    ),
                    "mean_xenium_neighborhood_entropy": float(group.loc[valid_neighbors, "xenium_neighborhood_entropy"].mean()) if valid_neighbors.any() else 0.0,
                }
            )
        neighborhood_summary = pd.DataFrame(summary_rows)
        neighborhood_summary.to_csv(output_dir / "niche_xenium_neighborhood_summary.csv", index=False)
        neighborhood_summary_path = str(output_dir / "niche_xenium_neighborhood_summary.csv")

    summary = {
        "num_niches": n_niches,
        "niche_labels": {str(k): v for k, v in label_map.items()},
        "biological_summary_path": str(output_dir / "niche_biological_summary.csv"),
        "xenium_neighborhood_validation_path": str(output_dir / "niche_xenium_neighborhood_validation.csv")
        if neighborhood_summary_path
        else "",
        "xenium_neighborhood_summary_path": neighborhood_summary_path,
        "map_paths": map_paths,
    }
    (output_dir / "niche_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_nature_analysis(
    prepared_path: str | Path,
    run_dir: str | Path,
    output_dir: str | Path,
    comparison_run_dir: str | Path | None = None,
) -> dict[str, object]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = Path(comparison_run_dir) / "predicted_proportions.csv" if comparison_run_dir is not None else None
    results = {
        "proportion_maps": proportion_map_analysis(
            prepared_path,
            run_dir / "predicted_proportions.csv",
            output_dir,
        ),
        "reliability": reliability_analysis(
            prepared_path,
            run_dir / "predicted_proportions.csv",
            run_dir / "spot_uncertainty.csv",
            run_dir / "gate_weights.csv",
            output_dir,
            modality_reliability_path=run_dir / "modality_reliability.csv",
        ),
        "boundary": boundary_analysis(
            prepared_path,
            run_dir / "predicted_proportions.csv",
            output_dir,
            comparison_predictions_path=comparison_path,
        ),
        "niche": niche_analysis(
            prepared_path,
            run_dir / "predicted_proportions.csv",
            run_dir / "gate_weights.csv",
            run_dir / "agent_attention.csv",
            output_dir,
            uncertainty_path=run_dir / "spot_uncertainty.csv",
            modality_reliability_path=run_dir / "modality_reliability.csv",
        ),
    }
    (output_dir / "nature_analysis_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reliability, boundary, and niche analyses.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--comparison-run-dir", default=None)
    args = parser.parse_args()
    results = run_nature_analysis(args.prepared, args.run_dir, args.output_dir, args.comparison_run_dir)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
