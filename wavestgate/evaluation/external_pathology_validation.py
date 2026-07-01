"""Validate external no-retuning predictions against pathology metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from wavestgate.data.prepare_dataset import load_prepared_dataset
PATHOLOGY_GROUP_RULES = {
    "tumor": ["cancer", "invasive", "tumor", "dcis", "carcinoma"],
    "immune": ["lymphocyte", "immune", "t cell", "b cell"],
    "stromal": ["stroma", "stromal", "adipose", "normal"],
}

PATHOLOGY_CELL_TYPE_GROUPS = {
    "tumor": ["DCIS 1", "DCIS 2", "Invasive Tumor", "Prolif Invasive Tumor", "T Cell & Tumor Hybrid"],
    "immune": ["B Cells", "CD4+ T Cells", "CD8+ T Cells", "IRF7+ DCs", "LAMP3+ DCs", "Macrophages 1", "Macrophages 2", "Mast Cells"],
    "stromal": ["Endothelial", "Perivascular-Like", "Stromal", "Stromal & T Cell Hybrid", "Myoepi ACTA2+", "Myoepi KRT15+"],
}


def _read_prediction_table(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "spot_id" in frame.columns:
        frame = frame.set_index("spot_id")
    else:
        frame = frame.set_index(frame.columns[0])
    frame.index = frame.index.astype(str)
    return frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _read_metadata_table(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "spot_id" in frame.columns:
        frame = frame.set_index("spot_id")
    else:
        frame = frame.set_index(frame.columns[0])
    frame.index = frame.index.astype(str)
    return frame


def _group_fraction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    used: set[str] = set()
    for group, cell_types in PATHOLOGY_CELL_TYPE_GROUPS.items():
        present = [cell_type for cell_type in cell_types if cell_type in frame.columns]
        used.update(present)
        rows[group] = frame[present].sum(axis=1) if present else pd.Series(0.0, index=frame.index)
    other_cols = [cell_type for cell_type in frame.columns if cell_type not in used]
    rows["other"] = frame[other_cols].sum(axis=1) if other_cols else pd.Series(0.0, index=frame.index)
    grouped = pd.DataFrame(rows, index=frame.index)
    totals = grouped.sum(axis=1).replace(0.0, np.nan)
    return grouped.div(totals, axis=0).fillna(0.0)


def _expected_groups(classification: str) -> list[str]:
    text = str(classification).lower()
    if "artefact" in text or "artifact" in text:
        return ["other"]
    groups = []
    for group, keywords in PATHOLOGY_GROUP_RULES.items():
        if any(keyword in text for keyword in keywords):
            groups.append(group)
    return groups or ["other"]


def _safe_write_barplot(frame: pd.DataFrame, output_path: Path) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    if frame.empty:
        return ""
    plot_frame = frame.set_index("classification")[["mean_tumor_fraction", "mean_immune_fraction", "mean_stromal_fraction", "mean_other_fraction"]]
    plot_frame = plot_frame.sort_index()
    ax = plot_frame.plot(kind="bar", stacked=True, figsize=(9, 4.5), width=0.8)
    ax.set_ylabel("Mean predicted fraction")
    ax.set_xlabel("Pathology class")
    ax.legend(["tumor", "immune", "stromal", "other"], bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=180)
    plt.close(ax.figure)
    return str(output_path)


def _safe_write_niche_heatmap(frame: pd.DataFrame, output_path: Path) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    if frame.empty:
        return ""
    pivot = frame.pivot(index="classification", columns="external_niche", values="fraction_spots").fillna(0.0)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(pivot))), dpi=180)
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=max(float(pivot.to_numpy().max()), 1e-8))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("External predicted niche")
    ax.set_ylabel("Pathology class")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


def _dataset_records(
    dataset: str,
    prepared_dir: Path,
    prediction_dir: Path,
) -> pd.DataFrame:
    metadata_path = prepared_dir / "spot_metadata.csv"
    predictions_path = prediction_dir / "predicted_proportions.csv"
    gates_path = prediction_dir / "gate_weights.csv"
    uncertainty_path = prediction_dir / "spot_uncertainty.csv"
    agent_path = prediction_dir / "agent_attention.csv"
    if not metadata_path.exists() or not predictions_path.exists():
        return pd.DataFrame()
    meta = _read_metadata_table(metadata_path)
    pred = _read_prediction_table(predictions_path)
    gates = _read_prediction_table(gates_path) if gates_path.exists() else pd.DataFrame(index=pred.index)
    uncertainty = _read_prediction_table(uncertainty_path) if uncertainty_path.exists() else pd.DataFrame(index=pred.index)
    agent = _read_prediction_table(agent_path) if agent_path.exists() else pd.DataFrame(index=pred.index)
    batch, batch_meta, _ = load_prepared_dataset(prepared_dir / "prepared.pt", device="cpu") if (prepared_dir / "prepared.pt").exists() else (None, None, {})
    expression_total = pd.Series(dtype=float)
    if batch is not None:
        expression_total = pd.Series(batch.st_expression.cpu().sum(dim=1).numpy(), index=batch_meta.spot_ids)

    common = meta.index.intersection(pred.index)
    if len(common) == 0 or "Classification" not in meta.columns:
        return pd.DataFrame()
    pred = pred.reindex(common)
    grouped = _group_fraction_frame(pred)
    records = pd.DataFrame(index=common)
    records["dataset"] = dataset
    records["patientid"] = meta.reindex(common).get("patientid", pd.Series("", index=common)).astype(str)
    records["subtype"] = meta.reindex(common).get("subtype", pd.Series("", index=common)).astype(str)
    records["classification"] = meta.reindex(common)["Classification"].astype(str)
    for col in grouped.columns:
        records[f"predicted_{col}_fraction"] = grouped[col]
    records["predicted_dominant_group"] = grouped.idxmax(axis=1)
    records["expected_groups"] = records["classification"].map(lambda value: ";".join(_expected_groups(value)))
    records["pathology_group_agreement"] = [
        dominant in expected.split(";") for dominant, expected in zip(records["predicted_dominant_group"], records["expected_groups"])
    ]
    if not gates.empty:
        for col in gates.columns:
            records[f"gate_{col}"] = gates.reindex(common)[col]
    if not uncertainty.empty:
        column = "spot_uncertainty" if "spot_uncertainty" in uncertainty.columns else uncertainty.columns[0]
        records["spot_uncertainty"] = uncertainty.reindex(common)[column]
    if not agent.empty:
        records["top_agent"] = agent.reindex(common).idxmax(axis=1)
        records["top_agent_attention"] = agent.reindex(common).max(axis=1)
    if not expression_total.empty:
        records["total_expression"] = expression_total.reindex(common)
    records.index.name = "spot_id"
    return records.reset_index()


def validate_external_pathology(
    processed_root: str | Path = "data/processed",
    external_results_dir: str | Path = "results/nature_external_no_retuning",
    output_dir: str | Path = "results/nature_external_pathology_validation",
    dataset_prefix: str = "wu_swarbrick",
    n_niches: int = 5,
) -> dict[str, str]:
    """Compare external predictions with public pathology classification metadata."""

    processed_root = Path(processed_root)
    external_results_dir = Path(external_results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_frames = []
    for prediction_dir in sorted(external_results_dir.glob(f"{dataset_prefix}*")):
        if not prediction_dir.is_dir():
            continue
        prepared_dir = processed_root / prediction_dir.name
        frame = _dataset_records(prediction_dir.name, prepared_dir, prediction_dir)
        if not frame.empty:
            dataset_frames.append(frame)
    if not dataset_frames:
        raise ValueError("No external pathology metadata/prediction pairs found")

    spot_df = pd.concat(dataset_frames, ignore_index=True)
    features = spot_df[["predicted_tumor_fraction", "predicted_immune_fraction", "predicted_stromal_fraction", "predicted_other_fraction"]].to_numpy()
    n_niches = max(1, min(int(n_niches), len(spot_df)))
    spot_df["external_niche"] = KMeans(n_clusters=n_niches, random_state=17, n_init=20).fit_predict(features)
    spot_path = output_dir / "pathology_spot_predictions.csv"
    spot_df.to_csv(spot_path, index=False)

    summary_rows = []
    for classification, group in spot_df.groupby("classification"):
        expected = sorted({value for values in group["expected_groups"].str.split(";") for value in values})
        summary_rows.append(
            {
                "classification": classification,
                "expected_groups": ";".join(expected),
                "num_spots": int(len(group)),
                "num_datasets": int(group["dataset"].nunique()),
                "agreement_rate": float(group["pathology_group_agreement"].mean()),
                "dominant_predicted_group": str(
                    group[["predicted_tumor_fraction", "predicted_immune_fraction", "predicted_stromal_fraction", "predicted_other_fraction"]]
                    .mean()
                    .idxmax()
                    .replace("predicted_", "")
                    .replace("_fraction", "")
                ),
                "mean_tumor_fraction": float(group["predicted_tumor_fraction"].mean()),
                "mean_immune_fraction": float(group["predicted_immune_fraction"].mean()),
                "mean_stromal_fraction": float(group["predicted_stromal_fraction"].mean()),
                "mean_other_fraction": float(group["predicted_other_fraction"].mean()),
                "mean_spot_uncertainty": float(group["spot_uncertainty"].mean()) if "spot_uncertainty" in group.columns else 0.0,
                "mean_expression_gate": float(group["gate_expression"].mean()) if "gate_expression" in group.columns else 0.0,
                "mean_image_gate": float(group["gate_image"].mean()) if "gate_image" in group.columns else 0.0,
                "mean_reference_gate": float(group["gate_reference"].mean()) if "gate_reference" in group.columns else 0.0,
            }
        )
    class_summary = pd.DataFrame(summary_rows).sort_values(["num_spots", "classification"], ascending=[False, True])
    class_path = output_dir / "pathology_class_summary.csv"
    class_summary.to_csv(class_path, index=False)

    patient_summary = (
        spot_df.groupby(["dataset", "patientid", "subtype", "classification"], dropna=False)
        .agg(
            num_spots=("spot_id", "count"),
            agreement_rate=("pathology_group_agreement", "mean"),
            mean_tumor_fraction=("predicted_tumor_fraction", "mean"),
            mean_immune_fraction=("predicted_immune_fraction", "mean"),
            mean_stromal_fraction=("predicted_stromal_fraction", "mean"),
            mean_other_fraction=("predicted_other_fraction", "mean"),
        )
        .reset_index()
    )
    patient_path = output_dir / "pathology_patient_summary.csv"
    patient_summary.to_csv(patient_path, index=False)

    niche_rows = []
    for niche, group in spot_df.groupby("external_niche"):
        class_counts = group["classification"].value_counts(normalize=True)
        niche_rows.append(
            {
                "external_niche": int(niche),
                "num_spots": int(len(group)),
                "dominant_pathology_class": str(class_counts.index[0]),
                "dominant_pathology_fraction": float(class_counts.iloc[0]),
                "agreement_rate": float(group["pathology_group_agreement"].mean()),
                "mean_tumor_fraction": float(group["predicted_tumor_fraction"].mean()),
                "mean_immune_fraction": float(group["predicted_immune_fraction"].mean()),
                "mean_stromal_fraction": float(group["predicted_stromal_fraction"].mean()),
                "mean_other_fraction": float(group["predicted_other_fraction"].mean()),
                "mean_spot_uncertainty": float(group["spot_uncertainty"].mean()) if "spot_uncertainty" in group.columns else 0.0,
            }
        )
    niche_summary = pd.DataFrame(niche_rows).sort_values("external_niche")
    niche_path = output_dir / "pathology_niche_summary.csv"
    niche_summary.to_csv(niche_path, index=False)

    niche_by_class = (
        spot_df.groupby(["classification", "external_niche"], dropna=False)
        .size()
        .reset_index(name="num_spots")
    )
    totals = niche_by_class.groupby("classification")["num_spots"].transform("sum")
    niche_by_class["fraction_spots"] = niche_by_class["num_spots"] / totals.clip(lower=1)
    niche_by_class_path = output_dir / "pathology_niche_by_class.csv"
    niche_by_class.to_csv(niche_by_class_path, index=False)

    composition_plot = _safe_write_barplot(class_summary, output_dir / "pathology_group_composition.png")
    niche_heatmap = _safe_write_niche_heatmap(niche_by_class, output_dir / "pathology_niche_by_class_heatmap.png")
    manifest = {
        "processed_root": str(processed_root),
        "external_results_dir": str(external_results_dir),
        "output_dir": str(output_dir),
        "dataset_prefix": dataset_prefix,
        "num_datasets": int(spot_df["dataset"].nunique()),
        "num_spots": int(len(spot_df)),
        "num_pathology_classes": int(spot_df["classification"].nunique()),
        "overall_agreement_rate": float(spot_df["pathology_group_agreement"].mean()),
        "spot_predictions_path": str(spot_path),
        "class_summary_path": str(class_path),
        "patient_summary_path": str(patient_path),
        "niche_summary_path": str(niche_path),
        "niche_by_class_path": str(niche_by_class_path),
        "composition_plot": composition_plot,
        "niche_heatmap": niche_heatmap,
        "notes": (
            "Agreement maps public Wu/Swarbrick pathology classification keywords to tumor/immune/stromal/other groups. "
            "This validates biological/pathology correspondence for external no-retuning predictions, not deconvolution ground-truth accuracy."
        ),
    }
    manifest_path = output_dir / "external_pathology_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "spot_predictions_path": str(spot_path),
        "class_summary_path": str(class_path),
        "patient_summary_path": str(patient_path),
        "niche_summary_path": str(niche_path),
        "niche_by_class_path": str(niche_by_class_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate external predictions against pathology classification metadata.")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--external-results-dir", default="results/nature_external_no_retuning")
    parser.add_argument("--output-dir", default="results/nature_external_pathology_validation")
    parser.add_argument("--dataset-prefix", default="wu_swarbrick")
    parser.add_argument("--n-niches", type=int, default=5)
    args = parser.parse_args()
    outputs = validate_external_pathology(
        processed_root=args.processed_root,
        external_results_dir=args.external_results_dir,
        output_dir=args.output_dir,
        dataset_prefix=args.dataset_prefix,
        n_niches=args.n_niches,
    )
    print(json.dumps(outputs, indent=2), flush=True)


if __name__ == "__main__":
    main()
