"""Generate a datasheet and data dictionary for the Xenium-to-Visium benchmark."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BENCHMARK_DIR = Path("data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55")
DEFAULT_OUTPUT_DIR = Path("results/nature_benchmark_datasheet")


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _artifact(path: Path, role: str, frame: pd.DataFrame | None = None) -> dict[str, Any]:
    if frame is None:
        frame = _read_csv(path) if path.suffix.lower() == ".csv" else pd.DataFrame()
    return {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "rows": int(len(frame)) if not frame.empty else 0,
        "columns": list(frame.columns) if not frame.empty else [],
        "num_columns": int(len(frame.columns)) if not frame.empty else 0,
    }


def _numeric_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"n": 0}
    return {
        "n": int(values.shape[0]),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _truthy_series(series: pd.Series, index: pd.Index | None = None) -> pd.Series:
    """Return a robust boolean series for CSV-loaded truth values."""

    if series.empty:
        return pd.Series(False, index=index if index is not None else pd.RangeIndex(0))
    if series.dtype == bool:
        values = series.fillna(False).astype(bool)
    else:
        values = series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    if index is not None:
        values = values.reindex(index, fill_value=False)
    return values.astype(bool)


def _column_dictionary() -> list[dict[str, str]]:
    return [
        {"file": "xenium_cell_counts.csv", "column": "spot_id", "description": "Visium/CytAssist spot barcode."},
        {"file": "xenium_cell_counts.csv", "column": "<cell type>", "description": "Integer number of Xenium cells assigned to this spot and cell type."},
        {"file": "xenium_cell_proportions.csv", "column": "spot_id", "description": "Visium/CytAssist spot barcode."},
        {"file": "xenium_cell_proportions.csv", "column": "<cell type>", "description": "Cell-type proportion among Xenium cells aggregated to this spot; all zero for spots without Xenium ground truth."},
        {"file": "spot_ground_truth_qc.csv", "column": "xenium_cell_count", "description": "Total aggregated Xenium cell count in the spot neighborhood."},
        {"file": "spot_ground_truth_qc.csv", "column": "ground_truth_entropy", "description": "Entropy of spot-level Xenium cell-type proportions."},
        {"file": "spot_ground_truth_qc.csv", "column": "dominant_cell_type", "description": "Cell type with the largest aggregated proportion for the spot."},
        {"file": "spot_ground_truth_qc.csv", "column": "has_xenium_ground_truth", "description": "Whether the spot has at least one aggregated Xenium cell."},
        {"file": "spot_splits.csv", "column": "split", "description": "Deterministic spatial holdout split label: train, val, or test."},
    ]


def _check_integrity(manifest: dict[str, Any], counts: pd.DataFrame, proportions: pd.DataFrame, qc: pd.DataFrame, splits: pd.DataFrame) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    expected_spots = int(manifest.get("num_spots", 0) or 0)
    expected_cell_types = int(manifest.get("num_cell_types", 0) or 0)
    frames = {"counts": counts, "proportions": proportions, "qc": qc, "splits": splits}
    for name, frame in frames.items():
        checks.append({"name": f"{name}_row_count_matches_manifest", "status": "pass" if len(frame) == expected_spots else "fail", "observed": int(len(frame)), "expected": expected_spots})
    if all("spot_id" in frame.columns for frame in frames.values()):
        spot_sets = {name: set(frame["spot_id"].astype(str)) for name, frame in frames.items()}
        reference = next(iter(spot_sets.values()))
        checks.append({"name": "spot_id_sets_match_across_artifacts", "status": "pass" if all(values == reference for values in spot_sets.values()) else "fail"})
    cell_cols = [col for col in counts.columns if col != "spot_id"]
    prop_cols = [col for col in proportions.columns if col != "spot_id"]
    checks.append({"name": "cell_type_columns_match_counts_and_proportions", "status": "pass" if cell_cols == prop_cols else "fail", "observed": len(cell_cols), "expected": len(prop_cols)})
    checks.append({"name": "cell_type_count_matches_manifest", "status": "pass" if len(cell_cols) == expected_cell_types else "fail", "observed": len(cell_cols), "expected": expected_cell_types})
    if cell_cols and "xenium_cell_count" in qc.columns:
        count_totals = counts[cell_cols].sum(axis=1).astype(float).to_numpy()
        qc_totals = pd.to_numeric(qc["xenium_cell_count"], errors="coerce").fillna(-1).astype(float).to_numpy()
        checks.append({"name": "qc_cell_count_matches_count_matrix", "status": "pass" if np.allclose(count_totals, qc_totals) else "fail"})
        if "has_xenium_ground_truth" in qc.columns:
            has_gt = _truthy_series(qc["has_xenium_ground_truth"])
            checks.append({"name": "gt_flag_matches_positive_cell_count", "status": "pass" if bool(np.array_equal(has_gt.to_numpy(), count_totals > 0)) else "fail"})
    if prop_cols:
        row_sums = proportions[prop_cols].sum(axis=1).astype(float)
        checks.append({"name": "proportion_rows_sum_to_one_or_zero", "status": "pass" if bool(((row_sums.sub(1.0).abs() < 1e-5) | (row_sums.abs() < 1e-5)).all()) else "fail"})
        if "has_xenium_ground_truth" in qc.columns and len(qc) == len(proportions):
            has_gt = _truthy_series(qc["has_xenium_ground_truth"])
            expected_sums = has_gt.astype(float)
            checks.append({"name": "proportion_row_sums_match_gt_flag", "status": "pass" if bool(np.allclose(row_sums.to_numpy(), expected_sums.to_numpy(), atol=1e-5)) else "fail"})
    if "split" in splits.columns:
        allowed = {"train", "val", "test"}
        observed = set(splits["split"].astype(str))
        checks.append({"name": "split_labels_are_expected", "status": "pass" if observed.issubset(allowed) else "fail", "observed": sorted(observed)})
    manifest_artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    for name, artifact_path in manifest_artifacts.items():
        exists = Path(str(artifact_path)).exists()
        checks.append({"name": f"manifest_artifact_{name}_exists", "status": "pass" if exists else "fail", "path": str(artifact_path)})
    return checks


def _cell_type_summary(counts: pd.DataFrame, proportions: pd.DataFrame, qc: pd.DataFrame) -> list[dict[str, Any]]:
    cell_types = [col for col in counts.columns if col != "spot_id"]
    gt_mask = pd.Series(True, index=counts.index)
    if not qc.empty and {"spot_id", "has_xenium_ground_truth"}.issubset(qc.columns) and "spot_id" in counts.columns:
        qc_indexed = qc.set_index("spot_id")
        aligned_gt = _truthy_series(qc_indexed["has_xenium_ground_truth"], index=counts["spot_id"])
        gt_mask = pd.Series(aligned_gt.to_numpy(), index=counts.index)
    rows = []
    for cell_type in cell_types:
        count_values = pd.to_numeric(counts[cell_type], errors="coerce").fillna(0)
        prop_values = pd.to_numeric(proportions[cell_type], errors="coerce").fillna(0) if cell_type in proportions.columns else pd.Series(dtype=float)
        prop_gt = prop_values.loc[gt_mask] if len(prop_values) == len(gt_mask) else prop_values[prop_values > 0]
        rows.append(
            {
                "cell_type": cell_type,
                "total_xenium_cells": int(count_values.sum()),
                "spots_with_cells": int((count_values > 0).sum()),
                "mean_proportion_on_gt_spots": float(prop_gt.mean()) if not prop_gt.empty else 0.0,
            }
        )
    return rows


def _split_summary(splits: pd.DataFrame, qc: pd.DataFrame) -> list[dict[str, Any]]:
    if splits.empty or "split" not in splits.columns:
        return []
    merged = splits.copy()
    if not qc.empty and {"spot_id", "has_xenium_ground_truth"}.issubset(qc.columns):
        merged = merged.merge(qc[["spot_id", "has_xenium_ground_truth"]], on="spot_id", how="left")
    rows = []
    for split, frame in merged.groupby("split", dropna=False):
        rows.append(
            {
                "split": str(split),
                "num_spots": int(len(frame)),
                "num_spots_with_ground_truth": int(_truthy_series(frame["has_xenium_ground_truth"]).sum()) if "has_xenium_ground_truth" in frame.columns else 0,
            }
        )
    return sorted(rows, key=lambda row: row["split"])


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# WaveST-Gate Xenium-to-Visium Benchmark Datasheet",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Overview",
        "",
        f"- Spots: `{payload['overview'].get('num_spots', '')}`",
        f"- Xenium-supervised spots: `{payload['overview'].get('num_spots_with_ground_truth', '')}`",
        f"- Xenium cells: `{payload['overview'].get('num_cells', '')}`",
        f"- Cell types: `{payload['overview'].get('num_cell_types', '')}`",
        f"- Spot radius: `{payload['overview'].get('spot_radius', '')}`",
        "",
        "## Artifacts",
        "",
    ]
    for artifact in payload["artifacts"]:
        lines.append(f"- {artifact['role']}: `{artifact['path']}` ({artifact['rows']} rows, {artifact['num_columns']} columns, exists: {artifact['exists']})")
    lines.extend(["", "## Integrity Checks", ""])
    for check in payload["integrity_checks"]:
        lines.append(f"- `{check['name']}`: `{check['status']}`")
    lines.extend(["", "## Split Summary", ""])
    for row in payload["split_summary"]:
        lines.append(f"- `{row['split']}`: {row['num_spots']} spots, {row['num_spots_with_ground_truth']} with ground truth")
    lines.extend(["", "## QC Summary", ""])
    for key, value in payload["qc_summary"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Cell Types", ""])
    for row in payload["cell_type_summary"]:
        lines.append(f"- `{row['cell_type']}`: {row['total_xenium_cells']} cells across {row['spots_with_cells']} spots")
    lines.extend(["", "## Data Dictionary", ""])
    for row in payload["column_dictionary"]:
        lines.append(f"- `{row['file']}` / `{row['column']}`: {row['description']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_benchmark_datasheet(
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    protocol_path: str | Path = "docs/xenium_to_visium_benchmark_protocol.md",
) -> dict[str, Any]:
    """Build a datasheet and data dictionary for benchmark artifacts."""

    benchmark_dir = Path(benchmark_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = Path(protocol_path)
    manifest_path = benchmark_dir / "xenium_visium_benchmark_manifest.json"
    counts_path = benchmark_dir / "xenium_cell_counts.csv"
    proportions_path = benchmark_dir / "xenium_cell_proportions.csv"
    qc_path = benchmark_dir / "spot_ground_truth_qc.csv"
    splits_path = benchmark_dir / "spot_splits.csv"

    manifest = _read_json(manifest_path)
    counts = _read_csv(counts_path)
    proportions = _read_csv(proportions_path)
    qc = _read_csv(qc_path)
    splits = _read_csv(splits_path)

    artifacts = [
        _artifact(manifest_path, "benchmark manifest", pd.DataFrame()),
        _artifact(counts_path, "Xenium cell counts", counts),
        _artifact(proportions_path, "Xenium cell proportions", proportions),
        _artifact(qc_path, "spot ground-truth QC", qc),
        _artifact(splits_path, "deterministic spot splits", splits),
        {"role": "benchmark protocol", "path": str(protocol_path), "exists": protocol_path.exists(), "rows": 0, "columns": [], "num_columns": 0},
    ]
    integrity_checks = _check_integrity(manifest, counts, proportions, qc, splits)
    qc_summary: dict[str, Any] = {}
    if not qc.empty:
        gt_mask = _truthy_series(qc["has_xenium_ground_truth"]) if "has_xenium_ground_truth" in qc.columns else pd.Series(False, index=qc.index)
        gt_qc = qc.loc[gt_mask]
        qc_summary = {
            "num_spots": int(len(qc)),
            "num_spots_with_ground_truth": int(gt_mask.sum()) if "has_xenium_ground_truth" in qc.columns else 0,
            "fraction_spots_with_ground_truth": float(gt_mask.mean()) if len(gt_mask) else 0.0,
            "xenium_cell_count": _numeric_summary(qc.get("xenium_cell_count", pd.Series(dtype=float))),
            "xenium_cell_count_on_gt_spots": _numeric_summary(gt_qc.get("xenium_cell_count", pd.Series(dtype=float))),
            "ground_truth_entropy": _numeric_summary(qc.get("ground_truth_entropy", pd.Series(dtype=float))),
            "ground_truth_entropy_on_gt_spots": _numeric_summary(gt_qc.get("ground_truth_entropy", pd.Series(dtype=float))),
            "dominant_cell_type_counts_on_gt_spots": gt_qc.get("dominant_cell_type", pd.Series(dtype=str)).astype(str).value_counts().to_dict() if "dominant_cell_type" in gt_qc.columns else {},
        }
    failed_checks = [check for check in integrity_checks if check["status"] != "pass"]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if all(artifact["exists"] for artifact in artifacts) and not failed_checks else "partial",
        "benchmark_dir": str(benchmark_dir),
        "overview": {
            "protocol": manifest.get("protocol", ""),
            "num_spots": manifest.get("num_spots", ""),
            "num_cells": manifest.get("num_cells", ""),
            "num_cell_types": manifest.get("num_cell_types", ""),
            "num_spots_with_ground_truth": manifest.get("num_spots_with_ground_truth", ""),
            "spot_radius": manifest.get("spot_radius", ""),
            "columns": manifest.get("columns", {}),
            "split_protocol": manifest.get("split_protocol", {}),
            "metrics": manifest.get("metrics", []),
        },
        "artifacts": artifacts,
        "column_dictionary": _column_dictionary(),
        "cell_type_summary": _cell_type_summary(counts, proportions, qc) if not counts.empty else [],
        "split_summary": _split_summary(splits, qc),
        "qc_summary": qc_summary,
        "integrity_checks": integrity_checks,
        "failed_integrity_checks": failed_checks,
        "outputs": {
            "json": str(output_dir / "benchmark_datasheet.json"),
            "markdown": str(output_dir / "benchmark_datasheet.md"),
        },
    }
    json_path = output_dir / "benchmark_datasheet.json"
    markdown_path = output_dir / "benchmark_datasheet.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(payload, markdown_path)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a WaveST-Gate Xenium-to-Visium benchmark datasheet.")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--protocol-path", type=Path, default=Path("docs/xenium_to_visium_benchmark_protocol.md"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    payload = build_benchmark_datasheet(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        protocol_path=args.protocol_path,
    )
    print(json.dumps(payload["outputs"], indent=2), flush=True)


if __name__ == "__main__":
    main()
