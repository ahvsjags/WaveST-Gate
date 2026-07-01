"""Generate evidence-linked manuscript figure legends for WaveST-Gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("results/nature_manuscript_figure_legends")
DEFAULT_FIGURES_DIR = Path("results/nature_manuscript_figures")
DEFAULT_TABLES_DIR = Path("results/nature_manuscript_tables")
DEFAULT_RUN_DIR = Path("results/nature_main/cytassist_rep2_radius55")
DEFAULT_BENCHMARK_DIR = Path("data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55")


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


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float(value)
    if number is None:
        return str(value) if value not in (None, "") else "not available"
    return f"{number:.{digits}g}"


def _row(frame: pd.DataFrame, column: str, value: str) -> dict[str, Any]:
    if frame.empty or column not in frame.columns:
        return {}
    hits = frame.loc[frame[column].astype(str).eq(value)]
    if hits.empty:
        return {}
    return hits.iloc[0].to_dict()


def _metric_value(frame: pd.DataFrame, key_col: str, value_col: str, key: str) -> Any:
    row = _row(frame, key_col, key)
    return row.get(value_col, "")


def _evidence(path: str | Path, role: str) -> dict[str, Any]:
    path = Path(path)
    return {"role": role, "path": str(path), "exists": path.exists()}


def _legend(
    figure: str,
    title: str,
    legend: str,
    key_findings: list[str],
    evidence: list[dict[str, Any]],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "figure": figure,
        "title": title,
        "legend": legend,
        "key_findings": key_findings,
        "metrics": metrics or {},
        "evidence": evidence,
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# WaveST-Gate Figure Legends And Claim Evidence",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        "",
        "These legends are generated from the current figure manifest, manuscript tables, and evidence files. Refresh after changing figures or results.",
        "",
    ]
    for entry in payload["legends"]:
        lines.extend([f"## {entry['figure']}. {entry['title']}", "", entry["legend"], ""])
        if entry["key_findings"]:
            lines.append("Key findings:")
            for finding in entry["key_findings"]:
                lines.append(f"- {finding}")
            lines.append("")
        if entry["metrics"]:
            lines.append("Key values:")
            for key, value in entry["metrics"].items():
                lines.append(f"- `{key}`: `{value}`")
            lines.append("")
        lines.append("Evidence:")
        for evidence in entry["evidence"]:
            exists = "yes" if evidence["exists"] else "no"
            lines.append(f"- {evidence['role']}: `{evidence['path']}` (exists: {exists})")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_manuscript_figure_legends(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    tables_dir: str | Path = DEFAULT_TABLES_DIR,
    run_dir: str | Path = DEFAULT_RUN_DIR,
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    methods_dir: str | Path = "results/nature_manuscript_methods",
) -> dict[str, Any]:
    """Build figure legends with explicit result and artifact evidence."""

    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    tables_dir = Path(tables_dir)
    run_dir = Path(run_dir)
    benchmark_dir = Path(benchmark_dir)
    methods_dir = Path(methods_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_manifest_path = figures_dir / "figure_manifest.json"
    figure_manifest = _read_json(figure_manifest_path)
    benchmark = _read_json(benchmark_dir / "xenium_visium_benchmark_manifest.json")
    main_table = _read_csv(tables_dir / "table_2_main_model_performance.csv")
    baseline_table = _read_csv(tables_dir / "table_3_baseline_comparison.csv")
    evidence_table = _read_csv(tables_dir / "table_5_reliability_boundary_niche.csv")
    robustness_table = _read_csv(tables_dir / "table_7_robustness_summary.csv")

    wavestgate = _row(baseline_table, "method", "WaveST-Gate")
    non_model = baseline_table.loc[~baseline_table["method"].astype(str).eq("WaveST-Gate")] if not baseline_table.empty and "method" in baseline_table.columns else pd.DataFrame()
    best_baseline = non_model.iloc[0].to_dict() if not non_model.empty else {}
    p_values = [_float(value) for value in baseline_table.get("paired_permutation_p", pd.Series(dtype=float)).tolist()]
    p_values = [value for value in p_values if value is not None]

    def ev_metric(metric: str) -> str:
        return _fmt(_metric_value(evidence_table, "metric", "value", metric))

    figure_paths = {
        "Figure 1": figures_dir / "figure_1_workflow_schematic.png",
        "Figure 2": figures_dir / "figure_2_spatial_cell_composition.png",
        "Figure 3": figures_dir / "figure_3_baseline_performance.png",
        "Figure 4": figures_dir / "figure_4_reliability_calibration.png",
        "Figure 5": figures_dir / "figure_5_boundary_niche_pathology.png",
        "Supplementary Figure S1": figures_dir / "supplementary_figure_s1_robustness.png",
    }
    robustness_scenarios = robustness_table["scenario"].astype(str).tolist() if "scenario" in robustness_table.columns else []

    legends = [
        _legend(
            "Figure 1",
            "Reproducible Xenium-to-Visium benchmark and WaveST-Gate workflow",
            (
                "Workflow schematic for the WaveST-Gate study. Xenium single-cell annotations are spatially aggregated into Visium/CytAssist spot neighborhoods to create a reproducible breast cancer deconvolution benchmark. "
                "H&E image patches, spot expression, and scRNA-derived prototype agents are then processed by wavelet morphology encoding, reference-aware agents, local cross-modal refinement, and reliability-calibrated fusion to produce cell-type proportions, expression reconstruction, uncertainty, attention, and niche outputs."
            ),
            [
                f"Benchmark contains {benchmark.get('num_spots', 'NA')} spots, {benchmark.get('num_spots_with_ground_truth', 'NA')} Xenium-supervised spots, {benchmark.get('num_cells', 'NA')} typed cells, and {benchmark.get('num_cell_types', 'NA')} cell types.",
                f"Aggregation radius is fixed at {benchmark.get('spot_radius', 'NA')}.",
            ],
            [
                _evidence(figure_paths["Figure 1"], "Figure 1 image"),
                _evidence(benchmark_dir / "xenium_visium_benchmark_manifest.json", "benchmark manifest"),
                _evidence("docs/xenium_to_visium_benchmark_protocol.md", "benchmark protocol"),
                _evidence("wavestgate/models/wavestgate.py", "model implementation"),
                _evidence(methods_dir / "manuscript_methods.json", "methods draft"),
            ],
            {
                "spots": benchmark.get("num_spots", ""),
                "supervised_spots": benchmark.get("num_spots_with_ground_truth", ""),
                "xenium_cells": benchmark.get("num_cells", ""),
                "cell_types": benchmark.get("num_cell_types", ""),
                "radius": benchmark.get("spot_radius", ""),
            },
        ),
        _legend(
            "Figure 2",
            "Spatial cell-type composition predicted by WaveST-Gate",
            (
                "Spatial maps summarize WaveST-Gate cell-type deconvolution on the real breast cancer benchmark, including high-priority cell-type maps and tumor/immune/stromal group-level organization. "
                "The maps are linked to the model prediction files, Xenium-supervised evaluation metrics, and proportion-map manifest."
            ),
            [
                f"Main model JSD is {_fmt(_metric_value(main_table, 'metric', 'value', 'JSD'))} with spotwise cosine {_fmt(_metric_value(main_table, 'metric', 'value', 'spotwise_cosine'))}.",
                f"Proportion maps cover {_fmt(_metric_value(main_table, 'metric', 'value', 'proportion_map_cell_types'))} cell types and {_fmt(_metric_value(main_table, 'metric', 'value', 'priority_cell_types_mapped'))} priority mapped cell types.",
            ],
            [
                _evidence(figure_paths["Figure 2"], "Figure 2 image"),
                _evidence(run_dir / "predicted_proportions.csv", "predicted proportions"),
                _evidence(run_dir / "metrics.csv", "main model metrics"),
                _evidence(run_dir / "nature_analysis" / "proportion_maps" / "proportion_map_manifest.json", "proportion map manifest"),
                _evidence(tables_dir / "table_2_main_model_performance.csv", "main model manuscript table"),
            ],
            {
                "jsd": _fmt(_metric_value(main_table, "metric", "value", "JSD")),
                "spotwise_cosine": _fmt(_metric_value(main_table, "metric", "value", "spotwise_cosine")),
                "mean_celltype_pearson": _fmt(_metric_value(main_table, "metric", "value", "mean_celltype_pearson")),
                "proportion_map_cell_types": _fmt(_metric_value(main_table, "metric", "value", "proportion_map_cell_types")),
            },
        ),
        _legend(
            "Figure 3",
            "Formal baseline comparison on matched Xenium-supervised spots",
            (
                "Bar plot comparing WaveST-Gate with formal and simple deconvolution baselines using the same supervised spots, reference/gene panel where applicable, and shared metrics. "
                "The comparison records runtime, peak memory, bootstrap mean and standard deviation, and paired permutation evidence."
            ),
            [
                f"WaveST-Gate ranks {_fmt(wavestgate.get('rank_by_jsd'))} of {len(baseline_table)} methods by JSD.",
                f"WaveST-Gate JSD is {_fmt(wavestgate.get('jsd'))}; the strongest non-WaveST-Gate row is {best_baseline.get('method', 'not available')} with JSD {_fmt(best_baseline.get('jsd'))}.",
                f"Minimum paired permutation p-value across recorded baselines is {_fmt(min(p_values) if p_values else None)}.",
            ],
            [
                _evidence(figure_paths["Figure 3"], "Figure 3 image"),
                _evidence(run_dir / "baseline_comparison" / "baseline_comparison.csv", "baseline comparison table"),
                _evidence(run_dir / "baseline_comparison" / "baseline_split_bootstrap_summary.csv", "bootstrap summary"),
                _evidence(run_dir / "baseline_comparison" / "baseline_bootstrap_paired_improvement.csv", "paired statistics"),
                _evidence(tables_dir / "table_3_baseline_comparison.csv", "manuscript baseline table"),
            ],
            {
                "num_methods": len(baseline_table),
                "wavestgate_jsd": _fmt(wavestgate.get("jsd")),
                "best_baseline": best_baseline.get("method", ""),
                "best_baseline_jsd": _fmt(best_baseline.get("jsd")),
                "min_paired_permutation_p": _fmt(min(p_values) if p_values else None),
            },
        ),
        _legend(
            "Figure 4",
            "Reliability-calibrated modality fusion and uncertainty analysis",
            (
                "Reliability panels show H&E, expression, and reference gate maps together with spot-level uncertainty, calibration bins, and risk-coverage behavior. "
                "The figure supports the interpretation that WaveST-Gate learns calibrated modality reliability rather than a static attention mixture."
            ),
            [
                f"Uncertainty-error Pearson correlation is {ev_metric('uncertainty_error_pearson')}.",
                f"Calibration-bin Pearson correlation is {ev_metric('calibration_bin_pearson')}.",
            ],
            [
                _evidence(figure_paths["Figure 4"], "Figure 4 image"),
                _evidence(run_dir / "gate_weights.csv", "gate weights"),
                _evidence(run_dir / "spot_uncertainty.csv", "spot uncertainty"),
                _evidence(run_dir / "nature_analysis" / "reliability_summary.json", "reliability summary"),
                _evidence(run_dir / "nature_analysis" / "risk_coverage_curve.csv", "risk coverage curve"),
                _evidence(run_dir / "nature_analysis" / "uncertainty_calibration_bins.csv", "calibration bins"),
            ],
            {
                "uncertainty_error_pearson": ev_metric("uncertainty_error_pearson"),
                "calibration_bin_pearson": ev_metric("calibration_bin_pearson"),
            },
        ),
        _legend(
            "Figure 5",
            "Boundary preservation and tumor-immune-stromal niche interpretation",
            (
                "Boundary and niche panels evaluate whether morphology-aware constraints preserve tissue transitions and whether predicted cell composition, marker enrichment, pathology correspondence, Xenium neighborhoods, gate reliability, and agent attention support interpretable tumor-immune-stromal niches."
            ),
            [
                f"Boundary-to-interior jump ratio is {ev_metric('boundary_to_interior_jump_ratio')}.",
                f"The analysis identifies {ev_metric('num_niches')} biological niches.",
                f"External pathology-class agreement is {ev_metric('spot_weighted_pathology_agreement_rate')}.",
            ],
            [
                _evidence(figure_paths["Figure 5"], "Figure 5 image"),
                _evidence(run_dir / "nature_analysis" / "boundary_summary.json", "boundary summary"),
                _evidence(run_dir / "nature_analysis" / "niche_summary.json", "niche summary"),
                _evidence(run_dir / "nature_analysis" / "niche_biological_summary.csv", "niche biological summary"),
                _evidence(run_dir / "nature_analysis" / "niche_xenium_neighborhood_summary.csv", "Xenium neighborhood validation"),
                _evidence(tables_dir / "table_5_reliability_boundary_niche.csv", "reliability/boundary/niche table"),
            ],
            {
                "boundary_to_interior_jump_ratio": ev_metric("boundary_to_interior_jump_ratio"),
                "num_niches": ev_metric("num_niches"),
                "pathology_agreement": ev_metric("spot_weighted_pathology_agreement_rate"),
            },
        ),
        _legend(
            "Supplementary Figure S1",
            "Robustness to realistic input and reference perturbations",
            (
                "Supplementary robustness panel summarizes model behavior under patch-size changes, gene-panel restrictions, gene dropout, reference mismatch, missing cell types, prototype perturbation, H&E stain perturbation, low-cell-count or low-expression subgroups, and alternative splits."
            ),
            [
                f"Robustness summary contains {len(robustness_scenarios)} scenario rows: {', '.join(robustness_scenarios)}.",
                "Patch-size robustness includes 32, 64, 128, and 256 pixel settings.",
            ],
            [
                _evidence(figure_paths["Supplementary Figure S1"], "Supplementary Figure S1 image"),
                _evidence(run_dir / "robustness" / "robustness_summary.csv", "robustness summary"),
                _evidence(run_dir / "robustness" / "robustness_manifest.json", "robustness manifest"),
                _evidence(run_dir / "patch_size_robustness" / "patch_size_summary.csv", "patch-size robustness"),
                _evidence(tables_dir / "table_7_robustness_summary.csv", "manuscript robustness table"),
            ],
            {"robustness_rows": len(robustness_scenarios), "scenarios": "; ".join(robustness_scenarios)},
        ),
    ]

    figure_failures = int(figure_manifest.get("num_fail", 1) or 0)
    all_evidence = [evidence for entry in legends for evidence in entry["evidence"]]
    missing_evidence = [item["path"] for item in all_evidence if not item["exists"]]
    status = "complete" if len(legends) == 6 and figure_failures == 0 and not missing_evidence else "partial"
    json_path = output_dir / "figure_legends.json"
    markdown_path = output_dir / "figure_legends.md"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "figure_manifest": str(figure_manifest_path),
        "figure_manifest_status": {
            "num_figures": figure_manifest.get("num_figures", 0),
            "num_pass": figure_manifest.get("num_pass", 0),
            "num_fail": figure_manifest.get("num_fail", 0),
        },
        "outputs": {"json": str(json_path), "markdown": str(markdown_path)},
        "missing_evidence": missing_evidence,
        "legends": legends,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(payload, markdown_path)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build WaveST-Gate manuscript figure legends and claim evidence.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--methods-dir", type=Path, default=Path("results/nature_manuscript_methods"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    payload = build_manuscript_figure_legends(
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
        run_dir=args.run_dir,
        benchmark_dir=args.benchmark_dir,
        methods_dir=args.methods_dir,
    )
    print(json.dumps(payload["outputs"], indent=2), flush=True)


if __name__ == "__main__":
    main()
