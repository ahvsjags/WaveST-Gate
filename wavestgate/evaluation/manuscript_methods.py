"""Generate evidence-linked manuscript Methods text for WaveST-Gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("results/nature_manuscript_methods")
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


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _safe_float(value)
    if number is None:
        return str(value) if value not in (None, "") else "not available"
    return f"{number:.{digits}g}"


def _first_row(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _row_by_value(frame: pd.DataFrame, column: str, value: str) -> dict[str, Any]:
    if frame.empty or column not in frame.columns:
        return {}
    hits = frame.loc[frame[column].astype(str).eq(value)]
    if hits.empty:
        return {}
    return hits.iloc[0].to_dict()


def _evidence(path: str | Path, role: str) -> dict[str, Any]:
    path = Path(path)
    return {"role": role, "path": str(path), "exists": path.exists()}


def _section(title: str, text: str, evidence: list[dict[str, Any]], metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"title": title, "text": text, "metrics": metrics or {}, "evidence": evidence}


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# WaveST-Gate Methods And Statistical Analysis Draft",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        "",
        "This draft is generated from the current evidence files. Refresh it after changing data, models, baselines, figures, tables, or release artifacts.",
        "",
    ]
    for section in payload["sections"]:
        lines.extend([f"## {section['title']}", "", section["text"], ""])
        if section["metrics"]:
            lines.append("Key values:")
            for key, value in section["metrics"].items():
                lines.append(f"- `{key}`: `{value}`")
            lines.append("")
        lines.append("Evidence:")
        for evidence in section["evidence"]:
            exists = "yes" if evidence["exists"] else "no"
            lines.append(f"- {evidence['role']}: `{evidence['path']}` (exists: {exists})")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_manuscript_methods(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    tables_dir: str | Path = DEFAULT_TABLES_DIR,
    run_dir: str | Path = DEFAULT_RUN_DIR,
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    environment_report_path: str | Path = "results/nature_release/environment_report.json",
    statements_dir: str | Path = "results/nature_manuscript_statements",
) -> dict[str, Any]:
    """Create a manuscript Methods/Statistical Analysis draft from evidence."""

    output_dir = Path(output_dir)
    tables_dir = Path(tables_dir)
    run_dir = Path(run_dir)
    benchmark_dir = Path(benchmark_dir)
    environment_report_path = Path(environment_report_path)
    statements_dir = Path(statements_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = _read_json(benchmark_dir / "xenium_visium_benchmark_manifest.json")
    benchmark_table = _read_csv(tables_dir / "table_1_benchmark_summary.csv")
    model_table = _read_csv(tables_dir / "table_2_main_model_performance.csv")
    baseline_table = _read_csv(tables_dir / "table_3_baseline_comparison.csv")
    ablation_table = _read_csv(tables_dir / "table_4_ablation_delta.csv")
    evidence_table = _read_csv(tables_dir / "table_5_reliability_boundary_niche.csv")
    external_table = _read_csv(tables_dir / "table_6_external_generalization.csv")
    robustness_table = _read_csv(tables_dir / "table_7_robustness_summary.csv")
    split_table = _read_csv(tables_dir / "table_8_split_sensitivity.csv")
    imagegate_table = _read_csv(tables_dir / "table_9_imagegate_supplement.csv")
    benchmark_sensitivity_table = _read_csv(tables_dir / "table_10_benchmark_sensitivity.csv")
    rep1_budget_table = _read_csv(tables_dir / "table_11_rep1_retune_budget_curve.csv")
    environment = _read_json(environment_report_path)

    main_metrics = _first_row(_read_csv(run_dir / "metrics.csv"))
    imagegate_dir = run_dir.parent / f"{run_dir.name}_imagegate"
    noimage_dir = run_dir.parent / f"{run_dir.name}_imagegate_noimage"
    imagegate_metrics = _first_row(_read_csv(imagegate_dir / "metrics.csv"))
    noimage_metrics = _first_row(_read_csv(noimage_dir / "metrics.csv"))
    image_contribution = _read_json(imagegate_dir / "image_contribution" / "image_contribution_summary.json")
    top_model = _row_by_value(baseline_table, "method", "WaveST-Gate") or _first_row(baseline_table)
    strongest_baseline = {}
    if not baseline_table.empty and "method" in baseline_table.columns:
        non_model = baseline_table.loc[~baseline_table["method"].astype(str).eq("WaveST-Gate")]
        strongest_baseline = _first_row(non_model)
    full_ablation = _row_by_value(ablation_table, "ablation", "full")
    ablation_names = ablation_table["ablation"].astype(str).tolist() if "ablation" in ablation_table.columns else []
    methods = baseline_table["method"].astype(str).tolist() if "method" in baseline_table.columns else []
    original_val_split = _first_row(
        split_table.loc[
            split_table.get("analysis", pd.Series(dtype=str)).astype(str).eq("primary_spatial_holdout")
            & split_table.get("split", pd.Series(dtype=str)).astype(str).eq("val")
        ]
    )
    original_test_split = _first_row(
        split_table.loc[
            split_table.get("analysis", pd.Series(dtype=str)).astype(str).eq("primary_spatial_holdout")
            & split_table.get("split", pd.Series(dtype=str)).astype(str).eq("test")
        ]
    )
    stratified_val = _first_row(
        split_table.loc[
            split_table.get("analysis", pd.Series(dtype=str)).astype(str).eq("supplementary_gt_stratified_sensitivity")
            & split_table.get("split", pd.Series(dtype=str)).astype(str).eq("val")
        ]
    )
    primary_radius_row = _first_row(
        benchmark_sensitivity_table.loc[
            benchmark_sensitivity_table.get("analysis", pd.Series(dtype=str)).astype(str).eq("radius_cell_count_metric")
            & pd.to_numeric(benchmark_sensitivity_table.get("radius", pd.Series(dtype=float)), errors="coerce").eq(55.0)
            & pd.to_numeric(benchmark_sensitivity_table.get("min_xenium_cells", pd.Series(dtype=float)), errors="coerce").eq(1.0)
        ]
    )
    rep1_zero = _first_row(rep1_budget_table.loc[rep1_budget_table.get("budget_steps", pd.Series(dtype=str)).astype(str).isin({"0", "0.0"})])
    rep1_beating = pd.DataFrame()
    if not rep1_budget_table.empty and "beats_best_baseline" in rep1_budget_table.columns:
        rep1_beating = rep1_budget_table.loc[rep1_budget_table["beats_best_baseline"].astype(str).str.lower().isin({"true", "1"})].copy()
        if "budget_steps" in rep1_beating.columns:
            rep1_beating["_budget_numeric"] = pd.to_numeric(rep1_beating["budget_steps"], errors="coerce")
            rep1_beating = rep1_beating.sort_values("_budget_numeric")
    rep1_first_beating = _first_row(rep1_beating)
    paired_p_values = []
    if "paired_permutation_p" in baseline_table.columns:
        paired_p_values = [
            _safe_float(value)
            for value in baseline_table["paired_permutation_p"].tolist()
            if _safe_float(value) is not None
        ]
    environment_torch = environment.get("torch") if isinstance(environment.get("torch"), dict) else {}
    devices = environment_torch.get("devices") if isinstance(environment_torch.get("devices"), list) else []
    gpu_names = [str(device.get("name", "")) for device in devices if isinstance(device, dict) and device.get("name")]

    sections = [
        _section(
            "Xenium-to-Visium benchmark construction",
            (
                "Xenium cells were aggregated into Visium/CytAssist spot neighborhoods using a fixed radius in the aligned coordinate system. "
                f"The current benchmark contains {benchmark.get('num_spots', 'NA')} spots, {benchmark.get('num_spots_with_ground_truth', 'NA')} "
                f"Xenium-supervised spots, {benchmark.get('num_cells', 'NA')} typed Xenium cells, {benchmark.get('num_cell_types', 'NA')} cell types, "
                f"and radius {benchmark.get('spot_radius', 'NA')}. Cell counts, normalized proportions, QC fields, entropy, dominant cell type, "
                "coordinate/radius metadata, and deterministic train/validation/test splits are stored as independent artifacts."
            ),
            [
                _evidence(benchmark_dir / "xenium_visium_benchmark_manifest.json", "benchmark manifest"),
                _evidence(benchmark_dir / "xenium_cell_counts.csv", "spot-level cell counts"),
                _evidence(benchmark_dir / "xenium_cell_proportions.csv", "spot-level proportions"),
                _evidence(benchmark_dir / "spot_ground_truth_qc.csv", "spot QC and entropy"),
                _evidence(benchmark_dir / "spot_splits.csv", "fixed spot split"),
                _evidence("docs/xenium_to_visium_benchmark_protocol.md", "benchmark protocol"),
            ],
            {
                "spots": benchmark.get("num_spots", ""),
                "xenium_supervised_spots": benchmark.get("num_spots_with_ground_truth", ""),
                "xenium_cells": benchmark.get("num_cells", ""),
                "cell_types": benchmark.get("num_cell_types", ""),
                "spot_radius": benchmark.get("spot_radius", ""),
            },
        ),
        _section(
            "Split sensitivity and benchmark confidence controls",
            (
                "The primary split is kept as the original deterministic spatial holdout to preserve the pre-specified benchmark. "
                f"Its validation split contains {_fmt(original_val_split.get('num_supervised_spots'))} Xenium-supervised spots, while the test split contains {_fmt(original_test_split.get('num_supervised_spots'))}; therefore validation is documented as an unsupervised spatial holdout rather than a supervised model-selection endpoint. "
                f"A supplementary GT-stratified split sensitivity analysis generated five seeds with supervised validation/test spots; the validation mean JSD was {_fmt(stratified_val.get('mean_jsd'))} across a mean of {_fmt(stratified_val.get('num_supervised_spots'))} supervised validation spots. "
                "Ground-truth construction was also stress-tested across aggregation radii 45/55/65/75 and Xenium cell-count thresholds 1/5/10/20/50. "
                f"The primary radius 55, minimum-cell threshold 1 setting yielded {_fmt(primary_radius_row.get('num_spots_passing_threshold'))} supervised spots and JSD {_fmt(primary_radius_row.get('jsd'))}."
            ),
            [
                _evidence("wavestgate/evaluation/split_sensitivity.py", "split sensitivity CLI"),
                _evidence(run_dir / "split_sensitivity" / "original_split_gt_summary.csv", "primary split GT inventory"),
                _evidence(run_dir / "split_sensitivity" / "split_sensitivity_aggregate.csv", "GT-stratified split aggregate"),
                _evidence("wavestgate/evaluation/benchmark_sensitivity.py", "radius/cell-count sensitivity CLI"),
                _evidence(run_dir / "benchmark_sensitivity" / "radius_cell_count_sensitivity.csv", "radius and cell-count metric table"),
                _evidence(tables_dir / "table_8_split_sensitivity.csv", "manuscript split sensitivity table"),
                _evidence(tables_dir / "table_10_benchmark_sensitivity.csv", "manuscript benchmark sensitivity table"),
            ],
            {
                "original_val_supervised_spots": _fmt(original_val_split.get("num_supervised_spots")),
                "original_test_supervised_spots": _fmt(original_test_split.get("num_supervised_spots")),
                "gt_stratified_val_mean_jsd": _fmt(stratified_val.get("mean_jsd")),
                "primary_radius55_min1_jsd": _fmt(primary_radius_row.get("jsd")),
                "primary_radius55_min1_supervised_spots": _fmt(primary_radius_row.get("num_spots_passing_threshold")),
            },
        ),
        _section(
            "WaveST-Gate model",
            (
                "WaveST-Gate integrates H&E image patches, spot expression, and scRNA-derived cell-type prototypes. "
                "The image branch uses a pure-PyTorch wavelet-guided morphology encoder to represent low-frequency tissue context and directional high-frequency boundary/texture information. "
                "The expression branch encodes spot-level gene expression with an MLP, while reference prototypes are transformed into cell-type agents that interact with spot features. "
                "A cross-modal reliability gate predicts spot-wise modality weights for image, expression, and reference-agent features; local cross-modal refinement is applied before decoders produce cell-type proportions, reconstructed expression, uncertainty, attention, and optional niche logits."
            ),
            [
                _evidence("wavestgate/models/wavestgate.py", "model composition"),
                _evidence("wavestgate/models/wavelet_encoder.py", "wavelet morphology encoder"),
                _evidence("wavestgate/models/expression_encoder.py", "expression encoder"),
                _evidence("wavestgate/models/celltype_agents.py", "cell-type prototype agents"),
                _evidence("wavestgate/models/cross_modal_gate.py", "cross-modal reliability gate"),
                _evidence("wavestgate/models/local_refinement.py", "local cross-modal refinement"),
                _evidence("wavestgate/models/deconv_decoder.py", "proportion and expression decoders"),
            ],
        ),
        _section(
            "Training objective and optimization",
            (
                "The real-data training entry point combines expression reconstruction, supervised proportion loss where Xenium-derived labels exist, entropy/sparsity regularization, spatial smoothness, boundary-aware constraints, contrastive alignment, uncertainty calibration, and optional niche supervision. "
                "An optional image-gate participation term is available for morphology-contribution controls; it uses a hinge target for calibrated and raw image gates, with higher targets in morphology-boundary regions, and is disabled unless explicitly requested in the training configuration. "
                f"The main run finished at step {_fmt(main_metrics.get('step'))} with JSD {_fmt(main_metrics.get('jsd'))}, spotwise cosine {_fmt(main_metrics.get('spotwise_cosine'))}, "
                f"mean cell-type Pearson {_fmt(main_metrics.get('mean_celltype_pearson'))}, and expression log1p RMSE {_fmt(main_metrics.get('expression_log1p_rmse'))}."
            ),
            [
                _evidence("wavestgate/training/train_real.py", "real-data training CLI"),
                _evidence("wavestgate/training/losses.py", "loss functions"),
                _evidence(run_dir / "training_history.csv", "training history"),
                _evidence(run_dir / "metrics.csv", "main metrics"),
                _evidence(run_dir / "checkpoint.pt", "trained checkpoint"),
            ],
            {
                "step": _fmt(main_metrics.get("step")),
                "jsd": _fmt(main_metrics.get("jsd")),
                "spotwise_cosine": _fmt(main_metrics.get("spotwise_cosine")),
                "mean_celltype_pearson": _fmt(main_metrics.get("mean_celltype_pearson")),
                "expression_log1p_rmse": _fmt(main_metrics.get("expression_log1p_rmse")),
            },
        ),
        _section(
            "Image modality contribution control",
            (
                "To test whether the H&E branch contributed usable information rather than a nominal module, an image-gate-enhanced control was trained with light modality dropout and the image-gate participation term, then compared with a matched no-image control using paired Xenium-supervised spot errors. "
                f"The enhanced image run had mean calibrated image gate {_fmt(image_contribution.get('mean_image_gate'))}, raw image gate {_fmt(image_contribution.get('mean_raw_image_gate'))}, JSD {_fmt(imagegate_metrics.get('jsd'))}, and mean cell-type Pearson {_fmt(imagegate_metrics.get('mean_celltype_pearson'))}. "
                f"The matched no-image control had JSD {_fmt(noimage_metrics.get('jsd'))} and mean cell-type Pearson {_fmt(noimage_metrics.get('mean_celltype_pearson'))}. "
                f"The paired no-image-minus-image JSD improvement was {_fmt(image_contribution.get('mean_paired_jsd_improvement_noimage_minus_imagegate'))} overall and {_fmt(image_contribution.get('high_texture_mean_paired_jsd_improvement'))} in the highest H&E texture quartile."
            ),
            [
                _evidence("experiments/nature_main/train_cytassist_rep2_radius55_imagegate.yaml", "image-gate enhanced training config"),
                _evidence("experiments/nature_main/train_cytassist_rep2_radius55_imagegate_noimage.yaml", "matched no-image control config"),
                _evidence("wavestgate/training/losses.py", "image-gate participation loss"),
                _evidence("wavestgate/evaluation/image_contribution.py", "image contribution analysis CLI"),
                _evidence(imagegate_dir / "metrics.csv", "image-gate enhanced metrics"),
                _evidence(noimage_dir / "metrics.csv", "matched no-image metrics"),
                _evidence(imagegate_dir / "image_contribution" / "image_contribution_summary.json", "paired image contribution summary"),
                _evidence(imagegate_dir / "image_contribution" / "image_contribution_texture_groups.csv", "texture-stratified image contribution table"),
                _evidence(tables_dir / "table_9_imagegate_supplement.csv", "manuscript image-gate supplement table"),
            ],
            {
                "imagegate_mean_image_gate": _fmt(image_contribution.get("mean_image_gate")),
                "imagegate_jsd": _fmt(imagegate_metrics.get("jsd")),
                "noimage_jsd": _fmt(noimage_metrics.get("jsd")),
                "paired_jsd_improvement": _fmt(
                    image_contribution.get("mean_paired_jsd_improvement_noimage_minus_imagegate")
                ),
                "high_texture_paired_jsd_improvement": _fmt(
                    image_contribution.get("high_texture_mean_paired_jsd_improvement")
                ),
            },
        ),
        _section(
            "Baseline comparison and fairness",
            (
                "All baseline methods were evaluated against the same Xenium-derived spot-level ground truth, shared reference/gene panel where applicable, identical supervised spots, and common metrics. "
                f"The comparison table contains {len(methods)} methods: {', '.join(methods)}. "
                f"WaveST-Gate ranked {_fmt(top_model.get('rank_by_jsd'))} by JSD with JSD {_fmt(top_model.get('jsd'))}; "
                f"the strongest non-WaveST-Gate row was {strongest_baseline.get('method', 'not available')} with JSD {_fmt(strongest_baseline.get('jsd'))}. "
                "Runtime, peak memory, bootstrap mean/standard deviation, and paired permutation tests are recorded for auditability."
            ),
            [
                _evidence(run_dir / "baseline_comparison" / "baseline_comparison.csv", "baseline comparison table"),
                _evidence(run_dir / "baseline_comparison" / "baseline_split_bootstrap_summary.csv", "bootstrap mean/std summary"),
                _evidence(run_dir / "baseline_comparison" / "baseline_bootstrap_paired_improvement.csv", "paired bootstrap/permutation evidence"),
                _evidence(run_dir / "baseline_environment_audit.json", "baseline environment audit"),
                _evidence(tables_dir / "table_3_baseline_comparison.csv", "manuscript baseline table"),
            ],
            {
                "num_methods": len(methods),
                "wavestgate_jsd": _fmt(top_model.get("jsd")),
                "best_baseline": strongest_baseline.get("method", ""),
                "best_baseline_jsd": _fmt(strongest_baseline.get("jsd")),
                "min_paired_permutation_p": _fmt(min(paired_p_values) if paired_p_values else None),
            },
        ),
        _section(
            "Ablation design",
            (
                "Ablations tested whether each component was necessary rather than simply increasing model size. "
                f"The panel contains {len(ablation_names)} settings: {', '.join(ablation_names)}. "
                f"The full ablation run reached JSD {_fmt(full_ablation.get('jsd'))}; the no-gate/mean-fusion and raw-gate-without-uncertainty settings quantify the reliability gate and calibration contribution, while wavelet, agent, boundary, modality-only, and refinement settings isolate the proposed biological modeling components."
            ),
            [
                _evidence(run_dir / "ablations250" / "ablation_summary.csv", "ablation summary"),
                _evidence(run_dir / "ablations250" / "configs", "ablation configs"),
                _evidence(tables_dir / "table_4_ablation_delta.csv", "manuscript ablation table"),
            ],
            {"num_ablations": len(ablation_names), "full_jsd": _fmt(full_ablation.get("jsd"))},
        ),
        _section(
            "Reliability, boundary, and niche analyses",
            (
                "Reliability analyses relate predicted uncertainty to spot-level error, risk-coverage behavior, calibration bins, modality gate maps, and failure cases. "
                "Boundary analyses compare tumor-stroma, ductal, and immune-edge transitions against interior regions and smoothness-only controls. "
                "Niche analyses combine predicted cell composition, marker enrichment, H&E/pathology correspondence, Xenium neighborhood validation, gate reliability by niche, and agent attention by niche."
            ),
            [
                _evidence(run_dir / "nature_analysis" / "reliability_summary.json", "reliability summary"),
                _evidence(run_dir / "nature_analysis" / "risk_coverage_curve.csv", "risk coverage curve"),
                _evidence(run_dir / "nature_analysis" / "uncertainty_calibration_bins.csv", "calibration bins"),
                _evidence(run_dir / "nature_analysis" / "boundary_summary.json", "boundary summary"),
                _evidence(run_dir / "nature_analysis" / "boundary_type_summary.csv", "boundary type summary"),
                _evidence(run_dir / "nature_analysis" / "niche_summary.json", "niche summary"),
                _evidence(run_dir / "nature_analysis" / "niche_biological_summary.csv", "niche biological summary"),
                _evidence(tables_dir / "table_5_reliability_boundary_niche.csv", "manuscript reliability/boundary/niche table"),
            ],
            {row.get("metric", f"metric_{idx}"): row.get("value", "") for idx, row in evidence_table.iterrows()},
        ),
        _section(
            "External generalization and robustness",
            (
                "External experiments cover no-retuning predictions, Rep1-to-Rep2/matched-ground-truth evaluation, minimal retuning, external pathology validation, and multi-sample summaries. "
                f"The external summary includes {len(external_table)} rows. For Rep1, no-retuning is reported as a domain-shift case with held-out test JSD {_fmt(rep1_zero.get('jsd'))} versus best baseline {rep1_zero.get('best_baseline_method', 'not available')} JSD {_fmt(rep1_zero.get('best_baseline_jsd'))}. "
                f"The smallest minimal-retuning budget that beat the best baseline was {_fmt(rep1_first_beating.get('budget_steps'))} steps with JSD {_fmt(rep1_first_beating.get('jsd'))}. "
                "Robustness experiments include patch sizes 32/64/128/256, common/top/marker gene panels, dropout, reference mismatch and missing cell types, prototype perturbations, H&E stain perturbation, low-cell-count/low-expression subgroups, split sensitivity, and radius/cell-count benchmark sensitivity."
            ),
            [
                _evidence(tables_dir / "table_6_external_generalization.csv", "external generalization table"),
                _evidence(tables_dir / "table_7_robustness_summary.csv", "robustness table"),
                _evidence(tables_dir / "table_11_rep1_retune_budget_curve.csv", "Rep1 retune budget table"),
                _evidence("results/nature_external_no_retuning/external_no_retuning_summary.csv", "no-retuning external summary"),
                _evidence("results/nature_external_matched_gt/external_matched_gt_summary.csv", "matched GT external summary"),
                _evidence("results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/rep1_retune_budget_curve/rep1_minimal_retune_budget_curve.csv", "Rep1 no-retuning/minimal-retuning budget curve"),
                _evidence(run_dir / "robustness" / "robustness_summary.csv", "robustness summary"),
                _evidence(run_dir / "patch_size_robustness" / "patch_size_summary.csv", "patch-size robustness"),
            ],
            {
                "external_rows": len(external_table),
                "robustness_scenarios": len(robustness_table),
                "rep1_no_retune_jsd": _fmt(rep1_zero.get("jsd")),
                "rep1_min_budget_beating_best_baseline": _fmt(rep1_first_beating.get("budget_steps")),
                "rep1_min_budget_jsd": _fmt(rep1_first_beating.get("jsd")),
            },
        ),
        _section(
            "Statistical analysis, reproducibility, and compute",
            (
                "Primary proportion metrics were JSD, spot-wise cosine similarity, mean cell-type Pearson correlation, and RMSE. "
                "Baseline uncertainty was summarized with split/bootstrap statistics and paired permutation tests; reliability was evaluated with uncertainty-error correlations, risk-coverage curves, and calibration bins. "
                f"The recorded compute environment used PyTorch {environment_torch.get('version', 'not available')}, CUDA version {environment_torch.get('cuda_version', 'not available')}, "
                f"{environment_torch.get('device_count', 'not available')} GPU(s)"
                + (f" ({', '.join(gpu_names)})" if gpu_names else "")
                + ". The final release chain regenerates tables, figures, statements, bundle, verification, readiness, and handoff artifacts."
            ),
            [
                _evidence(environment_report_path, "environment report"),
                _evidence(statements_dir / "manuscript_availability_statements.json", "availability and reproducibility statements"),
                _evidence("results/nature_release/release_verification.json", "release verification"),
                _evidence("results/nature_submission_readiness/readiness_report.json", "submission readiness audit"),
                _evidence("results/nature_release/final_submission_handoff.json", "final handoff manifest"),
            ],
            {
                "torch": environment_torch.get("version", ""),
                "cuda_version": environment_torch.get("cuda_version", ""),
                "gpu_count": environment_torch.get("device_count", ""),
                "gpu_names": ", ".join(gpu_names),
            },
        ),
    ]

    required_paths = [
        benchmark_dir / "xenium_visium_benchmark_manifest.json",
        run_dir / "metrics.csv",
        run_dir / "baseline_comparison" / "baseline_comparison.csv",
        run_dir / "ablations250" / "ablation_summary.csv",
        tables_dir / "table_5_reliability_boundary_niche.csv",
        tables_dir / "table_6_external_generalization.csv",
        tables_dir / "table_7_robustness_summary.csv",
        tables_dir / "table_8_split_sensitivity.csv",
        tables_dir / "table_9_imagegate_supplement.csv",
        tables_dir / "table_10_benchmark_sensitivity.csv",
        tables_dir / "table_11_rep1_retune_budget_curve.csv",
        environment_report_path,
    ]
    status = "complete" if all(path.exists() for path in required_paths) and len(sections) >= 9 else "partial"
    json_path = output_dir / "manuscript_methods.json"
    markdown_path = output_dir / "manuscript_methods.md"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "outputs": {"json": str(json_path), "markdown": str(markdown_path)},
        "required_paths": [str(path) for path in required_paths],
        "missing_required_paths": [str(path) for path in required_paths if not path.exists()],
        "sections": sections,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(payload, markdown_path)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build WaveST-Gate manuscript Methods and statistical analysis draft.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--environment-report-path", type=Path, default=Path("results/nature_release/environment_report.json"))
    parser.add_argument("--statements-dir", type=Path, default=Path("results/nature_manuscript_statements"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    payload = build_manuscript_methods(
        output_dir=args.output_dir,
        tables_dir=args.tables_dir,
        run_dir=args.run_dir,
        benchmark_dir=args.benchmark_dir,
        environment_report_path=args.environment_report_path,
        statements_dir=args.statements_dir,
    )
    print(json.dumps(payload["outputs"], indent=2), flush=True)


if __name__ == "__main__":
    main()
