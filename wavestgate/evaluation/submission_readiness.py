"""Submission readiness audit for WaveST-Gate evidence packages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


STATUS_ORDER = {"complete": 0, "optional_pending": 1, "partial": 2, "missing": 3}


@dataclass
class EvidenceRecord:
    stage: str
    requirement: str
    status: str
    evidence_path: str
    detail: str
    metric: str = ""


def _as_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path)


def _path_text(path: str | Path | None) -> str:
    if path is None:
        return ""
    return str(Path(path))


def _exists(path: str | Path | None) -> bool:
    return path is not None and Path(path).exists()


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _last_row(path: str | Path) -> dict[str, Any]:
    frame = _read_csv(path)
    if frame.empty:
        return {}
    return frame.iloc[-1].to_dict()


def _status_from_files(paths: Iterable[str | Path]) -> str:
    return "complete" if all(Path(path).exists() for path in paths) else "missing"


def _format_metric(values: dict[str, Any], keys: Iterable[str]) -> str:
    parts = []
    for key in keys:
        if key not in values:
            continue
        value = values[key]
        if pd.isna(value):
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.6g}")
        else:
            parts.append(f"{key}={value}")
    return "; ".join(parts)


DOI_PATTERN = re.compile(r"(?:doi\.org/)?10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def _has_doi(*paths: str | Path) -> bool:
    for path in paths:
        p = Path(path)
        if not p.exists() or p.is_dir():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore").lower()
        if DOI_PATTERN.search(text):
            return True
    return False


def _release_doi_from_json(path: str | Path) -> str:
    """Return a DOI only from explicit release/deposition fields.

    Zenodo metadata may cite upstream public datasets; those should not be
    counted as the WaveST-Gate release DOI.
    """

    payload = _read_json(path)
    if not payload:
        return ""
    for key in ("doi", "conceptdoi", "concept_doi"):
        value = payload.get(key)
        if isinstance(value, str) and DOI_PATTERN.search(value):
            return value
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        prereserved = metadata.get("prereserve_doi")
        if isinstance(prereserved, dict):
            value = prereserved.get("doi")
            if isinstance(value, str) and DOI_PATTERN.search(value):
                return value
    response = payload.get("deposition_response")
    if isinstance(response, dict):
        value = _release_doi_from_payload(response)
        if value:
            return value
    return ""


def _release_doi_from_payload(payload: dict[str, Any]) -> str:
    for key in ("doi", "conceptdoi", "concept_doi"):
        value = payload.get(key)
        if isinstance(value, str) and DOI_PATTERN.search(value):
            return value
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in ("doi", "conceptdoi", "concept_doi"):
            value = metadata.get(key)
            if isinstance(value, str) and DOI_PATTERN.search(value):
                return value
        prereserved = metadata.get("prereserve_doi")
        if isinstance(prereserved, dict):
            value = prereserved.get("doi")
            if isinstance(value, str) and DOI_PATTERN.search(value):
                return value
    return ""


def _release_deposition_summary(release_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    candidates = [
        release_dir / "zenodo_deposition_result.json",
        release_dir / "release_bundle_manifest.json",
    ]
    if output_dir is not None:
        candidates.append(output_dir / "zenodo_release_manifest.json")
    summary: dict[str, Any] = {}
    for path in candidates:
        payload = _read_json(path)
        if not payload:
            continue
        if not summary:
            summary = {
                "source_path": str(path),
                "release_status": payload.get("release_status", ""),
                "doi": _release_doi_from_json(path),
                "zenodo_deposition_id": payload.get("zenodo_deposition_id", ""),
                "zenodo_record_url": payload.get("zenodo_record_url", ""),
                "zenodo_deposition_result": payload.get("zenodo_deposition_result", str(path) if path.name == "zenodo_deposition_result.json" else ""),
            }
        doi = _release_doi_from_json(path)
        if doi:
            summary.update(
                {
                    "source_path": str(path),
                    "release_status": payload.get("release_status", "zenodo_doi_recorded"),
                    "doi": doi,
                    "zenodo_deposition_id": payload.get("zenodo_deposition_id", summary.get("zenodo_deposition_id", "")),
                    "zenodo_record_url": payload.get("zenodo_record_url", summary.get("zenodo_record_url", "")),
                    "zenodo_deposition_result": payload.get("zenodo_deposition_result", summary.get("zenodo_deposition_result", "")),
                }
            )
            break
    return summary


def _sha256(path: Path, max_bytes: int) -> tuple[str, str]:
    size = path.stat().st_size
    if size > max_bytes:
        return "", "skipped_large_file"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), "hashed"


def _stage_status(records: list[EvidenceRecord], stage: str) -> str:
    stage_records = [record for record in records if record.stage == stage]
    if not stage_records:
        return "missing"
    worst = max(stage_records, key=lambda record: STATUS_ORDER.get(record.status, 99))
    return worst.status


def _add(records: list[EvidenceRecord], stage: str, requirement: str, status: str, evidence_path: str | Path | None, detail: str, metric: str = "") -> None:
    records.append(
        EvidenceRecord(
            stage=stage,
            requirement=requirement,
            status=status,
            evidence_path=_path_text(evidence_path),
            detail=detail,
            metric=metric,
        )
    )


def _audit_stage_1(records: list[EvidenceRecord], benchmark_dir: Path, docs_dir: Path, output_dir: Path, release_dir: Path) -> None:
    stage = "1. Xenium-to-Visium benchmark"
    manifest_path = benchmark_dir / "xenium_visium_benchmark_manifest.json"
    counts_path = benchmark_dir / "xenium_cell_counts.csv"
    proportions_path = benchmark_dir / "xenium_cell_proportions.csv"
    qc_path = benchmark_dir / "spot_ground_truth_qc.csv"
    splits_path = benchmark_dir / "spot_splits.csv"
    protocol_path = docs_dir / "xenium_to_visium_benchmark_protocol.md"
    required_files = [manifest_path, counts_path, proportions_path, qc_path, splits_path, protocol_path]
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    metric = _format_metric(
        manifest,
        ["num_spots", "num_spots_with_ground_truth", "num_cells", "num_cell_types", "spot_radius"],
    )
    _add(
        records,
        stage,
        "spot-level Xenium aggregation artifacts",
        _status_from_files(required_files),
        manifest_path,
        "Counts, proportions, QC, splits, manifest, and protocol are required.",
        metric,
    )
    qc = _read_csv(qc_path)
    expected_qc = {"xenium_cell_count", "ground_truth_entropy", "dominant_cell_type", "has_xenium_ground_truth"}
    qc_status = "complete" if expected_qc.issubset(set(qc.columns)) else "missing"
    qc_detail = "QC includes cell coverage, entropy, dominant cell type, and GT coverage flag." if qc_status == "complete" else "QC columns are incomplete."
    _add(records, stage, "spot QC fields", qc_status, qc_path, qc_detail, f"columns={','.join(qc.columns)}" if not qc.empty else "")

    release_manifest_path = output_dir / "zenodo_release_manifest.json"
    deposition_summary = _release_deposition_summary(release_dir, output_dir)
    release_doi = str(deposition_summary.get("doi", ""))
    release_published = deposition_summary.get("release_status") == "zenodo_published"
    release_files = [
        release_dir / "release_bundle_manifest.json",
        release_dir / "zenodo_metadata.json",
        release_dir / "release_upload_manifest.csv",
        release_dir / "zenodo_deposition_instructions.md",
    ]
    bundle_manifests = _read_json(release_dir / "release_bundle_manifest.json") if (release_dir / "release_bundle_manifest.json").exists() else {}
    bundle_path = Path(str(bundle_manifests.get("bundle_path", ""))) if bundle_manifests.get("bundle_path") else release_dir / "wavestgate_submission_evidence_v0.1.0.tar.gz"
    release_files.append(bundle_path)
    _add(
        records,
        stage,
        "Zenodo-ready release bundle and metadata",
        _status_from_files(release_files),
        release_dir / "release_bundle_manifest.json",
        "Release bundle, upload manifest, Zenodo metadata, and deposition instructions are prepared.",
        _format_metric(bundle_manifests, ["num_files", "bundle_bytes"]),
    )
    verification_path = release_dir / "release_verification.json"
    verification = _read_json(verification_path) if verification_path.exists() else {}
    verification_ok = verification.get("bundle_integrity_status") == "passed" and int(verification.get("num_failures", 1) or 0) == 0
    _add(
        records,
        stage,
        "release bundle integrity verification",
        "complete" if verification_ok else ("partial" if verification_path.exists() else "missing"),
        verification_path,
        (
            "Release verifier checked tar readability, upload manifest members, critical artifacts, and dry-run bundle consistency."
            if verification_ok
            else "Release verifier has not confirmed bundle integrity."
        ),
        _format_metric(
            verification,
            ["overall_status", "bundle_integrity_status", "doi_status", "tar_member_count", "critical_artifacts_checked", "num_failures"],
        ),
    )
    environment_path = release_dir / "environment_report.json"
    environment = _read_json(environment_path) if environment_path.exists() else {}
    environment_ok = bool(environment.get("python")) and bool(environment.get("packages")) and bool(environment.get("torch"))
    _add(
        records,
        stage,
        "software and hardware environment report",
        "complete" if environment_ok and (release_dir / "environment_report.md").exists() else ("partial" if environment_path.exists() else "missing"),
        environment_path,
        (
            "Environment report records Python, package versions, PyTorch/CUDA/GPU, R packages, and baseline environment status."
            if environment_ok
            else "Environment report is missing or incomplete."
        ),
        _format_metric(
            {
                "python": (environment.get("python") or {}).get("version", "").split(" ")[0],
                "torch": (environment.get("torch") or {}).get("version", ""),
                "cuda_available": (environment.get("torch") or {}).get("cuda_available", ""),
                "gpu_count": (environment.get("torch") or {}).get("device_count", ""),
            },
            ["python", "torch", "cuda_available", "gpu_count"],
        ),
    )
    _add(
        records,
        stage,
        "Zenodo DOI / release deposition",
        "complete" if release_doi and release_published else "partial",
        deposition_summary.get("source_path") or release_manifest_path,
        (
            "A WaveST-Gate release DOI was found in explicit published Zenodo deposition fields."
            if release_doi and release_published
            else (
                "A Zenodo draft/reserved DOI exists, but it is not yet a public published release."
                if release_doi
                else "Release automation is present, but a real Zenodo DOI still must be deposited and filled in."
            )
        ),
        _format_metric(deposition_summary, ["release_status", "doi", "zenodo_deposition_id"]),
    )


def _audit_benchmark_datasheet(records: list[EvidenceRecord], datasheet_dir: Path) -> None:
    stage = "1. Xenium-to-Visium benchmark"
    json_path = datasheet_dir / "benchmark_datasheet.json"
    markdown_path = datasheet_dir / "benchmark_datasheet.md"
    payload = _read_json(json_path) if json_path.exists() else {}
    failed_checks = payload.get("failed_integrity_checks") if isinstance(payload.get("failed_integrity_checks"), list) else []
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), list) else []
    cell_types = payload.get("cell_type_summary") if isinstance(payload.get("cell_type_summary"), list) else []
    dictionary = payload.get("column_dictionary") if isinstance(payload.get("column_dictionary"), list) else []
    status = _status_from_files([json_path, markdown_path])
    if status == "complete":
        status = "complete" if payload.get("status") == "complete" and not failed_checks and artifacts and cell_types and dictionary else "partial"
    _add(
        records,
        stage,
        "benchmark datasheet and data dictionary",
        status,
        json_path,
        "Datasheet records artifact inventory, QC summary, split summary, cell-type totals, data dictionary, and machine-checkable integrity checks for the Xenium-to-Visium benchmark.",
        _format_metric(
            {
                "status": payload.get("status", ""),
                "artifacts": len(artifacts),
                "cell_types": len(cell_types),
                "failed_integrity_checks": len(failed_checks),
            },
            ["status", "artifacts", "cell_types", "failed_integrity_checks"],
        ),
    )


def _audit_completion_audit(records: list[EvidenceRecord], completion_audit_dir: Path) -> None:
    stage = "1. Xenium-to-Visium benchmark"
    json_path = completion_audit_dir / "goal_completion_audit.json"
    markdown_path = completion_audit_dir / "goal_completion_audit.md"
    payload = _read_json(json_path) if json_path.exists() else {}
    requirements = payload.get("requirements") if isinstance(payload.get("requirements"), list) else []
    missing = int(payload.get("num_missing", 1) or 0)
    status = _status_from_files([json_path, markdown_path])
    if status == "complete":
        status = "complete" if payload.get("overall_status") in {"complete", "complete_except_project_doi"} and missing == 0 and len(requirements) >= 10 else "partial"
    _add(
        records,
        stage,
        "requirement-by-requirement goal completion audit",
        status,
        json_path,
        "Machine-readable audit maps every Nature-level goal requirement to evidence files and failed checks, with the project DOI kept as an explicit external action.",
        _format_metric(
            {
                "status": payload.get("overall_status", ""),
                "requirements": len(requirements),
                "complete": payload.get("num_complete", ""),
                "partial": payload.get("num_partial", ""),
                "missing": payload.get("num_missing", ""),
            },
            ["status", "requirements", "complete", "partial", "missing"],
        ),
    )


def _audit_stage_2(records: list[EvidenceRecord], run_dir: Path, nature_dir: Path) -> None:
    stage = "2. Main model training"
    required_files = [
        run_dir / "checkpoint.pt",
        run_dir / "predicted_proportions.csv",
        run_dir / "reconstructed_expression.csv",
        run_dir / "gate_weights.csv",
        run_dir / "spot_uncertainty.csv",
        run_dir / "agent_attention.csv",
        run_dir / "training_history.csv",
        run_dir / "metrics.csv",
        nature_dir / "spot_uncertainty_map.png",
        nature_dir / "proportion_maps" / "proportion_map_manifest.json",
        nature_dir / "proportion_maps" / "predicted_top_celltypes_panel.png",
        nature_dir / "proportion_maps" / "predicted_tumor_immune_stromal_group_panel.png",
        nature_dir / "image_gate_map.png",
        nature_dir / "expression_gate_map.png",
        nature_dir / "reference_gate_map.png",
        nature_dir / "niche_map.png",
    ]
    metrics = _last_row(run_dir / "metrics.csv")
    _add(
        records,
        stage,
        "checkpoint, predictions, reliability maps, and training curve",
        _status_from_files(required_files),
        run_dir,
        "Main real-data run must provide model state, proportions, expression reconstruction, gates, uncertainty, attention, maps, and training history.",
        _format_metric(metrics, ["step", "jsd", "spotwise_cosine", "mean_celltype_pearson", "num_supervised_spots"]),
    )


def _audit_figure_assets(records: list[EvidenceRecord], figures_dir: Path) -> None:
    stage = "2. Main model training"
    manifest_path = figures_dir / "figure_manifest.json"
    required_files = [
        manifest_path,
        figures_dir / "figure_manifest.csv",
        figures_dir / "figure_manifest.md",
        figures_dir / "figure_1_workflow_schematic.png",
        figures_dir / "figure_2_spatial_cell_composition.png",
        figures_dir / "figure_3_baseline_performance.png",
        figures_dir / "figure_4_reliability_calibration.png",
        figures_dir / "figure_5_boundary_niche_pathology.png",
    ]
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    status = _status_from_files(required_files)
    if status == "complete":
        num_fail = int(manifest.get("num_fail", 1) or 0)
        num_figures = int(manifest.get("num_figures", 0) or 0)
        status = "complete" if num_fail == 0 and num_figures >= 5 else "partial"
    _add(
        records,
        stage,
        "manuscript Figure 1-5 assets",
        status,
        manifest_path,
        "Figure-level assets should exist for workflow, spatial composition, baselines, reliability, boundary/niche/pathology, and pass nonblank validation.",
        _format_metric(manifest, ["num_figures", "num_pass", "num_fail"]),
    )


def _audit_figure_legends(records: list[EvidenceRecord], legends_dir: Path) -> None:
    stage = "2. Main model training"
    json_path = legends_dir / "figure_legends.json"
    markdown_path = legends_dir / "figure_legends.md"
    payload = _read_json(json_path) if json_path.exists() else {}
    legends = payload.get("legends") if isinstance(payload.get("legends"), list) else []
    missing_evidence = payload.get("missing_evidence") if isinstance(payload.get("missing_evidence"), list) else []
    manifest_status = payload.get("figure_manifest_status") if isinstance(payload.get("figure_manifest_status"), dict) else {}
    status = _status_from_files([json_path, markdown_path])
    if status == "complete":
        status = "complete" if len(legends) >= 6 and not missing_evidence and int(manifest_status.get("num_fail", 1) or 0) == 0 else "partial"
    _add(
        records,
        stage,
        "manuscript figure legends and claim evidence",
        status,
        json_path,
        "Generated Figure 1-5 and Supplementary Figure S1 legends link each visual to key claims, values, and evidence files.",
        _format_metric(
            {
                "status": payload.get("status", ""),
                "legends": len(legends),
                "num_fail": manifest_status.get("num_fail", ""),
                "missing_evidence": len(missing_evidence),
            },
            ["status", "legends", "num_fail", "missing_evidence"],
        ),
    )


def _audit_manuscript_statements(records: list[EvidenceRecord], statements_dir: Path) -> None:
    stage = "1. Xenium-to-Visium benchmark"
    json_path = statements_dir / "manuscript_availability_statements.json"
    markdown_path = statements_dir / "manuscript_availability_statements.md"
    payload = _read_json(json_path) if json_path.exists() else {}
    statements = payload.get("statements") if isinstance(payload.get("statements"), list) else []
    release = payload.get("release") if isinstance(payload.get("release"), dict) else {}
    status = _status_from_files([json_path, markdown_path])
    if status == "complete":
        titles = {str(statement.get("title", "")) for statement in statements if isinstance(statement, dict)}
        required_titles = {"Data Availability", "Code Availability", "Reproducibility", "Computing Environment"}
        project_doi = str(release.get("doi", ""))
        status = "complete" if required_titles.issubset(titles) and (not project_doi or DOI_PATTERN.search(project_doi)) else "partial"
    _add(
        records,
        stage,
        "manuscript availability and reproducibility statements",
        status,
        json_path,
        "Generated Data availability, Code availability, Reproducibility, and Computing environment statements are evidence-linked and keep the project release DOI separate from upstream dataset DOIs.",
        _format_metric(release, ["status", "doi_status", "doi", "deposition_id"]),
    )


def _audit_manuscript_methods(records: list[EvidenceRecord], methods_dir: Path) -> None:
    stage = "1. Xenium-to-Visium benchmark"
    json_path = methods_dir / "manuscript_methods.json"
    markdown_path = methods_dir / "manuscript_methods.md"
    payload = _read_json(json_path) if json_path.exists() else {}
    sections = payload.get("sections") if isinstance(payload.get("sections"), list) else []
    missing_required = payload.get("missing_required_paths") if isinstance(payload.get("missing_required_paths"), list) else []
    required_titles = {
        "Xenium-to-Visium benchmark construction",
        "WaveST-Gate model",
        "Training objective and optimization",
        "Baseline comparison and fairness",
        "Ablation design",
        "Reliability, boundary, and niche analyses",
        "External generalization and robustness",
        "Statistical analysis, reproducibility, and compute",
    }
    status = _status_from_files([json_path, markdown_path])
    if status == "complete":
        titles = {str(section.get("title", "")) for section in sections if isinstance(section, dict)}
        status = "complete" if required_titles.issubset(titles) and not missing_required else "partial"
    _add(
        records,
        stage,
        "manuscript methods and statistical analysis draft",
        status,
        json_path,
        "Generated Methods/Supplementary Methods draft links benchmark construction, model specification, training losses, baseline fairness, ablations, reliability, boundary, niche, external generalization, robustness, statistics, compute, and reproducibility evidence.",
        _format_metric({"status": payload.get("status", ""), "sections": len(sections), "missing_required": len(missing_required)}, ["status", "sections", "missing_required"]),
    )


def _audit_reviewer_preflight(records: list[EvidenceRecord], preflight_dir: Path) -> None:
    stage = "1. Xenium-to-Visium benchmark"
    json_path = preflight_dir / "reviewer_preflight.json"
    markdown_path = preflight_dir / "reviewer_preflight.md"
    payload = _read_json(json_path) if json_path.exists() else {}
    claims = payload.get("headline_claims") if isinstance(payload.get("headline_claims"), list) else []
    concerns = payload.get("reviewer_concerns") if isinstance(payload.get("reviewer_concerns"), list) else []
    missing_evidence = payload.get("missing_evidence") if isinstance(payload.get("missing_evidence"), list) else []
    status = _status_from_files([json_path, markdown_path])
    if status == "complete":
        status = "complete" if len(claims) >= 8 and len(concerns) >= 8 and not missing_evidence else "partial"
    _add(
        records,
        stage,
        "reviewer preflight dossier",
        status,
        json_path,
        "Reviewer-facing dossier links headline claims, likely reviewer concerns, preemptive responses, stage status, and evidence paths.",
        _format_metric(
            {"status": payload.get("status", ""), "claims": len(claims), "concerns": len(concerns), "missing_evidence": len(missing_evidence)},
            ["status", "claims", "concerns", "missing_evidence"],
        ),
    )


def _audit_stage_3(records: list[EvidenceRecord], run_dir: Path, multisample_baseline_dir: Path) -> None:
    stage = "3. Strong baseline comparison"
    comparison_path = run_dir / "baseline_comparison" / "baseline_comparison.csv"
    env_path = run_dir / "baseline_environment_audit.json"
    frame = _read_csv(comparison_path)
    methods = frame.get("method", pd.Series(dtype=str)).astype(str).tolist()
    method_text = " | ".join(methods)
    required = ["cell2location", "RCTD", "CARD", "Tangram", "SpatialDWLS", "SpatialDWLS/Seurat", "BayesPrism", "SPOTlight"]
    missing = [name for name in required if name.lower() not in method_text.lower()]
    _add(
        records,
        stage,
        "formal baseline table with shared genes/reference/supervised spots",
        "complete" if not missing and comparison_path.exists() else "missing",
        comparison_path,
        "Completed formal baselines: " + (", ".join(methods) if methods else "none"),
        f"missing={','.join(missing)}; n_methods={len(methods)}",
    )
    columns = set(frame.columns)
    runtime_status = "complete" if {"runtime_seconds", "peak_cuda_memory_mb", "paired_permutation_p"}.issubset(columns) else "missing"
    _add(
        records,
        stage,
        "runtime, memory, and paired significance",
        runtime_status,
        comparison_path,
        "Comparison table should include runtime, memory, and paired per-spot significance columns.",
    )
    statistics_files = [
        run_dir / "baseline_comparison" / "baseline_split_bootstrap_metrics.csv",
        run_dir / "baseline_comparison" / "baseline_split_bootstrap_summary.csv",
        run_dir / "baseline_comparison" / "baseline_bootstrap_paired_improvement.csv",
        run_dir / "baseline_comparison" / "baseline_statistics_manifest.json",
    ]
    statistics_manifest = _read_json(run_dir / "baseline_comparison" / "baseline_statistics_manifest.json") if statistics_files[-1].exists() else {}
    _add(
        records,
        stage,
        "split/bootstrap mean and standard deviation",
        _status_from_files(statistics_files),
        run_dir / "baseline_comparison" / "baseline_split_bootstrap_summary.csv",
        "Split-wise and paired bootstrap mean/std statistics are required for the current matched benchmark.",
        _format_metric(statistics_manifest, ["n_methods", "n_supervised_spots", "n_bootstraps"]),
    )
    env = _read_json(env_path) if env_path.exists() else {}
    seurat_status = env.get("status", {}).get("SpatialDWLS/Seurat", "")
    giotto_files = [
        run_dir / "spatialdwls_giotto_baseline" / "spatialdwls_giotto_metrics.csv",
        run_dir / "spatialdwls_giotto_baseline" / "spatialdwls_giotto_proportions.csv",
        run_dir / "spatialdwls_giotto_baseline" / "spatialdwls_giotto_manifest.json",
        run_dir / "spatialdwls_giotto_baseline" / "spatialdwls_giotto_python_manifest.json",
    ]
    giotto_metrics = _last_row(giotto_files[0])
    giotto_status = "complete" if seurat_status == "ready" and _status_from_files(giotto_files) == "complete" else "optional_pending"
    _add(
        records,
        stage,
        "SpatialDWLS/Seurat package-stack rerun",
        giotto_status,
        giotto_files[0] if giotto_files[0].exists() else env_path,
        (
            "Full Giotto runDWLSDeconv package-stack rerun produced predictions, metrics, and manifests."
            if giotto_status == "complete"
            else "Standalone SpatialDWLS is present; full Giotto/Seurat package-stack rerun still needs complete output files."
        ),
        f"SpatialDWLS/Seurat={seurat_status}; " + _format_metric(giotto_metrics, ["jsd", "spotwise_cosine", "runtime_seconds"]),
    )
    multisample_files = [
        multisample_baseline_dir / "matched_multisample_baseline_dataset_metrics.csv",
        multisample_baseline_dir / "matched_multisample_baseline_summary.csv",
        multisample_baseline_dir / "matched_multisample_baseline_paired_improvement.csv",
        multisample_baseline_dir / "matched_multisample_baseline_manifest.json",
    ]
    multisample_manifest = _read_json(multisample_baseline_dir / "matched_multisample_baseline_manifest.json") if multisample_files[-1].exists() else {}
    multisample_summary = _read_csv(multisample_baseline_dir / "matched_multisample_baseline_summary.csv")
    complete_methods = multisample_manifest.get("methods_in_all_required_datasets", [])
    has_two_datasets = int(multisample_manifest.get("num_datasets", 0) or 0) >= 2
    has_formal_methods = any("rctd" in str(method).lower() for method in complete_methods) and any(
        "tangram" in str(method).lower() for method in complete_methods
    )
    multisample_status = (
        "complete"
        if _status_from_files(multisample_files) == "complete" and has_two_datasets and has_formal_methods
        else "partial"
    )
    top_metric = ""
    if not multisample_summary.empty:
        top_metric = _format_metric(multisample_summary.iloc[0].to_dict(), ["jsd_mean", "jsd_std", "num_datasets"])
    _add(
        records,
        stage,
        "independent matched-GT multi-sample mean and standard deviation",
        multisample_status,
        multisample_baseline_dir / "matched_multisample_baseline_summary.csv",
        "Independent matched-GT datasets should be aggregated for formal mean/std baseline reporting.",
        "; ".join(
            filter(
                None,
                [
                    _format_metric(multisample_manifest, ["num_datasets", "num_methods"]),
                    f"complete_methods={len(complete_methods)}",
                    top_metric,
                ],
            )
        ),
    )


def _audit_stage_4(records: list[EvidenceRecord], run_dir: Path) -> None:
    stage = "4. Ablation study"
    summary_path = run_dir / "ablations250" / "ablation_summary.csv"
    frame = _read_csv(summary_path)
    ablations = set(frame.get("ablation", pd.Series(dtype=str)).astype(str).tolist())
    required = {
        "full",
        "no_wavelet_cnn_replacement",
        "no_image_branch",
        "no_celltype_agents",
        "no_gate_mean_fusion",
        "raw_gate_without_uncertainty",
        "no_boundary_loss",
        "normal_smoothness_only",
        "no_local_refinement",
        "expression_only",
        "image_only",
        "reference_only",
    }
    missing = sorted(required - ablations)
    _add(
        records,
        stage,
        "module and modality ablations",
        "complete" if summary_path.exists() and not missing else "missing",
        summary_path,
        "Required ablations are present." if not missing else "Missing ablations: " + ", ".join(missing),
        f"n_ablations={len(ablations)}",
    )


def _audit_stage_5(records: list[EvidenceRecord], nature_dir: Path) -> None:
    stage = "5. Reliability and calibration"
    required_files = [
        nature_dir / "reliability_summary.json",
        nature_dir / "reliability_spot_errors.csv",
        nature_dir / "risk_coverage_curve.csv",
        nature_dir / "risk_coverage_curve.png",
        nature_dir / "uncertainty_calibration_bins.csv",
        nature_dir / "uncertainty_calibration.png",
        nature_dir / "failure_case_candidates.csv",
        nature_dir / "image_gate_map.png",
        nature_dir / "expression_gate_map.png",
        nature_dir / "reference_gate_map.png",
    ]
    summary = _read_json(nature_dir / "reliability_summary.json") if (nature_dir / "reliability_summary.json").exists() else {}
    _add(
        records,
        stage,
        "uncertainty calibration and modality reliability evidence",
        _status_from_files(required_files),
        nature_dir,
        "Uncertainty-error correlation, risk coverage, calibration, gate maps, and failure cases are required.",
        _format_metric(summary, ["uncertainty_error_pearson", "uncertainty_error_spearman", "risk_gap", "calibration_bin_pearson"]),
    )


def _audit_stage_6(records: list[EvidenceRecord], nature_dir: Path, external_pathology_dir: Path) -> None:
    stage = "6. Boundary preservation"
    required_files = [
        nature_dir / "boundary_summary.json",
        nature_dir / "boundary_edge_jumps.csv",
        nature_dir / "boundary_type_summary.csv",
        nature_dir / "boundary_marker_validation.csv",
        nature_dir / "boundary_sharpness_map.png",
        nature_dir / "boundary_he_overlay.png",
        nature_dir / "boundary_he_pathology_proxy.csv",
    ]
    summary = _read_json(nature_dir / "boundary_summary.json") if (nature_dir / "boundary_summary.json").exists() else {}
    _add(
        records,
        stage,
        "typed boundary sharpness and marker validation",
        _status_from_files(required_files),
        nature_dir,
        "Tumor-stroma, ductal, immune-edge, marker validation, H&E overlay, and no-boundary comparison artifacts are required.",
        _format_metric(summary, ["boundary_to_interior_jump_ratio", "mean_boundary_jump", "comparison_mean_boundary_jump"]),
    )
    pathology_files = [
        external_pathology_dir / "external_pathology_validation_manifest.json",
        external_pathology_dir / "pathology_class_summary.csv",
        external_pathology_dir / "pathology_patient_summary.csv",
        external_pathology_dir / "pathology_group_composition.png",
    ]
    manifest = _read_json(external_pathology_dir / "external_pathology_validation_manifest.json") if pathology_files[0].exists() else {}
    _add(
        records,
        stage,
        "independent pathology metadata validation",
        _status_from_files(pathology_files),
        external_pathology_dir / "pathology_class_summary.csv",
        "External Wu/Swarbrick pathology classifications are compared with no-retuning predicted tumor/immune/stromal composition.",
        _format_metric(manifest, ["num_datasets", "num_spots", "num_pathology_classes", "overall_agreement_rate"]),
    )


def _audit_stage_7(records: list[EvidenceRecord], nature_dir: Path, external_pathology_dir: Path) -> None:
    stage = "7. Biological niche interpretation"
    required_files = [
        nature_dir / "niche_assignments.csv",
        nature_dir / "niche_composition.csv",
        nature_dir / "niche_marker_enrichment.csv",
        nature_dir / "niche_biological_summary.csv",
        nature_dir / "gate_reliability_by_niche.csv",
        nature_dir / "agent_attention_by_niche.csv",
        nature_dir / "niche_map.png",
    ]
    summary = _read_json(nature_dir / "niche_summary.json") if (nature_dir / "niche_summary.json").exists() else {}
    _add(
        records,
        stage,
        "tumor-immune-stromal niche outputs",
        _status_from_files(required_files),
        nature_dir,
        "Niche composition, marker enrichment, gate-by-niche, agent attention-by-niche, and maps are required.",
        _format_metric(summary, ["num_niches"]),
    )
    neighborhood_files = [
        nature_dir / "niche_xenium_neighborhood_validation.csv",
        nature_dir / "niche_xenium_neighborhood_summary.csv",
    ]
    _add(
        records,
        stage,
        "Xenium neighborhood validation",
        _status_from_files(neighborhood_files),
        nature_dir / "niche_xenium_neighborhood_summary.csv",
        "Predicted niches should be checked against local Xenium tumor/immune/stromal neighborhood composition.",
    )
    pathology_niche_files = [
        external_pathology_dir / "external_pathology_validation_manifest.json",
        external_pathology_dir / "pathology_niche_summary.csv",
        external_pathology_dir / "pathology_niche_by_class.csv",
        external_pathology_dir / "pathology_niche_by_class_heatmap.png",
    ]
    manifest = _read_json(external_pathology_dir / "external_pathology_validation_manifest.json") if pathology_niche_files[0].exists() else {}
    _add(
        records,
        stage,
        "independent pathology correspondence",
        _status_from_files(pathology_niche_files),
        external_pathology_dir / "pathology_niche_summary.csv",
        "External predicted niches are cross-tabulated against public pathology classification labels.",
        _format_metric(manifest, ["num_datasets", "num_spots", "overall_agreement_rate"]),
    )


def _audit_stage_8(records: list[EvidenceRecord], external_dir: Path, external_matched_gt_dir: Path) -> None:
    stage = "8. External generalization"
    summary_path = external_dir / "external_no_retuning_summary.csv"
    frame = _read_csv(summary_path)
    datasets = frame.get("dataset", pd.Series(dtype=str)).astype(str).tolist()
    required_per_dataset = [
        "predicted_proportions.csv",
        "gate_weights.csv",
        "spot_uncertainty.csv",
        "agent_attention.csv",
        "aligned_prediction_metrics.csv",
        "aligned_prediction_manifest.json",
    ]
    missing_outputs = []
    for dataset in datasets:
        for name in required_per_dataset:
            if not (external_dir / dataset / name).exists():
                missing_outputs.append(f"{dataset}/{name}")
    external_10x = sum(not dataset.startswith("wu_swarbrick") for dataset in datasets)
    wu = sum(dataset.startswith("wu_swarbrick") for dataset in datasets)
    status = "complete" if summary_path.exists() and len(datasets) >= 10 and not missing_outputs else "missing"
    _add(
        records,
        stage,
        "no-retuning external prediction panel",
        status,
        summary_path,
        "External no-retuning predictions must include per-dataset proportions, gates, uncertainty, attention, metrics, and manifests.",
        f"n_datasets={len(datasets)}; external_10x={external_10x}; wu_swarbrick={wu}; missing_outputs={len(missing_outputs)}",
    )
    matched_files = [
        external_matched_gt_dir / "external_matched_gt_summary.csv",
        external_matched_gt_dir / "external_matched_gt_manifest.json",
        external_matched_gt_dir / "xenium_rep1_pseudospots_radius55_common297" / "matched_gt_metrics.csv",
        external_matched_gt_dir / "xenium_rep1_pseudospots_radius55_common297" / "comparison" / "baseline_comparison.csv",
    ]
    matched_manifest = _read_json(external_matched_gt_dir / "external_matched_gt_manifest.json") if matched_files[1].exists() else {}
    matched_summary = _read_csv(external_matched_gt_dir / "external_matched_gt_summary.csv")
    metric = _format_metric(matched_manifest, ["num_datasets"])
    if not matched_summary.empty:
        first = matched_summary.iloc[0].to_dict()
        metric = "; ".join(filter(None, [metric, _format_metric(first, ["num_spots", "model_jsd", "model_spotwise_cosine", "wavestgate_rank_by_jsd"])]))
    _add(
        records,
        stage,
        "external matched-GT performance metrics",
        _status_from_files(matched_files),
        external_matched_gt_dir / "external_matched_gt_summary.csv",
        "Rep1 Xenium pseudo-spots provide external no-retuning matched cell-type ground truth metrics and simple-baseline comparison.",
        metric,
    )
    minimal_dir = external_matched_gt_dir / "xenium_rep1_pseudospots_radius55_common297" / "wavestgate_minimal_retune"
    minimal_files = [
        minimal_dir / "config.yaml",
        minimal_dir / "checkpoint.pt",
        minimal_dir / "test_metrics.csv",
        minimal_dir / "predicted_proportions.csv",
        minimal_dir / "gate_weights.csv",
        minimal_dir / "spot_uncertainty.csv",
        minimal_dir / "agent_attention.csv",
        minimal_dir / "test_formal_comparison" / "baseline_comparison.csv",
        Path("results/nature_matched_multisample_baselines_minimal_retune") / "matched_multisample_baseline_summary.csv",
        Path("results/nature_matched_multisample_baselines_minimal_retune") / "matched_multisample_baseline_manifest.json",
    ]
    minimal_metrics = _last_row(minimal_dir / "test_metrics.csv") if (minimal_dir / "test_metrics.csv").exists() else {}
    minimal_comparison = _read_csv(minimal_dir / "test_formal_comparison" / "baseline_comparison.csv")
    minimal_metric = _format_metric(
        minimal_metrics,
        ["jsd", "spotwise_cosine", "mean_celltype_pearson", "num_train_spots", "num_eval_spots"],
    )
    if not minimal_comparison.empty:
        top = minimal_comparison.iloc[0].to_dict()
        minimal_metric = "; ".join(filter(None, [minimal_metric, f"top_method={top.get('method')}", _format_metric(top, ["jsd"])]))
    _add(
        records,
        stage,
        "Rep1 held-out minimal-retuning matched-GT adaptation",
        _status_from_files(minimal_files),
        minimal_dir / "test_formal_comparison" / "baseline_comparison.csv",
        "Rep2 checkpoint is minimally retuned on Rep1 train/val pseudo-spots and evaluated on held-out Rep1 test spots against formal baselines.",
        minimal_metric,
    )
    budget_dir = external_matched_gt_dir / "xenium_rep1_pseudospots_radius55_common297" / "rep1_retune_budget_curve"
    budget_files = [
        budget_dir / "rep1_minimal_retune_budget_curve.csv",
        budget_dir / "rep1_minimal_retune_budget_curve_manifest.json",
        budget_dir / "rep1_minimal_retune_budget_curve.md",
    ]
    budget_curve = _read_csv(budget_files[0])
    first_beating = {}
    no_retune = {}
    if not budget_curve.empty:
        no_retune_rows = budget_curve.loc[pd.to_numeric(budget_curve.get("budget_steps", pd.Series(dtype=float)), errors="coerce").eq(0)]
        if not no_retune_rows.empty:
            no_retune = no_retune_rows.iloc[0].to_dict()
        beating = budget_curve.loc[budget_curve.get("beats_best_baseline", pd.Series(dtype=str)).astype(str).str.lower().isin({"true", "1"})].copy()
        if not beating.empty:
            beating["_budget_numeric"] = pd.to_numeric(beating["budget_steps"], errors="coerce")
            first_beating = beating.sort_values("_budget_numeric").iloc[0].to_dict()
    _add(
        records,
        stage,
        "Rep1 no-retuning and minimal-retuning budget curve",
        _status_from_files(budget_files) if first_beating else "partial",
        budget_files[0],
        "Rep1 direct transfer is reported as a domain-shift case, and a held-out minimal-retuning budget curve quantifies the adaptation needed to beat the best baseline.",
        "; ".join(
            filter(
                None,
                [
                    _format_metric(no_retune, ["jsd", "best_baseline_jsd", "beats_best_baseline"]),
                    _format_metric(first_beating, ["budget_steps", "jsd", "jsd_margin_vs_best_baseline"]),
                ],
            )
        ),
    )


def _audit_stage_9(records: list[EvidenceRecord], run_dir: Path) -> None:
    stage = "9. Robustness"
    robustness_path = run_dir / "robustness" / "robustness_summary.csv"
    patch_path = run_dir / "patch_size_robustness" / "patch_size_summary.csv"
    split_sensitivity_files = [
        run_dir / "split_sensitivity" / "original_split_gt_summary.csv",
        run_dir / "split_sensitivity" / "split_sensitivity_aggregate.csv",
        run_dir / "split_sensitivity" / "split_sensitivity_manifest.json",
        run_dir / "split_sensitivity" / "split_sensitivity.md",
    ]
    benchmark_sensitivity_files = [
        run_dir / "benchmark_sensitivity" / "radius_cell_count_sensitivity.csv",
        run_dir / "benchmark_sensitivity" / "radius_coverage_summary.csv",
        run_dir / "benchmark_sensitivity" / "radius_cell_count_sensitivity_manifest.json",
        run_dir / "benchmark_sensitivity" / "radius_cell_count_sensitivity.md",
    ]
    robust = _read_csv(robustness_path)
    patch = _read_csv(patch_path)
    scenario_pairs = set(zip(robust.get("scenario", pd.Series(dtype=str)).astype(str), robust.get("level", pd.Series(dtype=str)).astype(str)))
    scenarios = set(robust.get("scenario", pd.Series(dtype=str)).astype(str).tolist())
    patch_sizes = set(patch.get("patch_size", pd.Series(dtype=int)).astype(str).tolist()) if not patch.empty else set()
    needed_gene_dropout = {("gene_dropout", "0.1"), ("gene_dropout", "0.3"), ("gene_dropout", "0.5")}
    needed_gene_panels = {("gene_panel", "top200_variance"), ("gene_panel", "top100_variance"), ("gene_panel", "marker_only")}
    missing_dropout = sorted(needed_gene_dropout - scenario_pairs)
    missing_panels = sorted(needed_gene_panels - scenario_pairs)
    missing_patch = sorted({"32", "64", "128", "256"} - patch_sizes)
    reference_rows = robust[robust.get("scenario", pd.Series(dtype=str)).astype(str) == "reference_missing_celltype"] if not robust.empty else pd.DataFrame()
    prototype_rows = robust[robust.get("scenario", pd.Series(dtype=str)).astype(str) == "prototype_perturbation"] if not robust.empty else pd.DataFrame()
    split_rows = robust[robust.get("scenario", pd.Series(dtype=str)).astype(str) == "split"] if not robust.empty else pd.DataFrame()
    subgroup_levels = set(
        robust.loc[robust.get("scenario", pd.Series(dtype=str)).astype(str) == "subgroup", "level"].astype(str).tolist()
    ) if not robust.empty else set()
    core_complete = (
        robustness_path.exists()
        and patch_path.exists()
        and not missing_dropout
        and not missing_panels
        and not missing_patch
        and len(reference_rows) >= 10
        and len(prototype_rows) >= 4
        and len(split_rows) >= 3
        and "he_perturbation" in scenarios
        and "subgroup" in scenarios
        and {"low_expression_spots", "low_cell_count_spots"}.issubset(subgroup_levels)
    )
    _add(
        records,
        stage,
        "core stress tests",
        "complete" if core_complete else "missing",
        robustness_path,
        "Core robustness should include gene dropout, gene panels, cell-type removal, prototype perturbation, H&E perturbations, low-expression/low-cell-count subgroups, different splits, and patch sizes.",
        f"n_rows={len(robust)}; removed_celltypes={len(reference_rows)}; prototype_rows={len(prototype_rows)}; split_rows={len(split_rows)}; missing_panels={missing_panels}; patch_sizes={','.join(sorted(patch_sizes))}",
    )
    _add(
        records,
        stage,
        "GT-stratified split sensitivity and benchmark-radius confidence controls",
        "complete" if _status_from_files(split_sensitivity_files) == "complete" and _status_from_files(benchmark_sensitivity_files) == "complete" else "partial",
        run_dir / "split_sensitivity" / "split_sensitivity_aggregate.csv",
        "Supplementary GT-stratified splits address the zero-GT validation split, while radius/cell-count sensitivity checks Xenium-to-Visium benchmark-label confidence.",
        f"split_files={_status_from_files(split_sensitivity_files)}; benchmark_files={_status_from_files(benchmark_sensitivity_files)}",
    )


def _write_evidence_manifest(records: list[EvidenceRecord], output_dir: Path) -> Path:
    path = output_dir / "evidence_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return path


def _write_checksums(records: list[EvidenceRecord], output_dir: Path, max_bytes: int = 256 * 1024 * 1024) -> Path:
    paths: list[Path] = []
    for record in records:
        if not record.evidence_path:
            continue
        path = Path(record.evidence_path)
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(child for child in path.glob("*") if child.is_file()))
    unique_paths = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)

    checksum_path = output_dir / "release_file_checksums.csv"
    with checksum_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256", "status"])
        writer.writeheader()
        for path in unique_paths:
            if not path.exists():
                continue
            digest, status = _sha256(path, max_bytes=max_bytes)
            writer.writerow({"path": str(path), "bytes": path.stat().st_size, "sha256": digest, "status": status})
    return checksum_path


def _write_release_manifest(records: list[EvidenceRecord], output_dir: Path, checksum_path: Path, release_dir: Path) -> Path:
    path = output_dir / "zenodo_release_manifest.json"
    complete_records = [record for record in records if record.status == "complete"]
    deposition_summary = _release_deposition_summary(release_dir, output_dir)
    release_doi = str(deposition_summary.get("doi", ""))
    release_status = str(deposition_summary.get("release_status") or "prepared_not_deposited")
    release_published = release_status == "zenodo_published"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_status": release_status,
        "doi": release_doi,
        "doi_required_before_submission": not bool(release_doi and release_published),
        "zenodo_deposition_id": deposition_summary.get("zenodo_deposition_id", ""),
        "zenodo_record_url": deposition_summary.get("zenodo_record_url", ""),
        "zenodo_deposition_result": deposition_summary.get("zenodo_deposition_result", ""),
        "notes": (
            "A WaveST-Gate release DOI is recorded in explicit published Zenodo deposition fields."
            if release_doi and release_published
            else (
                "A Zenodo draft/reserved DOI is recorded but is not public; publish Zenodo or provide a public GitHub repository before submission."
                if release_doi
                else "This manifest prepares evidence paths for deposition. It is not a Zenodo deposit and does not replace a DOI."
            )
        ),
        "num_evidence_records": len(records),
        "num_complete_records": len(complete_records),
        "evidence_manifest": str(output_dir / "evidence_manifest.csv"),
        "checksums": str(checksum_path),
        "records": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_json_report(records: list[EvidenceRecord], output_dir: Path, checksum_path: Path, release_path: Path) -> Path:
    stages = sorted({record.stage for record in records})
    stage_summary = {
        stage: {
            "status": _stage_status(records, stage),
            "num_complete": sum(record.stage == stage and record.status == "complete" for record in records),
            "num_partial": sum(record.stage == stage and record.status == "partial" for record in records),
            "num_missing": sum(record.stage == stage and record.status == "missing" for record in records),
            "num_optional_pending": sum(record.stage == stage and record.status == "optional_pending" for record in records),
        }
        for stage in stages
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "complete" if all(value["status"] == "complete" for value in stage_summary.values()) else "partial",
        "stage_summary": stage_summary,
        "evidence_manifest": str(output_dir / "evidence_manifest.csv"),
        "release_file_checksums": str(checksum_path),
        "zenodo_release_manifest": str(release_path),
        "records": [asdict(record) for record in records],
    }
    path = output_dir / "readiness_report.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_markdown_report(records: list[EvidenceRecord], output_dir: Path, json_path: Path) -> Path:
    stages = sorted({record.stage for record in records})
    lines = [
        "# WaveST-Gate Submission Readiness",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Machine-readable report: `{json_path}`",
        "",
        "## Stage Summary",
        "",
        "| Stage | Status | Complete | Partial | Optional pending | Missing |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for stage in stages:
        stage_records = [record for record in records if record.stage == stage]
        lines.append(
            "| {stage} | {status} | {complete} | {partial} | {optional_pending} | {missing} |".format(
                stage=stage,
                status=_stage_status(records, stage),
                complete=sum(record.status == "complete" for record in stage_records),
                partial=sum(record.status == "partial" for record in stage_records),
                optional_pending=sum(record.status == "optional_pending" for record in stage_records),
                missing=sum(record.status == "missing" for record in stage_records),
            )
        )
    lines.extend(["", "## Evidence Items", ""])
    for stage in stages:
        lines.extend([f"### {stage}", ""])
        for record in [record for record in records if record.stage == stage]:
            metric = f" Metric: {record.metric}" if record.metric else ""
            path_text = f" Evidence: `{record.evidence_path}`." if record.evidence_path else ""
            lines.append(f"- **{record.status}** - {record.requirement}.{path_text} {record.detail}{metric}")
        lines.append("")
    path = output_dir / "readiness_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_submission_readiness(
    benchmark_dir: str | Path = "data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55",
    run_dir: str | Path = "results/nature_main/cytassist_rep2_radius55",
    external_dir: str | Path = "results/nature_external_no_retuning",
    output_dir: str | Path = "results/nature_submission_readiness",
    docs_dir: str | Path = "docs",
    release_dir: str | Path = "results/nature_release",
    external_pathology_dir: str | Path = "results/nature_external_pathology_validation",
    external_matched_gt_dir: str | Path = "results/nature_external_matched_gt",
    multisample_baseline_dir: str | Path = "results/nature_matched_multisample_baselines",
    nature_analysis_dir: str | Path | None = None,
    manuscript_figures_dir: str | Path = "results/nature_manuscript_figures",
    manuscript_statements_dir: str | Path = "results/nature_manuscript_statements",
    manuscript_methods_dir: str | Path = "results/nature_manuscript_methods",
    manuscript_figure_legends_dir: str | Path = "results/nature_manuscript_figure_legends",
    reviewer_preflight_dir: str | Path = "results/nature_reviewer_preflight",
    benchmark_datasheet_dir: str | Path = "results/nature_benchmark_datasheet",
    completion_audit_dir: str | Path = "results/nature_completion_audit",
) -> dict[str, str]:
    """Generate a submission-readiness audit and evidence manifest."""

    benchmark_dir = Path(benchmark_dir)
    run_dir = Path(run_dir)
    external_dir = Path(external_dir)
    output_dir = Path(output_dir)
    docs_dir = Path(docs_dir)
    release_dir = Path(release_dir)
    external_pathology_dir = Path(external_pathology_dir)
    external_matched_gt_dir = Path(external_matched_gt_dir)
    multisample_baseline_dir = Path(multisample_baseline_dir)
    manuscript_figures_dir = Path(manuscript_figures_dir)
    manuscript_statements_dir = Path(manuscript_statements_dir)
    manuscript_methods_dir = Path(manuscript_methods_dir)
    manuscript_figure_legends_dir = Path(manuscript_figure_legends_dir)
    reviewer_preflight_dir = Path(reviewer_preflight_dir)
    benchmark_datasheet_dir = Path(benchmark_datasheet_dir)
    completion_audit_dir = Path(completion_audit_dir)
    nature_dir = Path(nature_analysis_dir) if nature_analysis_dir is not None else run_dir / "nature_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[EvidenceRecord] = []
    _audit_stage_1(records, benchmark_dir, docs_dir, output_dir, release_dir)
    _audit_benchmark_datasheet(records, benchmark_datasheet_dir)
    _audit_completion_audit(records, completion_audit_dir)
    _audit_manuscript_statements(records, manuscript_statements_dir)
    _audit_manuscript_methods(records, manuscript_methods_dir)
    _audit_reviewer_preflight(records, reviewer_preflight_dir)
    _audit_stage_2(records, run_dir, nature_dir)
    _audit_figure_assets(records, manuscript_figures_dir)
    _audit_figure_legends(records, manuscript_figure_legends_dir)
    _audit_stage_3(records, run_dir, multisample_baseline_dir)
    _audit_stage_4(records, run_dir)
    _audit_stage_5(records, nature_dir)
    _audit_stage_6(records, nature_dir, external_pathology_dir)
    _audit_stage_7(records, nature_dir, external_pathology_dir)
    _audit_stage_8(records, external_dir, external_matched_gt_dir)
    _audit_stage_9(records, run_dir)

    evidence_path = _write_evidence_manifest(records, output_dir)
    checksum_path = _write_checksums(records, output_dir)
    release_path = _write_release_manifest(records, output_dir, checksum_path, release_dir)
    json_path = _write_json_report(records, output_dir, checksum_path, release_path)
    markdown_path = _write_markdown_report(records, output_dir, json_path)
    return {
        "evidence_manifest": str(evidence_path),
        "release_file_checksums": str(checksum_path),
        "zenodo_release_manifest": str(release_path),
        "readiness_report_json": str(json_path),
        "readiness_report_markdown": str(markdown_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate WaveST-Gate submission readiness reports.")
    parser.add_argument("--benchmark-dir", default="data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55")
    parser.add_argument("--run-dir", default="results/nature_main/cytassist_rep2_radius55")
    parser.add_argument("--external-dir", default="results/nature_external_no_retuning")
    parser.add_argument("--output-dir", default="results/nature_submission_readiness")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--release-dir", default="results/nature_release")
    parser.add_argument("--external-pathology-dir", default="results/nature_external_pathology_validation")
    parser.add_argument("--external-matched-gt-dir", default="results/nature_external_matched_gt")
    parser.add_argument("--multisample-baseline-dir", default="results/nature_matched_multisample_baselines")
    parser.add_argument("--nature-analysis-dir", default=None)
    parser.add_argument("--manuscript-figures-dir", default="results/nature_manuscript_figures")
    parser.add_argument("--manuscript-statements-dir", default="results/nature_manuscript_statements")
    parser.add_argument("--manuscript-methods-dir", default="results/nature_manuscript_methods")
    parser.add_argument("--manuscript-figure-legends-dir", default="results/nature_manuscript_figure_legends")
    parser.add_argument("--reviewer-preflight-dir", default="results/nature_reviewer_preflight")
    parser.add_argument("--benchmark-datasheet-dir", default="results/nature_benchmark_datasheet")
    parser.add_argument("--completion-audit-dir", default="results/nature_completion_audit")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    outputs = generate_submission_readiness(
        benchmark_dir=args.benchmark_dir,
        run_dir=args.run_dir,
        external_dir=args.external_dir,
        output_dir=args.output_dir,
        docs_dir=args.docs_dir,
        release_dir=args.release_dir,
        external_pathology_dir=args.external_pathology_dir,
        external_matched_gt_dir=args.external_matched_gt_dir,
        multisample_baseline_dir=args.multisample_baseline_dir,
        nature_analysis_dir=args.nature_analysis_dir,
        manuscript_figures_dir=args.manuscript_figures_dir,
        manuscript_statements_dir=args.manuscript_statements_dir,
        manuscript_methods_dir=args.manuscript_methods_dir,
        manuscript_figure_legends_dir=args.manuscript_figure_legends_dir,
        reviewer_preflight_dir=args.reviewer_preflight_dir,
        benchmark_datasheet_dir=args.benchmark_datasheet_dir,
        completion_audit_dir=args.completion_audit_dir,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
