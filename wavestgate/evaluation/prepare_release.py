"""Prepare Zenodo-ready release metadata and evidence bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_INCLUDE_PATHS = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "codemeta.json",
    "pyproject.toml",
    "docs",
    "data_manifest",
    "experiments/nature_main",
    "wavestgate",
    "tests",
    "data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55",
    "data/processed/xenium_rep1_pseudospots_radius55_common297",
    "results/nature_main/cytassist_rep2_radius55/checkpoint.pt",
    "results/nature_main/cytassist_rep2_radius55/metrics.csv",
    "results/nature_main/cytassist_rep2_radius55/training_history.csv",
    "results/nature_main/cytassist_rep2_radius55/predicted_proportions.csv",
    "results/nature_main/cytassist_rep2_radius55/reconstructed_expression.csv",
    "results/nature_main/cytassist_rep2_radius55/gate_weights.csv",
    "results/nature_main/cytassist_rep2_radius55/raw_gate_weights.csv",
    "results/nature_main/cytassist_rep2_radius55/modality_reliability.csv",
    "results/nature_main/cytassist_rep2_radius55/spot_uncertainty.csv",
    "results/nature_main/cytassist_rep2_radius55/agent_attention.csv",
    "results/nature_main/cytassist_rep2_radius55/baseline_environment_audit.json",
    "results/nature_main/cytassist_rep2_radius55/baseline_comparison",
    "results/nature_main/cytassist_rep2_radius55/simple_baselines",
    "results/nature_main/cytassist_rep2_radius55/rctd_baseline_multi",
    "results/nature_main/cytassist_rep2_radius55/card_baseline",
    "results/nature_main/cytassist_rep2_radius55/cell2location_baseline",
    "results/nature_main/cytassist_rep2_radius55/spotlight_baseline",
    "results/nature_main/cytassist_rep2_radius55/bayesprism_baseline",
    "results/nature_main/cytassist_rep2_radius55/spatialdwls_baseline",
    "results/nature_main/cytassist_rep2_radius55/spatialdwls_giotto_baseline",
    "results/nature_main/cytassist_rep2_radius55/tangram_baseline",
    "results/nature_main/cytassist_rep2_radius55/tangram_baseline_uniform_prior",
    "results/nature_main/cytassist_rep2_radius55/ablations250/ablation_summary.csv",
    "results/nature_main/cytassist_rep2_radius55/nature_analysis",
    "results/nature_main/cytassist_rep2_radius55/robustness/robustness_summary.csv",
    "results/nature_main/cytassist_rep2_radius55/robustness/robustness_manifest.json",
    "results/nature_main/cytassist_rep2_radius55/patch_size_robustness/patch_size_summary.csv",
    "results/nature_main/cytassist_rep2_radius55/split_sensitivity",
    "results/nature_main/cytassist_rep2_radius55/benchmark_sensitivity",
    "results/nature_main/cytassist_rep2_radius55_split_guarded/metrics.csv",
    "results/nature_main/cytassist_rep2_radius55_split_guarded/training_history.csv",
    "results/nature_main/cytassist_rep2_radius55_split_guarded/predicted_proportions.csv",
    "results/nature_main/cytassist_rep2_radius55_split_guarded/gate_weights.csv",
    "results/nature_main/cytassist_rep2_radius55_split_guarded/raw_gate_weights.csv",
    "results/nature_main/cytassist_rep2_radius55_split_guarded/modality_reliability.csv",
    "results/nature_main/cytassist_rep2_radius55_split_guarded/spot_uncertainty.csv",
    "results/nature_main/cytassist_rep2_radius55_split_guarded/agent_attention.csv",
    "results/nature_main/cytassist_rep2_radius55_imagegate/checkpoint.pt",
    "results/nature_main/cytassist_rep2_radius55_imagegate/metrics.csv",
    "results/nature_main/cytassist_rep2_radius55_imagegate/training_history.csv",
    "results/nature_main/cytassist_rep2_radius55_imagegate/predicted_proportions.csv",
    "results/nature_main/cytassist_rep2_radius55_imagegate/reconstructed_expression.csv",
    "results/nature_main/cytassist_rep2_radius55_imagegate/gate_weights.csv",
    "results/nature_main/cytassist_rep2_radius55_imagegate/raw_gate_weights.csv",
    "results/nature_main/cytassist_rep2_radius55_imagegate/modality_reliability.csv",
    "results/nature_main/cytassist_rep2_radius55_imagegate/spot_uncertainty.csv",
    "results/nature_main/cytassist_rep2_radius55_imagegate/agent_attention.csv",
    "results/nature_main/cytassist_rep2_radius55_imagegate/nature_analysis",
    "results/nature_main/cytassist_rep2_radius55_imagegate/image_contribution",
    "results/nature_main/cytassist_rep2_radius55_imagegate_noimage",
    "results/nature_external_no_retuning/external_no_retuning_summary.csv",
    "results/nature_external_matched_gt",
    "results/nature_external_pathology_validation",
    "results/nature_manuscript_tables",
    "results/nature_manuscript_figures",
    "results/nature_manuscript_statements",
    "results/nature_manuscript_methods",
    "results/nature_manuscript_figure_legends",
    "results/nature_benchmark_datasheet",
    "results/nature_completion_audit",
    "results/nature_final_doi_gate",
    "results/nature_reviewer_preflight",
    "results/nature_leakage_fairness_audit",
    "results/nature_release/environment_report.json",
    "results/nature_release/environment_report.md",
    "results/nature_submission_readiness",
]

CRITICAL_ARTIFACT_PATHS = [
    "LICENSE",
    "CITATION.cff",
    "codemeta.json",
    "results/nature_release/environment_report.json",
    "results/nature_release/environment_report.md",
    "results/nature_main/cytassist_rep2_radius55/checkpoint.pt",
    "results/nature_main/cytassist_rep2_radius55/metrics.csv",
    "results/nature_main/cytassist_rep2_radius55/training_history.csv",
    "results/nature_main/cytassist_rep2_radius55/predicted_proportions.csv",
    "results/nature_main/cytassist_rep2_radius55/reconstructed_expression.csv",
    "results/nature_main/cytassist_rep2_radius55/gate_weights.csv",
    "results/nature_main/cytassist_rep2_radius55/raw_gate_weights.csv",
    "results/nature_main/cytassist_rep2_radius55/modality_reliability.csv",
    "results/nature_main/cytassist_rep2_radius55/spot_uncertainty.csv",
    "results/nature_main/cytassist_rep2_radius55/agent_attention.csv",
    "results/nature_main/cytassist_rep2_radius55/nature_analysis/proportion_maps/proportion_map_manifest.json",
    "results/nature_main/cytassist_rep2_radius55/nature_analysis/proportion_maps/predicted_top_celltypes_panel.png",
    "results/nature_main/cytassist_rep2_radius55/nature_analysis/proportion_maps/predicted_tumor_immune_stromal_group_panel.png",
    "results/nature_main/cytassist_rep2_radius55/nature_analysis/reliability_summary.json",
    "results/nature_main/cytassist_rep2_radius55/nature_analysis/boundary_summary.json",
    "results/nature_main/cytassist_rep2_radius55/nature_analysis/niche_summary.json",
    "results/nature_main/cytassist_rep2_radius55/split_sensitivity/original_split_gt_summary.csv",
    "results/nature_main/cytassist_rep2_radius55/split_sensitivity/split_sensitivity_aggregate.csv",
    "results/nature_main/cytassist_rep2_radius55/split_sensitivity/split_sensitivity_manifest.json",
    "results/nature_main/cytassist_rep2_radius55/benchmark_sensitivity/radius_cell_count_sensitivity.csv",
    "results/nature_main/cytassist_rep2_radius55/benchmark_sensitivity/radius_coverage_summary.csv",
    "results/nature_main/cytassist_rep2_radius55/benchmark_sensitivity/radius_cell_count_sensitivity_manifest.json",
    "results/nature_main/cytassist_rep2_radius55_split_guarded/metrics.csv",
    "results/nature_main/cytassist_rep2_radius55_split_guarded/predicted_proportions.csv",
    "results/nature_main/cytassist_rep2_radius55_imagegate/checkpoint.pt",
    "results/nature_main/cytassist_rep2_radius55_imagegate/metrics.csv",
    "results/nature_main/cytassist_rep2_radius55_imagegate/training_history.csv",
    "results/nature_main/cytassist_rep2_radius55_imagegate/gate_weights.csv",
    "results/nature_main/cytassist_rep2_radius55_imagegate/raw_gate_weights.csv",
    "results/nature_main/cytassist_rep2_radius55_imagegate/nature_analysis/reliability_summary.json",
    "results/nature_main/cytassist_rep2_radius55_imagegate/nature_analysis/boundary_summary.json",
    "results/nature_main/cytassist_rep2_radius55_imagegate/image_contribution/image_contribution_summary.json",
    "results/nature_main/cytassist_rep2_radius55_imagegate/image_contribution/image_contribution_texture_groups.csv",
    "results/nature_main/cytassist_rep2_radius55_imagegate_noimage/metrics.csv",
    "results/nature_main/cytassist_rep2_radius55_imagegate_noimage/predicted_proportions.csv",
    "results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/rep1_retune_budget_curve/rep1_minimal_retune_budget_curve.csv",
    "results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/rep1_retune_budget_curve/rep1_minimal_retune_budget_curve_manifest.json",
    "results/nature_manuscript_tables/table_8_split_sensitivity.csv",
    "results/nature_manuscript_tables/table_9_imagegate_supplement.csv",
    "results/nature_manuscript_tables/table_10_benchmark_sensitivity.csv",
    "results/nature_manuscript_tables/table_11_rep1_retune_budget_curve.csv",
    "results/nature_manuscript_figures/figure_manifest.json",
    "results/nature_manuscript_figures/figure_manifest.csv",
    "results/nature_manuscript_figures/figure_manifest.md",
    "results/nature_manuscript_figures/figure_1_workflow_schematic.png",
    "results/nature_manuscript_figures/figure_2_spatial_cell_composition.png",
    "results/nature_manuscript_figures/figure_3_baseline_performance.png",
    "results/nature_manuscript_figures/figure_4_reliability_calibration.png",
    "results/nature_manuscript_figures/figure_5_boundary_niche_pathology.png",
    "results/nature_manuscript_statements/manuscript_availability_statements.json",
    "results/nature_manuscript_statements/manuscript_availability_statements.md",
    "results/nature_manuscript_methods/manuscript_methods.json",
    "results/nature_manuscript_methods/manuscript_methods.md",
    "results/nature_manuscript_figure_legends/figure_legends.json",
    "results/nature_manuscript_figure_legends/figure_legends.md",
    "results/nature_benchmark_datasheet/benchmark_datasheet.json",
    "results/nature_benchmark_datasheet/benchmark_datasheet.md",
    "results/nature_completion_audit/goal_completion_audit.json",
    "results/nature_completion_audit/goal_completion_audit.md",
    "results/nature_final_doi_gate/final_doi_gate.json",
    "results/nature_final_doi_gate/final_doi_gate.md",
    "results/nature_reviewer_preflight/reviewer_preflight.json",
    "results/nature_reviewer_preflight/reviewer_preflight.md",
    "results/nature_leakage_fairness_audit/leakage_fairness_audit.json",
    "results/nature_leakage_fairness_audit/leakage_fairness_audit.md",
    "results/nature_leakage_fairness_audit/leakage_fairness_checklist.csv",
    "results/nature_leakage_fairness_audit/baseline_fairness_table.csv",
    "results/nature_leakage_fairness_audit/label_permutation_control.csv",
]

EXCLUDE_SUFFIXES = {".pt", ".pth", ".tar", ".tgz", ".zip", ".h5", ".h5ad", ".tif", ".tiff", ".svs", ".ndpi"}
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".git"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(paths: Iterable[str | Path], max_file_bytes: int) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        if path.is_file():
            candidates = [path]
            explicit_file = True
        else:
            candidates = [candidate for candidate in path.rglob("*") if candidate.is_file()]
            explicit_file = False
        for candidate in candidates:
            if any(part in EXCLUDE_PARTS for part in candidate.parts):
                continue
            if not explicit_file and candidate.suffix.lower() in EXCLUDE_SUFFIXES:
                continue
            try:
                if candidate.stat().st_size > max_file_bytes:
                    continue
            except FileNotFoundError:
                continue
            files.append(candidate)
    unique = []
    seen = set()
    for path in sorted(files):
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _write_upload_manifest(files: list[Path], output_dir: Path) -> Path:
    path = output_dir / "release_upload_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        for file_path in files:
            writer.writerow({"path": str(file_path), "bytes": file_path.stat().st_size, "sha256": _sha256(file_path)})
    return path


def _write_zenodo_metadata(output_dir: Path, version: str, title: str) -> Path:
    path = output_dir / "zenodo_metadata.json"
    metadata = {
        "metadata": {
            "title": title,
            "upload_type": "software",
            "publication_type": "article",
            "description": (
                "WaveST-Gate release package for a reproducible Xenium-to-Visium breast cancer "
                "spatial deconvolution benchmark, reliability-calibrated multimodal model evidence, "
                "baseline comparisons, ablations, boundary analysis, niche interpretation, external "
                "generalization summaries, and robustness statistics."
            ),
            "creators": [{"name": "WaveST-Gate contributors"}],
            "version": version,
            "license": "MIT",
            "access_right": "open",
            "keywords": [
                "spatial transcriptomics",
                "Xenium",
                "Visium",
                "breast cancer",
                "deconvolution",
                "multimodal learning",
                "reliability calibration",
            ],
            "communities": [],
            "related_identifiers": [
                {
                    "identifier": "10.5281/zenodo.4739739",
                    "relation": "cites",
                    "resource_type": "dataset",
                },
            ],
            "notes": (
                "This metadata is ready for a Zenodo deposition draft. A real DOI is only created "
                "after uploading the bundle and publishing/reserving a DOI through Zenodo."
            ),
        }
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def _critical_artifact_manifest(files: list[Path], critical_artifact_paths: Iterable[str | Path] | None = None) -> list[dict[str, object]]:
    bundled = {str(path) for path in files}
    rows: list[dict[str, object]] = []
    for raw_path in critical_artifact_paths or CRITICAL_ARTIFACT_PATHS:
        raw_path = str(raw_path)
        path = Path(raw_path)
        exists = path.exists()
        rows.append(
            {
                "path": raw_path,
                "exists": exists,
                "bundled": raw_path in bundled,
                "bytes": path.stat().st_size if exists else 0,
                "sha256": _sha256(path) if exists else "",
            }
        )
    return rows


def _write_deposition_instructions(output_dir: Path, metadata_path: Path, bundle_path: Path, manifest_path: Path) -> Path:
    path = output_dir / "zenodo_deposition_instructions.md"
    token_present = bool(os.environ.get("ZENODO_ACCESS_TOKEN") or os.environ.get("ZENODO_TOKEN"))
    lines = [
        "# Zenodo Deposition Instructions",
        "",
        "This workspace has prepared a Zenodo-ready metadata file and evidence bundle.",
        "",
        f"- Metadata: `{metadata_path}`",
        f"- Bundle: `{bundle_path}`",
        f"- Upload manifest: `{manifest_path}`",
        f"- Zenodo token present in environment: `{token_present}`",
        "",
        "A machine-executable deposition helper is available:",
        "",
        "```bash",
        "python -m wavestgate.evaluation.zenodo_deposit --dry-run",
        "ZENODO_ACCESS_TOKEN=<token> python -m wavestgate.evaluation.zenodo_deposit --sandbox",
        "ZENODO_ACCESS_TOKEN=<token> python -m wavestgate.evaluation.zenodo_deposit",
        "ZENODO_ACCESS_TOKEN=<token> python -m wavestgate.evaluation.zenodo_deposit --publish",
        "```",
        "",
        "Use `--sandbox` first to validate the API flow. Omit `--sandbox` for the production draft. Use `--publish` only after reviewing the draft, because publishing registers the DOI and makes the record public. The helper writes `zenodo_deposition_result.json` and updates `release_bundle_manifest.json` with the deposition id, DOI, and record URL when Zenodo returns them.",
        "",
        "The bundle intentionally excludes raw public data and unlisted large binaries. Explicit manuscript-critical artifacts, including the main WaveST-Gate checkpoint, are bundled when listed in the release manifest. Public raw data are tracked by manifests and source accessions; reproducible benchmark tables and result evidence are included.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def prepare_release_bundle(
    output_dir: str | Path = "results/nature_release",
    version: str = "0.1.0",
    title: str = "WaveST-Gate: Xenium-to-Visium Breast Cancer Spatial Deconvolution Benchmark and Evidence",
    include_paths: list[str | Path] | None = None,
    critical_artifact_paths: list[str | Path] | None = None,
    max_file_bytes: int = 64 * 1024 * 1024,
) -> dict[str, str]:
    """Create release metadata, upload manifest, and a compact evidence tarball."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_paths = include_paths or DEFAULT_INCLUDE_PATHS
    files = _iter_files(include_paths, max_file_bytes=max_file_bytes)
    manifest_path = _write_upload_manifest(files, output_dir)
    metadata_path = _write_zenodo_metadata(output_dir, version=version, title=title)

    bundle_path = output_dir / f"wavestgate_submission_evidence_v{version}.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tar:
        for file_path in files:
            tar.add(file_path, arcname=file_path)
        tar.add(manifest_path, arcname=manifest_path)
        tar.add(metadata_path, arcname=metadata_path)

    instructions_path = _write_deposition_instructions(output_dir, metadata_path, bundle_path, manifest_path)
    critical_artifacts = _critical_artifact_manifest(files, critical_artifact_paths=critical_artifact_paths)
    bundle_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "title": title,
        "release_status": "zenodo_ready_not_deposited",
        "doi": "",
        "zenodo_token_present": bool(os.environ.get("ZENODO_ACCESS_TOKEN") or os.environ.get("ZENODO_TOKEN")),
        "num_files": len(files),
        "bundle_path": str(bundle_path),
        "bundle_bytes": bundle_path.stat().st_size,
        "bundle_sha256": _sha256(bundle_path),
        "upload_manifest": str(manifest_path),
        "zenodo_metadata": str(metadata_path),
        "instructions": str(instructions_path),
        "critical_artifacts": critical_artifacts,
        "num_critical_artifacts": len(critical_artifacts),
        "missing_critical_artifacts": [row["path"] for row in critical_artifacts if not row["exists"]],
        "unbundled_critical_artifacts": [row["path"] for row in critical_artifacts if row["exists"] and not row["bundled"]],
        "excluded_large_or_raw_data_note": "Raw public data and large binary files are referenced through manifests rather than bundled.",
    }
    bundle_manifest_path = output_dir / "release_bundle_manifest.json"
    bundle_manifest_path.write_text(json.dumps(bundle_manifest, indent=2), encoding="utf-8")
    return {
        "bundle_manifest": str(bundle_manifest_path),
        "bundle_path": str(bundle_path),
        "upload_manifest": str(manifest_path),
        "zenodo_metadata": str(metadata_path),
        "instructions": str(instructions_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a Zenodo-ready WaveST-Gate release bundle.")
    parser.add_argument("--output-dir", default="results/nature_release")
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--title", default="WaveST-Gate: Xenium-to-Visium Breast Cancer Spatial Deconvolution Benchmark and Evidence")
    parser.add_argument("--max-file-mb", type=int, default=64)
    args = parser.parse_args()
    outputs = prepare_release_bundle(
        output_dir=args.output_dir,
        version=args.version,
        title=args.title,
        max_file_bytes=args.max_file_mb * 1024 * 1024,
    )
    print(json.dumps(outputs, indent=2), flush=True)


if __name__ == "__main__":
    main()
