"""Run the final WaveST-Gate submission packaging chain."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wavestgate.evaluation.benchmark_datasheet import build_benchmark_datasheet
from wavestgate.evaluation.completion_audit import build_completion_audit
from wavestgate.evaluation.environment_report import collect_environment_report
from wavestgate.evaluation.final_doi_gate import build_final_doi_gate
from wavestgate.evaluation.leakage_fairness_audit import build_leakage_fairness_audit
from wavestgate.evaluation.manuscript_figure_legends import build_manuscript_figure_legends
from wavestgate.evaluation.manuscript_figures import build_manuscript_figures
from wavestgate.evaluation.manuscript_methods import build_manuscript_methods
from wavestgate.evaluation.manuscript_statements import build_manuscript_statements
from wavestgate.evaluation.manuscript_tables import build_arg_parser as build_tables_arg_parser
from wavestgate.evaluation.manuscript_tables import build_manuscript_tables
from wavestgate.evaluation.prepare_release import prepare_release_bundle
from wavestgate.evaluation.reviewer_preflight import build_reviewer_preflight
from wavestgate.evaluation.submission_readiness import generate_submission_readiness
from wavestgate.evaluation.verify_release import verify_release_bundle
from wavestgate.evaluation.zenodo_deposit import deposit_release_bundle


def _write_handoff_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# WaveST-Gate Final Submission Handoff",
        "",
        f"Generated UTC: {summary['generated_at_utc']}",
        "",
        f"- Overall readiness: `{summary['readiness_overall_status']}`",
        f"- Bundle integrity: `{summary['release_verification'].get('bundle_integrity_status', '')}`",
        f"- DOI status: `{summary['release_verification'].get('doi_status', '')}`",
        f"- Release status: `{summary['zenodo_result'].get('release_status', '')}`",
        f"- Bundle: `{summary['release_bundle'].get('bundle_path', '')}`",
        f"- Bundle SHA256: `{summary['release_bundle'].get('bundle_sha256', '')}`",
        "",
        "## Outputs",
        "",
        "| Output | Path |",
        "| --- | --- |",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Remaining External Action",
            "",
            "A real Zenodo DOI requires `ZENODO_ACCESS_TOKEN` and a token-backed",
            "deposition. Run with `--deposit` after reviewing the prepared bundle.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_tables_args(args: argparse.Namespace) -> argparse.Namespace:
    return build_tables_arg_parser().parse_args(
        [
            "--output-dir",
            str(args.tables_dir),
            "--run-dir",
            str(args.run_dir),
            "--benchmark-manifest",
            str(args.benchmark_dir / "xenium_visium_benchmark_manifest.json"),
            "--spot-qc",
            str(args.benchmark_dir / "spot_ground_truth_qc.csv"),
            "--main-metrics",
            str(args.run_dir / "metrics.csv"),
            "--external-no-retuning",
            str(args.external_dir / "external_no_retuning_summary.csv"),
            "--external-matched-gt",
            str(args.external_matched_gt_dir / "external_matched_gt_summary.csv"),
            "--minimal-retune-multisample",
            str(args.minimal_retune_multisample),
            "--external-pathology-dir",
            str(args.external_pathology_dir),
            "--split-sensitivity-dir",
            str(args.run_dir / "split_sensitivity"),
            "--benchmark-sensitivity-dir",
            str(args.run_dir / "benchmark_sensitivity"),
            "--image-contribution-dir",
            str(args.run_dir.parent / f"{args.run_dir.name}_imagegate" / "image_contribution"),
            "--rep1-budget-curve",
            str(args.external_matched_gt_dir / "xenium_rep1_pseudospots_radius55_common297" / "rep1_retune_budget_curve" / "rep1_minimal_retune_budget_curve.csv"),
        ]
    )


def _refresh_release_verification_readiness(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, Any], dict[str, Any], dict[str, str]]:
    release_outputs = prepare_release_bundle(output_dir=args.release_dir, version=args.version)
    zenodo_result = deposit_release_bundle(
        bundle_manifest_path=release_outputs["bundle_manifest"],
        output_path=args.release_dir / "zenodo_deposition_result.json",
        token=args.zenodo_token,
        sandbox=args.sandbox,
        publish=args.publish,
        dry_run=not args.deposit,
        deposition_id=args.deposition_id,
    )
    verification = verify_release_bundle(
        bundle_manifest_path=release_outputs["bundle_manifest"],
        deposition_result_path=zenodo_result["zenodo_deposition_result"],
        output_path=args.release_dir / "release_verification.json",
        markdown_path=args.release_dir / "release_verification.md",
        require_doi=args.deposit,
    )
    readiness = generate_submission_readiness(
        benchmark_dir=args.benchmark_dir,
        run_dir=args.run_dir,
        external_dir=args.external_dir,
        output_dir=args.readiness_dir,
        docs_dir=args.docs_dir,
        release_dir=args.release_dir,
        external_pathology_dir=args.external_pathology_dir,
        external_matched_gt_dir=args.external_matched_gt_dir,
        multisample_baseline_dir=args.multisample_baseline_dir,
        manuscript_figures_dir=args.figures_dir,
        manuscript_statements_dir=args.statements_dir,
        manuscript_methods_dir=args.methods_dir,
        manuscript_figure_legends_dir=args.figure_legends_dir,
        reviewer_preflight_dir=args.reviewer_preflight_dir,
        benchmark_datasheet_dir=args.benchmark_datasheet_dir,
        completion_audit_dir=args.completion_audit_dir,
    )
    return release_outputs, zenodo_result, verification, readiness


def finalize_submission(args: argparse.Namespace) -> dict[str, Any]:
    """Refresh tables, figures, release bundle, dry-run/deposit, verification, and readiness."""

    for attr in [
        "benchmark_dir",
        "run_dir",
        "external_dir",
        "external_matched_gt_dir",
        "external_pathology_dir",
        "multisample_baseline_dir",
        "minimal_retune_multisample",
        "benchmark_datasheet_dir",
        "completion_audit_dir",
        "final_doi_gate_dir",
        "tables_dir",
        "figures_dir",
        "statements_dir",
        "methods_dir",
        "figure_legends_dir",
        "reviewer_preflight_dir",
        "leakage_fairness_dir",
        "release_dir",
        "readiness_dir",
        "docs_dir",
        "handoff_json",
        "handoff_md",
        "environment_report_json",
        "environment_report_md",
    ]:
        setattr(args, attr, Path(getattr(args, attr)))

    environment = collect_environment_report(
        output_path=args.environment_report_json,
        markdown_path=args.environment_report_md,
        baseline_environment_path=args.run_dir / "baseline_environment_audit.json",
    )
    benchmark_datasheet = build_benchmark_datasheet(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.benchmark_datasheet_dir,
        protocol_path=args.docs_dir / "xenium_to_visium_benchmark_protocol.md",
    )
    tables = build_manuscript_tables(_build_tables_args(args))
    figures = build_manuscript_figures(
        output_dir=args.figures_dir,
        run_dir=args.run_dir,
        tables_dir=args.tables_dir,
        external_pathology_dir=args.external_pathology_dir,
    )
    statements = build_manuscript_statements(
        output_dir=args.statements_dir,
        benchmark_manifest=args.benchmark_dir / "xenium_visium_benchmark_manifest.json",
        data_manifest_dir=Path("data_manifest"),
        release_dir=args.release_dir,
        readiness_dir=args.readiness_dir,
        environment_report_path=args.environment_report_json,
    )
    methods = build_manuscript_methods(
        output_dir=args.methods_dir,
        tables_dir=args.tables_dir,
        run_dir=args.run_dir,
        benchmark_dir=args.benchmark_dir,
        environment_report_path=args.environment_report_json,
        statements_dir=args.statements_dir,
    )
    figure_legends = build_manuscript_figure_legends(
        output_dir=args.figure_legends_dir,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
        run_dir=args.run_dir,
        benchmark_dir=args.benchmark_dir,
        methods_dir=args.methods_dir,
    )
    reviewer_preflight = build_reviewer_preflight(
        output_dir=args.reviewer_preflight_dir,
        readiness_report_path=args.readiness_dir / "readiness_report.json",
        release_dir=args.release_dir,
        tables_dir=args.tables_dir,
        run_dir=args.run_dir,
        benchmark_dir=args.benchmark_dir,
        figure_legends_dir=args.figure_legends_dir,
        methods_dir=args.methods_dir,
        statements_dir=args.statements_dir,
        docs_dir=args.docs_dir,
        external_dir=args.external_dir,
        external_matched_gt_dir=args.external_matched_gt_dir,
        minimal_retune_dir=args.minimal_retune_multisample.parent,
        external_pathology_dir=args.external_pathology_dir,
        benchmark_datasheet_dir=args.benchmark_datasheet_dir,
        completion_audit_dir=args.completion_audit_dir,
    )
    leakage_fairness = build_leakage_fairness_audit(
        benchmark_dir=args.benchmark_dir,
        run_dir=args.run_dir,
        output_dir=args.leakage_fairness_dir,
    )
    release_outputs, zenodo_result, verification, readiness = _refresh_release_verification_readiness(args)
    if args.deposit and zenodo_result.get("zenodo_deposition_id") and not args.deposition_id:
        args.deposition_id = str(zenodo_result["zenodo_deposition_id"])
    completion_audit = build_completion_audit(
        benchmark_dir=args.benchmark_dir,
        run_dir=args.run_dir,
        external_dir=args.external_dir,
        external_matched_gt_dir=args.external_matched_gt_dir,
        external_pathology_dir=args.external_pathology_dir,
        multisample_dir=args.multisample_baseline_dir,
        minimal_multisample_dir=args.minimal_retune_multisample.parent,
        datasheet_dir=args.benchmark_datasheet_dir,
        release_dir=args.release_dir,
        tables_dir=args.tables_dir,
        docs_dir=args.docs_dir,
        output_dir=args.completion_audit_dir,
    )
    final_doi_gate = build_final_doi_gate(
        release_dir=args.release_dir,
        completion_audit_dir=args.completion_audit_dir,
        readiness_dir=args.readiness_dir,
        output_dir=args.final_doi_gate_dir,
    )
    reviewer_preflight = build_reviewer_preflight(
        output_dir=args.reviewer_preflight_dir,
        readiness_report_path=readiness["readiness_report_json"],
        release_dir=args.release_dir,
        tables_dir=args.tables_dir,
        run_dir=args.run_dir,
        benchmark_dir=args.benchmark_dir,
        figure_legends_dir=args.figure_legends_dir,
        methods_dir=args.methods_dir,
        statements_dir=args.statements_dir,
        docs_dir=args.docs_dir,
        external_dir=args.external_dir,
        external_matched_gt_dir=args.external_matched_gt_dir,
        minimal_retune_dir=args.minimal_retune_multisample.parent,
        external_pathology_dir=args.external_pathology_dir,
        benchmark_datasheet_dir=args.benchmark_datasheet_dir,
        completion_audit_dir=args.completion_audit_dir,
    )
    release_outputs, zenodo_result, verification, readiness = _refresh_release_verification_readiness(args)
    readiness_report = _read_json(readiness["readiness_report_json"])
    release_manifest = _read_json(release_outputs["bundle_manifest"])
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": args.version,
        "deposit_requested": bool(args.deposit),
        "sandbox": bool(args.sandbox),
        "publish_requested": bool(args.publish),
        "readiness_overall_status": readiness_report.get("overall_status", ""),
        "readiness_stage_summary": readiness_report.get("stage_summary", {}),
        "release_bundle": {
            "bundle_path": release_manifest.get("bundle_path", ""),
            "bundle_bytes": release_manifest.get("bundle_bytes", 0),
            "bundle_sha256": release_manifest.get("bundle_sha256", ""),
            "num_files": release_manifest.get("num_files", 0),
            "num_critical_artifacts": release_manifest.get("num_critical_artifacts", 0),
            "missing_critical_artifacts": release_manifest.get("missing_critical_artifacts", []),
            "unbundled_critical_artifacts": release_manifest.get("unbundled_critical_artifacts", []),
        },
        "zenodo_result": {
            "release_status": zenodo_result.get("release_status", ""),
            "doi": zenodo_result.get("doi", ""),
            "zenodo_deposition_id": zenodo_result.get("zenodo_deposition_id", ""),
            "zenodo_record_url": zenodo_result.get("zenodo_record_url", ""),
        },
        "release_verification": {
            "overall_status": verification.get("overall_status", ""),
            "bundle_integrity_status": verification.get("bundle_integrity_status", ""),
            "doi_status": verification.get("doi_status", ""),
            "num_failures": verification.get("num_failures", 0),
            "num_warnings": verification.get("num_warnings", 0),
            "tar_member_count": verification.get("tar_member_count", 0),
            "upload_manifest_rows": verification.get("upload_manifest_rows", 0),
            "critical_artifacts_checked": verification.get("critical_artifacts_checked", 0),
        },
        "environment": {
            "python": (environment.get("python") or {}).get("version", "").split(" ")[0],
            "torch": (environment.get("torch") or {}).get("version", ""),
            "cuda_available": (environment.get("torch") or {}).get("cuda_available", ""),
            "cuda_version": (environment.get("torch") or {}).get("cuda_version", ""),
            "gpu_count": (environment.get("torch") or {}).get("device_count", ""),
        },
        "outputs": {
            "environment_report_json": args.environment_report_json,
            "environment_report_md": args.environment_report_md,
            "benchmark_datasheet_json": benchmark_datasheet["outputs"]["json"],
            "benchmark_datasheet_markdown": benchmark_datasheet["outputs"]["markdown"],
            "completion_audit_json": completion_audit["outputs"]["json"],
            "completion_audit_markdown": completion_audit["outputs"]["markdown"],
            "final_doi_gate_json": final_doi_gate["outputs"]["json"],
            "final_doi_gate_markdown": final_doi_gate["outputs"]["markdown"],
            "tables_manifest": args.tables_dir / "manuscript_tables_manifest.json",
            "figures_manifest": args.figures_dir / "figure_manifest.json",
            "statements_json": statements["outputs"]["json"],
            "statements_markdown": statements["outputs"]["markdown"],
            "methods_json": methods["outputs"]["json"],
            "methods_markdown": methods["outputs"]["markdown"],
            "figure_legends_json": figure_legends["outputs"]["json"],
            "figure_legends_markdown": figure_legends["outputs"]["markdown"],
            "reviewer_preflight_json": reviewer_preflight["outputs"]["json"],
            "reviewer_preflight_markdown": reviewer_preflight["outputs"]["markdown"],
            "leakage_fairness_json": leakage_fairness["outputs"]["summary_json"],
            "leakage_fairness_markdown": leakage_fairness["outputs"]["summary_markdown"],
            "release_bundle_manifest": release_outputs["bundle_manifest"],
            "zenodo_deposition_result": zenodo_result["zenodo_deposition_result"],
            "release_verification_json": args.release_dir / "release_verification.json",
            "release_verification_md": args.release_dir / "release_verification.md",
            "readiness_report_json": readiness["readiness_report_json"],
            "readiness_report_markdown": readiness["readiness_report_markdown"],
            "handoff_json": args.handoff_json,
            "handoff_md": args.handoff_md,
        },
    }
    summary["outputs"] = {key: str(value) for key, value in summary["outputs"].items()}
    args.handoff_json.parent.mkdir(parents=True, exist_ok=True)
    args.handoff_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_handoff_markdown(summary, args.handoff_md)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize WaveST-Gate submission artifacts in the correct order.")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55"))
    parser.add_argument("--run-dir", type=Path, default=Path("results/nature_main/cytassist_rep2_radius55"))
    parser.add_argument("--external-dir", type=Path, default=Path("results/nature_external_no_retuning"))
    parser.add_argument("--external-matched-gt-dir", type=Path, default=Path("results/nature_external_matched_gt"))
    parser.add_argument("--external-pathology-dir", type=Path, default=Path("results/nature_external_pathology_validation"))
    parser.add_argument("--multisample-baseline-dir", type=Path, default=Path("results/nature_matched_multisample_baselines"))
    parser.add_argument(
        "--minimal-retune-multisample",
        type=Path,
        default=Path("results/nature_matched_multisample_baselines_minimal_retune/matched_multisample_baseline_summary.csv"),
    )
    parser.add_argument("--benchmark-datasheet-dir", type=Path, default=Path("results/nature_benchmark_datasheet"))
    parser.add_argument("--completion-audit-dir", type=Path, default=Path("results/nature_completion_audit"))
    parser.add_argument("--final-doi-gate-dir", type=Path, default=Path("results/nature_final_doi_gate"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/nature_manuscript_tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/nature_manuscript_figures"))
    parser.add_argument("--statements-dir", type=Path, default=Path("results/nature_manuscript_statements"))
    parser.add_argument("--methods-dir", type=Path, default=Path("results/nature_manuscript_methods"))
    parser.add_argument("--figure-legends-dir", type=Path, default=Path("results/nature_manuscript_figure_legends"))
    parser.add_argument("--reviewer-preflight-dir", type=Path, default=Path("results/nature_reviewer_preflight"))
    parser.add_argument("--leakage-fairness-dir", type=Path, default=Path("results/nature_leakage_fairness_audit"))
    parser.add_argument("--release-dir", type=Path, default=Path("results/nature_release"))
    parser.add_argument("--readiness-dir", type=Path, default=Path("results/nature_submission_readiness"))
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--handoff-json", type=Path, default=Path("results/nature_release/final_submission_handoff.json"))
    parser.add_argument("--handoff-md", type=Path, default=Path("results/nature_release/final_submission_handoff.md"))
    parser.add_argument("--environment-report-json", type=Path, default=Path("results/nature_release/environment_report.json"))
    parser.add_argument("--environment-report-md", type=Path, default=Path("results/nature_release/environment_report.md"))
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--deposit", action="store_true", help="Create/update a real Zenodo deposition instead of dry-run.")
    parser.add_argument("--zenodo-token", default=None)
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--deposition-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = finalize_submission(args)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
