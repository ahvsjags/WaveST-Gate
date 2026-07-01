import json
from pathlib import Path

import wavestgate.evaluation.finalize_submission as finalize_module
from wavestgate.evaluation.finalize_submission import build_arg_parser, finalize_submission


def test_finalize_submission_writes_handoff_manifest(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_environment(**kwargs):
        calls.append("environment")
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text(json.dumps({"python": {"version": "test"}}), encoding="utf-8")
        Path(kwargs["markdown_path"]).write_text("# env\n", encoding="utf-8")
        return {"python": {"version": "test"}}

    def fake_benchmark_datasheet(**kwargs):
        calls.append("benchmark_datasheet")
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "benchmark_datasheet.json"
        markdown_path = output_dir / "benchmark_datasheet.md"
        payload = {"outputs": {"json": str(json_path), "markdown": str(markdown_path)}, "status": "complete"}
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        markdown_path.write_text("# datasheet\n", encoding="utf-8")
        return payload

    def fake_completion_audit(**kwargs):
        calls.append("completion_audit")
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "goal_completion_audit.json"
        markdown_path = output_dir / "goal_completion_audit.md"
        payload = {"outputs": {"json": str(json_path), "markdown": str(markdown_path)}, "overall_status": "complete_except_project_doi"}
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        markdown_path.write_text("# completion audit\n", encoding="utf-8")
        return payload

    def fake_final_doi_gate(**kwargs):
        calls.append("final_doi_gate")
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "final_doi_gate.json"
        markdown_path = output_dir / "final_doi_gate.md"
        payload = {"outputs": {"json": str(json_path), "markdown": str(markdown_path)}, "status": "ready_for_token_deposition"}
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        markdown_path.write_text("# final doi gate\n", encoding="utf-8")
        return payload

    def fake_tables(args):
        calls.append("tables")
        path = Path(args.output_dir) / "manuscript_tables_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tables": {}}), encoding="utf-8")
        return {"markdown_index": str(Path(args.output_dir) / "manuscript_tables.md")}

    def fake_figures(**kwargs):
        calls.append("figures")
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "figure_manifest.json").write_text(json.dumps({"num_fail": 0}), encoding="utf-8")
        return {"num_figures": 6, "num_pass": 6, "num_fail": 0}

    def fake_figure_legends(**kwargs):
        calls.append("figure_legends")
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "figure_legends.json"
        markdown_path = output_dir / "figure_legends.md"
        payload = {"outputs": {"json": str(json_path), "markdown": str(markdown_path)}, "status": "complete"}
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        markdown_path.write_text("# legends\n", encoding="utf-8")
        return payload

    def fake_reviewer_preflight(**kwargs):
        calls.append("reviewer_preflight")
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "reviewer_preflight.json"
        markdown_path = output_dir / "reviewer_preflight.md"
        payload = {"outputs": {"json": str(json_path), "markdown": str(markdown_path)}, "status": "complete_except_project_doi"}
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        markdown_path.write_text("# reviewer preflight\n", encoding="utf-8")
        return payload

    def fake_leakage_fairness(**kwargs):
        calls.append("leakage_fairness")
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "leakage_fairness_audit.json"
        markdown_path = output_dir / "leakage_fairness_audit.md"
        payload = {
            "outputs": {"summary_json": str(json_path), "summary_markdown": str(markdown_path)},
            "overall_status": "complete_with_primary_benchmark_caveat",
        }
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        markdown_path.write_text("# leakage fairness\n", encoding="utf-8")
        return payload

    def fake_statements(**kwargs):
        calls.append("statements")
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "manuscript_availability_statements.json"
        markdown_path = output_dir / "manuscript_availability_statements.md"
        payload = {"outputs": {"json": str(json_path), "markdown": str(markdown_path)}, "release": {"doi": ""}}
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        markdown_path.write_text("# statements\n", encoding="utf-8")
        return payload

    def fake_methods(**kwargs):
        calls.append("methods")
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "manuscript_methods.json"
        markdown_path = output_dir / "manuscript_methods.md"
        payload = {"outputs": {"json": str(json_path), "markdown": str(markdown_path)}, "status": "complete"}
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        markdown_path.write_text("# methods\n", encoding="utf-8")
        return payload

    def fake_release(output_dir, version):
        calls.append("release")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        bundle = output_dir / "bundle.tar.gz"
        bundle.write_bytes(b"bundle")
        manifest = {
            "bundle_path": str(bundle),
            "bundle_bytes": 6,
            "bundle_sha256": "abc",
            "num_files": 2,
            "num_critical_artifacts": 1,
            "missing_critical_artifacts": [],
            "unbundled_critical_artifacts": [],
            "zenodo_metadata": str(output_dir / "zenodo_metadata.json"),
        }
        manifest_path = output_dir / "release_bundle_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return {"bundle_manifest": str(manifest_path), "bundle_path": str(bundle)}

    def fake_deposit(**kwargs):
        calls.append("deposit")
        assert kwargs["dry_run"] is True
        path = Path(kwargs["output_path"])
        payload = {"zenodo_deposition_result": str(path), "release_status": "dry_run_token_missing", "doi": "", "zenodo_deposition_id": ""}
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fake_verify(**kwargs):
        calls.append("verify")
        return {
            "overall_status": "passed_with_warnings",
            "bundle_integrity_status": "passed",
            "doi_status": "pending_token",
            "num_failures": 0,
            "num_warnings": 1,
            "tar_member_count": 4,
            "upload_manifest_rows": 2,
            "critical_artifacts_checked": 1,
        }

    def fake_readiness(**kwargs):
        calls.append("readiness")
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "overall_status": "partial",
            "stage_summary": {
                "1. Xenium-to-Visium benchmark": {"status": "partial"},
                "2. Main model training": {"status": "complete"},
            },
        }
        json_path = output_dir / "readiness_report.json"
        md_path = output_dir / "readiness_report.md"
        json_path.write_text(json.dumps(report), encoding="utf-8")
        md_path.write_text("# report\n", encoding="utf-8")
        return {"readiness_report_json": str(json_path), "readiness_report_markdown": str(md_path)}

    monkeypatch.setattr(finalize_module, "collect_environment_report", fake_environment)
    monkeypatch.setattr(finalize_module, "build_benchmark_datasheet", fake_benchmark_datasheet)
    monkeypatch.setattr(finalize_module, "build_completion_audit", fake_completion_audit)
    monkeypatch.setattr(finalize_module, "build_final_doi_gate", fake_final_doi_gate)
    monkeypatch.setattr(finalize_module, "build_manuscript_tables", fake_tables)
    monkeypatch.setattr(finalize_module, "build_manuscript_figures", fake_figures)
    monkeypatch.setattr(finalize_module, "build_manuscript_figure_legends", fake_figure_legends)
    monkeypatch.setattr(finalize_module, "build_manuscript_statements", fake_statements)
    monkeypatch.setattr(finalize_module, "build_manuscript_methods", fake_methods)
    monkeypatch.setattr(finalize_module, "build_reviewer_preflight", fake_reviewer_preflight)
    monkeypatch.setattr(finalize_module, "build_leakage_fairness_audit", fake_leakage_fairness)
    monkeypatch.setattr(finalize_module, "prepare_release_bundle", fake_release)
    monkeypatch.setattr(finalize_module, "deposit_release_bundle", fake_deposit)
    monkeypatch.setattr(finalize_module, "verify_release_bundle", fake_verify)
    monkeypatch.setattr(finalize_module, "generate_submission_readiness", fake_readiness)

    args = build_arg_parser().parse_args(
        [
            "--benchmark-dir",
            str(tmp_path / "benchmark"),
            "--run-dir",
            str(tmp_path / "run"),
            "--external-dir",
            str(tmp_path / "external"),
            "--external-matched-gt-dir",
            str(tmp_path / "matched"),
            "--external-pathology-dir",
            str(tmp_path / "pathology"),
            "--multisample-baseline-dir",
            str(tmp_path / "multi"),
            "--minimal-retune-multisample",
            str(tmp_path / "minimal.csv"),
            "--benchmark-datasheet-dir",
            str(tmp_path / "datasheet"),
            "--completion-audit-dir",
            str(tmp_path / "completion_audit"),
            "--final-doi-gate-dir",
            str(tmp_path / "final_doi_gate"),
            "--tables-dir",
            str(tmp_path / "tables"),
            "--figures-dir",
            str(tmp_path / "figures"),
            "--statements-dir",
            str(tmp_path / "statements"),
            "--methods-dir",
            str(tmp_path / "methods"),
            "--figure-legends-dir",
            str(tmp_path / "figure_legends"),
            "--reviewer-preflight-dir",
            str(tmp_path / "reviewer_preflight"),
            "--leakage-fairness-dir",
            str(tmp_path / "leakage_fairness"),
            "--release-dir",
            str(tmp_path / "release"),
            "--readiness-dir",
            str(tmp_path / "readiness"),
            "--docs-dir",
            str(tmp_path / "docs"),
            "--handoff-json",
            str(tmp_path / "release" / "handoff.json"),
            "--handoff-md",
            str(tmp_path / "release" / "handoff.md"),
            "--environment-report-json",
            str(tmp_path / "release" / "environment_report.json"),
            "--environment-report-md",
            str(tmp_path / "release" / "environment_report.md"),
        ]
    )

    summary = finalize_submission(args)

    assert calls == [
        "environment",
        "benchmark_datasheet",
        "tables",
        "figures",
        "statements",
        "methods",
        "figure_legends",
        "reviewer_preflight",
        "leakage_fairness",
        "release",
        "deposit",
        "verify",
        "readiness",
        "completion_audit",
        "final_doi_gate",
        "reviewer_preflight",
        "release",
        "deposit",
        "verify",
        "readiness",
    ]
    assert summary["readiness_overall_status"] == "partial"
    assert summary["release_verification"]["bundle_integrity_status"] == "passed"
    assert summary["environment"]["python"] == "test"
    assert Path(summary["outputs"]["handoff_json"]).exists()
    assert Path(summary["outputs"]["handoff_md"]).exists()
    assert Path(summary["outputs"]["benchmark_datasheet_json"]).exists()
    assert Path(summary["outputs"]["completion_audit_json"]).exists()
    assert Path(summary["outputs"]["final_doi_gate_json"]).exists()
    assert Path(summary["outputs"]["statements_json"]).exists()
    assert Path(summary["outputs"]["methods_json"]).exists()
    assert Path(summary["outputs"]["figure_legends_json"]).exists()
    assert Path(summary["outputs"]["reviewer_preflight_json"]).exists()
    assert Path(summary["outputs"]["leakage_fairness_json"]).exists()


def test_finalize_submission_reuses_first_deposition_id_when_depositing(tmp_path: Path, monkeypatch) -> None:
    calls = []
    deposit_deposition_ids = []

    def fake_environment(**kwargs):
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text(json.dumps({"python": {"version": "test"}}), encoding="utf-8")
        Path(kwargs["markdown_path"]).write_text("# env\n", encoding="utf-8")
        return {"python": {"version": "test"}}

    def fake_builder(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"outputs": {"json": str(output_dir / "out.json"), "markdown": str(output_dir / "out.md")}}

    def fake_tables(args):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        return {}

    def fake_release(output_dir, version):
        calls.append("release")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "release_bundle_manifest.json"
        bundle_path = output_dir / "bundle.tar.gz"
        bundle_path.write_bytes(b"bundle")
        manifest_path.write_text(json.dumps({"bundle_path": str(bundle_path), "bundle_sha256": "abc", "missing_critical_artifacts": [], "unbundled_critical_artifacts": []}), encoding="utf-8")
        return {"bundle_manifest": str(manifest_path), "bundle_path": str(bundle_path)}

    def fake_deposit(**kwargs):
        calls.append("deposit")
        deposit_deposition_ids.append(kwargs["deposition_id"])
        path = Path(kwargs["output_path"])
        payload = {"zenodo_deposition_result": str(path), "release_status": "zenodo_draft_reserved", "doi": "10.5072/zenodo.123", "zenodo_deposition_id": "123"}
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fake_verify(**kwargs):
        calls.append("verify")
        return {"overall_status": "passed", "bundle_integrity_status": "passed", "doi_status": "recorded", "num_failures": 0}

    def fake_readiness(**kwargs):
        calls.append("readiness")
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "readiness_report.json"
        md_path = output_dir / "readiness_report.md"
        json_path.write_text(json.dumps({"overall_status": "complete", "stage_summary": {}}), encoding="utf-8")
        md_path.write_text("# report\n", encoding="utf-8")
        return {"readiness_report_json": str(json_path), "readiness_report_markdown": str(md_path)}

    def fake_reviewer_preflight(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "reviewer_preflight.json"
        md_path = output_dir / "reviewer_preflight.md"
        json_path.write_text(json.dumps({"outputs": {"json": str(json_path), "markdown": str(md_path)}}), encoding="utf-8")
        md_path.write_text("# preflight\n", encoding="utf-8")
        return {"outputs": {"json": str(json_path), "markdown": str(md_path)}}

    def fake_leakage_fairness(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "leakage_fairness_audit.json"
        md_path = output_dir / "leakage_fairness_audit.md"
        json_path.write_text(json.dumps({"outputs": {"summary_json": str(json_path), "summary_markdown": str(md_path)}}), encoding="utf-8")
        md_path.write_text("# audit\n", encoding="utf-8")
        return {"outputs": {"summary_json": str(json_path), "summary_markdown": str(md_path)}}

    monkeypatch.setattr(finalize_module, "collect_environment_report", fake_environment)
    monkeypatch.setattr(finalize_module, "build_benchmark_datasheet", fake_builder)
    monkeypatch.setattr(finalize_module, "build_completion_audit", fake_builder)
    monkeypatch.setattr(finalize_module, "build_final_doi_gate", fake_builder)
    monkeypatch.setattr(finalize_module, "build_manuscript_tables", fake_tables)
    monkeypatch.setattr(finalize_module, "build_manuscript_figures", lambda **kwargs: {})
    monkeypatch.setattr(finalize_module, "build_manuscript_statements", fake_builder)
    monkeypatch.setattr(finalize_module, "build_manuscript_methods", fake_builder)
    monkeypatch.setattr(finalize_module, "build_manuscript_figure_legends", fake_builder)
    monkeypatch.setattr(finalize_module, "build_reviewer_preflight", fake_reviewer_preflight)
    monkeypatch.setattr(finalize_module, "build_leakage_fairness_audit", fake_leakage_fairness)
    monkeypatch.setattr(finalize_module, "prepare_release_bundle", fake_release)
    monkeypatch.setattr(finalize_module, "deposit_release_bundle", fake_deposit)
    monkeypatch.setattr(finalize_module, "verify_release_bundle", fake_verify)
    monkeypatch.setattr(finalize_module, "generate_submission_readiness", fake_readiness)

    args = build_arg_parser().parse_args(
        [
            "--benchmark-dir", str(tmp_path / "benchmark"),
            "--run-dir", str(tmp_path / "run"),
            "--external-dir", str(tmp_path / "external"),
            "--external-matched-gt-dir", str(tmp_path / "matched"),
            "--external-pathology-dir", str(tmp_path / "pathology"),
            "--multisample-baseline-dir", str(tmp_path / "multi"),
            "--minimal-retune-multisample", str(tmp_path / "minimal.csv"),
            "--benchmark-datasheet-dir", str(tmp_path / "datasheet"),
            "--completion-audit-dir", str(tmp_path / "completion"),
            "--final-doi-gate-dir", str(tmp_path / "gate"),
            "--tables-dir", str(tmp_path / "tables"),
            "--figures-dir", str(tmp_path / "figures"),
            "--statements-dir", str(tmp_path / "statements"),
            "--methods-dir", str(tmp_path / "methods"),
            "--figure-legends-dir", str(tmp_path / "legends"),
            "--reviewer-preflight-dir", str(tmp_path / "preflight"),
            "--leakage-fairness-dir", str(tmp_path / "leakage"),
            "--release-dir", str(tmp_path / "release"),
            "--readiness-dir", str(tmp_path / "readiness"),
            "--docs-dir", str(tmp_path / "docs"),
            "--handoff-json", str(tmp_path / "release" / "handoff.json"),
            "--handoff-md", str(tmp_path / "release" / "handoff.md"),
            "--environment-report-json", str(tmp_path / "release" / "environment_report.json"),
            "--environment-report-md", str(tmp_path / "release" / "environment_report.md"),
            "--deposit",
        ]
    )

    finalize_submission(args)

    assert deposit_deposition_ids == [None, "123"]
