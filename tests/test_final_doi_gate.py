import json
from pathlib import Path

from wavestgate.evaluation.final_doi_gate import build_final_doi_gate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_final_doi_gate_ready_for_token_deposition(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ZENODO_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("ZENODO_TOKEN", raising=False)
    release = tmp_path / "release"
    _write_json(
        release / "release_verification.json",
        {"bundle_integrity_status": "passed", "num_failures": 0, "doi_status": "pending_token"},
    )
    _write_json(
        release / "release_bundle_manifest.json",
        {"missing_critical_artifacts": [], "unbundled_critical_artifacts": [], "doi": "", "zenodo_deposition_id": ""},
    )
    _write_json(release / "zenodo_deposition_result.json", {"doi": "", "zenodo_deposition_id": ""})
    audit = tmp_path / "audit"
    _write_json(audit / "goal_completion_audit.json", {"overall_status": "complete_except_project_doi", "num_missing": 0})
    readiness = tmp_path / "readiness"
    _write_json(readiness / "readiness_report.json", {"overall_status": "partial", "stage_summary": {"stage": {"num_missing": 0}}})

    payload = build_final_doi_gate(release_dir=release, completion_audit_dir=audit, readiness_dir=readiness, output_dir=tmp_path / "gate")

    assert payload["status"] == "ready_for_token_deposition"
    assert payload["ready_to_deposit"] is True
    assert payload["complete_for_submission"] is False
    assert "No ZENODO_ACCESS_TOKEN" in " ".join(payload["blocking_reasons"])
    assert Path(payload["outputs"]["json"]).exists()
    assert Path(payload["outputs"]["markdown"]).exists()


def test_final_doi_gate_complete_when_doi_recorded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZENODO_ACCESS_TOKEN", "token")
    release = tmp_path / "release"
    _write_json(
        release / "release_verification.json",
        {"bundle_integrity_status": "passed", "num_failures": 0, "doi_status": "published"},
    )
    _write_json(
        release / "release_bundle_manifest.json",
        {
            "missing_critical_artifacts": [],
            "unbundled_critical_artifacts": [],
            "release_status": "zenodo_published",
            "doi": "10.5072/zenodo.123",
            "zenodo_deposition_id": "123",
        },
    )
    _write_json(
        release / "zenodo_deposition_result.json",
        {"release_status": "zenodo_published", "doi": "10.5072/zenodo.123", "zenodo_deposition_id": "123"},
    )
    audit = tmp_path / "audit"
    _write_json(audit / "goal_completion_audit.json", {"overall_status": "complete", "num_missing": 0})
    readiness = tmp_path / "readiness"
    _write_json(readiness / "readiness_report.json", {"overall_status": "complete", "stage_summary": {"stage": {"num_missing": 0}}})

    payload = build_final_doi_gate(release_dir=release, completion_audit_dir=audit, readiness_dir=readiness, output_dir=tmp_path / "gate")

    assert payload["status"] == "complete"
    assert payload["ready_to_deposit"] is False
    assert payload["complete_for_submission"] is True
    assert payload["blocking_reasons"] == []
