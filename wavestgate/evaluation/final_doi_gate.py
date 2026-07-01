"""Final DOI gate for WaveST-Gate submission readiness."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RELEASE_DIR = Path("results/nature_release")
DEFAULT_COMPLETION_AUDIT_DIR = Path("results/nature_completion_audit")
DEFAULT_READINESS_DIR = Path("results/nature_submission_readiness")
DEFAULT_OUTPUT_DIR = Path("results/nature_final_doi_gate")


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _token_present() -> bool:
    return bool(os.environ.get("ZENODO_ACCESS_TOKEN") or os.environ.get("ZENODO_TOKEN"))


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# WaveST-Gate Final DOI Gate",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        "",
        f"Status: `{payload['status']}`",
        f"Ready to deposit: `{payload['ready_to_deposit']}`",
        f"Complete for submission: `{payload['complete_for_submission']}`",
        "",
        "## Checks",
        "",
    ]
    for check in payload["checks"]:
        lines.append(f"- `{check['name']}`: `{check['status']}` - {check['detail']}")
    if payload["blocking_reasons"]:
        lines.extend(["", "## Blocking Reasons", ""])
        for reason in payload["blocking_reasons"]:
            lines.append(f"- {reason}")
    lines.extend(["", "## Next Commands", ""])
    for command in payload["next_commands"]:
        lines.extend(["```bash", command, "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_final_doi_gate(
    release_dir: str | Path = DEFAULT_RELEASE_DIR,
    completion_audit_dir: str | Path = DEFAULT_COMPLETION_AUDIT_DIR,
    readiness_dir: str | Path = DEFAULT_READINESS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Write a final DOI gate report.

    The gate is intentionally strict: local evidence may be complete, but the
    submission is not marked complete until a project Zenodo DOI and deposition
    id are recorded in explicit published deposition fields.
    """

    release_dir = Path(release_dir)
    completion_audit_dir = Path(completion_audit_dir)
    readiness_dir = Path(readiness_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    deposition = _read_json(release_dir / "zenodo_deposition_result.json")
    verification = _read_json(release_dir / "release_verification.json")
    bundle_manifest = _read_json(release_dir / "release_bundle_manifest.json")
    completion = _read_json(completion_audit_dir / "goal_completion_audit.json")
    readiness = _read_json(readiness_dir / "readiness_report.json")

    doi = str(deposition.get("doi") or bundle_manifest.get("doi") or "")
    deposition_id = str(deposition.get("zenodo_deposition_id") or bundle_manifest.get("zenodo_deposition_id") or "")
    release_status = str(deposition.get("release_status") or bundle_manifest.get("release_status") or "")
    doi_published = bool(doi and deposition_id and release_status == "zenodo_published")
    token_present = _token_present()
    bundle_ok = (
        verification.get("bundle_integrity_status") == "passed"
        and int(verification.get("num_failures", 1) or 0) == 0
        and not bundle_manifest.get("missing_critical_artifacts")
        and not bundle_manifest.get("unbundled_critical_artifacts")
    )
    local_goal_ok = completion.get("overall_status") in {"complete", "complete_except_project_doi"} and int(completion.get("num_missing", 1) or 0) == 0
    readiness_ok = readiness.get("overall_status") in {"complete", "partial"} and not any(
        summary.get("num_missing", 0) for summary in readiness.get("stage_summary", {}).values()
    )
    doi_recorded = bool(doi and deposition_id)
    verification_doi_recorded = verification.get("doi_status") in {"published", "recorded"}
    verification_doi_published = verification.get("doi_status") == "published"

    checks = [
        {"name": "release_bundle_integrity", "status": "pass" if bundle_ok else "fail", "detail": f"bundle_integrity_status={verification.get('bundle_integrity_status', '')}; num_failures={verification.get('num_failures', '')}"},
        {"name": "critical_artifacts_bundled", "status": "pass" if not bundle_manifest.get("missing_critical_artifacts") and not bundle_manifest.get("unbundled_critical_artifacts") else "fail", "detail": f"missing={len(bundle_manifest.get('missing_critical_artifacts', []))}; unbundled={len(bundle_manifest.get('unbundled_critical_artifacts', []))}"},
        {"name": "completion_audit_local_requirements", "status": "pass" if local_goal_ok else "fail", "detail": f"overall_status={completion.get('overall_status', '')}; num_missing={completion.get('num_missing', '')}"},
        {"name": "readiness_no_missing_records", "status": "pass" if readiness_ok else "fail", "detail": f"overall_status={readiness.get('overall_status', '')}"},
        {"name": "zenodo_token_present", "status": "pass" if token_present else "warn", "detail": "A token is needed only when creating/updating the real deposition."},
        {"name": "project_doi_recorded", "status": "pass" if doi_recorded else "fail", "detail": f"doi={doi}; zenodo_deposition_id={deposition_id}"},
        {"name": "project_doi_published", "status": "pass" if doi_published else "fail", "detail": f"release_status={release_status}; doi={doi}; zenodo_deposition_id={deposition_id}"},
        {"name": "release_verification_doi_recorded", "status": "pass" if verification_doi_recorded else "fail", "detail": f"doi_status={verification.get('doi_status', '')}"},
        {"name": "release_verification_doi_published", "status": "pass" if verification_doi_published else "fail", "detail": f"doi_status={verification.get('doi_status', '')}"},
    ]
    blocking_reasons = []
    if not bundle_ok:
        blocking_reasons.append("Release bundle integrity or critical artifact bundling is not clean.")
    if not local_goal_ok:
        blocking_reasons.append("Requirement-by-requirement completion audit still has missing non-DOI evidence.")
    if not doi_recorded:
        blocking_reasons.append("Project Zenodo DOI and deposition id are not recorded.")
    if doi_recorded and not doi_published:
        blocking_reasons.append("Project Zenodo DOI is draft/reserved but not published publicly.")
    if not verification_doi_recorded:
        blocking_reasons.append("Release verification has not confirmed a recorded project DOI.")
    if verification_doi_recorded and not verification_doi_published:
        blocking_reasons.append("Release verification has not confirmed a published public project DOI.")
    if not token_present and not doi_recorded:
        blocking_reasons.append("No ZENODO_ACCESS_TOKEN or ZENODO_TOKEN is present, so the real deposition cannot be created from this environment.")

    ready_to_deposit = bool(bundle_ok and local_goal_ok and not doi_published)
    complete_for_submission = bool(bundle_ok and local_goal_ok and doi_published and verification_doi_published)
    status = "complete" if complete_for_submission else ("ready_for_token_deposition" if ready_to_deposit else "not_ready")
    next_commands = [
        "python -m wavestgate.evaluation.final_doi_gate --strict",
        "ZENODO_ACCESS_TOKEN=<token> python -m wavestgate.evaluation.finalize_submission --deposit",
        "python -m wavestgate.evaluation.final_doi_gate --strict",
    ]
    if not token_present:
        next_commands.insert(1, "export ZENODO_ACCESS_TOKEN=<token>")

    json_path = output_dir / "final_doi_gate.json"
    markdown_path = output_dir / "final_doi_gate.md"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "ready_to_deposit": ready_to_deposit,
        "complete_for_submission": complete_for_submission,
        "token_present": token_present,
        "doi": doi,
        "zenodo_deposition_id": deposition_id,
        "zenodo_record_url": deposition.get("zenodo_record_url") or bundle_manifest.get("zenodo_record_url", ""),
        "checks": checks,
        "blocking_reasons": blocking_reasons,
        "next_commands": next_commands,
        "outputs": {"json": str(json_path), "markdown": str(markdown_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(payload, markdown_path)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check the final WaveST-Gate DOI gate.")
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--completion-audit-dir", type=Path, default=DEFAULT_COMPLETION_AUDIT_DIR)
    parser.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless the project DOI is recorded and verified.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    payload = build_final_doi_gate(
        release_dir=args.release_dir,
        completion_audit_dir=args.completion_audit_dir,
        readiness_dir=args.readiness_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload["outputs"], indent=2), flush=True)
    if args.strict and not payload["complete_for_submission"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
