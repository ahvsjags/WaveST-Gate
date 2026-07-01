"""Verify WaveST-Gate release bundle integrity and DOI handoff state."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    digest = hashlib.sha256()
    handle = archive.extractfile(member)
    if handle is None:
        return ""
    with handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arcname(path: str | Path) -> str:
    return str(path).lstrip("/")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _add_check(checks: list[dict[str, str]], name: str, status: str, detail: str, metric: str = "") -> None:
    checks.append({"name": name, "status": status, "detail": detail, "metric": metric})


def _read_upload_manifest_from_tar(archive: tarfile.TarFile, member: tarfile.TarInfo) -> list[dict[str, str]]:
    handle = archive.extractfile(member)
    if handle is None:
        return []
    with handle:
        text = io.TextIOWrapper(handle, encoding="utf-8", newline="")
        return list(csv.DictReader(text))


def _write_markdown_report(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# WaveST-Gate Release Verification",
        "",
        f"Generated UTC: {result['generated_at_utc']}",
        "",
        f"- Overall status: `{result['overall_status']}`",
        f"- Bundle integrity: `{result['bundle_integrity_status']}`",
        f"- DOI status: `{result['doi_status']}`",
        f"- Bundle: `{result['bundle_path']}`",
        f"- SHA256: `{result['bundle_sha256']}`",
        "",
        "| Check | Status | Detail | Metric |",
        "| --- | --- | --- | --- |",
    ]
    for check in result["checks"]:
        lines.append(f"| {check['name']} | `{check['status']}` | {check['detail']} | {check.get('metric', '')} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def verify_release_bundle(
    bundle_manifest_path: str | Path = "results/nature_release/release_bundle_manifest.json",
    deposition_result_path: str | Path | None = "results/nature_release/zenodo_deposition_result.json",
    output_path: str | Path | None = "results/nature_release/release_verification.json",
    markdown_path: str | Path | None = "results/nature_release/release_verification.md",
    require_doi: bool = False,
) -> dict[str, Any]:
    """Verify bundle checksums, tar members, critical artifacts, and DOI handoff."""

    bundle_manifest_path = Path(bundle_manifest_path)
    manifest = _read_json(bundle_manifest_path)
    checks: list[dict[str, str]] = []
    failures = 0
    warnings = 0

    def check(name: str, condition: bool, ok: str, bad: str, metric: str = "", warn: bool = False) -> None:
        nonlocal failures, warnings
        if condition:
            _add_check(checks, name, "pass", ok, metric)
        else:
            if warn:
                warnings += 1
                _add_check(checks, name, "warn", bad, metric)
            else:
                failures += 1
                _add_check(checks, name, "fail", bad, metric)

    check("bundle_manifest_exists", bool(manifest), f"Read {bundle_manifest_path}.", f"Missing or unreadable {bundle_manifest_path}.")
    bundle_path = Path(manifest.get("bundle_path", "")) if manifest else Path("")
    bundle_exists = bundle_path.exists()
    check("bundle_exists", bundle_exists, f"Found {bundle_path}.", f"Missing bundle {bundle_path}.")

    actual_bytes = bundle_path.stat().st_size if bundle_exists else 0
    expected_bytes = int(manifest.get("bundle_bytes", 0) or 0)
    check(
        "bundle_bytes_match",
        bundle_exists and actual_bytes == expected_bytes,
        "Bundle byte size matches manifest.",
        "Bundle byte size does not match manifest.",
        f"expected={expected_bytes}; actual={actual_bytes}",
    )

    actual_sha = _sha256_path(bundle_path) if bundle_exists else ""
    expected_sha = str(manifest.get("bundle_sha256", ""))
    check(
        "bundle_sha256_match",
        bool(actual_sha) and actual_sha == expected_sha,
        "Bundle SHA256 matches manifest.",
        "Bundle SHA256 does not match manifest.",
        f"expected={expected_sha}; actual={actual_sha}",
    )

    tar_names: set[str] = set()
    upload_rows: list[dict[str, str]] = []
    critical_missing: list[str] = []
    critical_hash_mismatch: list[str] = []
    upload_missing: list[str] = []
    upload_hash_mismatch: list[str] = []

    if bundle_exists:
        try:
            with tarfile.open(bundle_path, "r:gz") as archive:
                members = {member.name: member for member in archive.getmembers() if member.isfile()}
                tar_names = set(members)
                _add_check(checks, "tar_readable", "pass", "Bundle tarball is readable.", f"members={len(tar_names)}")

                upload_manifest_name = _arcname(manifest.get("upload_manifest", ""))
                metadata_name = _arcname(manifest.get("zenodo_metadata", ""))
                check(
                    "upload_manifest_in_bundle",
                    upload_manifest_name in members,
                    "Upload manifest is present in the bundle.",
                    "Upload manifest is absent from the bundle.",
                    upload_manifest_name,
                )
                check(
                    "zenodo_metadata_in_bundle",
                    metadata_name in members,
                    "Zenodo metadata is present in the bundle.",
                    "Zenodo metadata is absent from the bundle.",
                    metadata_name,
                )

                if upload_manifest_name in members:
                    upload_rows = _read_upload_manifest_from_tar(archive, members[upload_manifest_name])
                    expected_count = int(manifest.get("num_files", 0) or 0)
                    check(
                        "upload_manifest_row_count",
                        len(upload_rows) == expected_count,
                        "Upload manifest row count matches release manifest.",
                        "Upload manifest row count does not match release manifest.",
                        f"expected={expected_count}; actual={len(upload_rows)}",
                    )
                    for row in upload_rows:
                        name = _arcname(row.get("path", ""))
                        member = members.get(name)
                        if member is None:
                            upload_missing.append(name)
                            continue
                        expected_member_bytes = int(float(row.get("bytes", 0) or 0))
                        expected_member_sha = str(row.get("sha256", ""))
                        member_sha = _sha256_member(archive, member)
                        if member.size != expected_member_bytes or member_sha != expected_member_sha:
                            upload_hash_mismatch.append(name)
                    check(
                        "upload_manifest_members_present",
                        not upload_missing,
                        "All upload manifest files are present in the tarball.",
                        "Some upload manifest files are missing from the tarball.",
                        f"missing={len(upload_missing)}",
                    )
                    check(
                        "upload_manifest_member_hashes",
                        not upload_hash_mismatch,
                        "All upload manifest member byte sizes and SHA256 hashes match.",
                        "Some upload manifest member hashes or byte sizes differ.",
                        f"mismatch={len(upload_hash_mismatch)}",
                    )

                for artifact in manifest.get("critical_artifacts", []):
                    name = _arcname(artifact.get("path", ""))
                    member = members.get(name)
                    if not artifact.get("exists") or not artifact.get("bundled") or member is None:
                        critical_missing.append(name)
                        continue
                    member_sha = _sha256_member(archive, member)
                    expected_artifact_bytes = int(artifact.get("bytes", 0) or 0)
                    expected_artifact_sha = str(artifact.get("sha256", ""))
                    if member.size != expected_artifact_bytes or member_sha != expected_artifact_sha:
                        critical_hash_mismatch.append(name)
                check(
                    "critical_artifacts_present",
                    not critical_missing,
                    "All critical artifacts are present and marked bundled.",
                    "Critical artifacts are missing or not marked bundled.",
                    f"missing={len(critical_missing)}",
                )
                check(
                    "critical_artifact_hashes",
                    not critical_hash_mismatch,
                    "All critical artifact byte sizes and SHA256 hashes match tar members.",
                    "Critical artifact hashes or byte sizes differ from tar members.",
                    f"mismatch={len(critical_hash_mismatch)}",
                )
        except tarfile.TarError as exc:
            failures += 1
            _add_check(checks, "tar_readable", "fail", f"Could not read tarball: {exc}")

    deposition_result_path = Path(deposition_result_path) if deposition_result_path else None
    deposition = _read_json(deposition_result_path) if deposition_result_path else {}
    if deposition_result_path:
        check(
            "deposition_result_exists",
            bool(deposition),
            f"Read {deposition_result_path}.",
            f"Missing or unreadable {deposition_result_path}.",
            warn=not require_doi,
        )
    if deposition:
        deposition_bundle_match = (
            deposition.get("bundle_sha256") == expected_sha
            and int(deposition.get("bundle_bytes", 0) or 0) == expected_bytes
            and deposition.get("bundle_path") == manifest.get("bundle_path")
        )
        check(
            "deposition_bundle_matches_manifest",
            deposition_bundle_match,
            "Deposition dry-run/result references the same bundle path, byte size, and SHA256.",
            "Deposition dry-run/result does not match the release manifest bundle.",
            f"release_status={deposition.get('release_status', '')}",
        )
        has_doi = bool(deposition.get("doi") and deposition.get("zenodo_deposition_id"))
        is_published = deposition.get("release_status") == "zenodo_published"
        check(
            "zenodo_doi_recorded",
            has_doi,
            "Zenodo DOI and deposition id are recorded.",
            "Zenodo DOI/deposition id are not recorded yet.",
            f"release_status={deposition.get('release_status', '')}; doi={deposition.get('doi', '')}; zenodo_deposition_id={deposition.get('zenodo_deposition_id', '')}",
            warn=not require_doi,
        )
        check(
            "zenodo_release_published",
            has_doi and is_published,
            "Zenodo release is published and publicly accessible.",
            "Zenodo DOI is absent or still draft/reserved, not a public release.",
            f"release_status={deposition.get('release_status', '')}",
            warn=not require_doi,
        )

    bundle_integrity_status = "failed" if failures else "passed"
    if deposition.get("doi") and deposition.get("zenodo_deposition_id") and deposition.get("release_status") == "zenodo_published":
        doi_status = "published"
    elif deposition.get("doi") and deposition.get("zenodo_deposition_id"):
        doi_status = "draft_reserved"
    else:
        doi_status = "pending_token"
    if failures:
        overall_status = "failed"
    elif warnings:
        overall_status = "passed_with_warnings"
    else:
        overall_status = "passed"

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "bundle_integrity_status": bundle_integrity_status,
        "doi_status": doi_status,
        "num_failures": failures,
        "num_warnings": warnings,
        "bundle_manifest_path": str(bundle_manifest_path),
        "bundle_path": str(bundle_path),
        "bundle_bytes": actual_bytes,
        "bundle_sha256": actual_sha,
        "tar_member_count": len(tar_names),
        "upload_manifest_rows": len(upload_rows),
        "critical_artifacts_checked": len(manifest.get("critical_artifacts", [])),
        "upload_manifest_missing": upload_missing,
        "upload_manifest_hash_mismatch": upload_hash_mismatch,
        "critical_artifacts_missing": critical_missing,
        "critical_artifact_hash_mismatch": critical_hash_mismatch,
        "checks": checks,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if markdown_path:
        markdown_path = Path(markdown_path)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown_report(result, markdown_path)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify a WaveST-Gate release bundle against its manifests.")
    parser.add_argument("--bundle-manifest", default="results/nature_release/release_bundle_manifest.json")
    parser.add_argument("--deposition-result", default="results/nature_release/zenodo_deposition_result.json")
    parser.add_argument("--output", default="results/nature_release/release_verification.json")
    parser.add_argument("--markdown-output", default="results/nature_release/release_verification.md")
    parser.add_argument("--require-doi", action="store_true", help="Fail verification unless a Zenodo DOI/deposition id are recorded.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = verify_release_bundle(
        bundle_manifest_path=args.bundle_manifest,
        deposition_result_path=args.deposition_result,
        output_path=args.output,
        markdown_path=args.markdown_output,
        require_doi=args.require_doi,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
