"""Create a Zenodo deposition for a prepared WaveST-Gate release bundle."""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PRODUCTION_API = "https://zenodo.org"
SANDBOX_API = "https://sandbox.zenodo.org"


JsonRequest = Callable[[str, str, str, dict[str, Any] | None], dict[str, Any]]
FileUpload = Callable[[str, str, Path], dict[str, Any]]


def _with_token(url: str, token: str) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urllib.parse.urlencode({'access_token': token})}"


def _request_json(method: str, url: str, token: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(_with_token(url, token), data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Zenodo API {method} {url} failed with HTTP {exc.code}: {detail}") from exc
    return json.loads(body) if body else {}


def _upload_file(bucket_url: str, token: str, file_path: Path) -> dict[str, Any]:
    upload_url = f"{bucket_url.rstrip('/')}/{urllib.parse.quote(file_path.name)}"
    request = urllib.request.Request(
        _with_token(upload_url, token),
        data=file_path.read_bytes(),
        headers={"Content-Type": "application/octet-stream", "Accept": "application/json"},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Zenodo file upload failed with HTTP {exc.code}: {detail}") from exc
    return json.loads(body) if body else {}


def _load_metadata(metadata_path: Path, reserve_doi: bool) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if reserve_doi:
        metadata.setdefault("metadata", {})["prereserve_doi"] = True
    return metadata


def _extract_doi(response: dict[str, Any]) -> str:
    metadata = response.get("metadata", {}) if isinstance(response, dict) else {}
    for value in [
        response.get("doi") if isinstance(response, dict) else None,
        metadata.get("doi"),
        metadata.get("prereserve_doi", {}).get("doi") if isinstance(metadata.get("prereserve_doi"), dict) else None,
    ]:
        if value:
            return str(value)
    return ""


def _write_result(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _update_bundle_manifest(bundle_manifest_path: Path, deposition_result: dict[str, Any]) -> None:
    if not bundle_manifest_path.exists():
        return
    manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    for key in [
        "release_status",
        "doi",
        "zenodo_deposition_id",
        "zenodo_record_url",
        "zenodo_bucket_url",
        "zenodo_deposition_result",
    ]:
        value = deposition_result.get(key)
        if value is not None:
            manifest[key] = value
    bundle_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def deposit_release_bundle(
    bundle_manifest_path: str | Path = "results/nature_release/release_bundle_manifest.json",
    metadata_path: str | Path | None = None,
    bundle_path: str | Path | None = None,
    output_path: str | Path | None = None,
    token: str | None = None,
    sandbox: bool = False,
    publish: bool = False,
    reserve_doi: bool = True,
    dry_run: bool = False,
    deposition_id: str | None = None,
    request_json: JsonRequest = _request_json,
    upload_file: FileUpload = _upload_file,
) -> dict[str, Any]:
    """Create/update a Zenodo draft, upload the bundle, and optionally publish it."""

    bundle_manifest_path = Path(bundle_manifest_path)
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    metadata_path = Path(metadata_path or bundle_manifest["zenodo_metadata"])
    bundle_path = Path(bundle_path or bundle_manifest["bundle_path"])
    output_path = Path(output_path or bundle_manifest_path.with_name("zenodo_deposition_result.json"))
    token = token or os.environ.get("ZENODO_ACCESS_TOKEN") or os.environ.get("ZENODO_TOKEN")
    api_base = SANDBOX_API if sandbox else PRODUCTION_API
    metadata = _load_metadata(metadata_path, reserve_doi=reserve_doi)

    base_result: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_base": api_base,
        "sandbox": bool(sandbox),
        "publish_requested": bool(publish),
        "reserve_doi_requested": bool(reserve_doi),
        "bundle_manifest_path": str(bundle_manifest_path),
        "metadata_path": str(metadata_path),
        "bundle_path": str(bundle_path),
        "bundle_bytes": int(bundle_path.stat().st_size) if bundle_path.exists() else 0,
        "bundle_sha256": bundle_manifest.get("bundle_sha256", ""),
        "zenodo_deposition_result": str(output_path),
    }

    if dry_run or not token:
        result = {
            **base_result,
            "release_status": "dry_run_token_missing" if not token else "dry_run_not_deposited",
            "doi": "",
            "zenodo_deposition_id": str(deposition_id or ""),
            "zenodo_record_url": "",
            "zenodo_bucket_url": "",
            "metadata_preview": metadata,
            "notes": "No network deposition was performed. Provide ZENODO_ACCESS_TOKEN or --token to create a real draft.",
        }
        _write_result(output_path, result)
        return result

    depositions_url = f"{api_base}/api/deposit/depositions"
    if deposition_id:
        deposition = request_json("GET", f"{depositions_url}/{deposition_id}", token, None)
    else:
        deposition = request_json("POST", depositions_url, token, {})
        deposition_id = str(deposition.get("id"))
    if not deposition_id:
        raise RuntimeError("Zenodo did not return a deposition id.")

    updated = request_json("PUT", f"{depositions_url}/{deposition_id}", token, metadata)
    bucket_url = str(updated.get("links", {}).get("bucket") or deposition.get("links", {}).get("bucket") or "")
    if not bucket_url:
        raise RuntimeError("Zenodo deposition response did not include a bucket URL.")
    upload_response = upload_file(bucket_url, token, bundle_path)

    final_response = updated
    release_status = "zenodo_draft_reserved"
    if publish:
        final_response = request_json("POST", f"{depositions_url}/{deposition_id}/actions/publish", token, {})
        release_status = "zenodo_published"

    doi = _extract_doi(final_response) or _extract_doi(updated)
    result = {
        **base_result,
        "release_status": release_status,
        "doi": doi,
        "zenodo_deposition_id": str(deposition_id),
        "zenodo_record_url": str(final_response.get("links", {}).get("html") or updated.get("links", {}).get("html") or ""),
        "zenodo_bucket_url": bucket_url,
        "upload_response": upload_response,
        "deposition_response": final_response,
    }
    _write_result(output_path, result)
    _update_bundle_manifest(bundle_manifest_path, result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create a Zenodo deposition for the prepared WaveST-Gate release bundle.")
    parser.add_argument("--bundle-manifest", default="results/nature_release/release_bundle_manifest.json")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--no-reserve-doi", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--deposition-id", default=None)
    args = parser.parse_args(argv)
    result = deposit_release_bundle(
        bundle_manifest_path=args.bundle_manifest,
        metadata_path=args.metadata,
        bundle_path=args.bundle,
        output_path=args.output,
        token=args.token,
        sandbox=args.sandbox,
        publish=args.publish,
        reserve_doi=not args.no_reserve_doi,
        dry_run=args.dry_run,
        deposition_id=args.deposition_id,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
