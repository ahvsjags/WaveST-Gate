"""Build manuscript availability and reproducibility statements.

The generated text is intentionally evidence-linked. It can be used in a
manuscript draft, while the JSON companion preserves the machine-readable
artifact paths and release DOI status.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from wavestgate.evaluation.submission_readiness import DOI_PATTERN, _release_deposition_summary


DEFAULT_OUTPUT_DIR = Path("results/nature_manuscript_statements")


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _evidence(path: str | Path, role: str) -> dict[str, Any]:
    path = Path(path)
    return {"role": role, "path": str(path), "exists": path.exists()}


def _data_manifest_summary(data_manifest_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(data_manifest_dir.glob("*.yaml")):
        payload = _read_yaml(path)
        items = payload.get("items") if isinstance(payload.get("items"), list) else []
        sources = payload.get("sources") if isinstance(payload.get("sources"), list) else []
        upstream_dois = []
        for source in sources:
            if not isinstance(source, dict):
                continue
            doi = str(source.get("doi", "")).strip()
            if doi and DOI_PATTERN.search(doi):
                upstream_dois.append(doi)
        summaries.append(
            {
                "path": str(path),
                "dataset_id": payload.get("dataset_id", path.stem),
                "description": str(payload.get("description", "")).strip(),
                "num_items": len(items),
                "num_required_items": sum(1 for item in items if isinstance(item, dict) and bool(item.get("required"))),
                "upstream_dois": sorted(set(upstream_dois)),
            }
        )
    return summaries


def _release_status(release_dir: Path, statements_dir: Path) -> dict[str, Any]:
    deposition = _release_deposition_summary(release_dir)
    release_manifest = _read_json(release_dir / "release_bundle_manifest.json")
    verification = _read_json(release_dir / "release_verification.json")
    doi = str(deposition.get("doi", "")).strip()
    return {
        "status": deposition.get("release_status", release_manifest.get("release_status", "")),
        "doi": doi,
        "doi_status": "recorded" if doi else str(verification.get("doi_status", "pending_token") or "pending_token"),
        "deposition_id": deposition.get("zenodo_deposition_id", ""),
        "record_url": deposition.get("zenodo_record_url", ""),
        "bundle_manifest": str(release_dir / "release_bundle_manifest.json"),
        "deposition_result": str(release_dir / "zenodo_deposition_result.json"),
        "release_verification": str(release_dir / "release_verification.json"),
        "statements_json": str(statements_dir / "manuscript_availability_statements.json"),
        "statements_markdown": str(statements_dir / "manuscript_availability_statements.md"),
    }


def _environment_summary(environment: dict[str, Any]) -> dict[str, Any]:
    torch = environment.get("torch") if isinstance(environment.get("torch"), dict) else {}
    python = environment.get("python") if isinstance(environment.get("python"), dict) else {}
    r = environment.get("r") if isinstance(environment.get("r"), dict) else {}
    devices = torch.get("devices") if isinstance(torch.get("devices"), list) else []
    device_names = [str(device.get("name", "")) for device in devices if isinstance(device, dict) and device.get("name")]
    return {
        "python": str(python.get("version", "")).split(" ")[0],
        "torch": torch.get("version", ""),
        "cuda_available": torch.get("cuda_available", ""),
        "cuda_version": torch.get("cuda_version", ""),
        "gpu_count": torch.get("device_count", ""),
        "gpu_names": device_names,
        "rscript_version": r.get("rscript_version", ""),
    }


def _benchmark_text(benchmark: dict[str, Any]) -> str:
    return (
        f"The Xenium-to-Visium benchmark contains {benchmark.get('num_spots', 'NA')} Visium spots, "
        f"{benchmark.get('num_spots_with_ground_truth', 'NA')} spots with Xenium-derived ground truth, "
        f"{benchmark.get('num_cells', 'NA')} mapped Xenium cells, {benchmark.get('num_cell_types', 'NA')} cell types, "
        f"and a spot aggregation radius of {benchmark.get('spot_radius', 'NA')}. "
        "The benchmark artifacts include cell counts, proportions, spot-level QC, deterministic splits, "
        "coordinate/radius metadata, and a protocol document."
    )


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# WaveST-Gate Manuscript Availability And Reproducibility Statements",
        "",
        f"Generated UTC: {payload['generated_at_utc']}",
        "",
        "These statements are generated from workspace manifests and should be refreshed after a new release deposition.",
        "",
    ]
    for statement in payload["statements"]:
        lines.extend(
            [
                f"## {statement['title']}",
                "",
                statement["text"],
                "",
                "Evidence:",
            ]
        )
        for evidence in statement["evidence"]:
            exists = "yes" if evidence["exists"] else "no"
            lines.append(f"- {evidence['role']}: `{evidence['path']}` (exists: {exists})")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_manuscript_statements(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    benchmark_manifest: str | Path = "data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/xenium_visium_benchmark_manifest.json",
    data_manifest_dir: str | Path = "data_manifest",
    release_dir: str | Path = "results/nature_release",
    readiness_dir: str | Path = "results/nature_submission_readiness",
    environment_report_path: str | Path = "results/nature_release/environment_report.json",
) -> dict[str, Any]:
    """Generate manuscript statements and their evidence-linked JSON."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_manifest = Path(benchmark_manifest)
    data_manifest_dir = Path(data_manifest_dir)
    release_dir = Path(release_dir)
    readiness_dir = Path(readiness_dir)
    environment_report_path = Path(environment_report_path)

    benchmark = _read_json(benchmark_manifest)
    data_manifests = _data_manifest_summary(data_manifest_dir)
    release = _release_status(release_dir, output_dir)
    environment = _environment_summary(_read_json(environment_report_path))
    readiness = _read_json(readiness_dir / "readiness_report.json")
    upstream_dois = sorted({doi for manifest in data_manifests for doi in manifest["upstream_dois"]})

    if release["doi"]:
        project_release_sentence = f"The WaveST-Gate evidence bundle is deposited on Zenodo with DOI {release['doi']}."
    else:
        project_release_sentence = (
            "The WaveST-Gate evidence bundle is Zenodo-ready, but the project release DOI is pending a token-backed "
            "Zenodo deposition; the current DOI status is recorded as "
            f"{release['doi_status']} in the release verification artifacts."
        )

    upstream_sentence = ""
    if upstream_dois:
        upstream_sentence = " Upstream public datasets with DOIs include: " + ", ".join(upstream_dois) + "."

    data_text = (
        "Raw public data are not redistributed in the evidence bundle. Source URLs, required files, expected sizes, "
        "and licenses/DOIs where available are recorded in the data manifests. "
        + _benchmark_text(benchmark)
        + " "
        + project_release_sentence
        + upstream_sentence
    )
    code_text = (
        "The WaveST-Gate source package, tests, examples, release metadata, MIT license, CITATION.cff, and codemeta.json "
        "are included in the release bundle. The code availability statement should cite the Zenodo release DOI once "
        "a real deposition has been reserved or published."
    )
    reproducibility_text = (
        "The full artifact refresh is driven by `python -m wavestgate.evaluation.finalize_submission`. "
        "This command regenerates the environment report, manuscript tables, manuscript figures, release bundle, "
        "Zenodo dry-run/deposition result, release verification, readiness audit, and final handoff. "
        f"The latest readiness status is {readiness.get('overall_status', 'not_available')}, and the release verifier "
        "records tar integrity, upload-manifest consistency, and critical-artifact coverage."
    )
    compute_text = (
        f"The recorded compute environment uses Python {environment.get('python', '')}, PyTorch {environment.get('torch', '')}, "
        f"CUDA available={environment.get('cuda_available', '')}, CUDA version {environment.get('cuda_version', '')}, "
        f"{environment.get('gpu_count', '')} GPU(s)"
        + (f" ({', '.join(environment['gpu_names'])})" if environment.get("gpu_names") else "")
        + f", and {environment.get('rscript_version', '')} for R-based baselines."
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if (benchmark_manifest.exists() and data_manifests and environment_report_path.exists()) else "partial",
        "benchmark": benchmark,
        "data_manifests": data_manifests,
        "release": release,
        "environment": environment,
        "statements": [
            {
                "title": "Data Availability",
                "text": data_text,
                "evidence": [
                    _evidence(benchmark_manifest, "xenium-to-visium benchmark manifest"),
                    _evidence(data_manifest_dir, "public data source manifests"),
                    _evidence(release_dir / "zenodo_deposition_result.json", "project release DOI status"),
                    _evidence(release_dir / "release_bundle_manifest.json", "release bundle manifest"),
                ],
            },
            {
                "title": "Code Availability",
                "text": code_text,
                "evidence": [
                    _evidence("wavestgate", "source package"),
                    _evidence("tests", "test suite"),
                    _evidence("LICENSE", "software license"),
                    _evidence("CITATION.cff", "citation metadata"),
                    _evidence("codemeta.json", "CodeMeta metadata"),
                ],
            },
            {
                "title": "Reproducibility",
                "text": reproducibility_text,
                "evidence": [
                    _evidence(readiness_dir / "readiness_report.json", "submission readiness audit"),
                    _evidence(release_dir / "release_verification.json", "release integrity verification"),
                    _evidence(release_dir / "final_submission_handoff.json", "final handoff manifest"),
                ],
            },
            {
                "title": "Computing Environment",
                "text": compute_text,
                "evidence": [
                    _evidence(environment_report_path, "software and hardware environment report"),
                    _evidence(release_dir / "environment_report.md", "environment report summary"),
                ],
            },
        ],
    }
    json_path = output_dir / "manuscript_availability_statements.json"
    markdown_path = output_dir / "manuscript_availability_statements.md"
    payload["outputs"] = {"json": str(json_path), "markdown": str(markdown_path)}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(payload, markdown_path)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build WaveST-Gate manuscript availability statements.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--benchmark-manifest", type=Path, default=Path("data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/xenium_visium_benchmark_manifest.json"))
    parser.add_argument("--data-manifest-dir", type=Path, default=Path("data_manifest"))
    parser.add_argument("--release-dir", type=Path, default=Path("results/nature_release"))
    parser.add_argument("--readiness-dir", type=Path, default=Path("results/nature_submission_readiness"))
    parser.add_argument("--environment-report-path", type=Path, default=Path("results/nature_release/environment_report.json"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    payload = build_manuscript_statements(
        output_dir=args.output_dir,
        benchmark_manifest=args.benchmark_manifest,
        data_manifest_dir=args.data_manifest_dir,
        release_dir=args.release_dir,
        readiness_dir=args.readiness_dir,
        environment_report_path=args.environment_report_path,
    )
    print(json.dumps(payload["outputs"], indent=2), flush=True)


if __name__ == "__main__":
    main()
