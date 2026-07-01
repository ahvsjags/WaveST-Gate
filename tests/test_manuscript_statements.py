import json
from pathlib import Path

from wavestgate.evaluation.manuscript_statements import build_manuscript_statements


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_manuscript_statements_separates_upstream_and_project_doi(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark" / "xenium_visium_benchmark_manifest.json"
    _write_json(
        benchmark,
        {
            "num_spots": 10,
            "num_spots_with_ground_truth": 6,
            "num_cells": 100,
            "num_cell_types": 3,
            "spot_radius": 55,
        },
    )
    data_manifest_dir = tmp_path / "data_manifest"
    data_manifest_dir.mkdir()
    (data_manifest_dir / "external.yaml").write_text(
        "\n".join(
            [
                "dataset_id: external",
                "sources:",
                "  - name: upstream",
                "    doi: 10.5281/zenodo.4739739",
                "items:",
                "  - id: counts",
                "    required: true",
            ]
        ),
        encoding="utf-8",
    )
    release_dir = tmp_path / "release"
    _write_json(
        release_dir / "release_bundle_manifest.json",
        {"release_status": "zenodo_ready_not_deposited", "doi": "", "bundle_path": "bundle.tar.gz"},
    )
    _write_json(
        release_dir / "release_verification.json",
        {"doi_status": "pending_token", "bundle_integrity_status": "passed"},
    )
    _write_json(
        release_dir / "zenodo_deposition_result.json",
        {"release_status": "dry_run_token_missing", "doi": "", "zenodo_deposition_id": ""},
    )
    _write_json(
        release_dir / "environment_report.json",
        {
            "python": {"version": "3.12.0"},
            "torch": {"version": "2.6.0+cu124", "cuda_available": True, "cuda_version": "12.4", "device_count": 1},
            "r": {"rscript_version": "R scripting front-end version 4.1.2"},
        },
    )
    readiness_dir = tmp_path / "readiness"
    _write_json(readiness_dir / "readiness_report.json", {"overall_status": "partial"})

    payload = build_manuscript_statements(
        output_dir=tmp_path / "statements",
        benchmark_manifest=benchmark,
        data_manifest_dir=data_manifest_dir,
        release_dir=release_dir,
        readiness_dir=readiness_dir,
        environment_report_path=release_dir / "environment_report.json",
    )

    assert payload["release"]["doi"] == ""
    assert payload["release"]["doi_status"] == "pending_token"
    assert payload["data_manifests"][0]["upstream_dois"] == ["10.5281/zenodo.4739739"]
    assert Path(payload["outputs"]["json"]).exists()
    assert Path(payload["outputs"]["markdown"]).exists()
    markdown = Path(payload["outputs"]["markdown"]).read_text(encoding="utf-8")
    assert "project release DOI is pending" in markdown
    assert "10.5281/zenodo.4739739" in markdown
