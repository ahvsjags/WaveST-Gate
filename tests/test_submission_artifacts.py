import json
import tarfile
from pathlib import Path

import pandas as pd
import torch
from scipy import sparse
import h5py

from wavestgate.data.assemble import BatchMetadata
from wavestgate.data.prepare_dataset import save_prepared_dataset
from wavestgate.data.prepare_xenium_pseudospots import prepare_xenium_pseudospots
from wavestgate.evaluation.baseline_statistics import collect_baseline_statistics
from wavestgate.evaluation.external_pathology_validation import validate_external_pathology
from wavestgate.evaluation.matched_multisample_baselines import collect_matched_multisample_baselines
from wavestgate.evaluation.prepare_release import prepare_release_bundle
from wavestgate.evaluation.verify_release import verify_release_bundle
from wavestgate.evaluation.zenodo_deposit import deposit_release_bundle
from wavestgate.models.types import WaveSTGateBatch


def test_collect_baseline_statistics(tmp_path):
    prepared_path = tmp_path / "prepared.pt"
    spot_ids = ["s1", "s2", "s3", "s4"]
    cell_types = ["A", "B"]
    batch = WaveSTGateBatch(
        image_patches=torch.zeros(4, 3, 8, 8),
        st_expression=torch.ones(4, 3),
        reference_prototypes=torch.ones(2, 3),
        coords=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        proportion_gt=torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.0, 0.0]], dtype=torch.float32),
    )
    metadata = BatchMetadata(spot_ids=spot_ids, gene_names=["G1", "G2", "G3"], cell_types=cell_types)
    save_prepared_dataset(prepared_path, batch, metadata)

    model_pred = tmp_path / "model.csv"
    baseline_pred = tmp_path / "baseline.csv"
    pd.DataFrame([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.5, 0.5]], index=spot_ids, columns=cell_types).to_csv(
        model_pred,
        index_label="spot_id",
    )
    pd.DataFrame([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], index=spot_ids, columns=cell_types).to_csv(
        baseline_pred,
        index_label="spot_id",
    )
    comparison = tmp_path / "comparison.csv"
    pd.DataFrame(
        [
            {"method": "WaveST-Gate", "predictions_path": str(model_pred)},
            {"method": "uniform", "predictions_path": str(baseline_pred)},
        ]
    ).to_csv(comparison, index=False)
    splits = tmp_path / "splits.csv"
    pd.DataFrame({"spot_id": spot_ids, "split": ["train", "test", "test", "train"]}).to_csv(splits, index=False)

    outputs = collect_baseline_statistics(
        prepared_path=prepared_path,
        comparison_path=comparison,
        output_dir=tmp_path / "stats",
        split_path=splits,
        n_bootstraps=5,
        seed=7,
    )
    summary = pd.read_csv(outputs["summary_path"])
    assert summary.iloc[0]["method"] == "WaveST-Gate"
    paired = pd.read_csv(outputs["paired_improvement_path"])
    assert paired.iloc[0]["method"] == "uniform"
    assert Path(outputs["manifest_path"]).exists()


def test_collect_matched_multisample_baselines(tmp_path):
    rep2 = tmp_path / "rep2.csv"
    rep1 = tmp_path / "rep1.csv"
    rows = [
        {"method": "WaveST-Gate", "jsd": 0.1, "spotwise_cosine": 0.9, "runtime_seconds": 10},
        {"method": "RCTD (multi)", "jsd": 0.3, "spotwise_cosine": 0.7, "runtime_seconds": 20},
        {"method": "Tangram", "jsd": 0.4, "spotwise_cosine": 0.6, "runtime_seconds": 30},
    ]
    pd.DataFrame(rows).to_csv(rep2, index=False)
    pd.DataFrame([{**row, "jsd": row["jsd"] + 0.1} for row in rows]).to_csv(rep1, index=False)

    outputs = collect_matched_multisample_baselines(
        comparisons=[("rep2", rep2), ("rep1", rep1)],
        output_dir=tmp_path / "multi",
    )
    summary = pd.read_csv(outputs["summary_path"])
    assert summary.iloc[0]["method"] == "WaveST-Gate"
    assert summary.iloc[0]["num_datasets"] == 2
    paired = pd.read_csv(outputs["paired_improvement_path"])
    assert set(paired["method"]) == {"RCTD (multi)", "Tangram"}
    manifest = json.loads(Path(outputs["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["num_datasets"] == 2
    assert "WaveST-Gate" in manifest["methods_in_all_required_datasets"]


def test_prepare_release_bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("hello", encoding="utf-8")
    outputs = prepare_release_bundle(
        output_dir=tmp_path / "release",
        version="0.0.test",
        include_paths=[source / "README.md"],
    )
    manifest = json.loads(Path(outputs["bundle_manifest"]).read_text(encoding="utf-8"))
    assert manifest["release_status"] == "zenodo_ready_not_deposited"
    assert manifest["num_files"] == 1
    assert Path(outputs["bundle_path"]).exists()
    assert Path(outputs["zenodo_metadata"]).exists()


def test_prepare_release_bundle_includes_explicit_checkpoint_but_skips_directory_pt(tmp_path):
    source = tmp_path / "source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    checkpoint = source / "checkpoint.pt"
    checkpoint.write_bytes(b"model")
    (nested / "prepared.pt").write_bytes(b"prepared")
    (nested / "table.csv").write_text("a\n1\n", encoding="utf-8")

    outputs = prepare_release_bundle(
        output_dir=tmp_path / "release",
        version="0.0.test",
        include_paths=[checkpoint, nested],
    )

    with tarfile.open(outputs["bundle_path"], "r:gz") as archive:
        names = set(archive.getnames())
    assert str(checkpoint).lstrip("/") in names
    assert str(nested / "table.csv").lstrip("/") in names
    assert str(nested / "prepared.pt").lstrip("/") not in names


def test_zenodo_deposit_dry_run_without_token(tmp_path, monkeypatch):
    monkeypatch.delenv("ZENODO_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("ZENODO_TOKEN", raising=False)
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("hello", encoding="utf-8")
    outputs = prepare_release_bundle(
        output_dir=tmp_path / "release",
        version="0.0.test",
        include_paths=[source / "README.md"],
    )

    result = deposit_release_bundle(
        bundle_manifest_path=outputs["bundle_manifest"],
        output_path=tmp_path / "release" / "zenodo_deposition_result.json",
        dry_run=True,
    )

    assert result["release_status"] == "dry_run_token_missing"
    assert result["doi"] == ""
    assert result["metadata_preview"]["metadata"]["prereserve_doi"] is True
    assert Path(result["zenodo_deposition_result"]).exists()


def test_verify_release_bundle_checks_tar_and_critical_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv("ZENODO_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("ZENODO_TOKEN", raising=False)
    source = tmp_path / "source"
    source.mkdir()
    readme = source / "README.md"
    checkpoint = source / "checkpoint.pt"
    readme.write_text("hello", encoding="utf-8")
    checkpoint.write_bytes(b"model")
    outputs = prepare_release_bundle(
        output_dir=tmp_path / "release",
        version="0.0.test",
        include_paths=[readme, checkpoint],
        critical_artifact_paths=[checkpoint],
    )
    deposition = deposit_release_bundle(
        bundle_manifest_path=outputs["bundle_manifest"],
        output_path=tmp_path / "release" / "zenodo_deposition_result.json",
        dry_run=True,
    )

    result = verify_release_bundle(
        bundle_manifest_path=outputs["bundle_manifest"],
        deposition_result_path=deposition["zenodo_deposition_result"],
        output_path=tmp_path / "release" / "release_verification.json",
        markdown_path=tmp_path / "release" / "release_verification.md",
    )

    assert result["overall_status"] == "passed_with_warnings"
    assert result["bundle_integrity_status"] == "passed"
    assert result["doi_status"] == "pending_token"
    assert result["critical_artifacts_missing"] == []
    assert result["critical_artifact_hash_mismatch"] == []
    assert Path(tmp_path / "release" / "release_verification.json").exists()
    assert Path(tmp_path / "release" / "release_verification.md").exists()


def test_zenodo_deposit_updates_manifest_with_reserved_doi(tmp_path, monkeypatch):
    monkeypatch.delenv("ZENODO_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("ZENODO_TOKEN", raising=False)
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("hello", encoding="utf-8")
    outputs = prepare_release_bundle(
        output_dir=tmp_path / "release",
        version="0.0.test",
        include_paths=[source / "README.md"],
    )
    calls = []

    def fake_request_json(method, url, token, payload):
        calls.append((method, url, payload))
        if method == "POST" and url.endswith("/api/deposit/depositions"):
            return {"id": 123, "links": {"bucket": "https://zenodo.example/bucket/123", "html": "https://zenodo.example/deposit/123"}}
        if method == "PUT" and url.endswith("/api/deposit/depositions/123"):
            assert payload["metadata"]["prereserve_doi"] is True
            return {
                "id": 123,
                "links": {"bucket": "https://zenodo.example/bucket/123", "html": "https://zenodo.example/deposit/123"},
                "metadata": {"prereserve_doi": {"doi": "10.5072/zenodo.12345"}},
            }
        raise AssertionError(f"unexpected request {method} {url}")

    def fake_upload_file(bucket_url, token, file_path):
        assert bucket_url == "https://zenodo.example/bucket/123"
        assert token == "token"
        assert file_path.exists()
        return {"filename": file_path.name, "checksum": "md5:abc"}

    result = deposit_release_bundle(
        bundle_manifest_path=outputs["bundle_manifest"],
        token="token",
        sandbox=True,
        request_json=fake_request_json,
        upload_file=fake_upload_file,
    )

    assert result["release_status"] == "zenodo_draft_reserved"
    assert result["doi"] == "10.5072/zenodo.12345"
    assert result["zenodo_deposition_id"] == "123"
    assert [call[0] for call in calls] == ["POST", "PUT"]
    manifest = json.loads(Path(outputs["bundle_manifest"]).read_text(encoding="utf-8"))
    assert manifest["release_status"] == "zenodo_draft_reserved"
    assert manifest["doi"] == "10.5072/zenodo.12345"
    assert manifest["zenodo_deposition_id"] == "123"


def test_validate_external_pathology(tmp_path):
    processed = tmp_path / "processed"
    results = tmp_path / "external"
    dataset = "wu_swarbrick_demo"
    prepared_dir = processed / dataset
    result_dir = results / dataset
    spot_ids = ["s1", "s2", "s3"]
    cell_types = ["Invasive Tumor", "Stromal", "B Cells"]
    batch = WaveSTGateBatch(
        image_patches=torch.zeros(3, 3, 8, 8),
        st_expression=torch.ones(3, 2),
        reference_prototypes=torch.ones(3, 2),
        coords=torch.arange(6, dtype=torch.float32).reshape(3, 2),
    )
    metadata = BatchMetadata(spot_ids=spot_ids, gene_names=["G1", "G2"], cell_types=cell_types)
    save_prepared_dataset(prepared_dir / "prepared.pt", batch, metadata)
    pd.DataFrame(
        {
            "spot_id": spot_ids,
            "patientid": ["p1", "p1", "p1"],
            "subtype": ["TNBC", "TNBC", "TNBC"],
            "Classification": ["Invasive cancer", "Stroma", "Lymphocytes"],
        }
    ).to_csv(prepared_dir / "spot_metadata.csv", index=False)
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]],
        index=spot_ids,
        columns=cell_types,
    ).to_csv(result_dir / "predicted_proportions.csv", index_label="spot_id")
    pd.DataFrame([[0.1, 0.8, 0.1]] * 3, index=spot_ids, columns=["image", "expression", "reference"]).to_csv(
        result_dir / "gate_weights.csv",
        index_label="spot_id",
    )
    pd.DataFrame({"spot_uncertainty": [0.1, 0.2, 0.3]}, index=spot_ids).to_csv(result_dir / "spot_uncertainty.csv", index_label="spot_id")
    pd.DataFrame([[0.7, 0.2, 0.1]] * 3, index=spot_ids, columns=cell_types).to_csv(result_dir / "agent_attention.csv", index_label="spot_id")

    outputs = validate_external_pathology(
        processed_root=processed,
        external_results_dir=results,
        output_dir=tmp_path / "pathology",
        dataset_prefix="wu_swarbrick",
        n_niches=2,
    )
    manifest = json.loads(Path(outputs["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["num_datasets"] == 1
    assert manifest["num_spots"] == 3
    summary = pd.read_csv(outputs["class_summary_path"])
    assert set(summary["classification"]) == {"Invasive cancer", "Stroma", "Lymphocytes"}


def test_prepare_xenium_pseudospots(tmp_path):
    matrix_path = tmp_path / "cell_feature_matrix.h5"
    values = sparse.csc_matrix(
        [
            [5, 4, 0, 0],
            [0, 0, 3, 4],
            [1, 1, 1, 1],
        ],
        dtype="int32",
    )
    with h5py.File(matrix_path, "w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("data", data=values.data)
        group.create_dataset("indices", data=values.indices)
        group.create_dataset("indptr", data=values.indptr)
        group.create_dataset("shape", data=values.shape)
        group.create_dataset("barcodes", data=[b"1", b"2", b"3", b"4"])
        features = group.create_group("features")
        features.create_dataset("name", data=[b"G1", b"G2", b"G3"])
        features.create_dataset("id", data=[b"G1", b"G2", b"G3"])
        features.create_dataset("feature_type", data=[b"Gene Expression", b"Gene Expression", b"Gene Expression"])

    typed_cells = tmp_path / "typed_cells.csv"
    pd.DataFrame(
        {
            "cell_id": ["1", "2", "3", "4"],
            "x": [4.0, 5.0, 10.0, 11.0],
            "y": [4.0, 5.0, 10.0, 11.0],
            "cell_type": ["A", "A", "B", "B"],
        }
    ).to_csv(typed_cells, index=False)
    reference = tmp_path / "reference.csv"
    pd.DataFrame({"cell_type": ["A", "B"], "G1": [4.5, 0.0], "G2": [0.0, 3.5], "G3": [1.0, 1.0]}).to_csv(reference, index=False)
    image = tmp_path / "he.png"
    from PIL import Image

    Image.new("RGB", (24, 24), (180, 120, 160)).save(image)

    manifest = prepare_xenium_pseudospots(
        typed_cells_path=typed_cells,
        cell_feature_matrix_path=matrix_path,
        he_image_path=image,
        reference_prototypes_path=reference,
        output_dir=tmp_path / "pseudo",
        radius=3.0,
        stride=6.0,
        min_cells=1,
        patch_size=8,
        graph_k=1,
    )
    assert manifest["num_spots"] >= 2
    assert manifest["num_genes"] == 3
    assert (tmp_path / "pseudo" / "prepared.pt").exists()
    assert (tmp_path / "pseudo" / "xenium_cell_proportions.csv").exists()
