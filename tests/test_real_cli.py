import json
import gzip
import tarfile

import h5py
import numpy as np
import pandas as pd
import torch
from scipy import io as scipy_io
from scipy import sparse
from PIL import Image

from wavestgate.data.assemble import BatchMetadata
from wavestgate.data.prepare_dataset import load_prepared_dataset, prepare_from_config, save_prepared_dataset
from wavestgate.data.prepare_10x_visium import convert_10x_visium
from wavestgate.data.prepare_scrna_reference import convert_scrna_reference
from wavestgate.data.prepare_wu_swarbrick_visium import convert_wu_atlas
from wavestgate.data.download import verify_manifest
from wavestgate.data.preprocess_scrna import load_reference_prototypes_table
from wavestgate.data.preprocess_st import load_spot_expression
from wavestgate.evaluation.evaluate_real import evaluate_real
from wavestgate.evaluation.nature_analysis import run_nature_analysis
from wavestgate.evaluation.prepare_baseline_inputs import create_baseline_bundle
from wavestgate.evaluation.run_baselines import collect_baseline_comparison
from wavestgate.evaluation.xenium_visium_benchmark import build_xenium_visium_benchmark
from wavestgate.models.types import WaveSTGateBatch
from wavestgate.training.predict_real import predict_real
from wavestgate.training.train_real import train_real_from_config


def _write_real_style_inputs(tmp_path):
    spot_expression = tmp_path / "spot_expression.csv"
    spot_coords = tmp_path / "spot_coords.csv"
    scrna = tmp_path / "scrna.csv"
    xenium = tmp_path / "xenium_cells.csv"
    he_image = tmp_path / "he.png"

    pd.DataFrame(
        {
            "spot_id": ["s1", "s2", "s3"],
            "G1": [2.0, 0.2, 0.5],
            "G2": [0.1, 3.0, 0.5],
            "G3": [1.0, 0.5, 2.0],
            "G4": [0.0, 1.0, 2.5],
        }
    ).to_csv(spot_expression, index=False)
    pd.DataFrame(
        {
            "spot_id": ["s1", "s2", "s3"],
            "x": [8.0, 20.0, 32.0],
            "y": [8.0, 20.0, 32.0],
        }
    ).to_csv(spot_coords, index=False)
    pd.DataFrame(
        {
            "cell_id": ["c1", "c2", "c3", "c4"],
            "cell_type": ["B", "B", "T", "T"],
            "G1": [3.0, 2.0, 0.1, 0.2],
            "G2": [0.1, 0.2, 3.0, 2.5],
            "G3": [1.0, 1.0, 1.0, 1.0],
            "G4": [0.0, 0.1, 1.0, 1.2],
        }
    ).to_csv(scrna, index=False)
    pd.DataFrame(
        {
            "cell_type": ["B", "B", "T", "T"],
            "x": [8.0, 9.0, 20.0, 32.0],
            "y": [8.0, 9.0, 20.0, 32.0],
        }
    ).to_csv(xenium, index=False)
    Image.new("RGB", (48, 48), (130, 80, 120)).save(he_image)
    return spot_expression, spot_coords, scrna, xenium, he_image


def test_prepare_dataset_and_train_real(tmp_path):
    spot_expression, spot_coords, scrna, xenium, he_image = _write_real_style_inputs(tmp_path)
    prepared_path = tmp_path / "prepared.pt"
    counts_path = tmp_path / "counts.csv"
    proportions_path = tmp_path / "xenium_props.csv"
    niche_labels_path = tmp_path / "niche_labels.csv"
    pd.DataFrame({"spot_id": ["s1", "s2", "s3"], "niche": ["tumor", "immune", "immune"]}).to_csv(
        niche_labels_path,
        index=False,
    )
    prepare_config = {
        "data": {
            "spot_expression_path": str(spot_expression),
            "spot_coords_path": str(spot_coords),
            "he_image_path": str(he_image),
            "scrna_expression_path": str(scrna),
            "xenium_cells_path": str(xenium),
            "niche_labels_path": str(niche_labels_path),
            "output_path": str(prepared_path),
            "xenium_counts_output_path": str(counts_path),
            "xenium_proportions_output_path": str(proportions_path),
        },
        "patches": {"patch_size": 24},
        "graph": {"k": 1},
        "xenium": {"spot_radius": 3.0},
    }
    prepare_config_path = tmp_path / "prepare.yaml"
    prepare_config_path.write_text(json.dumps(prepare_config), encoding="utf-8")

    batch, metadata, extra = prepare_from_config(prepare_config_path)
    assert prepared_path.exists()
    assert counts_path.exists()
    assert proportions_path.exists()
    assert batch.image_patches.shape == (3, 3, 24, 24)
    assert batch.proportion_gt is not None
    assert batch.niche_gt is not None
    assert metadata.cell_types == ["B", "T"]
    assert extra["has_xenium_ground_truth"] is True
    assert extra["has_niche_labels"] is True

    loaded_batch, loaded_metadata, _ = load_prepared_dataset(prepared_path)
    assert loaded_batch.st_expression.shape == (3, 4)
    assert loaded_batch.niche_gt.tolist() == [1, 0, 0]
    assert loaded_metadata.spot_ids == ["s1", "s2", "s3"]

    train_config = {
        "data": {"prepared_path": str(prepared_path)},
        "model": {
            "latent_dim": 16,
            "hidden_dim": 32,
            "patch_size": 24,
            "loss_expr_weight": 1.0,
            "loss_prop_weight": 0.5,
            "loss_sparse_weight": 0.01,
            "loss_spatial_weight": 0.01,
            "loss_contrast_weight": 0.01,
            "loss_uncertainty_weight": 0.01,
            "loss_boundary_weight": 0.01,
            "loss_niche_weight": 0.01,
        },
        "training": {
            "steps": 2,
            "learning_rate": 0.001,
            "seed": 5,
            "device": "cpu",
            "checkpoint_path": str(tmp_path / "real_checkpoint.pt"),
            "metrics_path": str(tmp_path / "real_metrics.csv"),
        },
        "predictions": {
            "proportions_path": str(tmp_path / "predicted_proportions.csv"),
            "gates_path": str(tmp_path / "gate_weights.csv"),
            "uncertainty_path": str(tmp_path / "spot_uncertainty.csv"),
            "modality_reliability_path": str(tmp_path / "modality_reliability.csv"),
            "niche_logits_path": str(tmp_path / "niche_logits.csv"),
            "agent_attention_path": str(tmp_path / "agent_attention.csv"),
        },
    }
    train_config_path = tmp_path / "train_real.yaml"
    train_config_path.write_text(json.dumps(train_config), encoding="utf-8")
    metrics = train_real_from_config(train_config_path)

    assert (tmp_path / "real_checkpoint.pt").exists()
    assert (tmp_path / "real_metrics.csv").exists()
    assert (tmp_path / "predicted_proportions.csv").exists()
    assert (tmp_path / "gate_weights.csv").exists()
    assert (tmp_path / "spot_uncertainty.csv").exists()
    assert (tmp_path / "modality_reliability.csv").exists()
    assert (tmp_path / "niche_logits.csv").exists()
    assert (tmp_path / "agent_attention.csv").exists()
    assert "expression_log1p_rmse" in metrics
    assert "spotwise_cosine" in metrics
    assert "uncertainty_error_pearson" in metrics
    assert "niche_accuracy" in metrics

    predict_paths = predict_real(
        checkpoint_path=tmp_path / "real_checkpoint.pt",
        prepared_path=prepared_path,
        proportions_path=tmp_path / "predict_cli_proportions.csv",
        gates_path=tmp_path / "predict_cli_gates.csv",
        uncertainty_path=tmp_path / "predict_cli_uncertainty.csv",
        modality_reliability_path=tmp_path / "predict_cli_modality_reliability.csv",
        reconstructed_expression_path=tmp_path / "predict_cli_expression.csv",
        niche_logits_path=tmp_path / "predict_cli_niche_logits.csv",
    )
    assert "proportions_path" in predict_paths
    assert (tmp_path / "predict_cli_proportions.csv").exists()
    assert (tmp_path / "predict_cli_gates.csv").exists()
    assert (tmp_path / "predict_cli_uncertainty.csv").exists()
    assert (tmp_path / "predict_cli_modality_reliability.csv").exists()
    assert (tmp_path / "predict_cli_expression.csv").exists()
    assert (tmp_path / "predict_cli_niche_logits.csv").exists()

    eval_metrics = evaluate_real(
        predictions_path=tmp_path / "predict_cli_proportions.csv",
        prepared_path=prepared_path,
        output_metrics_path=tmp_path / "evaluation_metrics.csv",
        uncertainty_path=tmp_path / "predict_cli_uncertainty.csv",
    )
    assert (tmp_path / "evaluation_metrics.csv").exists()
    assert "spotwise_cosine" in eval_metrics
    assert "uncertainty_risk_gap" in eval_metrics

    nature_results = run_nature_analysis(
        prepared_path=prepared_path,
        run_dir=tmp_path,
        output_dir=tmp_path / "nature_analysis",
        comparison_run_dir=tmp_path,
    )
    assert "reliability" in nature_results
    assert "boundary" in nature_results
    assert "niche" in nature_results
    assert (tmp_path / "nature_analysis" / "uncertainty_calibration_bins.csv").exists()
    assert (tmp_path / "nature_analysis" / "failure_case_candidates.csv").exists()
    assert (tmp_path / "nature_analysis" / "boundary_type_summary.csv").exists()
    assert (tmp_path / "nature_analysis" / "niche_biological_summary.csv").exists()

    benchmark_manifest = build_xenium_visium_benchmark(
        cells_path=xenium,
        spots_path=spot_coords,
        output_dir=tmp_path / "xenium_visium_benchmark",
        spot_radius=3.0,
        val_fraction=0.2,
        test_fraction=0.2,
    )
    assert benchmark_manifest["num_spots"] == 3
    assert benchmark_manifest["num_spots_with_ground_truth"] == 3
    assert (tmp_path / "xenium_visium_benchmark" / "xenium_cell_proportions.csv").exists()
    assert (tmp_path / "xenium_visium_benchmark" / "spot_splits.csv").exists()
    assert (tmp_path / "xenium_visium_benchmark" / "spot_ground_truth_qc.csv").exists()
    assert (tmp_path / "xenium_visium_benchmark" / "xenium_visium_benchmark_manifest.json").exists()


def test_train_real_split_eval_outputs_only_requested_spots(tmp_path):
    prepared_path = tmp_path / "prepared.pt"
    spot_ids = ["s1", "s2", "s3"]
    batch = WaveSTGateBatch(
        image_patches=torch.zeros(3, 3, 8, 8),
        st_expression=torch.ones(3, 2),
        reference_prototypes=torch.ones(2, 2),
        coords=torch.arange(6, dtype=torch.float32).reshape(3, 2),
        proportion_gt=torch.tensor([[1.0, 0.0], [0.2, 0.8], [0.0, 1.0]], dtype=torch.float32),
    )
    metadata = BatchMetadata(spot_ids=spot_ids, gene_names=["G1", "G2"], cell_types=["A", "B"])
    save_prepared_dataset(prepared_path, batch, metadata)
    splits = tmp_path / "spot_splits.csv"
    pd.DataFrame({"spot_id": spot_ids, "split": ["train", "val", "test"]}).to_csv(splits, index=False)
    config = {
        "data": {"prepared_path": str(prepared_path), "split_path": str(splits)},
        "model": {
            "latent_dim": 8,
            "hidden_dim": 16,
            "patch_size": 8,
            "loss_expr_weight": 1.0,
            "loss_prop_weight": 0.5,
            "loss_uncertainty_weight": 0.01,
        },
        "training": {
            "steps": 1,
            "learning_rate": 0.001,
            "seed": 5,
            "device": "cpu",
            "checkpoint_path": str(tmp_path / "split_checkpoint.pt"),
            "metrics_path": str(tmp_path / "split_metrics.csv"),
            "train_splits": ["train", "val"],
        },
        "evaluation": {"eval_splits": ["test"]},
        "predictions": {
            "proportions_path": str(tmp_path / "split_predicted_proportions.csv"),
            "gates_path": str(tmp_path / "split_gate_weights.csv"),
            "uncertainty_path": str(tmp_path / "split_spot_uncertainty.csv"),
        },
    }
    config_path = tmp_path / "split_train.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    metrics = train_real_from_config(config_path)
    predicted = pd.read_csv(tmp_path / "split_predicted_proportions.csv")
    assert predicted["spot_id"].tolist() == ["s3"]
    assert metrics["num_train_spots"] == 2.0
    assert metrics["num_eval_spots"] == 1.0
    checkpoint = torch.load(tmp_path / "split_checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint["training"]["train_splits"] == ["train", "val"]
    assert checkpoint["training"]["eval_splits"] == ["test"]


def test_collect_baseline_comparison_recomputes_eval_split(tmp_path):
    prepared_path = tmp_path / "prepared.pt"
    spot_ids = ["s1", "s2", "s3"]
    cell_types = ["A", "B"]
    batch = WaveSTGateBatch(
        image_patches=torch.zeros(3, 3, 8, 8),
        st_expression=torch.ones(3, 2),
        reference_prototypes=torch.ones(2, 2),
        coords=torch.arange(6, dtype=torch.float32).reshape(3, 2),
        proportion_gt=torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
    )
    metadata = BatchMetadata(spot_ids=spot_ids, gene_names=["G1", "G2"], cell_types=cell_types)
    save_prepared_dataset(prepared_path, batch, metadata)
    splits = tmp_path / "splits.csv"
    pd.DataFrame({"spot_id": spot_ids, "split": ["train", "test", "test"]}).to_csv(splits, index=False)

    model_pred = tmp_path / "model_pred.csv"
    baseline_pred = tmp_path / "baseline_pred.csv"
    pd.DataFrame([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], index=spot_ids, columns=cell_types).to_csv(
        model_pred,
        index_label="spot_id",
    )
    pd.DataFrame([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], index=spot_ids, columns=cell_types).to_csv(
        baseline_pred,
        index_label="spot_id",
    )
    model_metrics = tmp_path / "model_metrics.csv"
    pd.DataFrame([{"jsd": 9.0, "spotwise_cosine": 0.0, "mean_celltype_pearson": 0.0, "rmse": 9.0}]).to_csv(
        model_metrics,
        index=False,
    )
    simple_metrics = tmp_path / "simple_metrics.csv"
    pd.DataFrame(
        [
            {
                "method": "bad_baseline",
                "predictions_path": str(baseline_pred),
                "jsd": 0.0,
                "spotwise_cosine": 1.0,
                "mean_celltype_pearson": 1.0,
                "rmse": 0.0,
            }
        ]
    ).to_csv(simple_metrics, index=False)

    comparison = collect_baseline_comparison(
        prepared_path=prepared_path,
        model_metrics_path=model_metrics,
        model_predictions_path=model_pred,
        output_dir=tmp_path / "comparison",
        simple_metrics_path=simple_metrics,
        split_path=splits,
        eval_splits=["test"],
        permutations=100,
        seed=5,
    )
    assert comparison.iloc[0]["method"] == "WaveST-Gate"
    assert comparison.iloc[0]["num_supervised_spots"] == 2
    assert comparison.iloc[0]["jsd"] < comparison.iloc[1]["jsd"]
    manifest = json.loads((tmp_path / "comparison" / "baseline_comparison_manifest.json").read_text(encoding="utf-8"))
    assert manifest["eval_splits"] == ["test"]
    assert manifest["num_evaluation_spots"] == 2


def test_download_manifest_verify(tmp_path):
    root = tmp_path / "raw"
    (root / "ok").mkdir(parents=True)
    (root / "ok" / "file.txt").write_text("abc", encoding="utf-8")
    (root / "bad").mkdir(parents=True)
    (root / "bad" / "file.txt").write_text("abcd", encoding="utf-8")
    manifest = {
        "root_dir": str(root),
        "items": [
            {"id": "ok", "path": "ok/file.txt", "expected_bytes": 3, "required": True},
            {"id": "missing", "path": "missing/file.txt", "expected_bytes": 2, "required": True},
            {"id": "bad", "path": "bad/file.txt", "expected_bytes": 3, "required": True},
            {
                "id": "private",
                "path": "private/UNAVAILABLE.txt",
                "unavailable": True,
                "reason": "Private accession",
            },
        ],
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    results = verify_manifest(manifest_path)
    statuses = {result.item_id: result.status for result in results}
    assert statuses == {
        "ok": "exists",
        "missing": "missing",
        "bad": "size_mismatch",
        "private": "unavailable",
    }
    assert (root / "verify_status.json").exists()


def test_convert_10x_visium(tmp_path):
    matrix_h5 = tmp_path / "filtered_feature_bc_matrix.h5"
    spatial_tar = tmp_path / "spatial.tar.gz"
    image_path = tmp_path / "image.tif"
    Image.new("RGB", (64, 64), (200, 180, 190)).save(image_path)

    spot_by_gene = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 3],
                [0, 2, 0],
                [4, 5, 6],
            ],
            dtype=np.int32,
        )
    )
    gene_by_spot = spot_by_gene.T.tocsc()
    with h5py.File(matrix_h5, "w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("data", data=gene_by_spot.data)
        group.create_dataset("indices", data=gene_by_spot.indices)
        group.create_dataset("indptr", data=gene_by_spot.indptr)
        group.create_dataset("shape", data=gene_by_spot.shape)
        group.create_dataset("barcodes", data=np.array([b"s1", b"s2", b"s3"]))
        features = group.create_group("features")
        features.create_dataset("id", data=np.array([b"ENSG1", b"ENSG2", b"ENSG3"]))
        features.create_dataset("name", data=np.array([b"G1", b"G2", b"G3"]))
        features.create_dataset("feature_type", data=np.array([b"Gene Expression", b"Gene Expression", b"Gene Expression"]))

    positions = tmp_path / "tissue_positions.csv"
    pd.DataFrame(
        {
            "barcode": ["s1", "s2", "s3"],
            "in_tissue": [1, 0, 1],
            "array_row": [1, 2, 3],
            "array_col": [4, 5, 6],
            "pxl_col_in_fullres": [10, 20, 30],
            "pxl_row_in_fullres": [11, 21, 31],
        }
    ).to_csv(positions, index=False)
    with tarfile.open(spatial_tar, "w:gz") as archive:
        archive.add(positions, arcname="spatial/tissue_positions.csv")

    out = convert_10x_visium(
        matrix_h5_path=matrix_h5,
        spatial_tar_path=spatial_tar,
        image_path=image_path,
        output_dir=tmp_path / "converted",
        dataset_id="tiny_visium",
        max_genes=2,
    )
    assert out["num_spots"] == 2
    assert out["num_genes"] == 2
    assert (tmp_path / "converted" / "spot_expression.csv.gz").exists()
    assert (tmp_path / "converted" / "spot_coords.csv").exists()
    assert (tmp_path / "converted" / "baseline_manifest.json").exists()
    assert (tmp_path / "converted" / "prepare_dataset.template.yaml").exists()

    spots = load_spot_expression(tmp_path / "converted" / "spot_expression.csv.gz")
    assert spots.spot_ids == ["s1", "s3"]
    assert spots.gene_names == ["G1", "G2"]
    coords = pd.read_csv(tmp_path / "converted" / "spot_coords.csv")
    assert coords["spot_id"].tolist() == ["s1", "s3"]


def test_convert_wu_swarbrick_visium_atlas(tmp_path):
    root = tmp_path / "wu"
    matrix_dir = root / "filtered_count_matrices" / "S1_filtered_count_matrix"
    spatial_dir = root / "spatial" / "S1_spatial"
    metadata_dir = root / "metadata"
    matrix_dir.mkdir(parents=True)
    spatial_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    gene_by_spot = sparse.coo_matrix(np.array([[1, 0, 4], [0, 2, 5], [3, 0, 6]], dtype=np.int32))
    scipy_io.mmwrite(matrix_dir / "matrix.mtx", gene_by_spot)
    (matrix_dir / "matrix.mtx").rename(matrix_dir / "matrix.mtx.gz")
    pd.DataFrame(["s1", "s2", "s3"]).to_csv(matrix_dir / "barcodes.tsv.gz", sep="\t", header=False, index=False, compression=None)
    pd.DataFrame(
        [["ENSG1", "G1", "Gene Expression"], ["ENSG2", "G2", "Gene Expression"], ["ENSG3", "G3", "Gene Expression"]]
    ).to_csv(matrix_dir / "features.tsv.gz", sep="\t", header=False, index=False, compression=None)

    pd.DataFrame(
        [
            ["s1", 1, 1, 1, 100, 120],
            ["s2", 0, 1, 2, 200, 220],
            ["s3", 1, 2, 1, 300, 320],
        ]
    ).to_csv(spatial_dir / "tissue_positions_list.csv", header=False, index=False)
    (spatial_dir / "scalefactors_json.json").write_text(json.dumps({"tissue_hires_scalef": 0.25}), encoding="utf-8")
    Image.new("RGB", (128, 128), (210, 180, 190)).save(spatial_dir / "tissue_hires_image.png")
    pd.DataFrame({"spot_id": ["s1", "s3"], "pathology": ["tumor", "stroma"]}).to_csv(metadata_dir / "S1_metadata.csv", index=False)

    filtered_tar = tmp_path / "filtered.tar.gz"
    spatial_tar = tmp_path / "spatial.tar.gz"
    metadata_tar = tmp_path / "metadata.tar.gz"
    with tarfile.open(filtered_tar, "w:gz") as archive:
        archive.add(root / "filtered_count_matrices", arcname="filtered_count_matrices")
    with tarfile.open(spatial_tar, "w:gz") as archive:
        archive.add(root / "spatial", arcname="spatial")
    with tarfile.open(metadata_tar, "w:gz") as archive:
        archive.add(root / "metadata", arcname="metadata")

    genes = tmp_path / "genes.txt"
    genes.write_text("G1\nG2\nMISSING\n", encoding="utf-8")
    results = convert_wu_atlas(
        filtered_count_matrices_tar=filtered_tar,
        spatial_tar=spatial_tar,
        metadata_tar=metadata_tar,
        extracted_root=tmp_path / "extracted",
        output_root=tmp_path / "processed",
        genes_path=genes,
        allow_missing_genes=True,
    )

    assert len(results) == 1
    assert results[0]["num_spots"] == 2
    assert results[0]["num_genes"] == 2
    assert results[0]["missing_requested_genes"] == ["MISSING"]
    out_dir = tmp_path / "processed" / "wu_swarbrick_S1_xenium_common301"
    assert (out_dir / "spot_expression.csv.gz").exists()
    assert (out_dir / "spot_coords.csv").exists()
    coords = pd.read_csv(out_dir / "spot_coords.csv")
    assert coords["spot_id"].tolist() == ["s1", "s3"]
    assert coords["x"].tolist() == [25.0, 75.0]
    assert coords["y"].tolist() == [30.0, 80.0]
    expr = pd.read_csv(out_dir / "spot_expression.csv.gz")
    assert expr.columns.tolist() == ["spot_id", "G1", "G2"]
    assert (out_dir / "spot_metadata.csv").exists()


def test_convert_scrna_reference_and_prepare_with_prototypes(tmp_path):
    matrix_h5 = tmp_path / "scrna_feature_matrix.h5"
    cell_by_gene = sparse.csr_matrix(
        np.array(
            [
                [2, 0, 1],
                [4, 0, 3],
                [0, 5, 1],
            ],
            dtype=np.int32,
        )
    )
    gene_by_cell = cell_by_gene.T.tocsc()
    with h5py.File(matrix_h5, "w") as handle:
        group = handle.create_group("matrix")
        group.create_dataset("data", data=gene_by_cell.data)
        group.create_dataset("indices", data=gene_by_cell.indices)
        group.create_dataset("indptr", data=gene_by_cell.indptr)
        group.create_dataset("shape", data=gene_by_cell.shape)
        group.create_dataset("barcodes", data=np.array([b"c1", b"c2", b"c3"]))
        features = group.create_group("features")
        features.create_dataset("id", data=np.array([b"ENSG1", b"ENSG2", b"ENSG3"]))
        features.create_dataset("name", data=np.array([b"G1", b"G2", b"G3"]))
        features.create_dataset("feature_type", data=np.array([b"Gene Expression", b"Gene Expression", b"Gene Expression"]))

    labels = tmp_path / "labels.csv"
    pd.DataFrame({"Barcode": ["c1", "c2", "c3"], "Annotation": ["Tumor", "Tumor", "Immune"]}).to_csv(labels, index=False)
    result = convert_scrna_reference(
        matrix_h5_path=matrix_h5,
        labels_path=labels,
        output_dir=tmp_path / "reference",
        dataset_id="tiny_scrna",
        write_cell_expression=True,
        write_mtx=True,
    )
    assert result["num_labelled_cells"] == 3
    assert result["num_cell_types"] == 2
    assert (tmp_path / "reference" / "cell_expression.csv.gz").exists()
    assert (tmp_path / "reference" / "reference_gene_by_cell.mtx.gz").exists()
    reference = load_reference_prototypes_table(tmp_path / "reference" / "reference_prototypes.csv.gz")
    assert reference.cell_types == ["Immune", "Tumor"]
    assert reference.gene_names == ["G1", "G2", "G3"]
    with gzip.open(tmp_path / "reference" / "reference_gene_by_cell.mtx.gz", "rb") as handle:
        ref_mtx = scipy_io.mmread(handle)
    assert ref_mtx.shape == (3, 3)

    spot_expression = tmp_path / "spot_expression.csv"
    spot_coords = tmp_path / "spot_coords.csv"
    he_image = tmp_path / "he.png"
    pd.DataFrame({"spot_id": ["s1", "s2"], "G1": [1, 2], "G2": [0, 3], "G3": [1, 0]}).to_csv(spot_expression, index=False)
    pd.DataFrame({"spot_id": ["s1", "s2"], "x": [8, 24], "y": [8, 24]}).to_csv(spot_coords, index=False)
    Image.new("RGB", (32, 32), (130, 80, 120)).save(he_image)
    prepare_config = {
        "data": {
            "spot_expression_path": str(spot_expression),
            "spot_coords_path": str(spot_coords),
            "he_image_path": str(he_image),
            "reference_prototypes_path": str(tmp_path / "reference" / "reference_prototypes.csv.gz"),
            "output_path": str(tmp_path / "prepared_from_prototypes.pt"),
        },
        "patches": {"patch_size": 16},
    }
    config_path = tmp_path / "prepare_with_prototypes.yaml"
    config_path.write_text(json.dumps(prepare_config), encoding="utf-8")
    batch, metadata, extra = prepare_from_config(config_path)
    assert batch.reference_prototypes.shape == (2, 3)
    assert metadata.cell_types == ["Immune", "Tumor"]
    assert extra["num_spots"] == 2


def test_create_baseline_bundle(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    spot_expression = dataset_dir / "spot_expression.csv"
    spot_coords = dataset_dir / "spot_coords.csv"
    genes = dataset_dir / "genes.csv"
    raw_h5 = dataset_dir / "raw.h5"
    spatial_tar = dataset_dir / "spatial.tar.gz"
    image = dataset_dir / "image.tif"
    raw_h5.write_text("placeholder", encoding="utf-8")
    spatial_tar.write_text("placeholder", encoding="utf-8")
    Image.new("RGB", (12, 12), (1, 2, 3)).save(image)
    pd.DataFrame({"spot_id": ["s1", "s2"], "G1": [1, 0], "G2": [0, 2]}).to_csv(spot_expression, index=False)
    pd.DataFrame({"spot_id": ["s1", "s2"], "x": [1, 2], "y": [3, 4]}).to_csv(spot_coords, index=False)
    pd.DataFrame({"gene_name": ["G1", "G2"], "gene_id": ["ENSG1", "ENSG2"], "feature_type": ["Gene Expression", "Gene Expression"]}).to_csv(
        genes, index=False
    )
    (dataset_dir / "baseline_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "tiny_dataset",
                "raw_matrix_h5_path": str(raw_h5),
                "raw_spatial_tar_path": str(spatial_tar),
                "image_path": str(image),
                "spot_expression_path": str(spot_expression),
                "spot_coords_path": str(spot_coords),
                "genes_path": str(genes),
            }
        ),
        encoding="utf-8",
    )

    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    ref_genes = reference_dir / "genes.csv"
    labels = reference_dir / "cell_labels.csv"
    prototypes = reference_dir / "reference_prototypes.csv.gz"
    counts = reference_dir / "cell_type_counts.csv"
    ref_h5 = reference_dir / "scrna.h5"
    ref_h5.write_text("placeholder", encoding="utf-8")
    pd.DataFrame({"gene_name": ["G1", "G2"], "gene_id": ["ENSG1", "ENSG2"], "feature_type": ["Gene Expression", "Gene Expression"]}).to_csv(
        ref_genes, index=False
    )
    pd.DataFrame({"cell_id": ["c1", "c2"], "cell_type": ["A", "B"]}).to_csv(labels, index=False)
    pd.DataFrame({"cell_type": ["A", "B"], "num_cells": [1, 1]}).to_csv(counts, index=False)
    pd.DataFrame({"cell_type": ["A", "B"], "G1": [1.0, 0.0], "G2": [0.0, 1.0]}).to_csv(prototypes, index=False)
    reference_manifest = reference_dir / "baseline_scrna_reference.json"
    reference_manifest.write_text(
        json.dumps(
            {
                "raw_matrix_h5_path": str(ref_h5),
                "cell_labels_path": str(labels),
                "cell_expression_path": str(prototypes),
                "reference_prototypes_path": str(prototypes),
                "reference_mtx_path": str(reference_dir / "reference_gene_by_cell.mtx.gz"),
                "reference_cells_path": str(reference_dir / "reference_cells.tsv"),
                "reference_genes_path": str(reference_dir / "reference_genes.tsv"),
                "genes_path": str(ref_genes),
                "cell_type_counts_path": str(counts),
                "num_labelled_cells": 2,
                "num_cell_types": 2,
                "num_genes": 2,
            }
        ),
        encoding="utf-8",
    )
    with gzip.open(reference_dir / "reference_gene_by_cell.mtx.gz", "wb") as handle:
        scipy_io.mmwrite(handle, sparse.csr_matrix(np.eye(2)))
    pd.DataFrame({"cell_id": ["c1", "c2"]}).to_csv(reference_dir / "reference_cells.tsv", sep="\t", index=False)
    pd.DataFrame({"gene_name": ["G1", "G2"]}).to_csv(reference_dir / "reference_genes.tsv", sep="\t", index=False)

    report = create_baseline_bundle(
        datasets_manifest=None,
        dataset_dirs=[str(dataset_dir)],
        reference_manifest_path=reference_manifest,
        output_dir=tmp_path / "bundle",
    )
    assert report["datasets"][0]["status"] == "ok"
    assert (tmp_path / "bundle" / "benchmark_registry.csv").exists()
    assert (tmp_path / "bundle" / "master_spot_splits.csv").exists()
    spatial_mtx = tmp_path / "bundle" / "datasets" / "tiny_dataset" / "baseline_inputs" / "spatial_gene_by_spot.mtx.gz"
    assert spatial_mtx.exists()
    with gzip.open(spatial_mtx, "rb") as handle:
        matrix = scipy_io.mmread(handle)
    assert matrix.shape == (2, 2)
    for method in ["cell2location", "RCTD", "CARD", "Tangram", "SPOTlight", "BayesPrism", "SpatialDWLS", "SpatialDWLS_Seurat"]:
        assert (tmp_path / "bundle" / "datasets" / "tiny_dataset" / "method_configs" / f"{method}.yaml").exists()
