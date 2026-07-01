"""Convert the Wu/Swarbrick breast cancer Visium atlas into WaveST-Gate tables."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import tarfile
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

from wavestgate.data.prepare_10x_visium import (
    TenxMatrix,
    _load_gene_list,
    _select_gene_indices,
    _write_expression_csv_gz,
)
from wavestgate.data.prepare_dataset import prepare_from_config


def _require_mtx_dependencies():
    try:
        from scipy import io as scipy_io
    except Exception as exc:  # pragma: no cover - exercised only when optional deps are absent.
        raise RuntimeError("prepare_wu_swarbrick_visium requires optional dependency: scipy") from exc
    return scipy_io


def _safe_extract(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        root = output_dir.resolve()
        for member in archive.getmembers():
            target = (output_dir / member.name).resolve()
            if not str(target).startswith(str(root)):
                raise ValueError(f"Unsafe archive member: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Could not extract archive member: {member.name}")
                with source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle)


def extract_wu_archives(
    filtered_count_matrices_tar: str | Path,
    spatial_tar: str | Path,
    metadata_tar: str | Path,
    extracted_root: str | Path,
) -> Path:
    """Extract the Wu/Swarbrick atlas archives into a reusable raw directory."""

    extracted = Path(extracted_root)
    _safe_extract(Path(filtered_count_matrices_tar), extracted)
    _safe_extract(Path(spatial_tar), extracted)
    _safe_extract(Path(metadata_tar), extracted)
    return extracted


def read_10x_mtx_folder(matrix_dir: str | Path) -> TenxMatrix:
    """Read a 10x MatrixMarket folder into spot-by-gene orientation."""

    scipy_io = _require_mtx_dependencies()
    matrix_dir = Path(matrix_dir)
    matrix_path = matrix_dir / "matrix.mtx.gz"
    try:
        matrix = scipy_io.mmread(matrix_path).T.tocsr()
    except gzip.BadGzipFile:
        with matrix_path.open("rb") as handle:
            matrix = scipy_io.mmread(handle).T.tocsr()
    barcodes = _read_tsv_maybe_gzip(matrix_dir / "barcodes.tsv.gz")[0].astype(str).tolist()
    features = _read_tsv_maybe_gzip(matrix_dir / "features.tsv.gz")
    if features.shape[1] == 1:
        gene_ids = features[0].astype(str).tolist()
        gene_names = gene_ids
        feature_types = ["Gene Expression"] * len(gene_names)
    else:
        gene_ids = features[0].astype(str).tolist()
        gene_names = features[1].astype(str).tolist()
        feature_types = features[2].astype(str).tolist() if features.shape[1] > 2 else ["Gene Expression"] * len(gene_names)
    return TenxMatrix(
        matrix=matrix,
        barcodes=barcodes,
        gene_ids=gene_ids,
        gene_names=gene_names,
        feature_types=feature_types,
    )


def _read_tsv_maybe_gzip(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t", header=None, compression="gzip")
    except gzip.BadGzipFile:
        return pd.read_csv(path, sep="\t", header=None, compression=None)


def read_wu_spatial_folder(spatial_dir: str | Path, scale_to_hires: bool = True) -> pd.DataFrame:
    """Read Wu/Swarbrick Space Ranger spatial positions.

    The archive provides `tissue_hires_image.png` rather than a full-resolution
    TIFF. When `scale_to_hires` is true, full-resolution spot coordinates are
    multiplied by `tissue_hires_scalef` so patch extraction aligns to the hires
    PNG used by the generated prepare config.
    """

    spatial_dir = Path(spatial_dir)
    positions = pd.read_csv(spatial_dir / "tissue_positions_list.csv", header=None)
    positions.columns = ["spot_id", "in_tissue", "array_row", "array_col", "fullres_x", "fullres_y"]
    with (spatial_dir / "scalefactors_json.json").open("r", encoding="utf-8") as handle:
        scalefactors = json.load(handle)
    hires_scale = float(scalefactors.get("tissue_hires_scalef", 1.0))
    positions["spot_id"] = positions["spot_id"].astype(str)
    for column in ["in_tissue", "array_row", "array_col"]:
        positions[column] = pd.to_numeric(positions[column], errors="raise").astype(int)
    for column in ["fullres_x", "fullres_y"]:
        positions[column] = pd.to_numeric(positions[column], errors="raise")
    if scale_to_hires:
        positions["x"] = positions["fullres_x"] * hires_scale
        positions["y"] = positions["fullres_y"] * hires_scale
    else:
        positions["x"] = positions["fullres_x"]
        positions["y"] = positions["fullres_y"]
    positions["tissue_hires_scalef"] = hires_scale
    return positions[["spot_id", "x", "y", "fullres_x", "fullres_y", "in_tissue", "array_row", "array_col", "tissue_hires_scalef"]]


def _write_metadata_subset(metadata_path: Path, output_path: Path, spot_ids: Sequence[str]) -> str | None:
    if not metadata_path.exists():
        return None
    metadata = pd.read_csv(metadata_path)
    if metadata.empty:
        return None
    id_candidates = ["spot_id", "barcode", "Barcode", "barcodes"]
    id_col = next((col for col in id_candidates if col in metadata.columns), metadata.columns[0])
    metadata[id_col] = metadata[id_col].astype(str)
    selected = metadata[metadata[id_col].isin(set(spot_ids))].copy()
    if selected.empty:
        selected = metadata.copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)
    return str(output_path)


def convert_wu_sample(
    sample_id: str,
    extracted_root: str | Path,
    output_dir: str | Path,
    reference_prototypes_path: str | Path | None = None,
    scrna_expression_path: str | Path | None = None,
    genes_path: str | Path | None = None,
    allow_missing_genes: bool = False,
    max_genes: int | None = None,
    in_tissue_only: bool = True,
    scale_to_hires: bool = True,
    patch_size: int = 32,
    graph_k: int = 6,
    chunk_size: int = 128,
    prepare: bool = False,
) -> dict[str, object]:
    """Convert one Wu/Swarbrick sample into standard files and optional tensors."""

    extracted = Path(extracted_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matrix_dir = extracted / "filtered_count_matrices" / f"{sample_id}_filtered_count_matrix"
    spatial_dir = extracted / "spatial" / f"{sample_id}_spatial"
    metadata_path = extracted / "metadata" / f"{sample_id}_metadata.csv"
    image_path = spatial_dir / "tissue_hires_image.png"
    if not matrix_dir.exists():
        raise FileNotFoundError(f"Missing matrix directory for sample {sample_id}: {matrix_dir}")
    if not spatial_dir.exists():
        raise FileNotFoundError(f"Missing spatial directory for sample {sample_id}: {spatial_dir}")
    if not image_path.exists():
        raise FileNotFoundError(f"Missing hires image for sample {sample_id}: {image_path}")

    tenx = read_10x_mtx_folder(matrix_dir)
    spatial = read_wu_spatial_folder(spatial_dir, scale_to_hires=scale_to_hires).set_index("spot_id")
    selected_rows = []
    selected_spots = []
    missing_positions = []
    for idx, barcode in enumerate(tenx.barcodes):
        if barcode not in spatial.index:
            missing_positions.append(barcode)
            continue
        if in_tissue_only and int(spatial.loc[barcode, "in_tissue"]) != 1:
            continue
        selected_rows.append(idx)
        selected_spots.append(barcode)
    if missing_positions:
        raise ValueError(f"Spatial folder is missing barcodes from matrix: {missing_positions[:5]}")
    if not selected_rows:
        raise ValueError(f"No spots selected for {sample_id} after applying in_tissue filter")

    requested_genes = _load_gene_list(genes_path)
    missing_requested_genes: list[str] = []
    if requested_genes is not None and allow_missing_genes:
        available = set(tenx.gene_names)
        missing_requested_genes = [gene for gene in requested_genes if gene not in available]
        requested_genes = [gene for gene in requested_genes if gene in available]
    gene_indices, selected_genes = _select_gene_indices(tenx.gene_names, requested_genes, max_genes=max_genes)

    expression_path = output / "spot_expression.csv.gz"
    coords_path = output / "spot_coords.csv"
    _write_expression_csv_gz(
        expression_path,
        tenx.matrix,
        tenx.barcodes,
        selected_genes,
        selected_rows,
        gene_indices,
        chunk_size=chunk_size,
    )
    coords = spatial.loc[selected_spots].reset_index()
    coords.to_csv(coords_path, index=False)

    genes_path_out = output / "genes.csv"
    pd.DataFrame(
        {
            "gene_name": selected_genes,
            "gene_id": [tenx.gene_ids[idx] for idx in gene_indices],
            "feature_type": [tenx.feature_types[idx] for idx in gene_indices],
        }
    ).to_csv(genes_path_out, index=False)
    metadata_out = _write_metadata_subset(metadata_path, output / "spot_metadata.csv", selected_spots)

    baseline_manifest = {
        "dataset_id": f"wu_swarbrick_{sample_id}",
        "source": "Wu/Swarbrick Zenodo 4739739",
        "matrix_dir": str(matrix_dir),
        "spatial_dir": str(spatial_dir),
        "image_path": str(image_path),
        "metadata_path": metadata_out,
        "spot_expression_path": str(expression_path),
        "spot_coords_path": str(coords_path),
        "genes_path": str(genes_path_out),
        "scrna_expression_path": str(scrna_expression_path) if scrna_expression_path is not None else None,
        "reference_prototypes_path": str(reference_prototypes_path) if reference_prototypes_path is not None else None,
        "num_spots": len(selected_spots),
        "num_genes": len(selected_genes),
        "missing_requested_genes": missing_requested_genes,
        "in_tissue_only": in_tissue_only,
        "coords_scaled_to_hires_image": scale_to_hires,
    }
    baseline_manifest_path = output / "baseline_manifest.json"
    baseline_manifest_path.write_text(json.dumps(baseline_manifest, indent=2), encoding="utf-8")

    data_config = {
        "spot_expression_path": str(expression_path),
        "spot_coords_path": str(coords_path),
        "he_image_path": str(image_path),
        "output_path": str(output / "prepared.pt"),
    }
    if reference_prototypes_path is not None:
        data_config["reference_prototypes_path"] = str(reference_prototypes_path)
    else:
        data_config["scrna_expression_path"] = str(scrna_expression_path) if scrna_expression_path is not None else "CHANGE_ME_scrna_expression.csv"

    prepare_config = {
        "data": data_config,
        "patches": {"patch_size": patch_size},
        "graph": {"k": graph_k},
        "columns": {"spot_id_col": "spot_id", "spot_x_col": "x", "spot_y_col": "y"},
    }
    config_name = "prepare_dataset.yaml" if scrna_expression_path is not None or reference_prototypes_path is not None else "prepare_dataset.template.yaml"
    prepare_config_path = output / config_name
    prepare_config_path.write_text(yaml.safe_dump(prepare_config, sort_keys=False), encoding="utf-8")

    prepared_path = None
    if prepare:
        prepare_from_config(prepare_config_path)
        prepared_path = str(output / "prepared.pt")

    return {
        "dataset_id": f"wu_swarbrick_{sample_id}",
        "sample_id": sample_id,
        "spot_expression_path": str(expression_path),
        "spot_coords_path": str(coords_path),
        "genes_path": str(genes_path_out),
        "metadata_path": metadata_out,
        "baseline_manifest_path": str(baseline_manifest_path),
        "prepare_config_path": str(prepare_config_path),
        "prepared_path": prepared_path,
        "num_spots": len(selected_spots),
        "num_genes": len(selected_genes),
        "missing_requested_genes": missing_requested_genes,
    }


def discover_wu_samples(extracted_root: str | Path) -> list[str]:
    matrices = Path(extracted_root) / "filtered_count_matrices"
    samples = []
    for path in sorted(matrices.glob("*_filtered_count_matrix")):
        samples.append(path.name.removesuffix("_filtered_count_matrix"))
    return samples


def convert_wu_atlas(
    filtered_count_matrices_tar: str | Path,
    spatial_tar: str | Path,
    metadata_tar: str | Path,
    output_root: str | Path,
    extracted_root: str | Path,
    reference_prototypes_path: str | Path | None = None,
    scrna_expression_path: str | Path | None = None,
    genes_path: str | Path | None = None,
    allow_missing_genes: bool = False,
    max_genes: int | None = None,
    prepare: bool = False,
    patch_size: int = 32,
    graph_k: int = 6,
    chunk_size: int = 128,
) -> list[dict[str, object]]:
    extracted = extract_wu_archives(filtered_count_matrices_tar, spatial_tar, metadata_tar, extracted_root)
    results = []
    for sample_id in discover_wu_samples(extracted):
        results.append(
            convert_wu_sample(
                sample_id=sample_id,
                extracted_root=extracted,
                output_dir=Path(output_root) / f"wu_swarbrick_{sample_id}_xenium_common301",
                reference_prototypes_path=reference_prototypes_path,
                scrna_expression_path=scrna_expression_path,
                genes_path=genes_path,
                allow_missing_genes=allow_missing_genes,
                max_genes=max_genes,
                prepare=prepare,
                patch_size=patch_size,
                graph_k=graph_k,
                chunk_size=chunk_size,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the Wu/Swarbrick Visium atlas into WaveST-Gate standard files.")
    parser.add_argument("--filtered-count-matrices-tar", default="data/raw/wavestgate_wu_swarbrick_visium/zenodo/filtered_count_matrices.tar.gz")
    parser.add_argument("--spatial-tar", default="data/raw/wavestgate_wu_swarbrick_visium/zenodo/spatial.tar.gz")
    parser.add_argument("--metadata-tar", default="data/raw/wavestgate_wu_swarbrick_visium/zenodo/metadata.tar.gz")
    parser.add_argument("--extracted-root", default="data/raw/wavestgate_wu_swarbrick_visium/extracted")
    parser.add_argument("--output-root", default="data/processed")
    parser.add_argument("--reference-prototypes-path", default=None)
    parser.add_argument("--scrna-expression-path", default=None)
    parser.add_argument("--genes", default=None, help="Optional newline-delimited gene list.")
    parser.add_argument("--allow-missing-genes", action="store_true", help="Use the overlap when requested genes are absent from a Wu sample.")
    parser.add_argument("--max-genes", type=int, default=None)
    parser.add_argument("--prepare", action="store_true", help="Also run prepare_dataset for each converted sample.")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--graph-k", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()
    results = convert_wu_atlas(
        filtered_count_matrices_tar=args.filtered_count_matrices_tar,
        spatial_tar=args.spatial_tar,
        metadata_tar=args.metadata_tar,
        output_root=args.output_root,
        extracted_root=args.extracted_root,
        reference_prototypes_path=args.reference_prototypes_path,
        scrna_expression_path=args.scrna_expression_path,
        genes_path=args.genes,
        allow_missing_genes=args.allow_missing_genes,
        max_genes=args.max_genes,
        prepare=args.prepare,
        patch_size=args.patch_size,
        graph_k=args.graph_k,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
