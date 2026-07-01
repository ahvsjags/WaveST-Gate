# Xenium-to-Visium Breast Deconvolution Benchmark Protocol

## Objective

Construct a reproducible single-cell-resolved benchmark for breast cancer
spatial deconvolution by aggregating supervised Xenium Rep2 cells to matched
CytAssist Visium spots.

## Inputs

- Visium expression: `data/processed/cytassist_xenium_rep2_common297/spot_expression.csv.gz`
- Visium spots: `data/processed/cytassist_xenium_rep2_common297/spot_coords.csv`
- H&E image: `data/raw/wavestgate_breast_core/10x/xenium_rep2/Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.tif`
- Xenium cells: `data/raw/wavestgate_breast_core/10x/xenium_rep2/extracted/outs/cells.csv.gz`
- Xenium labels: `Cell_Barcode_Type_Matrices.xlsx`, sheet `Xenium R2 Fig1-5 (supervised)`
- Alignment matrix: `Xenium_FFPE_Human_Breast_Cancer_Rep2_he_imagealignment.csv`
- Reference prototypes: `data/processed/scffpe_reference_xenium_common301/reference_prototypes.csv.gz`

## Coordinate And Label Processing

1. Join Xenium `cells.csv.gz` with supervised workbook labels on `cell_id`/`Barcode`.
2. Normalize cell-type names by replacing underscores with spaces.
3. Drop `Unlabeled` and `Undefined` cells.
4. Apply the inverse 3x3 H&E alignment matrix to map Xenium centroids into
   Visium/H&E pixel coordinates.
5. Aggregate cells into Visium spots with radius `55` pixels.

## Outputs

- Typed cells: `data/processed/xenium_rep2_typed_cells/cells_he_aligned.csv`
- Cell-label QC: `data/processed/xenium_rep2_typed_cells/qc.json`
- Benchmark counts: `data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/xenium_cell_counts.csv`
- Benchmark proportions: `data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/xenium_cell_proportions.csv`
- Spot QC: `data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/spot_ground_truth_qc.csv`
- Splits: `data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/spot_splits.csv`
- Manifest: `data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/xenium_visium_benchmark_manifest.json`
- Training batch: `data/processed/cytassist_xenium_rep2_common297/prepared_xenium_gt_radius55.pt`

## Current Benchmark Size

- Visium spots: `4,992`
- Xenium labelled cells: `115,275`
- Cell types: `19`
- Visium spots with Xenium ground truth: `485`
- Gene panel: `297` genes; missing from CytAssist relative to common301:
  `ANGPT2`, `AKR1C1`, `CD8B`, `BTNL9`

## Reproduction Commands

```bash
python -m wavestgate.data.prepare_xenium_cells \
  --cells data/raw/wavestgate_breast_core/10x/xenium_rep2/extracted/outs/cells.csv.gz \
  --labels data/raw/wavestgate_breast_core/10x/xenium_rep1/Cell_Barcode_Type_Matrices.xlsx \
  --sheet "Xenium R2 Fig1-5 (supervised)" \
  --affine data/raw/wavestgate_breast_core/10x/xenium_rep2/Xenium_FFPE_Human_Breast_Cancer_Rep2_he_imagealignment.csv \
  --affine-direction inverse \
  --output data/processed/xenium_rep2_typed_cells/cells_he_aligned.csv \
  --qc data/processed/xenium_rep2_typed_cells/qc.json
```

```bash
python -m wavestgate.evaluation.xenium_visium_benchmark \
  --config examples/real_data_demo/xenium_visium_benchmark.yaml
```

## DOI / Zenodo Status

The benchmark is locally reproducible and manifest-backed. Zenodo deposition is
pending because it requires account credentials and final author approval.
