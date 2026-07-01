# WaveST-Gate

WaveST-Gate is a PyTorch MVP for reliability-aware, morphology-guided spatial
transcriptomics deconvolution. It combines H&E image patches, ST spot
expression, and scRNA cell-type reference prototypes to estimate spot-level
cell-type proportions, modality reliability, prediction uncertainty, and
optional biological niches.

This repository currently contains two parts:

- `CV...`: the original extracted CV module collection, kept as read-only
  reference material.
- `wavestgate`: the clean biological method package built from those ideas.

## Citation and Release Metadata

Software citation metadata are provided in `CITATION.cff`, CodeMeta metadata in
`codemeta.json`, and the software license in `LICENSE`. The release bundle
also records these files as critical artifacts, so the Zenodo-ready package can
be machine-checked before deposition.

The current public archived release is available on Zenodo:

- DOI: `10.5281/zenodo.20550855`
- Record: `https://zenodo.org/records/20550855`

Refresh the final local submission package with:

```bash
python -m wavestgate.evaluation.finalize_submission
```

This refreshes environment, benchmark datasheet/data dictionary, requirement
completion audit, tables, figures, figure legends, manuscript availability
statements, manuscript Methods/statistical-analysis draft, reviewer preflight
dossier, release bundle, verification, readiness, and final handoff files.
It performs a dry-run Zenodo handoff by default. Use `--deposit` only after
reviewing the bundle and providing a real `ZENODO_ACCESS_TOKEN`.

Check the final DOI gate before submission:

```bash
python -m wavestgate.evaluation.final_doi_gate --strict
```

Refresh the reviewer-facing leakage and baseline fairness supplement with:

```bash
python -m wavestgate.evaluation.leakage_fairness_audit
```

## MVP workflow

```bash
python -m wavestgate.training.train --config examples/small_breast_demo/config.yaml
```

The demo uses synthetic data to validate import, forward, backward, checkpoint
writing, and metrics output. Real Visium/Xenium/scRNA inputs can be prepared
through the lightweight interfaces below.

## Public API

- `WaveSTGateConfig`: model dimensions and loss weights.
- `WaveSTGateBatch`: image patches, ST expression, reference prototypes, and
  optional supervision.
- `WaveSTGateOutput`: deconvolution outputs, calibrated gate weights,
  spot-level uncertainty, modality uncertainty/reliability, agent attention,
  and optional niche logits.

## Lightweight real-data interfaces

The MVP now includes CSV/TSV/NPY/PIL interfaces for the next data milestone:

- `load_spot_expression` and `load_spot_coordinates` for spot tables.
- `build_reference_prototypes` for scRNA cell-type mean prototypes.
- `extract_spot_patches` for spot-centered H&E patch extraction from regular
  image files.
- `build_knn_graph` and `build_radius_graph` for spatial smoothness graphs.
- `count_cells_in_spots` and `proportions_from_counts` for Xenium-to-Visium
  ground-truth construction.
- `assemble_wavestgate_batch` to align genes/cell types and build a model-ready
  batch.

Prepare a real-data tensor bundle:

```bash
python -m wavestgate.data.prepare_dataset --config examples/real_data_demo/prepare_dataset.yaml
```

Train from the prepared bundle:

```bash
python -m wavestgate.training.train_real --config examples/real_data_demo/train_real.yaml
```

Predict and evaluate a held-out prepared bundle:

```bash
python -m wavestgate.training.predict_real \
  --checkpoint examples/real_data_demo/real_checkpoint.pt \
  --prepared examples/real_data_demo/prepared.pt \
  --proportions examples/real_data_demo/predicted_proportions.csv \
  --gates examples/real_data_demo/gate_weights.csv \
  --uncertainty examples/real_data_demo/spot_uncertainty.csv \
  --modality-reliability examples/real_data_demo/modality_reliability.csv

python -m wavestgate.evaluation.evaluate_real \
  --predictions examples/real_data_demo/predicted_proportions.csv \
  --prepared examples/real_data_demo/prepared.pt \
  --uncertainty examples/real_data_demo/spot_uncertainty.csv \
  --output-metrics examples/real_data_demo/evaluation_metrics.csv
```

For morphology-contribution controls, the package includes an optional
image-gate participation loss and a paired image-vs-no-image analysis:

```bash
python -m wavestgate.training.train_real \
  --config experiments/nature_main/train_cytassist_rep2_radius55_imagegate.yaml

python -m wavestgate.training.train_real \
  --config experiments/nature_main/train_cytassist_rep2_radius55_imagegate_noimage.yaml

python -m wavestgate.evaluation.image_contribution \
  --prepared data/processed/cytassist_xenium_rep2_common297/prepared_xenium_gt_radius55.pt \
  --image-run-dir results/nature_main/cytassist_rep2_radius55_imagegate \
  --no-image-run-dir results/nature_main/cytassist_rep2_radius55_imagegate_noimage \
  --baseline-run-dir results/nature_main/cytassist_rep2_radius55 \
  --output-dir results/nature_main/cytassist_rep2_radius55_imagegate/image_contribution
```

Additional reviewer-facing sensitivity analyses are included for the primary
benchmark split, Xenium aggregation radius/cell-count confidence, and Rep1
minimal-retuning budget:

```bash
python -m wavestgate.evaluation.split_sensitivity \
  --prepared data/processed/cytassist_xenium_rep2_common297/prepared_xenium_gt_radius55.pt \
  --predictions results/nature_main/cytassist_rep2_radius55/predicted_proportions.csv \
  --splits data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/spot_splits.csv \
  --qc data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/spot_ground_truth_qc.csv \
  --output-dir results/nature_main/cytassist_rep2_radius55/split_sensitivity

python -m wavestgate.evaluation.benchmark_sensitivity \
  --prepared data/processed/cytassist_xenium_rep2_common297/prepared_xenium_gt_radius55.pt \
  --predictions results/nature_main/cytassist_rep2_radius55/predicted_proportions.csv \
  --benchmark-manifest data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/xenium_visium_benchmark_manifest.json \
  --output-dir results/nature_main/cytassist_rep2_radius55/benchmark_sensitivity

python -m wavestgate.evaluation.rep1_retune_budget_curve \
  --base-config results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/wavestgate_minimal_retune/config.yaml \
  --no-retune-predictions results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/predicted_proportions.csv \
  --baseline-comparison results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/wavestgate_minimal_retune/test_formal_comparison/baseline_comparison.csv \
  --output-dir results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/rep1_retune_budget_curve \
  --budgets 0 25 50 100 250 500
```

The real-data configs expect ordinary tabular inputs first: spot expression,
spot coordinates, scRNA expression with cell-type labels, an H&E image, and
optionally a Xenium cell table for spot-level ground-truth proportions and a
spot-level niche-label table for supervised niche analysis.

Build a reproducible Xenium-to-Visium benchmark protocol with ground-truth
counts/proportions, spatial holdout splits, QC, and a manifest:

```bash
python -m wavestgate.evaluation.xenium_visium_benchmark \
  --config examples/real_data_demo/xenium_visium_benchmark.yaml
```

Public benchmark downloads are tracked in:

```bash
python -m wavestgate.data.download \
  --manifest data_manifest/wavestgate_breast_core.yaml \
  --verify-only
```

Additional independent-test and generalization data are tracked in:

```bash
python -m wavestgate.data.download \
  --manifest data_manifest/wavestgate_submission_breast_external.yaml \
  --verify-only
```

The Wu/Swarbrick six-sample breast cancer Visium atlas is tracked in:

```bash
python -m wavestgate.data.download \
  --manifest data_manifest/wavestgate_wu_swarbrick_visium.yaml \
  --verify-only
```

Convert official 10x Visium files into the standard real-data table format:

```bash
python -m wavestgate.data.prepare_10x_visium \
  --matrix-h5 data/raw/wavestgate_submission_breast_external/10x/breast_block_a_section_1/V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5 \
  --spatial-tar data/raw/wavestgate_submission_breast_external/10x/breast_block_a_section_1/V1_Breast_Cancer_Block_A_Section_1_spatial.tar.gz \
  --image data/raw/wavestgate_submission_breast_external/10x/breast_block_a_section_1/V1_Breast_Cancer_Block_A_Section_1_image.tif \
  --output-dir data/processed/breast_block_a_section_1 \
  --dataset-id breast_block_a_section_1
```

Convert the Wu/Swarbrick atlas into standard tables and prepared batches:

```bash
python -m wavestgate.data.prepare_wu_swarbrick_visium \
  --genes data/processed/gene_panels/xenium_breast_panel_common_visium_scffpe.txt \
  --allow-missing-genes \
  --reference-prototypes-path data/processed/scffpe_reference_xenium_common301/reference_prototypes.csv.gz \
  --prepare \
  --patch-size 32 \
  --graph-k 6
```

Prepare a labelled scRNA reference from 10x h5 plus the Janesick cell-type
annotation workbook:

```bash
python -m wavestgate.data.prepare_scrna_reference \
  --matrix-h5 data/raw/wavestgate_breast_core/geo/GSE243275/extracted/GSM7782698_count_raw_feature_bc_matrix.h5 \
  --labels data/raw/wavestgate_breast_core/geo/GSE243275/GSE243275_Barcode_Cell_Type_Matrices.xlsx \
  --labels-sheet scFFPE-Seq \
  --output-dir data/processed/scffpe_reference \
  --dataset-id scffpe_reference
```

Create baseline-compatible configs and fixed test splits for the external
Xenium-panel benchmark:

```bash
python -m wavestgate.evaluation.prepare_baseline_inputs \
  --datasets-manifest examples/baseline_benchmark/external_xenium_common301_datasets.yaml \
  --reference-manifest data/processed/scffpe_reference_xenium_common301/baseline_scrna_reference.json \
  --output-dir data/processed/baseline_bundles/external_xenium_common301 \
  --split-mode all_test
```

Create the expanded 10-sample submission bundle including Wu/Swarbrick:

```bash
python -m wavestgate.evaluation.prepare_baseline_inputs \
  --datasets-manifest examples/baseline_benchmark/submission_xenium_common301_plus_wu_datasets.yaml \
  --reference-manifest data/processed/scffpe_reference_xenium_common301/baseline_scrna_reference.json \
  --output-dir data/processed/baseline_bundles/submission_xenium_common301_plus_wu \
  --split-mode all_test
```

This bundle writes per-dataset `cell2location`, `RCTD`, `CARD`, and `Tangram`
configs plus sparse MatrixMarket spatial/reference matrices for method runners.

See `docs/project_brief.md` for the implementation roadmap.
