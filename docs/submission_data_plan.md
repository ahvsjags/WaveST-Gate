# Submission Data Plan

This document tracks the data scope needed for a high-quality WaveST-Gate
submission. It separates the already downloaded Xenium-supervised core benchmark
from additional public datasets for independent testing, generalization, and
baseline interoperability.

## Data Tiers

### Tier 1: Core Supervised Benchmark

Manifest:

```bash
data_manifest/wavestgate_breast_core.yaml
```

Role:

- Train and validate the full WaveST-Gate model.
- Build Xenium-to-Visium spot-level cell-type proportions.
- Evaluate supervised deconvolution and interpret reliability gates.
- Provide the shared breast cancer scRNA reference from `GSE243275_RAW.tar`
  for baseline methods.
- Add the public Xenium Rep2 output bundle as a same-cohort hold-out for
  Xenium-level transfer and robustness checks.

Current status:

- Public core files are downloaded and byte-verified.
- The core benchmark depends only on public Janesick breast cancer data:
  `GSE243168` for Xenium, `GSE243275` for scRNA-seq/Visium, and `GSE243280` as
  the public SuperSeries record.
- Neighboring accessions outside this SuperSeries, including private or
  non-breast records, are excluded from the submission benchmark.
- Xenium Rep2 is downloaded, byte-verified, and md5-verified as same-cohort
  hold-out data.
- Xenium scale now includes Rep1 with 167,780 cells / 42,638,083 transcripts
  and Rep2 with 118,752 cells / 31,997,227 transcripts.

### Tier 2: Independent Test Sections

Manifest:

```bash
data_manifest/wavestgate_submission_breast_external.yaml
```

Datasets:

- 10x Human Breast Cancer Block A Section 1.
- 10x Human Breast Cancer Block A Section 2.

Role:

- Held-out section/replicate tests.
- Evaluate whether a model trained on the core benchmark transfers to related
  breast cancer Visium sections.
- Support reviewer-facing evidence that the model is not only fit to one slide.

### Tier 3: External Generalization

Manifest:

```bash
data_manifest/wavestgate_submission_breast_external.yaml
```

Datasets:

- 10x Invasive Ductal Carcinoma stained with fluorescent CD3 antibody.
- 10x FFPE Human Breast Cancer DCIS/Invasive Carcinoma.

Role:

- Cross-sample generalization.
- Cross-image-modality stress test: H&E versus IF/CD3 imaging.
- Cross-preservation stress test: fresh frozen versus FFPE.
- Optional weak validation with pathologist annotation image in the FFPE
  dataset.

### Tier 4: Wu/Swarbrick Visium Atlas

Manifest:

```bash
data_manifest/wavestgate_wu_swarbrick_visium.yaml
```

Dataset:

- Wu/Swarbrick single-cell and spatially resolved atlas of human breast cancers,
  Zenodo DOI `10.5281/zenodo.4739739`.

Role:

- Six additional primary breast cancer Visium samples for external-scale
  validation.
- H&E/annotation image documentation plus spot pathology metadata for
  pathology-aware qualitative checks.
- Public CC-BY-4.0 dataset that strengthens reviewer-facing reproducibility
  without relying on private GEO accessions.

Current status:

- All five Zenodo files are downloaded, byte-verified, and md5-verified.
- Six model-ready batches have been generated with the shared scFFPE
  common-panel reference:
  `data/processed/wu_swarbrick_*_xenium_common301/prepared.pt`.
- The Wu/Swarbrick prepared atlas contributes 15,601 Visium spots. The first
  two samples use 300 overlapping panel genes; the remaining four use 296
  overlapping panel genes, with missing requested genes recorded in each
  `baseline_manifest.json`.
- The current external Visium pool is 30,631 spots: 15,030 spots from the four
  10x external datasets plus 15,601 spots from Wu/Swarbrick.

### Optional: 10x Xenium Breast Biomarker Dataset

10x also provides a newer multi-sample Xenium breast biomarker/custom-panel
dataset. It is useful for future robustness analysis, but it is not part of the
current main submission benchmark because its panel differs from the Janesick
Xenium breast panel used to align the present scFFPE reference, Visium datasets,
and baselines.

## Required Standardized Inputs

Each Visium dataset must be converted into the standard WaveST-Gate input
tables:

```bash
spot_expression.csv
spot_coords.csv
he_image.tif
scrna.csv
xenium_cells.csv optional
proportion_gt.csv optional
```

For external datasets without Xenium cells, `proportion_gt.csv` is not required.
They should still produce model predictions, expression reconstruction metrics,
spatial maps, and baseline-compatible input files.

## Paper Gene Panel

The official Xenium breast panel is stored at:

```bash
data/processed/gene_panels/xenium_breast_panel.csv
data/processed/gene_panels/xenium_breast_panel.txt
```

Coverage across the external Visium datasets and the shared scFFPE reference is
recorded at:

```bash
data/processed/gene_panels/xenium_breast_panel_coverage.csv
```

The full Xenium breast panel has 313 genes. The FFPE external Visium dataset
covers 301 of them, so the current cross-dataset paper panel is:

```bash
data/processed/gene_panels/xenium_breast_panel_common_visium_scffpe.txt
```

This common panel is used for the main external prepared batches and baseline
bundle.

## Baseline Compatibility Target

The same standardized expression/reference files should support:

- `cell2location`
- `RCTD`
- `CARD`
- `Tangram`

Minimum shared artifacts:

- Visium count matrix with spot barcodes and gene names.
- Spot coordinate table in full-resolution image pixels.
- scRNA count matrix or cell-type prototype table.
- Cell-type labels for the scRNA reference.
- Image path and Space Ranger scale factors for image-aware baselines.
- Train/test split table so all methods evaluate on identical spots.

Reference source:

- `data/raw/wavestgate_breast_core/geo/GSE243275/GSE243275_RAW.tar`
  contains the public 5' scRNA, 3' scRNA, Flex scRNA, and Visium CytAssist
  files from the `GSE243275` subseries.
- The labelled scFFPE reference has been standardized at
  `data/processed/scffpe_reference/`.
- Its model-ready prototype table is
  `data/processed/scffpe_reference/reference_prototypes.csv.gz`.
- Its baseline label file is
  `data/processed/scffpe_reference/cell_labels.csv`.
- The common 301-gene reference used for the current external benchmark is
  `data/processed/scffpe_reference_xenium_common301/`.

## Conversion Tools

The current `prepare_dataset` CLI consumes simple CSV/TSV/NPY/PIL inputs. The
10x adapter converts:

- `filtered_feature_bc_matrix.h5`
- `spatial.tar.gz`
- full-resolution image file
- shared scRNA reference

into the standard tables above and a `prepare_dataset` config.

```bash
python -m wavestgate.data.prepare_10x_visium --help
python -m wavestgate.data.prepare_wu_swarbrick_visium --help
python -m wavestgate.data.prepare_scrna_reference --help
python -m wavestgate.data.prepare_xenium_pseudospots --help
python -m wavestgate.evaluation.prepare_baseline_inputs --help
python -m wavestgate.evaluation.cell2location_baseline --help
python -m wavestgate.evaluation.tangram_baseline --help
python -m wavestgate.evaluation.card_baseline --help
python -m wavestgate.evaluation.spotlight_baseline --help
python -m wavestgate.evaluation.bayesprism_baseline --help
python -m wavestgate.evaluation.spatialdwls_baseline --help
python -m wavestgate.evaluation.run_baselines --help
python -m wavestgate.evaluation.baseline_statistics --help
python -m wavestgate.evaluation.matched_multisample_baselines --help
python -m wavestgate.evaluation.external_pathology_validation --help
python -m wavestgate.evaluation.robustness --help
python -m wavestgate.evaluation.benchmark_datasheet --help
python -m wavestgate.evaluation.completion_audit --help
python -m wavestgate.evaluation.final_doi_gate --help
python -m wavestgate.evaluation.prepare_release --help
python -m wavestgate.evaluation.submission_readiness --help
```

The baseline bundle generator emits method configs for:

- `cell2location`
- `RCTD`
- `CARD`
- `Tangram`
- `SPOTlight`
- `BayesPrism`
- `SpatialDWLS`
- `SpatialDWLS/Seurat`

It also verifies that expression spots match coordinate rows and all spatial
genes are present in the shared scRNA reference.

The submission readiness generator consolidates the current benchmark, model,
baseline, ablation, reliability, boundary, niche, external generalization, and
robustness evidence into a report plus release checksum manifest:

The benchmark datasheet/data dictionary can be regenerated before readiness
auditing:

```bash
python -m wavestgate.evaluation.benchmark_datasheet \
  --output-dir results/nature_benchmark_datasheet
```

The requirement-by-requirement completion audit checks the full Nature-level
goal list against concrete evidence files and reports whether anything besides
the project DOI remains:

```bash
python -m wavestgate.evaluation.completion_audit \
  --output-dir results/nature_completion_audit
```

The final DOI gate is the strict pre-submission check. In the current audited
release it passes because an explicit project Zenodo DOI and deposition id are
recorded:

```bash
python -m wavestgate.evaluation.final_doi_gate --strict
```

```bash
python -m wavestgate.evaluation.submission_readiness \
  --output-dir results/nature_submission_readiness
```

The generated report is deliberately conservative. It marks directly supported
claims as complete and keeps the project release DOI visible. In the current
audited state, all nine readiness stages are complete, and the release record
contains Zenodo deposition id `20550855`, DOI `10.5281/zenodo.20550855`, and
record URL `https://zenodo.org/records/20550855`.

Zenodo release deposition is scripted so the DOI step is reproducible rather
than manual. The dry-run records metadata and bundle checks without network
upload; sandbox should be used before the production draft:

```bash
python -m wavestgate.evaluation.zenodo_deposit --dry-run \
  --bundle-manifest results/nature_release/release_bundle_manifest.json \
  --output results/nature_release/zenodo_deposition_result.json

ZENODO_ACCESS_TOKEN=<token> python -m wavestgate.evaluation.zenodo_deposit \
  --sandbox \
  --bundle-manifest results/nature_release/release_bundle_manifest.json

ZENODO_ACCESS_TOKEN=<token> python -m wavestgate.evaluation.zenodo_deposit \
  --bundle-manifest results/nature_release/release_bundle_manifest.json

ZENODO_ACCESS_TOKEN=<token> python -m wavestgate.evaluation.zenodo_deposit \
  --publish \
  --bundle-manifest results/nature_release/release_bundle_manifest.json

ZENODO_ACCESS_TOKEN=<token> python -m wavestgate.evaluation.finalize_submission --deposit
python -m wavestgate.evaluation.final_doi_gate --strict
```

The helper writes `zenodo_deposition_result.json` and updates
`release_bundle_manifest.json` with the deposition id, DOI, record URL, and
release status returned by Zenodo. Readiness only marks the DOI stage complete
when a project release DOI appears in those explicit deposition fields. The
current published release satisfies that gate; the current bundle file count,
byte size, and SHA256 are recorded in `release_bundle_manifest.json` after each
refresh.

The release bundle integrity verifier can be run after bundle/dry-run refresh:

```bash
python -m wavestgate.evaluation.verify_release \
  --bundle-manifest results/nature_release/release_bundle_manifest.json \
  --deposition-result results/nature_release/zenodo_deposition_result.json \
  --output results/nature_release/release_verification.json \
  --markdown-output results/nature_release/release_verification.md
```

It verifies tar readability, upload manifest row count, per-member bytes/SHA256,
critical artifacts, and bundle/deposition consistency. In the current release,
`--require-doi` is expected to pass because the Zenodo DOI is recorded.

The final local submission chain can be refreshed in the correct order with:

```bash
python -m wavestgate.evaluation.finalize_submission
```

This regenerates the software/hardware environment report, benchmark
datasheet/data dictionary, requirement completion audit, manuscript tables,
Figure 1-5 assets, figure legends/claim evidence, manuscript
availability/reproducibility statements, Methods/statistical-analysis draft
material, reviewer preflight dossier, the Zenodo-ready bundle, Zenodo handoff,
release verification, readiness reports, and final handoff
files in `results/nature_release/final_submission_handoff.{json,md}`. The
handoff files are written after bundle creation so they can record the final
bundle SHA256 without creating self-referential tarball hashes. Use
`--deposit` only when a real `ZENODO_ACCESS_TOKEN` is available and the bundle
has been reviewed.

The environment report can also be refreshed directly with:

```bash
python -m wavestgate.evaluation.environment_report \
  --output results/nature_release/environment_report.json \
  --markdown-output results/nature_release/environment_report.md
```

It records Python, package versions, PyTorch/CUDA/GPU, R packages, and the
formal-baseline environment audit.

The current robustness panel can be refreshed with:

```bash
python -m wavestgate.evaluation.robustness \
  --checkpoint results/nature_main/cytassist_rep2_radius55/checkpoint.pt \
  --prepared data/processed/cytassist_xenium_rep2_common297/prepared_xenium_gt_radius55.pt \
  --output-dir results/nature_main/cytassist_rep2_radius55/robustness \
  --batch-size 512 \
  --device cuda \
  --splits data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/spot_splits.csv \
  --xenium-counts data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/xenium_cell_counts.csv
```

The reviewer-facing split and benchmark-label confidence sensitivity analyses
can be refreshed with:

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
  --output-dir results/nature_main/cytassist_rep2_radius55/benchmark_sensitivity \
  --radii 45 55 65 75 \
  --min-cell-counts 1 5 10 20 50
```

The current baseline split/bootstrap statistics can be refreshed with:

```bash
python -m wavestgate.evaluation.baseline_statistics \
  --prepared data/processed/cytassist_xenium_rep2_common297/prepared_xenium_gt_radius55.pt \
  --comparison results/nature_main/cytassist_rep2_radius55/baseline_comparison/baseline_comparison.csv \
  --splits data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/spot_splits.csv \
  --output-dir results/nature_main/cytassist_rep2_radius55/baseline_comparison \
  --n-bootstraps 500 \
  --seed 2026
```

The external pathology metadata validation can be refreshed with:

```bash
python -m wavestgate.evaluation.external_pathology_validation \
  --processed-root data/processed \
  --external-results-dir results/nature_external_no_retuning \
  --output-dir results/nature_external_pathology_validation \
  --dataset-prefix wu_swarbrick \
  --n-niches 5
```

The Rep1 external matched-GT pseudo-spot benchmark can be refreshed with:

```bash
python -m wavestgate.data.prepare_xenium_pseudospots \
  --typed-cells data/processed/xenium_rep1_typed_cells/cells_he_aligned.csv \
  --cell-feature-matrix data/raw/wavestgate_breast_core/10x/xenium_rep1/extracted/outs/cell_feature_matrix.h5 \
  --he-image data/raw/wavestgate_breast_core/10x/xenium_rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif \
  --reference-prototypes data/processed/scffpe_reference_xenium_common301/reference_prototypes.csv.gz \
  --target-genes data/processed/gene_panels/xenium_breast_panel_cytassist_common297.txt \
  --output-dir data/processed/xenium_rep1_pseudospots_radius55_common297 \
  --radius 55 \
  --stride 110 \
  --min-cells 5 \
  --patch-size 128 \
  --graph-k 6
```

The Rep1 no-retuning matched-GT prediction and evaluation can be refreshed with:

```bash
python -m wavestgate.training.predict_aligned \
  --checkpoint results/nature_main/cytassist_rep2_radius55/checkpoint.pt \
  --prepared data/processed/xenium_rep1_pseudospots_radius55_common297/prepared.pt \
  --output-dir results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297 \
  --batch-size 8 \
  --device cuda

python -m wavestgate.evaluation.evaluate_real \
  --predictions results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/predicted_proportions.csv \
  --prepared data/processed/xenium_rep1_pseudospots_radius55_common297/prepared.pt \
  --uncertainty results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/spot_uncertainty.csv \
  --output-metrics results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/matched_gt_metrics.csv
```

The Rep1 formal matched-GT comparison currently includes Tangram,
SpatialDWLS, cell2location, RCTD multi, CARD, and simple baselines:

```bash
python -m wavestgate.evaluation.run_baselines collect \
  --prepared data/processed/xenium_rep1_pseudospots_radius55_common297/prepared.pt \
  --model-metrics results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/matched_gt_metrics.csv \
  --model-predictions results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/predicted_proportions.csv \
  --simple-metrics results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/simple_baselines/simple_baseline_metrics.csv \
  --tangram-metrics results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/tangram_baseline/tangram_metrics.csv \
  --baseline-metrics results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/spatialdwls_baseline/spatialdwls_metrics.csv \
  --baseline-metrics results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/cell2location_baseline/cell2location_metrics.csv \
  --baseline-metrics results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/rctd_baseline_multi/rctd_metrics.csv \
  --baseline-metrics results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/card_baseline/card_metrics.csv \
  --output-dir results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/formal_comparison \
  --permutations 10000 \
  --seed 2026
```

BayesPrism and SPOTlight were attempted on Rep1 external pseudo-spots but did
not produce valid Rep1 metrics under the current 50GB RAM/time constraints.
They remain present in the main CytAssist/Rep2 benchmark comparison.

The Rep1 held-out minimal-retuning adaptation can be refreshed with:

```bash
python -m wavestgate.training.train_real \
  --config results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/wavestgate_minimal_retune/config.yaml

python -m wavestgate.evaluation.run_baselines collect \
  --prepared data/processed/xenium_rep1_pseudospots_radius55_common297/prepared.pt \
  --model-metrics results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/wavestgate_minimal_retune/test_metrics.csv \
  --model-predictions results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/wavestgate_minimal_retune/predicted_proportions.csv \
  --simple-metrics results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/simple_baselines/simple_baseline_metrics.csv \
  --tangram-metrics results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/tangram_baseline/tangram_metrics.csv \
  --baseline-metrics results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/spatialdwls_baseline/spatialdwls_metrics.csv \
  --baseline-metrics results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/cell2location_baseline/cell2location_metrics.csv \
  --baseline-metrics results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/rctd_baseline_multi/rctd_metrics.csv \
  --baseline-metrics results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/card_baseline/card_metrics.csv \
  --splits data/processed/xenium_rep1_pseudospots_radius55_common297/spot_splits.csv \
  --eval-splits test \
  --output-dir results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/wavestgate_minimal_retune/test_formal_comparison \
  --permutations 10000 \
  --seed 2026
```

The current minimal-retuning result uses 975 Rep1 train/val pseudo-spots and
171 held-out Rep1 test pseudo-spots. It reaches test JSD `0.0385`, compared
with RCTD multi test JSD `0.1782`.

The honest Rep1 no-retuning/minimal-retuning budget curve can be refreshed
with:

```bash
python -m wavestgate.evaluation.rep1_retune_budget_curve \
  --base-config results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/wavestgate_minimal_retune/config.yaml \
  --no-retune-predictions results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/predicted_proportions.csv \
  --baseline-comparison results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/wavestgate_minimal_retune/test_formal_comparison/baseline_comparison.csv \
  --output-dir results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/rep1_retune_budget_curve \
  --budgets 0 25 50 100 250 500
```

The curve reports direct Rep1 no-retuning as a domain-shift case and shows
that 25 minimal-retuning steps already beat the best Rep1 baseline on held-out
test spots.

Independent matched-GT multi-sample baseline mean/std can be refreshed with:

```bash
python -m wavestgate.evaluation.matched_multisample_baselines \
  --comparison cytassist_rep2_radius55=results/nature_main/cytassist_rep2_radius55/baseline_comparison/baseline_comparison.csv \
  --comparison xenium_rep1_pseudospots_radius55_common297=results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/formal_comparison/baseline_comparison.csv \
  --output-dir results/nature_matched_multisample_baselines

python -m wavestgate.evaluation.matched_multisample_baselines \
  --comparison cytassist_rep2_radius55=results/nature_main/cytassist_rep2_radius55/baseline_comparison/baseline_comparison.csv \
  --comparison xenium_rep1_minimal_retune_test=results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/wavestgate_minimal_retune/test_formal_comparison/baseline_comparison.csv \
  --output-dir results/nature_matched_multisample_baselines_minimal_retune
```

Manuscript-ready CSV tables can be refreshed with:

```bash
python -m wavestgate.evaluation.manuscript_tables \
  --output-dir results/nature_manuscript_tables
```

This writes benchmark, main model, baseline, ablation, reliability/boundary/
niche, external generalization, robustness, split sensitivity, image-gate,
radius/cell-count sensitivity, and Rep1 budget-curve tables plus a Markdown
index.

Manuscript figure assets can be refreshed with:

```bash
python -m wavestgate.evaluation.manuscript_figures \
  --output-dir results/nature_manuscript_figures
```

This writes Figure 1-5 assets, a robustness supplement, and
`figure_manifest.{csv,json,md}` with dimensions, SHA256, and nonblank checks.

Manuscript figure legends and claim evidence can be refreshed with:

```bash
python -m wavestgate.evaluation.manuscript_figure_legends \
  --output-dir results/nature_manuscript_figure_legends
```

This writes `figure_legends.{json,md}` linking Figure 1-5 and Supplementary
Figure S1 to key claims, values, and evidence files.

Manuscript Methods and statistical-analysis draft material can be refreshed
with:

```bash
python -m wavestgate.evaluation.manuscript_methods \
  --output-dir results/nature_manuscript_methods
```

This writes `manuscript_methods.{json,md}` covering benchmark construction,
model specification, training losses, baseline fairness, ablations,
reliability/calibration, boundary preservation, biological niches, external
generalization, robustness, statistics, compute, and reproducibility evidence.

Reviewer preflight and response evidence can be refreshed with:

```bash
python -m wavestgate.evaluation.reviewer_preflight \
  --output-dir results/nature_reviewer_preflight
```

This writes `reviewer_preflight.{json,md}` linking headline claims, likely
reviewer concerns, preemptive responses, current stage status, and evidence
paths. It keeps the project DOI as pending until a real Zenodo deposition
returns explicit DOI fields.

The Zenodo-ready release bundle can be refreshed with:

```bash
python -m wavestgate.evaluation.prepare_release \
  --output-dir results/nature_release \
  --version 0.1.0
```

The release manifest records `critical_artifacts` for the main model
checkpoint, predicted proportions, reconstructed expression, gate/reliability
outputs, uncertainty, agent attention, and proportion/reliability/boundary/
niche summaries. Directory scans still exclude raw public data and unlisted
large binaries; explicitly listed manuscript-critical artifacts are bundled.

The current formal cell2location run uses an isolated Python 3.10 environment
at `/root/miniconda3/envs/cell2loc_env/bin/python` with CUDA PyTorch,
cell2location `0.1.5`, scvi-tools `1.3.3`, mean raw scRNA cell-type
signatures, and the same benchmark genes/supervised spots:

```bash
python -m wavestgate.evaluation.cell2location_baseline \
  --prepared data/processed/cytassist_xenium_rep2_common297/prepared_xenium_gt_radius55.pt \
  --spatial-expression data/processed/cytassist_xenium_rep2_common297/spot_expression.csv.gz \
  --scrna-expression data/processed/scffpe_reference_xenium_common301/cell_expression.csv.gz \
  --scrna-labels data/processed/scffpe_reference_xenium_common301/cell_labels.csv \
  --output-dir results/nature_main/cytassist_rep2_radius55/cell2location_baseline \
  --max-epochs 1000 \
  --batch-size 512 \
  --num-samples 500 \
  --accelerator gpu
```

The current formal Tangram run is executed in cluster mode with identical
scRNA labels, benchmark genes, supervised spots, and metrics:

```bash
python -m wavestgate.evaluation.tangram_baseline \
  --prepared data/processed/cytassist_xenium_rep2_common297/prepared_xenium_gt_radius55.pt \
  --scrna-expression data/processed/scffpe_reference_xenium_common301/cell_expression.csv.gz \
  --scrna-labels data/processed/scffpe_reference_xenium_common301/cell_labels.csv \
  --output-dir results/nature_main/cytassist_rep2_radius55/tangram_baseline \
  --device cuda \
  --num-epochs 1000
```

The current formal RCTD run uses the `spacexr` R package in multi-mode with the
same spot expression, coordinates, scRNA reference, benchmark genes, supervised
spots, and metrics:

```bash
python -m wavestgate.evaluation.rctd_baseline \
  --prepared data/processed/cytassist_xenium_rep2_common297/prepared_xenium_gt_radius55.pt \
  --spatial-expression data/processed/cytassist_xenium_rep2_common297/spot_expression.csv.gz \
  --spatial-coords data/processed/cytassist_xenium_rep2_common297/spot_coords.csv \
  --scrna-expression data/processed/scffpe_reference_xenium_common301/cell_expression.csv.gz \
  --scrna-labels data/processed/scffpe_reference_xenium_common301/cell_labels.csv \
  --output-dir results/nature_main/cytassist_rep2_radius55/rctd_baseline_multi \
  --doublet-mode multi \
  --max-cores 10
```

The current formal CARD run uses the R `CARD` package plus its `MuSiC`
runtime dependency with the same standardized benchmark inputs:

```bash
python -m wavestgate.evaluation.card_baseline \
  --prepared data/processed/cytassist_xenium_rep2_common297/prepared_xenium_gt_radius55.pt \
  --spatial-expression data/processed/cytassist_xenium_rep2_common297/spot_expression.csv.gz \
  --spatial-coords data/processed/cytassist_xenium_rep2_common297/spot_coords.csv \
  --scrna-expression data/processed/scffpe_reference_xenium_common301/cell_expression.csv.gz \
  --scrna-labels data/processed/scffpe_reference_xenium_common301/cell_labels.csv \
  --output-dir results/nature_main/cytassist_rep2_radius55/card_baseline \
  --min-count-gene 1 \
  --min-count-spot 1
```

The current formal SPOTlight run uses SPOTlight `1.2.0` on R `4.1.2`,
seeded NMF with 25 marker genes per cell type, and a fixed 500-cell
per-cell-type scRNA subsample because the full-reference NMF attempt exceeded
the machine's 50GB RAM limit:

```bash
python -m wavestgate.evaluation.spotlight_baseline \
  --prepared data/processed/cytassist_xenium_rep2_common297/prepared_xenium_gt_radius55.pt \
  --spatial-expression data/processed/cytassist_xenium_rep2_common297/spot_expression.csv.gz \
  --scrna-expression data/processed/scffpe_reference_xenium_common301/cell_expression.csv.gz \
  --scrna-labels data/processed/scffpe_reference_xenium_common301/cell_labels.csv \
  --output-dir results/nature_main/cytassist_rep2_radius55/spotlight_baseline \
  --top-markers 25 \
  --max-cells-per-type 500 \
  --nrun 1 \
  --max-iter 200 \
  --min-prop 0
```

The current formal BayesPrism run uses BayesPrism `2.2.3` with the same
benchmark genes, supervised spots, and scFFPE reference. Because BayesPrism is
bulk-oriented, this run uses GEP mode from the same 27,472 reference cells,
disables the bulk outlier filter for spatial spots, skips one zero-expression
spot during inference, and realigns predictions to all benchmark spots:

```bash
python -m wavestgate.evaluation.bayesprism_baseline \
  --prepared data/processed/cytassist_xenium_rep2_common297/prepared_xenium_gt_radius55.pt \
  --spatial-expression data/processed/cytassist_xenium_rep2_common297/spot_expression.csv.gz \
  --scrna-expression data/processed/scffpe_reference_xenium_common301/cell_expression.csv.gz \
  --scrna-labels data/processed/scffpe_reference_xenium_common301/cell_labels.csv \
  --output-dir results/nature_main/cytassist_rep2_radius55/bayesprism_baseline \
  --input-type GEP \
  --n-cores 10 \
  --chain-length 30 \
  --burn-in 10 \
  --thinning 2 \
  --no-update-gibbs \
  --which-theta first \
  --optimizer MLE \
  --outlier-cut 1 \
  --outlier-fraction 1 \
  --seed 123
```

The current formal SpatialDWLS run uses a standalone `Matrix`/`quadprog`
implementation of the Giotto `runDWLSDeconv` DWLS formulation. It builds a
top-marker scFFPE signature matrix, performs first-pass DWLS cell-type
screening at `1/n_cell`, then reruns dampened weighted least squares on the
screened cell types:

```bash
python -m wavestgate.evaluation.spatialdwls_baseline \
  --prepared data/processed/cytassist_xenium_rep2_common297/prepared_xenium_gt_radius55.pt \
  --spatial-expression data/processed/cytassist_xenium_rep2_common297/spot_expression.csv.gz \
  --scrna-expression data/processed/scffpe_reference_xenium_common301/cell_expression.csv.gz \
  --scrna-labels data/processed/scffpe_reference_xenium_common301/cell_labels.csv \
  --output-dir results/nature_main/cytassist_rep2_radius55/spatialdwls_baseline \
  --top-markers 25 \
  --n-cell 50 \
  --dampening-j 2 \
  --max-iter 100 \
  --tol 0.01
```

The full Giotto/Seurat package-stack rerun is also available and has been run
on all 4,992 benchmark spots with metrics evaluated on the same 485
Xenium-supervised spots:

```bash
python -m wavestgate.evaluation.spatialdwls_giotto_baseline \
  --prepared data/processed/cytassist_xenium_rep2_common297/prepared_xenium_gt_radius55.pt \
  --spatial-expression data/processed/cytassist_xenium_rep2_common297/spot_expression.csv.gz \
  --scrna-expression data/processed/scffpe_reference_xenium_common301/cell_expression.csv.gz \
  --scrna-labels data/processed/scffpe_reference_xenium_common301/cell_labels.csv \
  --output-dir results/nature_main/cytassist_rep2_radius55/spatialdwls_giotto_baseline \
  --top-markers 25 \
  --cluster-size 64 \
  --cutoff 2 \
  --n-cell 50
```

This rerun uses Seurat `5.5.0`, Giotto `4.2.3` from the R4.1-compatible
Giotto branch, GiottoClass `0.5.1`, GiottoUtils `0.2.5`, GiottoVisuals
`0.2.15`, Rfast `2.1.5.2`, and quadprog `1.5.8`.

Completed model and baseline metrics are merged with:

```bash
python -m wavestgate.evaluation.run_baselines collect \
  --prepared data/processed/cytassist_xenium_rep2_common297/prepared_xenium_gt_radius55.pt \
  --model-metrics results/nature_main/cytassist_rep2_radius55/metrics.csv \
  --model-predictions results/nature_main/cytassist_rep2_radius55/predicted_proportions.csv \
  --simple-metrics results/nature_main/cytassist_rep2_radius55/simple_baselines/simple_baseline_metrics.csv \
  --tangram-metrics results/nature_main/cytassist_rep2_radius55/tangram_baseline/tangram_metrics.csv \
  --tangram-metrics results/nature_main/cytassist_rep2_radius55/tangram_baseline_uniform_prior/tangram_metrics.csv \
  --baseline-metrics results/nature_main/cytassist_rep2_radius55/rctd_baseline_multi/rctd_metrics.csv \
  --baseline-metrics results/nature_main/cytassist_rep2_radius55/card_baseline/card_metrics.csv \
  --baseline-metrics results/nature_main/cytassist_rep2_radius55/cell2location_baseline/cell2location_metrics.csv \
  --baseline-metrics results/nature_main/cytassist_rep2_radius55/spotlight_baseline/spotlight_metrics.csv \
  --baseline-metrics results/nature_main/cytassist_rep2_radius55/bayesprism_baseline/bayesprism_metrics.csv \
  --baseline-metrics results/nature_main/cytassist_rep2_radius55/spatialdwls_baseline/spatialdwls_metrics.csv \
  --baseline-metrics results/nature_main/cytassist_rep2_radius55/spatialdwls_giotto_baseline/spatialdwls_giotto_metrics.csv \
  --output-dir results/nature_main/cytassist_rep2_radius55/baseline_comparison
```

Spatial proportion maps, reliability calibration, boundary preservation, and
biological niche analysis are generated with:

```bash
python -m wavestgate.evaluation.nature_analysis \
  --prepared data/processed/cytassist_xenium_rep2_common297/prepared_xenium_gt_radius55.pt \
  --run-dir results/nature_main/cytassist_rep2_radius55 \
  --comparison-run-dir results/nature_main/cytassist_rep2_radius55/ablations250/no_boundary_loss \
  --output-dir results/nature_main/cytassist_rep2_radius55/nature_analysis
```

This emits per-cell-type and tumor/immune/stromal proportion maps, calibration
bins and plots, risk-coverage curves, high-error failure-case candidates,
typed boundary summaries, boundary marker validation, and named
tumor/stromal/immune niche summaries.

For baseline methods that prefer matrix inputs, the current bundle also emits:

- per-dataset `baseline_inputs/spatial_gene_by_spot.mtx.gz`
- per-dataset `baseline_inputs/spatial_spots.tsv`
- per-dataset `baseline_inputs/spatial_genes.tsv`
- per-dataset `baseline_inputs/spatial_coords.csv`
- reference `reference_gene_by_cell.mtx.gz`
- reference `reference_cells.tsv`
- reference `reference_genes.tsv`
- reference `cell_expression.csv.gz`

## Current Prepared External Batches

Panel-limited smoke/pilot batches have been prepared with 64 genes, the shared
scFFPE prototype reference, and 32 px image patches:

```bash
data/processed/breast_block_a_section_1_panel64/prepared.pt
data/processed/breast_block_a_section_2_panel64/prepared.pt
data/processed/invasive_ductal_carcinoma_cd3_if_panel64/prepared.pt
data/processed/ffpe_dcis_invasive_carcinoma_panel64/prepared.pt
```

These are intended for fast pipeline and held-out inference checks. Full
analysis should rerun `prepare_10x_visium` with a paper gene panel or without
`--max-genes`, depending on the final benchmark design.

The current paper-panel external batches use the common 301-gene Xenium breast
panel, the shared scFFPE common301 reference, and 32 px image patches:

```bash
data/processed/breast_block_a_section_1_xenium_common301/prepared.pt
data/processed/breast_block_a_section_2_xenium_common301/prepared.pt
data/processed/invasive_ductal_carcinoma_cd3_if_xenium_common301/prepared.pt
data/processed/ffpe_dcis_invasive_carcinoma_xenium_common301/prepared.pt
```

The Wu/Swarbrick atlas batches use per-sample overlap with the same common
panel/reference:

```bash
data/processed/wu_swarbrick_1142243F_xenium_common301/prepared.pt
data/processed/wu_swarbrick_1160920F_xenium_common301/prepared.pt
data/processed/wu_swarbrick_CID4290_xenium_common301/prepared.pt
data/processed/wu_swarbrick_CID4465_xenium_common301/prepared.pt
data/processed/wu_swarbrick_CID44971_xenium_common301/prepared.pt
data/processed/wu_swarbrick_CID4535_xenium_common301/prepared.pt
```

Their spot/gene counts are summarized in:

```bash
data/processed/wu_swarbrick_xenium_common301_summary.csv
```

The current baseline bundle for these data is:

```bash
data/processed/baseline_bundles/external_xenium_common301/
```

Wu/Swarbrick-only and combined submission-scale baseline bundles are:

```bash
data/processed/baseline_bundles/wu_swarbrick_xenium_common301/
data/processed/baseline_bundles/submission_xenium_common301_plus_wu/
```

It contains:

- `benchmark_registry.csv`
- `master_spot_splits.csv`
- `baseline_bundle_report.json`
- per-dataset `spot_splits.csv`
- per-dataset method configs for `cell2location`, `RCTD`, `CARD`, `Tangram`,
  `SPOTlight`, `BayesPrism`, `SpatialDWLS`, and `SpatialDWLS/Seurat`
- per-dataset sparse MatrixMarket spatial count matrices
- shared sparse MatrixMarket and cell-level expression files for the scFFPE
  reference
