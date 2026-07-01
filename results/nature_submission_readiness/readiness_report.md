# WaveST-Gate Submission Readiness

Generated UTC: 2026-06-05T06:13:55.776361+00:00

Machine-readable report: `results/nature_submission_readiness/readiness_report.json`

## Stage Summary

| Stage | Status | Complete | Partial | Optional pending | Missing |
| --- | --- | ---: | ---: | ---: | ---: |
| 1. Xenium-to-Visium benchmark | complete | 11 | 0 | 0 | 0 |
| 2. Main model training | complete | 3 | 0 | 0 | 0 |
| 3. Strong baseline comparison | complete | 5 | 0 | 0 | 0 |
| 4. Ablation study | complete | 1 | 0 | 0 | 0 |
| 5. Reliability and calibration | complete | 1 | 0 | 0 | 0 |
| 6. Boundary preservation | complete | 2 | 0 | 0 | 0 |
| 7. Biological niche interpretation | complete | 3 | 0 | 0 | 0 |
| 8. External generalization | complete | 4 | 0 | 0 | 0 |
| 9. Robustness | complete | 2 | 0 | 0 | 0 |

## Evidence Items

### 1. Xenium-to-Visium benchmark

- **complete** - spot-level Xenium aggregation artifacts. Evidence: `data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/xenium_visium_benchmark_manifest.json`. Counts, proportions, QC, splits, manifest, and protocol are required. Metric: num_spots=4992; num_spots_with_ground_truth=485; num_cells=115275; num_cell_types=19; spot_radius=55
- **complete** - spot QC fields. Evidence: `data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/spot_ground_truth_qc.csv`. QC includes cell coverage, entropy, dominant cell type, and GT coverage flag. Metric: columns=spot_id,xenium_cell_count,ground_truth_entropy,dominant_cell_type,has_xenium_ground_truth
- **complete** - Zenodo-ready release bundle and metadata. Evidence: `results/nature_release/release_bundle_manifest.json`. Release bundle, upload manifest, Zenodo metadata, and deposition instructions are prepared. Metric: num_files=615; bundle_bytes=104284966
- **complete** - release bundle integrity verification. Evidence: `results/nature_release/release_verification.json`. Release verifier checked tar readability, upload manifest members, critical artifacts, and dry-run bundle consistency. Metric: overall_status=passed; bundle_integrity_status=passed; doi_status=published; tar_member_count=617; critical_artifacts_checked=73; num_failures=0
- **complete** - software and hardware environment report. Evidence: `results/nature_release/environment_report.json`. Environment report records Python, package versions, PyTorch/CUDA/GPU, R packages, and baseline environment status. Metric: python=3.12.4; torch=2.6.0+cu124; cuda_available=True; gpu_count=1
- **complete** - Zenodo DOI / release deposition. Evidence: `results/nature_release/zenodo_deposition_result.json`. A WaveST-Gate release DOI was found in explicit published Zenodo deposition fields. Metric: release_status=zenodo_published; doi=10.5281/zenodo.20550855; zenodo_deposition_id=20550855
- **complete** - benchmark datasheet and data dictionary. Evidence: `results/nature_benchmark_datasheet/benchmark_datasheet.json`. Datasheet records artifact inventory, QC summary, split summary, cell-type totals, data dictionary, and machine-checkable integrity checks for the Xenium-to-Visium benchmark. Metric: status=complete; artifacts=6; cell_types=19; failed_integrity_checks=0
- **complete** - requirement-by-requirement goal completion audit. Evidence: `results/nature_completion_audit/goal_completion_audit.json`. Machine-readable audit maps every Nature-level goal requirement to evidence files and failed checks, with the project DOI kept as an explicit external action. Metric: status=complete; requirements=11; complete=11; partial=0; missing=0
- **complete** - manuscript availability and reproducibility statements. Evidence: `results/nature_manuscript_statements/manuscript_availability_statements.json`. Generated Data availability, Code availability, Reproducibility, and Computing environment statements are evidence-linked and keep the project release DOI separate from upstream dataset DOIs. Metric: status=zenodo_published; doi_status=recorded; doi=10.5281/zenodo.20550855; deposition_id=20550855
- **complete** - manuscript methods and statistical analysis draft. Evidence: `results/nature_manuscript_methods/manuscript_methods.json`. Generated Methods/Supplementary Methods draft links benchmark construction, model specification, training losses, baseline fairness, ablations, reliability, boundary, niche, external generalization, robustness, statistics, compute, and reproducibility evidence. Metric: status=complete; sections=10; missing_required=0
- **complete** - reviewer preflight dossier. Evidence: `results/nature_reviewer_preflight/reviewer_preflight.json`. Reviewer-facing dossier links headline claims, likely reviewer concerns, preemptive responses, stage status, and evidence paths. Metric: status=complete_except_project_doi; claims=9; concerns=12; missing_evidence=0

### 2. Main model training

- **complete** - checkpoint, predictions, reliability maps, and training curve. Evidence: `results/nature_main/cytassist_rep2_radius55`. Main real-data run must provide model state, proportions, expression reconstruction, gates, uncertainty, attention, maps, and training history. Metric: step=499; jsd=0.0129275; spotwise_cosine=0.993274; mean_celltype_pearson=0.92827; num_supervised_spots=485
- **complete** - manuscript Figure 1-5 assets. Evidence: `results/nature_manuscript_figures/figure_manifest.json`. Figure-level assets should exist for workflow, spatial composition, baselines, reliability, boundary/niche/pathology, and pass nonblank validation. Metric: num_figures=6; num_pass=6; num_fail=0
- **complete** - manuscript figure legends and claim evidence. Evidence: `results/nature_manuscript_figure_legends/figure_legends.json`. Generated Figure 1-5 and Supplementary Figure S1 legends link each visual to key claims, values, and evidence files. Metric: status=complete; legends=6; num_fail=0; missing_evidence=0

### 3. Strong baseline comparison

- **complete** - formal baseline table with shared genes/reference/supervised spots. Evidence: `results/nature_main/cytassist_rep2_radius55/baseline_comparison/baseline_comparison.csv`. Completed formal baselines: WaveST-Gate, BayesPrism, RCTD (multi), SpatialDWLS/Seurat, CARD, reference_cosine, reference_nnls, SpatialDWLS, Tangram (rna_count_based, 1000 epochs), cell2location, Tangram (uniform, 1000 epochs), uniform, SPOTlight Metric: missing=; n_methods=13
- **complete** - runtime, memory, and paired significance. Evidence: `results/nature_main/cytassist_rep2_radius55/baseline_comparison/baseline_comparison.csv`. Comparison table should include runtime, memory, and paired per-spot significance columns.
- **complete** - split/bootstrap mean and standard deviation. Evidence: `results/nature_main/cytassist_rep2_radius55/baseline_comparison/baseline_split_bootstrap_summary.csv`. Split-wise and paired bootstrap mean/std statistics are required for the current matched benchmark. Metric: n_methods=13; n_supervised_spots=485; n_bootstraps=500
- **complete** - SpatialDWLS/Seurat package-stack rerun. Evidence: `results/nature_main/cytassist_rep2_radius55/spatialdwls_giotto_baseline/spatialdwls_giotto_metrics.csv`. Full Giotto runDWLSDeconv package-stack rerun produced predictions, metrics, and manifests. Metric: SpatialDWLS/Seurat=ready; jsd=0.252396; spotwise_cosine=0.69183; runtime_seconds=1659.74
- **complete** - independent matched-GT multi-sample mean and standard deviation. Evidence: `results/nature_matched_multisample_baselines/matched_multisample_baseline_summary.csv`. Independent matched-GT datasets should be aggregated for formal mean/std baseline reporting. Metric: num_datasets=2; num_methods=12; complete_methods=9; jsd_mean=0.136431; jsd_std=0.174661; num_datasets=2

### 4. Ablation study

- **complete** - module and modality ablations. Evidence: `results/nature_main/cytassist_rep2_radius55/ablations250/ablation_summary.csv`. Required ablations are present. Metric: n_ablations=12

### 5. Reliability and calibration

- **complete** - uncertainty calibration and modality reliability evidence. Evidence: `results/nature_main/cytassist_rep2_radius55/nature_analysis`. Uncertainty-error correlation, risk coverage, calibration, gate maps, and failure cases are required. Metric: uncertainty_error_pearson=0.526288; uncertainty_error_spearman=0.598622; risk_gap=0.0107746; calibration_bin_pearson=0.961906

### 6. Boundary preservation

- **complete** - typed boundary sharpness and marker validation. Evidence: `results/nature_main/cytassist_rep2_radius55/nature_analysis`. Tumor-stroma, ductal, immune-edge, marker validation, H&E overlay, and no-boundary comparison artifacts are required. Metric: boundary_to_interior_jump_ratio=2.36171; mean_boundary_jump=1.12348
- **complete** - independent pathology metadata validation. Evidence: `results/nature_external_pathology_validation/pathology_class_summary.csv`. External Wu/Swarbrick pathology classifications are compared with no-retuning predicted tumor/immune/stromal composition. Metric: num_datasets=6; num_spots=15601; num_pathology_classes=19; overall_agreement_rate=0.828024

### 7. Biological niche interpretation

- **complete** - tumor-immune-stromal niche outputs. Evidence: `results/nature_main/cytassist_rep2_radius55/nature_analysis`. Niche composition, marker enrichment, gate-by-niche, agent attention-by-niche, and maps are required. Metric: num_niches=5
- **complete** - Xenium neighborhood validation. Evidence: `results/nature_main/cytassist_rep2_radius55/nature_analysis/niche_xenium_neighborhood_summary.csv`. Predicted niches should be checked against local Xenium tumor/immune/stromal neighborhood composition.
- **complete** - independent pathology correspondence. Evidence: `results/nature_external_pathology_validation/pathology_niche_summary.csv`. External predicted niches are cross-tabulated against public pathology classification labels. Metric: num_datasets=6; num_spots=15601; overall_agreement_rate=0.828024

### 8. External generalization

- **complete** - no-retuning external prediction panel. Evidence: `results/nature_external_no_retuning/external_no_retuning_summary.csv`. External no-retuning predictions must include per-dataset proportions, gates, uncertainty, attention, metrics, and manifests. Metric: n_datasets=10; external_10x=4; wu_swarbrick=6; missing_outputs=0
- **complete** - external matched-GT performance metrics. Evidence: `results/nature_external_matched_gt/external_matched_gt_summary.csv`. Rep1 Xenium pseudo-spots provide external no-retuning matched cell-type ground truth metrics and simple-baseline comparison. Metric: num_datasets=1; num_spots=1146
- **complete** - Rep1 held-out minimal-retuning matched-GT adaptation. Evidence: `results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/wavestgate_minimal_retune/test_formal_comparison/baseline_comparison.csv`. Rep2 checkpoint is minimally retuned on Rep1 train/val pseudo-spots and evaluated on held-out Rep1 test spots against formal baselines. Metric: jsd=0.0385261; spotwise_cosine=0.960481; mean_celltype_pearson=0.670521; num_train_spots=975; num_eval_spots=171; top_method=WaveST-Gate; jsd=0.0385261
- **complete** - Rep1 no-retuning and minimal-retuning budget curve. Evidence: `results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/rep1_retune_budget_curve/rep1_minimal_retune_budget_curve.csv`. Rep1 direct transfer is reported as a domain-shift case, and a held-out minimal-retuning budget curve quantifies the adaptation needed to beat the best baseline. Metric: jsd=0.365606; best_baseline_jsd=0.178227; beats_best_baseline=False; budget_steps=25; jsd=0.125886; jsd_margin_vs_best_baseline=0.0523413

### 9. Robustness

- **complete** - core stress tests. Evidence: `results/nature_main/cytassist_rep2_radius55/robustness/robustness_summary.csv`. Core robustness should include gene dropout, gene panels, cell-type removal, prototype perturbation, H&E perturbations, low-expression/low-cell-count subgroups, different splits, and patch sizes. Metric: n_rows=42; removed_celltypes=19; prototype_rows=4; split_rows=5; missing_panels=[]; patch_sizes=128,256,32,64
- **complete** - GT-stratified split sensitivity and benchmark-radius confidence controls. Evidence: `results/nature_main/cytassist_rep2_radius55/split_sensitivity/split_sensitivity_aggregate.csv`. Supplementary GT-stratified splits address the zero-GT validation split, while radius/cell-count sensitivity checks Xenium-to-Visium benchmark-label confidence. Metric: split_files=complete; benchmark_files=complete
