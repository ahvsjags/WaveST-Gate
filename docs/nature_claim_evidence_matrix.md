# WaveST-Gate Nature Claim-to-Evidence Matrix

This matrix links each submission-level claim to the exact reproducible
evidence in the workspace. It is meant to be used when drafting the manuscript,
responding to reviewers, or checking the Zenodo release before submission.

## Headline Claims

| Claim | Evidence | Current status | Reviewer risk addressed |
| --- | --- | --- | --- |
| WaveST-Gate provides a reproducible Xenium-to-Visium breast cancer spatial deconvolution benchmark. | `data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/xenium_visium_benchmark_manifest.json`; `data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/spot_ground_truth_qc.csv`; `results/nature_main/cytassist_rep2_radius55/split_sensitivity/`; `results/nature_main/cytassist_rep2_radius55/benchmark_sensitivity/`; `docs/xenium_to_visium_benchmark_protocol.md`; `results/nature_release/release_bundle_manifest.json` | Complete locally and archived on Zenodo with DOI `10.5281/zenodo.20550855`. Primary validation is documented as unsupervised; GT-stratified split sensitivity and radius/cell-count confidence sensitivity are complete. | Prevents the work from being framed as only a private model experiment or an arbitrary-label benchmark. |
| WaveST-Gate is accurate on real multimodal ST data, not only synthetic data. | `results/nature_main/cytassist_rep2_radius55/metrics.csv`; `results/nature_main/cytassist_rep2_radius55/training_history.csv`; `results/nature_main/cytassist_rep2_radius55/predicted_proportions.csv`; `results/nature_main/cytassist_rep2_radius55/nature_analysis/proportion_maps/` | Complete. Main supervised JSD `0.0129275`, spotwise cosine `0.993274`, mean cell-type Pearson `0.92827`. | Counters "toy demo" and "metric-only" concerns. |
| The method is not a loose CNN/attention/gate stack; each component has measurable necessity. | `results/nature_main/cytassist_rep2_radius55/ablations250/ablation_summary.csv` | Complete. Twelve ablations cover wavelet/CNN replacement, agents, gate, uncertainty, boundary loss, local refinement, and modality-only variants. | Counters the "module soup" criticism. |
| The reliability gate is calibrated and interpretable rather than ordinary attention. | `results/nature_main/cytassist_rep2_radius55/nature_analysis/reliability_summary.json`; `uncertainty_calibration_bins.csv`; `risk_coverage_curve.csv`; `failure_case_candidates.csv`; gate maps in `nature_analysis/` | Complete. Uncertainty-error Pearson `0.526288`, Spearman `0.598622`, calibration-bin Pearson `0.961906`. | Shows the gate has reliability semantics and failure-case behavior. |
| Morphology-aware constraints protect tumor-stroma, ductal, and immune boundaries. | `results/nature_main/cytassist_rep2_radius55/nature_analysis/boundary_summary.json`; typed boundary summaries; marker validation tables; H&E overlays; no-boundary-loss comparison | Complete. Boundary-to-interior jump ratio `2.36171`. | Counters the concern that spatial losses simply over-smooth biology. |
| WaveST-Gate reveals breast cancer tumor-immune-stromal niches. | `results/nature_main/cytassist_rep2_radius55/nature_analysis/niche_summary.csv`; `niche_xenium_neighborhood_summary.csv`; gate-by-niche and agent-attention-by-niche summaries; `results/nature_external_pathology_validation/pathology_niche_summary.csv` | Complete. Five named niches with Xenium neighborhood and external pathology correspondence. | Moves the result beyond accuracy into biological interpretation. |
| Results generalize beyond one sample. | `results/nature_external_no_retuning/external_no_retuning_summary.csv`; `results/nature_external_matched_gt/external_matched_gt_summary.csv`; `results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/wavestgate_minimal_retune/test_formal_comparison/baseline_comparison.csv`; `results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/rep1_retune_budget_curve/`; `results/nature_matched_multisample_baselines_minimal_retune/matched_multisample_baseline_summary.csv` | Complete. Includes 10 external no-retuning datasets plus Rep1 matched-GT minimal-retuning evaluation. Rep1 direct no-retuning is reported as domain shift, and the budget curve shows 25 steps beat the best Rep1 baseline on held-out test. | Counters "works on one sample" concerns without hiding the Rep1 no-retuning limitation. |
| The method is robust to realistic perturbations. | `results/nature_main/cytassist_rep2_radius55/robustness/robustness_summary.csv`; `robustness_manifest.json`; `patch_size_robustness/patch_size_summary.csv`; `split_sensitivity/`; `benchmark_sensitivity/` | Complete. Forty-two stress-test rows cover patch sizes, gene panels/dropout, reference mismatch/removal, prototype perturbation, H&E perturbation, low-count/low-expression spots, split variation, GT-stratified split sensitivity, and radius/cell-count benchmark sensitivity. | Counters fragility, split dependence, and parameter sensitivity concerns. |
| H&E morphology contributes despite a low mean image gate in the primary model. | `results/nature_main/cytassist_rep2_radius55_imagegate/image_contribution/image_contribution_summary.json`; `imagegate_run_comparison.csv`; `image_contribution_texture_groups.csv`; `results/nature_manuscript_tables/table_9_imagegate_supplement.csv` | Complete. Image-gate-enhanced control mean image gate `0.132`, raw image gate `0.2068`; matched no-image control is worse, and high-texture spots show positive paired improvement. | Counters the question "why include H&E if the average gate is small?" |

## Stage Evidence Checklist

| Stage | Required deliverable | Evidence | Status |
| --- | --- | --- | --- |
| 1. Xenium-to-Visium protocol | cell counts, proportions, QC, coverage, entropy, dominant cell type, fixed splits, radius/coordinate/cell-type mapping, manifest, protocol | `data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/`; `docs/xenium_to_visium_benchmark_protocol.md` | Complete. |
| 1. Zenodo/DOI | release bundle, metadata, upload manifest, deposition result | `results/nature_release/release_bundle_manifest.json`; `results/nature_release/zenodo_metadata.json`; `results/nature_release/zenodo_deposition_result.json` | Complete. Zenodo record `20550855` and DOI `10.5281/zenodo.20550855` are recorded. |
| 2. Main model | checkpoint, proportions, reconstructed expression, reliability/gate/uncertainty/agent/niche maps, curves, metrics | `results/nature_main/cytassist_rep2_radius55/`; `results/nature_main/cytassist_rep2_radius55/nature_analysis/proportion_maps/` | Complete. |
| 3. Baselines | cell2location, RCTD, CARD, Tangram, SpatialDWLS/Seurat, BayesPrism, shared inputs, means/std, significance, runtime/memory | `results/nature_main/cytassist_rep2_radius55/baseline_comparison/baseline_comparison.csv`; `baseline_split_bootstrap_summary.csv`; `baseline_environment_audit.json`; per-method baseline directories | Complete. |
| 4. Ablations | full model and all requested module/modality removals | `results/nature_main/cytassist_rep2_radius55/ablations250/ablation_summary.csv` | Complete. |
| 5. Reliability/calibration | uncertainty-error correlation, risk coverage, calibration, gate maps, failure cases | `results/nature_main/cytassist_rep2_radius55/nature_analysis/` | Complete. |
| 6. Boundary preservation | tumor-stroma, ductal, immune-edge sharpness, smoothness comparison, H&E overlay, marker/pathology validation | `results/nature_main/cytassist_rep2_radius55/nature_analysis/`; `results/nature_external_pathology_validation/pathology_class_summary.csv` | Complete. |
| 7. Biological niche | composition, marker enrichment, H&E/pathology correspondence, Xenium neighborhood validation, gate/agent-by-niche | `results/nature_main/cytassist_rep2_radius55/nature_analysis/`; `results/nature_external_pathology_validation/pathology_niche_summary.csv` | Complete. |
| 8. External generalization | Rep1/Rep2 transfer, external 10x/Wu-Swarbrick datasets, no-retuning, minimal-retuning, and Rep1 budget curve summaries | `results/nature_external_no_retuning/`; `results/nature_external_matched_gt/`; `results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/rep1_retune_budget_curve/`; `results/nature_matched_multisample_baselines_minimal_retune/` | Complete. |
| 9. Robustness | patch size, gene panel/dropout, reference mismatch/removal/perturbation, H&E perturbation, low-quality spots, splits, radius/cell-count confidence | `results/nature_main/cytassist_rep2_radius55/robustness/`; `results/nature_main/cytassist_rep2_radius55/patch_size_robustness/`; `results/nature_main/cytassist_rep2_radius55/split_sensitivity/`; `results/nature_main/cytassist_rep2_radius55/benchmark_sensitivity/` | Complete. |

## Baseline Positioning

The formal comparison table currently includes WaveST-Gate, BayesPrism, RCTD
multi, SpatialDWLS/Seurat, CARD, reference cosine, reference NNLS,
standalone SpatialDWLS, Tangram with RNA-count prior, cell2location, Tangram
with uniform prior, uniform, and SPOTlight. The primary matched benchmark
ranks WaveST-Gate first by JSD, and the report includes runtime, memory, paired
spot-level significance, split summaries, and bootstrap statistics.

## DOI Handoff

The submission release is recorded as published on Zenodo with DOI
`10.5281/zenodo.20550855` and record URL
`https://zenodo.org/records/20550855`. The local final DOI gate is therefore
complete for submission. For a future versioned release, refresh the deposition
with:

```bash
ZENODO_ACCESS_TOKEN=<token> python -m wavestgate.evaluation.zenodo_deposit \
  --bundle-manifest results/nature_release/release_bundle_manifest.json

python -m wavestgate.evaluation.submission_readiness \
  --output-dir results/nature_submission_readiness
```

Use `--sandbox` first for API validation and `--publish` only after reviewing
the draft. The readiness audit marks the DOI stage complete when
`zenodo_deposition_result.json` and `release_bundle_manifest.json` contain an
explicit project release DOI/deposition id returned by Zenodo; the current
record satisfies that requirement.
