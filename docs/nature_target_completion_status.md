# WaveST-Gate Nature-Level Target Status

## Completed Evidence In This Workspace

### 1. Xenium-to-Visium Benchmark

- Typed Xenium Rep2 cell table: `data/processed/xenium_rep2_typed_cells/cells_he_aligned.csv`
- Cell type QC: `data/processed/xenium_rep2_typed_cells/qc.json`
- Spot-level counts/proportions/QC/splits/manifest:
  `data/processed/xenium_to_visium_benchmark/cytassist_rep2_radius55/`
- Benchmark datasheet/data dictionary:
  `results/nature_benchmark_datasheet/benchmark_datasheet.{json,md}`
- Requirement-by-requirement goal completion audit:
  `results/nature_completion_audit/goal_completion_audit.{json,md}`
- Prepared supervised batch:
  `data/processed/cytassist_xenium_rep2_common297/prepared_xenium_gt_radius55.pt`

### 2. Main WaveST-Gate Training

- CUDA/RTX 4090 training completed for 500 steps.
- Outputs: `results/nature_main/cytassist_rep2_radius55/`
- Supervised spots: `485`
- Main metrics: JSD `0.0129`, spotwise cosine `0.9933`,
  mean cell-type Pearson `0.9283`.
- Spatial cell-composition maps were generated in
  `results/nature_main/cytassist_rep2_radius55/nature_analysis/proportion_maps/`,
  including the top cell-type panel, tumor/immune/stromal group panel, and
  per-cell-type maps for 11 priority breast cancer cell types.
- Reliability: uncertainty-error Pearson `0.5263`; high-uncertainty JSD is
  higher than low-uncertainty JSD.
- A modality-dropout variant was also trained:
  `results/nature_main/cytassist_rep2_radius55_moddrop/`.
  It improves H&E gate participation while preserving strong accuracy
  (JSD `0.0187`, mean cell-type Pearson `0.9090`,
  uncertainty-error Pearson `0.5172`).

### 3. Baseline Comparison

- Lightweight fair baselines completed:
  `results/nature_main/cytassist_rep2_radius55/simple_baselines/`
- Methods: uniform, reference cosine, reference NNLS.
- Formal cell2location baseline completed in an isolated Python 3.10
  `cell2loc_env` environment with CUDA PyTorch, cell2location `0.1.5`,
  scvi-tools `1.3.3`, the same 297-gene panel, scFFPE reference, and
  supervised spot metrics:
  `results/nature_main/cytassist_rep2_radius55/cell2location_baseline/`
- Formal Tangram cluster-mode baseline completed with the same scFFPE
  reference, 297-gene panel, supervised spot set, and metrics:
  `results/nature_main/cytassist_rep2_radius55/tangram_baseline/`
- Tangram sensitivity run with uniform density prior completed:
  `results/nature_main/cytassist_rep2_radius55/tangram_baseline_uniform_prior/`
- Formal RCTD baseline completed through `spacexr` multi-mode after installing
  R and required R dependencies:
  `results/nature_main/cytassist_rep2_radius55/rctd_baseline_multi/`
- Formal CARD baseline completed through the R `CARD` package plus its
  `MuSiC` runtime dependency with the same spot expression, coordinates,
  scRNA reference, benchmark genes, supervised spots, and metrics:
  `results/nature_main/cytassist_rep2_radius55/card_baseline/`
- Formal SPOTlight baseline completed with SPOTlight `1.2.0`, seeded NMF,
  the same 297-gene panel, supervised spot metrics, and a fixed 500-cell
  per-cell-type reference subsample after a full-reference run exceeded the
  50GB RAM limit:
  `results/nature_main/cytassist_rep2_radius55/spotlight_baseline/`
- Formal BayesPrism baseline completed with BayesPrism `2.2.3`, the same
  297-gene panel, supervised spot set, and the same 27,472-cell scFFPE
  reference aggregated to GEP mode. The run disables BayesPrism's bulk
  outlier filter for spatial spots (`outlier_cut=1`, `outlier_fraction=1`),
  skips one zero-expression spot during inference, and realigns predictions
  to all 4,992 benchmark spots:
  `results/nature_main/cytassist_rep2_radius55/bayesprism_baseline/`
- Formal SpatialDWLS-compatible baseline completed with a standalone
  `Matrix`/`quadprog` runner following the Giotto `runDWLSDeconv` DWLS
  formulation: scFFPE top-marker signature matrix, first-pass DWLS cell-type
  screening at `1/n_cell`, second-pass dampened weighted least squares, and
  the same 297-gene panel/supervised spot set:
  `results/nature_main/cytassist_rep2_radius55/spatialdwls_baseline/`
- Full Giotto/Seurat package-stack SpatialDWLS rerun completed through
  `Giotto::runDWLSDeconv` on the R4.1-compatible Giotto branch. It produced
  predictions for all 4,992 benchmark spots and evaluated the same 485
  Xenium-supervised spots:
  `results/nature_main/cytassist_rep2_radius55/spatialdwls_giotto_baseline/`
- Unified comparison table with paired per-spot JSD permutation tests:
  `results/nature_main/cytassist_rep2_radius55/baseline_comparison/baseline_comparison.csv`
- Split-wise and paired bootstrap mean/std statistics completed:
  `baseline_split_bootstrap_summary.csv`,
  `baseline_split_bootstrap_metrics.csv`, and
  `baseline_bootstrap_paired_improvement.csv`. The current 500-bootstrap
  summary keeps WaveST-Gate ranked first by mean JSD.
- Independent matched-GT multi-sample baseline summary completed across
  CytAssist/Rep2 and Rep1 pseudo-Visium:
  `results/nature_matched_multisample_baselines/matched_multisample_baseline_summary.csv`.
  WaveST-Gate ranks first by two-dataset mean JSD (`0.1364 +/- 0.1747`),
  followed by RCTD multi (`0.1827 +/- 0.0871`). The paired dataset table also
  records the key caveat that RCTD beats no-retuning WaveST-Gate on Rep1 alone.
- A second independent matched-GT summary tracks the minimal-retuning transfer
  setting:
  `results/nature_matched_multisample_baselines_minimal_retune/matched_multisample_baseline_summary.csv`.
  In this setting, Rep2-initialized WaveST-Gate is retuned only on Rep1
  train/val pseudo-spots and evaluated on held-out Rep1 test spots; WaveST-Gate
  ranks first with two-dataset mean JSD `0.0257 +/- 0.0181`.
- Current comparison: WaveST-Gate JSD `0.0129`; BayesPrism JSD `0.2377`;
  RCTD multi JSD `0.2443`; SpatialDWLS/Seurat JSD `0.2524`; CARD JSD
  `0.2726`; reference cosine JSD `0.2848`; standalone SpatialDWLS JSD
  `0.3128`; Tangram rna-count prior JSD `0.3604`; cell2location JSD
  `0.3677`; SPOTlight JSD `0.5691`.
  All completed baselines show paired JSD improvement versus WaveST-Gate with
  permutation p value `0.0001`.
- Baseline environment audit:
  `results/nature_main/cytassist_rep2_radius55/baseline_environment_audit.json`
  shows Tangram/AnnData/Scanpy, external cell2location/scvi/CUDA PyTorch,
  RCTD/spacexr, CARD/MuSiC, SPOTlight, BayesPrism, standalone SpatialDWLS,
  and full SpatialDWLS/Seurat are ready. The Giotto rerun used Seurat `5.5.0`,
  Giotto `4.2.3`, GiottoClass `0.5.1`, GiottoUtils `0.2.5`, GiottoVisuals
  `0.2.15`, Rfast `2.1.5.2`, and quadprog `1.5.8`.

### 4. Ablations

- 12 ablations completed at 250 steps:
  `results/nature_main/cytassist_rep2_radius55/ablations250/`
- Covered: full, CNN replacement, no image, no agents, no gate, raw gate
  without uncertainty, no boundary loss, normal smoothness only, no local
  refinement, expression-only, image-only, reference-only.

### 5. Reliability / Calibration

- Outputs: `results/nature_main/cytassist_rep2_radius55/nature_analysis/`
- Includes reliability spot errors, risk-coverage curve, uncertainty map,
  image/expression/reference gate maps.
- Added calibration-grade outputs: `uncertainty_calibration_bins.csv`,
  `uncertainty_calibration.png`, `risk_coverage_curve.png`, and
  `failure_case_candidates.csv`.
- Current reliability evidence: uncertainty-error Pearson `0.5263`,
  Spearman `0.5986`; high-vs-low uncertainty JSD permutation p value
  `0.0001`; calibration-bin Pearson `0.9619`.

### 6. Boundary Preservation

- Boundary edge analysis:
  `results/nature_main/cytassist_rep2_radius55/nature_analysis/boundary_edge_jumps.csv`
- Full model boundary-to-interior jump ratio: `2.36`
- Comparison against no-boundary-loss ablation is included.
- Added typed boundary summaries and marker validation:
  `boundary_type_summary.csv`, `boundary_marker_validation.csv`, and
  `boundary_sharpness_map.png`.
- Added H&E patch-color overlay and image/pathology proxy table:
  `boundary_he_overlay.png` and `boundary_he_pathology_proxy.csv`.
- Boundary categories include tumor-stroma boundary, ductal boundary, immune
  infiltration edge, immune-stromal edge, and within-compartment edges.
- External pathology metadata validation completed on six Wu/Swarbrick Visium
  samples:
  `results/nature_external_pathology_validation/pathology_class_summary.csv`.
  The validation compares no-retuning predicted tumor/immune/stromal
  composition with public pathology `Classification` labels across 15,601
  spots.

### 7. Biological Niches

- Niche assignments, composition, gate reliability by niche, agent attention by
  niche, and marker enrichment:
  `results/nature_main/cytassist_rep2_radius55/nature_analysis/`
- Niche map: `niche_map.png`
- Added named biological niche summary:
  `niche_biological_summary.csv`, with labels such as
  HER2/tumor-associated niche, stromal remodeling niche, and
  macrophage-rich immune niche.
- Added explicit Xenium neighborhood validation:
  `niche_xenium_neighborhood_validation.csv` and
  `niche_xenium_neighborhood_summary.csv`. The summary reports
  tumor/immune/stromal neighborhood agreement rates by niche.
- Added external pathology-niche correspondence:
  `results/nature_external_pathology_validation/pathology_niche_summary.csv`
  and `pathology_niche_by_class.csv`, with a pathology/niche heatmap.

### 8. External Generalization

- No-retuning predictions completed on 10 external datasets:
  `results/nature_external_no_retuning/`
- Includes 4 external 10x Visium samples and 6 Wu/Swarbrick breast cancer
  Visium samples.
- Summary: `results/nature_external_no_retuning/external_no_retuning_summary.csv`
- External matched-GT evaluation completed on Rep1 Xenium pseudo-spots:
  `results/nature_external_matched_gt/external_matched_gt_summary.csv`.
  The Rep1 Xenium cell-feature matrix and typed cells were aggregated into
  1,146 pseudo-Visium spots with 297 genes and 19 cell types, then evaluated
  no-retuning using the Rep2-trained WaveST-Gate checkpoint.
- Rep1 matched-GT result: WaveST-Gate JSD `0.2599`, spotwise cosine `0.5653`,
  mean cell-type Pearson `0.2700`. Against formal baselines, WaveST-Gate ranks
  2 by JSD; RCTD multi is best on Rep1 with JSD `0.1211`. WaveST-Gate remains
  slightly better than CARD (`0.2623`), Tangram (`0.2667`), cell2location
  (`0.2758`), reference cosine (`0.3135`), and SpatialDWLS (`0.3260`).
- Minimal-retuning Rep1 adaptation completed:
  `results/nature_external_matched_gt/xenium_rep1_pseudospots_radius55_common297/wavestgate_minimal_retune/`.
  The Rep2 checkpoint was initialized and retuned on 975 Rep1 train/val
  pseudo-spots, then evaluated on 171 held-out Rep1 test pseudo-spots. Test JSD
  is `0.0385`, spotwise cosine `0.9605`, and WaveST-Gate ranks 1 versus RCTD
  multi (`0.1782`), cell2location (`0.2344`), Tangram (`0.2498`), CARD
  (`0.2735`), reference cosine (`0.2908`), and SpatialDWLS (`0.3909`), with
  paired JSD permutation p value `0.0001` against each baseline.

### 9. Robustness

- Stress tests completed:
  `results/nature_main/cytassist_rep2_radius55/robustness/robustness_summary.csv`
- Covered: gene dropout 10/30/50%, gene panels top200/top100/marker-only,
  reference missing cell types, prototype noise/dropout, H&E brightness/noise
  perturbation, low/high expression subgroup, low/high Xenium cell-count
  subgroup, manifest splits, and random holdout splits.
- Patch-size robustness completed for 32/64/128/256:
  `results/nature_main/cytassist_rep2_radius55/patch_size_robustness/patch_size_summary.csv`
- A local-refinement OOM at large patch sizes was fixed by pooled
  cross-attention.

### Submission Readiness Audit

- A machine-checkable readiness audit is generated by:
  `python -m wavestgate.evaluation.submission_readiness --output-dir results/nature_submission_readiness`
- The Xenium-to-Visium benchmark datasheet/data dictionary is generated by:
  `python -m wavestgate.evaluation.benchmark_datasheet --output-dir results/nature_benchmark_datasheet`
- The full goal completion audit is generated by:
  `python -m wavestgate.evaluation.completion_audit --output-dir results/nature_completion_audit`
- Manuscript-ready summary tables are generated by:
  `python -m wavestgate.evaluation.manuscript_tables --output-dir results/nature_manuscript_tables`
  and currently include seven CSV tables plus `manuscript_tables.md`:
  benchmark summary, main model performance, formal baseline comparison,
  ablation deltas, reliability/boundary/niche validation, external
  generalization, and robustness.
- Manuscript figure assets are generated by:
  `python -m wavestgate.evaluation.manuscript_figures --output-dir results/nature_manuscript_figures`
  and currently include Figure 1-5 PNGs, Supplementary Figure S1, and
  `figure_manifest.{csv,json,md}` with dimensions, SHA256, and nonblank
  validation.
- Manuscript figure legends and claim evidence are generated by:
  `python -m wavestgate.evaluation.manuscript_figure_legends --output-dir results/nature_manuscript_figure_legends`
  and link Figure 1-5 plus Supplementary Figure S1 to key claims, values,
  and evidence files.
- Manuscript Methods/statistical-analysis draft material is generated by:
  `python -m wavestgate.evaluation.manuscript_methods --output-dir results/nature_manuscript_methods`
  and covers benchmark construction, model specification, training losses,
  baseline fairness, ablations, reliability/calibration, boundary
  preservation, niche interpretation, external generalization, robustness,
  statistical analysis, compute, and reproducibility evidence.
- Reviewer preflight and response evidence is generated by:
  `python -m wavestgate.evaluation.reviewer_preflight --output-dir results/nature_reviewer_preflight`
  and links headline claims, likely reviewer concerns, preemptive responses,
  stage status, and evidence paths. The current release records explicit
  Zenodo DOI fields.
- The local finalization chain is generated by:
  `python -m wavestgate.evaluation.finalize_submission`. It refreshes
  the software/hardware environment report, benchmark datasheet/data
  dictionary, requirement completion audit, manuscript tables, Figure 1-5
  assets, figure legends/claim evidence, manuscript
  availability/reproducibility statements, Methods/statistical-analysis draft
  material, reviewer preflight dossier, the Zenodo-ready release bundle,
  Zenodo handoff, release verification, readiness reports, and
  `results/nature_release/final_submission_handoff.{json,md}`.
- Current outputs:
  `results/nature_submission_readiness/readiness_report.json`,
  `results/nature_submission_readiness/readiness_report.md`,
  `results/nature_submission_readiness/evidence_manifest.csv`,
  `results/nature_submission_readiness/release_file_checksums.csv`, and
  `results/nature_submission_readiness/zenodo_release_manifest.json`.
- Current audited overall status is `complete`. The readiness report records
  all nine stages as complete, and the final DOI gate records the project as
  complete for submission.
- Zenodo-ready release assets are prepared in `results/nature_release/`:
  `wavestgate_submission_evidence_v0.1.0.tar.gz`,
  `zenodo_metadata.json`, `release_upload_manifest.csv`,
  `release_bundle_manifest.json`, and
  `zenodo_deposition_instructions.md`. The deposition helper
  `python -m wavestgate.evaluation.zenodo_deposit` performs dry-run,
  sandbox, production draft, DOI reservation, upload, and optional publish
  flows. The release manifest also records manuscript-critical artifacts,
  including the main checkpoint, predicted proportions, reconstructed
  expression, gate weights, uncertainty, agent attention, and key
  proportion/reliability/boundary/niche summaries. The current bundle file
  count, byte size, and SHA256 are recorded in
  `results/nature_release/release_bundle_manifest.json` after each refresh.
  Bundle integrity can be independently checked with
  `python -m wavestgate.evaluation.verify_release`, which writes
  `results/nature_release/release_verification.json` and
  `results/nature_release/release_verification.md`.
  Current published deposition output is
  `results/nature_release/zenodo_deposition_result.json`, with Zenodo
  deposition id `20550855`, DOI `10.5281/zenodo.20550855`, and record URL
  `https://zenodo.org/records/20550855`.
- The audit keeps submission-critical items visible through
  `results/nature_submission_readiness/readiness_report.json` and
  `results/nature_final_doi_gate/final_doi_gate.json`; both currently report a
  complete submission state.

## Remaining For A True Nature-Level Submission

- Keep manuscript text, figure legends, and data/code availability statements
  aligned with Zenodo DOI `10.5281/zenodo.20550855`.
- Optional: expand matched single-cell ground truth beyond the current
  Rep2/Rep1 matched-GT evidence if additional paired data are obtained.
