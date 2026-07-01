# WaveST-Gate Supporting Package QA Report

## Scope

This report summarizes the generated supporting materials for the manuscript planning target:

- Supporting Figures S1-S15.
- Supplementary Tables S1-S19.
- Top-tier journal style: complete evidence chain, source traceability, editable vector outputs and more diverse visualization forms.

## File Integrity

- Figures: 15 PNG, 15 TIFF, 15 PDF and 15 SVG files.
- Tables: 19 CSV and 19 Markdown files.
- PDF check: all 15 figure PDFs opened as single-page figure files.
- Raster check: all PNG and TIFF files opened successfully.
- SVG check: all SVG files contain editable text objects.
- Missing-data check: `missing_or_server_needed.csv` contains only the header, so this round did not require server-side file recovery.

## Visual Style Revision

- Replaced colored rounded panel labels with restrained black lowercase panel labels.
- Reduced large in-figure headings to small, unobtrusive figure identifiers.
- Lowered color saturation across the suite to avoid a synthetic report-like appearance.
- Reworked S1 workflow schematic from crowded horizontal boxes into a quieter vertical thin-line schematic.
- Rebuilt S3 directly from source coordinates and predictions instead of embedding pre-rendered PNG maps, improving clarity and avoiding raster blur.
- Rebuilt S4 baseline fairness panels into a more evidence-oriented package with runtime-accuracy Pareto view, paired-bootstrap forest plot and bootstrap stability intervals.
- Rebuilt S5 from summary bars/table into an audit-style figure: audited prediction-error ranking, label-permutation negative control, leakage/fairness verdict matrix and prediction-file integrity matrix.
- Rebuilt S13 pathology composition from tabular source data instead of relying on a dense pre-rendered image.
- Rebuilt S14 from a simple release workflow into a reproducibility evidence figure: release manifest size spectrum, release verification matrix, checksum-covered core artifact spectrum and release/environment identity table.
- Fixed S2 coverage plotting by using the true coverage column in the radius summary table.
- Rebuilt S7 from a simple summary-bar layout into a six-panel per-spot paired-control figure: spatial paired gain, paired spot-level error, paired gain distribution, texture-stratified gain, gate-texture response and run-level control profile.
- Rebuilt S15 from simple runtime/memory bar charts into a methods-style compute figure: runtime-accuracy envelope, memory-runtime profile, training loss components and compact compute context.

## Figure Logic

- S1: Benchmark construction and QC, including aggregation logic, supervised spot coverage, split balance, cell-count QC and typed Xenium inventory.
- S2: Aggregation sensitivity across radius and minimum-cell thresholds, linking coverage and performance.
- S3: Full 19-cell-type spatial atlas from predicted proportion maps.
- S4: Baseline fairness package for matched spots, genes, references, runtime and paired improvement evidence.
- S5: Leakage and negative controls, including audited prediction error, shuffled-label control, leakage/fairness verdicts and prediction-file integrity.
- S6: Ablation mechanism wall for wavelet, agent, gate, uncertainty, boundary and refinement components.
- S7: H&E morphology contribution using imagegate-enhanced/no-image controls and texture-stratified paired error.
- S8: Reliability calibration details through uncertainty-error scatter, calibration bins, risk coverage and failure candidates.
- S9: Boundary preservation details using boundary maps, H&E overlay, typed transitions and marker validation.
- S10: Niche biological validation using spatial niche map, cell-type composition heatmap, marker dot plot, Xenium neighborhood agreement and gate/agent profile.
- S11: External generalization using no-retuning transfer, Rep1 matched-GT adaptation, minimal-retuning curve and multi-sample forest plot.
- S12: Robustness stress tests using perturbation JSD strip plot, patch-size curve, split sensitivity and radius/cell-count heatmap.
- S13: External pathology validation using pathology-niche heatmap, agreement by class, pathology group composition and patient-level summary.
- S14: Reproducibility/release map for source data, code, Zenodo DOI, release verification, checksum coverage and software environment.
- S15: Compute/scalability summary for runtime, memory, training trace and hardware/software context.

## Table Logic

- Tables S1-S4: data, benchmark inventory, cell-type mapping and model parameters.
- Tables S5-S8: main model metrics, baseline configuration, full baseline results and ablation results.
- Tables S9-S12: reliability, H&E contribution, boundary validation and niche validation.
- Tables S13-S16: external generalization, Rep1 retuning budget, robustness and radius/cell-count sensitivity.
- Tables S17-S19: leakage/fairness audit, source-data/release/checksum records and compute/hardware/software evidence.

## Notes For Manuscript Writing

- The main text can use S1-S5 to defend benchmark fairness and data construction before presenting claims of superiority.
- S6-S10 should be cited when explaining mechanism: which model components matter, when H&E helps, why uncertainty is meaningful, how boundaries are preserved and how niches align biologically.
- S11-S13 should support generalization and translational relevance.
- S14-S15 should be used in Methods, Data Availability, Code Availability and reviewer-facing reproducibility statements.
