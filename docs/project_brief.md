# WaveST-Gate Project Brief

WaveST-Gate is a reliability-aware, morphology-guided spatial transcriptomics
deconvolution framework. The current implementation turns the MVP into a
submission-oriented method scaffold with uncertainty, boundary-aware spatial
regularization, Xenium-to-Visium benchmarking, and biological niche outputs.

## Current milestone

- Build a clean `wavestgate` package.
- Keep the original CV module archive untouched.
- Reimplement the core ideas without heavy CV-only dependencies.
- Provide synthetic data tests and a toy training CLI.
- Provide lightweight real-data interfaces for spot expression, scRNA reference
  prototypes, H&E patches, spatial graphs, and Xenium-derived proportions.
- Provide CLI entrypoints to prepare a model-ready dataset bundle and train from
  that prepared bundle.
- Estimate spot-level uncertainty and modality-specific uncertainty/reliability
  so gate weights can be evaluated as calibrated evidence rather than generic
  feature fusion.
- Add morphology-aware boundary-preserving spatial regularization that smooths
  cell-type proportions across similar histology while reducing smoothing across
  putative tissue boundaries.
- Provide a reproducible Xenium-to-Visium benchmark builder with spot-level
  counts/proportions, spatial holdout splits, QC, metrics, and a manifest.
- Support optional biological niche supervision and niche-level interpretation
  summaries for cell-type composition and modality reliability.

## Module mapping

- `132a. DW-GCA.py` maps to a pure-torch wavelet morphology encoder.
- `170a. SDWA.py` maps to directional wavelet attention without
  `pytorch_wavelets`.
- `120a. XMSGF.py` maps to a cross-modal reliability gate.
- Reliability modeling extends the gate with uncertainty calibration so the
  model can report when H&E, ST expression, or scRNA reference evidence is weak.
- `24a...DAEM.py` maps to scRNA-initialized cell-type prototype agents.
- `50a...py` maps to local cross-modal refinement without `ultralytics`.

## Future milestones

- Add AnnData/Scanpy adapters as optional bio extras.
- Run baseline methods and ablations on the Xenium-to-Visium benchmark.
- Produce Bioinformatics-style manuscript figures, calibration plots,
  morphology-boundary maps, niche case studies, and supplementary material.
