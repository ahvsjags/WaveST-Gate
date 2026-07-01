# Zenodo Deposition Instructions

This workspace has prepared a Zenodo-ready metadata file and evidence bundle.

- Metadata: `results/nature_release/zenodo_metadata.json`
- Bundle: `results/nature_release/wavestgate_submission_evidence_v0.1.0.tar.gz`
- Upload manifest: `results/nature_release/release_upload_manifest.csv`
- Zenodo token present in environment: `False`

A machine-executable deposition helper is available:

```bash
python -m wavestgate.evaluation.zenodo_deposit --dry-run
ZENODO_ACCESS_TOKEN=<token> python -m wavestgate.evaluation.zenodo_deposit --sandbox
ZENODO_ACCESS_TOKEN=<token> python -m wavestgate.evaluation.zenodo_deposit
ZENODO_ACCESS_TOKEN=<token> python -m wavestgate.evaluation.zenodo_deposit --publish
```

Use `--sandbox` first to validate the API flow. Omit `--sandbox` for the production draft. Use `--publish` only after reviewing the draft, because publishing registers the DOI and makes the record public. The helper writes `zenodo_deposition_result.json` and updates `release_bundle_manifest.json` with the deposition id, DOI, and record URL when Zenodo returns them.

The bundle intentionally excludes raw public data and unlisted large binaries. Explicit manuscript-critical artifacts, including the main WaveST-Gate checkpoint, are bundled when listed in the release manifest. Public raw data are tracked by manifests and source accessions; reproducible benchmark tables and result evidence are included.