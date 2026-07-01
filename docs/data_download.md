# WaveST-Gate Data Download Notes

The primary benchmark data source is the public 10x Genomics Janesick et al.
human breast cancer Xenium/Visium preview dataset plus GEO supplementary files
from the Janesick SuperSeries `GSE243280`.

## Manifest

All known public download targets are listed in:

```bash
data_manifest/wavestgate_breast_core.yaml
```

Additional independent-test and generalization datasets are listed in:

```bash
data_manifest/wavestgate_submission_breast_external.yaml
```

The Wu/Swarbrick six-sample breast cancer Visium atlas is listed in:

```bash
data_manifest/wavestgate_wu_swarbrick_visium.yaml
```

The manifest includes:

- Xenium FFPE Human Breast Cancer Rep1 output bundle.
- Xenium FFPE Human Breast Cancer Rep2 output bundle for same-cohort hold-out
  validation.
- 10x/GEO cell-type annotation workbook.
- CytAssist FFPE Human Breast Cancer Visium h5 and spatial bundle.
- Xenium metadata, panel, gene panel, H&E/IF/alignment files.
- GEO supplementary raw files for the public Janesick breast cancer series:
  `GSE243168` for Xenium, `GSE243275` for scRNA-seq/Visium, and the
  `GSE243280` SuperSeries package.

Neighboring accessions such as `GSE243281`, `GSE243282`, and `GSE243283` are
not used by the breast benchmark. They are outside the Janesick breast cancer
SuperSeries scope or unavailable, so the submission data package has no private
GEO dependency.

## Download

Download all manifest items:

```bash
python -m wavestgate.data.download \
  --manifest data_manifest/wavestgate_breast_core.yaml
```

Download additional submission datasets:

```bash
python -m wavestgate.data.download \
  --manifest data_manifest/wavestgate_submission_breast_external.yaml
```

Download the Wu/Swarbrick breast cancer Visium atlas:

```bash
python -m wavestgate.data.download \
  --manifest data_manifest/wavestgate_wu_swarbrick_visium.yaml
```

Download only required primary benchmark items:

```bash
python -m wavestgate.data.download \
  --manifest data_manifest/wavestgate_breast_core.yaml \
  --required-only
```

Download selected items:

```bash
python -m wavestgate.data.download \
  --manifest data_manifest/wavestgate_breast_core.yaml \
  --id xenium_rep1_outs \
  --id xenium_rep2_outs \
  --id geo_gse243275_raw
```

Verify selected files without downloading:

```bash
python -m wavestgate.data.download \
  --manifest data_manifest/wavestgate_breast_core.yaml \
  --verify-only
```

Verify the additional submission datasets:

```bash
python -m wavestgate.data.download \
  --manifest data_manifest/wavestgate_submission_breast_external.yaml \
  --verify-only
```

Verify the Wu/Swarbrick atlas:

```bash
python -m wavestgate.data.download \
  --manifest data_manifest/wavestgate_wu_swarbrick_visium.yaml \
  --verify-only
```

The downloader uses `aria2c` when available and falls back to `wget --continue`
or resumable `curl`, so interrupted downloads can be resumed by rerunning the
same command.

Downloaded files are stored under:

```bash
data/raw/wavestgate_breast_core/
```

Additional submission files are stored under:

```bash
data/raw/wavestgate_submission_breast_external/
```

Wu/Swarbrick atlas files are stored under:

```bash
data/raw/wavestgate_wu_swarbrick_visium/
```

Convert the downloaded Wu/Swarbrick atlas into WaveST-Gate standard files and
model-ready batches:

```bash
python -m wavestgate.data.prepare_wu_swarbrick_visium \
  --genes data/processed/gene_panels/xenium_breast_panel_common_visium_scffpe.txt \
  --allow-missing-genes \
  --reference-prototypes-path data/processed/scffpe_reference_xenium_common301/reference_prototypes.csv.gz \
  --prepare \
  --patch-size 32 \
  --graph-k 6
```

The converted Wu/Swarbrick sample batches are stored under:

```bash
data/processed/wu_swarbrick_*_xenium_common301/
```

The conversion summary is stored at:

```bash
data/processed/wu_swarbrick_xenium_common301_summary.csv
```

The downloader writes a run status file at:

```bash
data/raw/wavestgate_breast_core/download_status.json
```

Verification writes:

```bash
data/raw/wavestgate_breast_core/verify_status.json
```

Each manifest root also receives its own `download_status.json` and
`verify_status.json`.
