# WaveST-Gate Release Verification

Generated UTC: 2026-06-05T06:11:32.990592+00:00

- Overall status: `passed`
- Bundle integrity: `passed`
- DOI status: `published`
- Bundle: `results/nature_release/wavestgate_submission_evidence_v0.1.0.tar.gz`
- SHA256: `dd849c78a8b90d9ae8f5d1a9f470ee3c27dd80e23117db97468ae38d8426b24f`

| Check | Status | Detail | Metric |
| --- | --- | --- | --- |
| bundle_manifest_exists | `pass` | Read results/nature_release/release_bundle_manifest.json. |  |
| bundle_exists | `pass` | Found results/nature_release/wavestgate_submission_evidence_v0.1.0.tar.gz. |  |
| bundle_bytes_match | `pass` | Bundle byte size matches manifest. | expected=104284966; actual=104284966 |
| bundle_sha256_match | `pass` | Bundle SHA256 matches manifest. | expected=dd849c78a8b90d9ae8f5d1a9f470ee3c27dd80e23117db97468ae38d8426b24f; actual=dd849c78a8b90d9ae8f5d1a9f470ee3c27dd80e23117db97468ae38d8426b24f |
| tar_readable | `pass` | Bundle tarball is readable. | members=617 |
| upload_manifest_in_bundle | `pass` | Upload manifest is present in the bundle. | results/nature_release/release_upload_manifest.csv |
| zenodo_metadata_in_bundle | `pass` | Zenodo metadata is present in the bundle. | results/nature_release/zenodo_metadata.json |
| upload_manifest_row_count | `pass` | Upload manifest row count matches release manifest. | expected=615; actual=615 |
| upload_manifest_members_present | `pass` | All upload manifest files are present in the tarball. | missing=0 |
| upload_manifest_member_hashes | `pass` | All upload manifest member byte sizes and SHA256 hashes match. | mismatch=0 |
| critical_artifacts_present | `pass` | All critical artifacts are present and marked bundled. | missing=0 |
| critical_artifact_hashes | `pass` | All critical artifact byte sizes and SHA256 hashes match tar members. | mismatch=0 |
| deposition_result_exists | `pass` | Read results/nature_release/zenodo_deposition_result.json. |  |
| deposition_bundle_matches_manifest | `pass` | Deposition dry-run/result references the same bundle path, byte size, and SHA256. | release_status=zenodo_published |
| zenodo_doi_recorded | `pass` | Zenodo DOI and deposition id are recorded. | release_status=zenodo_published; doi=10.5281/zenodo.20550855; zenodo_deposition_id=20550855 |
| zenodo_release_published | `pass` | Zenodo release is published and publicly accessible. | release_status=zenodo_published |