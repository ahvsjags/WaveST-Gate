# WaveST-Gate Final DOI Gate

Generated UTC: 2026-06-10T08:41:46.056542+00:00

Status: `complete`
Ready to deposit: `False`
Complete for submission: `True`

## Checks

- `release_bundle_integrity`: `pass` - bundle_integrity_status=passed; num_failures=0
- `critical_artifacts_bundled`: `pass` - missing=0; unbundled=0
- `completion_audit_local_requirements`: `pass` - overall_status=complete; num_missing=0
- `readiness_no_missing_records`: `pass` - overall_status=complete
- `zenodo_token_present`: `warn` - A token is needed only when creating/updating the real deposition.
- `project_doi_recorded`: `pass` - doi=10.5281/zenodo.20550855; zenodo_deposition_id=20550855
- `project_doi_published`: `pass` - release_status=zenodo_published; doi=10.5281/zenodo.20550855; zenodo_deposition_id=20550855
- `release_verification_doi_recorded`: `pass` - doi_status=published
- `release_verification_doi_published`: `pass` - doi_status=published

## Next Commands

```bash
python -m wavestgate.evaluation.final_doi_gate --strict
```

```bash
export ZENODO_ACCESS_TOKEN=<token>
```

```bash
ZENODO_ACCESS_TOKEN=<token> python -m wavestgate.evaluation.finalize_submission --deposit
```

```bash
python -m wavestgate.evaluation.final_doi_gate --strict
```
