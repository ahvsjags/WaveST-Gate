import json
from pathlib import Path

from wavestgate.evaluation.environment_report import collect_environment_report


def test_collect_environment_report_writes_json_and_markdown(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline_environment_audit.json"
    baseline.write_text(json.dumps({"status": {"Tangram": "ready"}}), encoding="utf-8")

    report = collect_environment_report(
        output_path=tmp_path / "environment_report.json",
        markdown_path=tmp_path / "environment_report.md",
        baseline_environment_path=baseline,
    )

    assert report["python"]["executable"]
    assert "torch" in report["packages"]
    assert "cuda_available" in report["torch"]
    assert report["baseline_environment_audit"]["status"]["Tangram"] == "ready"
    assert (tmp_path / "environment_report.json").exists()
    assert (tmp_path / "environment_report.md").exists()
