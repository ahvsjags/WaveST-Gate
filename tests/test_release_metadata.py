import json
from pathlib import Path

import yaml

from wavestgate.evaluation.prepare_release import CRITICAL_ARTIFACT_PATHS, DEFAULT_INCLUDE_PATHS


def test_release_metadata_files_are_valid() -> None:
    root = Path.cwd()
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    codemeta = json.loads((root / "codemeta.json").read_text(encoding="utf-8"))
    license_text = (root / "LICENSE").read_text(encoding="utf-8")

    assert citation["cff-version"] == "1.2.0"
    assert citation["title"].startswith("WaveST-Gate")
    assert citation["version"] == "0.1.0"
    assert citation["doi"] == "10.5281/zenodo.20550855"
    assert citation["url"] == "https://zenodo.org/records/20550855"
    assert codemeta["@type"] == "SoftwareSourceCode"
    assert codemeta["name"] == "WaveST-Gate"
    assert codemeta["identifier"] == "https://doi.org/10.5281/zenodo.20550855"
    assert codemeta["codeRepository"] == "https://zenodo.org/records/20550855"
    assert codemeta["license"].endswith("/MIT.html")
    assert "MIT License" in license_text


def test_release_metadata_is_bundled_and_critical() -> None:
    for path in ["LICENSE", "CITATION.cff", "codemeta.json"]:
        assert path in DEFAULT_INCLUDE_PATHS
        assert path in CRITICAL_ARTIFACT_PATHS
