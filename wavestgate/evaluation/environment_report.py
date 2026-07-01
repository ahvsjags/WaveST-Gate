"""Collect software and hardware environment metadata for reproducibility."""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CORE_PACKAGES = [
    "wavestgate",
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "scikit-learn",
    "pillow",
    "pyyaml",
    "tqdm",
    "scipy",
    "h5py",
    "anndata",
    "scanpy",
    "tangram-sc",
]

R_PACKAGES = [
    "Seurat",
    "Giotto",
    "GiottoClass",
    "GiottoUtils",
    "GiottoVisuals",
    "Rfast",
    "quadprog",
    "spacexr",
    "CARD",
    "MuSiC",
    "BayesPrism",
    "SPOTlight",
]


def _run(command: list[str], timeout: int = 15) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command, "stdout": "", "stderr": "", "returncode": None}
    completed = subprocess.run([executable, *command[1:]], capture_output=True, text=True, timeout=timeout, check=False)
    return {
        "available": True,
        "command": [executable, *command[1:]],
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "returncode": completed.returncode,
    }


def _package_versions(packages: list[str]) -> dict[str, dict[str, Any]]:
    versions: dict[str, dict[str, Any]] = {}
    for package in packages:
        try:
            versions[package] = {"available": True, "version": importlib_metadata.version(package)}
        except importlib_metadata.PackageNotFoundError:
            versions[package] = {"available": False, "version": ""}
    return versions


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - only hit if torch import is broken
        return {"available": False, "error": repr(exc)}
    cuda_available = bool(torch.cuda.is_available())
    devices = []
    if cuda_available:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_mb": int(props.total_memory / (1024**2)),
                    "major": int(props.major),
                    "minor": int(props.minor),
                }
            )
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
        "device_count": len(devices),
        "devices": devices,
    }


def _r_package_versions(packages: list[str]) -> dict[str, Any]:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return {"rscript": "", "packages": {package: {"available": False, "version": ""} for package in packages}}
    expr = """
packages <- strsplit(Sys.getenv("WAVESTGATE_R_PACKAGES"), ",", fixed=TRUE)[[1]]
for (pkg in packages) {
  if (requireNamespace(pkg, quietly=TRUE)) {
    cat(pkg, as.character(utils::packageVersion(pkg)), "\\n", sep="\\t")
  } else {
    cat(pkg, "", "\\n", sep="\\t")
  }
}
"""
    completed = subprocess.run(
        [rscript, "-e", expr],
        env={**os.environ, "WAVESTGATE_R_PACKAGES": ",".join(packages)},
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    parsed = {package: {"available": False, "version": ""} for package in packages}
    for line in completed.stdout.splitlines():
        parts = line.split("\t")
        if not parts:
            continue
        package = parts[0]
        version = parts[1] if len(parts) > 1 else ""
        if package in parsed:
            parsed[package] = {"available": bool(version), "version": version}
    rscript_version = _run(["Rscript", "--version"])
    return {
        "rscript": rscript,
        "rscript_version": rscript_version.get("stderr") or rscript_version.get("stdout", ""),
        "returncode": completed.returncode,
        "packages": parsed,
    }


def collect_environment_report(
    output_path: str | Path = "results/nature_release/environment_report.json",
    markdown_path: str | Path = "results/nature_release/environment_report.md",
    baseline_environment_path: str | Path = "results/nature_main/cytassist_rep2_radius55/baseline_environment_audit.json",
) -> dict[str, Any]:
    output_path = Path(output_path)
    markdown_path = Path(markdown_path)
    baseline_environment_path = Path(baseline_environment_path)
    baseline_environment = {}
    if baseline_environment_path.exists():
        baseline_environment = json.loads(baseline_environment_path.read_text(encoding="utf-8"))
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": _package_versions(CORE_PACKAGES),
        "torch": _torch_info(),
        "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "r": _r_package_versions(R_PACKAGES),
        "baseline_environment_audit": {
            "path": str(baseline_environment_path),
            "status": baseline_environment.get("status", {}),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# WaveST-Gate Environment Report",
        "",
        f"Generated UTC: {report['generated_at_utc']}",
        "",
        f"- Python: `{report['python']['version'].split()[0]}`",
        f"- Platform: `{report['platform']['system']} {report['platform']['release']} {report['platform']['machine']}`",
        f"- Torch: `{report['torch'].get('version', '')}`",
        f"- CUDA available: `{report['torch'].get('cuda_available', False)}`",
        f"- CUDA version: `{report['torch'].get('cuda_version', '')}`",
        f"- GPU count: `{report['torch'].get('device_count', 0)}`",
        f"- Rscript: `{report['r'].get('rscript_version', '')}`",
        "",
        "## Python Packages",
        "",
        "| Package | Available | Version |",
        "| --- | --- | --- |",
    ]
    for package, info in report["packages"].items():
        lines.append(f"| `{package}` | `{info['available']}` | `{info['version']}` |")
    lines.extend(["", "## R Packages", "", "| Package | Available | Version |", "| --- | --- | --- |"])
    for package, info in report["r"]["packages"].items():
        lines.append(f"| `{package}` | `{info['available']}` | `{info['version']}` |")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect WaveST-Gate software/hardware environment metadata.")
    parser.add_argument("--output", default="results/nature_release/environment_report.json")
    parser.add_argument("--markdown-output", default="results/nature_release/environment_report.md")
    parser.add_argument("--baseline-environment", default="results/nature_main/cytassist_rep2_radius55/baseline_environment_audit.json")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    report = collect_environment_report(
        output_path=args.output,
        markdown_path=args.markdown_output,
        baseline_environment_path=args.baseline_environment,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
