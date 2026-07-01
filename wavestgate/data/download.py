"""Dataset manifest downloader for WaveST-Gate public data."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DownloadResult:
    item_id: str
    path: str
    status: str
    bytes: int
    expected_bytes: int | None = None
    message: str = ""


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _item_path(root_dir: Path, item: dict[str, Any]) -> Path:
    return root_dir / item["path"]


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _matches_expected(path: Path, expected_bytes: int | None) -> bool:
    return path.exists() and (expected_bytes is None or _file_size(path) == expected_bytes)


def _download_with_wget(url: str, output: Path) -> None:
    cmd = [
        "wget",
        "--continue",
        "--tries=3",
        "--timeout=60",
        "--read-timeout=60",
        "--progress=dot:giga",
        "-O",
        str(output),
        url,
    ]
    subprocess.run(cmd, check=True)


def _download_with_aria2(url: str, output: Path) -> None:
    cmd = [
        "aria2c",
        "-c",
        "-x",
        "8",
        "-s",
        "8",
        "-k",
        "4M",
        "--file-allocation=none",
        "--summary-interval=30",
        "-d",
        str(output.parent),
        "-o",
        output.name,
        url,
    ]
    subprocess.run(cmd, check=True)


def _download_with_curl(url: str, output: Path) -> None:
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "3",
        "--connect-timeout",
        "60",
        "-C",
        "-",
        "-o",
        str(output),
        url,
    ]
    subprocess.run(cmd, check=True)


def download_item(root_dir: Path, item: dict[str, Any], overwrite: bool = False) -> DownloadResult:
    item_id = item["id"]
    if item.get("unavailable"):
        path = _item_path(root_dir, item)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item.get("reason", "Unavailable"), encoding="utf-8")
        return DownloadResult(item_id, str(path), "unavailable", _file_size(path), item.get("expected_bytes"), item.get("reason", ""))

    path = _item_path(root_dir, item)
    expected = item.get("expected_bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and path.exists():
        path.unlink()
    if _matches_expected(path, expected):
        return DownloadResult(item_id, str(path), "exists", _file_size(path), expected)

    try:
        if shutil.which("aria2c"):
            _download_with_aria2(item["url"], path)
        elif shutil.which("wget"):
            _download_with_wget(item["url"], path)
        elif shutil.which("curl"):
            _download_with_curl(item["url"], path)
        else:
            raise RuntimeError("Neither wget nor curl is available")
    except subprocess.CalledProcessError as exc:
        return DownloadResult(item_id, str(path), "failed", _file_size(path), expected, str(exc))

    if expected is not None and _file_size(path) != expected:
        return DownloadResult(
            item_id,
            str(path),
            "size_mismatch",
            _file_size(path),
            expected,
            f"Expected {expected} bytes, got {_file_size(path)}",
        )
    return DownloadResult(item_id, str(path), "downloaded", _file_size(path), expected)


def verify_item(root_dir: Path, item: dict[str, Any]) -> DownloadResult:
    item_id = item["id"]
    path = _item_path(root_dir, item)
    expected = item.get("expected_bytes")
    if item.get("unavailable"):
        return DownloadResult(item_id, str(path), "unavailable", _file_size(path), expected, item.get("reason", ""))
    if not path.exists():
        return DownloadResult(item_id, str(path), "missing", 0, expected, "File is not present")
    if expected is not None and _file_size(path) != expected:
        return DownloadResult(
            item_id,
            str(path),
            "size_mismatch",
            _file_size(path),
            expected,
            f"Expected {expected} bytes, got {_file_size(path)}",
        )
    return DownloadResult(item_id, str(path), "exists", _file_size(path), expected)


def _select_items(
    manifest: dict[str, Any],
    required_only: bool = False,
    categories: set[str] | None = None,
    ids: set[str] | None = None,
    max_items: int | None = None,
) -> list[dict[str, Any]]:
    selected = []
    for item in manifest.get("items", []):
        if required_only and not item.get("required", False):
            continue
        if categories is not None and item.get("category") not in categories:
            continue
        if ids is not None and item.get("id") not in ids:
            continue
        selected.append(item)
    if max_items is not None:
        selected = selected[:max_items]
    return selected


def download_manifest(
    manifest_path: str | Path,
    root_dir: str | Path | None = None,
    required_only: bool = False,
    categories: set[str] | None = None,
    ids: set[str] | None = None,
    max_items: int | None = None,
    overwrite: bool = False,
) -> list[DownloadResult]:
    manifest = load_manifest(manifest_path)
    root = Path(root_dir or manifest["root_dir"])
    results = []
    selected = _select_items(manifest, required_only=required_only, categories=categories, ids=ids, max_items=max_items)

    for item in selected:
        result = download_item(root, item, overwrite=overwrite)
        results.append(result)
        print(json.dumps(result.__dict__, ensure_ascii=False), flush=True)
    status_path = root / "download_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps([r.__dict__ for r in results], indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def verify_manifest(
    manifest_path: str | Path,
    root_dir: str | Path | None = None,
    required_only: bool = False,
    categories: set[str] | None = None,
    ids: set[str] | None = None,
    max_items: int | None = None,
) -> list[DownloadResult]:
    manifest = load_manifest(manifest_path)
    root = Path(root_dir or manifest["root_dir"])
    selected = _select_items(manifest, required_only=required_only, categories=categories, ids=ids, max_items=max_items)
    results = [verify_item(root, item) for item in selected]
    for result in results:
        print(json.dumps(result.__dict__, ensure_ascii=False), flush=True)
    status_path = root / "verify_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps([r.__dict__ for r in results], indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WaveST-Gate public dataset manifests.")
    parser.add_argument("--manifest", required=True, help="Path to dataset manifest YAML.")
    parser.add_argument("--root-dir", default=None, help="Override manifest root_dir.")
    parser.add_argument("--required-only", action="store_true", help="Download only required items.")
    parser.add_argument("--category", action="append", default=None, help="Limit to one or more categories.")
    parser.add_argument("--id", action="append", default=None, help="Limit to one or more item ids.")
    parser.add_argument("--max-items", type=int, default=None, help="Limit number of selected items.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload even when a file exists.")
    parser.add_argument("--verify-only", action="store_true", help="Verify selected manifest files without downloading.")
    args = parser.parse_args()
    kwargs = {
        "manifest_path": args.manifest,
        "root_dir": args.root_dir,
        "required_only": args.required_only,
        "categories": set(args.category) if args.category else None,
        "ids": set(args.id) if args.id else None,
        "max_items": args.max_items,
    }
    if args.verify_only:
        results = verify_manifest(**kwargs)
        bad_statuses = {"missing", "size_mismatch", "failed"}
        if any(result.status in bad_statuses for result in results):
            sys.exit(1)
    else:
        download_manifest(**kwargs, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
