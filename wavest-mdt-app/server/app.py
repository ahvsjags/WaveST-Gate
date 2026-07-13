"""Local-only API that connects the WaveST-MDT UI to WaveST-Gate inference."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

APP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(os.getenv("WAVEST_WORKSPACE_ROOT", APP_ROOT.parent)).resolve()
RUNTIME_ROOT = APP_ROOT / "runtime" / "cases"
DEFAULT_CHECKPOINT = Path(
    os.getenv(
        "WAVEST_CHECKPOINT",
        WORKSPACE_ROOT / "results" / "nature_main" / "cytassist_rep2_radius55" / "checkpoint.pt",
    )
).resolve()
DEVICE = os.getenv("WAVEST_DEVICE", "cuda")
BATCH_SIZE = max(1, int(os.getenv("WAVEST_BATCH_SIZE", "4")))
PATCH_SIZE = max(8, int(os.getenv("WAVEST_PATCH_SIZE", "32")))
MAX_UPLOAD_BYTES = int(os.getenv("WAVEST_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz"}

RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

app = FastAPI(
    title="WaveST-MDT local inference API",
    version="0.1.0",
    description="Local research-use service for WaveST-Gate case preparation and inference.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4173", "http://127.0.0.1:4173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_case_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-")
    return cleaned[:80] or "spatial-case"


def _job_snapshot(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown inference job")
        return dict(job)


def _update_job(job_id: str, **values: Any) -> None:
    with _jobs_lock:
        if job_id not in _jobs:
            return
        _jobs[job_id].update(values)
        _jobs[job_id]["updatedAt"] = _timestamp()


def _validate_suffix(upload: UploadFile, allowed: set[str], label: str) -> str:
    path = Path(upload.filename or "")
    suffix = "".join(path.suffixes[-2:]).lower() if path.suffix.lower() == ".gz" else path.suffix.lower()
    if suffix not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise HTTPException(status_code=415, detail=f"{label} must use one of: {allowed_text}")
    return suffix


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    written = 0
    with destination.open("wb") as handle:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                handle.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Uploaded file exceeds the local service size limit")
            handle.write(chunk)
    await upload.close()


def _run_command(job_id: str, command: list[str], *, stage: str, progress: int, log_path: Path) -> None:
    _update_job(job_id, status="running", stage=stage, progress=progress)
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    completed = subprocess.run(
        command,
        cwd=WORKSPACE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
        handle.write(completed.stdout)
        handle.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} failed. Review the local job log for details.")


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")


def _workstation_payload(job_id: str, case_name: str, case_dir: Path) -> dict[str, Any]:
    output_dir = case_dir / "output"
    proportions = pd.read_csv(output_dir / "predicted_proportions.csv", index_col="spot_id")
    gates = pd.read_csv(output_dir / "gate_weights.csv", index_col="spot_id")
    uncertainty = pd.read_csv(output_dir / "spot_uncertainty.csv", index_col="spot_id")["spot_uncertainty"]
    coords = _read_table(case_dir / "input" / "coordinates.csv")
    required = {"spot_id", "x", "y"}
    if not required.issubset(coords.columns):
        raise RuntimeError("Coordinate table must contain spot_id, x and y columns")
    coords = coords.assign(spot_id=coords["spot_id"].astype(str)).set_index("spot_id")
    common = [spot_id for spot_id in proportions.index.astype(str) if spot_id in coords.index]
    if not common:
        raise RuntimeError("No spot identifiers are shared between predictions and coordinates")

    proportions.index = proportions.index.astype(str)
    gates.index = gates.index.astype(str)
    uncertainty.index = uncertainty.index.astype(str)
    proportions = proportions.loc[common]
    gates = gates.reindex(common).fillna(0.0)
    uncertainty = uncertainty.reindex(common).fillna(0.0)
    coords = coords.loc[common]
    x = pd.to_numeric(coords["x"], errors="raise")
    y = pd.to_numeric(coords["y"], errors="raise")
    x_span = max(float(x.max() - x.min()), 1.0)
    y_span = max(float(y.max() - y.min()), 1.0)
    x_norm = ((x - x.min()) / x_span * 2.0 - 1.0).tolist()
    y_norm = ((y - y.min()) / y_span * 2.0 - 1.0).tolist()
    cell_types = [str(column) for column in proportions.columns]
    values = proportions.to_numpy(dtype=float)
    dominant = values.argmax(axis=1)
    gate_values = gates.reindex(columns=["image", "expression", "reference"], fill_value=0.0).to_numpy(dtype=float)
    gate_totals = gate_values.sum(axis=1, keepdims=True)
    gate_values = gate_values / gate_totals.clip(min=1e-8)

    spots = [
        {
            "id": spot_id,
            "x": round(float(x_norm[index]), 5),
            "y": round(float(y_norm[index]), 5),
            "values": [round(float(value), 6) for value in values[index]],
            "dominant": int(dominant[index]),
            "uncertainty": round(float(uncertainty.iloc[index]), 6),
            "gates": [round(float(value), 6) for value in gate_values[index]],
            "niche": 0,
            "nicheLabel": "Not assigned by this checkpoint",
        }
        for index, spot_id in enumerate(common)
    ]
    image_candidates = [
        path
        for path in (case_dir / "input").glob("he_image.*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    tissue_endpoint = f"/jobs/{job_id}/tissue" if image_candidates else None
    mean = values.mean(axis=0).tolist()
    return {
        "case": {
            "id": f"LOCAL-{job_id[:8].upper()}",
            "title": case_name,
            "titleZh": case_name,
            "specimen": "Locally supplied de-identified tissue",
            "specimenZh": "本地导入的去标识化组织病例",
            "validationContext": "On-demand local WaveST-Gate inference",
            "validationContextZh": "按需本地 WaveST-Gate 推理",
            "platform": "Uploaded H&E + spatial expression + reference atlas",
            "model": "WaveST-Gate 0.1.0",
        },
        "cellTypes": cell_types,
        "aggregate": {
            "spots": len(spots),
            "cellStates": len(cell_types),
            "supervisedSpots": 0,
            "typedCells": 0,
            "meanProportions": [round(float(value), 6) for value in mean],
            "uncertaintyCorrelation": None,
            "boundaryJumpRatio": None,
            "calibrationBinCorrelation": None,
        },
        "spots": spots,
        "inference": {
            "mode": "live_local",
            "nicheAvailable": False,
            "tissueEndpoint": tissue_endpoint,
            "message": "Uncertainty is model-derived; benchmark calibration and niche labels require a matched validation protocol.",
        },
    }


def _run_inference_job(job_id: str) -> None:
    job = _job_snapshot(job_id)
    case_dir = Path(job["caseDir"])
    log_path = case_dir / "job.log"
    try:
        if not DEFAULT_CHECKPOINT.exists():
            raise RuntimeError("No configured WaveST-Gate checkpoint is available on this workstation")
        config_path = case_dir / "prepare_config.json"
        prepared_path = case_dir / "prepared.pt"
        output_dir = case_dir / "output"
        _run_command(
            job_id,
            [sys.executable, "-m", "wavestgate.data.prepare_dataset", "--config", str(config_path)],
            stage="Preparing spatial and morphology inputs",
            progress=28,
            log_path=log_path,
        )
        _run_command(
            job_id,
            [
                sys.executable,
                "-m",
                "wavestgate.training.predict_aligned",
                "--checkpoint",
                str(DEFAULT_CHECKPOINT),
                "--prepared",
                str(prepared_path),
                "--output-dir",
                str(output_dir),
                "--batch-size",
                str(BATCH_SIZE),
                "--device",
                DEVICE,
            ],
            stage="Running WaveST-Gate on the local GPU",
            progress=62,
            log_path=log_path,
        )
        payload = _workstation_payload(job_id, job["caseName"], case_dir)
        result_path = case_dir / "workstation_result.json"
        result_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        _update_job(job_id, status="complete", stage="Results ready for MDT review", progress=100, resultReady=True)
    except Exception as error:
        _update_job(job_id, status="failed", stage="Inference stopped", progress=100, error=str(error))


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        device_name = torch.cuda.get_device_name(0) if cuda_available else None
    except Exception:
        cuda_available = False
        device_name = None
    return {
        "status": "ready" if DEFAULT_CHECKPOINT.exists() else "checkpoint_missing",
        "localOnly": True,
        "checkpointAvailable": DEFAULT_CHECKPOINT.exists(),
        "device": DEVICE,
        "cudaAvailable": cuda_available,
        "deviceName": device_name,
        "batchSize": BATCH_SIZE,
        "patchSize": PATCH_SIZE,
    }


@app.post("/api/jobs", status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    case_name: str = Form(...),
    h_e_image: UploadFile = File(...),
    expression: UploadFile = File(...),
    coordinates: UploadFile = File(...),
    reference: UploadFile = File(...),
) -> dict[str, Any]:
    if not DEFAULT_CHECKPOINT.exists():
        raise HTTPException(status_code=503, detail="No WaveST-Gate checkpoint is configured for local inference")
    image_suffix = _validate_suffix(h_e_image, IMAGE_SUFFIXES, "H&E image")
    expression_suffix = _validate_suffix(expression, TABLE_SUFFIXES, "Expression matrix")
    coordinate_suffix = _validate_suffix(coordinates, TABLE_SUFFIXES, "Coordinate table")
    reference_suffix = _validate_suffix(reference, TABLE_SUFFIXES, "Reference atlas")
    if coordinate_suffix != ".csv":
        raise HTTPException(status_code=415, detail="Coordinate table must be a CSV with spot_id, x and y columns")

    job_id = uuid.uuid4().hex
    case_dir = RUNTIME_ROOT / job_id
    input_dir = case_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)
    await _save_upload(h_e_image, input_dir / f"he_image{image_suffix}")
    await _save_upload(expression, input_dir / f"expression{expression_suffix}")
    await _save_upload(coordinates, input_dir / "coordinates.csv")
    await _save_upload(reference, input_dir / f"reference{reference_suffix}")
    config_path = case_dir / "prepare_config.json"
    prepared_path = case_dir / "prepared.pt"
    config = {
        "data": {
            "spot_expression_path": str(input_dir / f"expression{expression_suffix}"),
            "spot_coords_path": str(input_dir / "coordinates.csv"),
            "reference_prototypes_path": str(input_dir / f"reference{reference_suffix}"),
            "he_image_path": str(input_dir / f"he_image{image_suffix}"),
            "output_path": str(prepared_path),
        },
        "columns": {"spot_id_col": "spot_id", "spot_x_col": "x", "spot_y_col": "y", "cell_type_col": "cell_type"},
        "patches": {"patch_size": PATCH_SIZE},
        "graph": {"k": 6},
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    job = {
        "id": job_id,
        "caseName": _safe_case_name(case_name),
        "caseDir": str(case_dir),
        "status": "queued",
        "stage": "Files accepted for local QC",
        "progress": 8,
        "resultReady": False,
        "createdAt": _timestamp(),
        "updatedAt": _timestamp(),
    }
    with _jobs_lock:
        _jobs[job_id] = job
    background_tasks.add_task(_run_inference_job, job_id)
    return _job_snapshot(job_id)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = _job_snapshot(job_id)
    return {key: value for key, value in job.items() if key != "caseDir"}


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str) -> JSONResponse:
    job = _job_snapshot(job_id)
    if job["status"] != "complete":
        raise HTTPException(status_code=409, detail="Inference result is not ready")
    result_path = Path(job["caseDir"]) / "workstation_result.json"
    return JSONResponse(json.loads(result_path.read_text(encoding="utf-8")))


@app.get("/api/jobs/{job_id}/tissue")
def get_tissue(job_id: str) -> FileResponse:
    job = _job_snapshot(job_id)
    input_dir = Path(job["caseDir"]) / "input"
    candidates = [path for path in input_dir.glob("he_image.*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not candidates:
        raise HTTPException(status_code=415, detail="This H&E file is not browser-renderable; use PNG or JPEG for the interactive view")
    return FileResponse(candidates[0], headers={"Cache-Control": "no-store"})
