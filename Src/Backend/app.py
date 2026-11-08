"""FastAPI endpoints used by the EpiShield frontend."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import sys
from threading import Lock
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SRC_DIR.parent
PIPELINE_FILE = SRC_DIR / "Data Engineering" / "pipeline.py"
FRONTEND_DIR = SRC_DIR / "Frontend"
load_dotenv(PROJECT_DIR / ".env")


def _configured_origins() -> list[str]:
    raw = os.getenv("FRONTEND_ORIGINS", "")
    if raw.strip():
        return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ]


app = FastAPI(
    title="EpiShield AI Backend",
    version="1.0.0",
    description="Starts the epidemic-data ETL pipeline and reports its status.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_configured_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_state_lock = Lock()
_latest_run: dict[str, Any] = {
    "run_id": None,
    "status": "idle",
    "started_at": None,
    "completed_at": None,
    "rows_read_by_file": None,
    "rows_loaded_by_table": None,
    "total_rows_read": None,
    "total_rows_loaded": None,
    "error": None,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_snapshot() -> dict[str, Any]:
    with _state_lock:
        return dict(_latest_run)


def _load_pipeline_runner():
    if not PIPELINE_FILE.is_file():
        raise FileNotFoundError(f"ETL entry point was not found: {PIPELINE_FILE}")
    module_name = "_epishield_pipeline_entry"
    spec = spec_from_file_location(module_name, PIPELINE_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load ETL entry point: {PIPELINE_FILE}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.run_pipeline


def _safe_error_message(error: Exception) -> str:
    message = str(error).strip() or error.__class__.__name__
    password = os.getenv("DB_PASSWORD")
    if password:
        message = message.replace(password, "[redacted]")
    return message[:1200]


def _execute_pipeline(run_id: str) -> None:
    try:
        run_pipeline = _load_pipeline_runner()
        result = run_pipeline()
        completed_at = result.get("completed_at") or _utc_now()
        with _state_lock:
            if _latest_run["run_id"] == run_id:
                _latest_run.update(
                    {
                        "status": "success",
                        "completed_at": completed_at,
                        "rows_read_by_file": result.get("rows_read_by_file"),
                        "rows_loaded_by_table": result.get("rows_loaded_by_table"),
                        "total_rows_read": result.get("total_rows_read"),
                        "total_rows_loaded": result.get("total_rows_loaded"),
                        "error": None,
                    }
                )
    except Exception as error:
        with _state_lock:
            if _latest_run["run_id"] == run_id:
                _latest_run.update(
                    {
                        "status": "failed",
                        "completed_at": _utc_now(),
                        "error": _safe_error_message(error),
                    }
                )


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {"name": "EpiShield AI Backend", "docs": "/docs"}


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/pipeline/run", status_code=status.HTTP_202_ACCEPTED, tags=["pipeline"])
def start_pipeline(background_tasks: BackgroundTasks) -> dict[str, Any]:
    with _state_lock:
        if _latest_run["status"] == "running":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": "A pipeline run is already in progress.",
                    "run_id": _latest_run["run_id"],
                },
            )
        if _latest_run["status"] == "success":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "This database has already completed a pipeline run. "
                    "Use a fresh empty test database before running it again."
                ),
            )

        run_id = str(uuid4())
        _latest_run.update(
            {
                "run_id": run_id,
                "status": "running",
                "started_at": _utc_now(),
                "completed_at": None,
                "rows_read_by_file": None,
                "rows_loaded_by_table": None,
                "total_rows_read": None,
                "total_rows_loaded": None,
                "error": None,
            }
        )
        response = dict(_latest_run)

    background_tasks.add_task(_execute_pipeline, run_id)
    return response


@app.get("/pipeline/status", tags=["pipeline"])
def pipeline_status() -> dict[str, Any]:
    return _state_snapshot()


if FRONTEND_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
