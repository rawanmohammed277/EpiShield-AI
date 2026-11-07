"""Callable entry point for the EpiShield extract-transform-load workflow.

This module runs the existing preprocessing and ETL notebooks, then delegates
the SQL Server load and database validations to ``Database/final_load.py``.
It intentionally leaves the original source files and notebooks untouched.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from threading import Lock
from typing import Any

import nbformat
import pandas as pd
from nbclient import NotebookClient


SRC_DIR = Path(__file__).resolve().parents[1]
ENGINE_DIR = Path(__file__).resolve().parent
ETL_OUTPUT_DIR = SRC_DIR / "ETL Output"
PREPROCESSING_NOTEBOOK = ENGINE_DIR / "preprocessing.ipynb"
ETL_NOTEBOOK = ENGINE_DIR / "etl.ipynb"

_PIPELINE_LOCK = Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_inputs() -> dict[str, int]:
    """Read and validate the four CSV inputs without modifying source files."""
    if str(ENGINE_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_DIR))
    from extract import extract_data

    data = extract_data()
    required_columns = {
        "regioni": {
            "data", "codice_regione", "denominazione_regione",
            "nuovi_positivi", "deceduti", "note",
        },
        "nazionale": {"data", "nuovi_positivi", "deceduti", "note"},
        "popolazione": {
            "codice_regione", "denominazione_regione", "range_eta",
            "totale_genere_maschile", "totale_genere_femminile",
            "totale_generale",
        },
        "note": {"data", "note"},
    }
    missing = set(required_columns) - set(data)
    if missing:
        raise ValueError(f"Extraction did not return required datasets: {sorted(missing)}")
    for name, columns in required_columns.items():
        absent = columns - set(data[name].columns)
        if absent:
            raise ValueError(
                f"Input '{name}' is missing required columns: {sorted(absent)}"
            )
    return {name: int(frame.shape[0]) for name, frame in data.items()}


def _execute_notebook(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required ETL notebook not found: {path}")
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(ENGINE_DIR)}},
    )
    client.execute()


def _staging_row_counts() -> dict[str, int]:
    files = {
        "DimLocation": "dim_location_load.csv",
        "DimDate": "dim_date_load.csv",
        "FactPopulation": "fact_population_staging.csv",
        "FactNational": "fact_national_staging.csv",
        "FactRegional": "fact_regional_staging.csv",
        "ReportNotes": "report_notes_staging.csv",
    }
    counts: dict[str, int] = {}
    for table, filename in files.items():
        path = ETL_OUTPUT_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"ETL did not create required staging file: {path}")
        counts[table] = int(len(pd.read_csv(path)))
    return counts


def run_pipeline() -> dict[str, Any]:
    """Run extraction, preprocessing, transformation, and the SQL Server load.

    Returns a JSON-friendly summary on success. Errors are raised to the caller
    so an API layer can return a failed status and a useful error message.
    Only one pipeline run is allowed at a time in this Python process.
    """
    if not _PIPELINE_LOCK.acquire(blocking=False):
        raise RuntimeError("A pipeline run is already in progress.")

    started_at = _utc_now()
    try:
        input_rows = _check_inputs()
        _execute_notebook(PREPROCESSING_NOTEBOOK)
        _execute_notebook(ETL_NOTEBOOK)
        loaded_rows = _staging_row_counts()

        database_dir = SRC_DIR / "Database"
        if str(database_dir) not in sys.path:
            sys.path.insert(0, str(database_dir))
        import final_load

        # Reuse the existing validation, key mapping, transaction, and load code.
        final_load.main()

        return {
            "status": "success",
            "started_at": started_at,
            "completed_at": _utc_now(),
            "rows_read_by_file": input_rows,
            "rows_loaded_by_table": loaded_rows,
            "total_rows_read": sum(input_rows.values()),
            "total_rows_loaded": sum(loaded_rows.values()),
        }
    finally:
        _PIPELINE_LOCK.release()
