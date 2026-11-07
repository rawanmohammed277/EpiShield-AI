"""SQL Server connection settings for the EpiShield backend and ETL."""

import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine


PROJECT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_DIR / ".env")

SERVER = os.getenv("DB_SERVER", "tcp:100.103.10.17,1433")
DATABASE = os.getenv("DB_NAME", "EpiShieldDB_Test")
DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")
USERNAME = os.getenv("DB_USERNAME", "epi_backend")
PASSWORD = os.getenv("DB_PASSWORD")

if not PASSWORD:
    raise RuntimeError(
        "DB_PASSWORD is not set. Add it to the project-root .env file; "
        "never commit or share that file."
    )

connection_string = (
    f"DRIVER={{{DRIVER}}};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"UID={USERNAME};"
    f"PWD={PASSWORD};"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
)

connection_url = (
    "mssql+pyodbc:///?odbc_connect=" + quote_plus(connection_string)
)

engine = create_engine(
    connection_url,
    fast_executemany=True,
    pool_pre_ping=True,
)


def test_connection() -> dict[str, str | int]:
    """Check SQL login and read access without changing database contents."""
    with engine.connect() as connection:
        database_name = connection.exec_driver_sql("SELECT DB_NAME()").scalar_one()
        sample = connection.exec_driver_sql(
            "SELECT TOP (1) * FROM dbo.DimDate"
        ).first()
    result = {
        "status": "success",
        "database": str(database_name),
        "dim_date_readable": 1 if sample is not None else 0,
    }
    print(f"Connection successful. Database: {database_name}")
    print(f"DimDate readable: {sample is not None}")
    return result


if __name__ == "__main__":
    test_connection()
