"""Shared setup for the Lending Club analysis notebooks.

Usage (from a notebook in Notebooks/):

    import sys; sys.path.append("../src")
    from lending_db import get_connection
    con = get_connection()
"""

from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
SQL_DIR = PROJECT_ROOT / "SQL"
IMAGE_DIR = PROJECT_ROOT / "images"

RAW_FILES = {
    "accepted": "accepted_2007_to_2018Q4.csv",
    "rejected": "rejected_2007_to_2018Q4.csv",
}


def build_parquet(data_dir: Path = DATA_DIR) -> None:
    """Convert the raw Kaggle CSVs to Parquet once.

    The CSVs are ~1.7 GB each. Parquet is columnar and compressed, so queries
    that touch a few columns read only those columns (seconds instead of minutes).
    `sample_size = -1` scans the full file before inferring column types;
    `ignore_errors = true` skips the malformed summary rows at the end of the file.
    """
    con = duckdb.connect()
    for name, csv_name in RAW_FILES.items():
        parquet_path = data_dir / f"{name}.parquet"
        if parquet_path.exists():
            continue
        csv_path = data_dir / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found. See Data/README.md for download instructions."
            )
        con.sql(f"""
            COPY (
                SELECT *
                FROM read_csv('{csv_path}', sample_size = -1, ignore_errors = true)
            ) TO '{parquet_path}' (FORMAT parquet)
        """)
    con.close()


def get_connection(data_dir: Path = DATA_DIR) -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB connection with the analysis views created."""
    build_parquet(data_dir)
    con = duckdb.connect()
    ddl = (SQL_DIR / "00_create_views.sql").read_text()
    con.execute(ddl.replace("{data_dir}", str(data_dir)))
    IMAGE_DIR.mkdir(exist_ok=True)
    return con
