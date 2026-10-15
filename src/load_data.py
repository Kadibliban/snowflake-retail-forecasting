from __future__ import annotations

from pathlib import Path

import pandas as pd
from snowflake.connector.pandas_tools import write_pandas

from src.config import DEFAULT_SYNTHETIC_CSV, require_snowflake_config
from src.snowflake_client import get_connection


def validate_sales_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply required validation rules to raw sales data before loading."""
    if df is None or df.empty:
        raise ValueError("Sales data is empty. Add rows before loading.")

    required_columns = {"SALE_DATE", "REVENUE", "QUANTITY"}
    missing = required_columns.difference(set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = df.copy()
    cleaned["SALE_DATE"] = pd.to_datetime(cleaned["SALE_DATE"], errors="raise")
    cleaned["REVENUE"] = pd.to_numeric(cleaned["REVENUE"], errors="raise")
    cleaned["QUANTITY"] = pd.to_numeric(cleaned["QUANTITY"], errors="raise")

    if cleaned["SALE_DATE"].isna().any():
        raise ValueError("SALE_DATE contains missing values.")
    if cleaned["REVENUE"].isna().any() or cleaned["QUANTITY"].isna().any():
        raise ValueError("REVENUE and QUANTITY must not contain missing values.")
    if (cleaned["REVENUE"] <= 0).any():
        raise ValueError("REVENUE must be positive and greater than zero.")
    if (cleaned["QUANTITY"] < 0).any():
        raise ValueError("QUANTITY must be zero or greater.")
    if cleaned["SALE_DATE"].duplicated().any():
        duplicates = cleaned.loc[cleaned["SALE_DATE"].duplicated(keep=False), "SALE_DATE"].unique()
        raise ValueError(f"Duplicate dates detected in SALES_RAW: {sorted(duplicates.tolist())}. Investigate before loading.")

    return cleaned


def load_local_csv(path: str | Path = DEFAULT_SYNTHETIC_CSV) -> pd.DataFrame:
    """Read a CSV from the local data directory."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return validate_sales_dataframe(df)


def load_sales_to_snowflake(csv_path: str | Path, replace: bool = False) -> int:
    """Validate and upload local CSV data to the Snowflake raw table."""
    if replace:
        print("WARNING: replace=True will overwrite the Snowflake table contents.")

    conn = get_connection()
    df = pd.read_csv(csv_path)
    formatted = validate_sales_dataframe(df)
    formatted = formatted[["SALE_DATE", "REVENUE", "QUANTITY"]].copy()
    formatted["SALE_DATE"] = pd.to_datetime(formatted["SALE_DATE"]).dt.strftime("%Y-%m-%d")

    try:
        if replace:
            cursor = conn.cursor()
            cursor.execute("TRUNCATE TABLE RETAIL_ML.ANALYTICS.SALES_RAW")
            cursor.close()

        success = write_pandas(
            conn,
            formatted,
            table_name="SALES_RAW",
            database="RETAIL_ML",
            schema="ANALYTICS",
            quote_identifiers=False,
            auto_create_table=False,
            overwrite=replace,
        )
        if success is None:
            success = len(formatted)
        return int(success)
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load a local CSV into Snowflake.")
    parser.add_argument("--csv", default=str(DEFAULT_SYNTHETIC_CSV), help="CSV file to upload")
    parser.add_argument("--replace", action="store_true", help="Explicitly replace existing Snowflake data")
    args = parser.parse_args()

    try:
        rows_loaded = load_sales_to_snowflake(args.csv, replace=args.replace)
        print(f"Rows successfully loaded: {rows_loaded}")
    except Exception as exc:
        raise RuntimeError(f"Failed to load sales data: {exc}") from exc
