from __future__ import annotations

import pandas as pd

try:
    import snowflake.connector
except ImportError:  # pragma: no cover
    snowflake = None

from src.config import require_snowflake_config


def get_connection():
    """Create a Snowflake connection using environment variables."""
    if snowflake is None:
        raise RuntimeError(
            "The Snowflake Python connector is not installed. Install it with `pip install snowflake-connector-python[pandas]` in a Python 3.12 environment or use the synthetic mode."
        )

    config = require_snowflake_config()
    try:
        conn = snowflake.connector.connect(
            account=config["account"],
            user=config["user"],
            password=config["password"],
            warehouse=config["warehouse"],
            database=config["database"],
            schema=config["schema"],
        )
    except Exception as exc:  # pragma: no cover - depends on external service
        raise RuntimeError(
            "Snowflake connection failed. Check the .env values, account identifier, and role privileges."
        ) from exc
    return conn


def fetch_sales_data() -> pd.DataFrame:
    """Load retail sales raw data from Snowflake."""
    conn = get_connection()
    try:
        query = """
            SELECT
                SALE_DATE,
                REVENUE,
                QUANTITY
            FROM RETAIL_ML.ANALYTICS.SALES_RAW
            ORDER BY SALE_DATE
        """
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    if df.empty:
        raise ValueError("Snowflake returned no rows from SALES_RAW.")

    df = df.copy()
    df["SALE_DATE"] = pd.to_datetime(df["SALE_DATE"], errors="raise")
    df["REVENUE"] = pd.to_numeric(df["REVENUE"], errors="raise")
    df["QUANTITY"] = pd.to_numeric(df["QUANTITY"], errors="raise")
    return df
