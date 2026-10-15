from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_SYNTHETIC_CSV


def generate_synthetic_sales_dataframe(days: int = 730, seed: int = 42, start_date: str = "2023-01-01") -> pd.DataFrame:
    """Create a reproducible synthetic set of daily sales observations."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    day_index = np.arange(days)

    trend = np.linspace(120, 240, num=days)
    weekly_seasonality = 30 * np.sin((2 * np.pi * (day_index % 7)) / 7)
    annual_seasonality = 18 * np.sin((2 * np.pi * day_index / 365.0) + 1.2)
    noise = rng.normal(0, 8, size=days)
    revenue = trend + weekly_seasonality + annual_seasonality + noise
    revenue = np.maximum(revenue, 25.0)

    df = pd.DataFrame(
        {
            "SALE_DATE": dates,
            "REVENUE": np.round(revenue, 2),
            "QUANTITY": np.round(revenue / 18.0).astype(int),
            "IS_SYNTHETIC": True,
        }
    )

    return df


def save_synthetic_sales_csv(path: str | None = None) -> pd.DataFrame:
    """Write the synthetic sales data to a local CSV file for development."""
    output_path = Path(path) if path is not None else DEFAULT_SYNTHETIC_CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_sales_dataframe()
    df[["SALE_DATE", "REVENUE", "QUANTITY"]].to_csv(output_path, index=False)
    print(f"Saved synthetic sales data to {output_path}")
    return df


if __name__ == "__main__":
    save_synthetic_sales_csv()
