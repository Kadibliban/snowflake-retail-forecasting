from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import DEFAULT_FORECAST_PATH, DEFAULT_MODEL_PATH, DEFAULT_SYNTHETIC_CSV
from src.features import FEATURE_COLUMNS, build_feature_frame, validate_daily_series


def validate_history(df: pd.DataFrame) -> pd.DataFrame:
    """Validate historical sales data before generating recursive forecasts."""
    validated = validate_daily_series(df, date_col="SALE_DATE", value_col="REVENUE")
    return validated.sort_values("SALE_DATE").reset_index(drop=True)


def recursive_forecast(history_df: pd.DataFrame, model, horizon: int = 14) -> pd.DataFrame:
    """Generate a recursive 14-day forecast using prior predictions as inputs."""
    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")

    history = validate_history(history_df)
    if len(history) < 14:
        raise ValueError("At least 14 days of history are required for a stable recursive forecast.")

    feature_history = build_feature_frame(history).copy()
    revenue_series = feature_history["REVENUE"].tolist()
    future_dates = pd.date_range(start=feature_history["SALE_DATE"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast_rows = []

    for sale_date in future_dates:
        lag_1 = revenue_series[-1]
        lag_7 = revenue_series[-7] if len(revenue_series) >= 7 else revenue_series[0]
        rolling_mean_7 = np.mean(revenue_series[-7:]) if len(revenue_series) >= 7 else np.mean(revenue_series)
        feature_value = {
            "LAG_1": lag_1,
            "LAG_7": lag_7,
            "ROLLING_MEAN_7": rolling_mean_7,
            "DAY_OF_WEEK": sale_date.dayofweek,
            "MONTH": sale_date.month,
        }
        row = pd.DataFrame([feature_value], columns=FEATURE_COLUMNS)
        predicted_value = float(model.predict(row)[0])
        predicted_value = max(predicted_value, 0.0)
        forecast_rows.append({"SALE_DATE": sale_date, "REVENUE": predicted_value})
        revenue_series.append(predicted_value)

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df["PREDICTED_REVENUE"] = forecast_df["REVENUE"]
    return forecast_df


def generate_future_forecast(
    model_path: str = str(DEFAULT_MODEL_PATH),
    history_csv: str | None = None,
    horizon: int = 14,
    history_df: pd.DataFrame | None = None,
    output_path: str = str(DEFAULT_FORECAST_PATH),
) -> pd.DataFrame:
    """Load the trained model and generate a future forecast for the next 14 days."""
    if history_df is None:
        csv_path = Path(history_csv) if history_csv is not None else DEFAULT_SYNTHETIC_CSV
        history_df = pd.read_csv(csv_path)
    history_df = history_df.copy()
    history_df["REVENUE"] = pd.to_numeric(history_df["REVENUE"], errors="raise")
    history_df["SALE_DATE"] = pd.to_datetime(history_df["SALE_DATE"], errors="raise")

    model = joblib.load(model_path)
    forecast_df = recursive_forecast(history_df, model, horizon=horizon)
    output_file = Path(output_path)
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(output_file, index=False)
    return forecast_df


if __name__ == "__main__":
    generate_future_forecast()
