from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import DEFAULT_FORECAST_PATH, DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH, DEFAULT_SYNTHETIC_CSV, DEFAULT_TEST_PREDICTIONS_PATH, OUTPUTS_DIR
from src.features import FEATURE_COLUMNS, build_feature_frame
from src.generate_data import save_synthetic_sales_csv
from src.predict import generate_future_forecast
from src.snowflake_client import fetch_sales_data


def load_training_data(data_source: str = "synthetic") -> pd.DataFrame:
    """Load sales data from a Snowflake connection when configured; otherwise use the local CSV."""
    if data_source.lower() == "snowflake":
        try:
            df = fetch_sales_data()
            print("Loaded sales data from Snowflake.")
            return df
        except Exception as exc:
            raise RuntimeError(
                "Snowflake data source was requested but the connection or credentials are unavailable. "
                "Switch to synthetic mode or configure the .env file."
            ) from exc

    csv_path = DEFAULT_SYNTHETIC_CSV
    if not csv_path.exists():
        save_synthetic_sales_csv(csv_path)
    df = pd.read_csv(csv_path)
    print(f"Loaded synthetic training data from {csv_path}.")
    return df


def split_train_test(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a chronological train/test split."""
    if len(df) < 10:
        raise ValueError("Insufficient daily data. At least 10 observations are needed for a chronological split.")

    cutoff = max(1, int(len(df) * (1 - test_fraction)))
    train = df.iloc[:cutoff].copy().reset_index(drop=True)
    test = df.iloc[cutoff:].copy().reset_index(drop=True)

    if len(train) < 14 or len(test) < 1:
        raise ValueError("The time split produced too little training or test data.")
    return train, test


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    """Compute MAE, RMSE, and WAPE when the denominator is valid."""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)

    mae = float(np.mean(np.abs(actual_array - predicted_array)))
    rmse = float(np.sqrt(np.mean((actual_array - predicted_array) ** 2)))
    denominator = float(np.sum(np.abs(actual_array)))
    wape = None if denominator == 0 else float(np.sum(np.abs(actual_array - predicted_array)) / denominator)

    return {"MAE": mae, "RMSE": rmse, "WAPE": wape}


def train_and_evaluate(data_source: str = "synthetic", model_path: str = str(DEFAULT_MODEL_PATH), metrics_path: str = str(DEFAULT_METRICS_PATH), test_predictions_path: str = str(DEFAULT_TEST_PREDICTIONS_PATH), forecast_path: str = str(DEFAULT_FORECAST_PATH), random_seed: int = 42) -> dict:
    """Train a RandomForest model, evaluate it against a naive baseline, and save artifacts."""
    data_df = load_training_data(data_source=data_source)
    data_df = data_df[["SALE_DATE", "REVENUE"]].copy()
    features_df = build_feature_frame(data_df)
    train_df, test_df = split_train_test(features_df, test_fraction=0.2)

    feature_columns = FEATURE_COLUMNS
    X_train = train_df[feature_columns]
    y_train = train_df["REVENUE"]
    X_test = test_df[feature_columns]
    y_test = test_df["REVENUE"]

    model = RandomForestRegressor(n_estimators=250, random_state=random_seed)
    print("Training RandomForestRegressor...")
    model.fit(X_train, y_train)

    model_predictions = model.predict(X_test)
    naive_predictions = test_df["LAG_7"]

    rf_metrics = calculate_metrics(y_test.values, model_predictions)
    naive_metrics = calculate_metrics(y_test.values, naive_predictions.values)

    predictions_df = pd.DataFrame(
        {
            "SALE_DATE": test_df["SALE_DATE"].dt.strftime("%Y-%m-%d"),
            "ACTUAL_REVENUE": y_test.values,
            "PREDICTED_REVENUE": model_predictions,
            "NAIVE_7D_REVENUE": naive_predictions.values,
        }
    )

    metrics_payload = {
        "data_source": data_source,
        "evaluation_period": {
            "start": test_df["SALE_DATE"].min().strftime("%Y-%m-%d"),
            "end": test_df["SALE_DATE"].max().strftime("%Y-%m-%d"),
        },
        "feature_list": feature_columns,
        "model": {
            "name": "RandomForestRegressor",
            "settings": {"n_estimators": 250, "random_state": random_seed},
        },
        "random_seed": random_seed,
        "model_metrics": rf_metrics,
        "naive_baseline_metrics": naive_metrics,
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    model_output_path = Path(model_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    # After scoring the untouched holdout, refit the deployable model on all
    # available observations, as they will be known when producing a forecast.
    production_model = RandomForestRegressor(n_estimators=250, random_state=random_seed)
    production_model.fit(features_df[feature_columns], features_df["REVENUE"])
    joblib.dump(production_model, model_output_path)

    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)

    predictions_df.to_csv(test_predictions_path, index=False)
    future_forecast = generate_future_forecast(
        model_path=model_path,
        horizon=14,
        history_df=data_df,
        output_path=forecast_path,
    )

    print(f"Saved test predictions to {test_predictions_path}")
    print(f"Saved future forecast to {forecast_path}")
    print(f"Saved model to {model_path}")
    print(f"Saved evaluation metrics to {metrics_path}")

    print("Model metrics:")
    print(json.dumps(rf_metrics, indent=2))
    print("Baseline metrics:")
    print(json.dumps(naive_metrics, indent=2))

    return {"model": production_model, "metrics": rf_metrics, "naive_metrics": naive_metrics, "test_predictions": predictions_df, "future_forecast": future_forecast, "model_path": model_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the retail sales forecasting model.")
    parser.add_argument(
        "--data-source",
        choices=("synthetic", "snowflake"),
        default="synthetic",
        help="Use local synthetic CSV data (default) or retrieve data from Snowflake.",
    )
    args = parser.parse_args()
    train_and_evaluate(data_source=args.data_source)
