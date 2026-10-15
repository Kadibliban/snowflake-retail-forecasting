from __future__ import annotations

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.features import build_feature_frame
from src.predict import recursive_forecast
from src.train import calculate_metrics, split_train_test


def test_chronological_split() -> None:
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame({"SALE_DATE": dates, "REVENUE": range(20)})
    train, test = split_train_test(df, test_fraction=0.2)

    assert len(train) == 16
    assert len(test) == 4
    assert train["SALE_DATE"].max() < test["SALE_DATE"].min()


def test_forecast_output_length_and_dates() -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({"SALE_DATE": dates, "REVENUE": [float(i) for i in range(30)]})
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    feature_df = build_feature_frame(df)
    model.fit(feature_df[["LAG_1", "LAG_7", "ROLLING_MEAN_7", "DAY_OF_WEEK", "MONTH"]], feature_df["REVENUE"])

    forecast = recursive_forecast(df, model, horizon=14)

    assert len(forecast) == 14
    assert forecast["SALE_DATE"].iloc[0] > df["SALE_DATE"].max()
    assert forecast["SALE_DATE"].iloc[-1] > df["SALE_DATE"].max()


def test_metrics_calculation_and_insufficient_data() -> None:
    actual = [10, 20, 30]
    predicted = [12, 18, 32]
    metrics = calculate_metrics(actual, predicted)

    assert round(metrics["MAE"], 4) == 2.0
    assert round(metrics["RMSE"], 4) == 2.0
    assert round(metrics["WAPE"], 4) == 0.1

    short_df = pd.DataFrame({"SALE_DATE": pd.to_datetime(["2024-01-01", "2024-01-02"]), "REVENUE": [1, 2]})
    with pytest.raises(ValueError, match="Insufficient daily data"):
        split_train_test(short_df)
