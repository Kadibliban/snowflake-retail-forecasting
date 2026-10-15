from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = ["LAG_1", "LAG_7", "ROLLING_MEAN_7", "DAY_OF_WEEK", "MONTH"]


def validate_daily_series(df: pd.DataFrame, date_col: str = "SALE_DATE", value_col: str = "REVENUE") -> pd.DataFrame:
    """Validate a daily observation series and reject duplicates or gaps."""
    if df is None or df.empty:
        raise ValueError("Sales data is empty. Provide a non-empty daily dataset.")

    cleaned = df.copy()
    cleaned[date_col] = pd.to_datetime(cleaned[date_col], errors="raise")
    cleaned[value_col] = pd.to_numeric(cleaned[value_col], errors="raise")

    if cleaned[date_col].duplicated().any():
        duplicates = cleaned.loc[cleaned[date_col].duplicated(keep=False), date_col].unique().tolist()
        raise ValueError(f"Duplicate dates detected: {duplicates}. Investigate before training or forecasting.")

    if cleaned[date_col].isna().any():
        raise ValueError("Missing dates were found in the dataset. Review the source data.")

    ordered = cleaned.sort_values(date_col).reset_index(drop=True)
    full_range = pd.date_range(start=ordered[date_col].min(), end=ordered[date_col].max(), freq="D")
    missing = full_range.difference(ordered[date_col])
    if len(missing) > 0:
        missing_dates = [d.strftime("%Y-%m-%d") for d in missing[:10]]
        raise ValueError(
            "Missing daily observations detected. Investigate gaps in the source data before proceeding. "
            f"Sample missing dates: {missing_dates}"
        )

    return ordered


def build_feature_frame(df: pd.DataFrame, date_col: str = "SALE_DATE", value_col: str = "REVENUE") -> pd.DataFrame:
    """Create lag and calendar features from daily sales data."""
    validated = validate_daily_series(df, date_col=date_col, value_col=value_col)
    features = validated.copy()
    features["LAG_1"] = features[value_col].shift(1)
    features["LAG_7"] = features[value_col].shift(7)
    features["ROLLING_MEAN_7"] = features[value_col].shift(1).rolling(window=7, min_periods=1).mean()
    features["DAY_OF_WEEK"] = features[date_col].dt.dayofweek
    features["MONTH"] = features[date_col].dt.month
    return features
