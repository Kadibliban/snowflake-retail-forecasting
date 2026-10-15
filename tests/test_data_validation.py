from __future__ import annotations

import pandas as pd
import pytest

from src.features import validate_daily_series


def test_duplicate_date_rejected() -> None:
    df = pd.DataFrame(
        {
            "SALE_DATE": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-04"]),
            "REVENUE": [100, 110, 110, 120],
        }
    )

    with pytest.raises(ValueError, match="Duplicate dates"):
        validate_daily_series(df)


def test_missing_daily_observation_detected() -> None:
    df = pd.DataFrame(
        {
            "SALE_DATE": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"]),
            "REVENUE": [100, 110, 120],
        }
    )

    with pytest.raises(ValueError, match="Missing daily observations"):
        validate_daily_series(df)
