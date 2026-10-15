from __future__ import annotations

import pandas as pd

from src.features import FEATURE_COLUMNS, build_feature_frame


def test_feature_values_for_known_sample() -> None:
    df = pd.DataFrame(
        {
            "SALE_DATE": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"]),
            "REVENUE": [100, 120, 130, 135, 140, 150, 160, 170],
        }
    )

    features = build_feature_frame(df)

    assert list(features.columns[:5]) == ["SALE_DATE", "REVENUE", "LAG_1", "LAG_7", "ROLLING_MEAN_7"]
    assert features.loc[7, "LAG_1"] == 160
    assert features.loc[7, "LAG_7"] == 100
    assert features.loc[7, "ROLLING_MEAN_7"] == 133.57142857142858
    assert features.loc[0, "DAY_OF_WEEK"] == 0
    assert features.loc[0, "MONTH"] == 1
    assert FEATURE_COLUMNS == ["LAG_1", "LAG_7", "ROLLING_MEAN_7", "DAY_OF_WEEK", "MONTH"]
