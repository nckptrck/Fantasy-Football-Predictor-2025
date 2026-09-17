import math

import pandas as pd
import pytest

from src.evaluate_predictions import combine_predictions, rmse_by_week, rmse_by_week_and_horizon


def test_combine_predictions_uses_target_week_overlap_and_origin_recency():
    df = pd.DataFrame(
        [
            {
                "player_name": "A.J. Brown",
                "target_season": 2025,
                "target_week": 5,
                "horizon_step": 3,
                "origin_season": 2025,
                "origin_week": 2,
                "y_true": 15.0,
                "y_pred": 12.0,
                "target_played": True,
            },
            {
                "player_name": "A.J. Brown",
                "target_season": 2025,
                "target_week": 5,
                "horizon_step": 1,
                "origin_season": 2025,
                "origin_week": 4,
                "y_true": 15.0,
                "y_pred": 18.0,
                "target_played": True,
            },
            {
                "player_name": "A.J. Brown",
                "target_season": 2025,
                "target_week": 6,
                "horizon_step": 1,
                "origin_season": 2025,
                "origin_week": 5,
                "y_true": 20.0,
                "y_pred": 19.0,
                "target_played": True,
            },
        ]
    )

    combined = combine_predictions(df, group_columns=("player_name", "target_season", "target_week"))

    assert len(combined) == 2
    week_5 = combined[combined["target_week"] == 5].iloc[0]
    assert week_5["n_forecasts"] == 2
    assert week_5["y_pred"] > 12.0 and week_5["y_pred"] < 18.0


def test_weekly_and_horizon_metrics_are_produced():
    df = pd.DataFrame(
        [
            {"player_name": "P1", "target_season": 2025, "target_week": 1, "horizon_step": 1, "y_true": 10.0, "y_pred": 12.0, "target_played": True},
            {"player_name": "P1", "target_season": 2025, "target_week": 1, "horizon_step": 2, "y_true": 10.0, "y_pred": 9.0, "target_played": True},
            {"player_name": "P2", "target_season": 2025, "target_week": 2, "horizon_step": 1, "y_true": 8.0, "y_pred": 9.0, "target_played": True},
            {"player_name": "P2", "target_season": 2025, "target_week": 2, "horizon_step": 2, "y_true": 8.0, "y_pred": 5.0, "target_played": True},
        ]
    )

    weekly = rmse_by_week(df)
    horizon = rmse_by_week_and_horizon(df)

    assert list(weekly["target_week"]) == [1, 2]
    assert set(horizon["horizon_step"]) == {1, 2}
    assert all(math.isfinite(value) for value in horizon["rmse"].tolist())
