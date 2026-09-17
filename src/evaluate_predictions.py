"""Evaluate raw and combined multi-step fantasy forecasts."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


IDENTITY_COLUMNS = ["player_name", "target_season", "target_week", "horizon_step"]
REQUIRED_COLUMNS = IDENTITY_COLUMNS + ["y_true", "y_pred", "target_played"]


def _validate_predictions(predictions: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    if (predictions["horizon_step"] < 1).any():
        raise ValueError("horizon_step must start at 1")


def _forecast_distance_weeks(predictions: pd.DataFrame) -> pd.Series:
    """Measure how far each forecast is from its target week in weekly units."""
    if {"origin_season", "origin_week"}.issubset(predictions.columns):
        target_index = predictions["target_season"].astype(int) * 100 + predictions["target_week"].astype(int)
        origin_index = predictions["origin_season"].astype(int) * 100 + predictions["origin_week"].astype(int)
        return (target_index - origin_index).clip(lower=0).astype(float)
    if "forecast_age_days" in predictions.columns:
        return pd.to_numeric(predictions["forecast_age_days"], errors="coerce").fillna(0.0) / 7.0
    if {"origin_date", "target_date"}.issubset(predictions.columns):
        origin = pd.to_datetime(predictions["origin_date"])
        target = pd.to_datetime(predictions["target_date"])
        return ((target - origin).dt.days.clip(lower=0).fillna(0.0) / 7.0).astype(float)
    raise ValueError("Provide origin_season/origin_week, forecast_age_days, or origin_date/target_date")


def rmse_by_timedelta(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate RMSE by forecast distance, where step 1 is the next week."""
    _validate_predictions(predictions)
    scored = predictions[predictions["target_played"].astype(bool)].copy()
    if scored.empty:
        return pd.DataFrame(columns=["horizon_step", "n_predictions", "rmse"])
    scored["squared_error"] = (scored["y_pred"] - scored["y_true"]) ** 2
    result = (
        scored.groupby("horizon_step", as_index=False)
        .agg(n_predictions=("squared_error", "size"), mse=("squared_error", "mean"))
    )
    result["rmse"] = np.sqrt(result.pop("mse"))
    return result.sort_values("horizon_step").reset_index(drop=True)


def rmse_by_week(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate RMSE aggregated by target week, across all horizons."""
    _validate_predictions(predictions)
    scored = predictions[predictions["target_played"].astype(bool)].copy()
    if scored.empty:
        return pd.DataFrame(columns=["target_season", "target_week", "n_predictions", "rmse"])
    scored["squared_error"] = (scored["y_pred"] - scored["y_true"]) ** 2
    result = (
        scored.groupby(["target_season", "target_week"], as_index=False)
        .agg(n_predictions=("squared_error", "size"), mse=("squared_error", "mean"))
    )
    result["rmse"] = np.sqrt(result.pop("mse"))
    return result.sort_values(["target_season", "target_week"]).reset_index(drop=True)


def rmse_by_week_and_horizon(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate RMSE by target week and forecast horizon."""
    _validate_predictions(predictions)
    scored = predictions[predictions["target_played"].astype(bool)].copy()
    if scored.empty:
        return pd.DataFrame(columns=["target_season", "target_week", "horizon_step", "n_predictions", "rmse"])
    scored["squared_error"] = (scored["y_pred"] - scored["y_true"]) ** 2
    result = (
        scored.groupby(["target_season", "target_week", "horizon_step"], as_index=False)
        .agg(n_predictions=("squared_error", "size"), mse=("squared_error", "mean"))
    )
    result["rmse"] = np.sqrt(result.pop("mse"))
    return result.sort_values(["target_season", "target_week", "horizon_step"]).reset_index(drop=True)


def combine_predictions(
    predictions: pd.DataFrame,
    recency_decay: float = 0.25,
    horizon_decay: float = 0.02,
    group_columns: Iterable[str] = ("player_name", "target_season", "target_week"),
) -> pd.DataFrame:
    """Combine forecasts that share the same target bucket across different origin weeks.

    The nearest forecast to the target week gets the most weight, while farther-away
    starting points decay exponentially. This is the correct overlap pattern for
    multi-origin rolling forecasts where each player-target-week has up to 17 starts.
    """
    _validate_predictions(predictions)
    combined = predictions.copy()
    combined["forecast_distance_weeks"] = _forecast_distance_weeks(combined)
    combined["forecast_weight"] = np.exp(-recency_decay * combined["forecast_distance_weeks"]) * np.exp(
        -horizon_decay * (combined["horizon_step"] - 1)
    )
    combined["forecast_weight"] = combined["forecast_weight"].clip(lower=1e-12)
    group_columns = list(group_columns)

    def weighted_average(group: pd.DataFrame) -> pd.Series:
        weights = group["forecast_weight"].to_numpy(dtype=float)
        predictions_array = group["y_pred"].to_numpy(dtype=float)
        closest_row = group.loc[group["forecast_distance_weeks"].idxmin()]
        weighted_prediction = np.average(predictions_array, weights=weights)
        weighted_variance = np.average((predictions_array - weighted_prediction) ** 2, weights=weights)
        effective_forecasts = weights.sum() ** 2 / np.square(weights).sum()
        standard_error = np.sqrt(weighted_variance / effective_forecasts) if effective_forecasts > 1 else 0.0
        return pd.Series(
            {
                **{column: group.iloc[0][column] for column in group_columns},
                "origin_season": int(closest_row["origin_season"]),
                "origin_week": int(closest_row["origin_week"]),
                "horizon_step": int(closest_row["horizon_step"]),
                "forecast_distance_weeks": float(closest_row["forecast_distance_weeks"]),
                "y_true": group["y_true"].iloc[0],
                "y_pred": weighted_prediction,
                "prediction_std": np.sqrt(weighted_variance),
                "prediction_se": standard_error,
                "prediction_ci_low": max(0.0, weighted_prediction - 1.96 * standard_error),
                "prediction_ci_high": weighted_prediction + 1.96 * standard_error,
                "target_played": bool(group["target_played"].iloc[0]),
                "n_forecasts": len(group),
                "effective_forecasts": effective_forecasts,
                "weight_sum": weights.sum(),
            }
        )

    return (
        combined.groupby(group_columns, sort=True, dropna=False, group_keys=False)
        .apply(weighted_average)
        .reset_index(drop=True)
    )


def evaluate_raw_and_combined(predictions: pd.DataFrame, **combine_kwargs: float) -> dict[str, pd.DataFrame]:
    """Return horizon, weekly, and week+horizon RMSE tables before and after forecast aggregation."""
    combined = combine_predictions(predictions, **combine_kwargs)
    return {
        "raw": rmse_by_timedelta(predictions),
        "combined": rmse_by_timedelta(combined),
        "raw_by_week": rmse_by_week(predictions),
        "combined_by_week": rmse_by_week(combined),
        "raw_by_week_and_horizon": rmse_by_week_and_horizon(predictions),
        "combined_by_week_and_horizon": rmse_by_week_and_horizon(combined),
        "combined_predictions": combined,
    }
