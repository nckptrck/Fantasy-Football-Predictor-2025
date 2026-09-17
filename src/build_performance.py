"""Build file-backed model performance artifacts for the Streamlit dashboard."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prediction_paths_for_season(predictions_path: Path, season: int) -> list[tuple[int, Path]]:
    """Find saved season/start-week forecast runs for a season."""
    pattern = re.compile(rf"production_predictions_{season}_week_(\d+)\.csv$")
    paths = []
    for path in predictions_path.parent.glob(f"production_predictions_{season}_week_*.csv"):
        match = pattern.match(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    if not paths:
        paths.append((1, predictions_path))
    return sorted(paths)


def load_season_predictions(predictions_path: Path, season: int) -> pd.DataFrame:
    """Combine forecast runs, keeping the latest applicable run per target week."""
    frames = []
    for start_week, path in prediction_paths_for_season(predictions_path, season):
        frame = pd.read_csv(path, low_memory=False)
        frame = frame[frame["target_week"] >= start_week].copy()
        frame["forecast_run_start_week"] = start_week
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    predictions = predictions[
        predictions["target_season"].eq(season) & predictions["target_week"].between(1, 17)
    ].drop(columns=["y_true"], errors="ignore")
    return (
        predictions.sort_values("forecast_run_start_week")
        .drop_duplicates(["player_name", "target_season", "target_week"], keep="last")
        .drop(columns=["forecast_run_start_week"])
    )


def merge_actuals(predictions_path: Path, data_dir: Path, season: int) -> pd.DataFrame:
    """Join the latest weekly actuals onto one forecast season."""
    predictions = load_season_predictions(predictions_path, season)
    required = {"player_name", "target_season", "target_week", "y_pred"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {', '.join(sorted(missing))}")

    weekly = pd.read_csv(data_dir / "weekly.csv", low_memory=False)
    required_weekly = {"player_display_name", "season", "week", "fantasy_points_ppr"}
    missing_weekly = required_weekly.difference(weekly.columns)
    if missing_weekly:
        raise ValueError(f"weekly.csv is missing columns: {', '.join(sorted(missing_weekly))}")
    if "season_type" in weekly.columns:
        weekly = weekly[weekly["season_type"].eq("REG")]

    predictions = predictions.copy()
    actuals = weekly[
        weekly["season"].eq(season) & weekly["week"].between(1, 17)
    ][["player_display_name", "season", "week", "fantasy_points_ppr"]].rename(
        columns={
            "player_display_name": "player_name",
            "season": "target_season",
            "week": "target_week",
            "fantasy_points_ppr": "y_true",
        }
    )
    actuals = actuals.groupby(["player_name", "target_season", "target_week"], as_index=False)["y_true"].sum()
    scored = predictions.merge(actuals, on=["player_name", "target_season", "target_week"], how="inner")
    scored["error"] = scored["y_pred"] - scored["y_true"]
    scored["absolute_error"] = scored["error"].abs()
    scored["squared_error"] = scored["error"].pow(2)
    scored["week"] = scored["target_week"].astype(int)
    return scored.sort_values(["target_week", "absolute_error"], ascending=[True, False])


def build_performance(
    predictions_path: Path,
    data_dir: Path,
    season: int,
    week: int | None = None,
) -> dict:
    scored = merge_actuals(predictions_path, data_dir, season)
    if week is not None:
        scored = scored[scored["target_week"].eq(week)].copy()
    def records(frame: pd.DataFrame) -> list[dict]:
        return json.loads(frame.to_json(orient="records"))

    weekly_summary = (
        scored.groupby("target_week", as_index=False)
        .agg(
            n_predictions=("error", "size"),
            mse=("squared_error", "mean"),
            mae=("absolute_error", "mean"),
            mean_error=("error", "mean"),
        )
    )
    weekly_summary["rmse"] = np.sqrt(weekly_summary["mse"])
    weekly_details = {}
    for scored_week, week_rows in scored.groupby("target_week"):
        weekly_details[str(int(scored_week))] = {
            "week": int(scored_week),
            "n_predictions": int(len(week_rows)),
            "mse": float(week_rows["squared_error"].mean()),
            "rmse": float(np.sqrt(week_rows["squared_error"].mean())),
            "mae": float(week_rows["absolute_error"].mean()),
            "mean_error": float(week_rows["error"].mean()),
            "top_errors": records(week_rows.nlargest(10, "absolute_error")),
            "closest_predictions": records(week_rows.nsmallest(10, "absolute_error")),
        }
    summary = {
        "season": season,
        "week_requested": week,
        "weeks_scored": sorted(scored["target_week"].unique().astype(int).tolist()),
        "n_predictions": int(len(scored)),
        "mse": float(scored["squared_error"].mean()) if not scored.empty else None,
        "rmse": float(np.sqrt(scored["squared_error"].mean())) if not scored.empty else None,
        "mae": float(scored["absolute_error"].mean()) if not scored.empty else None,
        "mean_error": float(scored["error"].mean()) if not scored.empty else None,
        "top_errors": records(scored.nlargest(10, "absolute_error")),
        "closest_predictions": records(scored.nsmallest(10, "absolute_error")),
        "weekly": records(weekly_summary),
        "weekly_details": weekly_details,
        "rows": records(scored),
    }
    if week is not None:
        selected = scored[scored["target_week"].eq(week)]
        summary["selected_week"] = {
            "week": week,
            "n_predictions": int(len(selected)),
            "mse": float(selected["squared_error"].mean()) if not selected.empty else None,
            "rmse": float(np.sqrt(selected["squared_error"].mean())) if not selected.empty else None,
            "mae": float(selected["absolute_error"].mean()) if not selected.empty else None,
            "mean_error": float(selected["error"].mean()) if not selected.empty else None,
            "top_errors": records(selected.nlargest(10, "absolute_error")),
            "closest_predictions": records(selected.nsmallest(10, "absolute_error")),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, default=None, help="Optional week to include as selected_week metrics")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--updated-predictions", type=Path, default=None, help="Optional CSV path for predictions joined to current actual scores")
    args = parser.parse_args()
    default_name = f"performance_{args.season}_week_{args.week}.json" if args.week is not None else f"performance_{args.season}.json"
    output = args.output or PROJECT_ROOT / "outputs" / default_name
    artifact = build_performance(args.predictions, args.data_dir, args.season, args.week)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2))
    updated_predictions = args.updated_predictions or output.with_name(
        f"predictions_with_actuals_{args.season}"
        f"_week_{args.week}.csv" if args.week is not None else f"predictions_with_actuals_{args.season}.csv"
    )
    merge_actuals(args.predictions, args.data_dir, args.season).to_csv(updated_predictions, index=False)
    print(f"Saved performance artifact to {output}")
    print(f"Saved predictions with actuals to {updated_predictions}")
    print(f"Scored {artifact['n_predictions']:,} prediction rows across weeks {artifact['weeks_scored']}")


if __name__ == "__main__":
    main()
