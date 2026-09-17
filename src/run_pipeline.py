"""Refresh data and run evaluation or current-season forecasting."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import dump

from create_tensors import create_tensors, load_tables, load_tensors, save_tensors
from evaluate_predictions import (
    combine_predictions,
    rmse_by_timedelta,
    rmse_by_week,
    rmse_by_week_and_horizon,
)
from train_model import (
    fit_production_model,
    prediction_table,
    run_validation_splits,
    tune_hyperparameters,
)

GET_DATA_SPEC = spec_from_file_location("get_data", Path(__file__).with_name("get-data.py"))
if GET_DATA_SPEC is None or GET_DATA_SPEC.loader is None:
    raise ImportError("Unable to load the data-fetch module")
get_data = module_from_spec(GET_DATA_SPEC)
GET_DATA_SPEC.loader.exec_module(get_data)
fetch_raw_data = get_data.fetch_raw_data
prepare_data = get_data.prepare_data
save_data = get_data.save_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "lstm_config_v1.yaml"


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as config_file:
        return yaml.safe_load(config_file)


def refresh_data(config: dict[str, Any], end_season: int) -> Path:
    """Download and persist every source table required by the pipeline."""
    start_season = int(config["data"].get("start_season", 2019))
    seasons = list(range(start_season, end_season + 1))
    data_dir = PROJECT_ROOT / config["data"].get("data_dir", "data")
    print(f"[PIPELINE] Refreshing data for seasons {start_season}-{end_season}...", flush=True)
    save_data(prepare_data(fetch_raw_data(seasons)), data_dir)
    print("[PIPELINE] Data refresh complete", flush=True)
    return data_dir


def latest_labeled_season(tables: dict[str, pd.DataFrame]) -> int:
    """Find the newest season with weekly player outcomes."""
    weekly = tables["weekly"]
    if "season_type" in weekly:
        weekly = weekly[weekly["season_type"] == "REG"]
    if weekly.empty:
        raise ValueError("No labeled weekly outcomes were found")
    return int(weekly["season"].max())


def resolve_forecast_start_week(
    tables: dict[str, pd.DataFrame],
    forecast_season: int,
    requested_week: int | None,
    regular_season_end: int,
) -> int:
    """Use the first week without current-season weekly outcomes as the forecast start."""
    weekly = tables["weekly"]
    season_weekly = weekly[weekly["season"].astype(int) == forecast_season]
    latest_week = int(season_weekly["week"].max()) if not season_weekly.empty else 0
    expected_week = min(latest_week + 1, regular_season_end)
    start_week = expected_week if requested_week is None else int(requested_week)
    if not 1 <= start_week <= regular_season_end:
        raise ValueError(f"start week must be between 1 and {regular_season_end}")
    if start_week != expected_week:
        raise ValueError(
            f"Forecast start week {start_week} does not match the first unobserved week "
            f"({expected_week}); latest available {forecast_season} weekly/PBP data is week {latest_week}."
        )
    return start_week


def _scale_with(scaler: Any, numeric_inputs: np.ndarray) -> np.ndarray:
    feature_count = numeric_inputs.shape[-1]
    return scaler.transform(numeric_inputs.reshape(-1, feature_count)).reshape(numeric_inputs.shape).astype(np.float32)


def _add_production_intervals(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calibrate production bounds from historical validation residuals."""
    validation_path = PROJECT_ROOT / "outputs" / "temporal_validation.csv"
    if not validation_path.exists() or "horizon_step" not in predictions.columns:
        predictions["prediction_ci_low"] = np.nan
        predictions["prediction_ci_high"] = np.nan
        return predictions
    validation = pd.read_csv(validation_path, usecols=["horizon_step", "y_true", "y_pred"])
    validation["residual"] = validation["y_true"] - validation["y_pred"]
    bounds = validation.groupby("horizon_step")["residual"].quantile([0.025, 0.975]).unstack()
    bounds.columns = ["residual_low", "residual_high"]
    bounds = bounds.reset_index()
    result = predictions.merge(bounds, on="horizon_step", how="left")
    global_bounds = validation["residual"].quantile([0.025, 0.975])
    result["residual_low"] = result["residual_low"].fillna(global_bounds.loc[0.025])
    result["residual_high"] = result["residual_high"].fillna(global_bounds.loc[0.975])
    result["prediction_ci_low"] = (result["y_pred"] + result["residual_low"]).clip(lower=0.0)
    result["prediction_ci_high"] = result["y_pred"] + result["residual_high"]
    return result.drop(columns=["residual_low", "residual_high"])


def _load_or_build_tensors(
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
    cache_path: Path,
    rebuild: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load a tensor cache or build it when missing/explicitly invalidated."""
    signature_payload = {
        "config": config,
        "kwargs": kwargs,
        "tables": {
            name: {
                "rows": len(frame),
                "seasons": sorted(frame["season"].dropna().astype(int).unique().tolist()) if "season" in frame else [],
                "max_week": int(frame["week"].max()) if "week" in frame and not frame.empty else None,
            }
            for name, frame in tables.items()
        },
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True, default=str).encode()).hexdigest()
    if cache_path.exists() and not rebuild:
        print(f"[PIPELINE] Reusing tensor cache {cache_path}...", flush=True)
        tensors = load_tensors(cache_path)
        if tensors.get("cache_signature") == signature:
            print(f"[PIPELINE] Loaded cached tensors ({len(tensors['metadata']):,} samples)", flush=True)
            return tensors
        print("[PIPELINE] Tensor cache is stale; rebuilding", flush=True)
    print(f"[PIPELINE] Building tensors for cache {cache_path}...", flush=True)
    tensors = create_tensors(tables, config, **kwargs)
    tensors["cache_signature"] = signature
    save_tensors(tensors, cache_path)
    print(f"[PIPELINE] Saved tensor cache ({len(tensors['metadata']):,} samples)", flush=True)
    return tensors


def run_evaluation(
    config: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    tuning_seasons: list[int],
    test_seasons: list[int],
    epochs: int | None,
    output_path: Path,
    tune: bool = False,
    trials: int = 10,
    rebuild_tensors: bool = False,
    tensor_cache_dir: Path | None = None,
    refit_on_tuning: bool = False,
    validation_week: int | None = None,
) -> None:
    """Train on all seasons before the tune season, tune on the chosen season, and evaluate on the final test season."""
    tuning_seasons = sorted(set(int(season) for season in tuning_seasons))
    test_seasons = sorted(set(int(season) for season in test_seasons))
    if not tuning_seasons:
        raise ValueError("Provide at least one tune season")
    if not test_seasons:
        raise ValueError("Provide at least one test season")
    train_upto = min(tuning_seasons) - 1
    train_seasons = [season for season in sorted(set(int(value) for value in tables["weekly"]["season"].dropna().unique())) if season < min(tuning_seasons)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = tensor_cache_dir or output_path.parent
    training_tensors = _load_or_build_tensors(
        tables,
        config,
        cache_dir / "validation_training_tensors.npz",
        rebuild_tensors,
    )
    validation_tensors = {
        season: _load_or_build_tensors(
            tables,
            config,
            cache_dir / f"validation_tensors_{season}.npz",
            rebuild_tensors,
            active_season=season,
            forecast_season=season,
            forecast_start_week=validation_week if season == max(test_seasons) else None,
        )
        for season in sorted(set(tuning_seasons + test_seasons))
    }
    print(f"[PIPELINE] Train seasons: {train_seasons}", flush=True)
    print(f"[PIPELINE] Tuning seasons: {tuning_seasons}; test seasons: {test_seasons}", flush=True)
    best_epoch = epochs
    if tune and tuning_seasons:
        print(f"[PIPELINE] Tuning hyperparameters with {trials} trials on {tuning_seasons}...", flush=True)
        config, tuning_results, best_epoch = tune_hyperparameters(training_tensors, config, tuning_seasons, trials, epochs, validation_tensors)
        tuning_results.to_csv(output_path.with_name(f"{output_path.stem}_tuning.csv"), index=False)
        output_path.with_name(f"{output_path.stem}_best_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"[PIPELINE] Best hyperparameters saved alongside {output_path}", flush=True)
    if refit_on_tuning:
        print(f"[PIPELINE] Refitting on train+tune seasons before final test: {train_seasons + tuning_seasons}", flush=True)
    if not test_seasons:
        raise ValueError("No evaluation season was provided")
    predictions, histories = run_validation_splits(training_tensors, config, test_seasons, best_epoch, validation_tensors)
    print("[PIPELINE] Combining overlapping validation forecasts by target week...", flush=True)
    combined = combine_predictions(predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[PIPELINE] Saving validation artifacts to {output_path.parent}...", flush=True)
    predictions.to_csv(output_path, index=False)
    combined.to_csv(output_path.with_name(f"{output_path.stem}_combined.csv"), index=False)
    raw_horizon = rmse_by_timedelta(predictions)
    combined_horizon = rmse_by_timedelta(combined)
    raw_week = rmse_by_week(predictions)
    combined_week = rmse_by_week(combined)
    raw_week_horizon = rmse_by_week_and_horizon(predictions)
    combined_week_horizon = rmse_by_week_and_horizon(combined)
    raw_overall = float(np.sqrt(np.mean((predictions["y_pred"] - predictions["y_true"]) ** 2)))
    combined_overall = float(np.sqrt(np.mean((combined["y_pred"] - combined["y_true"]) ** 2)))
    metrics = {
        "train_seasons": train_seasons,
        "tuning_seasons": tuning_seasons,
        "test_seasons": test_seasons,
        "epochs": {str(season): len(history) for season, history in histories.items()},
        "raw_rmse_overall": raw_overall,
        "combined_rmse_overall": combined_overall,
        "raw_rmse_by_horizon": raw_horizon.to_dict(orient="records"),
        "combined_rmse_by_horizon": combined_horizon.to_dict(orient="records"),
        "raw_rmse_by_week": raw_week.to_dict(orient="records"),
        "combined_rmse_by_week": combined_week.to_dict(orient="records"),
        "raw_rmse_by_week_and_horizon": raw_week_horizon.to_dict(orient="records"),
        "combined_rmse_by_week_and_horizon": combined_week_horizon.to_dict(orient="records"),
    }
    output_path.with_name(f"{output_path.stem}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved raw predictions to {output_path}")
    print(f"Saved combined predictions to {output_path.with_name(f'{output_path.stem}_combined.csv')}")
    print(f"Raw overall RMSE: {raw_overall:.4f}")
    print(f"Combined overall RMSE: {combined_overall:.4f}")
    print("Raw RMSE by horizon:")
    print(raw_horizon.to_string(index=False))
    print("Combined RMSE by horizon:")
    print(combined_horizon.to_string(index=False))
    print("Raw RMSE by week:")
    print(raw_week.to_string(index=False))
    print("Combined RMSE by week:")
    print(combined_week.to_string(index=False))


def run_forecast(
    config: dict[str, Any],
    tables: dict[str, pd.DataFrame],
    forecast_season: int,
    epochs: int | None,
    output_path: Path,
    tune: bool = False,
    trials: int = 10,
    rebuild_tensors: bool = False,
    tensor_cache_dir: Path | None = None,
    forecast_start_week: int = 1,
) -> None:
    """Fit on labeled history and forecast active players for one season."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = tensor_cache_dir or output_path.parent
    training_tensors = _load_or_build_tensors(
        tables,
        config,
        cache_dir / "training_tensors.npz",
        rebuild_tensors,
    )
    labeled_season = min(forecast_season - 1, latest_labeled_season(tables))
    train_indices = np.asarray([
        index for index, sample in enumerate(training_tensors["metadata"])
        if max(sample["target_seasons"]) <= labeled_season
    ], dtype=int)
    if not len(train_indices):
        raise ValueError("No labeled training windows are available")
    if tune:
        validation_seasons = [
            int(season)
            for season in config.get("evaluation", {}).get("validation_seasons", [])
            if int(season) <= labeled_season
        ]
        if not validation_seasons:
            raise ValueError("Tuning requires configured validation seasons before the forecast season")
        print(f"[PIPELINE] Tuning hyperparameters with {trials} trials...", flush=True)
        config, tuning_results, best_epoch = tune_hyperparameters(training_tensors, config, validation_seasons, trials, epochs)
        tuning_results.to_csv(output_path.with_name(f"{output_path.stem}_tuning.csv"), index=False)
        output_path.with_name(f"{output_path.stem}_best_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"[PIPELINE] Best hyperparameters saved alongside {output_path}", flush=True)
    print(f"[PIPELINE] Fitting production model on {len(train_indices):,} labeled windows...", flush=True)
    model, history, scaler = fit_production_model(training_tensors, config, train_indices, best_epoch if tune else epochs)

    forecast_tensors = _load_or_build_tensors(
        tables,
        config,
        cache_dir / f"forecast_tensors_{forecast_season}_week_{forecast_start_week}.npz",
        rebuild_tensors,
        active_season=forecast_season,
        forecast_season=forecast_season,
        forecast_start_week=forecast_start_week,
    )
    scaled_forecast = _scale_with(scaler, forecast_tensors["X_numeric"])
    forecast_inputs = {
        "numeric": scaled_forecast,
        "categorical": forecast_tensors["X_categorical"],
        "played_mask": forecast_tensors["input_played_mask"],
    }
    forecast_values = model.predict(forecast_inputs, verbose=0)
    print("[PIPELINE] Saving model, scaler, and forecast rows...", flush=True)
    forecast_rows = prediction_table(
        forecast_tensors,
        forecast_values,
        np.arange(len(forecast_tensors["metadata"])),
    )
    forecast_rows = _add_production_intervals(forecast_rows)
    forecast_rows = forecast_rows[forecast_rows["target_week"].between(forecast_start_week, int(config["data"]["regular_season_weeks"][1]))].copy()
    forecast_rows["forecast_start_week"] = forecast_start_week
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_rows.to_csv(output_path, index=False)
    model.save(output_path.with_suffix(".keras"))
    dump(scaler, output_path.with_name(f"{output_path.stem}_scaler.joblib"))
    print(f"Trained for {len(history)} epochs on {len(train_indices)} windows")
    print(f"Saved {len(forecast_rows)} forecast rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["evaluate", "forecast"], required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--end-season", type=int, default=date.today().year)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download and rebuild the prepared CSV tables before running",
    )
    parser.add_argument("--tune-season", type=int, nargs="+", default=None, help="One or more seasons used for hyperparameter tuning")
    parser.add_argument("--test-season", type=int, nargs="+", default=None, help="One or more seasons used as the final holdout test set")
    parser.add_argument("--validation-seasons", type=int, nargs="+", default=None, help="Deprecated alias: use --tune-season and --test-season")
    parser.add_argument("--validation-week", type=int, default=None, help="Latest completed week to use for validation/tuning")
    parser.add_argument("--forecast-season", type=int, default=None)
    parser.add_argument("--start-week", type=int, default=None, help="First unobserved target week; defaults to latest available week + 1")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--tune", action="store_true", help="Run YAML hyperparameter search before final training")
    parser.add_argument("--trials", type=int, default=10, help="Number of randomly sampled hyperparameter trials")
    parser.add_argument("--rebuild-tensors", action="store_true", help="Rebuild tensor caches instead of reusing them")
    parser.add_argument("--tensor-cache-dir", type=Path, default=None, help="Directory for saved tensor caches")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--refit-on-tuning", action="store_true", help="Optionally refit on train+tune seasons before the final test evaluation")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = PROJECT_ROOT / config["data"].get("data_dir", "data")
    required_tables = ["weekly", "team_weekly_stats", "schedules", "rosters"]
    if args.refresh_data:
        data_dir = refresh_data(config, args.end_season)
    elif not all((data_dir / f"{name}.csv").exists() for name in required_tables):
        missing = [name for name in required_tables if not (data_dir / f"{name}.csv").exists()]
        raise FileNotFoundError(
            f"Missing prepared data tables: {missing}. Re-run with --refresh-data."
        )
    else:
        print(f"[PIPELINE] Reusing prepared data in {data_dir} (use --refresh-data to rebuild)", flush=True)
    tables = load_tables(data_dir)
    forecast_season = args.forecast_season or args.end_season
    forecast_start_week = None
    if args.mode == "forecast":
        forecast_start_week = resolve_forecast_start_week(
            tables,
            forecast_season,
            args.start_week,
            int(config["data"]["regular_season_weeks"][1]),
        )
        print(f"[PIPELINE] Forecast start week: {forecast_start_week} (latest observed week: {forecast_start_week - 1})", flush=True)
    default_output = (
        PROJECT_ROOT / "outputs" / "validation_predictions.csv"
        if args.mode == "evaluate"
        else PROJECT_ROOT / "outputs" / f"production_predictions_{forecast_season}_week_{forecast_start_week}.csv"
    )
    output_path = args.output or default_output
    if args.mode == "evaluate":
        if args.tune_season is not None or args.test_season is not None:
            tuning_seasons = args.tune_season or []
            test_seasons = args.test_season or []
        else:
            validation_seasons = args.validation_seasons or config.get("evaluation", {}).get("validation_seasons", [])
            if len(validation_seasons) < 2:
                raise ValueError("Provide at least two seasons or use --tune-season and --test-season explicitly")
            tuning_seasons = validation_seasons[:-1]
            test_seasons = validation_seasons[-1:]
        if not tuning_seasons or not test_seasons:
            raise ValueError("Provide --tune-season and --test-season or a two-season validation list")
        run_evaluation(config, tables, tuning_seasons, test_seasons, args.epochs, output_path, args.tune, args.trials, args.rebuild_tensors or args.refresh_data, args.tensor_cache_dir, args.refit_on_tuning, args.validation_week)
    else:
        run_forecast(config, tables, forecast_season, args.epochs, output_path, args.tune, args.trials, args.rebuild_tensors or args.refresh_data, args.tensor_cache_dir, forecast_start_week)


if __name__ == "__main__":
    main()
