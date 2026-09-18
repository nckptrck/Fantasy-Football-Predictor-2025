"""Train and evaluate the configurable Keras seq2seq fantasy model."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import yaml
from sklearn.preprocessing import StandardScaler

from evaluate_predictions import rmse_by_timedelta
from model import build_loss, build_model, build_optimizer
from create_tensors import create_tensors, load_tables

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "lstm_config_v1.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "predictions_validation.csv"


class ProgressCallback(tf.keras.callbacks.Callback):
    """Print compact epoch progress during training and tuning."""

    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        logs = logs or {}
        metrics = " ".join(
            f"{name}={value:.4f}"
            for name, value in logs.items()
            if isinstance(value, (float, int))
        )
        print(f"[TRAIN] {self.label} epoch {epoch + 1}: {metrics}", flush=True)


def split_indices(tensors: Mapping[str, Any], validation_season: int) -> tuple[np.ndarray, np.ndarray]:
    """Split by complete target windows without allowing targets across the split."""
    metadata = tensors["metadata"]
    train_indices = []
    validation_indices = []
    for index, sample in enumerate(metadata):
        target_seasons = set(sample["target_seasons"])
        if target_seasons and max(target_seasons) < validation_season:
            train_indices.append(index)
        if target_seasons and min(target_seasons) == validation_season:
            validation_indices.append(index)
    if not train_indices:
        raise ValueError(f"No training samples before validation season {validation_season}")
    if not validation_indices:
        raise ValueError(f"No validation samples for season {validation_season}")
    return np.asarray(train_indices, dtype=int), np.asarray(validation_indices, dtype=int)


def scale_numeric_features(
    numeric_inputs: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> tuple[np.ndarray, StandardScaler]:
    """Fit a feature scaler using training sequences only."""
    scaler = StandardScaler()
    _, _, feature_count = numeric_inputs.shape
    scaler.fit(numeric_inputs[train_indices].reshape(-1, feature_count))
    scaled = scaler.transform(numeric_inputs.reshape(-1, feature_count)).reshape(numeric_inputs.shape)
    return scaled.astype(np.float32), scaler


def scale_validation_features(
    train_inputs: np.ndarray,
    validation_inputs: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit scaling on training windows and apply it to a separate validation set."""
    scaler = StandardScaler()
    _, _, feature_count = train_inputs.shape
    scaler.fit(train_inputs[train_indices].reshape(-1, feature_count))
    scaled_train = scaler.transform(train_inputs.reshape(-1, feature_count)).reshape(train_inputs.shape)
    scaled_validation = scaler.transform(validation_inputs.reshape(-1, feature_count)).reshape(validation_inputs.shape)
    return scaled_train.astype(np.float32), scaled_validation.astype(np.float32), scaler


def prediction_table(
    tensors: Mapping[str, Any],
    predictions: np.ndarray,
    indices: np.ndarray,
) -> pd.DataFrame:
    """Convert model outputs and tensor metadata to evaluator-compatible rows."""
    rows = []
    for row_index, tensor_index in enumerate(indices):
        sample = tensors["metadata"][int(tensor_index)]
        target_seasons = sample["target_seasons"]
        target_weeks = sample["target_weeks"]
        for step, (season, week) in enumerate(zip(target_seasons, target_weeks), start=1):
            rows.append({
                "player_name": sample["player_name"],
                "position": sample.get("position", "UNKNOWN"),
                "origin_season": sample["origin_season"],
                "origin_week": sample["origin_week"],
                "target_season": season,
                "target_week": week,
                "horizon_step": step,
                "y_true": float(tensors["y"][tensor_index, step - 1]),
                "y_pred": float(predictions[row_index, step - 1]),
                "target_played": bool(tensors["target_played_mask"][tensor_index, step - 1]),
                "forecast_age_days": (step - 1) * 7,
            })
    return pd.DataFrame(rows)


def train_and_validate(
    tensors: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_season: int,
    epochs: int | None = None,
    validation_tensors: Mapping[str, Any] | None = None,
) -> tuple[tf.keras.Model, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Train on pre-validation target windows and return validation predictions."""
    train_indices, validation_indices = split_indices(tensors, validation_season)
    validation_tensors = validation_tensors or tensors
    if validation_tensors is tensors:
        numeric_inputs, scaler = scale_numeric_features(tensors["X_numeric"], train_indices, validation_indices)
        validation_numeric = numeric_inputs[validation_indices]
    else:
        numeric_inputs, validation_numeric, scaler = scale_validation_features(
            tensors["X_numeric"], validation_tensors["X_numeric"], train_indices
        )
    categorical_inputs = tensors["X_categorical"]
    cardinalities = {name: len(values) for name, values in tensors["vocabularies"].items()}
    model = build_model(config, numeric_inputs.shape[-1], cardinalities)
    model.compile(optimizer=build_optimizer(model, config), loss=build_loss(config))
    train_inputs = {
        "numeric": numeric_inputs[train_indices],
        "categorical": categorical_inputs[train_indices],
        "played_mask": tensors["input_played_mask"][train_indices],
    }
    validation_inputs = {
        "numeric": validation_numeric,
        "categorical": validation_tensors["X_categorical"] if validation_tensors is not tensors else categorical_inputs[validation_indices],
        "played_mask": validation_tensors["input_played_mask"] if validation_tensors is not tensors else tensors["input_played_mask"][validation_indices],
    }
    configured_epochs = int(config["training"].get("epochs", 100))
    patience = int(config["training"].get("early_stopping_patience", 10))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ProgressCallback(f"validation {validation_season}"),
    ]
    print(f"[TRAIN] Fitting fold for validation season {validation_season}: {len(train_indices):,} train / {len(validation_indices):,} validation windows", flush=True)
    fit_history = model.fit(
        train_inputs,
        tensors["y"][train_indices],
        sample_weight=tensors["target_played_mask"][train_indices],
        validation_data=(
            validation_inputs,
            validation_tensors["y"] if validation_tensors is not tensors else tensors["y"][validation_indices],
            validation_tensors["target_played_mask"] if validation_tensors is not tensors else tensors["target_played_mask"][validation_indices],
        ),
        batch_size=int(config["training"].get("batch_size", 64)),
        epochs=epochs or configured_epochs,
        shuffle=False,
        callbacks=callbacks,
        verbose=0,
    )
    validation_predictions = model.predict(validation_inputs, verbose=0)
    prediction_rows = prediction_table(
        validation_tensors,
        validation_predictions,
        np.arange(len(validation_tensors["metadata"])) if validation_tensors is not tensors else validation_indices,
    )
    return model, prediction_rows, pd.DataFrame(fit_history.history), scaler


def fit_production_model(
    tensors: Mapping[str, Any],
    config: Mapping[str, Any],
    train_indices: np.ndarray | None = None,
    epochs: int | None = None,
) -> tuple[tf.keras.Model, pd.DataFrame, StandardScaler]:
    """Fit on all supplied training windows for production forecasting."""
    if train_indices is None:
        train_indices = np.arange(len(tensors["metadata"]), dtype=int)
    numeric_inputs, scaler = scale_numeric_features(tensors["X_numeric"], train_indices, train_indices)
    cardinalities = {name: len(values) for name, values in tensors["vocabularies"].items()}
    model = build_model(config, numeric_inputs.shape[-1], cardinalities)
    model.compile(optimizer=build_optimizer(model, config), loss=build_loss(config))
    train_inputs = {
        "numeric": numeric_inputs[train_indices],
        "categorical": tensors["X_categorical"][train_indices],
        "played_mask": tensors["input_played_mask"][train_indices],
    }
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=int(config["training"].get("early_stopping_patience", 10)), restore_best_weights=True),
        ProgressCallback("production"),
    ]
    print(f"[TRAIN] Fitting production model: {len(train_indices):,} windows, {numeric_inputs.shape[-1]} numeric features, {int(config['data']['look_forward'])}-week horizon", flush=True)
    fit_history = model.fit(
        train_inputs,
        tensors["y"][train_indices],
        sample_weight=tensors["target_played_mask"][train_indices],
        batch_size=int(config["training"].get("batch_size", 64)),
        epochs=epochs or int(config["training"].get("epochs", 100)),
        shuffle=False,
        callbacks=callbacks,
        verbose=0,
    )
    return model, pd.DataFrame(fit_history.history), scaler


def run_validation_splits(
    tensors: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_seasons: list[int],
    epochs: int | None = None,
    validation_tensors: Mapping[int, Mapping[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Train a fresh model for each temporal validation season."""
    split_predictions = []
    histories = {}
    for validation_season in validation_seasons:
        print(f"[TRAIN] Starting validation fold {validation_season}...", flush=True)
        season_tensors = validation_tensors.get(validation_season) if validation_tensors else None
        _, predictions, history, _ = train_and_validate(tensors, config, validation_season, epochs, season_tensors)
        print(f"[TRAIN] Finished validation fold {validation_season} after {len(history)} epochs", flush=True)
        predictions["validation_season"] = validation_season
        split_predictions.append(predictions)
        histories[validation_season] = history
    return pd.concat(split_predictions, ignore_index=True), histories


def _overall_rmse(predictions: pd.DataFrame) -> float:
    scored = predictions[predictions["target_played"].astype(bool)]
    if scored.empty:
        raise ValueError("No played validation targets are available for tuning")
    return float(np.sqrt(np.mean((scored["y_pred"] - scored["y_true"]) ** 2)))


def _yaml_hyperparameters(hp: kt.HyperParameters, config: Mapping[str, Any]) -> None:
    """Register the YAML search space with KerasTuner."""
    for name, specification in config.get("search_space", {}).items():
        if specification["type"] == "choices":
            values = specification["values"]
            if values and isinstance(values[0], list):
                hp.Int(f"{name}__index", min_value=0, max_value=len(values) - 1, step=1)
            else:
                hp.Choice(name, values=values)
        elif specification["type"] == "int":
            hp.Int(name, min_value=int(specification["start"]), max_value=int(specification["stop"]), step=int(specification["step"]))
        elif specification["type"] == "float":
            hp.Float(name, min_value=float(specification["start"]), max_value=float(specification["stop"]), step=float(specification["step"]))
        else:
            raise ValueError(f"Unsupported search-space type: {specification['type']}")


def _set_config_value(config: dict[str, Any], name: str, value: Any) -> None:
    target = config
    path = name.split(".")
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _config_from_hyperparameters(config: Mapping[str, Any], hp: kt.HyperParameters) -> dict[str, Any]:
    """Create a model config from one KerasTuner trial."""
    candidate = deepcopy(dict(config))
    for name, specification in config.get("search_space", {}).items():
        if specification["type"] == "choices" and specification["values"] and isinstance(specification["values"][0], list):
            value = specification["values"][hp.get(f"{name}__index")]
        else:
            value = hp.get(name)
        _set_config_value(candidate, name, value)
    return candidate


def _hyperparameter_values(hp: kt.HyperParameters) -> dict[str, Any]:
    """Return serializable trial values for logs and CSV artifacts."""
    return {name: value for name, value in hp.values.items()}


def tune_hyperparameters(
    tensors: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_seasons: list[int],
    trials: int,
    epochs: int | None = None,
    validation_tensors: Mapping[int, Mapping[str, Any]] | None = None,
    verbose: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame, int]:
    """Use KerasTuner Bayesian optimization and return the winning config."""
    if not config.get("search_space"):
        raise ValueError("The configured search space is empty")
    seed = int(config.get("experiment", {}).get("random_seed", 42))
    trial_records: dict[str, dict[str, Any]] = {}

    class FoldTuner(kt.engine.tuner.Tuner):
        def run_trial(self, trial: kt.engine.trial.Trial, *args: Any, **kwargs: Any) -> None:
            trial_number = len(trial_records) + 1
            candidate = _config_from_hyperparameters(config, trial.hyperparameters)
            print(f"\n[TUNE] Trial {trial_number}/{trials}", flush=True)
            print(f"[TUNE] Hyperparameters: {json.dumps(_hyperparameter_values(trial.hyperparameters), sort_keys=True)}", flush=True)
            predictions, histories = run_validation_splits(tensors, candidate, validation_seasons, epochs, validation_tensors)
            score = _overall_rmse(predictions)
            fold_scores = {
                int(season): _overall_rmse(predictions[predictions["validation_season"] == season])
                for season in validation_seasons
            }
            best_epochs = [int(history["val_loss"].idxmin()) + 1 for history in histories.values() if "val_loss" in history]
            record = {
                "trial": trial_number,
                "rmse": score,
                "fold_rmse": json.dumps(fold_scores, sort_keys=True),
                "best_epoch": int(round(np.mean(best_epochs))) if best_epochs else None,
                "final_train_loss": float(np.mean([float(h["loss"].iloc[-1]) for h in histories.values() if "loss" in h])),
                "best_validation_loss": float(np.mean([float(h["val_loss"].min()) for h in histories.values() if "val_loss" in h])),
                **_hyperparameter_values(trial.hyperparameters),
            }
            trial_records[trial.trial_id] = record
            self.oracle.update_trial(trial.trial_id, {"val_rmse": score})
            best_so_far = min((item["rmse"] for item in trial_records.values()), default=score)
            print(f"[TUNE] Fold RMSE: {fold_scores}", flush=True)
            print(f"[TUNE] Trial RMSE: {score:.4f} | Best so far: {best_so_far:.4f}", flush=True)

    def hypermodel(hp: kt.HyperParameters) -> tf.keras.Model:
        _yaml_hyperparameters(hp, config)
        return tf.keras.Sequential()

    tuner = FoldTuner(
        oracle=kt.oracles.BayesianOptimizationOracle(
            objective=kt.Objective("val_rmse", direction="min"),
            max_trials=max(int(trials), 1),
            seed=seed,
        ),
        hypermodel=hypermodel,
        directory=str(PROJECT_ROOT / "outputs" / "keras_tuner"),
        project_name="lstm_v1",
        overwrite=True,
    )
    tuner.search()
    best_trial = tuner.oracle.get_best_trials(1)[0]
    best_config = _config_from_hyperparameters(config, best_trial.hyperparameters)
    results_frame = pd.DataFrame(trial_records.values()).sort_values("rmse").reset_index(drop=True)
    best_epoch = int(results_frame.iloc[0]["best_epoch"] or (epochs or best_config["training"].get("epochs", 100)))
    print(f"[TUNE] Best trial RMSE: {results_frame.iloc[0]['rmse']:.4f}; refit epochs: {best_epoch}", flush=True)
    return best_config, results_frame, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validation-season", type=int, default=None)
    parser.add_argument("--validation-seasons", type=int, nargs="+", default=None)
    parser.add_argument("--active-season", type=int, default=None)
    parser.add_argument("--player-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    data_dir = PROJECT_ROOT / config["data"]["data_dir"]
    tables = load_tables(data_dir)
    latest_season = int(tables["weekly"]["season"].max())
    configured_validation_seasons = config.get("evaluation", {}).get("validation_seasons", [latest_season])
    validation_seasons = args.validation_seasons or ([args.validation_season] if args.validation_season else configured_validation_seasons)
    active_season = args.active_season or latest_season
    tensors = create_tensors(
        tables,
        config,
        player_name=args.player_name,
        active_season=active_season if args.player_name is None else None,
    )
    predictions, histories = run_validation_splits(tensors, config, validation_seasons, args.epochs)
    print(f"Players from season {active_season}: {predictions['player_name'].nunique()}")
    print(f"Validation seasons: {validation_seasons}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Training epochs: { {season: len(history) for season, history in histories.items()} }")
    print(f"Validation predictions: {len(predictions)} rows")
    print(rmse_by_timedelta(predictions).to_string(index=False))
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
