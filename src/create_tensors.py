"""Build leakage-safe player sequences for the configured LSTM experiment."""

from __future__ import annotations

import argparse
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "lstm_config_v1.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "lstm_v1_tensors.npz"


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load a YAML experiment configuration."""
    with config_path.open() as config_file:
        return yaml.safe_load(config_file)


def load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load prepared CSV tables and remove legacy index columns."""
    names = ["weekly", "team_weekly_stats", "schedules", "rosters"]
    tables = {}
    for name in names:
        frame = pd.read_csv(data_dir / f"{name}.csv", low_memory=False)
        tables[name] = frame.drop(columns=["Unnamed: 0"], errors="ignore")
    return tables


def _data_module():
    """Load get-data.py, whose filename is retained for CLI compatibility."""
    path = PROJECT_ROOT / "src" / "get-data.py"
    spec = spec_from_file_location("get_data", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _numeric_value(value: Any, missing_value: float) -> float:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return missing_value if pd.isna(value) else float(value)


def _schedule_opponent(schedules: pd.DataFrame, team: str, season: int, week: int) -> str:
    games = schedules[(schedules["season"] == season) & (schedules["week"] == week)]
    if games.empty:
        return "UNKNOWN"
    game = games[(games["home_team"] == team) | (games["away_team"] == team)]
    if game.empty:
        return "UNKNOWN"
    row = game.iloc[0]
    return row["away_team"] if row["home_team"] == team else row["home_team"]


def _feature_columns(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    groups = config["data"]["input_groups"]
    raw_columns = []
    for group in groups.values():
        raw_columns.extend(group["columns"])
    rolling_columns = [
        f"{column}_rolling_{window}"
        for column in raw_columns
        for window in config["data"].get("rolling_windows", [])
    ]
    return raw_columns, rolling_columns


def _build_player_timeline(
    player_name: str,
    season: int,
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Create one calendar-aligned row per regular-season week."""
    start_week, end_week = config["data"]["regular_season_weeks"]
    positions = config["experiment"]["positions"]
    weekly = tables["weekly"]
    rosters = tables["rosters"]
    weekly_player = weekly[
        (weekly["player_display_name"] == player_name)
        & (weekly["season"] == season)
        & weekly["position"].isin(positions)
    ]
    roster_player = rosters[
        (rosters["player_name"] == player_name)
        & (rosters["season"] == season)
        & rosters["position"].isin(positions)
        & rosters["week"].between(start_week, end_week)
    ]
    if roster_player.empty and weekly_player.empty:
        return pd.DataFrame()

    position = (roster_player["position"].mode().iloc[0] if not roster_player.empty else weekly_player["position"].iloc[0])
    team_by_week = roster_player.drop_duplicates("week").set_index("week")["team"].to_dict()
    weekly_by_week = weekly_player.drop_duplicates("week").set_index("week")
    rows = []
    for week in range(start_week, end_week + 1):
        weekly_row = weekly_by_week.loc[week] if week in weekly_by_week.index else None
        team = weekly_row["recent_team"] if weekly_row is not None else team_by_week.get(week)
        if team is None and weekly_row is not None:
            team = weekly_row["recent_team"]
        team = team if pd.notna(team) else "UNKNOWN"
        opponent = _schedule_opponent(tables["schedules"], team, season, week)
        row = {
            "player_name": player_name,
            "season": season,
            "week": week,
            "position": position,
            "team": team,
            "opponent_team": opponent,
            "played": int(weekly_row is not None),
        }
        for column in config["data"]["input_groups"]["player"]["columns"]:
            row[column] = _numeric_value(weekly_row.get(column), config["data"]["missing_value"]) if weekly_row is not None else config["data"]["missing_value"]
        row["target_played"] = row["played"]
        rows.append(row)

    timeline = pd.DataFrame(rows)
    team_stats = tables["team_weekly_stats"]
    team_stats = team_stats.rename(columns={"posteam": "team", "defteam": "opponent_team"})
    team_columns = config["data"]["input_groups"]["team_offense"]["columns"]
    opponent_columns = config["data"]["input_groups"]["opponent_defense"]["columns"]
    lookup_columns = ["season", "week", "team", "opponent_team"]
    team_lookup = {
        tuple(row[column] for column in lookup_columns): row
        for _, row in team_stats.iterrows()
    }
    for group_name, columns in [("team_offense", team_columns), ("opponent_defense", opponent_columns)]:
        for column in columns:
            output_column = f"{group_name}__{column}"
            values = []
            for _, row in timeline.iterrows():
                key = (season, row["week"], row["team"], row["opponent_team"])
                lookup_row = team_lookup.get(key)
                if lookup_row is not None and column in lookup_row.index:
                    value = lookup_row[column]
                    values.append(_numeric_value(value, config["data"]["missing_value"]))
                else:
                    values.append(config["data"]["missing_value"])
            timeline[output_column] = values
    return timeline


def _add_rolling_features(timeline: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add causal rolling means; each row uses only earlier rows."""
    rolling_features = {}
    for group_name, group in config["data"]["input_groups"].items():
        prefix = "" if group_name == "player" else f"{group_name}__"
        for column in group["columns"]:
            source = f"{prefix}{column}"
            if source not in timeline.columns:
                continue
            for window in config["data"].get("rolling_windows", []):
                rolling_features[f"{source}_rolling_{window}"] = timeline[source].shift(1).rolling(window, min_periods=1).mean().fillna(0.0)
    return pd.concat([timeline, pd.DataFrame(rolling_features, index=timeline.index)], axis=1)


def _add_cold_start_history(timeline: pd.DataFrame, config: dict[str, Any], forecast_season: int) -> pd.DataFrame:
    """Pad a current-season-only player so the model can forecast week one."""
    start_week, end_week = config["data"]["regular_season_weeks"]
    if timeline.empty or int(timeline["season"].min()) < forecast_season:
        return timeline

    history = timeline.iloc[: end_week - start_week + 1].copy()
    history["season"] = forecast_season - 1
    history["week"] = range(start_week, end_week + 1)
    history["played"] = 0
    history["target_played"] = 0
    for group_name, group in config["data"]["input_groups"].items():
        prefix = "" if group_name == "player" else f"{group_name}__"
        for column in group["columns"]:
            history[f"{prefix}{column}"] = config["data"]["missing_value"]
    return pd.concat([history, timeline], ignore_index=True)


def create_tensors(
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
    player_name: str | None = None,
    active_season: int | None = None,
    forecast_season: int | None = None,
    forecast_start_week: int | None = None,
) -> dict[str, Any]:
    """Create numeric/categorical input tensors and multi-week targets.

    ``active_season`` limits players to those active on that season's roster;
    ``forecast_season`` limits generated samples to forecast targets in that
    season, which is useful when building a current-season prediction set.
    """
    start_week, end_week = config["data"]["regular_season_weeks"]
    if forecast_start_week is not None and not start_week <= forecast_start_week <= end_week:
        raise ValueError(f"forecast_start_week must be between {start_week} and {end_week}")
    lookback = config["data"]["lookback"]
    horizon = config["data"]["look_forward"]
    weekly = tables["weekly"]
    roster_names = tables["rosters"]["player_name"]
    positions = set(config["experiment"]["positions"])
    if player_name:
        names = [player_name]
    elif active_season is not None:
        weekly_players = tables["weekly"][(tables["weekly"]["season"] == active_season) & tables["weekly"]["position"].isin(positions)]
        active_rosters = tables["rosters"][(tables["rosters"]["season"] == active_season) & (tables["rosters"]["status"] == "ACT") & tables["rosters"]["position"].isin(positions)]
        names = sorted(set(weekly_players["player_display_name"].dropna()) | set(active_rosters["player_name"].dropna()))
    else:
        names = sorted(set(roster_names.dropna()))
    names = [name for name in names if not tables["rosters"][(tables["rosters"]["player_name"] == name) & tables["rosters"]["position"].isin(positions)].empty]
    print(f"[TENSORS] Building sequences for {len(names):,} players", flush=True)
    print(f"[TENSORS] Configuration: {lookback} weeks in -> {horizon} weeks out; rolling windows={config['data'].get('rolling_windows', [])}", flush=True)
    raw_columns, rolling_columns = _feature_columns(config)
    numeric_names = []
    for group_name, group in config["data"]["input_groups"].items():
        columns = group["columns"]
        prefix = "" if group_name == "player" else f"{group_name}__"
        numeric_names.extend([f"{prefix}{column}" for column in columns])
    numeric_names.extend([f"{name}_rolling_{window}" for name in numeric_names for window in config["data"].get("rolling_windows", [])])
    categorical_names = config["data"]["categorical_features"]
    categorical_values = {name: sorted(set(["UNKNOWN"] + tables["rosters"].get("team", pd.Series(dtype=str)).dropna().astype(str).tolist())) for name in categorical_names}
    categorical_values["position"] = sorted(set(["UNKNOWN"] + list(positions)))
    categorical_values["opponent_team"] = sorted(set(["UNKNOWN"] + tables["schedules"]["home_team"].dropna().astype(str).tolist() + tables["schedules"]["away_team"].dropna().astype(str).tolist()))
    vocabularies = {name: {value: index for index, value in enumerate(values)} for name, values in categorical_values.items()}

    numeric_sequences, categorical_sequences, targets, input_masks, target_masks, metadata = [], [], [], [], [], []
    seasons = sorted(set(weekly["season"].dropna().astype(int)) | set(tables["rosters"]["season"].dropna().astype(int)))
    for player_index, name in enumerate(names, start=1):
        if player_index == 1 or player_index % 25 == 0 or player_index == len(names):
            print(f"[TENSORS] Processing player {player_index:,}/{len(names):,}: {name}", flush=True)
        season_timelines = [
            _build_player_timeline(name, season, tables, config)
            for season in seasons
        ]
        season_timelines = [timeline for timeline in season_timelines if not timeline.empty]
        if not season_timelines:
            continue
        timeline = _add_rolling_features(
            pd.concat(season_timelines, ignore_index=True).sort_values(["season", "week"]),
            config,
        ).reset_index(drop=True)
        if forecast_season is not None:
            timeline = _add_cold_start_history(timeline, config, forecast_season)
        target_indices = range(
            lookback,
            len(timeline) if forecast_start_week is not None else len(timeline) - horizon + 1,
        )
        for target_index in target_indices:
            input_frame = timeline.iloc[target_index - lookback:target_index]
            target_frame = timeline.iloc[target_index:target_index + horizon].copy()
            if forecast_start_week is not None and len(target_frame) < horizon:
                padding_count = horizon - len(target_frame)
                padding = target_frame.iloc[[-1]].copy() if not target_frame.empty else timeline.iloc[[-1]].copy()
                padding = pd.concat([padding] * padding_count, ignore_index=True)
                padding["week"] = range(int(target_frame.iloc[-1]["week"]) + 1, int(target_frame.iloc[-1]["week"]) + padding_count + 1)
                padding["played"] = 0
                padding["target_played"] = 0
                for group_name, group in config["data"]["input_groups"].items():
                    prefix = "" if group_name == "player" else f"{group_name}__"
                    for column in group["columns"]:
                        padding[f"{prefix}{column}"] = config["data"]["missing_value"]
                target_frame = pd.concat([target_frame, padding], ignore_index=True)
            if not config["data"].get("allow_cross_season_targets", False) and target_frame["season"].nunique() != 1:
                continue
            if forecast_season is not None and int(target_frame.iloc[0]["season"]) != forecast_season:
                continue
            if forecast_start_week is not None and int(target_frame.iloc[0]["week"]) != forecast_start_week:
                continue
            numeric_sequences.append(input_frame[numeric_names].to_numpy(dtype=np.float32))
            categorical_sequences.append(np.array([[vocabularies[column].get(str(row[column]), 0) for column in categorical_names] for _, row in input_frame.iterrows()], dtype=np.int64))
            targets.append(target_frame[config["data"]["target"]].to_numpy(dtype=np.float32) if config["data"]["target"] in target_frame else target_frame.get("fantasy_points_ppr", pd.Series(0.0, index=target_frame.index)).to_numpy(dtype=np.float32))
            input_masks.append(input_frame["played"].to_numpy(dtype=np.float32))
            target_masks.append(target_frame["target_played"].to_numpy(dtype=np.float32))
            metadata.append({
                "player_name": name,
                "origin_season": int(input_frame.iloc[-1]["season"]),
                "origin_week": int(input_frame.iloc[-1]["week"]),
                "target_seasons": target_frame["season"].astype(int).tolist(),
                "target_weeks": target_frame["week"].astype(int).tolist(),
            })
    if not numeric_sequences:
        raise ValueError("No tensor samples were created")
    print(f"[TENSORS] Finished: {len(numeric_sequences):,} samples; numeric shape={np.stack(numeric_sequences).shape}; target shape={np.stack(targets).shape}", flush=True)
    return {
        "X_numeric": np.stack(numeric_sequences),
        "X_categorical": np.stack(categorical_sequences),
        "y": np.stack(targets),
        "input_played_mask": np.stack(input_masks),
        "target_played_mask": np.stack(target_masks),
        "feature_names": numeric_names,
        "categorical_features": categorical_names,
        "vocabularies": vocabularies,
        "metadata": metadata,
    }


def save_tensors(tensors: dict[str, Any], output_path: Path = DEFAULT_OUTPUT) -> None:
    """Save arrays and JSON metadata in a portable NPZ artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in tensors.items() if isinstance(value, np.ndarray)}
    arrays["metadata_json"] = np.array(json.dumps({key: value for key, value in tensors.items() if not isinstance(value, np.ndarray)}))
    np.savez_compressed(output_path, **arrays)


def load_tensors(input_path: Path) -> dict[str, Any]:
    """Load arrays and JSON metadata from a saved NPZ tensor artifact."""
    with np.load(input_path, allow_pickle=False) as artifact:
        tensors = {
            key: artifact[key]
            for key in artifact.files
            if key != "metadata_json"
        }
        metadata = json.loads(artifact["metadata_json"].item())
    tensors.update(metadata)
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--player-name", default=None)
    parser.add_argument("--active-season", type=int, default=None)
    parser.add_argument("--forecast-season", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_config(args.config)
    data_dir = PROJECT_ROOT / config["data"]["data_dir"]
    tensors = create_tensors(
        load_tables(data_dir),
        config,
        args.player_name,
        active_season=args.active_season,
        forecast_season=args.forecast_season,
    )
    save_tensors(tensors, args.output)
    print(f"Saved {tensors['X_numeric'].shape[0]} samples to {args.output}")
    print(f"Numeric shape: {tensors['X_numeric'].shape}; target shape: {tensors['y'].shape}")


if __name__ == "__main__":
    main()
