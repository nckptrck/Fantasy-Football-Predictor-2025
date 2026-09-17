"""Download and prepare the nflverse tables used by the forecasting pipeline.

The functions in this module are intentionally independent of model code so
they can be called from notebooks, scheduled jobs, or the command line.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError

import nfl_data_py as nfl
import pandas as pd


NFLVERSE_RELEASE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
FIRST_SEASON = 2019
LAST_REGULAR_WEEK = 17
SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}


def available_seasons(start_season: int = FIRST_SEASON) -> list[int]:
    """Return seasons through the current calendar year.

    nflverse publishes partial data during an active season, which makes this
    useful for weekly updates without changing the script every year.
    """
    current_season = date.today().year
    if start_season > current_season:
        raise ValueError("start_season cannot be after the current year")
    return list(range(start_season, current_season + 1))


def _ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Add absent numeric nflverse columns as zero-filled columns."""
    frame = frame.copy()
    for column in columns:
        if column not in frame:
            frame[column] = 0
    return frame


def fetch_raw_data(seasons: list[int], roster_seasons: list[int] | None = None) -> dict[str, pd.DataFrame]:
    """Fetch the raw nflverse tables for the requested seasons."""
    if roster_seasons is None:
        roster_seasons = seasons
    weekly_frames = []
    available_weekly_seasons = []
    for season in seasons:
        print(f"[DATA] Downloading weekly player stats for {season}...", flush=True)
        try:
            weekly_frames.append(_fetch_current_weekly_data(season))
            available_weekly_seasons.append(season)
            print(f"[DATA] Weekly player stats for {season} loaded ({len(weekly_frames[-1]):,} rows)", flush=True)
        except HTTPError as error:
            if error.code != 404:
                raise
            print(f"[DATA] No weekly player stats published for {season}; skipping", flush=True)
    if not weekly_frames:
        raise ValueError("No weekly player-stat releases are available for the requested seasons")
    weekly = pd.concat(weekly_frames, ignore_index=True)
    print("[DATA] Downloading schedules...", flush=True)
    schedules = _fetch_current_schedules(seasons)
    print(f"[DATA] Schedules loaded ({len(schedules):,} rows)", flush=True)
    print(f"[DATA] Downloading play-by-play for {available_weekly_seasons}...", flush=True)
    pbp = nfl.import_pbp_data(available_weekly_seasons)
    print(f"[DATA] Play-by-play loaded ({len(pbp):,} rows)", flush=True)
    print(f"[DATA] Downloading weekly rosters for {roster_seasons}...", flush=True)
    rosters = nfl.import_weekly_rosters(roster_seasons)
    print(f"[DATA] Weekly rosters loaded ({len(rosters):,} rows)", flush=True)
    return {
        "weekly": weekly,
        "pbp": pbp,
        "schedules": schedules,
        "rosters": rosters,
    }


def _fetch_current_weekly_data(season: int) -> pd.DataFrame:
    """Fetch maintained weekly player stats and match the legacy column names."""
    url = f"{NFLVERSE_RELEASE_BASE}/stats_player/stats_player_week_{season}.parquet"
    weekly = pd.read_parquet(url)
    weekly = weekly.rename(columns={
        "passing_interceptions": "interceptions",
        "sacks_suffered": "sacks",
        "sack_yards_lost": "sack_yards",
        "team": "recent_team",
    })
    return weekly


def _fetch_current_schedules(seasons: list[int]) -> pd.DataFrame:
    """Fetch the maintained consolidated schedule release."""
    url = f"{NFLVERSE_RELEASE_BASE}/schedules/games.parquet"
    schedules = pd.read_parquet(url)
    return schedules[schedules["season"].isin(seasons)].copy()


def add_weekly_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add aggregate fantasy columns used by downstream feature builders."""
    columns = [
        "rushing_fumbles", "receiving_fumbles", "rushing_fumbles_lost",
        "receiving_fumbles_lost", "passing_first_downs", "rushing_first_downs",
        "receiving_first_downs",
    ]
    weekly = _ensure_columns(weekly, columns)
    weekly = weekly.copy()
    weekly["fumbles"] = weekly["rushing_fumbles"] + weekly["receiving_fumbles"]
    weekly["fumbles_lost"] = weekly["rushing_fumbles_lost"] + weekly["receiving_fumbles_lost"]
    weekly["first_downs"] = (
        weekly["passing_first_downs"]
        + weekly["rushing_first_downs"]
        + weekly["receiving_first_downs"]
    )
    return weekly


def build_team_weekly_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate regular-season offensive statistics by team and game."""
    required = [
        "passing_yards", "rushing_yards", "pass_touchdown", "rush_touchdown",
        "first_down", "yards_gained", "interception", "fumble_lost",
    ]
    pbp = _ensure_columns(pbp, required)
    pbp = pbp[(pbp["week"].between(1, LAST_REGULAR_WEEK)) & (pbp["season_type"] == "REG")].copy()
    pbp["turnover"] = pbp["interception"] + pbp["fumble_lost"]
    return (
        pbp.groupby(["game_id", "posteam", "defteam", "season", "week"], dropna=False)
        .agg(
            team_passing_yards=("passing_yards", "sum"),
            team_rushing_yards=("rushing_yards", "sum"),
            team_passing_tds=("pass_touchdown", "sum"),
            team_rushing_tds=("rush_touchdown", "sum"),
            team_total_plays=("posteam", "count"),
            team_first_downs=("first_down", "sum"),
            team_yards_per_play=("yards_gained", "mean"),
            team_turnovers=("turnover", "sum"),
        )
        .reset_index()
    )


def build_player_list(rosters: pd.DataFrame) -> pd.DataFrame:
    """Return active QB/RB/WR/TE player-week records."""
    return (
        rosters[
            rosters["position"].isin(SKILL_POSITIONS)
            & (rosters["status"] == "ACT")
            & (rosters["week"] < LAST_REGULAR_WEEK + 1)
        ][["season", "week", "team", "player_name", "position"]]
        .sort_values(["season", "week", "team", "player_name", "position"])
        .drop_duplicates(subset=["season", "week", "player_name"])
        .reset_index(drop=True)
    )


def prepare_data(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Transform raw nflverse data into the five persisted project tables."""
    print("[DATA] Adding player weekly features...", flush=True)
    weekly = add_weekly_features(raw["weekly"])
    print("[DATA] Aggregating team weekly statistics...", flush=True)
    team_weekly_stats = build_team_weekly_stats(raw["pbp"])
    print("[DATA] Building active player-week list...", flush=True)
    player_list = build_player_list(raw["rosters"])
    return {
        "weekly": weekly,
        "team_weekly_stats": team_weekly_stats,
        "schedules": raw["schedules"].copy(),
        "player_list": player_list,
        "rosters": raw["rosters"].copy(),
    }


def save_data(data: dict[str, pd.DataFrame], data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Write prepared tables without pandas' synthetic index column."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in data.items():
        print(f"[DATA] Saving {name}.csv ({len(frame):,} rows)...", flush=True)
        frame.to_csv(data_dir / f"{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=FIRST_SEASON)
    parser.add_argument("--end-season", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    end_season = args.end_season or date.today().year
    if end_season < args.start_season:
        parser.error("--end-season must be >= --start-season")
    seasons = list(range(args.start_season, end_season + 1))
    save_data(prepare_data(fetch_raw_data(seasons)), args.data_dir)
    print(f"Saved {len(seasons)} seasons to {args.data_dir}")


if __name__ == "__main__":
    main()
