from pathlib import Path
from math import erf, isfinite, sqrt
import json
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
PRODUCTION_PATH = PROJECT_ROOT / "outputs" / "production_predictions_2026.csv"
COMBINED_PATH = PROJECT_ROOT / "outputs" / "temporal_validation_combined.csv"
RAW_PATH = PROJECT_ROOT / "outputs" / "temporal_validation.csv"


st.set_page_config(page_title="Sunday Signal", page_icon="SS", layout="wide")

st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
	:root { --ink: #f4f1e8; --muted: #9ca39d; --line: #303733; --paper: #111513; --panel: #1a211e; --orange: #ff7650; --green: #9ed6b2; }
	.stApp { background: var(--paper); color: var(--ink); }
	[data-testid="stSidebar"] { background: #171d1a; border-right: 1px solid var(--line); }
	[data-testid="stHeader"] { background: #111513; }
	[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: var(--muted) !important; }
	[data-baseweb="select"] > div, [data-baseweb="input"] > div { background: #202824; border-color: #3a453e; color: var(--ink); }
	[data-testid="stRadio"] label { color: var(--ink) !important; }
	[data-testid="stRadio"] [role="radiogroup"] { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: .35rem .55rem; width: fit-content; }
	[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { color: var(--ink) !important; }
	.stDataFrame, [data-testid="stMetric"] { background: var(--panel); }
	[data-testid="stMetric"] { border: 1px solid var(--line); border-radius: 6px; padding: .75rem; }
	h1, h2, h3, p, label, div { font-family: 'Space Grotesk', sans-serif; }
	h1 { letter-spacing: 0; font-size: 3.2rem; line-height: 1; margin-bottom: .4rem; }
	.eyebrow { color: var(--orange); font-family: 'DM Mono', monospace; font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; }
	.lede { color: var(--muted); font-size: 1.05rem; max-width: 650px; margin-bottom: 1.8rem; }
	.metric-card { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 1rem 1.1rem; min-height: 106px; }
	.metric-label { color: var(--muted); font-family: 'DM Mono', monospace; font-size: .68rem; text-transform: uppercase; }
	.metric-value { color: var(--ink); font-size: 2rem; font-weight: 700; margin-top: .35rem; }
	.metric-note { color: var(--muted); font-size: .78rem; }
	.section-rule { border-top: 1px solid var(--line); margin: 2rem 0 1.2rem; }
	.source-note { color: var(--muted); font-family: 'DM Mono', monospace; font-size: .7rem; }
	</style>
	""",
	unsafe_allow_html=True,
)


def available_prediction_paths() -> list[Path]:
	forecast_paths = sorted(
		(PROJECT_ROOT / "outputs").glob("production_predictions_*.csv"),
		key=lambda path: path.stat().st_mtime,
		reverse=True,
	)
	return forecast_paths or [candidate for candidate in (PRODUCTION_PATH, COMBINED_PATH, RAW_PATH) if candidate.exists()]


def prediction_run_week(path: Path) -> int:
	match = re.match(r"production_predictions_(\d+)(?:_week_(\d+))?\.csv$", path.name)
	if not match:
		return 0
	return int(match.group(2) or 0)


def forecast_path_for(season: int, week: int) -> Path:
	paths = available_prediction_paths()
	pattern = re.compile(r"production_predictions_(\d+)(?:_week_(\d+))?\.csv$")
	matching = []
	for path in paths:
		match = pattern.match(path.name)
		if match and int(match.group(1)) == season:
			path_week = int(match.group(2) or 0)
			if path_week == 0 or path_week <= week:
				matching.append((path_week, path))
	if matching:
		return max(matching, key=lambda item: item[0])[1]
	legacy = PROJECT_ROOT / "outputs" / f"production_predictions_{season}.csv"
	if legacy.exists():
		return legacy
	return paths[0]


def load_predictions(path: Path) -> tuple[pd.DataFrame, str]:
	if not path.exists():
		raise FileNotFoundError("No prediction CSV found in outputs/.")
	match = re.match(r"production_predictions_(\d+)(?:_week_(\d+))?\.csv$", path.name)
	season = int(match.group(1)) if match else None
	season_files = []
	if season is not None:
		season_files = [
			candidate for candidate in available_prediction_paths()
			if re.match(rf"production_predictions_{season}(?:_week_\d+)?\.csv$", candidate.name)
		]
		season_files = sorted(season_files, key=lambda candidate: (prediction_run_week(candidate), candidate.stat().st_mtime))
	if not season_files:
		season_files = [path]
	frames = []
	for candidate in season_files:
		frame = pd.read_csv(candidate)
		required = {"player_name", "target_season", "target_week", "y_pred"}
		missing = required.difference(frame.columns)
		if missing:
			raise ValueError(f"Prediction CSV is missing columns: {', '.join(sorted(missing))}")
		frame["target_week"] = frame["target_week"].astype(int)
		frame["y_pred"] = frame["y_pred"].astype(float)
		if "forecast_start_week" not in frame.columns:
			frame["forecast_start_week"] = prediction_run_week(candidate)
		frames.append(frame)
	predictions = pd.concat(frames, ignore_index=True)
	predictions = predictions.sort_values(["forecast_start_week", "target_week", "player_name"], kind="mergesort")
	predictions = predictions.drop_duplicates(["player_name", "target_season", "target_week"], keep="last")
	if "y_true" in predictions.columns:
		predictions["y_true"] = pd.to_numeric(predictions["y_true"], errors="coerce")
	actuals_paths = sorted((PROJECT_ROOT / "outputs").glob(f"predictions_with_actuals_{season}*.csv"))
	if actuals_paths:
		actuals = pd.concat(
			[
				pd.read_csv(
					path,
					usecols=["player_name", "target_season", "target_week", "y_true"],
					low_memory=False,
				).rename(columns={"y_true": "actual_latest"})
				for path in actuals_paths
			],
			ignore_index=True,
		)
		actuals = actuals.drop_duplicates(["player_name", "target_season", "target_week"], keep="last")
	else:
		actuals_path = PROJECT_ROOT / "data" / "weekly.csv"
		if not actuals_path.exists():
			actuals = None
		else:
			weekly = pd.read_csv(actuals_path, usecols=["player_display_name", "season", "week", "fantasy_points_ppr"], low_memory=False)
			actuals = weekly.rename(
				columns={
					"player_display_name": "player_name",
					"season": "target_season",
					"week": "target_week",
					"fantasy_points_ppr": "actual_latest",
				}
			)
			actuals = actuals.groupby(["player_name", "target_season", "target_week"], as_index=False)["actual_latest"].sum()
	if actuals is not None:
		predictions = predictions.drop(columns=["y_true"], errors="ignore").merge(
			actuals,
			on=["player_name", "target_season", "target_week"],
			how="left",
		)
		predictions["y_true"] = pd.to_numeric(predictions["actual_latest"], errors="coerce")
		predictions = predictions.drop(columns=["actual_latest"])
	if {"prediction_ci_low", "prediction_ci_high"}.issubset(predictions.columns):
		predictions["projection_low"] = predictions["prediction_ci_low"]
		predictions["projection_high"] = predictions["prediction_ci_high"]
	predictions = predictions.drop(columns=["forecast_start_week"], errors="ignore")
	return predictions, path.name


def metric_card(label: str, value: str, note: str = "") -> None:
	st.markdown(
		f'<div class="metric-card"><div class="metric-label">{label}</div>'
		f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
		unsafe_allow_html=True,
	)


def available_performance_paths() -> list[Path]:
	return sorted(
		(PROJECT_ROOT / "outputs").glob("performance_*.json"),
		key=lambda path: path.stat().st_mtime,
		reverse=True,
	)


def performance_path_for(season: int, week: int | None) -> Path | None:
	paths = available_performance_paths()
	if week is not None:
		exact = PROJECT_ROOT / "outputs" / f"performance_{season}_week_{week}.json"
		if exact.exists():
			return exact
	season_path = PROJECT_ROOT / "outputs" / f"performance_{season}.json"
	return season_path if season_path.exists() else (paths[0] if paths else None)


def render_performance_page(path: Path, selected_week: int | None) -> None:
	artifact = json.loads(path.read_text())
	season = artifact["season"]
	st.markdown('<div class="eyebrow">Model evaluation / weekly performance</div>', unsafe_allow_html=True)
	st.title(f"{season} Performance")
	st.markdown('<div class="lede">Compare projections with actual fantasy points, track error by week, and inspect the model\'s biggest misses.</div>', unsafe_allow_html=True)

	if selected_week is None:
		metrics = artifact
	else:
		metrics = artifact.get("weekly_details", {}).get(str(selected_week), {})

	metric_cols = st.columns(4)
	for column, label, key in zip(metric_cols, ["MSE", "RMSE", "MAE", "Predictions"], ["mse", "rmse", "mae", "n_predictions"]):
		with column:
			value = metrics.get(key)
			metric_card(label, "—" if value is None else (f"{value:.2f}" if key != "n_predictions" else f"{value:,}"), "season" if selected_week is None else f"week {selected_week}")

	weekly = pd.DataFrame(artifact["weekly"])
	if selected_week is None and not weekly.empty:
		st.subheader("RMSE by week")
		rmse_chart = alt.Chart(weekly).mark_line(point=True).encode(
			x=alt.X("target_week:Q", scale=alt.Scale(domain=[1, 17]), axis=alt.Axis(values=list(range(1, 18)), title="Week")),
			y=alt.Y("rmse:Q", scale=alt.Scale(domain=[0, max(1.0, float(weekly["rmse"].max()) * 1.1)]), title="RMSE"),
			tooltip=[alt.Tooltip("target_week:Q", title="Week"), alt.Tooltip("rmse:Q", format=".2f", title="RMSE"), alt.Tooltip("n_predictions:Q", title="Predictions")],
		).properties(height=280).configure_view(stroke=None)
		st.altair_chart(rmse_chart, use_container_width=True)

	st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
	tab_all, tab_weeks = st.tabs(["All predictions", "Week by week"])
	with tab_all:
		rows = pd.DataFrame(artifact.get("rows", []))
		if selected_week is not None:
			rows = rows[rows["target_week"].eq(selected_week)]
		if rows.empty:
			st.info("No scored predictions are available for this selection yet.")
		else:
			sort_options = {
				"Largest absolute error": "absolute_error",
				"Largest signed error": "error",
				"Projected points": "y_pred",
				"Actual points": "y_true",
			}
			sort_col = st.selectbox("Sort predictions by", list(sort_options), key="performance_sort")
			ascending = st.checkbox("Ascending", value=False, key="performance_sort_ascending")
			rows = rows.sort_values(sort_options[sort_col], ascending=ascending)
			display = rows[["player_name", "target_week", "y_pred", "y_true", "error", "absolute_error"]].rename(
				columns={"player_name": "Player", "target_week": "Week", "y_pred": "Projected", "y_true": "Actual", "error": "Error", "absolute_error": "Absolute error"}
			)
			st.dataframe(display.style.format({"Projected": "{:.2f}", "Actual": "{:.2f}", "Error": "{:+.2f}", "Absolute error": "{:.2f}"}), use_container_width=True, height=560)
	with tab_weeks:
		st.dataframe(weekly.rename(columns={"target_week": "Week", "n_predictions": "Predictions", "mse": "MSE", "rmse": "RMSE", "mae": "MAE", "mean_error": "Mean error"}).style.format({"MSE": "{:.2f}", "RMSE": "{:.2f}", "MAE": "{:.2f}", "Mean error": "{:+.2f}"}), use_container_width=True)
		if not weekly.empty:
			chart = alt.Chart(weekly).mark_line(point=True).encode(
				x=alt.X("target_week:Q", scale=alt.Scale(domain=[1, 17]), title="Week"),
				y=alt.Y("rmse:Q", scale=alt.Scale(domain=[0, max(1.0, float(weekly["rmse"].max()) * 1.1)]), title="RMSE"),
			).properties(height=260).configure_view(stroke=None)
			st.altair_chart(chart, use_container_width=True)


def probability_a_over_b(player_a_row: pd.Series, player_b_row: pd.Series) -> tuple[float, bool]:
	"""Estimate P(A > B) from the players' 95% projection bounds."""
	a_projection = float(player_a_row["y_pred"])
	b_projection = float(player_b_row["y_pred"])
	if {"projection_low", "projection_high"}.issubset(player_a_row.index) and {"projection_low", "projection_high"}.issubset(player_b_row.index):
		a_sigma = max(0.0, float(player_a_row["projection_high"]) - float(player_a_row["projection_low"])) / 3.92
		b_sigma = max(0.0, float(player_b_row["projection_high"]) - float(player_b_row["projection_low"])) / 3.92
		difference_sigma = sqrt(a_sigma**2 + b_sigma**2)
		if difference_sigma > 0 and isfinite(difference_sigma):
			z_score = (a_projection - b_projection) / difference_sigma
			return 0.5 * (1.0 + erf(z_score / sqrt(2.0))), True
	projection_total = a_projection + b_projection
	return (a_projection / projection_total if projection_total > 0 else 0.5), False


st.markdown('<div class="eyebrow">Sunday Signal / fantasy football intelligence</div>', unsafe_allow_html=True)
st.title("Sunday Signal")
selected_view = st.radio("Choose a workspace", ["Predictions", "Performance"], horizontal=True)

if selected_view == "Performance":
	performance_paths = available_performance_paths()
	if not performance_paths:
		st.info("No performance artifact found. Run src/build_performance.py first.")
		st.stop()
	performance_seasons = sorted({int(match.group(1)) for path in performance_paths if (match := re.match(r"performance_(\d+)(?:_week_\d+)?\.json$", path.name))}, reverse=True)
	performance_season_col, performance_scope_col = st.columns([1, 1])
	with performance_season_col:
		selected_performance_season = st.selectbox("Performance season", performance_seasons)
	season_path = performance_path_for(int(selected_performance_season), None)
	season_artifact = json.loads(season_path.read_text()) if season_path else {"weeks_scored": []}
	with performance_scope_col:
		performance_scope = st.selectbox("Performance scope", ["Season"] + [f"Week {week}" for week in season_artifact.get("weeks_scored", [])])
	selected_performance_week = None if performance_scope == "Season" else int(performance_scope.split()[-1])
	selected_performance_path = performance_path_for(int(selected_performance_season), selected_performance_week)
	if selected_performance_path is None:
		st.info("No performance artifact exists for that season/week yet. Run the performance builder first.")
		st.stop()
	if selected_performance_week is not None and not selected_performance_path.name.endswith(f"_week_{selected_performance_week}.json"):
		st.warning("The weekly artifact has not been generated yet; showing the season artifact instead.")
	render_performance_page(selected_performance_path, selected_performance_week)
	st.stop()

prediction_paths = available_prediction_paths()
try:
	initial_data, _ = load_predictions(prediction_paths[0])
except (FileNotFoundError, ValueError) as error:
	st.error(str(error))
	st.stop()

seasons = sorted(initial_data["target_season"].dropna().unique(), reverse=True)
control_season, control_week = st.columns([1, 1])
with control_season:
	selected_season = st.selectbox("Season", seasons, index=0)
available_weeks = sorted(initial_data.loc[initial_data["target_season"] == selected_season, "target_week"].unique())
with control_week:
	selected_week = st.selectbox("Week", available_weeks, index=0)
st.caption("Forecasts load automatically from the saved season/week file.")

selected_path = forecast_path_for(int(selected_season), int(selected_week))
try:
	data, source_name = load_predictions(selected_path)
except (FileNotFoundError, ValueError) as error:
	st.error(str(error))
	st.stop()

season_data = data[data["target_season"] == selected_season].copy()
prior_weeks = season_data[season_data["target_week"] < selected_week].copy()
week_data = season_data[season_data["target_week"] == selected_week].copy().sort_values("y_pred", ascending=False)
week_data["projection_rounded"] = week_data["y_pred"].round(2)
current_summary = (
	prior_weeks.groupby("player_name", as_index=False)
	.agg(
		current_average=("y_pred", "mean"),
		current_games=("y_pred", "count"),
		actual_current_average=("y_true", "mean"),
	)
)
season_summary = (
	season_data.groupby("player_name", as_index=False)
	.agg(
		season_average=("y_pred", "mean"),
		season_total=("y_pred", "sum"),
		actual_season_average=("y_true", "mean"),
	)
)
rest_of_season = season_data[season_data["target_week"] >= selected_week]
rest_summary = (
	rest_of_season.groupby("player_name", as_index=False)
	.agg(
		ros_total=("y_pred", "sum"),
		ros_games=("y_pred", "count"),
	)
)
rest_summary["ros_average"] = np.where(rest_summary["ros_games"] > 0, rest_summary["ros_total"] / rest_summary["ros_games"], 0.0)
player_summary = season_summary.merge(current_summary, on="player_name", how="left").merge(rest_summary, on="player_name", how="left")
player_summary["current_average"] = player_summary["current_average"].fillna(0.0)
player_summary["current_games"] = player_summary["current_games"].fillna(0).astype(int)
player_summary["actual_current_average"] = player_summary["actual_current_average"].replace({np.nan: None})
player_summary["ros_total"] = player_summary["ros_total"].fillna(0.0)
player_summary["ros_games"] = player_summary["ros_games"].fillna(0).astype(int)
player_summary["ros_average"] = player_summary["ros_average"].fillna(0.0)
week_data = week_data.merge(player_summary[["player_name", "current_average", "actual_current_average", "ros_total", "ros_average"]], on="player_name", how="left")
plot_values = season_data[["y_pred"]].copy()
if {"projection_low", "projection_high"}.issubset(season_data.columns):
	plot_values = season_data[["y_pred", "projection_low", "projection_high"]]
plot_y_max = max(20.0, float(plot_values.max(numeric_only=True).max()) * 1.1)

st.markdown(
	f'<div class="lede">Compare two players for week {selected_week} and get a clean start/sit recommendation from the current projections.</div>',
	unsafe_allow_html=True,
)
st.caption(f"Projection scope: this week = week {selected_week}; ROS = weeks {selected_week}-17, inclusive.")

col_a, col_b = st.columns(2)
with col_a:
	player_a = st.selectbox("Player A", week_data["player_name"].unique(), index=0)
with col_b:
	player_b = st.selectbox("Player B", week_data["player_name"].unique(), index=1 if len(week_data["player_name"].unique()) > 1 else 0)

player_a_row = week_data[week_data["player_name"] == player_a].iloc[0]
player_b_row = week_data[week_data["player_name"] == player_b].iloc[0]

a_proj = float(player_a_row["y_pred"])
b_proj = float(player_b_row["y_pred"])
probability_a, probability_uses_bounds = probability_a_over_b(player_a_row, player_b_row)

if a_proj == b_proj:
	recommendation = "Even projection — lean on matchups and usage"
	winner = "Push"
else:
	recommendation = f"Start {player_a if a_proj > b_proj else player_b} over {player_b if a_proj > b_proj else player_a}"
	winner = player_a if a_proj > b_proj else player_b

comparison_cols = st.columns(4)
with comparison_cols[0]:
	metric_card(player_a, f"{a_proj:.2f}", "projected this week")
with comparison_cols[1]:
	metric_card(player_b, f"{b_proj:.2f}", "projected this week")
with comparison_cols[2]:
	metric_card("Edge", f"{abs(a_proj - b_proj):.2f}", "projected points")
with comparison_cols[3]:
	metric_card("P(A > B)", f"{probability_a:.0%}", "from 95% bounds" if probability_uses_bounds else "projection share")

if probability_uses_bounds:
	st.caption(f"{recommendation}. Confidence estimates the chance that {player_a} outscores {player_b}, using each projection's 95% low/high interval.")
else:
	st.caption(f"{recommendation}. Confidence uses projection share because this file has no usable 95% bounds.")

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
left, right = st.columns([1.3, 1], gap="large")
with left:
	st.subheader("Week ranking")
	board_columns = ["player_name", "projection_rounded", "current_average", "actual_current_average", "ros_total", "ros_average"]
	if {"projection_low", "projection_high"}.issubset(week_data.columns):
		board_columns[2:2] = ["projection_low", "projection_high"]
	board = week_data[board_columns].rename(
		columns={
			"player_name": "Player",
			"projection_rounded": "This week",
			"projection_low": "95% low",
			"projection_high": "95% high",
			"current_average": "Current avg",
			"actual_current_average": "Actual avg",
			"ros_total": "ROS total",
			"ros_average": "ROS / game",
		}
	).reset_index(drop=True)
	board.index = board.index + 1
	st.dataframe(
		board,
		column_config={
			"Player": st.column_config.TextColumn("Player", help="Player name."),
			"This week": st.column_config.NumberColumn("This week", format="%.2f", help="Projected fantasy points for the selected week."),
			"95% low": st.column_config.NumberColumn("95% low", format="%.2f", help="Lower bound of the 95% projection interval for this week."),
			"95% high": st.column_config.NumberColumn("95% high", format="%.2f", help="Upper bound of the 95% projection interval for this week."),
			"Current avg": st.column_config.NumberColumn("Current avg", format="%.2f", help="Average projected points across the completed weeks of the current season so far."),
			"Actual avg": st.column_config.NumberColumn("Actual avg", format="%.2f", help="Average actual fantasy points across completed weeks in the current season so far. Blank until actuals exist."),
			"ROS total": st.column_config.NumberColumn("ROS total", format="%.2f", help=f"Projected total points across weeks {selected_week} through 17, inclusive."),
			"ROS / game": st.column_config.NumberColumn("ROS / game", format="%.2f", help=f"ROS total divided by the number of remaining games, so this is the average projected points per game from week {selected_week} onward."),
		},
		use_container_width=True,
		height=500,
	)

with right:
	st.subheader("Player snapshots")
	for player_name in [player_a, player_b]:
		player_data = season_data[season_data["player_name"] == player_name].sort_values("target_week")
		player_week = player_data[player_data["target_week"] == selected_week]
		if player_week.empty:
			st.markdown(f"### {player_name}")
			st.caption("No projected row for this week.")
			continue
		latest = player_week.iloc[0]
		st.markdown(f"### {player_name}")
		st.metric("Projected", f"{float(latest['y_pred']):.2f}")
		profile = player_summary[player_summary["player_name"] == player_name].iloc[0]
		current_average = profile["current_average"]
		current_average_text = "—" if pd.isna(current_average) or (current_average == 0 and profile["current_games"] == 0) else f"{current_average:.2f}"
		actual_average = profile["actual_current_average"]
		actual_average_text = "—" if pd.isna(actual_average) else f"{actual_average:.2f}"
		st.caption(
			f"Current avg: {current_average_text} | "
			f"Actual avg: {actual_average_text} | "
			f"ROS (weeks {selected_week}-17): {profile['ros_total']:.2f} total / {profile['ros_games']} games = {profile['ros_average']:.2f} per game"
		)
		if "y_true" in latest and pd.notna(latest["y_true"]):
			st.caption(f"Actual: {float(latest['y_true']):.2f} | Error: {float(latest['y_pred']) - float(latest['y_true']):+.2f}")
		else:
			st.caption("No actual score yet for this week.")
		chart_data_source = player_data[player_data["target_week"].between(1, 17)].copy()
		chart_columns = ["y_pred"]
		chart_names = {"y_pred": "Projection"}
		if {"projection_low", "projection_high"}.issubset(chart_data_source.columns):
			chart_columns = ["projection_low", "y_pred", "projection_high"]
			chart_names = {"projection_low": "95% low", "y_pred": "Projection", "projection_high": "95% high"}
		chart_data = chart_data_source.set_index("target_week")[chart_columns].rename(columns=chart_names).reset_index()
		if "y_true" in chart_data_source.columns:
			actual = pd.to_numeric(chart_data_source["y_true"], errors="coerce")
			chart_data["Actual"] = actual.to_numpy()
		long_data = chart_data.melt("target_week", var_name="series", value_name="value")
		actual_only = chart_data[["target_week", "Actual"]].dropna().rename(columns={"Actual": "value"})
		actual_only["series"] = "Actual"
		actual_point_chart = alt.Chart(actual_only).mark_circle(size=80, color="#4da3ff").encode(
			x=alt.X("target_week:Q", scale=alt.Scale(domain=[1, 17]), axis=alt.Axis(values=list(range(1, 18)), title="Week")),
			y=alt.Y("value:Q", scale=alt.Scale(domain=[0, plot_y_max]), axis=alt.Axis(title="PPR points")),
			tooltip=[alt.Tooltip("target_week:Q", title="Week"), alt.Tooltip("value:Q", title="Actual", format=".2f")],
		)
		long_chart = alt.Chart(long_data[long_data["series"] != "Actual"]).encode(
			x=alt.X("target_week:Q", scale=alt.Scale(domain=[1, 17]), axis=alt.Axis(values=list(range(1, 18)), title="Week")),
			y=alt.Y("value:Q", scale=alt.Scale(domain=[0, plot_y_max]), axis=alt.Axis(title="PPR points")),
			color=alt.Color("series:N", scale=alt.Scale(domain=["95% low", "Projection", "95% high"], range=["#9eb7a5", "#f26b38", "#9eb7a5"]), legend=alt.Legend(title=None)),
		)
		chart = (long_chart.mark_line(point=True) + actual_point_chart).properties(height=190).configure_view(stroke=None)
		st.altair_chart(chart, use_container_width=True)
		st.caption("Full regular season, weeks 1-17. The chart uses fixed axes and is intentionally non-interactive.")
		if {"projection_low", "projection_high"}.issubset(chart_data_source.columns):
			st.caption("95% bounds are shown around the projection; actual scores appear only for played weeks.")
		else:
			st.caption("95% bounds are unavailable for this forecast file.")
		st.markdown("---")

st.markdown(f'<div class="source-note">SOURCE: {source_name} · TARGET SEASON: {selected_season} · ACTIVE ROSTER FORECAST</div>', unsafe_allow_html=True)