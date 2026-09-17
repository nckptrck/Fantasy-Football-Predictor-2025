# Project Overview

The goal of this project is to predict the **number of fantasy football points** (Full PPR scoring) a player will score in a given week. Using historical data from the [`nflverse`](https://nflverse.nflverse.com/) dataset combined with custom-engineered features, multiple machine learning models were tested to find the most accurate predictive model.

## Workflow
1. **Data Collection**  
   - Pulled historical NFL data (2019-2024) from `nflverse` sources.  
   - Integrated multiple datasets: play-by-play, weekly stats, schedules, and rosters.

2. **Feature Engineering**  
   - Created predictive features from raw stats, player performance history, and opponent matchups.  
   - Included rolling averages, team-level statistics, vegas lines and implied point totals.
   - Created a final training set containing 23,414 rows and 128 columns
     - Each row contains a player for a particular week and season (2020-2024) and their recent individual, team, and opponent statististics
     - Target column is their Full-PPR fantasy football points for that week

3. **Model Selection**  
   - Experimented with multiple models including:  
     - Linear Regression (OLS, Ridge, Lasso, Elastic Net)  
     - Partial Least Squares  
     - Random Forest Regressor
     - Gradient Boosting Regressor (planned)
     - XGBoost (planned)  
   - Used `scikit-learn` pipelines for preprocessing and model fitting.
   - Hyperparameter tuning with `GridSearchCV`.
   - Cross-validation to avoid overfitting.  
   - Evaluation using **Root Mean Squared Error (RMSE)** as the main performance metric.

4. **Model Evaluation (planned)**
   - Train model on 2020-2023 data, evaluate across full 2024 season
   - Evaluation of RMSE
     - Overall
     - Breakdown by Position (QB, WR, RB, TE)
     - Breakdown by week (effectiveness as season progresses)
   - Binary Evaluation
     - Pick random players of similar projections/fantasy ranking (imagine trying to figure out who you should start)
     - See how often the model chooses the correct player to start
     - This will show how effective the model is in practice
     - Evaluate on accuracy, F1-score
   - Real Life Evaluation
     - How the model performs for my team during the 2025 NFL season
       
5. **Model Deployment (planned)**
  - Create a simple frontend to input multiple players and use model results to give start/sit suggestion
  - Integrate an LLM to generate an explanation of the suggestion

### How to Replicate
- Clone this repo
- set up a virtual environment
- run `pip install -r requirements.txt` in terminal
- run `get-data.py` to fetch `nflverse` data
- run `create-training-data.py` to generate full training data
- run `model-selection.ipynb` to see cross-validation model comparison
- ...

### Current Pipeline

### Forecast Dashboard

The MVP frontend reads `outputs/temporal_validation_combined.csv` and provides a weekly projection board and player trend view:

```bash
streamlit run src/app.py
```

The dashboard automatically falls back to `outputs/temporal_validation.csv` if the combined artifact is unavailable. Predictions are currently file-backed; database storage and MLflow model versioning are future extensions.

The complete refresh, training, evaluation, and forecasting entry point is
`src/run_pipeline.py`. Evaluation uses independent temporal folds:

```bash
python src/run_pipeline.py \
  --mode evaluate \
  --end-season 2025 \
  --validation-seasons 2024 2025 \
  --epochs 50
```

For the rolling current-season process, keep 2024 as the tuning fold, test
the full 2025 season, and add only the latest completed 2026 week to the test
set:

```bash
.venv/bin/python src/run_pipeline.py \
  --mode evaluate \
  --tune-season 2024 \
  --test-season 2025 2026 \
  --validation-week 1 \
  --tune \
  --trials 10 \
  --epochs 25 \
  --output outputs/validation_2026_week_1.csv
```

This trains on 2019-2023, tunes on 2024, evaluates all of 2025, and evaluates
only the completed 2026 week 1. When week 2 is complete, change
`--validation-week` to `2`; the historical tuning and 2025 test remain intact.

For a weekly current-season forecast, fetch through the current season and
train only on seasons before the requested forecast season:

```bash
python src/run_pipeline.py \
  --mode forecast \
  --end-season 2026 \
  --forecast-season 2026 \
  --epochs 100
```

Forecast start weeks are aligned automatically to the first week without
current-season weekly data. For example, before 2026 weekly stats are
published, this writes `outputs/production_predictions_2026_week_1.csv`:

```bash
.venv/bin/python src/run_pipeline.py \
  --mode forecast \
  --end-season 2026 \
  --forecast-season 2026 \
  --config outputs/temporal_validation_best_config.yaml \
  --epochs 25
```

Use `--start-week` only when it matches the first unobserved week. The output
name includes the season and start week, and the CSV includes
`forecast_start_week` plus prediction bounds.

### Weekly Performance Artifacts

After refreshing `data/weekly.csv` and generating a forecast, build the
file-backed performance artifact used by the dashboard:

```bash
.venv/bin/python src/build_performance.py \
  --predictions outputs/production_predictions_2026_week_1.csv \
  --season 2026 \
  --week 1 \
  --output outputs/performance_2026_week_1.json
```

For the season-long artifact, omit `--week` and use
`outputs/performance_2026.json`. The week-specific artifact contains only that
week's scored rows; the season artifact contains the full scored season.

The artifacts contain MSE, RMSE, MAE, errors, and all scored predictions.
Start the UI with:

```bash
.venv/bin/streamlit run src/app.py
```

Choose `Performance` at the top of the app, then select the season and either
the full season or a specific week. The app automatically loads the matching
`performance_<season>.json` or `performance_<season>_week_<week>.json` file.
This is currently CSV/JSON-backed; database storage can replace these
artifacts later.

Forecast mode writes the prediction CSV, Keras model, and fitted scaler under
`outputs/`. The model is trained on labeled history only and includes active
roster players even when they have not yet recorded a current-season weekly
stat line.




