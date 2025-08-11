# Project Overview

This project predicts the **number of fantasy football points** (Full PPR scoring) a player will score in a given week. Using historical data from the [`nflverse`](https://nflverse.nflverse.com/) dataset combined with custom-engineered features, multiple machine learning models were tested to find the most accurate predictive approach.

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



