# Agentic Football Predictor

An advanced, professional-grade football (soccer) predictive analytics engine. This system leverages massive amounts of historical match data to predict the expected goals, shots, and corners for both the Home and Away teams. It then utilizes the Poisson distribution to derive betting market probabilities (Match Odds, Over/Under, Both Teams to Score).

## Architecture Overview

The system is built as an end-to-end Machine Learning pipeline.

1. **Data Ingestion** (`src/data_ingestion.py`): Automatically fetches historical football data from `football-data.co.uk` across 15 European leagues spanning back to the 2019/2020 season (over 28,000 matches). Caches raw CSVs locally.
2. **Feature Engineering** (`src/feature_engineering.py`): Transforms raw data into a strictly chronological, zero-data-leakage feature set. 
3. **Model Training** (`src/models.py`): Trains multiple algorithms (Baseline Averages, Random Forest, XGBoost, and a PyTorch Feed-Forward Neural Network). 
4. **Evaluation** (`src/evaluation.py`): Compares models using Mean Absolute Error (MAE).
5. **Dashboard** (`dashboard.py`): A Streamlit web application providing an interactive UI to visualize matchup predictions and betting odds.

## Core Features Engineered

To accurately predict match outcomes, the system computes 31 dynamic features for every match:

- **Advanced Elo Rating Engine (`src/elo.py`)**: A dynamic rating system initialized at 1500. It updates sequentially after every match. Crucially, it incorporates a **Margin of Victory (MoV)** multiplier based on World Football Elo Ratings standard, ensuring blowout wins shift ratings more than narrow victories. It also factors in Home Advantage.
- **Rolling Form Statistics**: Rolling 5-match averages for Goals For/Against, Shots on Target, and Corners.
- **Venue-Specific Form**: Rolling 5-match averages specifically for the Home team when playing at home, and the Away team when playing away.
- **Points Per Game (PPG)**: Rolling PPG to measure current league form.
- **Rest Days**: Computes the exact number of days since a team's previous match to quantify fatigue.
- **Head-to-Head History**: A rolling 10-match historical record of outcomes and goals between the two specific teams playing.
- **Geospatial Travel Distance (`src/geography.py`)**: Uses the `geopy` library to geocode European stadium locations and calculates the exact Haversine travel distance (in kilometers) the Away team had to travel for the fixture.

## Machine Learning Models

The pipeline trains 4 models to predict 6 target variables: `Home Goals`, `Away Goals`, `Home Shots on Target`, `Away Shots on Target`, `Home Corners`, `Away Corners`.

1. **Baseline**: Simply predicts the rolling average. (Overall MAE: ~1.71)
2. **Random Forest**: 300 trees. (Overall MAE: ~1.57)
3. **XGBoost**: 400 rounds. (Overall MAE: ~1.58)
4. **PyTorch Neural Network (Champion)**: A Feed-Forward dense neural network (25k params) utilizing ReLU activations and Dropout for regularization. It consistently achieves the best performance (Overall MAE: ~1.56).

### Experimental Models (`train_lstm.py`)
We also built a **Dual-Sequence LSTM** model as a proof of concept. It treats matches as a time-series, passing the Home team's last 5 matches and Away team's last 5 matches through parallel LSTMs. However, the Feed-Forward Neural Network leveraging engineered rolling features continues to outperform the naive LSTM.

## Betting Market Probabilities (`src/probabilities.py`)

The ML models output expected counts (e.g., Home Team: 2.1 Goals, Away Team: 0.8 Goals). The system then assumes goalscoring follows a **Poisson distribution** to expand these expectations into a probability matrix, calculating:
- **Match Odds (1X2)**: Probability of Home Win, Draw, or Away Win.
- **Over/Under 2.5 Goals**: Probability of the total goals being 3+ or 0-2.
- **Both Teams to Score (BTTS)**: Probability that both teams score at least 1 goal.

## How to Run

### Command Line Interface
Run the full data ingestion, feature engineering, and model training pipeline:
```bash
python main.py
```

### Streamlit Dashboard
Launch the interactive web application to view predictions:
```bash
streamlit run dashboard.py
```

### Jupyter Notebooks
Exploratory data analysis and Elo algorithm prototyping can be found in `notebooks/`.

## Dependencies
- `pandas`, `numpy`, `scikit-learn`
- `torch` (PyTorch)
- `xgboost`
- `geopy` (for stadium travel distances)
- `streamlit`, `plotly` (for the dashboard UI)
- `soccerdata` (available for future FBref xG integration)
