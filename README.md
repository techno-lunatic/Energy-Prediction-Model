# Hourly Energy Consumption Forecasting

A reproducible, one-hour-ahead electricity demand forecasting project using
historical hourly consumption data and an XGBoost regression model.

## Dataset

The project uses `DOM_hourly.csv` from the Kaggle **Hourly Energy Consumption**
dataset. The file contains the `Datetime` timestamp and `DOM_MW` demand columns.
The project loader validates these columns, removes invalid rows and duplicate
timestamps, and sorts observations chronologically.

The supplied ZIP includes the CSV in `data/`. The raw CSV is excluded from Git
through `.gitignore`; retain it locally when running the project.

## Method

1. Parse and clean the timestamp and demand columns.
2. Create calendar predictors: hour, day of week, quarter, month, year,
   day of year, and weekend indicator.
3. Create lag predictors at 1, 24, and 168 hours.
4. Create 24-hour and 168-hour rolling means using only prior observations.
5. Split chronologically into 70% training, 15% validation, and 15% test data.
6. Train an XGBoost regressor. Validation data is used for early stopping.
   The final test segment is held out until model selection is complete.
7. Compare the model with a persistence baseline (`lag_1`, i.e. the previous
   hour's demand) using MAE, RMSE, and R-squared.

The task is a one-step-ahead forecast. The model is not a multi-day forecasting
system and the results should not be interpreted as a validated operational
grid forecast.

## Project structure

```text
Energy-Prediction-Model/
├── data/
│   ├── DOM_hourly.csv
│   └── README.md
├── models/                 # Saved model and metadata after training
├── notebooks/
│   └── README.md
├── results/
│   └── plots/              # Evaluation plots after training
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Setup

Python 3.10 or newer is recommended.

```bash
# Create and activate a virtual environment (Windows Git Bash)
python -m venv energy_venv
source energy_venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

## Train and evaluate

Run from the project root:

```bash
python -m src.train
```

To use a different CSV path:

```bash
python -m src.train --data path/to/DOM_hourly.csv
```

Training saves:
- `models/energy_xgboost.joblib`
- `models/metadata.json`
- `results/metrics.csv`
- `results/test_predictions.csv`
- `results/feature_importance.csv`
- `results/plots/feature_importance.png`
- `results/plots/actual_vs_predicted.png`
- `results/plots/residuals.png`

The printed metrics compare XGBoost with the lag-1 baseline. MAE and RMSE are
in megawatts; lower is better. R-squared measures variance explained.

## Predict the next hour

After training has produced the saved model:

```bash
python -m src.predict
```

The prediction command loads the saved model and the historical CSV. It does
not retrain the model. At least 168 hourly observations are required to create
the lag and rolling-window features.

## Evaluate saved predictions

```bash
python -m src.evaluate
```

## Reproducibility notes

- The split is chronological, not randomly shuffled.
- Lag and rolling predictors use historical values only; rolling means are
  shifted to avoid including the target observation.
- The validation segment is used for early stopping. The test segment is used
  only for the final reported comparison.
- `random_state=42` is fixed for reproducibility.
- The model and plots are generated locally and are excluded from Git by
  `.gitignore`. Run the training command to regenerate them.

## Dataset snapshot

- Valid rows: 116,189
- Date range: 2005-05-01 01:00:00 to 2018-08-03 00:00:00
- Target: `DOM_MW` (megawatts)

## Training run generated with the supplied CSV

The ZIP includes a trained model and evaluation outputs generated from the uploaded dataset. The metrics below are from the chronological held-out test segment in this run; rerun training to reproduce them in your environment.

| model          |     MAE |    RMSE |       R2 |
|:---------------|--------:|--------:|---------:|
| XGBoost        | 133.329 | 191.775 | 0.993882 |
| Lag-1 baseline | 411.266 | 517.338 | 0.955481 |
