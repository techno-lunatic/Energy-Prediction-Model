import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(mean_squared_error(actual, predicted) ** 0.5),
        "R2": float(r2_score(actual, predicted)),
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate saved test predictions.")
    parser.add_argument("--predictions", default="results/test_predictions.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.predictions)
    print(calculate_metrics(df["actual"], df["prediction"]))

if __name__ == "__main__":
    main()
