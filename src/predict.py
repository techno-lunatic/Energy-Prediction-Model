import argparse
import json
from pathlib import Path
import joblib
from .data_loader import load_data
from .feature_engineering import create_next_hour_features

def main():
    parser = argparse.ArgumentParser(description="Predict demand for the next hour.")
    parser.add_argument("--data", default="data/DOM_hourly.csv")
    parser.add_argument("--model", default="models/energy_xgboost.joblib")
    args = parser.parse_args()

    history = load_data(args.data)
    model = joblib.load(args.model)
    features = create_next_hour_features(history)
    prediction = float(model.predict(features)[0])
    print(f"Forecast timestamp: {features.index[0]}")
    print(f"Predicted DOM_MW: {prediction:,.2f} MW")

if __name__ == "__main__":
    main()
