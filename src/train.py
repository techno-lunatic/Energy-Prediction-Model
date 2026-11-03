import argparse
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from .data_loader import load_data
from .feature_engineering import FEATURE_COLUMNS
from .preprocessing import chronological_split
from .evaluate import calculate_metrics

def main():
    parser = argparse.ArgumentParser(description="Train the hourly energy forecasting model.")
    parser.add_argument("--data", default="data/DOM_hourly.csv", help="Path to DOM_hourly.csv")
    parser.add_argument("--output-dir", default=".", help="Project output directory")
    args = parser.parse_args()

    output = Path(args.output_dir)
    model_dir, results_dir, plots_dir = output/"models", output/"results", output/"results"/"plots"
    for folder in (model_dir, results_dir, plots_dir):
        folder.mkdir(parents=True, exist_ok=True)

    raw = load_data(args.data)
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(raw)

    model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=50,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # The test segment is used only after fitting and model selection are complete.
    predictions = model.predict(X_test)
    baseline = X_test["lag_1"].to_numpy()
    model_metrics = calculate_metrics(y_test, predictions)
    baseline_metrics = calculate_metrics(y_test, baseline)

    comparison = pd.DataFrame([
        {"model": "XGBoost", **model_metrics},
        {"model": "Lag-1 baseline", **baseline_metrics},
    ])
    comparison.to_csv(results_dir/"metrics.csv", index=False)

    pred_df = pd.DataFrame({
        "Datetime": X_test.index,
        "actual": y_test.to_numpy(),
        "prediction": predictions,
        "lag_1_baseline": baseline,
    })
    pred_df.to_csv(results_dir/"test_predictions.csv", index=False)

    joblib.dump(model, model_dir/"energy_xgboost.joblib")
    metadata = {
        "model_type": "XGBRegressor",
        "target": "DOM_MW",
        "forecast_horizon": "1 hour ahead",
        "features": FEATURE_COLUMNS,
        "best_iteration": int(getattr(model, "best_iteration", model.n_estimators - 1)),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "data_start": str(raw.index.min()),
        "data_end": str(raw.index.max()),
    }
    (model_dir/"metadata.json").write_text(json.dumps(metadata, indent=2))

    # Feature importance
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(results_dir/"feature_importance.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.barh(importance["feature"][::-1], importance["importance"][::-1])
    plt.xlabel("Importance")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(plots_dir/"feature_importance.png", dpi=160)
    plt.close()

    # Show a representative chronological slice of the test predictions.
    sample = pred_df.head(min(500, len(pred_df)))
    plt.figure(figsize=(12, 5))
    plt.plot(pd.to_datetime(sample["Datetime"]), sample["actual"], label="Actual")
    plt.plot(pd.to_datetime(sample["Datetime"]), sample["prediction"], label="XGBoost")
    plt.xlabel("Datetime")
    plt.ylabel("Demand (MW)")
    plt.title("Actual vs Predicted Demand (first 500 test observations)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir/"actual_vs_predicted.png", dpi=160)
    plt.close()

    residuals = y_test.to_numpy() - predictions
    plt.figure(figsize=(8, 5))
    plt.scatter(predictions, residuals, s=8, alpha=0.35)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted demand (MW)")
    plt.ylabel("Residual (MW)")
    plt.title("Residual Analysis")
    plt.tight_layout()
    plt.savefig(plots_dir/"residuals.png", dpi=160)
    plt.close()

    print("Training complete.")
    print("Best iteration:", metadata["best_iteration"])
    print(comparison.to_string(index=False))
    print("Saved model:", model_dir/"energy_xgboost.joblib")
    print("Saved metrics and plots in:", results_dir)

if __name__ == "__main__":
    main()
