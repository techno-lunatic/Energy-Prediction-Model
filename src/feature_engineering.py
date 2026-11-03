import pandas as pd

FEATURE_COLUMNS = [
    "hour", "dayofweek", "quarter", "month", "year", "dayofyear",
    "is_weekend", "lag_1", "lag_24", "lag_168",
    
    "rolling_mean_24", "rolling_mean_168",
]

def create_features(df):
    """
    Create one-step-ahead predictors from the observed demand history.
    Rolling features are shifted by one row so the current target is never used.
    """
    result = df.copy()
    idx = result.index

    result["hour"] = idx.hour
    result["dayofweek"] = idx.dayofweek
    result["quarter"] = idx.quarter
    result["month"] = idx.month
    result["year"] = idx.year
    result["dayofyear"] = idx.dayofyear
    result["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    result["lag_1"] = result["DOM_MW"].shift(1)
    result["lag_24"] = result["DOM_MW"].shift(24)
    result["lag_168"] = result["DOM_MW"].shift(168)
    prior_values = result["DOM_MW"].shift(1)
    result["rolling_mean_24"] = prior_values.rolling(window=24).mean()
    result["rolling_mean_168"] = prior_values.rolling(window=168).mean()

    return result.dropna(subset=FEATURE_COLUMNS + ["DOM_MW"])

def create_next_hour_features(history):
    """Build a single feature row for the hour immediately after the history."""
    if len(history) < 168:
        raise ValueError("At least 168 hourly observations are required.")
    last_time = history.index[-1]
    next_time = last_time + pd.Timedelta(hours=1)
    values = history["DOM_MW"].astype(float)

    row = {
        "hour": next_time.hour,
        "dayofweek": next_time.dayofweek,
        "quarter": next_time.quarter,
        "month": next_time.month,
        "year": next_time.year,
        "dayofyear": next_time.dayofyear,
        "is_weekend": int(next_time.dayofweek >= 5),
        "lag_1": values.iloc[-1],
        "lag_24": values.iloc[-24],
        "lag_168": values.iloc[-168],
        "rolling_mean_24": values.iloc[-24:].mean(),
        "rolling_mean_168": values.iloc[-168:].mean(),
    }
    return pd.DataFrame([row], index=[next_time])[FEATURE_COLUMNS]
