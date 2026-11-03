from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"Datetime", "DOM_MW"}

def load_data(path):
    """Load, validate, clean and chronologically sort hourly demand data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place DOM_hourly.csv in the data/ folder "
            "or pass its path with --data."
        )

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df[["Datetime", "DOM_MW"]].copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["DOM_MW"] = pd.to_numeric(df["DOM_MW"], errors="coerce")
    df = df.dropna(subset=["Datetime", "DOM_MW"])
    df = df.drop_duplicates(subset=["Datetime"], keep="last")
    df = df.sort_values("Datetime").set_index("Datetime")
    df = df[~df.index.duplicated(keep="last")]
    return df
