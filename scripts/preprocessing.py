from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "Location1.csv"
ARTIFACT_DATA_DIR = ROOT / "artifacts" / "data"
ARTIFACT_DATA_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "temperature_2m",
    "relativehumidity_2m",
    "dewpoint_2m",
    "windspeed_10m",
    "windspeed_100m",
    "winddirection_10m",
    "winddirection_100m",
    "windgusts_10m",
]
TARGET_COL = "Power"


def create_sequences(features_array, target_array, window_size=24):
    """Convert time-series rows into sliding windows for one-hour-ahead forecasting."""
    X, y = [], []
    for i in range(len(features_array) - window_size):
        X.append(features_array[i : i + window_size])
        y.append(target_array[i + window_size])
    return np.array(X), np.array(y)


def main():
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["Time"])
    df.sort_values("Time", inplace=True)
    df.set_index("Time", inplace=True)
    df.dropna(inplace=True)

    features = df[FEATURE_COLS]
    target = df[TARGET_COL]

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    X, y = create_sequences(scaled_features, target.values, window_size=24)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    np.save(ARTIFACT_DATA_DIR / "X_train.npy", X_train)
    np.save(ARTIFACT_DATA_DIR / "y_train.npy", y_train)
    np.save(ARTIFACT_DATA_DIR / "X_test.npy", X_test)
    np.save(ARTIFACT_DATA_DIR / "y_test.npy", y_test)
    np.save(ARTIFACT_DATA_DIR / "X_full.npy", scaled_features)
    np.save(ARTIFACT_DATA_DIR / "y_full.npy", target.values)

    print("Preprocessing complete.")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_test shape:  {y_test.shape}")
    print(f"Saved processed arrays to {ARTIFACT_DATA_DIR}")


if __name__ == "__main__":
    main()
