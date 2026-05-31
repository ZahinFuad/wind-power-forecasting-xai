from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
DATA_DIR = ROOT / "artifacts" / "data"
MODEL_DIR = ROOT / "artifacts" / "models"
METRICS_DIR = ROOT / "artifacts" / "metrics"
FIGURE_DIR = ROOT / "artifacts" / "figures"
for directory in (MODEL_DIR, METRICS_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def main():
    tf.keras.utils.set_random_seed(SEED)

    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    model = Sequential()
    model.add(
        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="causal",
            input_shape=(X_train.shape[1], X_train.shape[2]),
        )
    )
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1)
    y_pred = model.predict(X_test).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    nrmse_mean = rmse / np.mean(y_test)
    nrmse_range = rmse / (np.max(y_test) - np.min(y_test))

    print(f"CNN-LSTM MAE: {mae:.4f}")
    print(f"CNN-LSTM RMSE: {rmse:.4f}")
    print(f"NRMSE (mean-normalized): {nrmse_mean:.4f}")
    print(f"NRMSE (range-normalized): {nrmse_range:.4f}")

    model.save(MODEL_DIR / "cnn_lstm_baseline_model.h5")
    np.save(DATA_DIR / "y_pred_cnn_lstm.npy", y_pred)
    np.save(DATA_DIR / "y_test_cnn_lstm.npy", y_test)
    pd.DataFrame(history.history).to_csv(
        METRICS_DIR / "training_history_cnn_lstm.csv", index=False
    )
    with open(METRICS_DIR / "model_metrics_cnn_lstm.txt", "w") as f:
        f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\n")

    plt.figure(figsize=(12, 5))
    plt.plot(y_test[:200], label="True")
    plt.plot(y_pred[:200], label="Predicted")
    plt.title("CNN-LSTM Baseline - Wind Power Forecasting - First 200 Samples")
    plt.xlabel("Time Index")
    plt.ylabel("Power Output")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "cnn_lstm_predictions_first_200_samples.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
