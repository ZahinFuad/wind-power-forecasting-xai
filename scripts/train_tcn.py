from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "artifacts" / "data"
MODEL_DIR = ROOT / "artifacts" / "models"
METRICS_DIR = ROOT / "artifacts" / "metrics"
FIGURE_DIR = ROOT / "artifacts" / "figures"
for directory in (DATA_DIR, MODEL_DIR, METRICS_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate):
    conv1 = Conv1D(
        filters, kernel_size, padding="causal", dilation_rate=dilation_rate
    )(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Dropout(dropout_rate)(conv1)

    conv2 = Conv1D(
        filters, kernel_size, padding="causal", dilation_rate=dilation_rate
    )(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = Dropout(dropout_rate)(conv2)

    if x.shape[-1] != filters:
        res_x = Conv1D(filters, 1, padding="same")(x)
    else:
        res_x = x

    return Add()([res_x, conv2])


def main():
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = tcn_block(input_layer, filters=32, kernel_size=3, dilation_rate=1, dropout_rate=0.2)
    x = tcn_block(x, filters=32, kernel_size=3, dilation_rate=2, dropout_rate=0.2)
    x = tcn_block(x, filters=32, kernel_size=3, dilation_rate=4, dropout_rate=0.2)
    x = Flatten()(x)
    output = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1)
    y_pred = model.predict(X_test).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    nrmse_mean = rmse / np.mean(y_test)
    nrmse_range = rmse / (np.max(y_test) - np.min(y_test))

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"NRMSE (mean-normalized): {nrmse_mean:.4f}")
    print(f"NRMSE (range-normalized): {nrmse_range:.4f}")

    model.save(MODEL_DIR / "tcn_baseline_model_15epochs.h5")
    np.save(DATA_DIR / "y_pred_tcn_15epochs.npy", y_pred)
    np.save(DATA_DIR / "y_test_tcn_15epochs.npy", y_test)

    pd.DataFrame(history.history).to_csv(
        METRICS_DIR / "training_history_tcn_15epochs.csv", index=False
    )
    with open(METRICS_DIR / "model_metrics_tcn_15epochs.txt", "w") as f:
        f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\n")

    plt.figure(figsize=(12, 5))
    plt.plot(y_test[:200], label="True")
    plt.plot(y_pred[:200], label="Predicted")
    plt.title("TCN - Wind Power Forecasting - First 200 Samples")
    plt.xlabel("Time Index")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tcn_predictions_first_200_samples.png", dpi=300)
    plt.close()

    print("Model, predictions, metrics, and plot saved successfully.")


if __name__ == "__main__":
    main()
