from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import (
    Add,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "artifacts" / "data"
MODEL_DIR = ROOT / "artifacts" / "models"
METRICS_DIR = ROOT / "artifacts" / "metrics"
FIGURE_DIR = ROOT / "artifacts" / "figures"
for directory in (MODEL_DIR, METRICS_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def transformer_block(inputs, head_size=64, num_heads=2, ff_dim=128, dropout=0.1):
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(
        inputs, inputs
    )
    attention = Dropout(dropout)(attention)
    attention = Add()([attention, inputs])
    attention = LayerNormalization(epsilon=1e-6)(attention)

    ff = Dense(ff_dim, activation="relu")(attention)
    ff = Dropout(dropout)(ff)
    ff = Dense(inputs.shape[-1])(ff)
    ff = Add()([ff, attention])
    return LayerNormalization(epsilon=1e-6)(ff)


def main():
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = transformer_block(input_layer)
    x = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    output = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1)
    y_pred = model.predict(X_test).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    nrmse_mean = rmse / np.mean(y_test)
    nrmse_range = rmse / (np.max(y_test) - np.min(y_test))

    print(f"Transformer MAE: {mae:.4f}")
    print(f"Transformer RMSE: {rmse:.4f}")
    print(f"NRMSE (mean-normalized): {nrmse_mean:.4f}")
    print(f"NRMSE (range-normalized): {nrmse_range:.4f}")

    model.save(MODEL_DIR / "transformer_baseline_model.h5")
    np.save(DATA_DIR / "y_pred_transformer.npy", y_pred)
    np.save(DATA_DIR / "y_test_transformer.npy", y_test)
    pd.DataFrame(history.history).to_csv(
        METRICS_DIR / "training_history_transformer.csv", index=False
    )
    with open(METRICS_DIR / "model_metrics_transformer.txt", "w") as f:
        f.write(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\n")

    plt.figure(figsize=(12, 5))
    plt.plot(y_test[:200], label="True")
    plt.plot(y_pred[:200], label="Predicted")
    plt.title("Transformer Baseline - Wind Power Forecasting - First 200 Samples")
    plt.xlabel("Time Index")
    plt.ylabel("Power Output")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "transformer_predictions_first_200_samples.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
