from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "artifacts" / "data"
MODEL_DIR = ROOT / "artifacts" / "models"
FIGURE_DIR = ROOT / "artifacts" / "figures"
METRICS_DIR = ROOT / "artifacts" / "metrics"
XAI_DIR = ROOT / "artifacts" / "xai"
for directory in (FIGURE_DIR, METRICS_DIR, XAI_DIR):
    directory.mkdir(parents=True, exist_ok=True)

BASE_FEATURES = [
    "temperature_2m",
    "relativehumidity_2m",
    "dewpoint_2m",
    "windspeed_10m",
    "windspeed_100m",
    "winddirection_10m",
    "winddirection_100m",
    "windgusts_10m",
]


def main():
    model = load_model(MODEL_DIR / "tcn_baseline_model_15epochs.h5")
    X_test = np.load(DATA_DIR / "X_test.npy")

    time_steps = X_test.shape[1]
    num_features = X_test.shape[2]

    X_background = X_test[:100].reshape(100, time_steps * num_features)
    X_explain = X_test[100:110].reshape(10, time_steps * num_features)

    feature_names = [
        f"{BASE_FEATURES[f]}_t-{time_steps - t - 1}"
        for t in range(time_steps)
        for f in range(num_features)
    ]

    def model_predict(x_flat):
        return model(x_flat.reshape((-1, time_steps, num_features)), training=False).numpy()

    print("Initializing SHAP...")
    masker = shap.maskers.Independent(X_background)
    explainer = shap.Explainer(
        model_predict, masker, feature_names=feature_names, algorithm="permutation"
    )

    print("Running SHAP...")
    start_time = time.time()
    shap_values = explainer(X_explain)
    runtime = time.time() - start_time
    print(f"SHAP completed in {runtime:.2f} seconds.")

    np.save(XAI_DIR / "shap_values_named_10samples.npy", shap_values.values)
    np.save(XAI_DIR / "shap_feature_names_named.npy", np.array(feature_names))

    plt.figure(figsize=(8, 6), dpi=300)
    shap.plots.bar(shap_values, max_display=18, show=False)
    plt.title("SHAP Global Feature Importance", fontsize=12)
    plt.xlabel("Mean |SHAP value|", fontsize=10)
    plt.ylabel("Features", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_global_named_bar_tcn.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4), dpi=300)
    shap.plots.waterfall(shap_values[0], max_display=20, show=False)
    plt.title("SHAP Waterfall Plot for Sample 0", fontsize=10)
    plt.subplots_adjust(left=0.5, right=0.85, top=0.85, bottom=0.15)
    plt.savefig(
        FIGURE_DIR / "shap_waterfall_named_sample0_tcn.png",
        bbox_inches="tight",
        pad_inches=0.5,
    )
    plt.close()

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
    ).sort_values(by="mean_abs_shap", ascending=False)
    shap_df.head(30).to_csv(METRICS_DIR / "shap_top30_named_features.csv", index=False)

    print("Saved SHAP artifacts, plots, and summary table.")


if __name__ == "__main__":
    main()
