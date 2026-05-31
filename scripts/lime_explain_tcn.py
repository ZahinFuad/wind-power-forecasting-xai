from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
DATA_DIR = ROOT / "artifacts" / "data"
MODEL_DIR = ROOT / "artifacts" / "models"
METRICS_DIR = ROOT / "artifacts" / "metrics"
LIME_DIR = ROOT / "artifacts" / "xai" / "lime_explanations"
for directory in (METRICS_DIR, LIME_DIR):
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
    np.random.seed(SEED)

    model = load_model(MODEL_DIR / "tcn_baseline_model_15epochs.h5")
    X_test = np.load(DATA_DIR / "X_test.npy")

    time_steps = X_test.shape[1]
    num_features = X_test.shape[2]

    X_explain_3d = X_test[100:110]
    X_explain = X_explain_3d.reshape((X_explain_3d.shape[0], -1))
    X_background = X_test[:100].reshape((100, time_steps * num_features))

    feature_names = [
        f"{BASE_FEATURES[f]}_t-{time_steps - t - 1}"
        for t in range(time_steps)
        for f in range(num_features)
    ]

    def model_predict(flat_input):
        reshaped_input = flat_input.reshape((-1, time_steps, num_features))
        return model.predict(reshaped_input)

    explainer = LimeTabularExplainer(
        training_data=X_background,
        feature_names=feature_names,
        mode="regression",
        verbose=True,
        random_state=SEED,
    )

    lime_summary = []
    for i in range(10):
        exp = explainer.explain_instance(X_explain[i], model_predict, num_features=20)

        fig = exp.as_pyplot_figure()
        fig.set_dpi(300)
        fig.set_size_inches(8, 6)
        plt.title(f"LIME Local Explanation for Sample {i}", fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(LIME_DIR / f"lime_sample{i}.png", bbox_inches="tight")
        plt.close(fig)

        for feature, weight in exp.as_list():
            lime_summary.append({"sample": i, "feature": feature, "weight": weight})

    lime_df = pd.DataFrame(lime_summary)
    lime_df.to_csv(METRICS_DIR / "lime_top_features_summary.csv", index=False)
    print(f"Saved LIME plots to {LIME_DIR}")
    print("Saved LIME summary table.")


if __name__ == "__main__":
    main()
