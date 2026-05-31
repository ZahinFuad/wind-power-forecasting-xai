from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "artifacts" / "data"
MODEL_DIR = ROOT / "artifacts" / "models"
METRICS_DIR = ROOT / "artifacts" / "metrics"
XAI_DIR = ROOT / "artifacts" / "xai"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    model = load_model(MODEL_DIR / "tcn_baseline_model_15epochs.h5")
    X_test = np.load(DATA_DIR / "X_test.npy")
    X_explain_orig = X_test[100:110]
    time_steps, num_features = X_explain_orig.shape[1], X_explain_orig.shape[2]

    shap_values = np.load(XAI_DIR / "shap_values_named_10samples.npy")
    shap_feature_names = np.load(
        XAI_DIR / "shap_feature_names_named.npy", allow_pickle=True
    )
    lime_df = pd.read_csv(METRICS_DIR / "lime_top_features_summary.csv")

    faithfulness_results = []
    for i in range(10):
        original_sample = X_explain_orig[i : i + 1].copy()
        flat = original_sample.reshape(-1)

        top_idx = np.argsort(np.abs(shap_values[i]))[::-1][:3]
        flat[top_idx] = 0
        ablated_sample = flat.reshape(1, time_steps, num_features)

        original_pred = model.predict(original_sample)[0][0]
        ablated_pred = model.predict(ablated_sample)[0][0]
        diff = abs(original_pred - ablated_pred)

        faithfulness_results.append(
            {
                "sample": i,
                "original_pred": original_pred,
                "ablated_pred": ablated_pred,
                "delta": diff,
                "top_features": [shap_feature_names[idx] for idx in top_idx],
            }
        )

    pd.DataFrame(faithfulness_results).to_csv(
        METRICS_DIR / "faithfulness_test_results.csv", index=False
    )

    consistency = []
    for i in range(10):
        shap_sample = shap_values[i]
        shap_ranking = pd.Series(
            np.abs(shap_sample), index=shap_feature_names
        ).rank(ascending=False)

        lime_sample_df = lime_df[lime_df["sample"] == i].copy()
        lime_sample_df["abs_weight"] = lime_sample_df["weight"].abs()
        lime_ranking = lime_sample_df.set_index("feature")["abs_weight"].rank(
            ascending=False
        )

        common_features = shap_ranking.index.intersection(lime_ranking.index)
        spearman_corr = spearmanr(
            shap_ranking[common_features], lime_ranking[common_features]
        )[0]
        overlap_top5 = len(
            set(shap_ranking.nsmallest(5).index) & set(lime_ranking.nsmallest(5).index)
        )

        consistency.append(
            {
                "sample": i,
                "spearman_rank_corr": spearman_corr,
                "top5_overlap_count": overlap_top5,
            }
        )

    pd.DataFrame(consistency).to_csv(
        METRICS_DIR / "shap_lime_consistency_metrics.csv", index=False
    )

    X_flat = X_explain_orig.reshape((10, time_steps * num_features))
    X_bg = X_test[:100].reshape((100, time_steps * num_features))
    feature_names = shap_feature_names.tolist()

    def model_predict(flat):
        reshaped = flat.reshape((-1, time_steps, num_features))
        return model.predict(reshaped)

    lime_explainer = LimeTabularExplainer(X_bg, feature_names=feature_names, mode="regression")
    start_lime = time()
    for i in range(10):
        lime_explainer.explain_instance(X_flat[i], model_predict, num_features=10)
    lime_time = time() - start_lime

    masker = shap.maskers.Independent(X_bg)
    shap_explainer = shap.Explainer(model_predict, masker, algorithm="permutation")
    start_shap = time()
    shap_explainer(X_flat)
    shap_time = time() - start_shap

    with open(METRICS_DIR / "xai_runtime_comparison.txt", "w") as f:
        f.write(f"SHAP time (10 samples): {shap_time:.2f} seconds\n")
        f.write(f"LIME time (10 samples): {lime_time:.2f} seconds\n")

    print("All XAI evaluation results saved.")


if __name__ == "__main__":
    main()
