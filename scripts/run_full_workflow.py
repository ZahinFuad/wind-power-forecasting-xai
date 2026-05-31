from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    "scripts/preprocessing.py",
    "scripts/train_tcn.py",
    "scripts/train_lstm_baseline.py",
    "scripts/train_gru_baseline.py",
    "scripts/train_cnn_lstm_baseline.py",
    "scripts/train_transformer_baseline.py",
    "scripts/shap_explain_tcn.py",
    "scripts/lime_explain_tcn.py",
    "scripts/evaluate_xai.py",
]


def main():
    for step in STEPS:
        print(f"\n=== Running {step} ===")
        subprocess.run([sys.executable, str(ROOT / step)], cwd=ROOT, check=True)

    print("\nFull workflow completed. Generated outputs are in artifacts/.")


if __name__ == "__main__":
    main()
