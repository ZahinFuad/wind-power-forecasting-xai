# Explainable Wind Power Forecasting with Deep Learning

This repository contains a research experiment on short-term wind power forecasting using deep learning models and explainable AI methods. The project compares several neural architectures for one-hour-ahead wind power prediction from hourly weather observations, then applies SHAP and LIME to interpret the strongest model's behavior.

This work was completed as part of my master's-level computer science research portfolio.

## Research Focus

- Forecast wind power output using meteorological time-series data.
- Compare TCN, LSTM, GRU, CNN-LSTM, and Transformer baselines.
- Evaluate model performance with MAE, RMSE, and NRMSE.
- Use SHAP and LIME to identify which recent weather features most influence model predictions.

## Associated Manuscript

This repository accompanies a research manuscript on wind power forecasting using deep learning models and explainable AI methods. The manuscript is currently awaiting publication/acceptance and is not included in this public repository.

The code and summarized experimental results are shared here as part of my academic research portfolio. A citation and manuscript link will be added after publication.

## Repository Structure

```text
.
├── data/raw/Location1.csv
├── docs/
│   ├── methodology.md
│   └── results_summary.md
├── results/metrics_summary.csv
├── scripts/
│   ├── preprocessing.py
│   ├── train_tcn.py
│   ├── train_lstm_baseline.py
│   ├── train_gru_baseline.py
│   ├── train_cnn_lstm_baseline.py
│   ├── train_transformer_baseline.py
│   ├── shap_explain_tcn.py
│   ├── lime_explain_tcn.py
│   └── evaluate_xai.py
└── requirements.txt
```

Generated arrays, models, figures, and local manuscript files are excluded from the public repo. When scripts are run, those outputs are written under `artifacts/`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce The Main Workflow

Prepare the time-series windows:

```powershell
python scripts/preprocessing.py
```

Train the main TCN model:

```powershell
python scripts/train_tcn.py
```

Train baseline models:

```powershell
python scripts/train_lstm_baseline.py
python scripts/train_gru_baseline.py
python scripts/train_cnn_lstm_baseline.py
python scripts/train_transformer_baseline.py
```

Run explainability analysis after training the TCN model:

```powershell
python scripts/shap_explain_tcn.py
python scripts/lime_explain_tcn.py
python scripts/evaluate_xai.py
```

## Summary Results

The strongest saved results were from the TCN and LSTM models. Full summarized metrics are available in `results/metrics_summary.csv`.

| Model | MAE | RMSE |
| --- | ---: | ---: |
| TCN, 15 epochs | 0.1195 | 0.1595 |
| LSTM | 0.1207 | 0.1567 |
| CNN-LSTM | 0.1212 | 0.1596 |
| GRU | 0.1293 | 0.1661 |
| Transformer | 0.1754 | 0.2226 |

The SHAP and LIME analyses both indicated that recent wind speed at 100 meters was one of the most influential predictors for wind power output.

