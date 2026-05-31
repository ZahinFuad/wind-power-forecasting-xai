# Methodology

## Dataset

The experiment uses `data/raw/Location1.csv`, an hourly weather and wind power time-series dataset. The target variable is `Power`.

Input features:

- `temperature_2m`
- `relativehumidity_2m`
- `dewpoint_2m`
- `windspeed_10m`
- `windspeed_100m`
- `winddirection_10m`
- `winddirection_100m`
- `windgusts_10m`

## Preprocessing

The preprocessing script sorts records chronologically, removes rows with missing values, scales weather features with MinMax scaling, and converts the time series into supervised learning windows.

Each training sample uses the previous 24 hourly observations to predict wind power for the next hour.

The dataset is split chronologically into training and test sets with no shuffling, preserving time-series order.

## Models

The experiment compares the following neural forecasting models:

- Temporal Convolutional Network (TCN)
- Long Short-Term Memory network (LSTM)
- Gated Recurrent Unit network (GRU)
- CNN-LSTM hybrid
- Transformer baseline

## Evaluation

Model performance is evaluated with:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Normalized RMSE (NRMSE) in selected experiments

## Explainable AI

SHAP and LIME are applied to the trained TCN model to estimate which time-lagged weather variables most strongly influenced individual and global predictions. Additional checks compare explanation consistency, faithfulness, and runtime.

This repository corresponds to a submitted BIM 2025 camera-ready manuscript. The manuscript PDF is intentionally not included while publication is pending.

