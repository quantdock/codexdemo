# ARIMA cycle detection utility

This repository contains a small command line utility for evaluating whether a
univariate time series exhibits a repeating cycle. The script compares the best
non-seasonal ARIMA configuration with seasonal variants (SARIMA) for a
user-specified period.

## Installation

The script relies on NumPy, pandas, and statsmodels. Install them with pip:

```bash
pip install numpy pandas statsmodels
```

## Usage

```bash
python detect_cycle_arima.py <path_to_csv> [options]
```

Key options:

- `--column`: name of the numeric column containing the target series. If
  omitted, the first numeric column is used.
- `--date-column`: optional column to parse as a datetime index (helpful for
  chronological data).
- `--period`: cycle length to test (default: 12).
- `--max-order`: largest AR/MA order to evaluate during model selection
  (default: 2).
- `--aic-threshold`: minimum AIC improvement the seasonal model must provide to
  flag a cycle (default: 2.0).
- `--show-warnings`: display statsmodels convergence and frequency warnings.

## Sample workflow

The snippet below demonstrates generating seasonal monthly data and running the
cycle detector. A `period` of 12 corresponds to one year of monthly seasonality.

```bash
python - <<'PY'
import numpy as np
import pandas as pd

np.random.seed(0)
periods = 120
index = pd.date_range('2010-01-01', periods=periods, freq='ME')
seasonal = 10 * np.sin(2 * np.pi * np.arange(periods) / 12)
trend = np.linspace(0, 5, periods)
noise = np.random.normal(scale=2, size=periods)
values = 50 + trend + seasonal + noise
pd.DataFrame({'date': index, 'value': values}).to_csv('sample_seasonal.csv', index=False)
PY

python detect_cycle_arima.py sample_seasonal.csv --column value --date-column date --period 12
```

With the synthetic dataset above the script reports a strong preference for the
seasonal model, confirming the yearly cycle.
