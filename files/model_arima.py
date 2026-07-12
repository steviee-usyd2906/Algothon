"""
Per-instrument ARIMA next-day return prediction.

Order selection: AIC over p, q in {0,1,2} (d=0; returns are already
stationary), fitted per instrument on the training returns (days 1-399).

Test predictions are true one-step-ahead forecasts: the train-estimated
parameters are held fixed and applied to the full return series via the
Kalman filter, so the forecast for day d uses actual returns through day
d-1 while no test-period data enters parameter estimation.

Runtime: 51 instruments x 9 candidate orders — expect a few minutes.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from features import load_prices, TRAIN_END_DAY, TEST_END_DAY
from common_eval import report_metrics, save_predictions

ORDERS = [(p, 0, q) for p in range(3) for q in range(3)]
PREDICTIONS_FILE = "arima_test_predictions.csv"

warnings.filterwarnings("ignore")

prices = load_prices()
tickers = list(prices.columns)
n_inst = prices.shape[1]
ret = prices.pct_change()
n_test = TEST_END_DAY - TRAIN_END_DAY

pred_mat = np.full((n_test, n_inst), np.nan)
chosen = []

for j, col in enumerate(tickers):
    y = ret[col].to_numpy()[1:]                  # positions 0..498 = days 1..499
    y_train = y[: TRAIN_END_DAY - 1]             # returns on days 1..399

    best_order, best_aic, best_params = None, np.inf, None
    for order in ORDERS:
        try:
            res = SARIMAX(y_train, order=order, trend="c").fit(disp=False)
            if np.isfinite(res.aic) and res.aic < best_aic:
                best_order, best_aic, best_params = order, res.aic, res.params
        except Exception:
            continue

    if best_order is None:                        # extremely unlikely fallback
        pred_mat[:, j] = y_train.mean()
        chosen.append((0, 0, 0))
        print(f"{col}: all fits failed, using train mean")
        continue

    # apply the train parameters to the full series (filter = no re-estimation);
    # one-step-ahead predicted mean at position p uses observations < p only.
    # Positions 399..498 correspond to the returns on days 400..499.
    res_full = SARIMAX(y, order=best_order, trend="c").filter(best_params)
    pm = res_full.get_prediction(start=TRAIN_END_DAY - 1,
                                 end=TEST_END_DAY - 2).predicted_mean
    pred_mat[:, j] = pm
    chosen.append(best_order)
    print(f"{col}: order={best_order}  AIC={best_aic:.1f}")

print("\nselected orders (count):")
print(pd.Series([str(o) for o in chosen]).value_counts().to_string())

y_true = ret.iloc[TRAIN_END_DAY:TEST_END_DAY].to_numpy()
target_days = np.repeat(np.arange(TRAIN_END_DAY, TEST_END_DAY), n_inst)
inst_ids = np.tile(np.arange(n_inst), n_test)

report_metrics("ARIMA", pred_mat.ravel(), y_true.ravel(), target_days)
save_predictions(PREDICTIONS_FILE, target_days, inst_ids, tickers,
                 y_true.ravel(), pred_mat.ravel())
