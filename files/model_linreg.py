"""
Ordinary least squares next-day return prediction on the pooled feature
panel (same features as ridge/lasso/xgboost/lightgbm; categoricals one-hot
encoded, continuous features standardised on training rows).

No hyperparameters, so the fit uses the full training window
(targets on days [0, 400)); test = targets on days [400, 500).
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from features import load_prices, build_panel, make_splits, encode_for_linear
from common_eval import report_metrics, save_predictions

PREDICTIONS_FILE = "linreg_test_predictions.csv"

prices = load_prices()
X, y, target_day, valid, tickers = build_panel(prices)
_, _, train_full, test_mask = make_splits(target_day, valid)

Xd = encode_for_linear(X, train_full)
print(f"features after encoding: {Xd.shape[1]}")
print(f"rows  train: {int(train_full.sum())}   test: {int(test_mask.sum())}")

model = LinearRegression()
model.fit(Xd[train_full], y[train_full])
pred = model.predict(Xd[test_mask])

report_metrics("Linear regression", pred, y[test_mask], target_day[test_mask])
save_predictions(PREDICTIONS_FILE, target_day[test_mask],
                 X.loc[test_mask, "inst_id"].astype(int).to_numpy(), tickers,
                 y[test_mask], pred)
