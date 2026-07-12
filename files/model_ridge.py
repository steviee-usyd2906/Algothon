"""
Ridge regression next-day return prediction on the pooled feature panel.

alpha is selected on the time-respecting validation slice (fit on targets
[0, 350), score on targets [350, 400)), then the model is refit with the
chosen alpha on the full training window (targets [0, 400)).
Test = targets on days [400, 500).
"""

import numpy as np
from sklearn.linear_model import Ridge

from features import load_prices, build_panel, make_splits, encode_for_linear
from common_eval import report_metrics, save_predictions

ALPHAS = np.logspace(-2, 6, 17)
PREDICTIONS_FILE = "ridge_test_predictions.csv"

prices = load_prices()
X, y, target_day, valid, tickers = build_panel(prices)
inner_train, val_mask, train_full, test_mask = make_splits(target_day, valid)

# --- alpha selection (scaler fit on inner-train rows only) -------------------
Xd_sel = encode_for_linear(X, inner_train)
print(f"features after encoding: {Xd_sel.shape[1]}")
print(f"rows  train: {int(inner_train.sum())}   val: {int(val_mask.sum())}   "
      f"test: {int(test_mask.sum())}")

best_alpha, best_val = None, np.inf
print("\nalpha selection (val MSE x 1e6):")
for a in ALPHAS:
    m = Ridge(alpha=a).fit(Xd_sel[inner_train], y[inner_train])
    v = float(np.mean((m.predict(Xd_sel[val_mask]) - y[val_mask]) ** 2))
    print(f"  alpha={a:10.4g}   val MSE={v * 1e6:.6f}")
    if v < best_val:
        best_alpha, best_val = a, v
print(f"selected alpha: {best_alpha:g}")

# --- refit on the full training window (scaler refit on those rows) ----------
Xd = encode_for_linear(X, train_full)
model = Ridge(alpha=best_alpha)
model.fit(Xd[train_full], y[train_full])
pred = model.predict(Xd[test_mask])

report_metrics("Ridge", pred, y[test_mask], target_day[test_mask])
save_predictions(PREDICTIONS_FILE, target_day[test_mask],
                 X.loc[test_mask, "inst_id"].astype(int).to_numpy(), tickers,
                 y[test_mask], pred)
