"""
Lasso next-day return prediction on the pooled feature panel.

alpha is selected on the time-respecting validation slice (fit on targets
[0, 350), score on targets [350, 400)), then refit on the full training
window (targets [0, 400)). Test = targets on days [400, 500).

Because the target has ~1e-2 scale, useful alphas are tiny (1e-7..1e-3).
The surviving nonzero coefficients act as lasso's feature selection and are
saved to lasso_coefficients.csv.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

from features import load_prices, build_panel, make_splits, encode_for_linear
from common_eval import report_metrics, save_predictions

ALPHAS = np.logspace(-7, -3, 17)
MAX_ITER = 100_000
COEF_FILE = "lasso_coefficients.csv"
PREDICTIONS_FILE = "lasso_test_predictions.csv"

warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
    m = Lasso(alpha=a, max_iter=MAX_ITER).fit(Xd_sel[inner_train], y[inner_train])
    v = float(np.mean((m.predict(Xd_sel[val_mask]) - y[val_mask]) ** 2))
    nnz = int(np.sum(m.coef_ != 0))
    print(f"  alpha={a:9.3g}   val MSE={v * 1e6:.6f}   nonzero={nnz}")
    if v < best_val:
        best_alpha, best_val = a, v
print(f"selected alpha: {best_alpha:g}")

# --- refit on the full training window (scaler refit on those rows) ----------
Xd = encode_for_linear(X, train_full)
model = Lasso(alpha=best_alpha, max_iter=MAX_ITER)
model.fit(Xd[train_full], y[train_full])
pred = model.predict(Xd[test_mask])

report_metrics("Lasso", pred, y[test_mask], target_day[test_mask])

# --- surviving coefficients (lasso's feature selection) -----------------------
coef = pd.DataFrame({"feature": Xd.columns, "coef": model.coef_})
coef["abs_coef"] = coef["coef"].abs()
coef = coef.sort_values("abs_coef", ascending=False).reset_index(drop=True)
coef.to_csv(COEF_FILE, index=False)
nonzero = coef[coef["coef"] != 0]
print(f"\nnonzero coefficients: {len(nonzero)} of {len(coef)} "
      f"(full table written to {COEF_FILE})")
print(nonzero.head(20)[["feature", "coef"]].to_string(
    index=False, formatters={"coef": "{:+.3e}".format}))

save_predictions(PREDICTIONS_FILE, target_day[test_mask],
                 X.loc[test_mask, "inst_id"].astype(int).to_numpy(), tickers,
                 y[test_mask], pred)
