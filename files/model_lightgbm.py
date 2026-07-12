"""
LightGBM next-day return prediction — maximal feature set + feature importance.

Uses the shared pooled panel from features.py (~90 scale-free features per
(day, instrument) row; no lookahead) and the shared split: targets on days
[0, 400) train (last 50 days peeled off for early stopping only), targets on
days [400, 500) test. Model and parameters are unchanged from the previous
self-contained version — only the feature/eval code moved to shared modules.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from features import load_prices, build_panel, make_splits
from common_eval import report_metrics, save_predictions

SEED = 42
IMPORTANCE_FILE = "lightgbm_feature_importance.csv"
PREDICTIONS_FILE = "lightgbm_test_predictions.csv"

prices = load_prices()
X, y, target_day, valid, tickers = build_panel(prices)
inner_train, val_mask, train_full, test_mask = make_splits(target_day, valid)

X_tr, y_tr = X[inner_train], y[inner_train]
X_val, y_val = X[val_mask], y[val_mask]
X_te, y_te = X[test_mask], y[test_mask]
print(f"features: {X.shape[1]}")
print(f"rows  train: {len(X_tr)}   val (early stopping): {len(X_val)}   test: {len(X_te)}")

model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=31,
    min_child_samples=100,
    colsample_bytree=0.7,
    subsample=0.8,
    subsample_freq=1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
)
print(f"best iteration: {model.best_iteration_}")

pred = model.predict(X_te)
report_metrics("LightGBM", pred, y_te, target_day[test_mask])

# --------------------------- feature importance ------------------------------
booster = model.booster_
imp = pd.DataFrame({
    "feature": booster.feature_name(),
    "gain": booster.feature_importance(importance_type="gain"),
    "split": booster.feature_importance(importance_type="split"),
})
imp["gain_pct"] = 100.0 * imp["gain"] / imp["gain"].sum()
imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
imp.to_csv(IMPORTANCE_FILE, index=False)

print(f"\n=== feature importance (top 30 of {len(imp)}, by gain) ===")
print(imp.head(30).to_string(
    index=False,
    formatters={"gain": "{:.3e}".format, "gain_pct": "{:.2f}".format},
))
print(f"\nfull table written to {IMPORTANCE_FILE}")

save_predictions(PREDICTIONS_FILE, target_day[test_mask],
                 X_te["inst_id"].astype(int).to_numpy(), tickers, y_te, pred)
