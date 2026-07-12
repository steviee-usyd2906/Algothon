"""
XGBoost next-day return prediction on the pooled feature panel, mirroring
the LightGBM setup: same features, same split (early stopping on targets
[350, 400), test on [400, 500)), feature importance saved.

Requires xgboost >= 1.6 (native categorical support with tree_method='hist').
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from features import load_prices, build_panel, make_splits
from common_eval import report_metrics, save_predictions

SEED = 42
IMPORTANCE_FILE = "xgboost_feature_importance.csv"
PREDICTIONS_FILE = "xgboost_test_predictions.csv"

prices = load_prices()
X, y, target_day, valid, tickers = build_panel(prices)
inner_train, val_mask, train_full, test_mask = make_splits(target_day, valid)

X_tr, y_tr = X[inner_train], y[inner_train]
X_val, y_val = X[val_mask], y[val_mask]
X_te, y_te = X[test_mask], y[test_mask]
print(f"features: {X.shape[1]}")
print(f"rows  train: {len(X_tr)}   val (early stopping): {len(X_val)}   test: {len(X_te)}")

model = xgb.XGBRegressor(
    n_estimators=3000,
    learning_rate=0.02,
    max_depth=6,
    min_child_weight=100,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    eval_metric="rmse",
    random_state=SEED,
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"best iteration: {model.best_iteration}")

pred = model.predict(X_te)
report_metrics("XGBoost", pred, y_te, target_day[test_mask])

# --------------------------- feature importance ------------------------------
booster = model.get_booster()
gain = booster.get_score(importance_type="total_gain")
split = booster.get_score(importance_type="weight")
imp = pd.DataFrame({"feature": list(X.columns)})
imp["gain"] = imp["feature"].map(gain).fillna(0.0)
imp["split"] = imp["feature"].map(split).fillna(0.0)
imp["gain_pct"] = 100.0 * imp["gain"] / imp["gain"].sum()
imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
imp.to_csv(IMPORTANCE_FILE, index=False)

print(f"\n=== feature importance (top 30 of {len(imp)}, by total gain) ===")
print(imp.head(30).to_string(
    index=False,
    formatters={"gain": "{:.3e}".format, "gain_pct": "{:.2f}".format},
))
print(f"\nfull table written to {IMPORTANCE_FILE}")

save_predictions(PREDICTIONS_FILE, target_day[test_mask],
                 X_te["inst_id"].astype(int).to_numpy(), tickers, y_te, pred)
