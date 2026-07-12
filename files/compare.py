"""
Side-by-side comparison of all model test predictions.

Reads each {model}_test_predictions.csv written by the model files,
verifies they all cover the same test targets and actual returns (a
mismatch means a stale file from a different run/data), recomputes the
standard metric block via common_eval.compute_metrics — the exact same
code path as the per-model reports — and prints one table sorted by
daily cross-sectional IC t-stat. Missing files are skipped with a note.

Writes the raw table to model_comparison.csv.
"""

import os

import numpy as np
import pandas as pd

from common_eval import compute_metrics

MODEL_FILES = {
    "ARIMA": "arima_test_predictions.csv",
    "GARCH": "garch_test_predictions.csv",
    "Linear regression": "linreg_test_predictions.csv",
    "Ridge": "ridge_test_predictions.csv",
    "XGBoost": "xgboost_test_predictions.csv",
    "LightGBM": "lightgbm_test_predictions.csv",
    "LSTM": "lstm_test_predictions.csv",
    "GRU": "gru_test_predictions.csv",
}
SUMMARY_FILE = "model_comparison.csv"

results = {}
ref = None            # reference (target_day, inst_id, y_true) for consistency check
ref_name = None

for name, path in MODEL_FILES.items():
    if not os.path.exists(path):
        print(f"missing: {path:35s} ({name} not run yet) — skipped")
        continue

    df = pd.read_csv(path).sort_values(["target_day", "inst_id"]).reset_index(drop=True)

    if ref is None:
        ref, ref_name = df, name
    else:
        same = (
            len(df) == len(ref)
            and df["target_day"].equals(ref["target_day"])
            and df["inst_id"].equals(ref["inst_id"])
            and np.allclose(df["y_true"], ref["y_true"])
        )
        if not same:
            print(f"WARNING: {path} covers different targets/actuals than "
                  f"{ref_name}'s file — stale run or different data?")

    m = compute_metrics(df["y_pred"].to_numpy(), df["y_true"].to_numpy(),
                        df["target_day"].to_numpy())
    m["n_rows"] = len(df)
    results[name] = m

if not results:
    raise SystemExit("no prediction files found — run the model files first")

table = pd.DataFrame(results).T
table = table[["mse", "mae", "r2_vs_zero", "dir_acc", "ic_pearson",
               "ic_spearman", "daily_cs_ic_mean", "daily_cs_ic_t", "n_rows"]]
table["n_rows"] = table["n_rows"].astype(int)
table = table.sort_values("daily_cs_ic_t", ascending=False)
table.to_csv(SUMMARY_FILE)

mse_zero = float(np.mean(ref["y_true"].to_numpy() ** 2))
d_min, d_max = int(ref["target_day"].min()), int(ref["target_day"].max())

print(f"\npredict-zero baseline MSE: {mse_zero:.8e}")
print(f"\n=== model comparison (targets: days {d_min}-{d_max}; "
      f"sorted by daily CS IC t-stat) ===")
print(table.to_string(formatters={
    "mse": "{:.4e}".format,
    "mae": "{:.4e}".format,
    "r2_vs_zero": "{:+.5f}".format,
    "dir_acc": "{:.4f}".format,
    "ic_pearson": "{:+.4f}".format,
    "ic_spearman": "{:+.4f}".format,
    "daily_cs_ic_mean": "{:+.4f}".format,
    "daily_cs_ic_t": "{:+.2f}".format,
}))
print(f"\ntable written to {SUMMARY_FILE}")