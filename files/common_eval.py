"""
Shared test-set evaluation. Every model file calls report_metrics() on the
same test targets (returns on days 400-499, all 51 instruments), so the
printed numbers are directly comparable across models.

compute_metrics() is the pure computation (no printing); it is also used by
compare_models.py so the comparison table goes through the exact same code
path as the per-model reports.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def compute_metrics(pred, y_true, target_days) -> dict:
    pred = np.asarray(pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    target_days = np.asarray(target_days)

    mse = float(np.mean((pred - y_true) ** 2))
    mae = float(np.mean(np.abs(pred - y_true)))
    mse_zero = float(np.mean(y_true ** 2))          # baseline: always predict 0
    r2 = 1.0 - mse / mse_zero
    nz = y_true != 0
    dir_acc = float(np.mean(np.sign(pred[nz]) == np.sign(y_true[nz])))

    varying = float(np.std(pred)) > 0
    ic_pearson = pearsonr(pred, y_true)[0] if varying else np.nan
    ic_spearman = spearmanr(pred, y_true)[0] if varying else np.nan

    # daily cross-sectional Spearman IC
    daily = []
    for d in np.unique(target_days):
        m = target_days == d
        if np.std(pred[m]) == 0 or np.std(y_true[m]) == 0:
            daily.append(np.nan)
        else:
            daily.append(spearmanr(pred[m], y_true[m])[0])
    daily = np.asarray(daily, dtype=float)
    ok = ~np.isnan(daily)
    n_ok = int(ok.sum())
    ic_mean = float(np.nanmean(daily)) if n_ok else np.nan
    ic_std = float(np.nanstd(daily, ddof=1)) if n_ok > 1 else np.nan
    ic_t = ic_mean / (ic_std / np.sqrt(n_ok)) if n_ok > 1 and ic_std > 0 else np.nan

    return {"mse": mse, "mae": mae, "mse_zero": mse_zero, "r2_vs_zero": r2,
            "dir_acc": dir_acc, "ic_pearson": ic_pearson,
            "ic_spearman": ic_spearman, "daily_cs_ic_mean": ic_mean,
            "daily_cs_ic_std": ic_std, "daily_cs_ic_t": ic_t,
            "n_ic_days": n_ok}


def report_metrics(name: str, pred, y_true, target_days, label: str = "test") -> dict:
    m = compute_metrics(pred, y_true, target_days)
    target_days = np.asarray(target_days)

    print(f"\n=== {name} — {label} metrics (targets: days "
          f"{int(target_days.min())}-{int(target_days.max())}) ===")
    print(f"MSE                 : {m['mse']:.8e}   "
          f"(predict-zero baseline: {m['mse_zero']:.8e})")
    print(f"MAE                 : {m['mae']:.8e}")
    print(f"R^2 vs zero         : {m['r2_vs_zero']:+.5f}")
    print(f"directional accuracy: {m['dir_acc']:.4f}")
    print(f"pooled Pearson IC   : {m['ic_pearson']:+.4f}")
    print(f"pooled Spearman IC  : {m['ic_spearman']:+.4f}")
    print(f"daily CS Spearman IC: mean {m['daily_cs_ic_mean']:+.4f}  "
          f"std {m['daily_cs_ic_std']:.4f}  t-stat {m['daily_cs_ic_t']:+.2f}  "
          f"({m['n_ic_days']} days)")
    return m


def save_predictions(path: str, target_days, inst_ids, tickers, y_true, pred,
                     extra: dict | None = None) -> None:
    inst_ids = np.asarray(inst_ids)
    df = pd.DataFrame({
        "target_day": np.asarray(target_days),
        "inst_id": inst_ids,
        "ticker": [tickers[i] for i in inst_ids],
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(pred, dtype=float),
    })
    if extra:
        for k, v in extra.items():
            df[k] = np.asarray(v)
    df.to_csv(path, index=False)
    print(f"test predictions written to {path}")