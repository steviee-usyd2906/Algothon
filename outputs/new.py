"""
Pair trading pipeline built on top of the correlation/clustering results.

Steps:
  1. Re-load prices.
  2. Take the top correlated pairs (from the prior correlation analysis)
     and test each for COINTEGRATION (Engle-Granger), not just correlation.
     Correlation tells you two things move together; cointegration tells
     you their spread is actually mean-reverting, which is the real
     requirement for pair trading.
  3. For cointegrated pairs, compute the hedge ratio, build the spread,
     convert to a rolling z-score, and backtest a simple threshold
     entry/exit strategy.
  4. Report performance stats (return, Sharpe, max drawdown, # trades)
     and save an equity curve + spread/z-score plot for the best pair.

Usage:
    python pair_trading.py prices.txt
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm

INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "prices.txt"
OUT_DIR = "."

# ---------------------------------------------------------------------
# Config — tune these
# ---------------------------------------------------------------------
N_CANDIDATE_PAIRS = None    # None = test ALL pairs, not just top-correlated ones
LOOKBACK = 30               # rolling window for z-score (days)
ENTRY_Z = 2.0                # open a position when |z| exceeds this
EXIT_Z = 0.5                 # close when |z| falls back below this
STOP_Z = 4.0                 # hard stop-loss if spread keeps diverging
COINT_PVAL_MAX = 0.05         # require Engle-Granger p-value below this

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv(INPUT_PATH, sep=r"\s+", header=0)
returns = np.log(df / df.shift(1)).dropna()
corr = returns.corr()

pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
pairs = pairs.dropna()
pairs = pairs.rename("correlation").reset_index()
pairs.columns = ["a", "b", "correlation"]
pairs = pairs.reindex(pairs.correlation.abs().sort_values(ascending=False).index)
candidates = pairs if N_CANDIDATE_PAIRS is None else pairs.head(N_CANDIDATE_PAIRS)
print(f"Testing {len(candidates)} pairs for cointegration...")

# ---------------------------------------------------------------------
# 2. Cointegration screen
# ---------------------------------------------------------------------
results = []
for _, row in candidates.iterrows():
    a, b = row["a"], row["b"]
    score, pval, _ = coint(df[a], df[b])
    results.append({"a": a, "b": b, "correlation": row["correlation"],
                     "coint_pvalue": pval})

coint_df = pd.DataFrame(results).sort_values("coint_pvalue")
coint_df.to_csv(f"{OUT_DIR}/cointegration_screen.csv", index=False)

print("Cointegration screen — top 20 pairs by p-value (lower = stronger evidence "
      "of a stable, tradeable spread):")
print(coint_df.head(20).to_string(index=False))

tradeable = coint_df[coint_df.coint_pvalue < COINT_PVAL_MAX]
print(f"\n{len(tradeable)} of {len(coint_df)} tested pairs are cointegrated "
      f"at p < {COINT_PVAL_MAX}")

if tradeable.empty:
    print("\nNo pairs cleared the cointegration bar. Consider testing more "
          "candidate pairs (raise N_CANDIDATE_PAIRS) or relaxing COINT_PVAL_MAX.")
    sys.exit(0)

# ---------------------------------------------------------------------
# 3. Backtest a simple z-score strategy on each cointegrated pair
# ---------------------------------------------------------------------
def backtest_pair(a, b, df, lookback=LOOKBACK, entry_z=ENTRY_Z,
                   exit_z=EXIT_Z, stop_z=STOP_Z):
    # Hedge ratio via OLS on price levels
    X = sm.add_constant(df[b])
    model = sm.OLS(df[a], X).fit()
    hedge_ratio = model.params[b]

    spread = df[a] - hedge_ratio * df[b]
    roll_mean = spread.rolling(lookback).mean()
    roll_std = spread.rolling(lookback).std()
    zscore = (spread - roll_mean) / roll_std

    position = np.zeros(len(df))  # +1 = long spread, -1 = short spread
    pos = 0
    for i in range(len(df)):
        z = zscore.iloc[i]
        if np.isnan(z):
            position[i] = pos
            continue
        if pos == 0:
            if z > entry_z:
                pos = -1
            elif z < -entry_z:
                pos = 1
        elif pos == 1:
            if z > -exit_z or z < -stop_z:
                pos = 0
        elif pos == -1:
            if z < exit_z or z > stop_z:
                pos = 0
        position[i] = pos

    position = pd.Series(position, index=df.index).shift(1).fillna(0)  # trade next bar
    spread_ret = spread.diff()
    strat_pnl = position * spread_ret
    equity = strat_pnl.cumsum()

    n_trades = int((position.diff().abs() > 0).sum())
    daily = strat_pnl.dropna()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else np.nan
    max_dd = (equity.cummax() - equity).max()

    return {
        "a": a, "b": b, "hedge_ratio": hedge_ratio,
        "total_pnl": equity.iloc[-1], "sharpe": sharpe,
        "max_drawdown": max_dd, "n_trades": n_trades,
        "equity": equity, "zscore": zscore, "spread": spread,
    }

backtests = [backtest_pair(r["a"], r["b"], df) for _, r in tradeable.iterrows()]
summary = pd.DataFrame([{k: v for k, v in b.items()
                          if k not in ("equity", "zscore", "spread")}
                         for b in backtests]).sort_values("sharpe", ascending=False)
summary.to_csv(f"{OUT_DIR}/pair_trading_backtest_summary.csv", index=False)

print("\nBacktest summary (ranked by Sharpe):")
print(summary.to_string(index=False))

# ---------------------------------------------------------------------
# 4. Plot the best pair
# ---------------------------------------------------------------------
best = max(backtests, key=lambda b: (b["sharpe"] if not np.isnan(b["sharpe"]) else -999))

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axes[0].plot(df[best["a"]], label=best["a"])
axes[0].plot(df[best["b"]] * best["hedge_ratio"], label=f"{best['b']} x hedge ratio")
axes[0].set_title(f"{best['a']} vs {best['b']} (hedge ratio={best['hedge_ratio']:.3f})")
axes[0].legend()

axes[1].plot(best["zscore"], color="purple")
axes[1].axhline(ENTRY_Z, color="red", linestyle="--", linewidth=0.8)
axes[1].axhline(-ENTRY_Z, color="red", linestyle="--", linewidth=0.8)
axes[1].axhline(EXIT_Z, color="green", linestyle=":", linewidth=0.8)
axes[1].axhline(-EXIT_Z, color="green", linestyle=":", linewidth=0.8)
axes[1].set_title("Spread z-score with entry/exit thresholds")

axes[2].plot(best["equity"], color="black")
axes[2].set_title(f"Strategy equity curve (Sharpe={best['sharpe']:.2f}, "
                   f"trades={best['n_trades']})")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/best_pair_backtest.png", dpi=150)
plt.close()

print(f"\nBest pair by Sharpe: {best['a']} / {best['b']} "
      f"(Sharpe={best['sharpe']:.2f}, PnL={best['total_pnl']:.2f}, "
      f"trades={best['n_trades']})")
print("Plot saved to best_pair_backtest.png")