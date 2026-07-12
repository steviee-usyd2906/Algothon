"""
Shared data loading + feature engineering for the pooled panel models
(linear regression / ridge / lasso / xgboost / lightgbm).

Panel layout: one row per (day t, instrument i), day-major order.
Every feature at day t uses prices up to and including day t only.
The target is the return from day t to t+1 (the return realised ON day t+1):
    - targets on days [0, 400)   -> training window
    - targets on days [350, 400) -> validation / early-stopping slice
    - targets on days [400, 500) -> test window
"""

import numpy as np
import pandas as pd

PRICE_FILE = "prices.txt"
N_DAYS = 500
N_INST = 51
TRAIN_END_DAY = 400        # targets on days [0, 400) are training
TEST_END_DAY = 500         # targets on days [400, 500) are test
VAL_TARGET_START = 350     # targets on days [350, 400): validation / early stopping
EPS = 1e-12


def load_prices(path: str = PRICE_FILE) -> pd.DataFrame:
    prices = pd.read_csv(path, sep=r"\s+")
    assert prices.shape[1] == N_INST, f"expected {N_INST} instruments, got {prices.shape[1]}"
    assert len(prices) >= N_DAYS, f"expected >= {N_DAYS} days, got {len(prices)}"
    return prices.iloc[:N_DAYS].reset_index(drop=True)


def _signed_streak(col: pd.Series) -> pd.Series:
    sgn = np.sign(col)
    grp = (sgn != sgn.shift()).cumsum()
    run = sgn.groupby(grp).cumcount() + 1
    return run * sgn


def build_panel(prices: pd.DataFrame):
    """Build the pooled (day, instrument) feature panel.

    Returns
    -------
    X          : DataFrame, one row per (day, instrument); inst_id and
                 day_mod_5 are category dtype
    y          : np.ndarray, next-day return (target)
    target_day : np.ndarray, day index of the target (= feature day + 1)
    valid      : boolean np.ndarray, rows with finite features and target
    tickers    : list of instrument names (column order = inst_id order)
    """
    n_days, n_inst = prices.shape
    tickers = list(prices.columns)
    ret = prices.pct_change()

    per_inst = {}   # name -> DataFrame (n_days x n_inst), value known at day t
    glob = {}       # name -> Series (n_days), market-wide value known at day t

    # ---- lagged returns (ret_lag_1 = most recent observed return at day t) --
    for k in range(1, 11):
        per_inst[f"ret_lag_{k}"] = ret.shift(k - 1)

    # ---- rolling return moments ---------------------------------------------
    for w in (5, 10, 20, 60):
        per_inst[f"ret_mean_{w}"] = ret.rolling(w).mean()
        per_inst[f"ret_std_{w}"] = ret.rolling(w).std()
    for w in (20, 60):
        per_inst[f"ret_skew_{w}"] = ret.rolling(w).skew()
        per_inst[f"ret_kurt_{w}"] = ret.rolling(w).kurt()

    # ---- momentum (cumulative return over window) ----------------------------
    for w in (5, 10, 20, 60):
        per_inst[f"mom_{w}"] = prices / prices.shift(w) - 1.0

    # ---- exponentially weighted return / volatility --------------------------
    per_inst["ewm_ret_hl5"] = ret.ewm(halflife=5).mean()
    per_inst["ewm_ret_hl20"] = ret.ewm(halflife=20).mean()
    per_inst["ewm_vol_94"] = ret.ewm(alpha=0.06).std()   # RiskMetrics lambda=0.94

    # ---- volatility regime ratios ---------------------------------------------
    per_inst["vol_ratio_5_20"] = per_inst["ret_std_5"] / (per_inst["ret_std_20"] + EPS)
    per_inst["vol_ratio_10_60"] = per_inst["ret_std_10"] / (per_inst["ret_std_60"] + EPS)

    # ---- rolling lag-1 autocorrelation ----------------------------------------
    for w in (20, 60):
        per_inst[f"autocorr1_{w}"] = ret.rolling(w).corr(ret.shift(1))

    # ---- signed up/down streak -------------------------------------------------
    per_inst["streak"] = ret.apply(_signed_streak)

    # ---- fraction of up days, extremes -----------------------------------------
    up = (ret > 0).astype(float).where(ret.notna())
    for w in (5, 10, 20):
        per_inst[f"frac_up_{w}"] = up.rolling(w).mean()
    per_inst["max_ret_20"] = ret.rolling(20).max()
    per_inst["min_ret_20"] = ret.rolling(20).min()

    # ---- price-level features (all scale-free) ----------------------------------
    for w in (20, 60):
        m, s = prices.rolling(w).mean(), prices.rolling(w).std()
        per_inst[f"price_z_{w}"] = (prices - m) / (s + EPS)
    for w in (5, 10, 20, 50):
        per_inst[f"px_over_ma_{w}"] = prices / prices.rolling(w).mean() - 1.0
    per_inst["ma_cross_5_20"] = prices.rolling(5).mean() / prices.rolling(20).mean() - 1.0
    per_inst["ma_cross_10_50"] = prices.rolling(10).mean() / prices.rolling(50).mean() - 1.0

    # ---- RSI ----------------------------------------------------------------------
    for w in (7, 14):
        gains = ret.clip(lower=0).rolling(w).mean()
        losses = (-ret).clip(lower=0).rolling(w).mean()
        per_inst[f"rsi_{w}"] = 100.0 * gains / (gains + losses + EPS)

    # ---- MACD (normalised by price) ------------------------------------------------
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    macd = (ema12 - ema26) / prices
    per_inst["macd"] = macd
    per_inst["macd_signal"] = macd.ewm(span=9).mean()
    per_inst["macd_hist"] = per_inst["macd"] - per_inst["macd_signal"]

    # ---- drawdown / run-up / days since high ---------------------------------------
    for w in (20, 60):
        per_inst[f"drawdown_{w}"] = prices / prices.rolling(w).max() - 1.0
        per_inst[f"runup_{w}"] = prices / prices.rolling(w).min() - 1.0
    per_inst["days_since_max_20"] = 19 - prices.rolling(20).apply(np.argmax, raw=True)

    # ---- cross-sectional features (across instruments at day t) --------------------
    mom5, mom20 = per_inst["mom_5"], per_inst["mom_20"]
    per_inst["cs_rank_ret1"] = ret.rank(axis=1, pct=True)
    per_inst["cs_rank_mom5"] = mom5.rank(axis=1, pct=True)
    per_inst["cs_rank_mom20"] = mom20.rank(axis=1, pct=True)
    per_inst["cs_rank_vol20"] = per_inst["ret_std_20"].rank(axis=1, pct=True)
    per_inst["cs_ret1_demeaned"] = ret.sub(ret.mean(axis=1), axis=0)
    per_inst["cs_z_mom20"] = mom20.sub(mom20.mean(axis=1), axis=0).div(mom20.std(axis=1) + EPS, axis=0)

    # ---- market (equal-weight of all 51) --------------------------------------------
    mkt_ret = ret.mean(axis=1)
    for k in range(1, 6):
        glob[f"mkt_ret_lag_{k}"] = mkt_ret.shift(k - 1)
    glob["mkt_vol_20"] = mkt_ret.rolling(20).std()
    glob["mkt_mom_5"] = mkt_ret.rolling(5).sum()
    glob["mkt_mom_20"] = mkt_ret.rolling(20).sum()
    glob["breadth_1"] = (ret > 0).astype(float).where(ret.notna()).mean(axis=1)
    glob["cs_disp_ma20"] = ret.std(axis=1).rolling(20).mean()

    # ---- beta / idiosyncratic return ---------------------------------------------------
    beta_60 = ret.rolling(60).cov(mkt_ret).div(mkt_ret.rolling(60).var() + EPS, axis=0)
    per_inst["beta_60"] = beta_60
    per_inst["corr_mkt_60"] = ret.rolling(60).corr(mkt_ret)
    per_inst["resid_ret_1"] = ret.sub(beta_60.mul(mkt_ret, axis=0))

    # ---- ALGO-vs-basket spread (known AR(1) structure in this dataset) -------------------
    basket_ret = ret.iloc[:, 1:].mean(axis=1)
    spread_ret = ret.iloc[:, 0] - basket_ret
    spread_cum = spread_ret.cumsum()
    glob["spread_ret_lag1"] = spread_ret
    glob["spread_z_20"] = (spread_cum - spread_cum.rolling(20).mean()) / (spread_cum.rolling(20).std() + EPS)
    glob["spread_z_60"] = (spread_cum - spread_cum.rolling(60).mean()) / (spread_cum.rolling(60).std() + EPS)

    # ---- assemble long panel: one row per (day, instrument), day-major -------------------
    day_idx = np.repeat(np.arange(n_days), n_inst)
    inst_idx = np.tile(np.arange(n_inst), n_days)

    data = {name: df.to_numpy().ravel() for name, df in per_inst.items()}
    for name, s in glob.items():
        data[name] = np.repeat(s.to_numpy(), n_inst)
    data["inst_id"] = inst_idx
    data["is_algo"] = (inst_idx == 0).astype(int)
    data["day_mod_5"] = day_idx % 5

    X = pd.DataFrame(data)
    y = ret.shift(-1).to_numpy().ravel()          # target: return from day t to t+1
    target_day = day_idx + 1

    X = X.replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1).to_numpy() & ~np.isnan(y)

    X["inst_id"] = X["inst_id"].astype("category")
    X["day_mod_5"] = X["day_mod_5"].astype("category")
    return X, y, target_day, valid, tickers


def make_splits(target_day: np.ndarray, valid: np.ndarray):
    """Boolean row masks: (inner_train, val, train_full, test)."""
    inner_train = valid & (target_day < VAL_TARGET_START)
    val_mask = valid & (target_day >= VAL_TARGET_START) & (target_day < TRAIN_END_DAY)
    train_full = valid & (target_day < TRAIN_END_DAY)
    test_mask = valid & (target_day >= TRAIN_END_DAY) & (target_day < TEST_END_DAY)
    return inner_train, val_mask, train_full, test_mask


def encode_for_linear(X: pd.DataFrame, fit_rows: np.ndarray) -> pd.DataFrame:
    """One-hot the categoricals and standardise continuous features for the
    linear models. Scaler statistics come from `fit_rows` only (no leakage).

    is_algo is dropped: together with the full set of inst_id dummies it is
    exactly collinear (is_algo == 1 - sum of the other instrument dummies).
    """
    Xd = pd.get_dummies(X.drop(columns=["is_algo"]),
                        columns=["inst_id", "day_mod_5"],
                        drop_first=True, dtype=float)
    dummy_cols = [c for c in Xd.columns
                  if c.startswith("inst_id_") or c.startswith("day_mod_5_")]
    cont = [c for c in Xd.columns if c not in dummy_cols]
    mu = Xd.loc[fit_rows, cont].mean()
    sd = Xd.loc[fit_rows, cont].std().replace(0.0, 1.0)
    Xd[cont] = (Xd[cont] - mu) / sd
    return Xd
