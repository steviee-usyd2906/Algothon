"""
Single-file Algothon submission: getMyPosition(prcSoFar) for eval.py.

Same signal engine and risk-management shape as before (multi-horizon
EWMAC trend + short-horizon mean reversion, trend-only rolling-drawdown
scale, whole-book CVaR cap, global fractional-Kelly haircut, Ledoit-Wolf
vol targeting, cointegrated pairs overlay). What's new: the four risk
thresholds that were previously hand-picked constants --

    DD_CAP_DOLLARS, DD_WINDOW, CVAR_DOLLAR_BUDGET, KELLY_FRACTION

-- are now CALIBRATED AUTOMATICALLY, lazily, the first time enough price
history is available, instead of being fixed in advance.

HOW THE LAZY SWEEP WORKS
-------------------------
The first time getMyPosition(prcSoFar) is called with enough history to
be useful, it:

  1. Holds out the last SWEEP_HOLDOUT_DAYS days of prcSoFar entirely --
     the sweep never sees them. This is a deliberate walk-forward gap:
     it stops the calibration from fitting the four thresholds to
     whatever regime happens to be freshest, which is exactly the
     stretch you'd otherwise be most likely to overfit to.
  2. On the remaining ("training") history, runs a full internal
     backtest simulation -- reusing the exact same trend / reversion /
     vol / Ledoit-Wolf-covariance machinery used live -- for every
     combination of the four thresholds on a small grid (see
     *_GRID below).
  3. Scores each combination on out-of-sample-free simulated daily P&L
     using a Sharpe-minus-drawdown-penalty objective (SWEEP_DD_PENALTY),
     matching the stated goal: reduce negative-P&L stretches AND
     minimise realised vol, not just maximise raw return.
  4. Keeps the winning combination in _state["params"] for the rest of
     the run. The sweep runs exactly once; every later call reuses the
     calibrated numbers.

If there isn't yet enough history to leave out SWEEP_HOLDOUT_DAYS AND
still have a meaningful training window, getMyPosition falls back to
the DEFAULT_* values below and retries calibration on the next call
once more history has accumulated. This only matters for the very first
stretch of an eval run; once calibrated, it's calibrated for good.

WHAT THE SWEEP DELIBERATELY LEAVES OUT (for speed and because they don't
depend on the four swept thresholds): the pairs overlay, position
smoothing, and the minimum-activity churn. None of those are functions
of DD_CAP_DOLLARS / DD_WINDOW / CVAR_DOLLAR_BUDGET / KELLY_FRACTION, so
including them in the sweep would only add runtime without changing
which combination wins. They are still fully active in the live path.

The trend/mean-reversion signal construction, the Ledoit-Wolf covariance
overlay, the pairs overlay, position smoothing, and the minimum-volume
guarantee are otherwise unchanged from the previous version.

MERGED FROM v1: the hysteretic PnL-feedback de-risker (whole-book kill
switch). It tracks the realised daily P&L of the smoothed signal book
(normalised back to full size so the history is scale-independent) and,
when trailing DERISK_WINDOW-day P&L drops below DERISK_ENTER_Z trailing
stdevs, multiplies the ENTIRE signal book by DERISK_SCALE until P&L
recovers above DERISK_EXIT_Z stdevs. Applied as the LAST multiplicative
overlay, after the Kelly haircut, Ledoit-Wolf vol targeting and CVaR cap.
Set DERISK_SCALE = 1.0 to disable it entirely (prior walk-forward found
this mechanism can whipsaw -- it helped some blocks and hurt others --
so validate it toggled on/off before trusting the combination). Note it
partially overlaps with the calibrated trend-book drawdown scale: both
react to losing stretches, so with both active a bad run can be
double-de-risked.
"""

import numpy as np

# ----------------------------- signal config ------------------------------
TREND_SPANS = [(8, 24), (16, 48), (32, 96)]
MIN_TREND_HISTORY = 2 * max(slow for _, slow in TREND_SPANS)

REV_LOOKBACK = 5
MIN_REV_HISTORY = REV_LOOKBACK + 5

MOM_WEIGHT = 0.6
REV_WEIGHT = 0.4

VOL_HALFLIFE_FAST = 20
VOL_HALFLIFE_SLOW = 60
VOL_BLEND = 0.5

# ----------------------------- inference config --------------------------
DEFAULT_DLR_POS_LIMIT = 10_000
INST0_DLR_POS_LIMIT = 100_000

RISK_FRACTION = 1.0
TAU = 0.1
ALGO_RISK_FRACTION = 1.0
SIGNAL_KEEP_TOP_K = 20

# --- swept thresholds: DEFAULTS used only until/unless calibration runs ---
DEFAULT_KELLY_FRACTION = 0.5
DEFAULT_DD_CAP_DOLLARS = 6_000.0
DEFAULT_DD_WINDOW = 60
DEFAULT_CVAR_DOLLAR_BUDGET = 5_000.0

# --- sweep grid for the four thresholds above ------------------------------
KELLY_GRID = [0.25, 0.5, 0.75, 1.0]
DD_CAP_GRID = [3_000.0, 6_000.0, 10_000.0]
DD_WINDOW_GRID = [30, 60, 90]
CVAR_BUDGET_GRID = [3_000.0, 5_000.0, 8_000.0]

SWEEP_HOLDOUT_DAYS = 100      # never calibrate on the most recent N days
SWEEP_MIN_TRAIN_DAYS = 60     # need at least this many simulated days to trust a score
SWEEP_DD_PENALTY = 0.25       # objective = Sharpe - SWEEP_DD_PENALTY * (maxDD / $10k)

# --- PnL-feedback de-risker (hysteretic kill switch, merged from v1) ------
DERISK_SCALE = 0.5        # 1.0 disables the de-risker entirely
DERISK_WINDOW = 50
DERISK_ENTER_Z = -1.5
DERISK_EXIT_Z = 0.0

# --- non-swept risk-management constants (fixed, same as before) ----------
DD_MIN_SCALE = 0.25
DD_RECOVER_FRAC = 0.5
CVAR_LOOKBACK = 250
CVAR_MIN_HISTORY = 40
CVAR_ALPHA = 0.05

# --- portfolio-level correlation overlay (Ledoit-Wolf, unchanged) --------
COV_MAX_LOOKBACK = 250
COV_MIN_HISTORY = 20
TARGET_PORTFOLIO_DAILY_VOL = 4000.0

# --- turnover / commission control ----------------------------------------
SMOOTH_ALPHA = 0.35
MIN_TRADE_DOLLARS = 200.0

# --- pairs-trading overlay (unchanged; deliberately NOT drawdown-managed,
# per Kaminski & Lo -- this book is mean-reverting by construction) --------
PAIRS = [
    (5, 21, -0.071235),
    (13, 45, 0.597452),
    (49, 50, 0.449571),
    (18, 32, 1.162910),
    (36, 50, 0.483469),
    (7, 40, 0.942283),
    (18, 42, 1.810502),
    (10, 46, 1.395985),
    (1, 20, 0.904842),
    (31, 43, 0.824046),
    (14, 36, -4.895907),
    (26, 32, 0.604611),
]
PAIR_Z_LOOKBACK = 60
PAIR_Z_ENTRY = 2.0
PAIR_Z_EXIT = 0.5
PAIR_TRADE_DOLLARS = 1500.0
PAIR_SIGNAL_GATE = True

# --- minimum-activity guarantee -------------------------------------------
MIN_TOTAL_DVOLUME = 25_000.0
VOLUME_TARGET = 1.5 * MIN_TOTAL_DVOLUME
VOLUME_CHECK_CALLS = 20
CHURN_DOLLARS = 2_000.0


# ==========================================================================
# statistics helpers (shared by the live path and the sweep simulation)
# ==========================================================================

def _ema(prices, halflife):
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    out = np.empty_like(prices, dtype=float)
    out[:, 0] = prices[:, 0]
    for i in range(1, prices.shape[1]):
        out[:, i] = alpha * prices[:, i] + (1.0 - alpha) * out[:, i - 1]
    return out


def _ewma_vol(log_ret, halflife):
    n = log_ret.shape[1]
    if n < 2:
        return np.full(log_ret.shape[0], 1e-6)
    halflife = max(1.0, min(halflife, n))
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(n - 1, -1, -1)
    w = w / w.sum()
    mean = (w * log_ret).sum(axis=1, keepdims=True)
    var = (w * (log_ret - mean) ** 2).sum(axis=1)
    return np.sqrt(np.maximum(var, 1e-12))


def _ledoit_wolf_shrinkage(returns):
    """Ledoit & Wolf (2004) analytic shrinkage covariance estimator,
    target = (trace(S)/N) * I, closed-form optimal shrinkage intensity."""
    T, N = returns.shape
    X = returns - returns.mean(axis=0, keepdims=True)
    S = (X.T @ X) / T
    mu = np.trace(S) / N
    F = mu * np.eye(N)

    d2 = np.sum((S - F) ** 2) / N

    b_bar2 = 0.0
    for t in range(T):
        xt = X[t]
        diff = np.outer(xt, xt) - S
        b_bar2 += np.sum(diff ** 2)
    b_bar2 /= (T ** 2 * N)

    b2 = min(b_bar2, d2)
    shrinkage = 0.0 if d2 < 1e-18 else b2 / d2
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))

    return shrinkage * F + (1.0 - shrinkage) * S


def _trend_signal(prcSoFar, vol):
    nInst, t = prcSoFar.shape
    if t < MIN_TREND_HISTORY:
        return np.zeros(nInst)
    scores = np.zeros(nInst)
    for fast, slow in TREND_SPANS:
        ema_fast = _ema(prcSoFar, fast)[:, -1]
        ema_slow = _ema(prcSoFar, slow)[:, -1]
        raw = (ema_fast - ema_slow) / (prcSoFar[:, -1] * np.maximum(vol, 1e-8))
        scores += raw
    return scores / len(TREND_SPANS)


def _reversion_signal(prcSoFar, vol):
    nInst, t = prcSoFar.shape
    if t < MIN_REV_HISTORY:
        return np.zeros(nInst)
    window = prcSoFar[:, -REV_LOOKBACK:]
    mean = window.mean(axis=1)
    std = window.std(axis=1)
    std = np.maximum(std, prcSoFar[:, -1] * np.maximum(vol, 1e-8) * np.sqrt(REV_LOOKBACK))
    z = (prcSoFar[:, -1] - mean) / std
    return -z


def _dollar_limits(nInst):
    limits = np.full(nInst, DEFAULT_DLR_POS_LIMIT, dtype=float)
    if nInst > 0:
        limits[0] = INST0_DLR_POS_LIMIT
    return limits


def _historical_cvar(dollar_position, window):
    """Mean $ P&L of the worst CVAR_ALPHA fraction of days in `window`
    (a nInst x T slice of log returns), simulating the CURRENT candidate
    book against each day's actual historical return vector."""
    simulated_pnl = dollar_position @ window
    n_tail = max(1, int(np.ceil(CVAR_ALPHA * simulated_pnl.shape[0])))
    tail = np.sort(simulated_pnl)[:n_tail]
    return float(tail.mean())


# ==========================================================================
# lazy sweep: precompute (once) the combo-independent pieces, then score
# every (KELLY_FRACTION, DD_CAP_DOLLARS, DD_WINDOW, CVAR_DOLLAR_BUDGET)
# combination cheaply against those precomputed pieces.
# ==========================================================================

def _precompute_for_sweep(train_prices):
    """One pass over the training window computing everything that does
    NOT depend on the four swept thresholds: the raw (pre-drawdown,
    pre-Kelly, pre-CVaR) trend/reversion dollar books each day, the
    Ledoit-Wolf covariance matrix each day, and the full-size trend
    book's own realised daily P&L (used later for the drawdown calc)."""
    nInst, T = train_prices.shape
    full_log_ret = np.diff(np.log(train_prices), axis=1)  # nInst x (T-1)

    dlr_limits = _dollar_limits(nInst)
    risk_frac = np.full(nInst, RISK_FRACTION)
    risk_frac[0] *= ALGO_RISK_FRACTION

    t_start = MIN_TREND_HISTORY
    if T - 1 <= t_start:
        return None  # not enough history to simulate even one day

    days, trend_list, rev_list, sigma_list, price_list = [], [], [], [], []
    for t in range(t_start, T - 1):
        prc_slice = train_prices[:, : t + 1]
        lr_slice = full_log_ret[:, :t]

        vol_fast = _ewma_vol(lr_slice, VOL_HALFLIFE_FAST)
        vol_slow = _ewma_vol(lr_slice, VOL_HALFLIFE_SLOW)
        vol = np.maximum(VOL_BLEND * vol_fast + (1 - VOL_BLEND) * vol_slow, 1e-6)

        trend_score = _trend_signal(prc_slice, vol)
        reversion_score = _reversion_signal(prc_slice, vol)
        raw_score = MOM_WEIGHT * trend_score + REV_WEIGHT * reversion_score

        trade_mask = np.ones(nInst, dtype=bool)
        if SIGNAL_KEEP_TOP_K is not None and 0 < SIGNAL_KEEP_TOP_K < nInst:
            kth = np.partition(np.abs(raw_score), -SIGNAL_KEEP_TOP_K)[-SIGNAL_KEEP_TOP_K]
            trade_mask = np.abs(raw_score) >= kth

        trend_dollar_full = np.where(
            trade_mask, np.tanh(MOM_WEIGHT * trend_score / TAU) * dlr_limits * risk_frac, 0.0
        )
        reversion_dollar = np.where(
            trade_mask, np.tanh(REV_WEIGHT * reversion_score / TAU) * dlr_limits * risk_frac, 0.0
        )

        hist_len = lr_slice.shape[1]
        Sigma = None
        if hist_len >= COV_MIN_HISTORY:
            cov_window = lr_slice[:, -min(hist_len, COV_MAX_LOOKBACK):].T
            Sigma = _ledoit_wolf_shrinkage(cov_window)

        days.append(t)
        trend_list.append(trend_dollar_full)
        rev_list.append(reversion_dollar)
        sigma_list.append(Sigma)
        price_list.append(train_prices[:, t])

    if len(days) < SWEEP_MIN_TRAIN_DAYS:
        return None

    prices = np.array(price_list)                                   # (Td, nInst)
    next_prices = train_prices[:, [d + 1 for d in days]].T           # (Td, nInst)
    trend_dollar_full = np.array(trend_list)                         # (Td, nInst)
    reversion_dollar = np.array(rev_list)                            # (Td, nInst)

    # full-size trend book's own realised daily $ P&L -- independent of
    # every swept threshold, so computed once here and reused by every combo
    trend_shares = trend_dollar_full / prices
    trend_pnl_series = np.sum(trend_shares * (next_prices - prices), axis=1)  # (Td,)

    return {
        "days": days,
        "trend_dollar_full": trend_dollar_full,
        "reversion_dollar": reversion_dollar,
        "sigma_list": sigma_list,
        "prices": prices,
        "next_prices": next_prices,
        "trend_pnl_series": trend_pnl_series,
        "full_log_ret": full_log_ret,
    }


def _simulate_combo(pre, kelly, dd_cap, dd_window, cvar_budget):
    """Cheap per-combo pass: reuses everything precomputed above, only
    re-running the parts that actually depend on the four swept values
    (drawdown scale trajectory, Kelly haircut, CVaR cap)."""
    days = pre["days"]
    Td = len(days)
    trend_dollar_full = pre["trend_dollar_full"]
    reversion_dollar = pre["reversion_dollar"]
    prices = pre["prices"]
    next_prices = pre["next_prices"]
    sigma_list = pre["sigma_list"]
    trend_pnl_series = pre["trend_pnl_series"]
    full_log_ret = pre["full_log_ret"]

    trend_scale = 1.0
    pnl_series = np.zeros(Td)

    for i in range(Td):
        window_pnls = trend_pnl_series[max(0, i - dd_window):i]
        if len(window_pnls) >= 2:
            wealth = np.cumsum(window_pnls)
            running_peak = np.maximum.accumulate(np.concatenate([[0.0], wealth]))[1:]
            drawdown = float(max(0.0, running_peak[-1] - wealth[-1]))
        else:
            drawdown = 0.0

        if drawdown >= dd_cap:
            target_scale = DD_MIN_SCALE
        elif drawdown <= DD_RECOVER_FRAC * dd_cap:
            target_scale = 1.0
        else:
            frac = (drawdown - DD_RECOVER_FRAC * dd_cap) / ((1 - DD_RECOVER_FRAC) * dd_cap)
            target_scale = 1.0 - frac * (1.0 - DD_MIN_SCALE)
        trend_scale = trend_scale + 0.5 * (target_scale - trend_scale)

        dollar_position = (trend_dollar_full[i] * trend_scale + reversion_dollar[i]) * kelly

        Sigma = sigma_list[i]
        if Sigma is not None and np.isfinite(TARGET_PORTFOLIO_DAILY_VOL):
            port_var = dollar_position @ Sigma @ dollar_position
            port_vol = np.sqrt(max(port_var, 0.0))
            if port_vol > 1e-9:
                dollar_position = dollar_position * min(1.0, TARGET_PORTFOLIO_DAILY_VOL / port_vol)

        t_full = days[i]
        if t_full >= CVAR_MIN_HISTORY:
            window = full_log_ret[:, max(0, t_full - CVAR_LOOKBACK):t_full]
            cvar = _historical_cvar(dollar_position, window)
            if cvar < -cvar_budget:
                dollar_position = dollar_position * (cvar_budget / (-cvar))

        shares = dollar_position / prices[i]
        pnl_series[i] = float(shares @ (next_prices[i] - prices[i]))

    return pnl_series


def _score(pnl_series):
    if pnl_series.std() < 1e-9:
        sharpe = 0.0
    else:
        sharpe = float(pnl_series.mean() / pnl_series.std() * np.sqrt(252))
    equity = np.cumsum(pnl_series)
    running_peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    max_dd = float(np.max(running_peak - equity)) if len(equity) else 0.0
    return sharpe - SWEEP_DD_PENALTY * (max_dd / 10_000.0)


def _calibrate_thresholds(prcSoFar):
    """Returns a params dict. If there isn't yet enough history to hold
    out SWEEP_HOLDOUT_DAYS and still simulate SWEEP_MIN_TRAIN_DAYS, returns
    None so the caller keeps using defaults and retries next call."""
    nInst, T = prcSoFar.shape
    if T <= SWEEP_HOLDOUT_DAYS:
        return None
    train_prices = prcSoFar[:, : T - SWEEP_HOLDOUT_DAYS]

    pre = _precompute_for_sweep(train_prices)
    if pre is None:
        return None

    best_score = -np.inf
    best_params = None
    for kelly in KELLY_GRID:
        for dd_cap in DD_CAP_GRID:
            for dd_window in DD_WINDOW_GRID:
                for cvar_budget in CVAR_BUDGET_GRID:
                    pnl_series = _simulate_combo(pre, kelly, dd_cap, dd_window, cvar_budget)
                    score = _score(pnl_series)
                    if score > best_score:
                        best_score = score
                        best_params = {
                            "KELLY_FRACTION": kelly,
                            "DD_CAP_DOLLARS": dd_cap,
                            "DD_WINDOW": dd_window,
                            "CVAR_DOLLAR_BUDGET": cvar_budget,
                        }

    best_params["_sweep_score"] = best_score
    best_params["_sweep_train_days"] = len(pre["days"])
    return best_params


# ------------------------------------------------------------------
# state
# ------------------------------------------------------------------
_state = {
    "dlr_limits": None,
    "last_pos": None,
    "last_trend_pos": None,       # full-size (pre-drawdown-scale) trend shares
    "last_reversion_pos": None,   # reversion shares (never drawdown-scaled)
    "trend_scale": 1.0,           # current continuous drawdown scale on trend book
    "trend_pnl_hist": [],         # full-size trend PnL, for the live drawdown calc
    "signal_pnl_hist": [],        # smoothed-signal-book PnL, for the de-risker
    "derisked": False,            # hysteretic de-risker state (merged from v1)
    "last_scale": 1.0,            # de-risk scale in force yesterday (for PnL normalisation)
    "pair_state": [{"dir": 0, "shares_a": 0, "shares_b": 0} for _ in PAIRS],
    "cum_dvolume": 0.0, "call_count": 0, "churn_sign": 1,
    "params": {                   # live values of the four swept thresholds;
        "KELLY_FRACTION": DEFAULT_KELLY_FRACTION,      # overwritten once
        "DD_CAP_DOLLARS": DEFAULT_DD_CAP_DOLLARS,      # _calibrate_thresholds
        "DD_WINDOW": DEFAULT_DD_WINDOW,                # succeeds
        "CVAR_DOLLAR_BUDGET": DEFAULT_CVAR_DOLLAR_BUDGET,
    },
    "calibrated": False,
}


def getMyPosition(prcSoFar):
    nInst, t = prcSoFar.shape

    if not _state["calibrated"]:
        result = _calibrate_thresholds(prcSoFar)
        if result is not None:
            _state["params"].update(
                {k: v for k, v in result.items() if not k.startswith("_")}
            )
            _state["calibrated"] = True
            # Uncomment to see what the sweep picked, and on how much data:
            # print(f"[calibration] {result}")

    params = _state["params"]
    kelly_fraction = params["KELLY_FRACTION"]
    dd_cap_dollars = params["DD_CAP_DOLLARS"]
    dd_window = params["DD_WINDOW"]
    cvar_dollar_budget = params["CVAR_DOLLAR_BUDGET"]

    if _state["dlr_limits"] is None:
        _state["dlr_limits"] = _dollar_limits(nInst)
    dlr_limits = _state["dlr_limits"]
    curPrices = prcSoFar[:, -1]

    full_log_ret = np.diff(np.log(prcSoFar), axis=1)

    vol_fast = _ewma_vol(full_log_ret, VOL_HALFLIFE_FAST)
    vol_slow = _ewma_vol(full_log_ret, VOL_HALFLIFE_SLOW)
    vol = VOL_BLEND * vol_fast + (1 - VOL_BLEND) * vol_slow
    vol = np.maximum(vol, 1e-6)

    # ---- signals, kept SEPARATE so they can be risk-managed asymmetrically
    trend_score = _trend_signal(prcSoFar, vol)
    reversion_score = _reversion_signal(prcSoFar, vol)
    raw_score = MOM_WEIGHT * trend_score + REV_WEIGHT * reversion_score

    trade_mask = np.ones(nInst, dtype=bool)
    if SIGNAL_KEEP_TOP_K is not None and 0 < SIGNAL_KEEP_TOP_K < nInst:
        kth = np.partition(np.abs(raw_score), -SIGNAL_KEEP_TOP_K)[-SIGNAL_KEEP_TOP_K]
        trade_mask = np.abs(raw_score) >= kth

    risk_frac = np.full(nInst, RISK_FRACTION)
    risk_frac[0] *= ALGO_RISK_FRACTION

    trend_dollar_full = np.where(
        trade_mask, np.tanh(MOM_WEIGHT * trend_score / TAU) * dlr_limits * risk_frac, 0.0
    )
    reversion_dollar = np.where(
        trade_mask, np.tanh(REV_WEIGHT * reversion_score / TAU) * dlr_limits * risk_frac, 0.0
    )

    # ---- continuous rolling-drawdown scale on the TREND book only --------
    if t >= 2 and _state["last_trend_pos"] is not None:
        dprice = prcSoFar[:, -1] - prcSoFar[:, -2]
        pnl = float(_state["last_trend_pos"] @ dprice)
        _state["trend_pnl_hist"].append(pnl)

    # ---- de-risker PnL bookkeeping (merged from v1): realised P&L of the
    # smoothed signal book, divided by yesterday's de-risk scale so the
    # history is comparable across de-risked and normal stretches ---------
    if t >= 2 and "_last_smoothed" in _state:
        dprice = prcSoFar[:, -1] - prcSoFar[:, -2]
        pnl = float(_state["_last_smoothed"] @ dprice)
        _state["signal_pnl_hist"].append(pnl / _state["last_scale"])

    hist = _state["trend_pnl_hist"]
    if len(hist) >= 2:
        window = np.array(hist[-dd_window:])
        wealth = np.cumsum(window)
        running_peak = np.maximum.accumulate(np.concatenate([[0.0], wealth]))[1:]
        drawdown = float(max(0.0, running_peak[-1] - wealth[-1]))

        prev_scale = _state["trend_scale"]
        if drawdown >= dd_cap_dollars:
            target_scale = DD_MIN_SCALE
        elif drawdown <= DD_RECOVER_FRAC * dd_cap_dollars:
            target_scale = 1.0
        else:
            frac = (drawdown - DD_RECOVER_FRAC * dd_cap_dollars) / (
                (1 - DD_RECOVER_FRAC) * dd_cap_dollars
            )
            target_scale = 1.0 - frac * (1.0 - DD_MIN_SCALE)

        _state["trend_scale"] = prev_scale + 0.5 * (target_scale - prev_scale)

    trend_dollar = trend_dollar_full * _state["trend_scale"]

    dollar_position = trend_dollar + reversion_dollar

    # ---- global fractional-Kelly haircut (calibrated) ---------------------
    dollar_position = dollar_position * kelly_fraction

    # ---- portfolio-level correlation overlay (Ledoit-Wolf, symmetric vol)
    hist_len = full_log_ret.shape[1]
    if hist_len >= COV_MIN_HISTORY and np.isfinite(TARGET_PORTFOLIO_DAILY_VOL):
        cov_window = full_log_ret[:, -min(hist_len, COV_MAX_LOOKBACK):].T
        Sigma = _ledoit_wolf_shrinkage(cov_window)
        port_var = dollar_position @ Sigma @ dollar_position
        port_vol = np.sqrt(max(port_var, 0.0))
        if port_vol > 1e-9:
            scale = min(1.0, TARGET_PORTFOLIO_DAILY_VOL / port_vol)
            dollar_position = dollar_position * scale

    # ---- empirical CVaR (expected shortfall) cap, whole book (calibrated) -
    if hist_len >= CVAR_MIN_HISTORY:
        window = full_log_ret[:, -min(hist_len, CVAR_LOOKBACK):]
        cvar = _historical_cvar(dollar_position, window)
        if cvar < -cvar_dollar_budget:
            cvar_scale = cvar_dollar_budget / (-cvar)
            dollar_position = dollar_position * cvar_scale

    # ---- hysteretic PnL-feedback de-risker (merged from v1) --------------
    if DERISK_SCALE < 1.0:
        hist = _state["signal_pnl_hist"]
        if len(hist) >= DERISK_WINDOW:
            trailing = float(np.sum(hist[-DERISK_WINDOW:]))
            band = float(np.std(hist)) * np.sqrt(DERISK_WINDOW)
            if band > 1e-9:
                if not _state["derisked"] and trailing < DERISK_ENTER_Z * band:
                    _state["derisked"] = True
                elif _state["derisked"] and trailing > DERISK_EXIT_Z * band:
                    _state["derisked"] = False
        if _state["derisked"]:
            dollar_position = dollar_position * DERISK_SCALE

    _state["last_scale"] = DERISK_SCALE if _state["derisked"] else 1.0

    target_shares = dollar_position / curPrices

    # ---- pairs-trading overlay (unchanged; not drawdown-managed) ---------
    pair_pos = np.zeros(nInst)
    if t > PAIR_Z_LOOKBACK:
        for k, (ia, ib, beta) in enumerate(PAIRS):
            if max(ia, ib) >= nInst:
                continue
            st = _state["pair_state"][k]
            spread = (prcSoFar[ia, -PAIR_Z_LOOKBACK:]
                      - beta * prcSoFar[ib, -PAIR_Z_LOOKBACK:])
            sd = spread.std()
            if sd < 1e-9:
                continue
            z = (spread[-1] - spread.mean()) / sd

            if st["dir"] == 0:
                want = 0
                if z >= PAIR_Z_ENTRY:
                    want = -1
                elif z <= -PAIR_Z_ENTRY:
                    want = +1
                if want != 0:
                    edge = (curPrices[ia] * raw_score[ia]
                            - beta * curPrices[ib] * raw_score[ib])
                    if (not PAIR_SIGNAL_GATE) or want * edge >= 0:
                        sa = int(round(want * PAIR_TRADE_DOLLARS / curPrices[ia]))
                        st["dir"] = want
                        st["shares_a"] = sa
                        st["shares_b"] = int(round(-beta * sa))
            elif st["dir"] * z >= -PAIR_Z_EXIT:
                st["dir"] = 0
                st["shares_a"] = 0
                st["shares_b"] = 0

            pair_pos[ia] += st["shares_a"]
            pair_pos[ib] += st["shares_b"]

    # ---- signal-book smoothing (turnover control) -------------------------
    prev_smoothed = _state.get("_last_smoothed", np.zeros(nInst))
    full_move_dollars = np.abs(target_shares - prev_smoothed) * curPrices
    worth_trading = full_move_dollars >= MIN_TRADE_DOLLARS

    blended_shares = np.where(
        worth_trading,
        SMOOTH_ALPHA * target_shares + (1 - SMOOTH_ALPHA) * prev_smoothed,
        prev_smoothed,
    )
    signal_positions = np.round(blended_shares).astype(int)
    _state["_last_smoothed"] = signal_positions.astype(float)

    pos_limits = (dlr_limits / curPrices).astype(int)
    positions = np.clip(
        signal_positions + pair_pos.astype(int), -pos_limits, pos_limits
    ).astype(int)

    # ---- minimum-activity guarantee ---------------------------------------
    _state["call_count"] += 1
    if (
        _state["call_count"] >= VOLUME_CHECK_CALLS
        and _state["cum_dvolume"] < VOLUME_TARGET
    ):
        churn_shares = int(round(CHURN_DOLLARS / curPrices[0]))
        if churn_shares > 0:
            algo_cap_shares = int(dlr_limits[0] / curPrices[0])
            nudged = positions[0] + _state["churn_sign"] * churn_shares
            if abs(nudged) > algo_cap_shares:
                _state["churn_sign"] *= -1
                nudged = positions[0] + _state["churn_sign"] * churn_shares
            positions[0] = int(np.clip(nudged, -algo_cap_shares, algo_cap_shares))
            _state["churn_sign"] *= -1

    if _state["last_pos"] is None:
        last_pos = np.zeros(nInst)
    else:
        last_pos = _state["last_pos"]
    _state["cum_dvolume"] += float(
        np.sum(np.abs(positions - last_pos) * curPrices)
    )

    _state["last_trend_pos"] = (trend_dollar_full / curPrices)
    _state["last_reversion_pos"] = (reversion_dollar / curPrices)
    _state["last_pos"] = positions.astype(float)
    return positions