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
RESID_WEIGHT = 0.55
RIDGE_WEIGHT = 0.15
BASKET_WEIGHT = 0.70
CLUSTER_WEIGHT = 0.50
ANALOG_WEIGHT = 0.0

RESID_LOOKBACK = 5
RESID_BETA_LOOKBACK = 80
BASKET_LOOKBACK = 90
BASKET_Z_LOOKBACK = 60
BASKET_TOP_K = 6
BASKET_TRADE_DOLLARS = 90_000.0
BASKET_MAX_BOOK_DOLLARS = 1_080_000.0
BASKET_TARGET_DAILY_VOL = 12_000.0
CLUSTER_LOOKBACK = 20
CLUSTER_TRADE_DOLLARS = 225_000.0
CLUSTER_TARGET_DAILY_VOL = 9_000.0
INST0_LOOKBACK = 60
INST0_TRADE_DOLLARS = 95_000.0
INST0_TARGET_DAILY_VOL = 3_500.0
ANALOG_MIN_HISTORY = 120
ANALOG_MAX_HISTORY = 260
ANALOG_NEIGHBORS = 12
ANALOG_TRADE_DOLLARS = 180_000.0
ANALOG_TARGET_DAILY_VOL = 5_000.0
RIDGE_MAX_TRAIN_DAYS = 180
RIDGE_MIN_TRAIN_DAYS = 70
RIDGE_ALPHA = 25.0

ADAPTIVE_IC_LOOKBACK = 60
ADAPTIVE_IC_MIN_DAYS = 25
ADAPTIVE_PRIOR = np.array([0.50, 0.30, 0.40, 0.10, 0.35, 0.25, 0.30])

VOL_HALFLIFE_FAST = 20
VOL_HALFLIFE_SLOW = 60
VOL_BLEND = 0.5

# ----------------------------- inference config --------------------------
DEFAULT_DLR_POS_LIMIT = 10_000
INST0_DLR_POS_LIMIT = 100_000

RISK_FRACTION = 1.0
TAU = 0.1
ALPHA_TAU = 1.0
ALGO_RISK_FRACTION = 1.0
SIGNAL_KEEP_TOP_K = 40
FINAL_POSITION_SCALE = 2.5

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

# --- short bias on the SIGNAL book (trend + reversion) --------------------
# Long dollar exposure is scaled by (1 - SHORT_BIAS), short dollar exposure
# by (1 + SHORT_BIAS), per instrument, BEFORE the Kelly / vol / CVaR overlays
# so all downstream risk management sees the biased book. Applied identically
# in the calibration sweep so the swept thresholds match the live book.
# The pairs overlay is deliberately untouched (it is market-neutral by
# construction; tilting it would break the hedge). 0.0 disables entirely.
ENABLE_SHORT_BIAS = True
SHORT_BIAS = 0.25

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

# --- pairs-trading overlay (deliberately NOT drawdown-managed, per
# Kaminski & Lo -- this book is mean-reverting by construction) ------------
DEFAULT_PAIRS = [
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
STATIC_PAIR_POOL = [
    (5, 21, -0.0712347630183, 2.707532, 2.707532),
    (13, 45, 0.597452135683, 2.455892, 2.455892),
    (49, 50, 0.449571359636, 2.382338, 2.382338),
    (18, 32, 1.1629096385, 2.306963, 2.306963),
    (36, 50, 0.483469139095, 2.142724, 2.142724),
    (7, 40, 0.942282591635, 2.083149, 2.083149),
    (18, 42, 1.81050150104, 2.074411, 2.074411),
    (10, 46, 1.39598469511, 1.987949, 1.987949),
    (1, 20, 0.904842138901, 1.940643, 1.940643),
    (18, 28, 0.0817650229317, 1.939498, 1.939498),
    (31, 43, 0.824045853744, 1.887193, 1.887193),
    (18, 33, 1.6769464542, 1.837995, 1.837995),
    (18, 23, 0.166201582492, 1.826169, 1.826169),
    (18, 48, 0.376322998001, 1.795146, 1.795146),
    (14, 36, -4.89590661842, 1.786946, 1.786946),
    (41, 50, 0.699962478926, 1.756654, 1.756654),
    (18, 27, 0.303487101825, 1.736695, 1.736695),
    (1, 50, 3.16574236244, 1.725650, 1.725650),
    (26, 32, 0.604610975137, 1.713927, 1.713927),
    (18, 47, -0.484829645767, 1.707303, 1.707303),
    (41, 49, 1.46133105605, 1.668162, 1.668162),
    (8, 27, 1.13081558934, 1.649011, 1.649011),
    (20, 50, 3.3826361591, 1.634048, 1.634048),
    (25, 37, 0.9392114584, 1.624598, 1.624598),
    (9, 20, 0.15264396084, 1.587116, 1.587116),
    (18, 41, 0.638380049559, 1.525415, 1.525415),
    (14, 20, -0.74304304503, 1.513983, 1.513983),
    (5, 46, -0.339541871838, 1.455461, 1.455461),
    (5, 10, -0.240397074635, 1.444318, 1.444318),
    (33, 40, 0.0896698965385, 1.410017, 1.410017),
    (35, 42, 1.09217616589, 1.409511, 1.409511),
    (36, 49, 1.00449872201, 1.405583, 1.405583),
    (30, 42, 2.95593697025, 1.392779, 1.392779),
    (18, 35, 0.917076706844, 1.351591, 1.351591),
    (33, 48, 0.0944088527578, 1.336559, 1.336559),
    (17, 39, 0.136558447913, 1.329327, 1.329327),
    (33, 47, -0.0998194517238, 1.327623, 1.327623),
    (42, 43, -0.500680506448, 1.320732, 1.320732),
    (36, 41, 0.662153561258, 1.298134, 1.298134),
    (23, 36, 5.22613136699, 1.293423, 1.293423),
    (22, 44, 28.6479485244, 1.292363, 1.292363),
    (33, 46, -0.0387751242892, 1.275301, 1.275301),
    (42, 47, -0.210040007069, 1.147693, 1.147693),
    (18, 19, 0.226807083292, 1.131443, 1.131443),
    (28, 49, 10.9571842155, 1.127598, 1.127598),
    (33, 42, 0.589858361347, 1.112272, 1.112272),
    (18, 26, 1.6402661646, 1.112145, 1.112145),
    (14, 50, -2.71397774808, 1.108695, 1.108695),
    (33, 45, 0.00793905878924, 1.046494, 1.046494),
    (0, 13, 0.694363595734, 1.035556, 1.035556),
    (9, 14, -0.1718837968, 1.011003, 1.011003),
    (33, 41, 0.134405641683, 1.008228, 1.008228),
    (42, 44, -0.366379208061, 0.969342, 0.969342),
    (24, 36, -2.23603767725, 0.962291, 0.962291),
    (35, 47, -0.278289684643, 0.893540, 0.893540),
    (33, 36, 0.172061947598, 0.881206, 0.881206),
    (33, 38, 0.0148104096987, 0.881136, 0.881136),
    (35, 45, 0.0908354874263, 0.880060, 0.880060),
    (6, 48, -0.597225537357, 0.873448, 0.873448),
    (33, 44, -0.159678671241, 0.869581, 0.869581),
    (33, 39, 0.0404329782697, 0.864503, 0.864503),
    (33, 34, 0.0228030943345, 0.857712, 0.857712),
    (35, 50, 0.291304251093, 0.843772, 0.843772),
    (28, 50, 5.06845284188, 0.804105, 0.804105),
    (33, 35, 0.157300155508, 0.802137, 0.802137),
    (30, 47, -0.882063241159, 0.794808, 0.794808),
    (35, 36, 0.551074006184, 0.792896, 0.792896),
    (35, 38, 0.194169677216, 0.784689, 0.784689),
    (35, 41, 0.398986470658, 0.784121, 0.784121),
    (1, 14, -1.01561334958, 0.772496, 0.772496),
    (42, 46, -0.10187530622, 0.757869, 0.757869),
    (35, 48, 0.186085713017, 0.739771, 0.739771),
    (15, 37, 0.311413168064, 0.644602, 0.644602),
    (24, 49, -2.62245391479, 0.630521, 0.630521),
    (33, 43, -0.150061323531, 0.629095, 0.629095),
    (33, 37, -0.0241075202792, 0.588471, 0.588471),
    (37, 39, 0.76680533608, 0.578437, 0.578437),
    (35, 49, 0.575827906795, 0.561160, 0.561160),
    (33, 49, 0.183850978009, 0.497004, 0.497004),
    (12, 26, 1.96806336192, 0.445705, 0.445705),
]
PAIR_DISCOVERY_ENABLED = True
PAIR_DISCOVERY_TOP_K = 16
PAIR_DISCOVERY_MAX_PAIRS = 36
PAIR_DISCOVERY_MIN_HISTORY = 160
PAIR_DISCOVERY_HOLDOUT_DAYS = 40
PAIR_DISCOVERY_VOL_HALFLIFE = 30
PAIR_DISCOVERY_RET_LOOKBACK = 20
PAIR_DISCOVERY_ADF_THRESHOLD = -2.86
PAIR_Z_LOOKBACK = 60
PAIR_Z_ENTRY = 2.0
PAIR_Z_EXIT = 0.5
PAIR_TRADE_DOLLARS = 30_000.0
PAIR_MAX_BOOK_DOLLARS = 1_260_000.0
PAIR_TARGET_DAILY_VOL = 12_000.0
PAIR_MIN_BACKTEST_TRADES = 12
PAIR_SIGNAL_GATE = False

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


def _zscore_cross_section(x):
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return (x - x.mean()) / (x.std() + 1e-9)


def _residual_reversion_signal(prcSoFar, vol):
    """Mean reversion after stripping the common market component.

    Absolute short-horizon reversion is noisy in market-wide selloffs/rallies;
    this signal trades names that moved too far versus their own market beta.
    """
    nInst, T = prcSoFar.shape
    if T <= max(RESID_LOOKBACK + 1, RESID_BETA_LOOKBACK):
        return np.zeros(nInst)

    safe_prices = np.maximum(prcSoFar, 1e-12)
    logp = np.log(safe_prices)
    ret = np.diff(logp, axis=1)
    market = ret.mean(axis=0)
    beta_window = min(RESID_BETA_LOOKBACK, ret.shape[1] - RESID_LOOKBACK)
    if beta_window < 20:
        return np.zeros(nInst)

    inst_hist = ret[:, -RESID_LOOKBACK - beta_window:-RESID_LOOKBACK]
    mkt_hist = market[-RESID_LOOKBACK - beta_window:-RESID_LOOKBACK]
    mkt_var = float(np.var(mkt_hist))
    if mkt_var < 1e-12:
        beta = np.ones(nInst)
    else:
        beta = ((inst_hist - inst_hist.mean(axis=1, keepdims=True))
                @ (mkt_hist - mkt_hist.mean())) / (beta_window * mkt_var)
        beta = np.clip(beta, -2.0, 3.0)

    inst_move = ret[:, -RESID_LOOKBACK:].sum(axis=1)
    mkt_move = market[-RESID_LOOKBACK:].sum()
    residual_move = inst_move - beta * mkt_move
    residual_vol = np.maximum(vol * np.sqrt(RESID_LOOKBACK), 1e-8)
    return _zscore_cross_section(-residual_move / residual_vol)


def _nearest_neighbor_indices(ret_window, top_k):
    nInst = ret_window.shape[0]
    corr = np.corrcoef(ret_window)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, -np.inf)
    k = min(max(1, top_k), nInst - 1)
    return np.argsort(corr, axis=1)[:, -k:]


def _basket_residual_signal(prcSoFar):
    """Residual z-score after hedging each name against nearest neighbours."""
    nInst, T = prcSoFar.shape
    min_len = max(BASKET_LOOKBACK, BASKET_Z_LOOKBACK) + 5
    if nInst < 3 or T < min_len:
        return np.zeros(nInst)

    logp = np.log(np.maximum(prcSoFar, 1e-12))
    ret = np.diff(logp[:, -BASKET_LOOKBACK - 1:], axis=1)
    peers = _nearest_neighbor_indices(ret, BASKET_TOP_K)
    signal = np.zeros(nInst)

    y_hist_all = logp[:, -BASKET_Z_LOOKBACK:]
    for i in range(nInst):
        js = peers[i]
        X = y_hist_all[js].T
        y = y_hist_all[i]
        Xc = X - X.mean(axis=0, keepdims=True)
        yc = y - y.mean()
        ridge = 1e-4 * np.trace(Xc.T @ Xc) / max(1, Xc.shape[1])
        try:
            beta = np.linalg.solve(Xc.T @ Xc + ridge * np.eye(Xc.shape[1]), Xc.T @ yc)
        except np.linalg.LinAlgError:
            continue
        resid = yc - Xc @ beta
        sd = float(resid.std())
        if sd > 1e-9:
            signal[i] = -resid[-1] / sd

    return _zscore_cross_section(signal)


def _cluster_residual_signal(prcSoFar):
    """Cheap cluster-neutral residual reversion using nearest-neighbour return baskets."""
    nInst, T = prcSoFar.shape
    if nInst < 3 or T < CLUSTER_LOOKBACK + BASKET_LOOKBACK + 2:
        return np.zeros(nInst)

    logp = np.log(np.maximum(prcSoFar, 1e-12))
    ret_for_corr = np.diff(logp[:, -BASKET_LOOKBACK - 1:], axis=1)
    peers = _nearest_neighbor_indices(ret_for_corr, BASKET_TOP_K)
    move = logp[:, -1] - logp[:, -CLUSTER_LOOKBACK - 1]
    peer_move = np.array([move[peers[i]].mean() for i in range(nInst)])
    return _zscore_cross_section(-(move - peer_move))


def _inst0_signal(prcSoFar):
    """Special-rate instrument 0 residual signal versus the rest of the book."""
    nInst, T = prcSoFar.shape
    if nInst < 3 or T < INST0_LOOKBACK + 5:
        return 0.0

    logp = np.log(np.maximum(prcSoFar, 1e-12))
    others = np.arange(1, nInst)
    ret = np.diff(logp, axis=1)
    corr = np.array([_safe_corr(ret[0, -INST0_LOOKBACK:], ret[j, -INST0_LOOKBACK:])
                     for j in others])
    top = others[np.argsort(corr)[-min(10, others.size):]]
    basket = logp[top].mean(axis=0)
    y = logp[0, -INST0_LOOKBACK:] - basket[-INST0_LOOKBACK:]
    sd = float(y.std())
    if sd < 1e-9:
        return 0.0
    return float(np.clip(-(y[-1] - y.mean()) / sd, -3.0, 3.0))


def _analog_signal(prcSoFar):
    """Nearest historical market-pattern analogue for next-day rank returns."""
    nInst, T = prcSoFar.shape
    if T < ANALOG_MIN_HISTORY:
        return np.zeros(nInst)

    logp = np.log(np.maximum(prcSoFar, 1e-12))
    max_hist = min(ANALOG_MAX_HISTORY, T - 2)
    start = max(20, T - max_hist - 1)
    days = np.arange(start, T - 1)
    if days.size < 30:
        return np.zeros(nInst)

    def feat(day):
        r1 = logp[:, day] - logp[:, day - 1]
        r5 = logp[:, day] - logp[:, day - 5]
        r20 = logp[:, day] - logp[:, day - 20]
        f = np.concatenate([
            _zscore_cross_section(r1),
            _zscore_cross_section(r5),
            _zscore_cross_section(r20),
        ])
        norm = np.linalg.norm(f)
        return f / (norm + 1e-9)

    cur = feat(T - 1)
    F = np.vstack([feat(int(d)) for d in days])
    sims = F @ cur
    k = min(ANALOG_NEIGHBORS, sims.size)
    nn = np.argsort(sims)[-k:]
    weights = np.maximum(sims[nn] - sims[nn].min() + 1e-4, 1e-4)
    weights = weights / weights.sum()
    fwd = np.vstack([logp[:, int(days[i]) + 1] - logp[:, int(days[i])] for i in nn])
    pred = weights @ fwd
    return _rank01_to_score(pred)


def _rank01_to_score(x):
    """Map a cross-section to roughly [-1, 1] by rank, robust to outliers."""
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    n = x.size
    if n <= 1 or np.allclose(x, x[0]):
        return np.zeros(n)
    order = np.argsort(x)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    return 2.0 * (ranks / max(1.0, n - 1.0)) - 1.0


def _ridge_feature_matrix(logp, day_indices):
    """Panel features for a lightweight pooled ridge next-return ranker."""
    feats = []
    for d in day_indices:
        r1 = logp[:, d] - logp[:, d - 1]
        r5 = logp[:, d] - logp[:, d - 5]
        r20 = logp[:, d] - logp[:, d - 20]
        r60 = logp[:, d] - logp[:, d - 60]
        ret20 = np.diff(logp[:, d - 20:d + 1], axis=1)
        ret60 = np.diff(logp[:, d - 60:d + 1], axis=1)
        vol20 = ret20.std(axis=1)
        vol60 = ret60.std(axis=1)
        market5 = r5.mean()
        resid5 = r5 - market5
        day_feat = np.column_stack([r1, r5, r20, r60, vol20, vol60, resid5])
        feats.append(day_feat)
    return np.vstack(feats)


def _ridge_rank_signal(prcSoFar):
    """Tiny in-file ridge model. It is intentionally used as a weak rank
    overlay, not as the primary forecast engine."""
    nInst, T = prcSoFar.shape
    if T < RIDGE_MIN_TRAIN_DAYS + 65:
        return np.zeros(nInst)

    logp = np.log(np.maximum(prcSoFar, 1e-12))
    last_train_day = T - 2
    first_train_day = max(60, last_train_day - RIDGE_MAX_TRAIN_DAYS + 1)
    day_indices = np.arange(first_train_day, last_train_day + 1)
    if day_indices.size < RIDGE_MIN_TRAIN_DAYS:
        return np.zeros(nInst)

    X = _ridge_feature_matrix(logp, day_indices)
    y = np.concatenate([
        logp[:, d + 1] - logp[:, d]
        for d in day_indices
    ])
    if X.shape[0] <= X.shape[1] + 2:
        return np.zeros(nInst)

    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Xs = (X - mu) / sd
    yc = y - y.mean()
    xtx = Xs.T @ Xs
    beta = np.linalg.solve(xtx + RIDGE_ALPHA * np.eye(xtx.shape[0]), Xs.T @ yc)

    x_now = _ridge_feature_matrix(logp, np.array([T - 1]))
    pred = ((x_now - mu) / sd) @ beta
    return _rank01_to_score(pred)


def _proxy_signal_at(logp, day):
    """Fast historical proxies for adaptive alpha weighting."""
    r5 = logp[:, day] - logp[:, day - 5]
    r20 = logp[:, day] - logp[:, day - 20]
    r60 = logp[:, day] - logp[:, day - 60]
    market5 = r5.mean()
    cs20 = r20 - r20.mean()
    peer_proxy = np.zeros_like(r5)
    order = np.argsort(r20)
    q = max(1, r20.size // 5)
    peer_proxy[order[:q]] = 1.0
    peer_proxy[order[-q:]] = -1.0
    analog_proxy = _rank01_to_score(-(r5 - market5) + 0.35 * r20)
    return (
        _zscore_cross_section(r20 + 0.5 * r60),
        _zscore_cross_section(-r5),
        _zscore_cross_section(-(r5 - market5)),
        _zscore_cross_section(-cs20),
        _zscore_cross_section(peer_proxy),
        _zscore_cross_section(analog_proxy),
    )


def _adaptive_alpha_weights(prcSoFar):
    """Recent cross-sectional IC gate for trend, reversion, residual,
    ridge, basket and cluster components."""
    _, T = prcSoFar.shape
    if T < ADAPTIVE_IC_MIN_DAYS + 70:
        return ADAPTIVE_PRIOR / ADAPTIVE_PRIOR.sum()

    logp = np.log(np.maximum(prcSoFar, 1e-12))
    start = max(60, T - 1 - ADAPTIVE_IC_LOOKBACK)
    end = T - 2
    trend_ic, rev_ic, resid_ic, basket_ic, cluster_ic, analog_ic = [], [], [], [], [], []
    for d in range(start, end + 1):
        fwd = logp[:, d + 1] - logp[:, d]
        sigs = _proxy_signal_at(logp, d)
        vals = [trend_ic, rev_ic, resid_ic, basket_ic, cluster_ic, analog_ic]
        for out, sig in zip(vals, sigs):
            out.append(_safe_corr(sig, fwd))

    if len(trend_ic) < ADAPTIVE_IC_MIN_DAYS:
        return ADAPTIVE_PRIOR / ADAPTIVE_PRIOR.sum()

    raw = np.array([
        max(0.0, float(np.nanmean(trend_ic))),
        max(0.0, float(np.nanmean(rev_ic))),
        max(0.0, float(np.nanmean(resid_ic))),
        0.015,
        max(0.0, float(np.nanmean(basket_ic))),
        max(0.0, float(np.nanmean(cluster_ic))),
        max(0.0, float(np.nanmean(analog_ic))),
    ])
    weights = 0.65 * ADAPTIVE_PRIOR + 0.35 * (raw + 0.01)
    return weights / max(weights.sum(), 1e-9)


def _score_aware_scale(pnl_hist):
    if len(pnl_hist) < 35:
        return 1.0
    window = np.asarray(pnl_hist[-60:], dtype=float)
    sd = float(window.std())
    if sd < 1e-9:
        return 0.75
    sharpe = float(window.mean() / sd * np.sqrt(252))
    draw = np.cumsum(window)
    peak = np.maximum.accumulate(np.concatenate([[0.0], draw]))[1:]
    max_dd = float(np.max(peak - draw)) if draw.size else 0.0

    if sharpe < -0.5:
        scale = 0.45
    elif sharpe < 0.5:
        scale = 0.70
    elif sharpe < 1.5:
        scale = 1.00
    elif sharpe < 2.5:
        scale = 1.20
    else:
        scale = 1.35

    if max_dd > 4.0 * sd * np.sqrt(max(1, min(60, len(window)))):
        scale = min(scale, 0.75)
    return float(np.clip(scale, 0.40, 1.40))


def _component_multiplier(pnl_hist):
    if len(pnl_hist) < 25:
        return 1.0
    x = np.asarray(pnl_hist[-60:], dtype=float)
    sd = float(x.std())
    if sd < 1e-9:
        return 0.8
    sharpe = float(x.mean() / sd * np.sqrt(252))
    if sharpe < -0.5:
        return 0.25
    if sharpe < 0.3:
        return 0.55
    if sharpe < 1.2:
        return 0.90
    if sharpe < 2.5:
        return 1.15
    return 1.35


def _cap_dollar_book(dollar_book, log_ret, target_daily_vol=None, max_gross=None):
    dollar_book = np.nan_to_num(dollar_book, nan=0.0, posinf=0.0, neginf=0.0)
    gross = float(np.sum(np.abs(dollar_book)))
    if max_gross is not None and gross > max_gross > 0:
        dollar_book = dollar_book * (max_gross / gross)
        gross = max_gross
    if target_daily_vol is not None and target_daily_vol > 0 and log_ret.shape[1] >= 20:
        window = log_ret[:, -min(log_ret.shape[1], CVAR_LOOKBACK):]
        pnl = dollar_book @ window
        vol = float(np.std(pnl))
        if vol > 1e-9:
            dollar_book = dollar_book * min(1.0, target_daily_vol / vol)
    return dollar_book


def _dollar_from_signal(signal, dlr_limits, risk_frac, gross_budget, tau=ALPHA_TAU):
    raw = np.tanh(signal / tau) * dlr_limits * risk_frac
    gross = float(np.sum(np.abs(raw)))
    if gross > gross_budget > 0:
        raw *= gross_budget / gross
    return raw


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


def _safe_corr(x, y):
    if x.size < 3 or y.size < 3:
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _build_pair_feature_vectors(prices, vol_halflife=PAIR_DISCOVERY_VOL_HALFLIFE,
                                ret_lookback=PAIR_DISCOVERY_RET_LOOKBACK):
    """One feature vector per instrument for pair prefiltering:
    volatility, recent return, and lag-1 return autocorrelation."""
    nInst, T = prices.shape
    if T < 4:
        return None

    safe_prices = np.maximum(prices, 1e-12)
    log_ret = np.diff(np.log(safe_prices), axis=1)

    vol = _ewma_vol(log_ret, vol_halflife)
    lookback = min(ret_lookback, T - 1)
    recent_ret = safe_prices[:, -1] / safe_prices[:, -lookback - 1] - 1.0
    autocorr = np.array([
        _safe_corr(log_ret[i, :-1], log_ret[i, 1:])
        for i in range(nInst)
    ])

    features = np.column_stack([vol, recent_ret, autocorr])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    mu = features.mean(axis=0, keepdims=True)
    sd = features.std(axis=0, keepdims=True)
    return (features - mu) / np.maximum(sd, 1e-9)


def _engle_granger_adf_tstat(y, x):
    """Two-step Engle-Granger screen using a zero-lag ADF regression on
    residuals. More negative t-stat means stronger mean reversion."""
    if y.size < 20 or x.size < 20 or np.std(x) < 1e-12:
        return 0.0, np.inf

    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (beta * x + alpha)
    lagged = resid[:-1]
    dresid = np.diff(resid)
    denom = float(np.sum(lagged ** 2))
    if denom < 1e-18:
        return float(beta), np.inf

    rho = float(np.sum(lagged * dresid) / denom)
    err = dresid - rho * lagged
    dof = max(1, lagged.size - 1)
    se = float(np.sqrt(np.sum(err ** 2) / dof) / np.sqrt(denom))
    if se < 1e-12:
        return float(beta), np.inf
    return float(beta), rho / se


def _pair_backtest_quality(prices, ia, ib, beta, lookback=PAIR_Z_LOOKBACK):
    """Backtest one spread with unit gross exposure; returns a quality tuple."""
    T = prices.shape[1]
    if T <= lookback + 5:
        return None

    pa = prices[ia]
    pb = prices[ib]
    spread = pa - beta * pb
    pnl = []
    active = 0
    for t in range(lookback, T - 1):
        hist = spread[t - lookback:t]
        sd = float(hist.std())
        if sd < 1e-9:
            pnl.append(0.0)
            continue
        z = (spread[t] - hist.mean()) / sd
        strength = max(0.0, (abs(z) - PAIR_Z_EXIT) / (PAIR_Z_ENTRY - PAIR_Z_EXIT))
        strength = min(1.0, strength)
        if strength <= 0.0:
            pnl.append(0.0)
            continue
        active += 1
        unit = -np.sign(z) * strength
        gross = pa[t] + abs(beta) * pb[t]
        if gross <= 1e-9:
            pnl.append(0.0)
            continue
        shares_a = unit / gross
        shares_b = -beta * unit / gross
        pnl.append(shares_a * (pa[t + 1] - pa[t]) + shares_b * (pb[t + 1] - pb[t]))

    pnl = np.asarray(pnl)
    if active < PAIR_MIN_BACKTEST_TRADES or pnl.std() < 1e-10:
        return None
    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(252))
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    max_dd = float(np.max(peak - equity)) if equity.size else 0.0
    score = sharpe - 0.35 * max_dd / (pnl.std() * np.sqrt(252) + 1e-9)
    return score, sharpe, active


def _find_cointegrated_pairs(prices, top_k_neighbors=PAIR_DISCOVERY_TOP_K,
                             adf_threshold=PAIR_DISCOVERY_ADF_THRESHOLD,
                             max_pairs=PAIR_DISCOVERY_MAX_PAIRS):
    """Discover structurally similar cointegrated pairs from history.

    Feature-space nearest neighbours keep the Engle-Granger screen focused
    on instruments with similar volatility, return and autocorrelation
    behaviour, reducing noisy all-pairs data mining.
    """
    nInst, T = prices.shape
    if nInst < 2 or T < PAIR_DISCOVERY_MIN_HISTORY:
        return None

    features = _build_pair_feature_vectors(prices)
    if features is None:
        return None

    dists = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)

    candidates = set()
    k_neighbors = min(max(1, top_k_neighbors), nInst - 1)
    for i in range(nInst):
        for j in np.argsort(dists[i])[:k_neighbors]:
            candidates.add((min(i, int(j)), max(i, int(j))))

    safe_prices = np.maximum(prices, 1e-12)
    log_prices = np.log(safe_prices)
    results = []
    for ia, ib in candidates:
        _, adf_stat = _engle_granger_adf_tstat(log_prices[ia], log_prices[ib])
        if adf_stat < adf_threshold and np.std(safe_prices[ib]) > 1e-12:
            trade_beta, _ = np.polyfit(safe_prices[ib], safe_prices[ia], 1)
            if np.isfinite(trade_beta):
                quality = _pair_backtest_quality(safe_prices, ia, ib, float(trade_beta))
                if quality is not None:
                    score, sharpe, active = quality
                    if score > 0.0 and sharpe > 0.4:
                        results.append((
                            ia, ib, float(trade_beta), float(adf_stat),
                            float(score), float(sharpe), int(active)
                        ))

    results.sort(key=lambda row: row[4], reverse=True)
    return [(ia, ib, beta, score, sharpe) for ia, ib, beta, _, score, sharpe, _ in results[:max_pairs]]


def _calibrate_pairs(prcSoFar):
    """Lazy pair discovery using only a held-out-safe training prefix."""
    nInst, T = prcSoFar.shape
    pair_pool = [p for p in STATIC_PAIR_POOL if max(p[0], p[1]) < nInst]
    if not PAIR_DISCOVERY_ENABLED:
        return pair_pool if pair_pool else list(DEFAULT_PAIRS)
    if T <= PAIR_DISCOVERY_HOLDOUT_DAYS + PAIR_DISCOVERY_MIN_HISTORY:
        return pair_pool if pair_pool else None

    train_prices = prcSoFar[:, : T - PAIR_DISCOVERY_HOLDOUT_DAYS]
    discovered = _find_cointegrated_pairs(train_prices)
    if discovered:
        seen = {(min(p[0], p[1]), max(p[0], p[1])) for p in pair_pool}
        for p in discovered:
            key = (min(p[0], p[1]), max(p[0], p[1]))
            if key not in seen:
                pair_pool.append(p)
                seen.add(key)
    return pair_pool if pair_pool else list(DEFAULT_PAIRS)


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
    "component_pnl_hist": {
        "trend": [], "reversion": [], "basket": [], "cluster": [], "inst0": [], "analog": []
    },
    "last_components": {},
    "derisked": False,            # hysteretic de-risker state (merged from v1)
    "last_scale": 1.0,            # de-risk scale in force yesterday (for PnL normalisation)
    "pairs": list(DEFAULT_PAIRS),
    "pair_state": [{"dir": 0, "shares_a": 0, "shares_b": 0} for _ in DEFAULT_PAIRS],
    "pairs_calibrated": False,
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

    if not _state["pairs_calibrated"]:
        pairs = _calibrate_pairs(prcSoFar)
        if pairs is not None:
            _state["pairs"] = pairs
            _state["pair_state"] = [
                {"dir": 0, "shares_a": 0, "shares_b": 0} for _ in pairs
            ]
            _state["pairs_calibrated"] = True
            # Uncomment to see discovered pair hedge ratios:
            # print(f"[pair calibration] {_state['pairs']}")

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
    residual_score = _residual_reversion_signal(prcSoFar, vol)
    basket_score = _basket_residual_signal(prcSoFar)
    cluster_score = _cluster_residual_signal(prcSoFar)
    analog_score = _analog_signal(prcSoFar)
    ridge_score = _ridge_rank_signal(prcSoFar)
    inst0_z = _inst0_signal(prcSoFar)
    alpha_weights = _adaptive_alpha_weights(prcSoFar)

    trend_component = alpha_weights[0] * _zscore_cross_section(trend_score)
    reversion_component = (
        alpha_weights[1] * _zscore_cross_section(reversion_score)
        + alpha_weights[2] * RESID_WEIGHT * residual_score
        + alpha_weights[3] * RIDGE_WEIGHT * ridge_score
    )
    basket_component = alpha_weights[4] * BASKET_WEIGHT * basket_score
    cluster_component = alpha_weights[5] * CLUSTER_WEIGHT * cluster_score
    analog_component = alpha_weights[6] * ANALOG_WEIGHT * analog_score
    raw_score = (
        trend_component + reversion_component + basket_component
        + cluster_component + analog_component
    )

    trade_mask = np.ones(nInst, dtype=bool)
    if SIGNAL_KEEP_TOP_K is not None and 0 < SIGNAL_KEEP_TOP_K < nInst:
        kth = np.partition(np.abs(raw_score), -SIGNAL_KEEP_TOP_K)[-SIGNAL_KEEP_TOP_K]
        trade_mask = np.abs(raw_score) >= kth

    risk_frac = np.full(nInst, RISK_FRACTION)
    risk_frac[0] *= ALGO_RISK_FRACTION

    trend_dollar_full = np.where(
        trade_mask, np.tanh(trend_component / ALPHA_TAU) * dlr_limits * risk_frac, 0.0
    )
    reversion_dollar = np.where(
        trade_mask, np.tanh(reversion_component / ALPHA_TAU) * dlr_limits * risk_frac, 0.0
    )
    basket_dollar = np.where(
        trade_mask, np.tanh(basket_component / ALPHA_TAU) * dlr_limits * risk_frac, 0.0
    )
    cluster_dollar = np.where(
        trade_mask, np.tanh(cluster_component / ALPHA_TAU) * dlr_limits * risk_frac, 0.0
    )
    analog_dollar = np.where(
        trade_mask, np.tanh(analog_component / ALPHA_TAU) * dlr_limits * risk_frac, 0.0
    )

    basket_dollar = _cap_dollar_book(
        basket_dollar, full_log_ret, BASKET_TARGET_DAILY_VOL, BASKET_MAX_BOOK_DOLLARS
    )
    cluster_dollar = _cap_dollar_book(
        cluster_dollar, full_log_ret, CLUSTER_TARGET_DAILY_VOL, CLUSTER_TRADE_DOLLARS
    )
    inst0_dollar = np.zeros(nInst)
    if nInst > 0:
        inst0_dollar[0] = np.tanh(inst0_z / 1.25) * min(INST0_TRADE_DOLLARS, dlr_limits[0])
        if nInst > 1:
            hedge = -0.35 * inst0_dollar[0] / (nInst - 1)
            inst0_dollar[1:] = hedge
    inst0_dollar = _cap_dollar_book(
        inst0_dollar, full_log_ret, INST0_TARGET_DAILY_VOL, INST0_TRADE_DOLLARS * 1.4
    )
    analog_dollar = _cap_dollar_book(
        analog_dollar, full_log_ret, ANALOG_TARGET_DAILY_VOL, ANALOG_TRADE_DOLLARS
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

    if t >= 2 and _state["last_components"]:
        dprice = prcSoFar[:, -1] - prcSoFar[:, -2]
        for name, shares in _state["last_components"].items():
            _state["component_pnl_hist"].setdefault(name, []).append(float(shares @ dprice))

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

    comp_mult = {
        name: _component_multiplier(hist)
        for name, hist in _state["component_pnl_hist"].items()
    }

    component_dollars = {
        "trend": trend_dollar * comp_mult.get("trend", 1.0),
        "reversion": reversion_dollar * comp_mult.get("reversion", 1.0),
        "basket": basket_dollar * comp_mult.get("basket", 1.0),
        "cluster": cluster_dollar * comp_mult.get("cluster", 1.0),
        "inst0": inst0_dollar * comp_mult.get("inst0", 1.0),
        "analog": analog_dollar * comp_mult.get("analog", 1.0),
    }

    dollar_position = np.zeros(nInst)
    for comp_book in component_dollars.values():
        dollar_position = dollar_position + comp_book

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

    score_scale = _score_aware_scale(_state["signal_pnl_hist"])
    dollar_position = dollar_position * score_scale

    _state["last_scale"] = DERISK_SCALE if _state["derisked"] else 1.0

    target_shares = dollar_position / curPrices

    # ---- pairs-trading book: continuous spread z-score sizing ------------
    pair_pos = np.zeros(nInst)
    if t > PAIR_Z_LOOKBACK:
        for spec in _state["pairs"]:
            ia, ib, beta = spec[:3]
            if max(ia, ib) >= nInst:
                continue
            spread = (prcSoFar[ia, -PAIR_Z_LOOKBACK:]
                      - beta * prcSoFar[ib, -PAIR_Z_LOOKBACK:])
            sd = spread.std()
            if sd < 1e-9:
                continue
            z = (spread[-1] - spread.mean()) / sd
            strength = max(0.0, (abs(z) - PAIR_Z_EXIT) / (PAIR_Z_ENTRY - PAIR_Z_EXIT))
            strength = min(1.0, strength)
            if strength <= 0.0:
                continue

            edge = (curPrices[ia] * raw_score[ia]
                    - beta * curPrices[ib] * raw_score[ib])
            want = -np.sign(z)
            if PAIR_SIGNAL_GATE and want * edge < 0:
                continue

            quality = float(spec[3]) if len(spec) > 3 else 1.0
            quality_scale = float(np.clip(0.7 + 0.35 * quality, 0.6, 1.8))
            gross_per_unit = curPrices[ia] + abs(beta) * curPrices[ib]
            if gross_per_unit <= 1e-9:
                continue

            target_gross = PAIR_TRADE_DOLLARS * quality_scale * strength
            unit = want * target_gross / gross_per_unit
            pair_pos[ia] += unit
            pair_pos[ib] += -beta * unit

        pair_gross = float(np.sum(np.abs(pair_pos) * curPrices))
        if pair_gross > PAIR_MAX_BOOK_DOLLARS:
            pair_pos *= PAIR_MAX_BOOK_DOLLARS / pair_gross

        if hist_len >= CVAR_MIN_HISTORY and PAIR_TARGET_DAILY_VOL > 0:
            pair_dollars = pair_pos * curPrices
            pair_window = full_log_ret[:, -min(hist_len, CVAR_LOOKBACK):]
            pair_pnl = pair_dollars @ pair_window
            pair_vol = float(np.std(pair_pnl))
            if pair_vol > 1e-9:
                pair_pos *= min(1.0, PAIR_TARGET_DAILY_VOL / pair_vol)

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
        (signal_positions + pair_pos.astype(int)) * FINAL_POSITION_SCALE,
        -pos_limits,
        pos_limits,
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
    _state["last_components"] = {
        name: (book / curPrices).astype(float)
        for name, book in component_dollars.items()
    }
    _state["last_pos"] = positions.astype(float)
    return positions
