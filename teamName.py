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
once more history has accumulated.

WALK-FORWARD RECALIBRATION (regime adaptation)
-----------------------------------------------
The sweep no longer runs "once and for good" -- it reruns every
RECALIBRATION_INTERVAL_DAYS on the same expanding-window / fixed-holdout-gap
basis. A post-mortem on an earlier version found the four thresholds fit
once on the first ~400 days of a 750-day run and never adapted, while the
market's trend-followability genuinely broke down around day ~600 --
Kelly/DD/CVaR stayed as loose as the (differently-behaved) training period
warranted, for the entire back half of the run. Periodic re-sweeps let
those thresholds tighten (or loosen) as the regime actually evolves,
without hand-fitting a "regime detector" to this specific price history.

ADAPTIVE ALPHA WEIGHTS: SIGNED IC, NOT FLOORED AT ZERO
--------------------------------------------------------
_adaptive_alpha_weights previously floored each signal's trailing IC at
0.0 before blending 65% prior / 35% live, so a signal with a persistently
NEGATIVE IC could shrink towards (but never below) ~65% of its static
prior share -- it could never be actively suppressed, only fail to grow.
It now uses the signed IC and a 40% prior / 60% live blend (floored at a
small epsilon, not zero, so no signal auto-flips sign): a component whose
edge has genuinely broken down gets pulled well below its no-skill
baseline instead of bleeding at a quarter-strength indefinitely.

TREND GETS ITS OWN FASTER RISK THROTTLE
------------------------------------------
Trend carries the single largest hard-coded prior weight of any signal
(ADAPTIVE_PRIOR[0] = 0.50) and, in the post-mortem, was the one component
that went net-negative in EVERY 50-day block of a 250-day test window
during the regime break. It's now throttled by its own
_trend_component_multiplier: a shorter 30-day (vs. 60-day) trailing-Sharpe
lookback and a much lower floor (0.05x vs. 0.25x shared by every other
component), so a genuine trend-following regime break gets shut down in
weeks, not months, layered on top of (not instead of) the existing
drawdown-based trend_scale.

These three changes were validated with a walk-forward harness across
four disjoint ~150-day regimes spanning the full price history (not just
the most recent test window) plus the official last-250-day eval.py
block, specifically to avoid curve-fitting a fix to one regime at the
expense of the others.

PAIRS ENGINE ADAPTATION (second-round changes, same harness discipline)
------------------------------------------------------------------------
Book-level attribution showed the pairs book is the dominant PnL engine
in every regime, and it had the same one-shot-calibration defect as the
risk thresholds. Changes, each regime-agnostic by construction:

  * Pairs are RE-DISCOVERED on the same walk-forward cadence as the risk
    thresholds (RECALIBRATION_INTERVAL_DAYS), refreshing hedge ratios and
    pool membership as relationships drift.
  * The hardcoded STATIC_PAIR_POOL is RE-VALIDATED at each recalibration
    against the same quality gates as discovered pairs, with quality
    scored on the most recent PAIR_QUALITY_WINDOW days -- stale pairs are
    dropped or resized instead of trusted forever.
  * PER-PAIR PnL THROTTLE (_pair_multiplier): a pair whose spread has
    stopped mean-reverting is shrunk based on its own trailing realised
    PnL, mirroring the per-component multipliers.
  * RISK-PARITY SIZING across pairs: each pair's gross is tilted by
    (median spread vol / its spread vol), clipped to [0.5, 2.0], so
    high-vol spreads stop dominating the book's risk. Relative scaling
    only -- no new dollar constant.
  * MULTI-WINDOW Z: the entry z-score is the average over 40/60/90-day
    windows (diversification across time-scales rather than one fragile
    fixed window).

INST0 DIRECTIONAL BOOK REMOVED (INST0_TRADE_DOLLARS = 0): per-component
attribution showed the instrument-0 basket-reversion signal was a
zero-edge coin flip with large variance (e.g. +4.2k then -4.6k in
adjacent 50-day blocks); removing it improved EVERY tested regime
simultaneously. Instrument 0 still participates in the pairs book, the
cross-sectional signals and the minimum-activity churn.

Validation snapshot (walk-forward harness, score = eval.py formula):
days 250-400: 504 | 400-550: 379 | 500-650: 323 | 600-750: 304 |
official 500-750: 327 (baseline before all changes: 501/427/171/141/123).

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
INST0_TRADE_DOLLARS = 0.0     # inst0 directional book disabled (see docstring)
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
FINAL_POSITION_SCALE = 3.5

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
# Final-position deadband: skip any per-instrument position CHANGE smaller
# than this many dollars (holds the previous submitted position instead).
# Applied after the final clip, on the whole book — unlike MIN_TRADE_DOLLARS
# which only gates the signal-book smoothing. 0.0 disables.
TURNOVER_DEADBAND_DOLLARS = 0.0

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
# 2.25 (was 2.0): validated across 100/250/500/700-day walk-forward windows
# (scores 157/321/400/447 vs 139/327/393/428 at 2.0). Tighter entries help
# most where the pairs edge has decayed (last ~200 days); the entry-threshold
# response curve is smooth from 1.75-2.5, so this is not a knife-edge fit.
PAIR_Z_ENTRY = 2.25
PAIR_Z_EXIT = 0.5
PAIR_TRADE_DOLLARS = 45_000.0
PAIR_MAX_BOOK_DOLLARS = 1_260_000.0
PAIR_TARGET_DAILY_VOL = 16_000.0
PAIR_MIN_BACKTEST_TRADES = 12
PAIR_QUALITY_WINDOW = 250     # score pair quality on this many recent days
PAIR_SIGNAL_GATE = False

# --- OU half-life SEPARATE pairs book (screened 2026-07-17) ---------------
# ou_pairs was the strongest standalone screen (positive in every window,
# incl. +97 in 550-650 where the live pairs book makes ~nothing). OU sizing
# INSIDE the live pairs loop was rejected (interferes with risk-parity +
# multi-window z); this is the separate-book variant. 0.0 disables.
# ADOPTED 2026-07-17 (3rd session): 1.5k vol + OU_PAPER_GATE + OU_HEADROOM_FIT.
# Suite 4704.2 -> 4756.7, ALL windows pass (worst -3.8%). 2k vol left
# 700-750 at -5.3% (borderline); headroom-fit is what made adoption
# possible at all (see HEDGE/OU notes below).
OU_TARGET_DAILY_VOL = 1_500.0
OU_TRADE_DOLLARS = 40_000.0
OU_FIT_WINDOW = 120
OU_MIN_HALFLIFE = 2.0
OU_MAX_HALFLIFE = 40.0
OU_MAX_BOOK_DOLLARS = 800_000.0
OU_Z_TAU = 1.5
# autocorr gate: scale the OU book by clip(1 - ac/OU_AC_MAX, 0, 1) where ac
# is the trailing market lag-1 autocorrelation — full size in MR regimes,
# zero once the market turns persistently trending. False disables.
OU_AC_GATE = False
OU_AC_WINDOW = 60
OU_AC_MAX = 0.03
# per-pair paper-PnL gate: simulate this pair's OU rule over the trailing
# OU_PAPER_WINDOW days (one fit, vectorized); if the paper PnL is negative
# the pair trades at OU_PAPER_FLOOR size. Self-referential and regime-
# agnostic (reacts to each spread's own current behavior). False disables.
OU_PAPER_GATE = True      # adopted with the OU book (see above)
OU_PAPER_WINDOW = 30
OU_PAPER_FLOOR = 0.25
# Kalman-filter dynamic hedge ratio: instead of the calibrated pair's fixed
# beta, run a scalar Kalman recursion (on window-demeaned prices, so the
# spread's equilibrium level doesn't bias the ratio) across OU_FIT_WINDOW
# and build the spread with the time-varying beta. Attacks hedge-ratio
# drift mid-window (the 600-700 cointegration-decay regime). Delta is the
# state-noise-to-observation-noise ratio; 0.0 disables (static beta).
OU_KALMAN_DELTA = 0.0

# --- hedge-preserving position limiter ------------------------------------
# Diagnosis 2026-07-17: ~37/50 instruments sit ON the $10k limit daily in
# 600-700 — DELIBERATELY: FINAL_POSITION_SCALE over-scales so the clip pins
# most names at max size; that saturation IS the sizing mechanism. Scaling
# pair units down to fit under limits (HEDGE_LIMIT_PRESERVE) guts the book:
# suite 4704 -> 2787 REJECTED (vol −35%, but mean falls far more; only the
# quiet 700-750 window improved). Kept for reference/testing only.
HEDGE_LIMIT_PRESERVE = False
HEDGE_LIMIT_ITERS = 4
# OU_HEADROOM_FIT: the OU book must not perturb the pinned main book (on
# saturated names a small OU contribution flips WHICH side the name pins
# to — "stealing" allocation instead of overlaying). When True, OU pair
# units are jointly scaled to fit the RESIDUAL per-instrument headroom
# left by signal+pairs+sm+i0, and the main book is untouched.
OU_HEADROOM_FIT = True    # adopted with the OU book (see above)

# --- spread-momentum book on pair spreads (candidate) ---------------------
# When cointegration weakens (days 500-700: lag-1 autocorr flips positive)
# pair spreads TREND; EWMAC on the spread itself earns exactly when the
# reversion book's edge decays. 0.0 disables.
SPREADMOM_TARGET_DAILY_VOL = 0.0
SPREADMOM_TRADE_DOLLARS = 40_000.0
SPREADMOM_SPANS = [(8, 24), (16, 48)]
SPREADMOM_HIST = 150
SPREADMOM_MAX_BOOK_DOLLARS = 800_000.0
SPREADMOM_ZBAND = 0.0   # >0: only trade a spread while its |z| < this band
# Fit spread-mom pair units into residual headroom (same mechanism as
# OU_HEADROOM_FIT) instead of adding pre-clip. Harmless while the book is
# off; the pre-clip adder variant was REJECTED (allocation stealing).
SPREADMOM_HEADROOM_FIT = True

# --- dispersion-conditioned sizing of the reversion-family sleeve ---------
# Cross-sectional reversion/pairs PnL is mechanically proportional to
# dispersion; in quiet low-dispersion stretches (days 700-750) expected
# edge per trade falls below commission. Continuous multiplier on the
# reversion-family books only (NOT a signal switch). False disables.
DISP_SCALE_ENABLED = False
DISP_WINDOW = 20
DISP_PCT_WINDOW = 150
DISP_LO = 0.6
DISP_HI = 1.4
DISP_APPLY_PAIRS = False   # also scale the pairs book by the same multiplier

# --- stalled-divergence sizing gate on live pair entries ------------------
# Only take full pair size once the spread has STOPPED widening (last-3-day
# spread move against the divergence or small); avoids catching knives in
# the 500-700 trending-spread regime. 1.0 disables (no gate).
PAIR_STALL_FACTOR = 1.0
PAIR_STALL_DAYS = 3
PAIR_STALL_TOL = 0.25

# --- inst0 AR(1) market-timing book ---------------------------------------
# inst0 is a near-perfect index proxy (corr 0.986 with the equal-weight
# market) with 5x cheaper commission and a 10x position limit: the only
# place a small market-timing edge is monetizable. Signal: trailing
# lag-1-autocorr times today's market return (adaptive AR(1) — fades the
# market in MR regimes, follows it in momentum regimes). Measured
# sign-edge 11-14bp/day in days 500-750, ~1-6bp earlier. 0.0 disables.
INST0_AR_DOLLARS = 0.0        # FINAL dollar size (post FINAL_POSITION_SCALE)
INST0_AR_WINDOW = 60
INST0_AR_TAU = 0.3            # tanh(pred / (tau * market vol)): ~sign but smooth
INST0_AR_MIN_AC = 0.0         # dead-zone: no trade unless |trailing ac| >= this
# Clip the inst0 AR book into the residual headroom on instrument 0 (the
# single-instrument analogue of OU_HEADROOM_FIT): never perturbs which side
# the main book pins inst0 to. Pre-clip adder REJECTED (700-750 −8-9%).
INST0_AR_HEADROOM_FIT = True

# --- Donchian channel breakout book ----------------------------------------
# Classic single-lookback channel breakout, per instrument: persistent state
# s in {-1, 0, +1}. Channel over days t-N..t-1 (EXCLUDES today). Enter long
# on close >= N-day high, short on close <= N-day low; exit to FLAT when the
# close crosses the mid-channel (H+L)/2; opposite band reverses directly.
# Orthogonal to EWMAC by being all-or-nothing WITH a genuine flat state:
# in chop/quiet regimes the book exits to zero instead of bleeding.
# Sized risk-parity (1/vol, clipped), vol-capped like the OU book (same
# pre-FINAL_POSITION_SCALE convention as OU_TARGET_DAILY_VOL), and clipped
# LAST into residual per-instrument headroom: never perturbs the pinned
# main book. Instrument 0 (index, $100k limit) is EXCLUDED: index trend =
# market timing, a repeatedly-failed family here.
# ADOPTED 2026-07-18 (session 5) at N=35 / 500: suite 4819.4 -> 5709.4
# (+18.5%), 700-750 +87% -> 472.5, last100 +63%, worst window 500-600
# -2.8%; 6/7 disjoint wins. N ridge 30-60 all >= +13%; N=35 is the ridge
# center AND the only all-windows-rule-clean value (30 fails 500-600
# -10.9%, 40 fails 100-200 -5.3%). Size hump mapped at N=50: 250/375/500/
# 750/1000/1500 -> 500 is the center. Jitter-robust late gains (+11-63%
# on shifted boundaries); soft spot 550-650 -7% (bounded).
DONCH_N = 35
DONCH_TARGET_DAILY_VOL = 500.0   # 0 disables the book entirely
DONCH_TRADE_DOLLARS = 40_000.0

# --- trend-agreement gate on the 5-day reversion signal -------------------
# Full reversion size only when fading a move WITHIN the direction of the
# slow trend (buy dips in uptrends); reduced size when fighting it.
# 1.0 disables. ADOPTED 2026-07-17 at 0.7: suite 4509.0 -> 4622.6,
# 700-750 +34%, worst window -2.0% (0.5 was rejected: last100 -15%).
REV_TREND_GATE = 0.7

# --- residual momentum component book -------------------------------------
# Beta-hedged medium-horizon momentum (cumulative residual return over
# RESIDMOM_LOOKBACK days, skipping the last RESIDMOM_SKIP to stay out of
# the 5d reversion book's way). Captures the 500-700 momentum regime in
# hedged space without the market exposure that killed raw long-horizon
# momentum mid-sample. 0.0 disables.
RESIDMOM_TARGET_DAILY_VOL = 0.0
RESIDMOM_TRADE_DOLLARS = 75_000.0
RESIDMOM_LOOKBACK = 40
RESIDMOM_SKIP = 5
RESIDMOM_BETA_LOOKBACK = 80

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


# --- lead-lag cross-prediction book (screened 2026-07-17) -----------------
LEADLAG_REFIT_DAYS = 5
LEADLAG_WINDOW = 150
LEADLAG_MIN_ABS = 0.15
# ADOPTED 2026-07-17 (3rd): 2k vol / 100k gross (was 1.5k/75k). Under
# REV_TREND_GATE=0.7 the suite went 4622.6 -> 4704.2 with ALL windows
# passing (worst -1.9%; 700-750 +14%, last100 +10%). NOTE the gross cap
# binds before the vol target: scale BOTH together or the change is a no-op.
LEADLAG_TRADE_DOLLARS = 150_000.0
LEADLAG_TARGET_DAILY_VOL = 3_000.0

_leadlag_cache = {"t": -999, "L": None}

# --- echo network: cross-instrument predictability at lags 2..5 -----------
# The adopted lead-lag book stops at lag 1; if information diffuses across
# the cointegration network over multiple days, later hops are unclaimed.
# Same construction per lag k: E_k[i,j] = corr(ret_i(t), ret_j(t+k)),
# thresholded, prediction = sum_k E_k.T @ z(t-k+1). 0.0 disables.
ECHO_TARGET_DAILY_VOL = 0.0
ECHO_TRADE_DOLLARS = 75_000.0
ECHO_LAGS = (2, 3, 4, 5)
ECHO_WINDOW = 150
ECHO_MIN_ABS = 0.15
ECHO_REFIT_DAYS = 5

_echo_cache = {"t": -999, "Ls": None}


def _echo_signal(prcSoFar, vol):
    nInst, T = prcSoFar.shape
    if T < ECHO_WINDOW + max(ECHO_LAGS) + 2:
        return np.zeros(nInst)
    ret = np.diff(np.log(np.maximum(prcSoFar, 1e-12)), axis=1)
    cache = _echo_cache
    if (cache["Ls"] is None or cache["Ls"][0].shape[0] != nInst
            or T - cache["t"] >= ECHO_REFIT_DAYS):
        Ls = []
        for k in ECHO_LAGS:
            R = ret[:, -(ECHO_WINDOW + k):]
            Rs = (R - R.mean(axis=1, keepdims=True)) / (R.std(axis=1, keepdims=True) + 1e-12)
            L = Rs[:, :-k] @ Rs[:, k:].T / (ECHO_WINDOW - 1)
            np.fill_diagonal(L, 0.0)
            L[np.abs(L) < ECHO_MIN_ABS] = 0.0
            Ls.append(L)
        cache["t"] = T
        cache["Ls"] = Ls
    pred = np.zeros(nInst)
    for L, k in zip(cache["Ls"], ECHO_LAGS):
        z_day = np.clip(ret[:, -k] / np.maximum(vol, 1e-8), -3.0, 3.0)
        pred += L.T @ z_day
    return _zscore_cross_section(pred)


def _leadlag_signal(prcSoFar, vol):
    """Cross-instrument lag-1 predictability: L[i, j] = corr(ret_i today,
    ret_j tomorrow) over a trailing window; today's standardized returns
    propagated through the thresholded network predict tomorrow's moves."""
    nInst, T = prcSoFar.shape
    if T < LEADLAG_WINDOW + 2:
        return np.zeros(nInst)
    ret = np.diff(np.log(np.maximum(prcSoFar, 1e-12)), axis=1)
    cache = _leadlag_cache
    if (cache["L"] is None or cache["L"].shape[0] != nInst
            or T - cache["t"] >= LEADLAG_REFIT_DAYS):
        R = ret[:, -LEADLAG_WINDOW:]
        Rs = (R - R.mean(axis=1, keepdims=True)) / (R.std(axis=1, keepdims=True) + 1e-12)
        L = Rs[:, :-1] @ Rs[:, 1:].T / (LEADLAG_WINDOW - 1)
        np.fill_diagonal(L, 0.0)
        L[np.abs(L) < LEADLAG_MIN_ABS] = 0.0
        cache["t"] = T
        cache["L"] = L
    z_today = np.clip(ret[:, -1] / np.maximum(vol, 1e-8), -3.0, 3.0)
    return _zscore_cross_section(cache["L"].T @ z_today)


def _residual_momentum_signal(prcSoFar, vol):
    """Cross-sectional momentum on beta-hedged cumulative returns, skipping
    the most recent RESIDMOM_SKIP days (left to the reversion book)."""
    nInst, T = prcSoFar.shape
    need = RESIDMOM_LOOKBACK + RESIDMOM_SKIP + RESIDMOM_BETA_LOOKBACK + 5
    if T < need:
        return np.zeros(nInst)
    logp = np.log(np.maximum(prcSoFar, 1e-12))
    ret = np.diff(logp, axis=1)
    market = ret.mean(axis=0)
    inst_hist = ret[:, -RESIDMOM_BETA_LOOKBACK:]
    mkt_hist = market[-RESIDMOM_BETA_LOOKBACK:]
    mkt_var = float(np.var(mkt_hist))
    if mkt_var < 1e-12:
        beta = np.ones(nInst)
    else:
        beta = ((inst_hist - inst_hist.mean(axis=1, keepdims=True))
                @ (mkt_hist - mkt_hist.mean())) / (RESIDMOM_BETA_LOOKBACK * mkt_var)
        beta = np.clip(beta, -2.0, 3.0)
    end = -1 - RESIDMOM_SKIP
    start = end - RESIDMOM_LOOKBACK
    move = logp[:, end] - logp[:, start]
    mkt_move = float(np.sum(market[start + 1: end + 1 if end + 1 < 0 else None]))
    resid = move - beta * mkt_move
    denom = np.maximum(vol * np.sqrt(RESIDMOM_LOOKBACK), 1e-8)
    return _zscore_cross_section(resid / denom)


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
        float(np.nanmean(trend_ic)),
        float(np.nanmean(rev_ic)),
        float(np.nanmean(resid_ic)),
        0.015,
        float(np.nanmean(basket_ic)),
        float(np.nanmean(cluster_ic)),
        float(np.nanmean(analog_ic)),
    ])
    raw = np.nan_to_num(raw, nan=0.0)
    # SIGNED IC (no floor-at-zero): a signal with a persistently negative
    # trailing IC now actively pulls its own weight down below the
    # no-skill baseline, instead of only ever failing to add to it. The
    # live/prior blend is shifted towards the live signal (0.40/0.60
    # instead of 0.65/0.35) so a sustained regime break can actually move
    # the needle, floored at a small epsilon so no signal auto-flips sign.
    weights = 0.40 * ADAPTIVE_PRIOR + 0.60 * np.clip(raw + 0.01, -0.5, None)
    eps = 0.02 * ADAPTIVE_PRIOR.sum() / ADAPTIVE_PRIOR.size
    weights = np.clip(weights, eps, None)
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


# Trend gets its OWN, faster/steeper throttle: shorter lookback (reacts in
# ~15-30 days instead of 25-60) and a much lower floor (0.05x instead of
# 0.25x). Trend carries the single largest hard-coded prior weight in the
# whole book, so when it enters a real regime break it should be able to
# shrink to near-zero quickly instead of bleeding at a quarter size for
# months (see the 500-750 day post-mortem: trend was negative in every
# 50-day block of that window and this multiplier never got the chance
# to react fast enough with the old, shared 60-day/0.25-floor curve).
# Per-pair PnL feedback: a pair whose spread has stopped mean-reverting
# (persistently losing) is shrunk, using the same trailing-Sharpe idea as
# the component multipliers. Regime-agnostic: it reacts to each pair's own
# realised PnL, not to any property fitted to a specific price window.
def _pair_multiplier(pnl_hist):
    if len(pnl_hist) < 15:
        return 1.0
    x = np.asarray(pnl_hist[-40:], dtype=float)
    sd = float(x.std())
    if sd < 1e-9:
        return 1.0
    sharpe = float(x.mean() / sd * np.sqrt(252))
    if sharpe < -0.5:
        return 0.20
    if sharpe < 0.0:
        return 0.55
    if sharpe < 1.5:
        return 1.00
    return 1.25


def _trend_component_multiplier(pnl_hist):
    if len(pnl_hist) < 15:
        return 1.0
    x = np.asarray(pnl_hist[-30:], dtype=float)
    sd = float(x.std())
    if sd < 1e-9:
        return 0.8
    sharpe = float(x.mean() / sd * np.sqrt(252))
    if sharpe < -0.3:
        return 0.05
    if sharpe < 0.2:
        return 0.30
    if sharpe < 1.0:
        return 0.85
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
                quality = _pair_backtest_quality(
                    safe_prices[:, -PAIR_QUALITY_WINDOW:], ia, ib, float(trade_beta)
                )
                if quality is not None:
                    score, sharpe, active = quality
                    if score > 0.0 and sharpe > 0.4:
                        results.append((
                            ia, ib, float(trade_beta), float(adf_stat),
                            float(score), float(sharpe), int(active)
                        ))

    results.sort(key=lambda row: row[4], reverse=True)
    return [(ia, ib, beta, score, sharpe) for ia, ib, beta, _, score, sharpe, _ in results[:max_pairs]]


def _ou_pair_book(prcSoFar, pairs, full_log_ret, hist_len):
    """Separate OU (Ornstein-Uhlenbeck) pairs book: fit an AR(1) to each
    calibrated pair's spread, trade only spreads with a credible
    mean-reversion half-life, size continuously by z from the OU stationary
    distribution and tilt by reversion SPEED (fast half-life = bigger).
    Deliberately independent of the live pairs loop's entry/exit + throttles.
    Returns (aggregate shares, per-pair unit map {(ia,ib): (sh_a, sh_b)})."""
    nInst, t = prcSoFar.shape
    pos = np.zeros(nInst)
    units = {}
    if OU_TARGET_DAILY_VOL <= 0 or t <= OU_FIT_WINDOW + 2:
        return pos, units
    curPrices = prcSoFar[:, -1]
    for spec in pairs:
        ia, ib, beta = spec[0], spec[1], spec[2]
        if max(ia, ib) >= nInst:
            continue
        if OU_KALMAN_DELTA > 0:
            pa = prcSoFar[ia, -OU_FIT_WINDOW:]
            pb = prcSoFar[ib, -OU_FIT_WINDOW:]
            pac = pa - pa.mean()
            pbc = pb - pb.mean()
            R = max(1e-8, float(np.var(np.diff(pac - beta * pbc))))
            Q = OU_KALMAN_DELTA * R
            kb = beta
            P = R
            betas = np.empty(pa.size)
            for i in range(pa.size):
                P += Q
                S = P * pbc[i] * pbc[i] + R
                K = P * pbc[i] / S
                kb += K * (pac[i] - kb * pbc[i])
                P *= 1.0 - K * pbc[i]
                betas[i] = kb
            s = pa - betas * pb
            beta = float(kb)
        else:
            s = prcSoFar[ia, -OU_FIT_WINDOW:] - beta * prcSoFar[ib, -OU_FIT_WINDOW:]
        ds = np.diff(s)
        s_lag = s[:-1]
        var = float(np.var(s_lag))
        if var < 1e-12:
            continue
        b = float(np.mean((s_lag - s_lag.mean()) * (ds - ds.mean())) / var)
        a = float(ds.mean() - b * s_lag.mean())
        if b >= -1e-4 or b <= -1.5:
            continue
        kappa = -np.log(max(1e-9, 1.0 + b))
        halflife = np.log(2.0) / kappa
        if not (OU_MIN_HALFLIFE <= halflife <= OU_MAX_HALFLIFE):
            continue
        denom = 1.0 - (1.0 + b) ** 2
        if denom <= 1e-9:
            continue
        resid = ds - (a + b * s_lag)
        sig_stat = float(np.sqrt(np.var(resid) / denom))
        if sig_stat < 1e-9:
            continue
        z = (s[-1] - (-a / b)) / sig_stat
        gross_per_unit = curPrices[ia] + abs(beta) * curPrices[ib]
        if gross_per_unit <= 1e-9:
            continue
        # speed tilt: 1.0 at a 10-day half-life, faster reversion sized up
        speed_w = float(np.clip(np.sqrt(10.0 / halflife), 0.5, 1.6))
        if OU_PAPER_GATE and s.size > OU_PAPER_WINDOW + 1:
            tail = s[-(OU_PAPER_WINDOW + 1):]
            z_tail = (tail[:-1] - (-a / b)) / sig_stat
            paper = float(np.sum(-np.tanh(z_tail / OU_Z_TAU) * np.diff(tail)))
            if paper < 0.0:
                speed_w *= OU_PAPER_FLOOR
        unit = -np.tanh(z / OU_Z_TAU) * OU_TRADE_DOLLARS * speed_w / gross_per_unit
        pos[ia] += unit
        pos[ib] += -beta * unit
        key = (min(ia, ib), max(ia, ib))
        prev = units.get(key, (0.0, 0.0))
        if ia < ib:
            units[key] = (prev[0] + unit, prev[1] - beta * unit)
        else:
            units[key] = (prev[0] - beta * unit, prev[1] + unit)
    gross = float(np.sum(np.abs(pos) * curPrices))
    scale = 1.0
    if gross > OU_MAX_BOOK_DOLLARS > 0:
        scale *= OU_MAX_BOOK_DOLLARS / gross
    if hist_len >= CVAR_MIN_HISTORY:
        dollars = pos * curPrices * scale
        window = full_log_ret[:, -min(hist_len, CVAR_LOOKBACK):]
        v = float(np.std(dollars @ window))
        if v > 1e-9:
            scale *= min(1.0, OU_TARGET_DAILY_VOL / v)
    pos *= scale
    units = {k: (a * scale, b * scale) for k, (a, b) in units.items()}
    return pos, units


def _hedge_preserving_limit(base_shares, unit_maps, limit_shares):
    """Jointly scale down pair units (BOTH legs) wherever the combined book
    would breach an instrument's position limit, so the final per-instrument
    clip no longer amputates one leg of a hedged spread. Mutates the unit
    maps in place; residual violations are left to the plain clip."""
    n = base_shares.shape[0]
    for _ in range(HEDGE_LIMIT_ITERS):
        overlay = np.zeros(n)
        for m in unit_maps:
            for (ia, ib), (sa, sb) in m.items():
                overlay[ia] += sa
                overlay[ib] += sb
        excess = np.abs(base_shares + overlay) - limit_shares
        if not np.any(excess > 1e-9):
            break
        f_inst = np.ones(n)
        over = excess > 1e-9
        headroom = np.maximum(limit_shares[over] - np.abs(base_shares[over]), 0.0)
        f_inst[over] = np.clip(headroom / (np.abs(overlay[over]) + 1e-9), 0.0, 1.0)
        for m in unit_maps:
            for key in list(m.keys()):
                f = min(f_inst[key[0]], f_inst[key[1]])
                if f < 1.0:
                    sa, sb = m[key]
                    m[key] = (sa * f, sb * f)
    aggregates = []
    for m in unit_maps:
        agg = np.zeros(n)
        for (ia, ib), (sa, sb) in m.items():
            agg[ia] += sa
            agg[ib] += sb
        aggregates.append(agg)
    return aggregates


def _spread_mom_book(prcSoFar, pairs, full_log_ret, hist_len):
    """EWMAC momentum on each calibrated pair's SPREAD (not the legs).
    Complements the reversion books: profits when a spread breaks from its
    equilibrium and trends, which is when the z-reversion book bleeds.
    Returns (aggregate shares, per-pair unit map {(ia,ib): (sh_a, sh_b)})."""
    nInst, t = prcSoFar.shape
    pos = np.zeros(nInst)
    units = {}
    if SPREADMOM_TARGET_DAILY_VOL <= 0 or t <= SPREADMOM_HIST + 2:
        return pos, units
    curPrices = prcSoFar[:, -1]
    for spec in pairs:
        ia, ib, beta = spec[0], spec[1], spec[2]
        if max(ia, ib) >= nInst:
            continue
        s = prcSoFar[ia, -SPREADMOM_HIST:] - beta * prcSoFar[ib, -SPREADMOM_HIST:]
        sv = float(np.std(np.diff(s)))
        if sv < 1e-9:
            continue
        if SPREADMOM_ZBAND > 0:
            tail = s[-PAIR_Z_LOOKBACK:]
            sd_l = float(tail.std())
            if sd_l < 1e-9:
                continue
            z_l = (tail[-1] - tail.mean()) / sd_l
            if abs(z_l) >= SPREADMOM_ZBAND:
                continue
        raw = 0.0
        for fast, slow in SPREADMOM_SPANS:
            af = 1.0 - 0.5 ** (1.0 / fast)
            asl = 1.0 - 0.5 ** (1.0 / slow)
            ef = s[0]
            es = s[0]
            for x in s[1:]:
                ef = af * x + (1 - af) * ef
                es = asl * x + (1 - asl) * es
            raw += (ef - es) / (sv * np.sqrt(slow))
        raw /= len(SPREADMOM_SPANS)
        gross_per_unit = curPrices[ia] + abs(beta) * curPrices[ib]
        if gross_per_unit <= 1e-9:
            continue
        unit = np.tanh(raw) * SPREADMOM_TRADE_DOLLARS / gross_per_unit
        pos[ia] += unit
        pos[ib] += -beta * unit
        key = (min(ia, ib), max(ia, ib))
        prev = units.get(key, (0.0, 0.0))
        if ia < ib:
            units[key] = (prev[0] + unit, prev[1] - beta * unit)
        else:
            units[key] = (prev[0] - beta * unit, prev[1] + unit)
    gross = float(np.sum(np.abs(pos) * curPrices))
    scale = 1.0
    if gross > SPREADMOM_MAX_BOOK_DOLLARS > 0:
        scale *= SPREADMOM_MAX_BOOK_DOLLARS / gross
    if hist_len >= CVAR_MIN_HISTORY:
        dollars = pos * curPrices * scale
        window = full_log_ret[:, -min(hist_len, CVAR_LOOKBACK):]
        v = float(np.std(dollars @ window))
        if v > 1e-9:
            scale *= min(1.0, SPREADMOM_TARGET_DAILY_VOL / v)
    pos *= scale
    units = {k: (a * scale, b * scale) for k, (a, b) in units.items()}
    return pos, units


def _donch_book(prcSoFar, vol, full_log_ret):
    """Donchian channel breakout book (see the DONCH_N constants block).

    Updates the persistent per-instrument state vector in _state (all-zero
    init; warms up naturally through the WARMUP_DAYS replay) and returns
    pre-FINAL_POSITION_SCALE shares — same sizing convention as the OU
    book: DONCH_TARGET_DAILY_VOL is measured on pre-scale dollars."""
    nInst, t = prcSoFar.shape
    state = _state.get("donch_state")
    if state is None or state.shape[0] != nInst:
        state = np.zeros(nInst)
    if t < DONCH_N + 1:
        _state["donch_state"] = state
        return np.zeros(nInst)
    close = prcSoFar[:, -1]
    chan = prcSoFar[:, -(DONCH_N + 1):-1]   # days t-N..t-1: EXCLUDES today
    hi = chan.max(axis=1)
    lo = chan.min(axis=1)
    mid = 0.5 * (hi + lo)
    new_state = state.copy()
    # exits first (mid-channel cross), then band touches enter/reverse
    new_state[(state > 0) & (close < mid)] = 0.0
    new_state[(state < 0) & (close > mid)] = 0.0
    new_state[close >= hi] = 1.0
    new_state[close <= lo] = -1.0
    new_state[0] = 0.0        # index excluded (see constants block)
    _state["donch_state"] = new_state
    if not np.any(new_state):
        return np.zeros(nInst)
    # risk-parity weights around the median vol, clipped to a fixed 2:1
    # band so no single low-vol name dominates (deliberately not a constant)
    safe_vol = np.maximum(vol, 1e-6)
    w = np.clip(np.median(safe_vol) / safe_vol, 0.5, 2.0)
    dollars = new_state * DONCH_TRADE_DOLLARS * w
    dollars = _cap_dollar_book(
        dollars, full_log_ret, target_daily_vol=DONCH_TARGET_DAILY_VOL
    )
    return dollars / np.maximum(close, 1e-9)


def _revalidate_static_pool(train_prices, nInst):
    """Re-score the hardcoded STATIC_PAIR_POOL against CURRENT training
    data, using the same quality gates as freshly discovered pairs. The
    hardcoded quality numbers were fitted on the earliest history; they
    directly set position sizes (quality_scale), so trusting them forever
    means sizing stale spreads as if they still worked."""
    # quality is scored on the most recent QUALITY_WINDOW days only: a
    # pair's position size should reflect whether it works NOW, not in
    # the distant past.
    safe_prices = np.maximum(train_prices[:, -PAIR_QUALITY_WINDOW:], 1e-12)
    out = []
    for p in STATIC_PAIR_POOL:
        ia, ib, beta = p[0], p[1], p[2]
        if max(ia, ib) >= nInst:
            continue
        quality = _pair_backtest_quality(safe_prices, ia, ib, float(beta))
        if quality is None:
            continue
        score, sharpe, _ = quality
        if score > 0.0 and sharpe > 0.4:
            out.append((ia, ib, float(beta), float(score), float(sharpe)))
    return out


def _calibrate_pairs(prcSoFar):
    """Lazy pair discovery using only a held-out-safe training prefix."""
    nInst, T = prcSoFar.shape
    pair_pool = [p for p in STATIC_PAIR_POOL if max(p[0], p[1]) < nInst]
    if not PAIR_DISCOVERY_ENABLED:
        return pair_pool if pair_pool else list(DEFAULT_PAIRS)
    if T <= PAIR_DISCOVERY_HOLDOUT_DAYS + PAIR_DISCOVERY_MIN_HISTORY:
        return pair_pool if pair_pool else None

    train_prices = prcSoFar[:, : T - PAIR_DISCOVERY_HOLDOUT_DAYS]
    pair_pool = _revalidate_static_pool(train_prices, nInst)
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
        "trend": [], "reversion": [], "basket": [], "cluster": [], "inst0": [], "analog": [], "leadlag": []
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
    "last_calib_day": None,       # walk-forward recalibration bookkeeping
    "pair_pnl_hist": {},          # per-pair daily PnL history, keyed (ia, ib)
    "last_pair_units": {},        # per-pair (shares_a, shares_b) from yesterday
    "donch_state": None,          # Donchian per-instrument state in {-1,0,+1}
}

# Recalibrate the four swept risk thresholds periodically (walk-forward,
# expanding training window, same SWEEP_HOLDOUT_DAYS gap) instead of once
# at the very first call. This lets Kelly/DD/CVaR react as new regimes
# accumulate in the price history, instead of freezing forever on
# whatever the earliest ~400 days happened to look like.
RECALIBRATION_INTERVAL_DAYS = 100

# Cold-start warm-up: on the very first call, replay the strategy over the
# trailing WARMUP_DAYS of already-available history so every PnL-feedback
# throttle (component multipliers, score-aware scale, de-risker, trend
# drawdown scale) starts informed instead of at its uninformed 1.0 default.
# 60 days is the longest throttle lookback, not a fitted constant. The
# per-pair throttles are deliberately left cold (see below).
WARMUP_DAYS = 60


def getMyPosition(prcSoFar):
    nInst, t = prcSoFar.shape

    if not _state.get("warmed_up", False):
        _state["warmed_up"] = True
        start = max(20, t - WARMUP_DAYS)
        if t >= 80 and start < t:
            for d in range(start, t):
                getMyPosition(prcSoFar[:, :d])
            # first live call should recalibrate on the freshest data, not
            # on what was available WARMUP_DAYS ago
            if _state["calibrated"]:
                _state["last_calib_day"] = t - RECALIBRATION_INTERVAL_DAYS
            # warm-up trades were simulated, not sent: reset the
            # minimum-activity bookkeeping and the position ledger
            _state["call_count"] = 0
            _state["cum_dvolume"] = 0.0
            _state["last_pos"] = None
            # keep the per-pair throttles COLD: in a mean-reverting regime a
            # recently-losing pair is at its widest spread, and pre-throttling
            # it cuts size exactly when the opportunity is largest (validated:
            # warm pair histories cost ~23% on the day-300-400 window)
            _state["pair_pnl_hist"] = {}

    due_for_recalib = (
        not _state["calibrated"]
        or (t - _state["last_calib_day"]) >= RECALIBRATION_INTERVAL_DAYS
    )
    if due_for_recalib:
        result = _calibrate_thresholds(prcSoFar)
        if result is not None:
            _state["params"].update(
                {k: v for k, v in result.items() if not k.startswith("_")}
            )
            _state["calibrated"] = True
            _state["last_calib_day"] = t
            # Uncomment to see what the sweep picked, and on how much data:
            # print(f"[calibration] day={t} {result}")

    # Pairs are re-discovered on the same walk-forward cadence as the risk
    # thresholds: hedge ratios and cointegration relationships drift, so a
    # pool fitted once on the earliest history goes stale in later regimes.
    if not _state["pairs_calibrated"] or due_for_recalib:
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
    if REV_TREND_GATE < 1.0 and t >= MIN_TREND_HISTORY:
        fast_span, slow_span = TREND_SPANS[-1]
        slow_trend = (_ema(prcSoFar, fast_span)[:, -1]
                      - _ema(prcSoFar, slow_span)[:, -1])
        fighting = np.sign(reversion_score) != np.sign(slow_trend)
        reversion_score = np.where(
            fighting, reversion_score * REV_TREND_GATE, reversion_score
        )
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

    # ---- dispersion-conditioned multiplier for the reversion-family sleeve
    disp_mult = 1.0
    if DISP_SCALE_ENABLED and full_log_ret.shape[1] >= DISP_PCT_WINDOW + DISP_WINDOW:
        disp_series = full_log_ret[:, -(DISP_PCT_WINDOW + DISP_WINDOW):].std(axis=0)
        disp_now = float(disp_series[-DISP_WINDOW:].mean())
        hist_means = np.convolve(
            disp_series, np.ones(DISP_WINDOW) / DISP_WINDOW, mode="valid"
        )
        pct = float((hist_means <= disp_now).mean())
        disp_mult = DISP_LO + (DISP_HI - DISP_LO) * pct
        reversion_dollar = reversion_dollar * disp_mult
        cluster_dollar = cluster_dollar * disp_mult

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
    leadlag_score = _leadlag_signal(prcSoFar, vol)
    leadlag_dollar = np.tanh(0.35 * leadlag_score) * dlr_limits * risk_frac
    leadlag_dollar = _cap_dollar_book(
        leadlag_dollar, full_log_ret, LEADLAG_TARGET_DAILY_VOL, LEADLAG_TRADE_DOLLARS
    )
    echo_dollar = np.zeros(nInst)
    if ECHO_TARGET_DAILY_VOL > 0:
        echo_score = _echo_signal(prcSoFar, vol)
        echo_dollar = np.tanh(0.35 * echo_score) * dlr_limits * risk_frac
        echo_dollar = _cap_dollar_book(
            echo_dollar, full_log_ret, ECHO_TARGET_DAILY_VOL, ECHO_TRADE_DOLLARS
        )
    residmom_dollar = np.zeros(nInst)
    if RESIDMOM_TARGET_DAILY_VOL > 0:
        residmom_score = _residual_momentum_signal(prcSoFar, vol)
        residmom_dollar = np.tanh(0.35 * residmom_score) * dlr_limits * risk_frac
        residmom_dollar = _cap_dollar_book(
            residmom_dollar, full_log_ret,
            RESIDMOM_TARGET_DAILY_VOL, RESIDMOM_TRADE_DOLLARS
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
    comp_mult["trend"] = _trend_component_multiplier(
        _state["component_pnl_hist"].get("trend", [])
    )

    component_dollars = {
        "trend": trend_dollar * comp_mult.get("trend", 1.0),
        "reversion": reversion_dollar * comp_mult.get("reversion", 1.0),
        "basket": basket_dollar * comp_mult.get("basket", 1.0),
        "cluster": cluster_dollar * comp_mult.get("cluster", 1.0),
        "inst0": inst0_dollar * comp_mult.get("inst0", 1.0),
        "analog": analog_dollar * comp_mult.get("analog", 1.0),
        "leadlag": leadlag_dollar * comp_mult.get("leadlag", 1.0),
        "residmom": residmom_dollar * comp_mult.get("residmom", 1.0),
        "echo": echo_dollar * comp_mult.get("echo", 1.0),
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

    # ---- per-pair PnL bookkeeping (for the pair throttle) -----------------
    if t >= 2 and _state["last_pair_units"]:
        dprice = prcSoFar[:, -1] - prcSoFar[:, -2]
        for key, (sh_a, sh_b) in _state["last_pair_units"].items():
            ia, ib = key
            if max(ia, ib) >= nInst:
                continue
            pnl = sh_a * dprice[ia] + sh_b * dprice[ib]
            _state["pair_pnl_hist"].setdefault(key, []).append(float(pnl))

    # ---- pairs-trading book: continuous spread z-score sizing ------------
    # Two passes: first collect each active pair's desired trade and its
    # realised spread volatility, then size with a RELATIVE risk-parity
    # tilt -- equal-gross sizing lets high-vol spreads dominate the book's
    # risk while low-vol spreads contribute nothing; scaling each pair by
    # (median spread vol / its spread vol), clipped to [0.5, 2.0], evens
    # the risk contributions without introducing any new dollar constant.
    pair_pos = np.zeros(nInst)
    new_pair_units = {}
    pair_cap_scale = 1.0
    if t > PAIR_Z_LOOKBACK:
        candidates = []
        for spec in _state["pairs"]:
            ia, ib, beta = spec[:3]
            if max(ia, ib) >= nInst:
                continue
            # Multi-window z: average the spread z-score over several
            # lookbacks instead of betting everything on one 60-day window
            # (diversification across time-scales, no new fitted constant).
            spread = (prcSoFar[ia, -PAIR_Z_LOOKBACK:]
                      - beta * prcSoFar[ib, -PAIR_Z_LOOKBACK:])
            z_vals = []
            for lb in (40, PAIR_Z_LOOKBACK, 90):
                if t <= lb:
                    continue
                sp = (prcSoFar[ia, -lb:] - beta * prcSoFar[ib, -lb:])
                sd_lb = sp.std()
                if sd_lb < 1e-9:
                    continue
                z_vals.append((sp[-1] - sp.mean()) / sd_lb)
            if not z_vals:
                continue
            z = float(np.mean(z_vals))
            strength = max(0.0, (abs(z) - PAIR_Z_EXIT) / (PAIR_Z_ENTRY - PAIR_Z_EXIT))
            strength = min(1.0, strength)
            if strength <= 0.0:
                continue

            if PAIR_STALL_FACTOR < 1.0 and spread.size > PAIR_STALL_DAYS:
                d_move = float(spread[-1] - spread[-1 - PAIR_STALL_DAYS])
                sd_diff = float(np.std(np.diff(spread)))
                still_diverging = (
                    np.sign(d_move) == np.sign(z)
                    and abs(d_move) > PAIR_STALL_TOL * sd_diff * np.sqrt(PAIR_STALL_DAYS)
                )
                if still_diverging:
                    strength *= PAIR_STALL_FACTOR

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

            spread_change_vol = float(np.std(np.diff(spread)))
            vol_per_gross = spread_change_vol / gross_per_unit
            candidates.append(
                (ia, ib, beta, want, strength, quality_scale,
                 gross_per_unit, vol_per_gross)
            )

        if candidates:
            vols = np.array([c[7] for c in candidates])
            med_vol = float(np.median(vols[vols > 1e-12])) if np.any(vols > 1e-12) else 0.0

        for ia, ib, beta, want, strength, quality_scale, gross_per_unit, vol_per_gross in candidates:
            rp_scale = 1.0
            if med_vol > 1e-12 and vol_per_gross > 1e-12:
                rp_scale = float(np.clip(med_vol / vol_per_gross, 0.5, 2.0))
            target_gross = PAIR_TRADE_DOLLARS * quality_scale * strength * rp_scale
            unit = want * target_gross / gross_per_unit
            key = (min(ia, ib), max(ia, ib))
            unit *= _pair_multiplier(_state["pair_pnl_hist"].get(key, []))
            pair_pos[ia] += unit
            pair_pos[ib] += -beta * unit
            prev_a, prev_b = new_pair_units.get(key, (0.0, 0.0))
            new_pair_units[key] = (prev_a + unit, prev_b - beta * unit)

        if DISP_APPLY_PAIRS and disp_mult != 1.0:
            pair_pos *= disp_mult
            new_pair_units = {
                k: (a * disp_mult, b * disp_mult)
                for k, (a, b) in new_pair_units.items()
            }

        pair_gross = float(np.sum(np.abs(pair_pos) * curPrices))
        if pair_gross > PAIR_MAX_BOOK_DOLLARS:
            pair_cap_scale = PAIR_MAX_BOOK_DOLLARS / pair_gross
            pair_pos *= pair_cap_scale

        if hist_len >= CVAR_MIN_HISTORY and PAIR_TARGET_DAILY_VOL > 0:
            pair_dollars = pair_pos * curPrices
            pair_window = full_log_ret[:, -min(hist_len, CVAR_LOOKBACK):]
            pair_pnl = pair_dollars @ pair_window
            pair_vol = float(np.std(pair_pnl))
            if pair_vol > 1e-9:
                book_scale = min(1.0, PAIR_TARGET_DAILY_VOL / pair_vol)
                pair_pos *= book_scale
                new_pair_units = {
                    k: (a * book_scale, b * book_scale)
                    for k, (a, b) in new_pair_units.items()
                }

    _state["last_pair_units"] = new_pair_units

    # ---- OU half-life pairs book (separate small book, see _ou_pair_book) -
    ou_pos = np.zeros(nInst)
    ou_units = {}
    if OU_TARGET_DAILY_VOL > 0:
        if t >= 2 and _state.get("last_ou_pos") is not None:
            dprice = prcSoFar[:, -1] - prcSoFar[:, -2]
            _state.setdefault("ou_pnl_hist", []).append(
                float(_state["last_ou_pos"] @ dprice)
            )
        ou_pos, ou_units = _ou_pair_book(
            prcSoFar, _state["pairs"], full_log_ret, hist_len
        )
        ou_mult = _pair_multiplier(_state.get("ou_pnl_hist", []))
        if OU_AC_GATE and full_log_ret.shape[1] > OU_AC_WINDOW + 2:
            w_m = full_log_ret.mean(axis=0)[-OU_AC_WINDOW:]
            ac_m = _safe_corr(w_m[1:], w_m[:-1])
            ou_mult *= float(np.clip(1.0 - ac_m / OU_AC_MAX, 0.0, 1.0))
        ou_pos *= ou_mult
        ou_units = {k: (a * ou_mult, b * ou_mult) for k, (a, b) in ou_units.items()}
        _state["last_ou_pos"] = ou_pos.copy()

    # ---- spread-momentum book (separate small book, see _spread_mom_book) -
    sm_pos = np.zeros(nInst)
    sm_units = {}
    if SPREADMOM_TARGET_DAILY_VOL > 0:
        if t >= 2 and _state.get("last_sm_pos") is not None:
            dprice = prcSoFar[:, -1] - prcSoFar[:, -2]
            _state.setdefault("sm_pnl_hist", []).append(
                float(_state["last_sm_pos"] @ dprice)
            )
        sm_pos, sm_units = _spread_mom_book(
            prcSoFar, _state["pairs"], full_log_ret, hist_len
        )
        sm_mult = _pair_multiplier(_state.get("sm_pnl_hist", []))
        sm_pos *= sm_mult
        sm_units = {k: (a * sm_mult, b * sm_mult) for k, (a, b) in sm_units.items()}
        _state["last_sm_pos"] = sm_pos.copy()

    # ---- inst0 AR(1) market-timing book (see INST0_AR_DOLLARS) ------------
    i0_pos = np.zeros(nInst)
    if INST0_AR_DOLLARS > 0 and full_log_ret.shape[1] > INST0_AR_WINDOW + 2:
        if t >= 2 and _state.get("last_i0_pos") is not None:
            dprice = prcSoFar[:, -1] - prcSoFar[:, -2]
            _state.setdefault("i0_pnl_hist", []).append(
                float(_state["last_i0_pos"] @ dprice)
            )
        mret = full_log_ret.mean(axis=0)
        w = mret[-INST0_AR_WINDOW:]
        ac = _safe_corr(w[1:], w[:-1])
        pred = ac * mret[-1]
        mvol = float(w.std()) + 1e-12
        i0_dollars = (INST0_AR_DOLLARS
                      * float(np.tanh(pred / (INST0_AR_TAU * mvol))))
        if abs(ac) < INST0_AR_MIN_AC:
            i0_dollars = 0.0
        # sized in FINAL dollars: undo the downstream FINAL_POSITION_SCALE
        i0_pos[0] = i0_dollars / curPrices[0] / FINAL_POSITION_SCALE
        i0_pos *= _pair_multiplier(_state.get("i0_pnl_hist", []))
        _state["last_i0_pos"] = i0_pos.copy()

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

    if HEDGE_LIMIT_PRESERVE:
        limit_shares_f = (dlr_limits / curPrices) / FINAL_POSITION_SCALE
        # effective pair units include the gross-cap factor, which the
        # tracked new_pair_units deliberately does NOT (adopted semantics)
        pair_units_eff = {
            k: (a * pair_cap_scale, b * pair_cap_scale)
            for k, (a, b) in new_pair_units.items()
        }
        unit_maps = [pair_units_eff]
        if OU_TARGET_DAILY_VOL > 0:
            unit_maps.append(ou_units)
        aggs = _hedge_preserving_limit(
            signal_positions.astype(float), unit_maps, limit_shares_f
        )
        pair_pos = aggs[0]
        _state["last_pair_units"] = pair_units_eff
        if OU_TARGET_DAILY_VOL > 0:
            ou_pos = aggs[1]
            _state["last_ou_pos"] = ou_pos.copy()

    pos_limits = (dlr_limits / curPrices).astype(int)
    if SPREADMOM_HEADROOM_FIT and SPREADMOM_TARGET_DAILY_VOL > 0:
        limit_shares_f = (dlr_limits / curPrices) / FINAL_POSITION_SCALE
        base_wo_sm = (signal_positions
                      + (pair_pos + i0_pos).astype(int)).astype(float)
        sm_pos = _hedge_preserving_limit(
            base_wo_sm, [sm_units], limit_shares_f
        )[0]
        _state["last_sm_pos"] = sm_pos.copy()
    if INST0_AR_HEADROOM_FIT and INST0_AR_DOLLARS > 0:
        limit_shares_f = (dlr_limits / curPrices) / FINAL_POSITION_SCALE
        base0 = float(signal_positions[0] + int(pair_pos[0] + sm_pos[0]))
        i0_pos[0] = float(
            np.clip(base0 + i0_pos[0], -limit_shares_f[0], limit_shares_f[0])
            - base0
        )
        _state["last_i0_pos"] = i0_pos.copy()
    if OU_HEADROOM_FIT and OU_TARGET_DAILY_VOL > 0:
        limit_shares_f = (dlr_limits / curPrices) / FINAL_POSITION_SCALE
        base_wo_ou = (signal_positions
                      + (pair_pos + sm_pos + i0_pos).astype(int)).astype(float)
        fitted = _hedge_preserving_limit(base_wo_ou, [ou_units], limit_shares_f)
        ou_pos = fitted[0]
        _state["last_ou_pos"] = ou_pos.copy()

    # ---- Donchian breakout book: headroom-fitted LAST ---------------------
    # Runs after every other book so its per-instrument clip sees the full
    # base (signal + pairs + sm + i0 + ou). Single-instrument book, so a
    # simple clip into residual headroom (vectorized INST0_AR pattern)
    # is exactly hedge-preserving.
    donch_pos = np.zeros(nInst)
    if DONCH_TARGET_DAILY_VOL > 0:
        donch_pos = _donch_book(prcSoFar, vol, full_log_ret)
        limit_shares_f = (dlr_limits / curPrices) / FINAL_POSITION_SCALE
        base_all = (signal_positions
                    + (pair_pos + ou_pos + sm_pos + i0_pos).astype(int)
                    ).astype(float)
        donch_pos = (
            np.clip(base_all + donch_pos, -limit_shares_f, limit_shares_f)
            - base_all
        )
    positions = np.clip(
        (signal_positions + (pair_pos + ou_pos + sm_pos + i0_pos).astype(int)
         + donch_pos.astype(int)) * FINAL_POSITION_SCALE,
        -pos_limits,
        pos_limits,
    ).astype(int)

    if TURNOVER_DEADBAND_DOLLARS > 0 and _state["last_pos"] is not None:
        prev_final = _state["last_pos"]
        small = (np.abs(positions - prev_final) * curPrices
                 < TURNOVER_DEADBAND_DOLLARS)
        positions = np.where(small, prev_final, positions).astype(int)

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
