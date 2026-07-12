"""
Per-instrument AR(1)-GARCH(1,1) next-day return prediction (arch package).

The AR(1) mean equation supplies the return forecast that is evaluated;
GARCH(1,1) supplies the volatility forecast (saved alongside). Parameters
are estimated on training returns only (days 1-399), then held FIXED and
applied to the full series, so a test forecast for day d uses actual
returns through day d-1 while no test-period data enters estimation.

Returns are scaled x100 for optimiser stability; forecasts are scaled back.

Requires: pip install arch
"""

import warnings

import numpy as np
from arch import arch_model

from features import load_prices, TRAIN_END_DAY, TEST_END_DAY
from common_eval import report_metrics, save_predictions

PREDICTIONS_FILE = "garch_test_predictions.csv"

warnings.filterwarnings("ignore")

prices = load_prices()
tickers = list(prices.columns)
n_inst = prices.shape[1]
ret = prices.pct_change()
n_test = TEST_END_DAY - TRAIN_END_DAY

pred_mat = np.full((n_test, n_inst), np.nan)
vol_mat = np.full((n_test, n_inst), np.nan)
persistence = []

for j, col in enumerate(tickers):
    r = 100.0 * ret[col].to_numpy()[1:]          # percent; positions 0..498 = days 1..499
    r_train = r[: TRAIN_END_DAY - 1]             # returns on days 1..399
    try:
        am_train = arch_model(r_train, mean="AR", lags=1,
                              vol="GARCH", p=1, q=1, dist="normal")
        res_train = am_train.fit(disp="off", show_warning=False)

        am_full = arch_model(r, mean="AR", lags=1,
                             vol="GARCH", p=1, q=1, dist="normal")
        res_fix = am_full.fix(res_train.params)

        # arch alignment: the forecast in row t is made with information
        # through t, and column h.1 is the forecast OF t+1. Row 398 therefore
        # predicts position 399 = the day-400 return.
        f = res_fix.forecast(horizon=1, start=TRAIN_END_DAY - 2, reindex=False)
        mu = f.mean["h.1"].to_numpy()
        var = f.variance["h.1"].to_numpy()
        assert len(mu) >= n_test
        pred_mat[:, j] = mu[:n_test] / 100.0
        vol_mat[:, j] = np.sqrt(var[:n_test]) / 100.0
        persistence.append(float(res_train.params["alpha[1]"]
                                 + res_train.params["beta[1]"]))
    except Exception as e:
        pred_mat[:, j] = r_train.mean() / 100.0   # fallback: constant train mean
        vol_mat[:, j] = r_train.std() / 100.0
        print(f"{col}: fit failed ({e}); using train mean/std")

if persistence:
    p = np.asarray(persistence)
    print(f"GARCH persistence (alpha+beta): mean {p.mean():.3f}  "
          f"median {np.median(p):.3f}  ({len(p)}/{n_inst} fits ok)")

y_true = ret.iloc[TRAIN_END_DAY:TEST_END_DAY].to_numpy()
target_days = np.repeat(np.arange(TRAIN_END_DAY, TEST_END_DAY), n_inst)
inst_ids = np.tile(np.arange(n_inst), n_test)

report_metrics("AR(1)-GARCH(1,1)", pred_mat.ravel(), y_true.ravel(), target_days)
save_predictions(PREDICTIONS_FILE, target_days, inst_ids, tickers,
                 y_true.ravel(), pred_mat.ravel(),
                 extra={"pred_vol": vol_mat.ravel()})
