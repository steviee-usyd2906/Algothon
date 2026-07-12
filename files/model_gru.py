"""
GRU next-day return prediction, pooled across all 51 instruments (PyTorch).

Identical setup to model_lstm.py with the recurrent cell swapped for a GRU.

Each sample: a LOOKBACK-day window (days d-30 .. d-1) of 3 channels for one
instrument — [own return, idiosyncratic return (own - market), market
return] — standardised with statistics from the inner-training window only.
Target: that instrument's return on day d.

Split (same as the panel models): targets [0, 350) inner-train, [350, 400)
validation for early stopping, [400, 500) test. Targets are scaled x100 for
training and predictions scaled back.
"""

import copy

import numpy as np
import torch
import torch.nn as nn

from features import load_prices, TRAIN_END_DAY, TEST_END_DAY, VAL_TARGET_START
from common_eval import report_metrics, save_predictions

SEED = 42
LOOKBACK = 30
HIDDEN = 32
DROPOUT = 0.2
BATCH = 512
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3
WEIGHT_DECAY = 1e-5
Y_SCALE = 100.0
PREDICTIONS_FILE = "gru_test_predictions.csv"

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------ data ----------------------------------------
prices = load_prices()
tickers = list(prices.columns)
n_days, n_inst = prices.shape
ret = prices.pct_change().to_numpy()                 # (500, 51); day 0 is NaN
mkt = ret.mean(axis=1)                               # equal-weight market return
chan = np.stack(
    [ret, ret - mkt[:, None], np.broadcast_to(mkt[:, None], ret.shape)],
    axis=-1,
)                                                    # (500, 51, 3)

# standardise each channel using data available during inner training only
stats = chan[1:VAL_TARGET_START]                     # days 1 .. 349
chan = (chan - stats.mean(axis=(0, 1))) / stats.std(axis=(0, 1))

# windows: target day d uses channel days d-LOOKBACK .. d-1 (all >= 1 => d >= 31)
sw = np.lib.stride_tricks.sliding_window_view(chan, LOOKBACK, axis=0)
sw = sw.transpose(0, 1, 3, 2)                        # (471, 51, 30, 3); window s = days s..s+29
d_arr = np.arange(LOOKBACK + 1, n_days)              # target days 31 .. 499
X_all = sw[d_arr - LOOKBACK]                         # (469, 51, 30, 3)
y_all = ret[d_arr]                                   # (469, 51)
assert not np.isnan(X_all).any() and not np.isnan(y_all).any()

target_day = np.repeat(d_arr, n_inst)
inst_id = np.tile(np.arange(n_inst), len(d_arr))
X_flat = X_all.reshape(-1, LOOKBACK, 3).astype(np.float32)
y_flat = y_all.reshape(-1).astype(np.float32)

train_mask = target_day < VAL_TARGET_START
val_mask = (target_day >= VAL_TARGET_START) & (target_day < TRAIN_END_DAY)
test_mask = (target_day >= TRAIN_END_DAY) & (target_day < TEST_END_DAY)
print(f"rows  train: {int(train_mask.sum())}   val (early stopping): "
      f"{int(val_mask.sum())}   test: {int(test_mask.sum())}")


def make_loader(mask, shuffle):
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_flat[mask]),
        torch.from_numpy(y_flat[mask] * Y_SCALE),
    )
    return torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=shuffle)


train_loader = make_loader(train_mask, shuffle=True)
val_loader = make_loader(val_mask, shuffle=False)

# ------------------------------ model ---------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(input_size=3, hidden_size=HIDDEN, batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(self.drop(out[:, -1])).squeeze(-1)


model = Net().to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()


def eval_mse(loader):
    model.eval()
    se, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            p = model(xb.to(device))
            se += float(((p - yb.to(device)) ** 2).sum())
            n += len(yb)
    return se / n


best_val, best_state, best_epoch, bad = np.inf, None, -1, 0
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    for xb, yb in train_loader:
        opt.zero_grad()
        loss = loss_fn(model(xb.to(device)), yb.to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    v = eval_mse(val_loader)
    if v < best_val - 1e-8:
        best_val, best_state, best_epoch, bad = v, copy.deepcopy(model.state_dict()), epoch, 0
    else:
        bad += 1
        if bad >= PATIENCE:
            break
print(f"stopped after epoch {epoch}; best epoch {best_epoch}, "
      f"val MSE {best_val / Y_SCALE ** 2:.8e}")
model.load_state_dict(best_state)

# ------------------------------ test ----------------------------------------
model.eval()
X_te = torch.from_numpy(X_flat[test_mask])
preds = []
with torch.no_grad():
    for i in range(0, len(X_te), BATCH):
        preds.append(model(X_te[i:i + BATCH].to(device)).cpu().numpy())
pred = np.concatenate(preds) / Y_SCALE

report_metrics("GRU", pred, y_flat[test_mask], target_day[test_mask])
save_predictions(PREDICTIONS_FILE, target_day[test_mask], inst_id[test_mask],
                 tickers, y_flat[test_mask], pred)
