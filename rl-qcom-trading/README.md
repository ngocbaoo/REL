# RL Trading Agent — QCOM (REL301m)

**Student:** Ta Bao Ngoc (HE191387) — **Assigned stock:** QCOM (Qualcomm, Inc.)

An A2C (Advantage Actor-Critic) agent with action masking, trained to trade
QCOM daily bars against a custom Gymnasium environment. On-policy actor-critic
with GAE and an entropy bonus — chosen over value-based DQN because it is far
more stable here and does not collapse into degenerate all-hold / all-buy
policies (see Results).

## Personal parameters (fixed)

| Param | Value |
|---|---|
| B_MIN / B_MAX | 27% / 62% |
| SELL_MIN / SELL_MAX | 8% / 88% |
| TRANSACTION_FEE | 3% |
| TRANSACTION_SESSION | 32 sessions (no-trade streak before a cash penalty; does not lock orders) |
| TRANSACTION_PENALTY | 2567 USD (deducted directly from cash) |
| TOTAL_ASSETS_INITIAL | 100,000 USD |
| TOTAL_ASSETS_THRESHOLD | 100,000 USD (lose condition, combined with cash floor — see below) |
| CASH_FLOOR | -5,000 USD |
| WIN_TARGET | 1,000,000 USD |

**Win/lose conditions:** the episode ends in a **loss** only when total_assets
< 100,000 USD **and** cash <= -5,000 USD happen *simultaneously*; it ends in a
**win** when total_assets reaches 1,000,000 USD. TRANSACTION_SESSION never
blocks a BUY/SELL — action masking only disallows BUY when cash <= 0 and SELL
when holdings <= 0.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Fetch data

```bash
python data/fetch_data.py                 # via yfinance (ticker QCOM)
python data/fetch_data.py --csv path.csv   # or a manually-downloaded OHLCV CSV
```

Produces `data/qcom_train.csv` (oldest 8 of the last 10 years) and
`data/qcom_test.csv` (most recent 2 years, out-of-sample).

## 2. Train

```bash
python train_a2c.py --config configs/config.yaml
```

- TensorBoard logs: `outputs/logs/a2c/` (`tensorboard --logdir outputs/logs`)
- Checkpoints: `outputs/checkpoints/a2c_best.pt` (highest total_assets on a
  held-out validation slice — last 6 months of train data) and `a2c_final.pt`.

## 3. Evaluate

```bash
python evaluate.py --checkpoint outputs/checkpoints/a2c_best.pt --split train
python evaluate.py --checkpoint outputs/checkpoints/a2c_best.pt --split test
```

Prints cumulative reward, final total_assets, `score = total_assets / 1,000,000`,
and a Buy-and-Hold baseline for comparison. Exports full trade logs to
`outputs/trade_history_train.csv` / `outputs/trade_history_test.csv`.

## 4. Tests

```bash
python -m pytest tests/ -q
```

Verifies BUY/SELL remain valid immediately after a trade (no cooldown lock),
that BUY/SELL are masked out when cash/holdings are insufficient, and that an
idle streak of TRANSACTION_SESSION steps triggers the cash penalty.

## Project structure

```
rl-qcom-trading/
├── data/               # fetch_data.py, qcom_train.csv, qcom_test.csv
├── env/trading_env.py  # QcomTradingEnv (Gymnasium)
├── agent/a2c_agent.py  # A2CAgent + ActorCritic network
├── train_a2c.py
├── evaluate.py
├── configs/config.yaml
├── outputs/            # checkpoints/, logs/, trade_history_*.csv
└── tests/
```

## Config flags (configs/config.yaml)

- `env.price_mode`: `close` (default) | `avg_close_high` | `prev_day_high`
- `env.total_assets_threshold` / `env.cash_floor`: both must be breached simultaneously to trigger a loss
- `env.win_target`: total_assets threshold that ends the episode in a win
- `env.transaction_session` / `env.transaction_penalty`: no-trade streak length and the cash penalty applied once it's reached (does not block orders)
- `env.episode_length`: `null` for one long episode over the full dataset, or e.g. `252` to chunk into shorter episodes with random start offsets during training

## Results summary

Run: 200,000 timesteps, `price_mode: close`, starting capital $100,000 on both
splits. `a2c_best.pt` is selected by highest total_assets on the 6-month
validation slice during training; `a2c_final.pt` is the last checkpoint.

| Checkpoint | Split | Final total_assets (USD) | Buy-and-Hold (USD) | Agent vs B&H |
|---|---|---|---|---|
| a2c_best.pt | Train | 95,603.42 | 347,059.72 | -251,456.31 USD (-72.45%) |
| a2c_best.pt | Test (out-of-sample) | **104,791.16** | 83,152.46 | **+21,638.71 USD (+26.02%)** |
| a2c_final.pt | Test (out-of-sample) | 104,526.93 | 83,152.46 | +21,374.48 USD (+25.71%) |

`a2c_best.pt` is **profitable out-of-sample** (+4.79% over 2 years) and beats
Buy-and-Hold by +26%. Unlike a value-based DQN — whose `final` checkpoint
degenerated into an all-hold policy and lost -38% on test — **both** A2C
checkpoints stay profitable, thanks to the entropy bonus preventing collapse.

### Honest caveat: the learned policy is dollar-cost-averaging (DCA)

The greedy A2C policy on test is **~492 BUY, 0 SELL** — it gradually accumulates
shares every session. A trivial `always BUY-max` baseline yields the *identical*
104,791.16 USD. In other words, the agent rediscovers DCA rather than learning
market timing; the +26% edge over Buy-and-Hold comes purely from averaging in at
lower prices during the dip-then-recover test window, and it still suffered a
-29% intra-period drawdown. On single-stock daily bars with technical features
there is little learnable timing signal, so DCA is the empirical ceiling —
switching algorithms (DQN → A2C) improves *stability*, and risk-adjusted reward
shaping only trades the DCA corner for an all-cash corner. Beating DCA would
require changing the problem (multi-asset, higher-frequency data, or genuinely
predictive features). Full trade logs: `outputs/trade_history_train.csv`,
`outputs/trade_history_test.csv`.
