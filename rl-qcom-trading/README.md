# RL Trading Agent — QCOM (REL301m)

**Student:** Ta Bao Ngoc (HE191387) — **Assigned stock:** QCOM (Qualcomm, Inc.)

A Double DQN + Dueling + Prioritized Experience Replay (PER) agent, with
action masking, trained to trade QCOM daily bars against a custom
Gymnasium environment.

## Personal parameters (fixed)

| Param | Value |
|---|---|
| B_MIN / B_MAX | 27% / 62% |
| SELL_MIN / SELL_MAX | 8% / 88% |
| TRANSACTION_FEE | 3% |
| TRANSACTION_SESSION | 32 steps |
| TRANSACTION_PENALTY | 2567 USD (reward-shaping only) |
| TOTAL_ASSETS_INITIAL | 1000 USD |

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
python train.py --config configs/config.yaml
```

- TensorBoard logs: `outputs/logs/` (`tensorboard --logdir outputs/logs`)
- Checkpoints: `outputs/checkpoints/best.pt` (highest total_assets on a
  held-out validation slice — last 6 months of train data) and `final.pt`.

## 3. Evaluate

```bash
python evaluate.py --checkpoint outputs/checkpoints/best.pt --split train
python evaluate.py --checkpoint outputs/checkpoints/best.pt --split test
```

Prints cumulative reward, final total_assets, `score = total_assets / 1,000,000`,
and a Buy-and-Hold baseline for comparison. Exports full trade logs to
`outputs/trade_history_train.csv` / `outputs/trade_history_test.csv`.

## 4. Tests

```bash
python -m pytest tests/ -q
```

Verifies action masking never allows a BUY/SELL within 32 steps of the
previous trade, and that BUY/SELL are masked out when cash/holdings are
insufficient.

## Project structure

```
rl-qcom-trading/
├── data/               # fetch_data.py, qcom_train.csv, qcom_test.csv
├── env/trading_env.py  # QcomTradingEnv (Gymnasium)
├── agent/              # networks.py, replay_buffer.py, dqn_agent.py
├── train.py
├── evaluate.py
├── configs/config.yaml
├── outputs/            # checkpoints/, logs/, trade_history_*.csv
└── tests/
```

## Config flags (configs/config.yaml)

- `env.price_mode`: `close` (default) | `avg_close_high` | `prev_day_high`
- `env.use_soft_cooldown_penalty`: ablation flag for the soft cooldown-violation reward penalty
- `env.idle_penalty_enabled` / `idle_penalty_value` / `idle_penalty_multiplier`
- `env.episode_length`: `null` for one long episode over the full dataset, or e.g. `252` to chunk into shorter episodes with random start offsets during training

## Results summary

Run: `outputs/checkpoints/best.pt` (selected by highest total_assets on the
6-month validation slice during training), 200,000 timesteps, `price_mode: close`.

| Split | Final total_assets (USD) | Score | Buy-and-Hold (USD) | Agent vs B&H |
|---|---|---|---|---|
| Train (best ckpt) | 3,262.82 | 0.003263 | 3,470.60 | -207.77 USD (-5.99%) |
| Test (out-of-sample) | 1,071.81 | 0.001072 | 831.52 | +240.29 USD (+28.90%) |

Starting capital was $1,000 (`TOTAL_ASSETS_INITIAL`) on both splits. The agent
slightly underperforms Buy-and-Hold on the in-sample train set (-6%) but
outperforms it by +28.9% out-of-sample on the test set, where Buy-and-Hold
suffered from QCOM's drawdown over that window while the agent's 32-step
cooldown and position sizing reduced exposure to it. Training reward curve:
episode cumulative reward rose steadily from ~0.25 (ep 0) to ~1.8 (ep 100) as
epsilon decayed from 1.0 to the 0.05 floor by ~ep 55; episode final
total_assets rose from ~$1,072 to a peak of ~$4,882 by the end of training.
