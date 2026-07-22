from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

N_ACTIONS = 11
HOLD = 0
BUY_ACTIONS = (1, 2, 3, 4, 5)
SELL_ACTIONS = (6, 7, 8, 9, 10)
N_MARKET_FEATURES = 9
N_PORTFOLIO_FEATURES = 5


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)
    return rsi


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line


def compute_market_features(df: pd.DataFrame) -> np.ndarray:
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]

    log_ret_open = np.log(o / o.shift(1))
    log_ret_high = np.log(h / h.shift(1))
    log_ret_low = np.log(l / l.shift(1))
    log_ret_close = np.log(c / c.shift(1))

    vol_mean = v.rolling(20, min_periods=20).mean()
    vol_std = v.rolling(20, min_periods=20).std()
    vol_zscore = (v - vol_mean) / vol_std.replace(0.0, np.nan)

    sma10 = c.rolling(10, min_periods=10).mean()
    sma20 = c.rolling(20, min_periods=20).mean()
    sma10_norm = (c - sma10) / sma10
    sma20_norm = (c - sma20) / sma20

    rsi_scaled = _rsi(c, 14) / 100.0

    macd_hist_norm = _macd_hist(c) / c

    feats = pd.concat(
        [
            log_ret_open,
            log_ret_high,
            log_ret_low,
            log_ret_close,
            vol_zscore,
            sma10_norm,
            sma20_norm,
            rsi_scaled,
            macd_hist_norm,
        ],
        axis=1,
    )
    feats.columns = range(N_MARKET_FEATURES)
    return feats.to_numpy(dtype=np.float64)


class QcomTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        lookback_window: int = 10,
        total_assets_initial: float = 100_000.0,
        b_min: float = 0.27,
        b_max: float = 0.62,
        sell_min: float = 0.08,
        sell_max: float = 0.88,
        transaction_fee: float = 0.03,
        transaction_session: int = 32,
        transaction_penalty: float = 2567.0,
        total_assets_threshold: float = 100_000.0,
        cash_floor: float = -5_000.0,
        win_target: float = 1_000_000.0,
        price_mode: str = "close",
        episode_length: int | None = None,
        random_start: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.dates = pd.to_datetime(self.df["Date"]).to_numpy()
        self.open = self.df["Open"].to_numpy(dtype=np.float64)
        self.high = self.df["High"].to_numpy(dtype=np.float64)
        self.low = self.df["Low"].to_numpy(dtype=np.float64)
        self.close = self.df["Close"].to_numpy(dtype=np.float64)
        self.volume = self.df["Volume"].to_numpy(dtype=np.float64)

        self.lookback_window = lookback_window
        self.total_assets_initial = total_assets_initial
        self.b_min = b_min
        self.b_max = b_max
        self.sell_min = sell_min
        self.sell_max = sell_max
        self.buy_ratios = np.linspace(b_min, b_max, 5)
        self.sell_ratios = np.linspace(sell_min, sell_max, 5)
        self.transaction_fee = transaction_fee
        self.transaction_session = transaction_session
        self.transaction_penalty = transaction_penalty
        self.total_assets_threshold = total_assets_threshold
        self.cash_floor = cash_floor
        self.win_target = win_target
        self.price_mode = price_mode
        self.episode_length = episode_length
        self.random_start = random_start

        market_feats = compute_market_features(self.df)
        valid_mask = ~np.isnan(market_feats).any(axis=1)
        if not valid_mask.any():
            raise ValueError("Not enough data to compute market features")
        first_valid_idx = int(np.argmax(valid_mask))
        self.market_features = market_feats
        self.min_start_idx = first_valid_idx + lookback_window - 1
        self.last_idx = len(self.df) - 1

        if self.min_start_idx > self.last_idx:
            raise ValueError("Dataset too short for lookback_window / feature warmup")

        self.action_space = spaces.Discrete(N_ACTIONS)
        obs_dim = N_MARKET_FEATURES * lookback_window + N_PORTFOLIO_FEATURES
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._np_random = np.random.default_rng(seed)

        self.t = self.min_start_idx
        self.episode_end_idx = self.last_idx
        self.cash = total_assets_initial
        self.holdings = 0.0
        self.avg_cost_basis = 0.0
        self.total_assets = total_assets_initial
        self.prev_total_assets = total_assets_initial
        self.steps_since_last_trade = 0

    def _execution_price(self, t: int) -> float:
        if self.price_mode == "close":
            return float(self.close[t])
        elif self.price_mode == "avg_close_high":
            return float((self.close[t] + self.high[t]) / 2.0)
        elif self.price_mode == "prev_day_high":
            prev_t = max(t - 1, 0)
            return float(self.high[prev_t])
        else:
            raise ValueError(f"Unknown price_mode: {self.price_mode}")

    def action_mask(self) -> np.ndarray:
        mask = np.ones(N_ACTIONS, dtype=bool)
        if self.cash <= 0:
            for a in BUY_ACTIONS:
                mask[a] = False
        if self.holdings <= 0:
            for a in SELL_ACTIONS:
                mask[a] = False
        return mask

    def _apply_mask(self, action: int) -> int:
        mask = self.action_mask()
        if not mask[action]:
            return HOLD
        return action

    def _get_obs(self) -> np.ndarray:
        start = self.t - self.lookback_window + 1
        window = self.market_features[start : self.t + 1]
        market_flat = window.flatten().astype(np.float32)

        price = self.close[self.t]
        cash_ratio = self.cash / self.total_assets if self.total_assets > 0 else 0.0
        holding_value = self.holdings * price
        holding_ratio = holding_value / self.total_assets if self.total_assets > 0 else 0.0
        if self.holdings > 0 and self.avg_cost_basis > 0:
            unrealized_pnl_pct = (price - self.avg_cost_basis) / self.avg_cost_basis
        else:
            unrealized_pnl_pct = 0.0
        no_trade_streak_norm = min(1.0, self.steps_since_last_trade / self.transaction_session)


        asset_ratio_raw = self.total_assets / self.total_assets_initial
        asset_ratio = float(np.tanh(np.log(max(asset_ratio_raw, 1e-9))))

        portfolio = np.array(
            [cash_ratio, holding_ratio, unrealized_pnl_pct, no_trade_streak_norm, asset_ratio],
            dtype=np.float32,
        )
        return np.concatenate([market_flat, portfolio])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        if self.episode_length is not None:
            latest_start = self.last_idx - self.episode_length
            if latest_start <= self.min_start_idx:
                start = self.min_start_idx
                end = self.last_idx
            else:
                if self.random_start:
                    start = int(self._np_random.integers(self.min_start_idx, latest_start + 1))
                else:
                    start = self.min_start_idx
                end = min(start + self.episode_length, self.last_idx)
            self.t = start
            self.episode_end_idx = end
        else:
            self.t = self.min_start_idx
            self.episode_end_idx = self.last_idx

        self.cash = self.total_assets_initial
        self.holdings = 0.0
        self.avg_cost_basis = 0.0
        self.total_assets = self.total_assets_initial
        self.prev_total_assets = self.total_assets_initial
        self.steps_since_last_trade = 0

        obs = self._get_obs()
        info = {
            "total_assets": self.total_assets,
            "cash": self.cash,
            "holdings": self.holdings,
            "action_mask": self.action_mask(),
        }
        return obs, info

    def step(self, action: int):

        action_raw = int(action)
        price = self._execution_price(self.t)

        action_taken = self._apply_mask(action_raw)


        traded = False

        if action_taken in BUY_ACTIONS:
            ratio = self.buy_ratios[action_taken - 1]
            spend = ratio * self.cash
            shares_bought = (spend * (1 - self.transaction_fee)) / price
            new_holdings = self.holdings + shares_bought
            if new_holdings > 0:
                self.avg_cost_basis = (
                    self.avg_cost_basis * self.holdings + spend
                ) / new_holdings
            self.cash -= spend
            self.holdings = new_holdings
            traded = True


        elif action_taken in SELL_ACTIONS:
            ratio = self.sell_ratios[action_taken - 6]
            shares_sold = ratio * self.holdings
            proceeds = shares_sold * price * (1 - self.transaction_fee)
            self.cash += proceeds
            self.holdings -= shares_sold
            if self.holdings <= 1e-12:
                self.holdings = 0.0
                self.avg_cost_basis = 0.0
            traded = True


        if traded:
            self.steps_since_last_trade = 0
        else:
            self.steps_since_last_trade += 1
            if self.steps_since_last_trade >= self.transaction_session:
                self.cash -= self.transaction_penalty
                self.steps_since_last_trade = 0


        self.total_assets = self.cash + self.holdings * price

        reward = (self.total_assets - self.prev_total_assets) / self.prev_total_assets


        lost = self.total_assets < self.total_assets_threshold and self.cash <= self.cash_floor


        won = self.total_assets >= self.win_target
        terminated = bool(lost or won)
        if lost:
            reward += -1.0

        self.prev_total_assets = self.total_assets

        truncated = self.t >= self.episode_end_idx
        if not terminated and not truncated:
            self.t += 1


        obs = self._get_obs()
        info = {
            "total_assets": self.total_assets,
            "cash": self.cash,
            "holdings": self.holdings,
            "action_taken": action_taken,
            "action_raw": action_raw,
            "action_mask": self.action_mask(),
            "date": self.dates[min(self.t, self.last_idx)],
            "price": price,
        }
        return obs, reward, terminated, truncated, info
