"""Unit test: agent must never execute BUY/SELL within TRANSACTION_SESSION
steps of the previous executed trade, whether acting randomly or greedily."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.trading_env import QcomTradingEnv, BUY_ACTIONS, SELL_ACTIONS
from agent.dqn_agent import DQNAgent


def _make_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    price = 100 + np.cumsum(rng.normal(0, 1, size=n))
    price = np.clip(price, 10, None)
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": price + rng.normal(0, 0.1, n),
            "High": price + np.abs(rng.normal(0, 0.5, n)),
            "Low": price - np.abs(rng.normal(0, 0.5, n)),
            "Close": price,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }
    )
    return df


def test_random_actions_respect_cooldown():
    df = _make_df()
    env = QcomTradingEnv(df, episode_length=None, seed=1)
    obs, info = env.reset(seed=1)

    rng = np.random.default_rng(2)
    last_trade_step = -10_000
    step = 0
    done = False
    while not done:
        action = int(rng.integers(0, 11))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if info["action_taken"] in (BUY_ACTIONS + SELL_ACTIONS):
            assert step - last_trade_step >= env.transaction_session, (
                f"Trade executed at step {step}, only {step - last_trade_step} "
                f"steps after previous trade at {last_trade_step} (< {env.transaction_session})"
            )
            last_trade_step = step
        step += 1


def test_greedy_agent_respects_cooldown():
    df = _make_df()
    env = QcomTradingEnv(df, episode_length=None, seed=3)
    obs, info = env.reset(seed=3)

    agent = DQNAgent(obs_dim=env.observation_space.shape[0], n_actions=11, seed=3)

    last_trade_step = -10_000
    step = 0
    done = False
    while not done:
        mask = info["action_mask"]
        assert mask[0], "HOLD must always be a valid action"
        action = agent.select_action(obs, mask, greedy=True)
        assert mask[action], "Selected action must be valid under the mask"

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if info["action_taken"] in (BUY_ACTIONS + SELL_ACTIONS):
            assert step - last_trade_step >= env.transaction_session
            last_trade_step = step
        step += 1


def test_action_mask_blocks_buy_without_cash_and_sell_without_holdings():
    df = _make_df()
    env = QcomTradingEnv(df, episode_length=None, seed=5)
    obs, info = env.reset(seed=5)

    env.cash = 0.0
    mask = env.action_mask()
    assert not mask[list(BUY_ACTIONS)].any()

    env.cash = 1000.0
    env.holdings = 0.0
    mask = env.action_mask()
    assert not mask[list(SELL_ACTIONS)].any()


if __name__ == "__main__":
    test_random_actions_respect_cooldown()
    test_greedy_agent_respects_cooldown()
    test_action_mask_blocks_buy_without_cash_and_sell_without_holdings()
    print("All action masking tests passed.")
