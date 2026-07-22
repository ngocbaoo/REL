import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.trading_env import QcomTradingEnv, HOLD, BUY_ACTIONS, SELL_ACTIONS
from agent.a2c_agent import A2CAgent


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


def test_no_cooldown_restriction_on_consecutive_trades():
    df = _make_df()
    env = QcomTradingEnv(df, episode_length=None, seed=1)
    obs, info = env.reset(seed=1)

    assert info["action_mask"][BUY_ACTIONS[0]]
    obs, reward, terminated, truncated, info = env.step(BUY_ACTIONS[0])
    assert not (terminated or truncated)
    assert info["action_mask"][BUY_ACTIONS[0]], "BUY must still be valid right after a BUY (no cooldown)"
    assert info["action_mask"][SELL_ACTIONS[0]], "SELL must be valid right after a BUY since holdings > 0"


def test_idle_streak_triggers_cash_penalty():
    df = _make_df()
    env = QcomTradingEnv(df, episode_length=None, seed=7)
    obs, info = env.reset(seed=7)

    cash_before = env.cash
    terminated = truncated = False
    for i in range(env.transaction_session):
        obs, reward, terminated, truncated, info = env.step(HOLD)
        if terminated or truncated:
            break

    if not (terminated or truncated):
        assert abs(env.cash - (cash_before - env.transaction_penalty)) < 1e-6
        assert env.steps_since_last_trade == 0


def test_greedy_agent_never_selects_masked_action():
    df = _make_df()
    env = QcomTradingEnv(df, episode_length=None, seed=3)
    obs, info = env.reset(seed=3)

    agent = A2CAgent(obs_dim=env.observation_space.shape[0], n_actions=11, seed=3)

    done = False
    steps = 0
    while not done and steps < 200:
        mask = info["action_mask"]
        assert mask[0], "HOLD must always be a valid action"
        action = agent.select_action(obs, mask, greedy=True)
        assert mask[action], "Selected action must be valid under the mask"

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1


if __name__ == "__main__":
    test_action_mask_blocks_buy_without_cash_and_sell_without_holdings()
    test_no_cooldown_restriction_on_consecutive_trades()
    test_idle_streak_triggers_cash_penalty()
    test_greedy_agent_never_selects_masked_action()
    print("All action masking tests passed.")
