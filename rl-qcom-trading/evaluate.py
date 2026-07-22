from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
import yaml

from agent.a2c_agent import A2CAgent
from env.trading_env import QcomTradingEnv, N_ACTIONS, BUY_ACTIONS, SELL_ACTIONS

ACTION_NAMES = {0: "HOLD"}
ACTION_NAMES.update({a: "BUY" for a in BUY_ACTIONS})
ACTION_NAMES.update({a: "SELL" for a in SELL_ACTIONS})


def make_env(df: pd.DataFrame, env_cfg: dict) -> QcomTradingEnv:
    return QcomTradingEnv(
        df,
        lookback_window=env_cfg["lookback_window"],
        total_assets_initial=env_cfg["total_assets_initial"],
        b_min=env_cfg["b_min"],
        b_max=env_cfg["b_max"],
        sell_min=env_cfg["sell_min"],
        sell_max=env_cfg["sell_max"],
        transaction_fee=env_cfg["transaction_fee"],
        transaction_session=env_cfg["transaction_session"],
        transaction_penalty=env_cfg["transaction_penalty"],
        total_assets_threshold=env_cfg["total_assets_threshold"],
        cash_floor=env_cfg["cash_floor"],
        win_target=env_cfg["win_target"],
        price_mode=env_cfg["price_mode"],
        episode_length=None,
        random_start=False,
        seed=0,
    )


def resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def buy_and_hold_baseline(df: pd.DataFrame, total_assets_initial: float, transaction_fee: float) -> float:
    first_price = df["Close"].iloc[0]
    last_price = df["Close"].iloc[-1]
    shares = (total_assets_initial * (1 - transaction_fee)) / first_price
    final_value = shares * last_price * (1 - transaction_fee)
    return final_value


def run_rollout(agent: A2CAgent, df: pd.DataFrame, env_cfg: dict):
    env = make_env(df, env_cfg)
    obs, info = env.reset()

    records = []
    cumulative_reward = 0.0
    done = False

    while not done:
        mask = info["action_mask"]
        action = agent.select_action(obs, mask, greedy=True)

        pre_cash, pre_holdings = env.cash, env.holdings
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        cumulative_reward += reward

        action_taken = info["action_taken"]
        price = info["price"]
        shares_traded = abs(env.holdings - pre_holdings) if action_taken != 0 else 0.0
        cash_delta = env.cash - pre_cash
        if action_taken in BUY_ACTIONS:
            spend = -cash_delta
            fee_paid = spend * env_cfg["transaction_fee"] / (1 - env_cfg["transaction_fee"]) if spend > 0 else 0.0
            ratio = env.buy_ratios[action_taken - 1]
        elif action_taken in SELL_ACTIONS:
            proceeds = cash_delta
            fee_paid = proceeds * env_cfg["transaction_fee"] / (1 - env_cfg["transaction_fee"]) if proceeds > 0 else 0.0
            ratio = env.sell_ratios[action_taken - 6]
        else:
            fee_paid = 0.0
            ratio = 0.0

        records.append(
            {
                "date": pd.Timestamp(info["date"]).date(),
                "action_id": action_taken,
                "action_type": ACTION_NAMES[action_taken],
                "ratio": ratio,
                "price": price,
                "shares_traded": shares_traded,
                "fee_paid": fee_paid,
                "cash": env.cash,
                "holdings": env.holdings,
                "total_assets": info["total_assets"],
            }
        )

    trade_log = pd.DataFrame(records)
    return cumulative_reward, info["total_assets"], trade_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/a2c_best.pt")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--split", type=str, choices=["train", "test"], required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]
    a2c_cfg = cfg["a2c"]
    train_cfg = cfg["train"]

    csv_path = cfg["data"]["train_csv"] if args.split == "train" else cfg["data"]["test_csv"]
    df = pd.read_csv(csv_path)

    device = resolve_device(train_cfg["device"])

    dummy_env = make_env(df, env_cfg)
    agent = A2CAgent(
        obs_dim=dummy_env.observation_space.shape[0],
        n_actions=N_ACTIONS,
        hidden_sizes=a2c_cfg["hidden_sizes"],
        gamma=a2c_cfg["gamma"],
        device=device,
        seed=0,
    )
    agent.load(args.checkpoint)

    cumulative_reward, final_assets, trade_log = run_rollout(agent, df, env_cfg)

    total_assets_initial = env_cfg["total_assets_initial"]
    score = final_assets / 1_000_000.0
    bh_final = buy_and_hold_baseline(df, total_assets_initial, env_cfg["transaction_fee"])

    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/trade_history_{args.split}.csv"
    trade_log.to_csv(out_path, index=False)

    n_trades = int((trade_log["action_type"] != "HOLD").sum())

    print(f"=== Evaluation: {args.split} split ({csv_path}) ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Rows: {len(df)}  Trades executed: {n_trades}")
    print(f"Cumulative reward: {cumulative_reward:.4f}")
    print(f"Final total_assets: {final_assets:.2f} USD")
    print(f"Score (total_assets / 1,000,000): {score:.6f}")
    print(f"Buy-and-Hold final total_assets: {bh_final:.2f} USD")
    print(f"Agent vs Buy-and-Hold: {final_assets - bh_final:+.2f} USD ({(final_assets / bh_final - 1) * 100:+.2f}%)")
    print(f"Trade history exported to {out_path}")


if __name__ == "__main__":
    main()
