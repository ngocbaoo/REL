from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from agent.a2c_agent import A2CAgent
from env.trading_env import QcomTradingEnv, N_ACTIONS

_UNSET = object()


def make_env(df: pd.DataFrame, env_cfg: dict, episode_length=_UNSET, random_start=_UNSET, seed=None) -> QcomTradingEnv:
    el = env_cfg["episode_length"] if episode_length is _UNSET else episode_length
    rs = env_cfg["random_start"] if random_start is _UNSET else random_start
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
        episode_length=el,
        random_start=rs,
        seed=seed,
    )


def split_validation(train_df: pd.DataFrame, validation_months: int):
    dates = pd.to_datetime(train_df["Date"])
    cutoff = dates.max() - pd.DateOffset(months=validation_months)
    core_df = train_df[dates < cutoff].reset_index(drop=True)
    val_df = train_df[dates >= cutoff].reset_index(drop=True)
    return core_df, val_df


@torch.no_grad()
def greedy_eval(agent: A2CAgent, df: pd.DataFrame, env_cfg: dict) -> float:
    env = make_env(df, env_cfg, episode_length=None, random_start=False, seed=0)
    obs, info = env.reset()
    done = False
    while not done:
        mask = info["action_mask"]
        action = agent.select_action(obs, mask, greedy=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info["total_assets"]


def resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]
    a2c_cfg = cfg["a2c"]
    train_cfg = cfg["train"]

    seed = train_cfg["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(train_cfg["log_dir"], exist_ok=True)
    os.makedirs(train_cfg["checkpoint_dir"], exist_ok=True)

    full_train_df = pd.read_csv(cfg["data"]["train_csv"])
    core_df, val_df = split_validation(full_train_df, train_cfg["validation_months"])
    print(f"Train-core: {len(core_df)} rows, Validation: {len(val_df)} rows")

    env = make_env(core_df, env_cfg, seed=seed)
    obs, info = env.reset(seed=seed)

    device = resolve_device(train_cfg["device"])
    print(f"Using device: {device}")

    agent = A2CAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=N_ACTIONS,
        hidden_sizes=a2c_cfg["hidden_sizes"],
        gamma=a2c_cfg["gamma"],
        gae_lambda=a2c_cfg["gae_lambda"],
        lr=a2c_cfg["lr"],
        value_coef=a2c_cfg["value_coef"],
        entropy_coef=a2c_cfg["entropy_coef"],
        max_grad_norm=a2c_cfg["max_grad_norm"],
        weight_decay=a2c_cfg.get("weight_decay", 0.0),
        device=device,
        seed=seed,
    )

    writer = SummaryWriter(log_dir=os.path.join(train_cfg["log_dir"], "a2c"))

    n_steps = a2c_cfg["n_steps"]
    total_timesteps = train_cfg["total_timesteps"]
    num_updates = total_timesteps // n_steps
    val_every_updates = 100

    episode_reward = 0.0
    episode_idx = 0
    best_val_assets = -np.inf
    rolling_best_assets = -np.inf

    for update in range(1, num_updates + 1):
        b_obs, b_actions, b_rewards, b_dones, b_masks = [], [], [], [], []

        for _ in range(n_steps):
            mask = info["action_mask"]
            action, _value = agent.act(obs, mask)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            b_obs.append(obs)
            b_actions.append(action)
            b_rewards.append(reward)
            b_dones.append(float(done))
            b_masks.append(mask)

            obs = next_obs
            info = next_info
            episode_reward += reward

            if done:
                final_assets = info["total_assets"]
                rolling_best_assets = max(rolling_best_assets, final_assets)
                writer.add_scalar("episode/cumulative_reward", episode_reward, episode_idx)
                writer.add_scalar("episode/final_total_assets", final_assets, episode_idx)
                writer.add_scalar("episode/rolling_best_total_assets", rolling_best_assets, episode_idx)
                episode_idx += 1
                episode_reward = 0.0
                obs, info = env.reset()


        last_value = 0.0 if b_dones[-1] else agent.value_of(obs)
        stats = agent.update(b_obs, b_actions, b_rewards, b_dones, b_masks, last_value)

        global_step = update * n_steps
        writer.add_scalar("train/loss", stats["loss"], global_step)
        writer.add_scalar("train/policy_loss", stats["policy_loss"], global_step)
        writer.add_scalar("train/value_loss", stats["value_loss"], global_step)
        writer.add_scalar("train/entropy", stats["entropy"], global_step)

        if update % val_every_updates == 0:
            val_assets = greedy_eval(agent, val_df, env_cfg)
            writer.add_scalar("validation/total_assets", val_assets, global_step)
            if val_assets > best_val_assets:
                best_val_assets = val_assets
                best_path = os.path.join(train_cfg["checkpoint_dir"], "a2c_best.pt")
                agent.save(best_path)
                print(f"[upd {update}] step {global_step}: new best val total_assets={val_assets:.2f} -> saved {best_path}")
            print(
                f"[upd {update}/{num_updates}] step {global_step} "
                f"ep={episode_idx} entropy={stats['entropy']:.3f} "
                f"rolling_best={rolling_best_assets:.2f}"
            )

    final_path = os.path.join(train_cfg["checkpoint_dir"], "a2c_final.pt")
    agent.save(final_path)
    print(f"Saved final checkpoint to {final_path}")
    if best_val_assets == -np.inf:
        agent.save(os.path.join(train_cfg["checkpoint_dir"], "a2c_best.pt"))
        print("No validation eval ran; copied final as a2c_best.pt")

    writer.close()


if __name__ == "__main__":
    main()
