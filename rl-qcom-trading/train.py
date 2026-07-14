"""Train the Double-DQN + Dueling + PER agent on QCOM data.

Usage:
    python train.py --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from agent.dqn_agent import DQNAgent
from env.trading_env import QcomTradingEnv, N_ACTIONS


def make_env(df: pd.DataFrame, env_cfg: dict, episode_length=None, random_start=None, seed=None) -> QcomTradingEnv:
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
        bankruptcy_frac=env_cfg["bankruptcy_frac"],
        price_mode=env_cfg["price_mode"],
        use_soft_cooldown_penalty=env_cfg["use_soft_cooldown_penalty"],
        idle_penalty_enabled=env_cfg["idle_penalty_enabled"],
        idle_penalty_value=env_cfg["idle_penalty_value"],
        idle_penalty_multiplier=env_cfg["idle_penalty_multiplier"],
        episode_length=env_cfg["episode_length"] if episode_length is None else episode_length,
        random_start=env_cfg["random_start"] if random_start is None else random_start,
        seed=seed,
    )


def split_validation(train_df: pd.DataFrame, validation_months: int):
    dates = pd.to_datetime(train_df["Date"])
    cutoff = dates.max() - pd.DateOffset(months=validation_months)
    core_df = train_df[dates < cutoff].reset_index(drop=True)
    val_df = train_df[dates >= cutoff].reset_index(drop=True)
    return core_df, val_df


@torch.no_grad()
def greedy_eval(agent: DQNAgent, df: pd.DataFrame, env_cfg: dict) -> float:
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

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]
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

    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=N_ACTIONS,
        hidden_sizes=agent_cfg["hidden_sizes"],
        gamma=agent_cfg["gamma"],
        lr=agent_cfg["lr"],
        buffer_size=agent_cfg["buffer_size"],
        batch_size=agent_cfg["batch_size"],
        target_update_freq=agent_cfg["target_update_freq"],
        tau=agent_cfg["tau"],
        epsilon_start=agent_cfg["epsilon_start"],
        epsilon_end=agent_cfg["epsilon_end"],
        epsilon_decay_steps=agent_cfg["epsilon_decay_steps"],
        per_alpha=agent_cfg["per_alpha"],
        per_beta_start=agent_cfg["per_beta_start"],
        per_beta_end=agent_cfg["per_beta_end"],
        per_beta_anneal_steps=agent_cfg["per_beta_anneal_steps"],
        per_epsilon=agent_cfg["per_epsilon"],
        grad_clip_norm=agent_cfg["grad_clip_norm"],
        device=device,
        seed=seed,
    )

    writer = SummaryWriter(log_dir=train_cfg["log_dir"])

    episode_reward = 0.0
    episode_len = 0
    episode_idx = 0
    best_val_assets = -np.inf
    rolling_best_assets = -np.inf

    total_timesteps = train_cfg["total_timesteps"]
    train_start_steps = agent_cfg["train_start_steps"]
    train_freq = agent_cfg["train_freq"]

    for global_step in range(1, total_timesteps + 1):
        mask = info["action_mask"]
        action = agent.select_action(obs, mask)

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        next_mask = next_info["action_mask"]

        agent.store(obs, action, reward, next_obs, done, next_mask)

        obs = next_obs
        info = next_info
        episode_reward += reward
        episode_len += 1

        loss = None
        if global_step > train_start_steps and global_step % train_freq == 0:
            loss = agent.learn()
            if loss is not None:
                writer.add_scalar("train/loss", loss, global_step)

        writer.add_scalar("train/epsilon", agent.epsilon_at(agent.env_step), global_step)

        if done:
            final_assets = info["total_assets"]
            rolling_best_assets = max(rolling_best_assets, final_assets)
            writer.add_scalar("episode/cumulative_reward", episode_reward, episode_idx)
            writer.add_scalar("episode/final_total_assets", final_assets, episode_idx)
            writer.add_scalar("episode/rolling_best_total_assets", rolling_best_assets, episode_idx)
            writer.add_scalar("episode/length", episode_len, episode_idx)

            if episode_idx % train_cfg["eval_freq_episodes"] == 0:
                val_assets = greedy_eval(agent, val_df, env_cfg)
                writer.add_scalar("validation/total_assets", val_assets, episode_idx)
                if val_assets > best_val_assets:
                    best_val_assets = val_assets
                    best_path = os.path.join(train_cfg["checkpoint_dir"], "best.pt")
                    agent.save(best_path)
                    print(f"[ep {episode_idx}] step {global_step}: new best val total_assets={val_assets:.2f} -> saved {best_path}")

            if episode_idx % 10 == 0:
                print(
                    f"[ep {episode_idx}] step {global_step}/{total_timesteps} "
                    f"reward={episode_reward:.4f} assets={final_assets:.2f} eps={agent.epsilon_at(agent.env_step):.3f}"
                )

            episode_idx += 1
            episode_reward = 0.0
            episode_len = 0
            obs, info = env.reset()

    final_path = os.path.join(train_cfg["checkpoint_dir"], "final.pt")
    agent.save(final_path)
    print(f"Saved final checkpoint to {final_path}")
    if best_val_assets == -np.inf:
        agent.save(os.path.join(train_cfg["checkpoint_dir"], "best.pt"))
        print("No validation eval ran during training; copied final checkpoint as best.pt")

    writer.close()


if __name__ == "__main__":
    main()
