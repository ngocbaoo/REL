from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

NEG_INF = -1e8


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_dim, n_actions)
        self.value_head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


class A2CAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes=(128, 128),
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 7e-4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        weight_decay: float = 0.0,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.net = ActorCritic(obs_dim, n_actions, hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

    def _masked_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask, logits, torch.full_like(logits, NEG_INF))

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        logits, value = self.net(obs_t)
        dist = Categorical(logits=self._masked_logits(logits, mask_t))
        action = dist.sample()
        return int(action.item()), float(value.item())

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, mask: np.ndarray, epsilon=None, greedy: bool = True) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        logits, _ = self.net(obs_t)
        masked = self._masked_logits(logits, mask_t)
        if greedy:
            return int(masked.argmax(dim=1).item())
        dist = Categorical(logits=masked)
        return int(dist.sample().item())

    @torch.no_grad()
    def value_of(self, obs: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, value = self.net(obs_t)
        return float(value.item())

    def update(self, obs, actions, rewards, dones, masks, last_value):
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(np.asarray(actions), dtype=torch.int64, device=self.device)
        rewards_np = np.asarray(rewards, dtype=np.float64)
        dones_np = np.asarray(dones, dtype=np.float64)
        masks_t = torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=self.device)

        logits, values = self.net(obs_t)
        values_np = values.detach().cpu().numpy().astype(np.float64)


        T = len(rewards_np)
        advantages = np.zeros(T, dtype=np.float64)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - dones_np[t]
            next_value = last_value if t == T - 1 else values_np[t + 1]
            delta = rewards_np[t] + self.gamma * next_value * next_nonterminal - values_np[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values_np

        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        dist = Categorical(logits=self._masked_logits(logits, masks_t))
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        policy_loss = -(adv_t * log_probs).mean()
        value_loss = F.mse_loss(values, returns_t)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }

    def save(self, path: str):
        torch.save({"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
