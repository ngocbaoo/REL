"""Dueling Q-network (PyTorch) for the QCOM trading agent."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int] = (256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        self.value_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        value = self.value_head(h)                       # (B, 1)
        advantage = self.advantage_head(h)                # (B, n_actions)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q
