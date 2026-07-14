"""Prioritized Experience Replay buffer backed by a sum-tree."""
from __future__ import annotations

import numpy as np


class SumTree:
    """Binary sum-tree supporting O(log n) update and prefix-sum sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self.size = 0
        self._max_priority_seen = 0.0

    def add(self, priority: float, data_idx: int):
        tree_idx = data_idx + self.capacity - 1
        self.update(tree_idx, priority)
        if priority > self._max_priority_seen:
            self._max_priority_seen = priority

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
        if priority > self._max_priority_seen:
            self._max_priority_seen = priority

    def get(self, s: float):
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                leaf = parent
                break
            if s <= self.tree[left]:
                parent = left
            else:
                s -= self.tree[left]
                parent = right
        data_idx = leaf - (self.capacity - 1)
        return leaf, self.tree[leaf], data_idx

    @property
    def total(self) -> float:
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        return self._max_priority_seen if self._max_priority_seen > 0 else 1.0


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 200_000,
        eps: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.eps = eps

        self.tree = SumTree(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.zeros((capacity, 11), dtype=bool)

        self.ptr = 0
        self.size = 0

    def beta_at(self, step: int) -> float:
        frac = min(1.0, step / max(1, self.beta_anneal_steps))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def add(self, obs, action, reward, next_obs, done, next_mask):
        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = float(done)
        self.next_masks[idx] = next_mask

        max_p = self.tree.max_priority
        self.tree.add(max_p, idx)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.tree.size = self.size

    def sample(self, batch_size: int, step: int):
        if self.size < batch_size:
            raise ValueError("Not enough samples in buffer to sample a batch")

        idxs = np.zeros(batch_size, dtype=np.int64)
        tree_idxs = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = np.random.uniform(a, b)
            tree_idx, priority, data_idx = self.tree.get(s)
            tree_idxs[i] = tree_idx
            priorities[i] = priority
            idxs[i] = data_idx

        probs = priorities / self.tree.total
        beta = self.beta_at(step)
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()

        batch = dict(
            obs=self.obs[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_obs=self.next_obs[idxs],
            dones=self.dones[idxs],
            next_masks=self.next_masks[idxs],
            weights=weights.astype(np.float32),
            tree_idxs=tree_idxs,
        )
        return batch

    def update_priorities(self, tree_idxs: np.ndarray, td_errors: np.ndarray):
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for tree_idx, priority in zip(tree_idxs, priorities):
            self.tree.update(tree_idx, float(priority))

    def __len__(self):
        return self.size
