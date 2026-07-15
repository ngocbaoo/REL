"""Double DQN + Dueling architecture + Prioritized Experience Replay agent
with action masking, for the QCOM trading environment."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.networks import DuelingQNetwork
from agent.replay_buffer import PrioritizedReplayBuffer

NEG_INF = -1e8 # Vô hiệu hóa Q-value bị mask


class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes=(256, 256),
        gamma: float = 0.99,
        lr: float = 1e-4,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        tau: float | None = None,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 100_000,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_anneal_steps: int = 200_000,
        per_epsilon: float = 1e-6,
        grad_clip_norm: float = 10.0,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.grad_clip_norm = grad_clip_norm
        self.device = torch.device(device)

        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.online_net = DuelingQNetwork(obs_dim, n_actions, hidden_sizes).to(self.device)
        self.target_net = DuelingQNetwork(obs_dim, n_actions, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        self.buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            obs_dim=obs_dim,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_end=per_beta_end,
            beta_anneal_steps=per_beta_anneal_steps,
            eps=per_epsilon,
        )

        self.learn_step = 0
        self.env_step = 0

    def epsilon_at(self, step: int) -> float:
        """
        Kiểm soát exploration vs exploitation (nội suy tuyến tính)
        """
        frac = min(1.0, step / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    @torch.no_grad()
    def _masked_greedy_action(self, net: nn.Module, obs: np.ndarray, mask: np.ndarray) -> int:
        """
        Chọn action tốt nhất trong số action đc phép.
        Trả về action có Q-value cao nhất trong số còn lại
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = net(obs_t).squeeze(0).cpu().numpy() # Dự đoán
        q = np.where(mask, q, NEG_INF)
        return int(np.argmax(q))

    def select_action(self, obs: np.ndarray, mask: np.ndarray, epsilon: float | None = None, greedy: bool = False) -> int:
        """
        Quyết định khi nào nên exploration hay exploitation
        """
        valid_actions = np.flatnonzero(mask) # Trả về mảng chỉ số mask khác 0
        if valid_actions.size == 0:
            return 0  # HOLD is always a fallback; masking guarantees index 0 valid in practice

        if not greedy: # Nếu explore
            eps = self.epsilon_at(self.env_step) if epsilon is None else epsilon
            if self.rng.random() < eps:
                return int(self.rng.choice(valid_actions))

        return self._masked_greedy_action(self.online_net, obs, mask) # Nếu exploit

    def store(self, obs, action, reward, next_obs, done, next_mask):
        """
        Lưu lại kinh nghiệm vào buffer
        """
        self.buffer.add(obs, action, reward, next_obs, done, next_mask)
        self.env_step += 1

    def _soft_update(self):
        """
        Làm mới target_net dựa trên online_net
        """
        with torch.no_grad():
            for target_p, online_p in zip(self.target_net.parameters(), self.online_net.parameters()):
                target_p.data.mul_(1 - self.tau).add_(self.tau * online_p.data)

    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size: # Nếu chưa đủ batch_size thì không thể học
            return None

        batch = self.buffer.sample(self.batch_size, self.learn_step) # batch là 1 dict chứa obs, actions, rewards, next_obs, dones, next_masks, weights, tree_idx

        # Chuyển từng mảng sang tensor Pytorch
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch["next_masks"], dtype=torch.bool, device=self.device)
        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device)

        # Dự đoán hiện tại
        q_values = self.online_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN
        with torch.no_grad():
            next_q_online = self.online_net(next_obs) # Tính s' bằng mạng đang học
            next_q_online_masked = next_q_online.masked_fill(~next_masks, NEG_INF) # Đảo ngược mask và ghi NEG_INF vào các vị trí ko hợp lệ tại s'
            next_actions = next_q_online_masked.argmax(dim=1, keepdim=True) # Chọn action

            next_q_target = self.target_net(next_obs) # Tính s' bằng mạng target
            next_q_selected = next_q_target.gather(1, next_actions).squeeze(1) # Đánh giá 

            td_target = rewards + self.gamma * (1 - dones) * next_q_selected # Công thức Bellman

        td_errors = td_target - q_values # Chênh lệch giữa mong muốn và predict
        loss = (weights * F.smooth_l1_loss(q_values, td_target, reduction="none")).mean() # Huber loss

        # Backward prop
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        # Ghi lại TD-error vào buffer
        self.buffer.update_priorities(batch["tree_idxs"], td_errors.detach().cpu().numpy())

        self.learn_step += 1
        if self.tau is not None:
            self._soft_update()
        elif self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def save(self, path: str):
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "learn_step": self.learn_step,
                "env_step": self.env_step,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.learn_step = ckpt.get("learn_step", 0)
        self.env_step = ckpt.get("env_step", 0)
