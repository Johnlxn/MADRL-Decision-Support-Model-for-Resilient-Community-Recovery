# -*- coding: utf-8 -*-
"""顺序更新的多智能体 PPO（论文 Eq.(16)(17) + Algorithm 1）。

论文核心点：
- 不是同时更新所有 agent 的策略，而是每轮更新时对 agent 做一个随机排列 i1:n
- 依次更新每个 agent，并在目标函数中引入前面已更新 agent 的联合概率比率项

本实现采用共享 critic V(s)，每个 agent 一个独立 actor。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from marl_recovery.models.actor_critic import ActorNet, CriticNet
from marl_recovery.algorithms.rl_utils import iter_minibatches


@dataclass
class RolloutBatch:
    """一次 PPO 更新用到的数据（已展平成 batch）。"""

    node_features: torch.Tensor  # (B,N,5)
    actions: torch.Tensor  # (B,n_agents)
    old_logps: torch.Tensor  # (B,n_agents)
    action_masks: List[torch.Tensor]  # len=n_agents, each (B,action_dim_i)
    advantages: torch.Tensor  # (B,)
    returns: torch.Tensor  # (B,)


class SeqMAPPOTrainer:
    def __init__(
        self,
        actors: List[ActorNet],
        critic: CriticNet,
        edges: Dict[str, torch.Tensor],
        actor_lrs: List[float],
        critic_lr: float,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.0,
        grad_clip: float = 0.5,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.actors = actors
        self.critic = critic
        self.edges = edges
        self.clip_eps = float(clip_eps)
        self.entropy_coef = float(entropy_coef)
        self.grad_clip = float(grad_clip)
        self.device = device
        self.rng = np.random.default_rng(seed)

        # optimizers
        self.actor_opts = []
        for actor, lr in zip(self.actors, actor_lrs):
            self.actor_opts.append(torch.optim.Adam(actor.parameters(), lr=float(lr)))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(critic_lr))

    def update(
        self,
        batch: RolloutBatch,
        update_epochs: int,
        minibatch_size: int,
        normalize_adv: bool = True,
    ) -> Dict[str, float]:
        """执行一次 PPO 更新。"""
        B = batch.node_features.shape[0]
        n_agents = len(self.actors)

        node_features = batch.node_features.to(self.device)
        actions = batch.actions.to(self.device)
        old_logps = batch.old_logps.to(self.device)
        advantages = batch.advantages.to(self.device)
        returns = batch.returns.to(self.device)
        masks = [m.to(self.device) for m in batch.action_masks]

        if normalize_adv:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        # 先更新 actor（论文 Algorithm 1 的顺序：先 actor 后 critic）
        actor_loss_total = 0.0
        entropy_total = 0.0

        for _ in range(int(update_epochs)):
            perm = self.rng.permutation(n_agents)
            updated_logps = old_logps.clone()  # (B,n_agents)，逐个 agent 更新其列

            prev_agents: List[int] = []
            for agent_idx in perm:
                # 前面已更新的 agent 概率比率项：exp(sum(logp_new - logp_old))
                if len(prev_agents) == 0:
                    prev_ratio = torch.ones((B,), dtype=advantages.dtype, device=self.device)
                else:
                    prev_ratio = torch.exp((updated_logps[:, prev_agents] - old_logps[:, prev_agents]).sum(dim=1))

                # 对当前 agent：M_{i1:m} = prev_ratio * A_hat
                M = prev_ratio * advantages

                # minibatch SGD
                for mb_idx in iter_minibatches(B, minibatch_size, self.rng):
                    mb = torch.as_tensor(mb_idx, dtype=torch.long, device=self.device)

                    # 当前 agent 的新 logp
                    new_logp = self.actors[agent_idx].log_prob(
                        node_features[mb],
                        self.edges,
                        masks[agent_idx][mb],
                        actions[mb, agent_idx],
                    )
                    ratio = torch.exp(new_logp - old_logps[mb, agent_idx])

                    # PPO clipped surrogate
                    unclipped = ratio * M[mb]
                    clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * M[mb]
                    obj = torch.min(unclipped, clipped).mean()

                    # entropy bonus（可选）
                    ent = self.actors[agent_idx].entropy(node_features[mb], self.edges, masks[agent_idx][mb]).mean()

                    loss = -obj - self.entropy_coef * ent

                    self.actor_opts[agent_idx].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), self.grad_clip)
                    self.actor_opts[agent_idx].step()

                    actor_loss_total += float(loss.item())
                    entropy_total += float(ent.item())

                # 用更新后的策略重算该 agent 的 logp（整批），供后续 agent 使用
                with torch.no_grad():
                    updated_col = self.actors[agent_idx].log_prob(
                        node_features,
                        self.edges,
                        masks[agent_idx],
                        actions[:, agent_idx],
                    )
                    updated_logps[:, agent_idx] = updated_col

                prev_agents.append(agent_idx)

        # 再更新 critic
        critic_loss_total = 0.0
        for _ in range(int(update_epochs)):
            for mb_idx in iter_minibatches(B, minibatch_size, self.rng):
                mb = torch.as_tensor(mb_idx, dtype=torch.long, device=self.device)
                v_pred = self.critic(node_features[mb], self.edges)
                loss_v = torch.mean((v_pred - returns[mb]) ** 2)

                self.critic_opt.zero_grad()
                loss_v.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                self.critic_opt.step()

                critic_loss_total += float(loss_v.item())

        # 取平均（近似）
        denom = float(update_epochs) * max(1.0, float(B) / float(minibatch_size))
        stats = {
            "actor_loss": actor_loss_total / max(1e-9, denom * n_agents),
            "critic_loss": critic_loss_total / max(1e-9, denom),
            "entropy": entropy_total / max(1e-9, denom * n_agents),
        }
        return stats
