# -*- coding: utf-8 -*-
"""HGMAT-R trainer.

采用：
- on-policy clipped objective（PPO 风格）
- joint autoregressive policy 的 joint log-prob
- twin critic（resilience / time80）
- self-predictive auxiliary losses

说明：
- 为了兼容当前项目环境和 rollout 方式，这里按“按 env 轨迹切分 mini-batch”做序列训练。
- 这是一套可运行的研究原型，不追求极致高效实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F


@dataclass
class HGMATRRolloutBatch:
    node_features: torch.Tensor          # (T,B,N,F)
    times: torch.Tensor                  # (T,B)
    prev_actions: torch.Tensor           # (T,B,A)
    prev_reward: torch.Tensor            # (T,B)
    episode_starts: torch.Tensor         # (T,B)
    actions: torch.Tensor                # (T,B,A)
    old_logps: torch.Tensor              # (T,B,A)
    action_masks: List[torch.Tensor]     # len=A, each (T,B,action_dim_i)

    rewards_res: torch.Tensor            # (T,B)
    rewards_t80: torch.Tensor            # (T,B)
    dones: torch.Tensor                  # (T,B)
    values_res: torch.Tensor             # (T,B)
    values_t80: torch.Tensor             # (T,B)
    advantages_res: torch.Tensor         # (T,B)
    advantages_t80: torch.Tensor         # (T,B)
    returns_res: torch.Tensor            # (T,B)
    returns_t80: torch.Tensor            # (T,B)

    target_next_fn: torch.Tensor         # (T,B)
    target_next_ac: torch.Tensor         # (T,B)
    target_completion: torch.Tensor      # (T,B)
    aux_valid: torch.Tensor              # (T,B)


class HGMATRTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        edges: Dict[str, torch.Tensor],
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        aux_coef: float = 0.1,
        grad_clip: float = 0.5,
        alpha_start: float = 0.5,
        alpha_end: float = 0.5,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.model = model
        self.edges = edges
        self.clip_eps = float(clip_eps)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.aux_coef = float(aux_coef)
        self.grad_clip = float(grad_clip)
        self.alpha_start = float(alpha_start)
        self.alpha_end = float(alpha_end)
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr))
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    def _normalize_adv(self, adv: torch.Tensor) -> torch.Tensor:
        return (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    def _mix_alpha(self, update_idx: int, total_updates: int) -> float:
        if total_updates <= 1:
            return self.alpha_end
        r = min(1.0, max(0.0, float(update_idx) / float(total_updates - 1)))
        return (1.0 - r) * self.alpha_start + r * self.alpha_end

    def update(
        self,
        batch: HGMATRRolloutBatch,
        update_idx: int,
        total_updates: int,
        update_epochs: int = 4,
        env_minibatch_size: int = 4,
    ) -> Dict[str, float]:
        T, B = batch.rewards_res.shape
        env_mb = max(1, min(int(env_minibatch_size), B))
        alpha = self._mix_alpha(update_idx, total_updates)

        actor_loss_total = 0.0
        critic_loss_total = 0.0
        aux_loss_total = 0.0
        entropy_total = 0.0
        total_loss_total = 0.0
        n_updates = 0

        old_joint_logp = batch.old_logps.sum(dim=-1)  # (T,B)
        adv_mix_all = alpha * batch.advantages_res + (1.0 - alpha) * batch.advantages_t80

        for _ in range(int(update_epochs)):
            perm = torch.randperm(B, generator=self.generator)
            for start in range(0, B, env_mb):
                idx = perm[start : start + env_mb].to(batch.node_features.device)

                out = self.model.evaluate_sequence(
                    node_features=batch.node_features[:, idx],
                    edges=self.edges,
                    times=batch.times[:, idx],
                    prev_actions=batch.prev_actions[:, idx],
                    prev_reward=batch.prev_reward[:, idx],
                    episode_starts=batch.episode_starts[:, idx],
                    actions=batch.actions[:, idx],
                    action_masks=[m[:, idx] for m in batch.action_masks],
                )

                adv_mix = self._normalize_adv(adv_mix_all[:, idx].detach())
                ratio = torch.exp(out["joint_logp"] - old_joint_logp[:, idx])
                surr1 = ratio * adv_mix
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_mix
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss_res = F.mse_loss(out["value_res"], batch.returns_res[:, idx])
                critic_loss_t80 = F.mse_loss(out["value_t80"], batch.returns_t80[:, idx])
                critic_loss = critic_loss_res + critic_loss_t80

                aux_valid = batch.aux_valid[:, idx]
                denom = aux_valid.sum().clamp_min(1.0)
                aux_loss_fn = (((out["pred_next_fn"] - batch.target_next_fn[:, idx]) ** 2) * aux_valid).sum() / denom
                aux_loss_ac = (((out["pred_next_ac"] - batch.target_next_ac[:, idx]) ** 2) * aux_valid).sum() / denom
                aux_loss_done = (
                    F.binary_cross_entropy_with_logits(
                        out["pred_completion_logit"],
                        batch.target_completion[:, idx],
                        reduction="none",
                    )
                    * aux_valid
                ).sum() / denom
                aux_loss = aux_loss_fn + aux_loss_ac + aux_loss_done

                entropy = out["entropy"].mean()
                loss = actor_loss + self.value_coef * critic_loss + self.aux_coef * aux_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                actor_loss_total += float(actor_loss.detach().item())
                critic_loss_total += float(critic_loss.detach().item())
                aux_loss_total += float(aux_loss.detach().item())
                entropy_total += float(entropy.detach().item())
                total_loss_total += float(loss.detach().item())
                n_updates += 1

        return {
            "loss": total_loss_total / max(1, n_updates),
            "actor_loss": actor_loss_total / max(1, n_updates),
            "critic_loss": critic_loss_total / max(1, n_updates),
            "aux_loss": aux_loss_total / max(1, n_updates),
            "entropy": entropy_total / max(1, n_updates),
            "alpha_mix": float(alpha),
        }
