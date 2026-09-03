"""PPO (Proximal Policy Optimization) trainer with action masking.

Clipped-surrogate PPO over self-play batches. Advantages are
returns - value_baseline (Monte-Carlo; a hand of mahjong is short enough
that GAE adds little). Entropy regularization keeps early exploration alive
so the policy discovers claims and wins before collapsing.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F


class PPOTrainer:
    def __init__(self, model, device="cpu", lr=3e-4, clip_eps=0.2,
                 epochs=3, minibatch_size=1024, value_coef=0.5,
                 entropy_coef=0.01, max_grad_norm=0.5):
        self.model = model
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def update(self, batch):
        """One PPO update over a self-play batch. Returns loss metrics."""
        dev = self.device
        planes = torch.from_numpy(batch["planes"]).to(dev)
        history = torch.from_numpy(batch["history"]).to(dev)
        masks = torch.from_numpy(batch["masks"]).to(dev)
        actions = torch.from_numpy(batch["actions"]).to(dev)
        old_logprobs = torch.from_numpy(batch["logprobs"]).to(dev)
        returns = torch.from_numpy(batch["returns"]).to(dev)
        old_values = torch.from_numpy(batch["values"]).to(dev)

        adv = returns - old_values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(actions)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        steps = 0
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=dev)
            for start in range(0, n, self.minibatch_size):
                idx = perm[start:start + self.minibatch_size]
                logits, values = self.model(planes[idx], history[idx],
                                            masks[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logprobs = dist.log_prob(actions[idx])
                ratio = torch.exp(logprobs - old_logprobs[idx])

                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * adv[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns[idx])
                entropy = dist.entropy().mean()

                loss = (policy_loss + self.value_coef * value_loss
                        - self.entropy_coef * entropy)
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_grad_norm)
                self.opt.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                steps += 1
        return {k: v / max(steps, 1) for k, v in metrics.items()}

    # ---------- checkpointing ----------
    def save(self, path, games_played, config=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"  # atomic: a mid-save kill can't corrupt the file
        torch.save({"model": self.model.state_dict(),
                    "optimizer": self.opt.state_dict(),
                    "games_played": games_played,
                    "config": config or {}}, tmp)
        os.replace(tmp, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        return ckpt.get("games_played", 0)
