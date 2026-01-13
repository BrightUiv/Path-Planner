from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -0.999999, 0.999999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


@dataclass
class ActOut:
    action: torch.Tensor
    logp: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


class SquashedNormal:
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self.mean = mean
        self.log_std = log_std.clamp(-5.0, 2.0)
        self.std = self.log_std.exp()
        self.base = Normal(self.mean, self.std)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.base.rsample()
        a = torch.tanh(z)
        logp = self.log_prob(a, z=z)
        entropy = self.base.entropy().sum(dim=-1)
        return a, logp, entropy

    def log_prob(self, a: torch.Tensor, *, z: torch.Tensor | None = None) -> torch.Tensor:
        if z is None:
            z = atanh(a)
        logp = self.base.log_prob(z).sum(dim=-1)
        log_det = torch.log(torch.clamp(1.0 - a * a, min=1e-6)).sum(dim=-1)
        return logp - log_det


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return mean, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> ActOut:
        mean, value = self(obs)
        dist = SquashedNormal(mean, self.log_std.expand_as(mean))
        action, logp, entropy = dist.sample()
        return ActOut(action=action, logp=logp, value=value, entropy=entropy)

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self(obs)
        dist = SquashedNormal(mean, self.log_std.expand_as(mean))
        logp = dist.log_prob(action)
        entropy = dist.base.entropy().sum(dim=-1)
        return logp, value, entropy


@dataclass
class OrcaParamBounds:
    time_horizon: tuple[float, float] = (0.3, 2.5)
    safety_margin: tuple[float, float] = (0.02, 0.35)
    avoid_weight: tuple[float, float] = (0.2, 3.0)
    max_speed: tuple[float, float] = (0.4, 2.0)


def decode_orca_params(action: torch.Tensor, bounds: OrcaParamBounds) -> dict[str, torch.Tensor]:
    if action.shape[-1] != 4:
        raise ValueError("Expected action dim=4 (time_horizon, safety_margin, avoid_weight, max_speed).")

    def _map01(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        u = (x + 1.0) * 0.5
        return lo + (hi - lo) * torch.clamp(u, 0.0, 1.0)

    return {
        "time_horizon": _map01(action[..., 0], *bounds.time_horizon),
        "safety_margin": _map01(action[..., 1], *bounds.safety_margin),
        "avoid_weight": _map01(action[..., 2], *bounds.avoid_weight),
        "max_speed": _map01(action[..., 3], *bounds.max_speed),
    }

